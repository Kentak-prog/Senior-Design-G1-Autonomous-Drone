"""
Min-snap planning for the gate course (no ROS).

  gate_from_pose       one vision PoseArray entry (position + FULL gate-frame
                       quaternion) -> gate_path.Gate; identity quaternion =
                       "normal unreliable" -> center only
  GateLatch            stable gate set from per-frame detections (match,
                       running mean, min_hits, course order by distance)
  look_at_yaw_profile  nose at the next gate instead of along the velocity
                       (keeps the gates in the camera for VIO)
  plan_course          gates + drone pose -> (traj, waypoints, TrajectoryReference),
                       mass and drag from control/drone_constraints.py
  trajectory_points    TrajectoryReference -> (times, 12 x M states, thrust)

Used by the ROS 2 node ros2_nodes/min_snap_node.py and by the Python
rehearsals. Tested in Trajectory/test_min_snap_planner.py.
"""

import os
import sys
from dataclasses import dataclass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_REPO, "control"), os.path.join(_REPO, "vision")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gate_path import Gate, plan_gate_trajectory, DEFAULT_GATE_SIDE   # noqa: E402
from trajectory_reference import TrajectoryReference                   # noqa: E402
from drone_constraints import drone_constraints                        # noqa: E402

POINT_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz", "p", "q", "r")


def _quat_to_rot_wxyz(qw, qx, qy, qz):
    """Rotation matrix of a unit quaternion (no Euler angles involved)."""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def upright_gate_frame(center, flight_dir):
    """T_world_gate (vision gate model: +X right, +Y down, +Z flight direction) for an upright gate."""
    z = np.array([flight_dir[0], flight_dir[1], 0.0], dtype=float)
    if np.linalg.norm(z) < 1e-6:
        z = np.array([1.0, 0.0, 0.0])
    z /= np.linalg.norm(z)
    y = np.array([0.0, 0.0, -1.0])
    x = np.cross(y, z)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, center
    return T


def gate_from_pose(position, orientation_xyzw, name="gate", side=DEFAULT_GATE_SIDE, viewer_pos=None):
    """
    One PoseArray entry -> Gate.

    The orientation IS the gate frame (vision/ros_camera*.py publish
    T_world_gate's rotation), so the rotation is used as a whole -- the old
    node kept only its yaw and rebuilt the frame with +Z up, which pointed
    every gate's flight direction straight up.

    An identity quaternion is the vision node's "normal unreliable" marker:
    the gate is then planned through its center only (normal_reliable=False),
    with a frame facing away from viewer_pos (or world +X) for clearance checks.
    """
    c = np.array([position.x, position.y, position.z], dtype=float)
    qx, qy, qz, qw = orientation_xyzw
    if abs(abs(qw) - 1.0) < 1e-9 and abs(qx) < 1e-9 and abs(qy) < 1e-9 and abs(qz) < 1e-9:
        d = c - np.asarray(viewer_pos, dtype=float) if viewer_pos is not None else np.array([1.0, 0.0, 0.0])
        return Gate(name, upright_gate_frame(c, d), side=side, normal_reliable=False)
    T = np.eye(4)
    T[:3, :3] = _quat_to_rot_wxyz(qw, qx, qy, qz)
    T[:3, 3] = c
    return Gate(name, T, side=side, normal_reliable=True)


@dataclass
class _LatchedGate:
    center: np.ndarray
    normal_sum: np.ndarray       # sum of reliable unit normals (sign-aligned)
    hits: int
    reliable_hits: int
    order_key: float
    side: float
    last_seen: float

    @property
    def normal(self):
        n = np.linalg.norm(self.normal_sum)
        return self.normal_sum / n if n > 0 else None


class GateLatch:
    """Stable gate set from per-frame detections (see module docstring, step 1)."""

    def __init__(self, match_dist=1.0, min_hits=3):
        self.match_dist = match_dist
        self.min_hits = min_hits
        self.gates = []
        self.last_new_gate_time = None

    def update(self, detections, now, viewer_pos):
        """detections: list of Gate from one frame; viewer_pos: drone position (ENU)."""
        viewer_pos = np.asarray(viewer_pos, dtype=float)
        for det in detections:
            match = None
            if self.gates:
                d = [np.linalg.norm(g.center - det.center) for g in self.gates]
                i = int(np.argmin(d))
                if d[i] < self.match_dist:
                    match = self.gates[i]
            if match is None:
                n = det.normal.copy() if det.normal_reliable else np.zeros(3)
                self.gates.append(_LatchedGate(det.center.copy(), n, 1, int(det.normal_reliable),
                                               float(np.linalg.norm(det.center - viewer_pos)), det.side, now))
                self.last_new_gate_time = now
                continue
            match.hits += 1
            match.center += (det.center - match.center) / match.hits          # running mean
            if det.normal_reliable:
                n = det.normal
                ref = match.normal
                match.normal_sum += -n if (ref is not None and n @ ref < 0) else n
                match.reliable_hits += 1
            match.last_seen = now

    def confirmed(self, viewer_pos=None):
        """Confirmed gates as Gate objects, in course order."""
        out = []
        for i, g in enumerate(sorted((g for g in self.gates if g.hits >= self.min_hits), key=lambda g: g.order_key)):
            if g.normal is not None:
                out.append(Gate(f"gate_{i}", upright_gate_frame(g.center, g.normal), side=g.side, normal_reliable=True))
            else:
                d = g.center - (np.asarray(viewer_pos, dtype=float) if viewer_pos is not None else g.center - [1, 0, 0])
                out.append(Gate(f"gate_{i}", upright_gate_frame(g.center, d), side=g.side, normal_reliable=False))
        return out


def look_at_yaw_profile(traj, wps, gates, start_yaw, smooth_sigma=0.4, blend_s=1.0):
    """
    Yaw that points the nose (and the front camera) at the next gate not yet
    reached, instead of along the velocity. On a slalom the velocity heading
    swings far off the next gate (-80 deg on the 3-gate course), which takes
    every gate out of the camera; a quadrotor can fly sideways, so the MPC can
    follow any yaw. After the last gate it looks along that gate's flight
    direction. Gaussian-smoothed (smooth_sigma s) so target switches are not
    steps, and blended in from start_yaw over blend_s.
    """
    centers = [(traj.t_breaks[i], gates[w['gate']].center) for i, w in enumerate(wps) if w['type'] == 'center']
    last_dir = gates[wps[[i for i, w in enumerate(wps) if w['type'] == 'center'][-1]]['gate']].normal if centers \
        else np.array([np.cos(start_yaw), np.sin(start_yaw), 0.0])

    def profile(times, pos, vel):
        t_c = np.array([c[0] for c in centers])
        targets = np.empty_like(pos)
        for k, t in enumerate(times):
            j = np.searchsorted(t_c, t, side='right') if len(t_c) else 0
            targets[k] = centers[j][1] if j < len(centers) else pos[k] + 10.0 * last_dir
        d = targets - pos
        raw = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        raw += 2 * np.pi * np.round((start_yaw - raw[0]) / (2 * np.pi))
        dt = times[1] - times[0]
        half = int(3 * smooth_sigma / dt)
        kernel = np.exp(-0.5 * (np.arange(-half, half + 1) * dt / smooth_sigma) ** 2)
        kernel /= kernel.sum()
        smooth = np.convolve(np.pad(raw, half, mode='edge'), kernel, 'valid')
        x = np.clip(times / max(blend_s, 1e-6), 0.0, 1.0)
        w = x * x * (3.0 - 2.0 * x)
        return start_yaw + w * (smooth - start_yaw)
    return profile


def plan_course(gates, start_pos, start_yaw, v_max=None, a_max=None, yaw_mode="velocity", optimize_times=True):
    """
    Plan from the drone's current pose; mass and drag come from drone_constraints.
    v_max / a_max: None = 80 % of the drone's limits (gate_path.DEFAULT_V_MAX / DEFAULT_A_MAX).
    optimize_times: time-optimal split of the time between segments (gate_path.plan_gate_trajectory).
    yaw_mode "velocity": face the direction of travel (TrajectoryReference's default);
             "next_gate": look at the next gate (keeps the gates in the camera for VIO).
    """
    traj, wps = plan_gate_trajectory(gates, np.asarray(start_pos, dtype=float), v_max=v_max, a_max=a_max,
                                     verbose=False, optimize_times=optimize_times)
    if yaw_mode == "next_gate":
        profile = look_at_yaw_profile(traj, wps, gates, start_yaw)
    elif yaw_mode == "velocity":
        profile = None
    else:
        raise ValueError(f"unknown yaw_mode {yaw_mode!r} (use 'velocity' or 'next_gate')")
    ref = TrajectoryReference(traj, start_yaw=start_yaw, m=drone_constraints["mass"], yaw_profile=profile)
    return traj, wps, ref


def trajectory_points(ref, dt=0.02):
    """(times, 12 x M states, thrust) sampled from a TrajectoryReference, for MinSnapPoint."""
    ts = np.arange(0.0, ref.duration + 1e-9, dt)
    states = ref.build_reference_horizon(0.0, len(ts) - 1, dt)
    thrust = np.array([ref.feedforward_thrust(t) for t in ts])
    return ts, states, thrust
