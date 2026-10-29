"""
MPC attitude + thrust setpoints -> PX4 SET_ATTITUDE_TARGET (no ROS).

  attitude_ned_frd / rates_frd / thrust_normalized / tilt_deg
                  ENU-FLU -> NED-FRD frame and thrust conversions
  AttitudeBridge  the OFFBOARD sequence: stream setpoints, switch to
                  OFFBOARD, forward each new setpoint, abort to AUTO.LOITER
                  on a stale / invalid / over-tilted setpoint, never fight a
                  mode switch. The link (PX4Link) and the setpoint source
                  are injected, so it is tested against a fake PX4
                  (px4/test_attitude_bridge.py).

The ROS 2 front end is ros2_nodes/mpc_offboard_bridge.py.

*** UNTESTED AGAINST THE CLUSTER. ***

PX4 keeps its own attitude and rate loops: the MPC decides WHERE to point the
thrust and HOW MUCH, PX4 turns that into motor commands.

FRAMES: setpoints are world ENU / body FLU; PX4 wants NED / FRD. The attitude
is converted as R_ned_frd = A R_enu_flu B (A swaps x/y and flips z, B flips
body y and z); body rates (p, q, r) FLU -> (p, -q, -r) FRD.

THRUST: SET_ATTITUDE_TARGET takes a normalized thrust 0..1. Default: invert
Pegasus' PX4 mapping for the Iris, per rotor omega = 1000 u + 100 rad/s and
F = 8.54858e-6 omega^2 (workspace/STARLING2_DYNAMICS.md, source S9) -- hover
(1.5 kg) is about u = 0.556. A measured hover throttle (PX4's MPC_THR_HOVER /
hover thrust estimate) can be given instead and scales linearly.
"""

import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_REPO, "control")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fly_min_snap_offboard import PX4Link, PX4_MAIN_OFFBOARD    # noqa: E402
from fly_gate_approach import ned_to_enu                         # noqa: E402
from drone_constraints import drone_constraints                 # noqa: E402


def rotation_from_quat_xyzw(qx, qy, qz, qw):
    """Body -> world rotation of a ROS (x, y, z, w) quaternion."""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


G = 9.81
A_NED_FROM_ENU = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
B_FLU_FROM_FRD = np.diag([1.0, -1.0, -1.0])

# Pegasus PX4 backend (Iris): omega = OMEGA_PER_U * u + OMEGA_OFFSET, F = KF * omega^2 per rotor
KF = 8.54858e-6
OMEGA_PER_U = 1000.0
OMEGA_OFFSET = 100.0
N_ROTORS = 4

# SET_ATTITUDE_TARGET type_mask: 0 = use attitude, body rates and thrust
ATT_TYPE_MASK_ALL = 0


# --------------------------------------------------------------------------
# Pure conversions (no pymavlink, no ROS)
# --------------------------------------------------------------------------

def quat_wxyz_from_rotation(R):
    """Unit quaternion (w, x, y, z) of a rotation matrix (Shepperd's method)."""
    R = np.asarray(R, dtype=float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * math.sqrt(tr + 1.0)
        q = (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = ((R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s)
    q = np.array(q)
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def attitude_ned_frd(q_xyzw_enu_flu):
    """ENU/FLU attitude quaternion (ROS x, y, z, w) -> PX4 NED/FRD quaternion (MAVLink w, x, y, z)."""
    R_enu_flu = rotation_from_quat_xyzw(*q_xyzw_enu_flu)
    return quat_wxyz_from_rotation(A_NED_FROM_ENU @ R_enu_flu @ B_FLU_FROM_FRD)


def rates_frd(rates_flu):
    p, q, r = (float(v) for v in rates_flu)
    return np.array([p, -q, -r])


def tilt_deg(q_xyzw):
    """Angle between body z and world z (deg)."""
    R = rotation_from_quat_xyzw(*q_xyzw)
    return math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))


def thrust_normalized(thrust_n, hover_thrust=None, mass=drone_constraints["mass"]):
    """
    Total thrust (N) -> PX4 normalized thrust 0..1.
    hover_thrust None: invert Pegasus' Iris mapping (per rotor omega = 1000 u + 100, F = kf omega^2).
    hover_thrust U:    linear, U at m*g (PX4's own hover-throttle model).
    """
    thrust_n = max(0.0, float(thrust_n))
    if hover_thrust is not None:
        u = float(hover_thrust) * thrust_n / (mass * G)
    else:
        omega = math.sqrt(thrust_n / (N_ROTORS * KF))
        u = (omega - OMEGA_OFFSET) / OMEGA_PER_U
    return float(min(1.0, max(0.0, u)))


# --------------------------------------------------------------------------
# The OFFBOARD sequence (link + setpoint source injectable for tests)
# --------------------------------------------------------------------------

class AttitudeBridge:
    """
    get_setpoint() -> (setpoint, arrival_time) or (None, None); setpoint has
    .orientation (x, y, z, w), .body_rate (x, y, z), .thrust (N), .valid.
    arrival_time is on `clock` (wall, monotonic) -- staleness never compares
    the ROS clock with PX4's.
    external_abort() -> reason (str) or None: an outside watchdog (e.g.
    ros2_nodes/flight_monitor.py); any reason aborts to AUTO.LOITER.
    on_offboard(): called once when PX4 confirms OFFBOARD (the bridge node
    tells min_snap_node to start the plan then).
    """

    def __init__(self, link: PX4Link, get_setpoint, rate_hz=50.0, stale_s=0.3, invalid_s=0.5,
                 max_tilt_deg=45.0, hover_thrust=None, mass=drone_constraints["mass"],
                 clock=time.monotonic, sleep=time.sleep, out=print, external_abort=None,
                 on_offboard=None):
        self.v = link
        self.get_setpoint = get_setpoint
        self.dt = 1.0 / rate_hz
        self.stale_s, self.invalid_s, self.max_tilt = stale_s, invalid_s, max_tilt_deg
        self.hover_thrust, self.mass = hover_thrust, mass
        self.clock, self.sleep, self.out = clock, sleep, out
        self.external_abort = external_abort or (lambda: None)
        self.on_offboard = on_offboard or (lambda: None)
        self.status = "not started"
        self.sent = 0
        self.rows = []

    def _send(self, sp):
        q = (sp.orientation.x, sp.orientation.y, sp.orientation.z, sp.orientation.w)
        q_ned = attitude_ned_frd(q)
        w = rates_frd((sp.body_rate.x, sp.body_rate.y, sp.body_rate.z))
        u = thrust_normalized(sp.thrust, self.hover_thrust, self.mass)
        self.v.m.mav.set_attitude_target_send(
            0, self.v.m.target_system, self.v.m.target_component, ATT_TYPE_MASK_ALL,
            [float(c) for c in q_ned], float(w[0]), float(w[1]), float(w[2]), u)
        self.sent += 1
        return q_ned, w, u

    def _check(self, sp, arrived, invalid_since, now):
        ext = self.external_abort()
        if ext:
            return f"watchdog: {ext}"
        if sp is None or arrived is None or now - arrived > self.stale_s:
            return f"no new MPC setpoint for more than {self.stale_s} s"
        if invalid_since is not None and now - invalid_since > self.invalid_s:
            return f"MPC reported invalid (solver failing) for more than {self.invalid_s} s"
        tilt = tilt_deg((sp.orientation.x, sp.orientation.y, sp.orientation.z, sp.orientation.w))
        if tilt > self.max_tilt:
            return f"setpoint tilt {tilt:.0f} deg > --max-tilt-deg {self.max_tilt}"
        return None

    def wait_for_setpoints(self, timeout=10.0):
        deadline = self.clock() + timeout
        while self.clock() < deadline:
            sp, arrived = self.get_setpoint()
            self.v.poll()
            if sp is not None and arrived is not None and self.clock() - arrived < self.stale_s and sp.valid:
                return sp
            self.sleep(0.02)
        return None

    def run(self, max_duration=120.0):
        invalid_since = None
        try:
            # PX4 needs setpoints streaming before it accepts OFFBOARD
            self.out("streaming MPC setpoints ...")
            t0 = self.clock()
            while self.clock() - t0 < 1.5:
                sp, arrived = self.get_setpoint()
                reason = self._check(sp, arrived, None, self.clock())
                if reason:
                    return self._abort(reason)
                self._send(sp)
                self.v.poll()
                self.sleep(self.dt)
            for _ in range(5):
                self.v.set_offboard()
                t1 = self.clock()
                while self.clock() - t1 < 0.5 and self.v.main_mode != PX4_MAIN_OFFBOARD:
                    sp, arrived = self.get_setpoint()
                    if sp is not None:
                        self._send(sp)
                    self.v.poll()
                    self.sleep(self.dt)
                if self.v.main_mode == PX4_MAIN_OFFBOARD:
                    break
            else:
                return self._abort("PX4 did not enter OFFBOARD")

            self.on_offboard()
            self.out(f"OFFBOARD: the MPC is flying (max {max_duration:.0f} s; Ctrl-C hands back to AUTO.LOITER)")
            t_start = self.clock()
            next_t = t_start
            while self.clock() - t_start < max_duration:
                now = self.clock()
                sp, arrived = self.get_setpoint()
                if sp is not None and not sp.valid:
                    invalid_since = invalid_since if invalid_since is not None else now
                else:
                    invalid_since = None
                reason = self._check(sp, arrived, invalid_since, now)
                if reason:
                    return self._abort(reason)
                self.v.poll()
                if self.v.main_mode != PX4_MAIN_OFFBOARD:
                    self.status = f"stopped: PX4 left OFFBOARD (main mode {self.v.main_mode}); not fighting it"
                    self.out(self.status)
                    return self.status
                q_ned, w, u = self._send(sp)
                pos = ned_to_enu(self.v.pos_ned) if self.v.pos_ned is not None else [np.nan] * 3
                self.rows.append((now - t_start, *pos, *q_ned, *w, u, sp.thrust))
                next_t += self.dt
                self.sleep(max(0.0, next_t - self.clock()))
            self.status = f"completed: reached --max-duration {max_duration:.0f} s"
            return self.status
        except KeyboardInterrupt:
            self.status = "stopped: Ctrl-C"
            self.out("\nCtrl-C")
            return self.status
        finally:
            if self.v.main_mode == PX4_MAIN_OFFBOARD or self.status.startswith(("aborted", "stopped: Ctrl")):
                self.out("commanding AUTO.LOITER")
                for _ in range(5):
                    self.v.set_loiter()
                    self.sleep(0.1)

    def _abort(self, reason):
        self.status = f"aborted: {reason}"
        self.out(f"ABORT: {reason}")
        return self.status
