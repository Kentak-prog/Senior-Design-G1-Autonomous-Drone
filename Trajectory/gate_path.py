"""
Gate poses (from vision/) -> waypoints -> minimum-snap trajectory.

Inputs come from the vision stack in one of two ways:
  - a manifest written by workspace/spawn_test_gates.py (test_gates.json),
    loaded through vision/eval_isaac_gates.py so the yaw convention matches
    the evaluator exactly, or
  - PnP detections lifted into the world with vision/gate_pose.py
    (gate_pose_world(det, T_world_cam)), e.g. from GatePoseNode.
Either way each gate ends up as a 4x4 T_world_gate in the vision gate-model
convention: column 0 = gate +X (right), column 1 = gate +Y (DOWN), column 2 =
gate +Z = FLIGHT DIRECTION through the gate. See vision/gate_pose.py and
vision/HANDOFF.md before changing anything about that.

Waypoints per gate come from gate_pose.approach_waypoints() (pre / center /
post on the gate normal). When a detection's normal is unreliable
(GateDetection.normal_is_reliable is False) the HANDOFF rule applies: hand
the planner the center alone, with no direction constraint.

World frame: Isaac Sim, Z-up, meters.
"""

import os
import sys
from dataclasses import dataclass

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("vision",):
    _p = os.path.join(_REPO, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gate_pose import approach_waypoints, DEFAULT_GATE_SIDE   # noqa: E402

from min_snap import fit_to_limits                             # noqa: E402


@dataclass
class Gate:
    name: str
    T_world_gate: np.ndarray            # 4x4, vision gate-model convention
    side: float = DEFAULT_GATE_SIDE
    normal_reliable: bool = True        # False -> plan through the center only

    @property
    def center(self) -> np.ndarray:
        return self.T_world_gate[:3, 3]

    @property
    def x_axis(self) -> np.ndarray:
        return self.T_world_gate[:3, 0]

    @property
    def y_axis(self) -> np.ndarray:
        return self.T_world_gate[:3, 1]

    @property
    def normal(self) -> np.ndarray:
        """Unit flight direction through the gate (gate +Z)."""
        n = self.T_world_gate[:3, 2]
        return n / np.linalg.norm(n)

    def corners(self) -> np.ndarray:
        h = self.side / 2.0
        return np.array([self.center + a * h * self.x_axis + b * h * self.y_axis
                         for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])


def gates_from_manifest(path, gate_side: float = DEFAULT_GATE_SIDE):
    """test_gates.json (spawn_test_gates.py) -> list of Gate, in manifest (flight) order."""
    from eval_isaac_gates import load_manifest, T_world_gate_from_spec
    return [Gate(g["name"], T_world_gate_from_spec(g["x"], g["y"], g["z"], g["yaw_deg"]),
                 side=g.get("side", gate_side))
            for g in load_manifest(path)]


# ---------------------------------------------------------------------------
# Waypoints
# ---------------------------------------------------------------------------

def gate_waypoints(gates, start, standoff: float = 1.0, finish_dist: float = 4.0,
                   min_spacing_frac: float = 0.4):
    """
    start -> [pre, center, post] per gate -> finish.

    Uses vision/gate_pose.approach_waypoints() for the pre/center/post points.
    The standoff shrinks when neighbouring gates are close so a post point
    does not land past the next gate. finish_dist leaves room to brake to a
    stop after the last gate (too short and the end needs a big snap spike).

    Returns a list of dicts: {'name', 'position', 'type', 'gate'} where type is
    start / pre / center / post / finish (plus 'avoid' if the planner inserts any).
    """
    start = np.asarray(start, dtype=float)
    wps = [{'name': 'start', 'position': start, 'type': 'start', 'gate': None}]

    for i, gate in enumerate(gates):
        if not gate.normal_reliable:
            wps.append({'name': gate.name, 'position': gate.center.copy(), 'type': 'center', 'gate': i})
            continue

        prev_pt = gates[i - 1].center if i > 0 else start
        next_pt = gates[i + 1].center if i < len(gates) - 1 else None
        d_pre = min(standoff, min_spacing_frac * np.linalg.norm(gate.center - prev_pt))
        d_post = (min(standoff, min_spacing_frac * np.linalg.norm(next_pt - gate.center))
                  if next_pt is not None else standoff)

        pre, center, _ = approach_waypoints(gate.T_world_gate, standoff=d_pre)
        _, _, post = approach_waypoints(gate.T_world_gate, standoff=d_post)
        wps.append({'name': f'{gate.name}_pre', 'position': pre, 'type': 'pre', 'gate': i})
        wps.append({'name': gate.name, 'position': center, 'type': 'center', 'gate': i})
        if next_pt is not None:
            wps.append({'name': f'{gate.name}_post', 'position': post, 'type': 'post', 'gate': i})

    # Finish: straight out of the last gate (or along the last leg if its normal is unknown)
    last = gates[-1]
    out_dir = last.normal if last.normal_reliable else \
        (last.center - wps[-2]['position']) / np.linalg.norm(last.center - wps[-2]['position'])
    wps.append({'name': 'finish', 'position': last.center + finish_dist * out_dir,
                'type': 'finish', 'gate': len(gates) - 1})
    return wps


# ---------------------------------------------------------------------------
# Gate crossing checks
# ---------------------------------------------------------------------------

def gate_crossings(times, pos, vel, gates):
    """
    For each gate, every time the path crosses the gate plane near the gate
    (within one gate side of its center). Works on any sampled path: the
    planned trajectory or the flown one.

    Returns [(gate, [ {time, offset (gate x,y), inside, forward, angle_deg, speed}, ... ]), ...]
    """
    times, pos, vel = map(np.asarray, (times, pos, vel))
    results = []
    for gate in gates:
        d = pos - gate.center
        local = np.column_stack((d @ gate.x_axis, d @ gate.y_axis, d @ gate.normal))
        z = local[:, 2]
        idx = np.nonzero((z[:-1] * z[1:] <= 0) & (z[:-1] != z[1:]))[0]
        crossings = []
        for k in idx:
            a = z[k] / (z[k] - z[k + 1])
            xy = (1 - a) * local[k, :2] + a * local[k + 1, :2]
            if np.all(np.abs(xy) < gate.side):
                v = (1 - a) * vel[k] + a * vel[k + 1]
                speed = float(np.linalg.norm(v))
                cosang = v @ gate.normal / max(speed, 1e-9)
                crossings.append({'time': float((1 - a) * times[k] + a * times[k + 1]),
                                  'offset': xy,
                                  'inside': bool(np.all(np.abs(xy) < gate.side / 2)),
                                  'forward': bool(cosang > 0),
                                  'angle_deg': float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))),
                                  'speed': speed})
        results.append((gate, crossings))
    return results


def _first_bad_crossing(traj, gates, clearance):
    # Gates without a trustworthy normal have no trustworthy plane either, so skip them
    trusted = [g for g in gates if g.normal_reliable]
    times, pos, vel, *_ = traj.sample(0.005)
    for gate, crossings in gate_crossings(times, pos, vel, trusted):
        for c in crossings:
            if c['inside'] and c['forward']:
                continue
            edge_dist = np.max(np.abs(c['offset'])) - gate.side / 2
            if c['inside'] or edge_dist < clearance:
                return gate, c, edge_dist
    return None


def _avoidance_point(gate, p0, p1, clearance):
    """
    Extra waypoint for a segment p0 -> p1 whose trajectory crosses `gate` badly.

    Usually the polynomial has just bulged away from the straight line, and
    the line's midpoint pulls it back. But when the straight line ITSELF goes
    through the gate's plane near the frame (two gates side by side, with the
    path looping back past one), the midpoint would sit on the same bad line.
    Then use the point where the line meets the gate plane, pushed out past
    the frame edge by `clearance` (+ a little margin, so the curve through the
    point doesn't land exactly on the clearance limit and re-trigger).
    """
    push_dist = gate.side / 2 + clearance + 0.1
    z0, z1 = (p - gate.center for p in (p0, p1))
    d0, d1 = z0 @ gate.normal, z1 @ gate.normal
    if d0 * d1 < 0:
        a = d0 / (d0 - d1)
        hit = (1 - a) * z0 + a * z1
        offset = np.array([hit @ gate.x_axis, hit @ gate.y_axis])
        if np.max(np.abs(offset)) < gate.side / 2 + clearance:
            if np.max(np.abs(offset)) < 1e-6:
                offset = np.array([1.0, 0.0])
            push = offset * push_dist / np.max(np.abs(offset))
            return gate.center + push[0] * gate.x_axis + push[1] * gate.y_axis
    return 0.5 * (p0 + p1)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def plan_gate_trajectory(gates, start, v_max: float = 4.0, a_max: float = 6.0,
                         standoff: float = 1.0, finish_dist: float = 4.0,
                         clearance: float = 0.5, max_iters: int = 10, verbose: bool = True):
    """
    Gates -> waypoints -> min-snap trajectory that respects v_max / a_max.

    Every gate with a reliable normal is crossed perpendicular to its plane
    (velocity direction constraint at the center). If the polynomial bulges
    out and crosses some gate plane backwards, or within `clearance` of a
    frame, the midpoint of that segment's straight line is inserted as an
    extra waypoint and the trajectory is re-solved.

    Returns (traj, waypoints).
    """
    wps = gate_waypoints(gates, start, standoff=standoff, finish_dist=finish_dist)

    traj = None
    for _ in range(max_iters):
        points = np.array([w['position'] for w in wps])
        fixed_dir = {i: gates[w['gate']].normal for i, w in enumerate(wps)
                     if w['type'] == 'center' and gates[w['gate']].normal_reliable}
        traj = fit_to_limits(points, fixed_dir, v_max=v_max, a_max=a_max)

        bad = _first_bad_crossing(traj, gates, clearance)
        if bad is None:
            return traj, wps

        gate, c, edge_dist = bad
        seg = int(traj.segment_index(c['time']))
        point = _avoidance_point(gate, wps[seg]['position'], wps[seg + 1]['position'], clearance)
        wps.insert(seg + 1, {'name': f'avoid_{gate.name}', 'position': point,
                             'type': 'avoid', 'gate': None})
        if verbose:
            problem = "backwards through" if c['inside'] else f"{edge_dist:.2f} m from the frame of"
            print(f"[plan] trajectory passes {problem} {gate.name}; inserted waypoint "
                  f"{np.round(point, 2)} between {wps[seg]['name']} and {wps[seg + 2]['name']}")

    print("[plan] WARNING: trajectory still crosses a gate badly after max iterations.")
    return traj, wps
