"""
The in-flight replanning that ros2_nodes/replan_node.py runs, without ROS, so
it is tested on its own (Trajectory/test_replan_session.py).

It wraps replanner.Replanner and gate_map.GateMap for a node that PUBLISHES
plans (MinSnapTrajectory) to mpc_node instead of feeding an MPC directly:

  map       GateMap seeded with a PRIOR course (the manifest, exact or
            perturbed by prior_sigma), corrected by camera gate poses
            (add_detections, world frame, e.g. ros_camera_pose_fallback.py)
  prepare   before engage: plan from the hover pose with the current map.
            The first plan is from rest, so the planner optimizes the segment
            times; that takes seconds and must not happen at engage.
  engage    re-stamps the prepared plan to start now (valid: the drone is
            still at rest where the plan starts), or plans again if the drone
            moved more than start_tol_m since
  tick      every control step: Replanner.update (gate passes, replans when a
            gate estimate moves, the window runs out, ...). Returns a NEW plan
            when there is one, else None.

Every plan starts from the CURRENT REFERENCE state at its t0 (Replanner does
this), so publishing it with header.stamp = t0 keeps mpc_node's reference
continuous: mpc_node evaluates the new plan at now - t0, where it agrees with
the old one, however long the planning took.

All times are seconds on one clock (sim time with use_sim_time).
"""

from dataclasses import dataclass

import numpy as np

from gate_path import Gate
from gate_map import GateMap, gate_transform
from min_snap_planner import gate_from_pose, trajectory_points
from replanner import Replanner


def perturbed_prior(true_gates, sigma, seed, yaw_std_deg=5.0):
    """The course as a rough prior: each center moved by N(0, sigma) per axis, yaw by N(0, yaw_std_deg).
    Same construction as simulate_replanning.make_prior (seeded the same way)."""
    if sigma <= 0:
        return [Gate(g.name, g.T_world_gate.copy(), side=g.side) for g in true_gates]
    rng = np.random.default_rng(seed + 1)
    prior = []
    for g in true_gates:
        c = g.center + rng.normal(0.0, sigma, 3)
        yaw = np.arctan2(g.normal[1], g.normal[0]) + np.radians(rng.normal(0.0, yaw_std_deg))
        prior.append(Gate(g.name, gate_transform(c, [np.cos(yaw), np.sin(yaw), 0.0]), side=g.side))
    return prior


@dataclass
class PublishedPlan:
    t0: float              # header.stamp: time of the plan's t = 0
    times: np.ndarray      # seconds from t0
    states: np.ndarray     # 12 x M
    thrust: np.ndarray     # M
    reason: str
    index: int             # 1 = first plan
    gate_names: list       # route entries it flies through, in order


class ReplanSession:
    def __init__(self, prior_gates, v_max=2.0, a_max=2.0, laps=1, close_loop=True,
                 start_tol_m=0.5, gate_side=1.5, min_hits=3, prior_sigma=0.5,
                 max_detection_range_m=None, allow_new_gates=True, **replanner_kw):
        """max_detection_range_m: camera gate poses farther than this from the drone are
        ignored (None or <= 0: no limit). With the Starling's wide C29 camera a gate 12 m
        away is ~60 px across, and its PnP pose is off by up to ~1 m and ~100 deg in yaw;
        fused in, it turned gate_1 sideways and every replan failed (Isaac, 2026-11-07)."""
        self.prior_gates = prior_gates
        self.v_max, self.a_max, self.laps, self.close_loop = v_max, a_max, laps, close_loop
        self.start_tol, self.gate_side = start_tol_m, gate_side
        self.max_range = max_detection_range_m if max_detection_range_m and max_detection_range_m > 0 else None
        self.n_far_ignored = 0
        self.map_kw = dict(min_hits=min_hits, prior_sigma=prior_sigma, allow_new_gates=allow_new_gates)
        self.replanner_kw = replanner_kw
        self.map = None
        self.replanner = None
        self.engaged = False
        self.n_published = 0
        self.n_detections = 0
        self.events = []                    # (t, text)
        self._ev_idx = 0                    # replanner.events already copied into self.events

    # ------------------------------------------------------------------ map
    def _ensure_map(self, start_pos):
        if self.map is None:
            self.map = GateMap(np.asarray(start_pos, dtype=float), prior=self.prior_gates, **self.map_kw)

    def add_detections(self, t, poses, drone_pos):
        """poses: [(position xyz, orientation xyzw)] in the world frame (PoseArray; identity = normal unreliable)."""
        drone_pos = np.asarray(drone_pos, dtype=float)
        self._ensure_map(drone_pos)
        dets = []
        for p, q in poses:
            P = type("P", (), {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])})()
            g = gate_from_pose(P, tuple(q), side=self.gate_side, viewer_pos=drone_pos)
            rng = float(np.linalg.norm(g.center - drone_pos))
            if self.max_range is not None and rng > self.max_range:
                self.n_far_ignored += 1
                continue
            dets.append({"center": g.center, "normal": g.normal, "normal_reliable": g.normal_reliable,
                         "center_std": np.nan, "normal_std": np.nan,
                         "range": rng, "side": self.gate_side})
        if dets:
            self.map.update(dets, t)
            self.n_detections += len(dets)

    def map_snapshot(self):
        """[(name, center, normal, normal_reliable, n_obs)] of the confirmed gates, in course order."""
        if self.map is None:
            return []
        return [(mg.name, mg.center.copy(), mg.normal.copy(), mg.normal_reliable, mg.n_obs)
                for mg in self.map.ordered()]

    # ------------------------------------------------------------------ plans
    def _new_replanner(self, pos, yaw):
        self._ensure_map(pos)
        self.replanner = Replanner(self.map, np.asarray(pos, dtype=float), float(yaw), v_max=self.v_max,
                                   a_max=self.a_max, laps=self.laps, close_loop=self.close_loop,
                                   verbose=False, **self.replanner_kw)

    def _log(self, t, text):
        self.events.append((t, text))

    def prepare(self, t, pos, yaw):
        """Before engage: (re)plan from the hover pose if there is no plan yet or the map has moved
        a planned gate. Returns a short summary string when a plan was made, else None."""
        if self.replanner is None:
            self._new_replanner(pos, yaw)
        elif self.replanner.plans:
            # Not flying yet: keep the plan's clock at "now" (the drone is still at rest at its
            # start), or the replanner would count the first gate as MISSED while we hover.
            self.replanner.plans[-1].t0 = t
        n = len(self.replanner.plans)
        self.replanner.update(t, pos)
        self._copy_events()
        if len(self.replanner.plans) == n:
            return None
        p = self.replanner.plans[-1]
        text = (f"plan ready ({p.reason}): {p.traj.duration:.2f} s through "
                f"{', '.join(self.map.gates[g].name for g in p.gate_ids)} from {np.round(pos, 2)}")
        self._log(t, text)
        return text

    def engage(self, t, pos, yaw, clock=None):
        """OFFBOARD confirmed at time t: returns the first PublishedPlan, starting now."""
        pos = np.asarray(pos, dtype=float)
        plan = self.replanner.plans[-1] if self.replanner and self.replanner.plans else None
        if plan is None or np.linalg.norm(plan.traj.evaluate(0.0, 0) - pos) > self.start_tol:
            moved = "no plan yet" if plan is None else \
                f"drone {np.linalg.norm(plan.traj.evaluate(0.0, 0) - pos):.2f} m from the prepared start"
            self._log(t, f"engage: {moved}; planning now")
            self._new_replanner(pos, yaw)
            self._ev_idx = 0
            self.replanner.update(t, pos)
            self._copy_events()
            plan = self.replanner.plans[-1]
            if clock is not None:
                t = clock()                       # from rest: valid to start it when it is ready
        plan.t0 = t                               # the drone is at rest at the plan's start
        self.replanner._last_plan_t = t
        self.engaged = True
        self._log(t, f"engaged: plan #{len(self.replanner.plans)} starts at t={t:.2f}")
        return self._published(plan)

    def tick(self, t, pos):
        """Every step after engage: returns a new PublishedPlan when the replanner made one, else None."""
        if not self.engaged:
            return None
        n = len(self.replanner.plans)
        self.replanner.update(t, np.asarray(pos, dtype=float))
        self._copy_events()
        if len(self.replanner.plans) == n:
            return None
        return self._published(self.replanner.plans[-1])

    def _copy_events(self):
        for tt, text in self.replanner.events[self._ev_idx:]:
            self._log(tt, text)
        self._ev_idx = len(self.replanner.events)

    def _published(self, plan):
        ts, states, thrust = trajectory_points(plan.ref)
        self.n_published += 1
        names = [self.map.gates[g].name for g in plan.gate_ids]
        return PublishedPlan(plan.t0, ts, states, thrust, plan.reason, len(self.replanner.plans), names)

    # ------------------------------------------------------------------ status
    def progress(self):
        """(route entries passed, route length)."""
        if self.replanner is None:
            return 0, 0
        return self.replanner.progress, len(self.replanner.route())

    def finished(self, t):
        return self.replanner is not None and self.replanner.finished(t)

    def passes(self):
        return list(self.replanner.passes) if self.replanner else []
