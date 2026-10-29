"""
In-flight replanning: keeps the NMPC reference up to date as the gate map
(gate_map.py) learns more about the course, for one or more laps.

Every control step, call update(t, est_pos) then build_reference_horizon(t, N, dt).

The route
  The course is the map's confirmed gates in course order. For `laps` laps the
  route is that order repeated `laps` times, then (close_loop=True) back
  through the first gate, which is the start/finish line:
      laps=1:  g0 g1 g2 g3 g0
      laps=2:  g0 g1 g2 g3 g0 g1 g2 g3 g0
  `progress` counts route entries passed; the route ahead is route[progress:].
  A lap is complete each time the drone passes the start/finish gate again.

What it does
  - Tracks gate passes (the estimated position crosses the next gate's plane
    near its opening), or marks a gate missed once the plan is miss_timeout
    seconds past the time it should have reached it.
  - Plans a rolling window of the next `horizon_gates` route entries (a whole
    multi-lap route at once would be dozens of waypoints and slow to solve).
    Each window ends at rest past its last gate, but the window is extended
    long before the drone gets there, so it never actually stops.
  - Replans when:
      * the planned window is running out (<= extend_at gates left) and more route remains
      * a gate in the window was seen for the first time / the route changed
      * a planned gate's estimate moved more than max(pos_tol, pos_tol_frac * distance)
        or turned more than ang_tol
      * a gate's normal became reliable (center-only -> full pre/center/post)
  - Does NOT replan inside the commit zone (the last `commit_dist` meters
    before the target gate): the drone is lined up, the gate is usually too
    close to see whole anyway, and a late swerve is worse than a small error.
  - Each new plan starts from the CURRENT REFERENCE state (position, velocity,
    acceleration, jerk, yaw at time t) rather than the noisy estimate, so the
    reference the controller sees is continuous through every replan.
  - Every known gate outside the window is handed to the planner as an
    obstacle, so the path keeps clear of frames it flies past (e.g. the backs
    of later gates on the way round to the start).
  - Before any gate is known it hovers and slowly yaws to look for one.

Limitation: without a prior course map, gates found partway through a lap are
slotted into the course order by distance from the start. If one turns up
"before" a gate already passed, the progress count shifts. Give the map a
prior course (GateMap(prior=...)) for multi-lap racing.
"""

from dataclasses import dataclass

import numpy as np

from gate_path import plan_gate_trajectory
from trajectory_reference import TrajectoryReference
from drone_constraints import drone_constraints     # importable: trajectory_reference puts control/ on sys.path


@dataclass
class Plan:
    t0: float                  # sim time the plan starts
    traj: object               # MinSnapTrajectory (local time 0 = t0)
    ref: TrajectoryReference
    wps: list
    route_start: int           # route index of the first gate this plan flies through
    gate_ids: list             # mapped gates it flies through, in order (may repeat across laps)
    snapshot: dict             # gate_id -> (center, normal, normal_reliable) used for this plan
    reason: str

    def local(self, t):
        return t - self.t0

    def end_time(self):
        return self.t0 + self.traj.duration

    def gate_time(self, route_idx):
        """Sim time this plan reaches route entry `route_idx` (None if it isn't in this plan)."""
        k = route_idx - self.route_start
        if not 0 <= k < len(self.gate_ids):
            return None
        centers = [i for i, w in enumerate(self.wps) if w['type'] == 'center']
        return self.t0 + self.traj.t_breaks[centers[k]]


class HoldReference:
    """
    Hover at a point. If no gate has been confirmed after `search_delay`
    seconds, start yawing at `yaw_rate` to look for one (the map needs a few
    sightings to confirm a gate, so the delay avoids spinning at takeoff).
    """

    def __init__(self, pos, yaw, t0, yaw_rate=0.0, search_delay=1.0):
        self.pos, self.yaw, self.t0 = np.asarray(pos, dtype=float), yaw, t0
        self.yaw_rate, self.search_delay = yaw_rate, search_delay

    def state(self, t):
        searching = t - self.t0 - self.search_delay
        x = np.zeros(12)
        x[0:3] = self.pos
        x[5] = self.yaw + self.yaw_rate * max(searching, 0.0)
        x[11] = self.yaw_rate if searching > 0 else 0.0
        return x


class Replanner:
    def __init__(self, gate_map, start_pos, start_yaw, v_max=None, a_max=None, m=drone_constraints["mass"],
                 laps=1, close_loop=True, horizon_gates=5, extend_at=2,
                 commit_dist=1.5, pos_tol=0.10, pos_tol_frac=0.02, ang_tol_deg=5.0, min_interval=0.2,
                 replan=True, search_yaw_rate=0.5, miss_timeout=2.0, verbose=True):
        self.map = gate_map
        self.v_max, self.a_max, self.m = v_max, a_max, m
        self.laps, self.close_loop = laps, close_loop
        self.horizon_gates, self.extend_at = horizon_gates, extend_at
        self.commit_dist, self.pos_tol, self.ang_tol = commit_dist, pos_tol, np.radians(ang_tol_deg)
        self.pos_tol_frac = pos_tol_frac
        self.min_interval = min_interval
        self.replan_enabled = replan
        self.search_yaw_rate = search_yaw_rate
        self.miss_timeout = miss_timeout
        self.verbose = verbose

        self.plans = []
        self.hold = HoldReference(start_pos, start_yaw, 0.0, yaw_rate=search_yaw_rate)
        self.progress = 0                # route entries passed
        self.passes = []                 # (t, route index, gate_id, how)
        self.events = []                 # (t, text)
        self._prev_s = None
        self._near_unreliable = False
        self._last_plan_t = -np.inf

    # ------------------------------------------------------------------ route
    def route(self):
        """Gate ids for the whole race, in flight order (repeats across laps)."""
        course = [mg.gate_id for mg in self.map.ordered()]
        if not course:
            return []
        r = course * self.laps
        if self.close_loop:
            r = r + course[:1]
        return r

    def route_ahead(self):
        return self.route()[self.progress:]

    def gates_ahead(self):
        return [self.map.gates[gid] for gid in self.route_ahead()]

    def lap_of(self, route_idx):
        """1-based lap number a route entry belongs to (the closing pass of g0 ends the last lap)."""
        n = len(self.map.ordered())
        return min(route_idx // max(n, 1), self.laps - 1) + 1 if n else 1

    # ------------------------------------------------------------------ helpers
    def _log(self, t, text):
        self.events.append((t, text))
        if self.verbose:
            print(f"[replan] t={t:6.2f}s  {text}")

    def active(self, t):
        """The plan (or hold) the controller is following at time t."""
        return self.plans[-1] if self.plans else self.hold

    def _ref_state(self, t):
        a = self.active(t)
        return a.ref.reference_state(a.local(t)) if isinstance(a, Plan) else a.state(t)

    # ------------------------------------------------------------------ gate passing
    def _update_passed(self, t, pos):
        ahead = self.gates_ahead()
        if not ahead:
            return
        mg = ahead[0]
        g = mg.as_gate()
        d = pos - g.center

        # Missed: the current plan should have reached this gate a while ago. Move
        # on rather than circle back forever (the flown-path report will show it).
        due = self.plans[-1].gate_time(self.progress) if self.plans else None
        if due is not None and t > due + self.miss_timeout:
            self._mark_passed(t, mg, f"MISSED -- {self.miss_timeout:.0f} s past its planned time")
            return

        if mg.normal_reliable:
            s = d @ g.normal                                   # > 0: past the gate plane
            lateral = np.linalg.norm(d - s * g.normal)
            crossed = self._prev_s is not None and self._prev_s < 0.0 <= s
            if crossed:
                # Only a crossing near the opening counts. The path can cross a gate's
                # plane off to the side (looping around it) before going through; the
                # planner keeps those >= 0.5 m outside the frame, i.e. >= side/2 + 0.5.
                if lateral < g.side / 2 + 0.25:
                    how = "through" if lateral < g.side / 2 else f"{lateral:.2f} m off-center"
                    self._mark_passed(t, mg, how)
                    return
                self._prev_s = None                            # went past beside it
                return
            # Remember "in front of the gate" only near its approach line. Behind the
            # gate (s >= 0) there is nothing to remember -- on lap 2+ the drone often
            # starts the approach from behind a gate and has to loop round to its front.
            self._prev_s = s if (s < 0.0 and lateral < 2 * g.side) else None
        else:
            # No trustworthy plane: passed once we got close and are moving away again
            dist = np.linalg.norm(d)
            if dist < 0.75:
                self._near_unreliable = True
            elif self._near_unreliable and dist > 1.0:
                self._mark_passed(t, mg, "center-only gate")

    def _mark_passed(self, t, mg, how):
        idx = self.progress
        self.progress += 1
        self.passes.append((t, idx, mg.gate_id, how))
        self._prev_s = None
        self._near_unreliable = False
        n = len(self.map.ordered())
        lap_note = ""
        if self.close_loop and n and idx > 0 and idx % n == 0:
            lap_note = f"  -- LAP {idx // n} COMPLETE"
        self._log(t, f"passed {mg.name} [route {idx}, lap {self.lap_of(idx)}] ({how}){lap_note}")

    def _in_commit_zone(self, pos):
        ahead = self.gates_ahead()
        if not ahead or not ahead[0].normal_reliable:
            return False
        g = ahead[0].as_gate()
        d = pos - g.center
        s = d @ g.normal
        return -self.commit_dist < s < 0.3 and np.linalg.norm(d - s * g.normal) < 2 * g.side

    # ------------------------------------------------------------------ replan decision
    def _window(self):
        return self.route_ahead()[:self.horizon_gates]

    def _replan_reason(self, t, pos):
        ahead = self.route_ahead()
        if not ahead:
            return None
        if not self.plans:
            return "first plan"
        if not self.replan_enabled:
            return None
        if t - self._last_plan_t < self.min_interval or self._in_commit_zone(pos):
            return None

        plan = self.plans[-1]
        planned = plan.gate_ids[self.progress - plan.route_start:]
        window = self._window()

        if len(planned) <= self.extend_at and len(ahead) > len(planned):
            return f"extend horizon ({len(planned)} planned gate(s) left)"
        if window[:len(planned)] != planned:
            new = [self.map.gates[gid].name for gid in window if gid not in plan.gate_ids]
            return ("new gate " + ", ".join(new)) if new else "route changed"
        for gid in dict.fromkeys(planned):            # each gate once, in order
            mg = self.map.gates[gid]
            c0, n0, rel0 = plan.snapshot[gid]
            if mg.normal_reliable and not rel0:
                return f"{mg.name} normal now reliable"
            moved = np.linalg.norm(mg.center - c0)
            # A far gate's estimate wobbling by 10-20 cm changes little; scale the tolerance
            tol = max(self.pos_tol, self.pos_tol_frac * np.linalg.norm(mg.center - pos))
            if moved > tol:
                return f"{mg.name} moved {moved:.2f} m"
            turned = np.arccos(np.clip(mg.normal @ n0, -1.0, 1.0))
            if mg.normal_reliable and turned > self.ang_tol:
                return f"{mg.name} turned {np.degrees(turned):.1f} deg"
        return None

    def _start_state(self, t):
        """(position, {1: vel, 2: acc, 3: jerk}, yaw) of the reference at time t."""
        a = self.active(t)
        if isinstance(a, Plan):
            tau = a.local(t)
            if tau >= a.traj.duration:                          # finished: at rest at the finish
                return a.traj.evaluate(a.traj.duration, 0), {}, a.ref.reference_state(tau)[5]
            pos = a.traj.evaluate(tau, 0)
            ss = {r: a.traj.evaluate(tau, r) for r in (1, 2, 3)}
            return pos, ss, a.ref.reference_state(tau)[5]
        x = a.state(t)
        return x[0:3], {}, x[5]

    def _replan(self, t, reason):
        window = self._window()
        gates = [self.map.gates[gid].as_gate() for gid in window]
        # Every other known gate is an obstacle: passed ones, and later ones whose
        # backs the path may fly past on the way round
        obstacles = [mg.as_gate() for mg in self.map.ordered()
                     if mg.gate_id not in window and mg.normal_reliable]
        start, ss, yaw = self._start_state(t)
        moving = any(np.linalg.norm(v) > 1e-6 for v in ss.values())
        try:
            traj, wps = plan_gate_trajectory(gates, start, v_max=self.v_max, a_max=self.a_max,
                                             verbose=False, start_state=ss if moving else None,
                                             obstacle_gates=obstacles)
        except np.linalg.LinAlgError as e:
            self._log(t, f"replan FAILED ({reason}): {e}; keeping the current plan")
            return
        ref = TrajectoryReference(traj, start_yaw=yaw, m=self.m, yaw_blend_time=0.5 if moving else None)
        snapshot = {gid: (self.map.gates[gid].center.copy(), self.map.gates[gid].normal.copy(),
                          self.map.gates[gid].normal_reliable) for gid in window}
        self.plans.append(Plan(t, traj, ref, wps, self.progress, list(window), snapshot, reason))
        self._last_plan_t = t
        remaining = len(self.route_ahead())
        self._log(t, f"plan #{len(self.plans)} ({reason}): window {len(window)} of {remaining} "
                     f"route gate(s) left, lap {self.lap_of(self.progress)}/{self.laps}")

    # ------------------------------------------------------------------ public API
    def update(self, t, est_pos):
        est_pos = np.asarray(est_pos, dtype=float)
        self._update_passed(t, est_pos)
        reason = self._replan_reason(t, est_pos)
        if reason:
            self._replan(t, reason)

    def reference_state(self, t):
        return self._ref_state(t)

    def build_reference_horizon(self, t0, N, dt):
        """12 x (N+1) reference from the plan active at t0 (same signature as Drone_Race_Simulation)."""
        a = self.active(t0)
        if isinstance(a, Plan):
            return a.ref.build_reference_horizon(a.local(t0), N, dt)
        return np.column_stack([a.state(t0 + k * dt) for k in range(N + 1)])

    def finished(self, t):
        a = self.active(t)
        return isinstance(a, Plan) and self.progress > 0 and not self.route_ahead() \
            and t >= a.end_time()

    def lap_times(self):
        """Seconds between successive passes of the start/finish gate (route indices 0, n, 2n, ...)."""
        n = len(self.map.ordered())
        if not n:
            return []
        t_line = [t for (t, idx, _, how) in self.passes if idx % n == 0 and not how.startswith("MISSED")]
        return list(np.diff(t_line))
