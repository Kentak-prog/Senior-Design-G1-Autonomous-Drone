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
  - A replan (in flight, or while hovering before the start) keeps the previous plan's timing: every waypoint the two
    plans share (matched by name, in order) keeps its arrival time, new ones are
    placed by distance between them, and the result is rescaled to the limits.
    Without this a moving-start plan falls back to distance-proportional timing,
    which made the rest of the 3-gate closed lap about 2x slower (2026-11-05).
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
        """t0: when the hover starts; None = at the first state() call. The clock can be
        sim time (thousands of seconds in Isaac): a fixed t0 = 0 there made the "search"
        yaw 0.5 rad/s x 2400 s = 1210 rad, the first plan inherited it, and the MPC's first
        solve asked for a 53 deg tilt (Stage 4R, 2026-11-07)."""
        self.pos, self.yaw, self.t0 = np.asarray(pos, dtype=float), yaw, t0
        self.yaw_rate, self.search_delay = yaw_rate, search_delay

    def state(self, t):
        if self.t0 is None:
            self.t0 = t
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
                 replan=True, search_yaw_rate=0.5, miss_timeout=2.0, verbose=True, fail_backoff=1.0,
                 yaw_mode="velocity"):
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
        self.fail_backoff = fail_backoff
        if yaw_mode not in ("velocity", "next_gate"):
            raise ValueError(f"unknown yaw_mode {yaw_mode!r} (use 'velocity' or 'next_gate')")
        self.yaw_mode = yaw_mode            # next_gate: look at the next gate (keeps gates in the camera)
        self.verbose = verbose

        self.plans = []
        self.hold = HoldReference(start_pos, start_yaw, None, yaw_rate=search_yaw_rate)   # t0 = first use
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

    def _timing_hint(self, t):
        """segment_times_hint for plan_gate_trajectory: the active plan's arrival times from t on."""
        a = self.active(t)
        if not isinstance(a, Plan):
            return None
        tau = a.local(t)
        breaks = a.traj.t_breaks
        # 'start' and 'finish' are rebuilt by every plan: matching the old finish to a new one that the
        # extended window moved further on would squeeze every gate in between into no time
        ahead = [(w['name'], float(breaks[j]) - tau) for j, w in enumerate(a.wps)
                 if breaks[j] > tau + 1e-3 and w['type'] not in ('start', 'finish')]
        if not ahead:
            return None

        def hint(wps):
            arrive, k = [0.0], 0
            for w in wps[1:]:
                m = next((i for i in range(k, len(ahead)) if ahead[i][0] == w['name']), None)
                if m is None:
                    arrive.append(None)
                else:
                    arrive.append(ahead[m][1])
                    k = m + 1
            known = [i for i, v in enumerate(arrive) if v is not None]
            if len(known) < 2:
                return None
            pts = np.array([w['position'] for w in wps])
            s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
            for i, v in enumerate(arrive):                  # unmatched: by distance between matched ones
                if v is not None:
                    continue
                lo = max(j for j in known if j < i)
                hi = next((j for j in known if j > i), None)
                if hi is None:                              # past the last match: keep its pace
                    j0 = known[-2] if len(known) > 1 else 0
                    pace = (arrive[lo] - arrive[j0]) / max(s[lo] - s[j0], 1e-6)
                    arrive[i] = arrive[lo] + pace * (s[i] - s[lo])
                else:
                    f = (s[i] - s[lo]) / max(s[hi] - s[lo], 1e-6)
                    arrive[i] = arrive[lo] + f * (arrive[hi] - arrive[lo])
            T = np.diff(arrive)
            seg_len = np.diff(s)
            # a sane split: every segment takes time, none asks for more than twice the speed limit
            v_lim = self.v_max if self.v_max else 1e9
            if not np.all(T > 0.05) or np.any(seg_len / T > 2.0 * v_lim):
                return None
            return T
        return hint

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
                                             obstacle_gates=obstacles,
                                             segment_times_hint=self._timing_hint(t))
        except (np.linalg.LinAlgError, MemoryError, ValueError) as e:
            self._fail(t, reason, f"{type(e).__name__}: {str(e)[:80]}")
            return
        # fit_to_limits can fail to converge from a moving start and keep stretching the times (seen on
        # the 3-gate course's 2-lap window extension, 2026-11-05: a 228 s plan for 15 m). Never switch
        # to a plan that is absurdly slower than the one being flown.
        a = self.active(t)
        if isinstance(a, Plan):
            left = max(a.end_time() - t, 0.0)
            if traj.duration > 3.0 * left + 10.0:
                self._fail(t, reason, f"new plan {traj.duration:.1f} s vs {left:.1f} s left on the current one")
                return
        if self.yaw_mode == "next_gate":
            from min_snap_planner import look_at_yaw_profile      # here: min_snap_planner is a sibling module
            ref = TrajectoryReference(traj, start_yaw=yaw, m=self.m,
                                      yaw_profile=look_at_yaw_profile(traj, wps, gates, yaw))
        else:
            ref = TrajectoryReference(traj, start_yaw=yaw, m=self.m, yaw_blend_time=0.5 if moving else None)
        snapshot = {gid: (self.map.gates[gid].center.copy(), self.map.gates[gid].normal.copy(),
                          self.map.gates[gid].normal_reliable) for gid in window}
        self.plans.append(Plan(t, traj, ref, wps, self.progress, list(window), snapshot, reason))
        self._last_plan_t = t
        remaining = len(self.route_ahead())
        self._log(t, f"plan #{len(self.plans)} ({reason}): window {len(window)} of {remaining} "
                     f"route gate(s) left, lap {self.lap_of(self.progress)}/{self.laps}")

    def _fail(self, t, reason, why):
        self._log(t, f"replan FAILED ({reason}): {why}; keeping the current plan")
        # back off: do not retry a failing replan every min_interval
        self._last_plan_t = t + max(0.0, self.fail_backoff - self.min_interval)

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
