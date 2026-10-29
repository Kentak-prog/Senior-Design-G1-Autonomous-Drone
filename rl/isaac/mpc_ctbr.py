"""
MpcCtbrController: the team's NMPC (rl/mpc_controller.MpcController) behind the
CTBR interface PX4 offboard accepts (collective thrust + body RATES), so it can
fly through PX4 in Isaac Sim like the PID / RL controllers.

    command = "ctbr"
    reset(n_envs, device, dtype)
    act(pos, vel, quat, gate_idx, dt, omega=...) -> (thrust N (N,), body-rate ref FLU rad/s (N,3))

NOTHING of the MPC math is copied: this subclasses MpcController (planning with
Trajectory/, the CasADi/IPOPT problem from control/MPC_Optimizer, the state
vector, yaw unwrapping). Only the solve wrapper is replaced (it must keep the
predicted state, not just U[0], and has a time budget) and the solver options
are re-set after create_drone_mpc_optimizer (control/ is untouched).

WHAT IS SENT
    thrust = U_sol[0, 0] (collective thrust along body +z, N), clamped to
        [0, max_total_thrust] exactly like MpcController.
    rates  = X_sol[9:12, k] with k = rate_lookahead (default 1): the PREDICTED body
        rates (p, q, r), x-forward / y-left / z-up = FLU. The 12-state is
        [x y z, roll pitch yaw, vx vy vz (world), p q r (body)]
        (control/New_Drone_State_Equations.py: the Euler-rate kinematics map p q r
        to phi/theta/psi dot with the body-FLU convention shared with rl/), so
        these are directly the body-frame FLU rates PX4 wants. Clamped per axis to
        params.max_rate.
    Why X[:, 1]: the MPC applies U[0] (a torque) for one step dt and expects to
    be at X[:, 1] afterwards, so X[9:12, 1] is the angular velocity the MPC wants
    at the END of this step. A rate-controlled vehicle (PX4's rate loop, or
    QuadSim.step_ctbr) tracks a rate setpoint over its own time constant (tens
    of ms), so the setpoint that makes the vehicle's attitude follow the MPC's
    predicted attitude is the rate the MPC reaches at the end of the interval
    it is commanding, not X[:, 0] (= the measured rate, which would only
    say "keep doing what you do" and add a full step of lag). A longer
    lookahead (k = 2) leads the rate loop's lag further but over-commands
    when the plan changes; see the lookahead table below (picked by QuadSim).

THE COMMAND LEVEL CHANGE IS LOSSY. The MPC plans with a torque input and
    models no rate loop; here the rate loop sits between the MPC and the
    motors. The attitude/position that results is no longer exactly what the
    MPC predicted, but the MPC re-solves from the measured state every tick, so
    the mismatch is corrected like any other model error.

SPEED (real time). The wrench MPC is ~105 ms mean / 270 ms max per solve (IPOPT
    via Opti, max_iter 50, no expand). After create_drone_mpc_optimizer we call
    opti.solver("ipopt", plugin_opts, solver_opts) again (SOLVER_PRESETS) and do
    one throw-away solve in reset() so the NLP build is not charged to tick 1.
    Measured with rl/isaac/bench_mpc_ctbr.py, QuadSim, N = 10, M-series CPU,
    one flight per cell, solve ms mean / p95 / max (laps are all identical to the
    wrench MPC: 9.56 / 22.68 / 28.46 s):

      preset            options (beyond print_level 0)                    straight3         oval6             slalom5
      default           max_iter 50, acc_tol 1e-4 (control/ as built)     106.6/112.9/282   (not re-run)      (not re-run)
      expand            + expand=True                                     13.2/13.7/14.3    13.4/13.9/35.1    13.6/14.3/44.5
      fast              expand, max_iter 15, tol 1e-3, acc_tol 1e-2,      7.6/8.1/8.8       7.8/8.4/11.4      7.8/8.3/34.1
                        acc_iter 1
      fast_iter8        fast with max_iter 8                              7.7/8.1/8.9       (same as fast)    (not run)
      fast_mu (chosen)  fast + mu_init 1e-3, warm_start_init_point yes    3.4/3.7/4.7       3.3/3.7/4.6       3.5/4.7/11.5

    (default row: first-solve build cost included in its 282 ms max; the other rows
    exclude it by the warm-up solve. Max values of 30-60 ms on a few rows are single
    outliers from the OS, not the solver: fast_mu max is 4-12 ms.) All presets give the same
    laps, tracking error and gate margins to the printed precision, so the fastest,
    fast_mu, is the default. The speed comes from expand=True (about 8x, MX -> SX
    graph) and from IPOPT converging in 2-4 iterations when started at a small
    barrier parameter from the shifted previous solution. Target (mean < 20 ms,
    p95 < 40 ms) is met with a wide margin. Not tried: a shorter horizon (not needed).

LOOKAHEAD TABLE (rate = X[9:12, k], preset fast, same flights; position tracking
    error vs the reference rms / max [m], laps all within 0.02 s):
      k = 1   straight3 0.417/0.538   oval6 0.451/0.617   slalom5 0.429/0.745
      k = 2   straight3 0.450/0.580   oval6 0.483/0.663   slalom5 0.458/0.791
      k = 3   straight3 0.496/0.641   oval6 0.532/0.729   slalom5 0.503/0.865
    Larger k is monotonically worse (the setpoint runs ahead of what the MPC itself
    wants this step and the rate loop over-rotates), so k = 1 is the default.
    (The 0.4 m rms is mostly the time-parametrised reference running ahead of a
    drone that starts at rest; the wrench MPC has the same lag.)

PER-TICK BUDGET (max_solve_ms). IPOPT's max_cpu_time is set to the budget. If a
    solve hits it (return status Maximum_CpuTime_Exceeded), the previous
    solution shifted one step is used instead: thrust U_prev_sol[0, 1], rate
    X_prev_sol[9:12, 2] (i.e. the guess that was about to seed the solve), and
    the shifted guess is kept for the next solve. The count is
    n_budget_fallbacks. IPOPT only checks the clock between iterations, so a
    solve can overrun by about one iteration (~1-3 ms). Default None = no budget
    (deterministic in tests); for the live sim pass e.g. max_solve_ms=15.

RE-PLANNING (update_course). For a vision gate map that changes in flight.
    update_course(course, pos, vel) compares the REMAINING gates (index >= the
    last gate_idx seen in act) of `course` with those of the course the current
    plan was made from. If any centre moved more than replan_dist (0.3 m), any
    yaw more than replan_yaw_deg (5 deg), or the gate count changed, it re-plans
    the min-snap trajectory over the remaining gates from the current position
    and restarts the reference clock. When nothing changed it is a handful of
    float compares (microseconds). plan_gate_trajectory / min_snap take only a
    START POSITION, not a start velocity, so the new plan starts AT REST at the
    current position (the heading of the reference starts at the current yaw).
    The reference therefore asks the drone to brake first; the MPC does that
    within its limits, so re-planning at speed costs a visible slow-down
    (replanning when slow is cheap). `vel` is accepted for the interface and
    used only to log the speed at the re-plan. Re-plan time is in the table
    below: 35 ms on oval6 (6 gates), 87 ms (248 ms once, with other jobs running) mid-flight on straight3 (2 remaining
    gates, includes TrajectoryReference build); that is 2-5 control ticks in one
    act() call, so call it from the vision thread/between ticks, not every tick.
    Mid-flight test: gates 0.6 m off the believed plan are missed without the
    update (margin 0.37 m at gate 1, gate 2 missed) and flown with it
    (min margin 0.69 m) but the lap goes from 9.56 s to 20.6 s because the new
    plan restarts at rest.

NOT REAL-TIME ON A SLOW MACHINE: SimVehicle (offline) steps sim time per tick
    regardless of how long act() took, so the offline tests do not prove real
    time; only the measured solve times do.
"""

import os
import sys
import threading
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_RL = os.path.dirname(_HERE)
if _RL not in sys.path:
    sys.path.insert(0, _RL)

import mpc_controller as mc                                          # noqa: E402  (sets up control/ + Trajectory/ paths)
from mpc_controller import MpcController, state_vector, unwrap_yaw   # noqa: E402
from gate_path import Gate as PathGate, plan_gate_trajectory          # noqa: E402
from trajectory_reference import TrajectoryReference                  # noqa: E402

# name -> (plugin_opts, solver_opts); solver_opts are IPOPT options WITHOUT the
# "ipopt." prefix (Opti.solver takes them that way).
SOLVER_PRESETS = {
    # what control/MPC_Optimizer sets (reference row; plugin print_time 0)
    "default": ({"print_time": False},
                {"print_level": 0, "max_iter": 50, "acceptable_tol": 1e-4}),
    "expand": ({"expand": True, "print_time": False},
               {"print_level": 0, "max_iter": 50, "acceptable_tol": 1e-4}),
    "fast": ({"expand": True, "print_time": False},
             {"print_level": 0, "max_iter": 15, "tol": 1e-3, "acceptable_tol": 1e-2,
              "acceptable_iter": 1}),
    "fast_mu": ({"expand": True, "print_time": False},
                {"print_level": 0, "max_iter": 15, "tol": 1e-3, "acceptable_tol": 1e-2,
                 "acceptable_iter": 1, "mu_init": 1e-3, "warm_start_init_point": "yes",
                 "warm_start_bound_push": 1e-6, "warm_start_mult_bound_push": 1e-6}),
    "fast_iter8": ({"expand": True, "print_time": False},
                   {"print_level": 0, "max_iter": 8, "tol": 1e-3, "acceptable_tol": 1e-2,
                    "acceptable_iter": 1}),
}
DEFAULT_PRESET = "fast_mu"


class MpcCtbrController(MpcController):
    command = "ctbr"

    def __init__(self, params, course, v_max=4.0, a_max=6.0, horizon=10, *,
                 solver=DEFAULT_PRESET, rate_lookahead=1, max_solve_ms=None,
                 replan_dist=0.3, replan_yaw_deg=5.0):
        super().__init__(params, course, v_max=v_max, a_max=a_max, horizon=horizon)
        if isinstance(solver, str):
            solver = SOLVER_PRESETS[solver]
        self.plugin_opts, self.solver_opts = dict(solver[0]), dict(solver[1])
        if not 1 <= rate_lookahead <= horizon:
            raise ValueError("rate_lookahead must be in [1, horizon]")
        self.k_rate = rate_lookahead
        self.max_solve_ms = max_solve_ms
        self.replan_dist, self.replan_yaw = replan_dist, replan_yaw_deg
        self._solver_set = False
        self._base_plan = None
        self.n_budget_fallbacks = 0
        self.n_replans = 0
        self.last_replan_ms = float("nan")
        # async_replan: plan in a background thread and swap the plan in at the
        # next act(). The live runner turns it on: on the pod a min-snap re-plan
        # took seconds while vision processing shared the CPU, no setpoints
        # reached PX4 meanwhile and the flight ended upside down (2026-10-24).
        self.async_replan = False
        self._replan_thread = None
        self._pending_plan = None
        self._plan_gen = 0

    # ---------------------------------------------------------------- reset
    def reset(self, n_envs, device="cpu", dtype=torch.float32):
        super().reset(n_envs, device, dtype)        # plans (once) + builds the Opti problem
        if not self._solver_set:
            self._apply_solver()
            self._base_plan = (self.traj, self._ref)
        elif self._ref is not self._base_plan[1]:   # a previous flight re-planned: back to the course plan
            self.traj, self._ref = self._base_plan
        self._plan_course = self.course
        self._plan_start_gate = 0
        self._t0_step = 0
        self._gate_idx = 0
        self._yaw0 = self.course.start_pose()[1]
        self.n_budget_fallbacks = 0
        self.n_replans = 0
        self._have_prev = [False] * n_envs
        self._plan_gen += 1             # a plan still being computed belongs to the old flight
        self._pending_plan = None

    def _apply_solver(self):
        opts = dict(self.solver_opts)
        if self.max_solve_ms is not None:
            opts["max_cpu_time"] = self.max_solve_ms / 1000.0
        self.opti.solver("ipopt", dict(self.plugin_opts), opts)
        self._solver_set = True
        # one throw-away solve at hover: the first call builds/expands the NLP
        # (~0.3 s) and must not land inside the first control tick
        p = self.p
        hover = np.array([[p.mass * p.gravity], [0.0], [0.0], [0.0]])
        x0 = np.zeros((12, 1))
        x0[2] = 1.0
        self.opti.set_value(self.X_init, x0)
        self.opti.set_value(self.U_prev, hover)
        self.opti.set_value(self.X_ref, np.tile(x0, (1, self.N + 1)))
        self.opti.set_initial(self.X, np.tile(x0, (1, self.N + 1)))
        self.opti.set_initial(self.U, np.tile(hover, (1, self.N)))
        try:
            self.opti.solve()
        except RuntimeError:
            pass

    # ---------------------------------------------------------------- solve
    def _solve(self, i, x0, ref_h):
        """Returns (thrust, rate_ref (3,), U[:, 0] or None on a budget fallback). Same orchestration as
        MpcController._solve; additionally keeps X_sol[9:12, k_rate] and applies
        the CPU-time budget."""
        opti = self.opti
        if self.X_guess[i] is None:
            self.X_guess[i] = np.tile(x0.reshape(12, 1), (1, self.N + 1))
        opti.set_value(self.X_init, x0.reshape(12, 1))
        opti.set_value(self.U_prev, self.u_prev[i])
        opti.set_value(self.X_ref, ref_h)
        opti.set_initial(self.X, self.X_guess[i])
        opti.set_initial(self.U, self.U_guess[i])
        t0 = time.perf_counter()
        budget_hit = False
        try:
            sol = opti.solve()
            X_sol, U_sol = sol.value(self.X), sol.value(self.U)
        except RuntimeError:
            if (self._have_prev[i] and self.max_solve_ms is not None
                    and opti.stats().get("return_status") == "Maximum_CpuTime_Exceeded"):
                budget_hit = True
            else:                       # did not converge: take IPOPT's last iterate
                self.n_failures += 1
                X_sol, U_sol = opti.debug.value(self.X), opti.debug.value(self.U)
        self.solve_times.append(time.perf_counter() - t0)

        if budget_hit:
            # X_guess / U_guess are the previous solution shifted one step: column 0
            # is what the previous solve expected NOW, column k what it expects k
            # steps from now.
            self.n_budget_fallbacks += 1
            T, w = float(self.U_guess[i][0, 0]), self.X_guess[i][9:12, self.k_rate].copy()
            self.X_guess[i] = np.hstack([self.X_guess[i][:, 1:], self.X_guess[i][:, -1:]])
            self.U_guess[i] = np.hstack([self.U_guess[i][:, 1:], self.U_guess[i][:, -1:]])
            return T, w, None
        X_sol, U_sol = np.asarray(X_sol).reshape(12, -1), np.asarray(U_sol).reshape(4, -1)
        self.X_guess[i] = np.hstack([X_sol[:, 1:], X_sol[:, -1:]])
        self.U_guess[i] = np.hstack([U_sol[:, 1:], U_sol[:, -1:]])
        self._have_prev[i] = True
        return float(U_sol[0, 0]), X_sol[9:12, self.k_rate].copy(), U_sol[:, 0]

    # ---------------------------------------------------------------- act
    def act(self, pos, vel, quat, gate_idx, dt, omega=None):
        if omega is None:
            raise ValueError("MpcCtbrController needs omega (body rates)")
        self._gate_idx = int(gate_idx.min()) if torch.is_tensor(gate_idx) else int(gate_idx)
        if self._pending_plan is not None:      # an async re-plan finished
            p, self._pending_plan = self._pending_plan, None
            self._swap_in(p)
        t = (self.step - self._t0_step) * self.dt
        ref_h = self._ref.build_reference_horizon(t, self.N, self.dt)
        xs = state_vector(pos, vel, quat, omega)
        thrust = np.zeros(self.n)
        rates = np.zeros((self.n, 3))
        for i in range(self.n):
            x0 = xs[i].copy()
            x0[5] = unwrap_yaw(x0[5], ref_h[5, 0])
            T, w, u_full = self._solve(i, x0, ref_h)
            if u_full is not None:      # the actuator-rate cost refers to the last real solve
                self.u_prev[i] = u_full.reshape(4, 1)
            thrust[i], rates[i] = T, w
        self._yaw0 = float(x0[5])
        self.step += 1
        mr = np.asarray(self.p.max_rate, dtype=float)
        rates = np.clip(rates, -mr, mr)
        thrust_t = torch.as_tensor(thrust, dtype=self.dtype, device=self.device)
        thrust_t = thrust_t.clamp(0.0, self.p.max_total_thrust())
        return thrust_t, torch.as_tensor(rates, dtype=self.dtype, device=self.device)

    # ---------------------------------------------------------------- replan
    def _gate_moved(self, a, b):
        dyaw = abs((a.yaw_deg - b.yaw_deg + 180.0) % 360.0 - 180.0)
        return (np.linalg.norm(a.center - b.center) > self.replan_dist
                or dyaw > self.replan_yaw)

    def needs_replan(self, course):
        """True if a remaining gate (index >= last gate_idx) differs from the plan."""
        old = self._plan_course
        if len(course) != len(old):
            return True
        for k in range(self._gate_idx, len(course)):
            if self._gate_moved(course.gates[k], old.gates[k]):
                return True
        return False

    def update_course(self, course, pos, vel=None):
        """Re-plan if the remaining gates moved (see module docstring). pos/vel
        ENU (3,). Returns True if it re-planned (async: if a plan was started;
        it is swapped in by the next act()). While an async plan is running,
        further updates are ignored; the next one after it compares against it."""
        if self._replan_thread is not None and self._replan_thread.is_alive():
            return False
        if not self.needs_replan(course):
            return False
        k0 = min(self._gate_idx, len(course) - 1)
        gates = [PathGate(g.name, g.T_world_gate(), g.side) for g in course.gates[k0:]]
        pos = np.asarray(pos, dtype=float).reshape(3).copy()
        yaw, gen = self._yaw0, self._plan_gen     # measured yaw of the last act() (unwrapped)
        # The plan starts at rest (vel unused). optimize_times=False: gate_path's segment-time
        # optimisation (origin/main 2026-10-28) made one mid-flight re-plan take 94 s.
        # NOT start_state=current velocity, although gate_path supports it since then:
        # min_snap.fit_to_limits never returned for a start on gate 0's plane at 1.7 m/s
        # (test_midflight_replan_straight3): its time rescale blew up and sampling the
        # trajectory at 5 ms ran the machine out of memory (2026-10-29). Revisit once fixed.

        def plan():
            t0 = time.perf_counter()
            traj, _ = plan_gate_trajectory(gates, pos, v_max=self.v_max, a_max=self.a_max,
                                           verbose=False, optimize_times=False)
            ref = TrajectoryReference(traj, yaw, m=self.p.mass)
            return gen, traj, ref, course, k0, 1e3 * (time.perf_counter() - t0)

        if not self.async_replan:
            self._swap_in(plan())
            return True
        self._pending_plan = None

        def worker():
            self._pending_plan = plan()

        self._replan_thread = threading.Thread(target=worker, daemon=True)
        self._replan_thread.start()
        return True

    def _swap_in(self, p):
        gen, traj, ref, course, k0, ms = p
        if gen != self._plan_gen:       # computed for a flight that has since been reset
            return
        self.traj, self._ref = traj, ref
        self._plan_course = course
        self._plan_start_gate = k0
        self._t0_step = self.step       # old warm start is kept (the state is continuous)
        self.n_replans += 1
        self.last_replan_ms = ms
