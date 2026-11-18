"""
MpcController: the team's NMPC (control/) flown on QuadSim, behind the same
controller interface as PidController / RlPolicyController:

    reset(n_envs, device, dtype)
    act(pos, vel, quat, gate_idx, dt, omega=None) -> (thrust [N] (N,), torque [N m] (N,3))

It is a WRENCH controller (class attribute command = "wrench"): the output is
collective thrust + body torque, which fly_course.run_controller feeds to
QuadSim.step_wrench (mixer + rotors, no rate loop). PID and RL are
command = "ctbr" (thrust + body-rate setpoint, QuadSim.step_ctbr).

WHAT IS REUSED (nothing reimplemented)
    Trajectory/gate_path.py        Gate, plan_gate_trajectory: gates -> min-snap path
    Trajectory/trajectory_reference.py  TrajectoryReference: path -> 12-state
                                   reference, build_reference_horizon(t, N, dt)
    control/MPC_Optimizer.py       create_drone_mpc_optimizer (IPOPT via CasADi
                                   Opti; model, RK4 discretiser, cost and
                                   limits come from control/*). Only the new
                                   optional inertia= keyword was added, so the
                                   model uses the Iris inertia of QuadParams
                                   rather than the hardcoded 0.0081/0.0142.
    The solve loop mirrors control/Drone_Race_Simulation.run_race_simulation:
    set X_init / U_prev / X_ref, warm start with the previous solution shifted
    one step, on RuntimeError fall back to opti.debug.value (last iterate).
    Differences: the first warm start (and U_prev) is hover thrust m g instead
    of zeros, as the spec for this comparison asks.

CONVENTIONS (verified to agree with rl/)
    World ENU z-up. Body FLU. R = Rz(yaw) Ry(pitch) Rx(roll), the MPC's
    translational equations use exactly this R. The 12-state is
    [x y z, roll pitch yaw, vx vy vz (world), p q r (body rates)]. Controls:
    U1 collective thrust along body +z [N], U2..U4 body torques [N m].
    Gate convention of gate_path.Gate.T_world_gate == course.Gate.T_world_gate().
    The measured yaw is UNWRAPPED onto the 2 pi branch nearest the reference
    psi at time t, because the reference psi is continuous and can leave
    (-pi, pi].

TIME: reference time t = steps since reset * dt, i.e. a time-parametrised
    trajectory (no re-timing). An env that falls behind stays behind.

CAVEAT: like the PID and the RL policy, this controller receives the TRUE
    simulator state (no EKF, no vision noise), so the three-way comparison
    isolates the control law. The MPC's internal model (no drag, rigid torque
    input) differs from QuadSim (linear drag, mixer clipping, gyroscopic
    terms): that mismatch is part of what is being compared.
"""

import os
import sys
import time

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("control", "Trajectory", "vision"):
    _p = os.path.join(_REPO, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from MPC_Optimizer import create_drone_mpc_optimizer          # noqa: E402
from gate_path import Gate as PathGate, plan_gate_trajectory  # noqa: E402
from trajectory_reference import TrajectoryReference          # noqa: E402

from course import Course                                     # noqa: E402
from drone_params import QuadParams                           # noqa: E402
from rotations import quat_to_rotmat, rotmat_to_euler_zyx     # noqa: E402


PLAN_CACHE_DIR = os.environ.get("PLAN_CACHE_DIR", os.path.expanduser("~/.cache/min_snap_plans"))


def _lap_hint(lap_traj, lap_wps):
    """segment_times_hint for a multi-lap route from the optimised ONE-lap plan: each segment gets the time
    of the one-lap segment between the same two waypoints (matched by name); segments the one-lap plan
    does not have (the join from the last gate into the next lap) get their length at the lap's mean speed.
    plan_gate_trajectory then fits the whole route to the limits in one go. Optimising every segment of a
    3-lap route from scratch would take far longer than the ~2 min of one lap."""
    seg = {}
    for i in range(len(lap_wps) - 1):
        seg.setdefault((lap_wps[i]["name"], lap_wps[i + 1]["name"]), float(lap_traj.T[i]))
    pts = np.array([w["position"] for w in lap_wps])
    v_mean = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)) / lap_traj.duration

    def hint(wps):
        T = []
        for i in range(len(wps) - 1):
            k = (wps[i]["name"], wps[i + 1]["name"])
            d = float(np.linalg.norm(np.asarray(wps[i + 1]["position"]) - np.asarray(wps[i]["position"])))
            T.append(seg.get(k, max(d / v_mean, 0.05)))
        return np.array(T)
    return hint


def _stretch_locally(traj, T0, gates, v_max, a_max, iters=15, clearance=0.5):
    """From segment times T0, stretch only the segments that break v_max / a_max (by their own peak ratio,
    re-solving a few times) instead of min_snap.fit_to_limits' uniform rescale of the whole route. A 3-lap
    speed_oval7 plan fitted uniformly flew 14.2-14.4 s laps against 11.2 s for one lap, because one
    segment (the join between laps) set the scale for all of them. Kept only if within the limits, clear
    of the gate frames like plan_gate_trajectory's result, and shorter; else `traj` is returned."""
    from min_snap import MinSnapTrajectory
    from gate_path import _first_bad_crossing
    T = np.asarray(T0, dtype=float).copy()
    if T.shape != np.asarray(traj.T).shape:
        return traj
    best = None
    for _ in range(iters):
        cand = MinSnapTrajectory(traj.waypoints, T, traj.fixed_dir)
        times, _, vel, acc, _, _ = cand.sample(0.01)
        seg = cand.segment_index(times)
        r = np.maximum(np.linalg.norm(vel, axis=1) / v_max, np.sqrt(np.linalg.norm(acc, axis=1) / a_max))
        ratio = np.array([r[seg == i].max() if np.any(seg == i) else 0.0 for i in range(len(T))])
        if ratio.max() <= 1.01:
            best = cand
            break
        T = np.where(ratio > 1.0, T * np.minimum(ratio * 1.02, 2.0), T)
    if best is None and ratio.max() < 1.2:
        # Neighbouring segments couple, so the last few percent do not settle (3-lap speed_oval7: 1.02-1.04
        # after 15 passes). From rest a uniform scale by s divides speeds by s and accelerations by s^2, so
        # scaling every time by the remaining ratio lands within the limits.
        T = T * ratio.max()
        best = MinSnapTrajectory(traj.waypoints, T, traj.fixed_dir)
    if best is None or best.duration >= traj.duration - 1e-3 or _first_bad_crossing(best, gates, clearance) is not None:
        print("[plan] local stretch not used; keeping the uniformly fitted route")
        return traj
    print(f"[plan] local stretch: {traj.duration:.2f} s -> {best.duration:.2f} s")
    return best


def cached_plan(gates, start, v_max, a_max, optimize_times, laps=1):
    """plan_gate_trajectory, cached on disk when optimize_times is on. The segment-time optimisation
    (min_snap.optimize_segment_times, Powell over every segment's share) took over 10 minutes for the
    7-gate speed_oval7 (2026-11-18): far too long inside reset(), so it runs once per (gates, start,
    limits) and the result is reloaded. PLAN_CACHE_DIR picks the folder; the files are a plain JSON
    recipe (waypoints, segment times, gate directions), so they move between NumPy versions (copy them
    to the pod to skip planning there). The trajectory is rebuilt from the recipe (one KKT solve)."""
    if not optimize_times:
        return plan_gate_trajectory(gates, start, v_max=v_max, a_max=a_max, verbose=False, optimize_times=False)
    import hashlib
    import json
    from min_snap import MinSnapTrajectory
    key = hashlib.sha1(repr((
        [(g.name, np.round(g.T_world_gate, 4).tolist(), round(float(g.side), 4)) for g in gates],
        np.round(start, 4).tolist(), round(float(v_max), 3), round(float(a_max), 3))
        + ((int(laps),) if laps > 1 else ())).encode()).hexdigest()[:16]
    path = os.path.join(PLAN_CACHE_DIR, f"plan_{key}.json")
    if os.path.isfile(path):
        with open(path) as f:
            r = json.load(f)
        traj = MinSnapTrajectory(np.array(r["points"]), np.array(r["T"]),
                                 {int(k): np.array(v) for k, v in r["fixed_dir"].items()})
        wps = [dict(w, position=np.array(w["position"])) for w in r["wps"]]
        print(f"[plan] optimised plan loaded from {path} ({traj.duration:.2f} s)")
        return traj, wps
    t0 = time.perf_counter()
    if laps > 1:                              # gates = one lap's gates repeated `laps` times
        n = len(gates) // laps
        lap_traj, lap_wps = cached_plan(gates[:n], start, v_max, a_max, True)
        print(f"[plan] {laps} laps from the one-lap plan's segment times; cache {path}")
        hint = _lap_hint(lap_traj, lap_wps)
        traj, wps = plan_gate_trajectory(gates, start, v_max=v_max, a_max=a_max, verbose=False, optimize_times=False,
                                         segment_times_hint=hint)
        traj = _stretch_locally(traj, hint(wps), gates, v_max, a_max)
    else:
        print(f"[plan] optimising segment times (can take minutes); cache {path}")
        traj, wps = plan_gate_trajectory(gates, start, v_max=v_max, a_max=a_max, verbose=False, optimize_times=True)
    save_plan(path, traj, wps)
    print(f"[plan] optimised plan {traj.duration:.2f} s in {time.perf_counter() - t0:.0f} s, saved")
    return traj, wps


def save_plan(path, traj, wps):
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rec = dict(points=traj.waypoints.tolist(), T=np.asarray(traj.T).tolist(),
               fixed_dir={str(k): np.asarray(v).tolist() for k, v in traj.fixed_dir.items()},
               wps=[dict(w, position=np.asarray(w["position"]).tolist()) for w in wps],
               duration=float(traj.duration))
    with open(path, "w") as f:
        json.dump(rec, f)


def unwrap_yaw(yaw, psi_ref):
    """Shift yaw by a multiple of 2 pi onto the branch nearest psi_ref."""
    return yaw + 2.0 * np.pi * np.round((psi_ref - yaw) / (2.0 * np.pi))


def state_vector(pos, vel, quat, omega):
    """Sim state (torch, batched (N,.)) -> (N, 12) numpy MPC state with the
    yaw in (-pi, pi]; callers unwrap per env."""
    R = quat_to_rotmat(quat)
    roll, pitch, yaw = rotmat_to_euler_zyx(R)
    x = torch.cat([pos, torch.stack([roll, pitch, yaw], -1), vel, omega], dim=-1)
    return x.detach().double().cpu().numpy()


class MpcController:
    command = "wrench"

    def __init__(self, params: QuadParams, course: Course, v_max: float = 4.0,
                 a_max: float = 6.0, horizon: int = 50, optimize_times: bool = False):
        self.p = params
        self.course = course
        self.v_max, self.a_max, self.N = v_max, a_max, horizon
        # gate_path.plan_gate_trajectory optimises segment times by default since
        # origin/main 2026-10-28: faster laps, but one plan took 12 s wall / 105 s CPU
        # for straight3 on an M-series Mac and minutes under load, inside reset().
        # Off by default here (distance-proportional times, as in the published runs).
        self.optimize_times = optimize_times
        self.laps = 1          # the course is one lap's gates repeated `laps` times (runner --laps)
        self.dt = params.control_dt
        self.solve_times = []
        self.n_failures = 0
        self._ref = None

    def reset(self, n_envs: int, device="cpu", dtype=torch.float32):
        p = self.p
        self.n, self.device, self.dtype = n_envs, device, dtype
        if self._ref is None:       # planning and the IPOPT problem are env-independent
            gates = [PathGate(g.name, g.T_world_gate(), g.side) for g in self.course.gates]
            start, start_yaw = self.course.start_pose()
            self.path_gates = gates
            self.traj, self.wps = cached_plan(gates, np.asarray(start, dtype=float), self.v_max, self.a_max,
                                              self.optimize_times, laps=self.laps)
            self._ref = TrajectoryReference(self.traj, start_yaw, m=p.mass)
            (self.opti, self.X, self.U, self.X_init, self.U_prev,
             self.X_ref) = create_drone_mpc_optimizer(N=self.N, dt=self.dt, m=p.mass,
                                                      inertia=p.inertia_diag)
        hover = np.array([[p.mass * p.gravity], [0.0], [0.0], [0.0]])
        self.u_prev = [hover.copy() for _ in range(n_envs)]
        self.U_guess = [np.tile(hover, (1, self.N)) for _ in range(n_envs)]
        self.X_guess = [None] * n_envs
        self.step = 0
        self.solve_times = []
        self.n_failures = 0

    def _solve(self, i, x0, ref_h):
        opti = self.opti
        if self.X_guess[i] is None:
            self.X_guess[i] = np.tile(x0.reshape(12, 1), (1, self.N + 1))
        opti.set_value(self.X_init, x0.reshape(12, 1))
        opti.set_value(self.U_prev, self.u_prev[i])
        opti.set_value(self.X_ref, ref_h)
        opti.set_initial(self.X, self.X_guess[i])
        opti.set_initial(self.U, self.U_guess[i])
        t0 = time.perf_counter()
        try:
            sol = opti.solve()
            X_sol, U_sol = sol.value(self.X), sol.value(self.U)
        except RuntimeError:        # IPOPT did not converge: take its last iterate
            self.n_failures += 1
            X_sol, U_sol = opti.debug.value(self.X), opti.debug.value(self.U)
        self.solve_times.append(time.perf_counter() - t0)
        X_sol, U_sol = np.asarray(X_sol).reshape(12, -1), np.asarray(U_sol).reshape(4, -1)
        self.X_guess[i] = np.hstack([X_sol[:, 1:], X_sol[:, -1:]])
        self.U_guess[i] = np.hstack([U_sol[:, 1:], U_sol[:, -1:]])
        return U_sol[:, 0]

    def act(self, pos, vel, quat, gate_idx, dt, omega=None):
        if omega is None:
            raise ValueError("MpcController needs omega (body rates)")
        t = self.step * self.dt
        ref_h = self._ref.build_reference_horizon(t, self.N, self.dt)
        xs = state_vector(pos, vel, quat, omega)
        out = np.zeros((self.n, 4))
        for i in range(self.n):
            x0 = xs[i].copy()
            x0[5] = unwrap_yaw(x0[5], ref_h[5, 0])
            u = self._solve(i, x0, ref_h)
            self.u_prev[i] = u.reshape(4, 1)
            out[i] = u
        self.step += 1
        thrust = torch.as_tensor(out[:, 0], dtype=self.dtype, device=self.device)
        thrust = thrust.clamp(0.0, self.p.max_total_thrust())
        torque = torch.as_tensor(out[:, 1:], dtype=self.dtype, device=self.device)
        return thrust, torque
