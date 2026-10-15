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
                 a_max: float = 6.0, horizon: int = 10):
        self.p = params
        self.course = course
        self.v_max, self.a_max, self.N = v_max, a_max, horizon
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
            self.traj, _ = plan_gate_trajectory(gates, np.asarray(start, dtype=float),
                                                v_max=self.v_max, a_max=self.a_max,
                                                verbose=False)
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
