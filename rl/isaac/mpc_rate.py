"""
RateAwareMpc: a CTBR MPC (collective thrust + body RATES out, like
MpcCtbrController) whose model includes PX4's rate loop and the motor lag,
plus delay compensation. Drop-in for MpcCtbrController: same act() interface
and command = "ctbr"; reuses MpcController's min-snap plan and reference.

WHY (2026-11-12, bench_mpc_delay.py). MpcCtbrController plans torques and
sends the body rate it predicts one 20 ms step ahead, i.e. it assumes the
drone reaches a commanded rate within a step. Tony measured the Max's rate
loop in Isaac at a 63 % rise of 77 / 67 / 156 ms (roll / pitch / yaw), so
each correction lands 3-4 ticks late and overshoots: a growing roll/pitch
weave, then a tilt crash on a large demand (live: MPC 2/12 on the pink 36 in
gates). In QuadSim with that rate loop it crashes slalom5_pink36 even with no
delay, and oval6_pink36 from 30 ms of delay. This MPC models the loop, so its
command IS the rate setpoint and it knows the drone gets there over the next
few ticks: 21/21 bench flights complete (0-60 ms delay, delay not told to the
MPC, rate loop 2x faster than the MPC assumes).

MODEL (continuous, RK4, world ENU / body FLU, quaternion WXYZ body->world)
    p' = v
    v' = R(q) [0, 0, T/m] - g e3 + R(q) (-c * (R(q)^T v)) / m    linear body drag
    q' = 0.5 q (x) [0, w]                                         body rates
    w' = (w_cmd - w) / tau_w        PX4 rate loop, first order per axis
    T' = (T_cmd - T) / tau_T        motor lag
    state x = [p v q w T] (14), input u = [T_cmd, w_cmd] (4)
    tau_w defaults to the bench plant fitted to Isaac's measured rise times
    (80 / 64 / 152 ms). T is not measured: the controller propagates its own
    estimate from the commanded thrust.

COST over N steps of dt_mpc (default 20 x 50 ms = 1 s): position, velocity,
attitude (1 - <q, q_ref>^2, sign-free), body rate vs the reference's,
thrust vs the plan's feedforward, rate-command size and input change.

DELAY COMPENSATION (delay_s > 0): the measured state is predicted forward by
delay_s with the commands already sent (what the drone will fly until this
one lands), and the reference is read at t + delay_s.

BACKENDS: "ipopt" (CasADi Opti + IPOPT, works everywhere, the reference;
18-72 ms per solve on a laptop: NOT real time) and "acados" (SQP-RTI on
generated C, delta-u formulation; 0.8 ms median, <= 1.6 ms p95 on the bench,
and equal or better on every case; see mpc_rate_acados.py and
README_mpc_rate.md). Use acados for anything that flies.
"""
import math
import os
import sys
import time

import casadi as ca
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mpc_controller import MpcController   # noqa: E402

# Bench plant fitted to the Max's Isaac rate response (RESULTS_2026-11-12_max_pink.md)
TAU_W_ISAAC = (0.080, 0.064, 0.152)
WEIGHTS = dict(p=(60.0, 60.0, 100.0), v=(4.0, 4.0, 6.0), att=30.0, w=0.05,
               T=0.02, wc=0.05, du=0.5, term=5.0)


def quat_from_euler_zyx_np(roll, pitch, yaw):
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.stack([cy * cp * cr + sy * sp * sr,
                     cy * cp * sr - sy * sp * cr,
                     cy * sp * cr + sy * cp * sr,
                     sy * cp * cr - cy * sp * sr])


def model_rhs(params, tau_w, tau_T):
    """CasADi SX expressions (x, u, xdot) of the model above."""
    m, g = params.mass, params.gravity
    c = np.asarray(params.drag_coeff, dtype=float)
    x, u = ca.SX.sym("x", 14), ca.SX.sym("u", 4)
    v, q, w, T = x[3:6], x[6:10], x[10:13], x[13]
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    R = ca.vertcat(
        ca.horzcat(1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
        ca.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)),
        ca.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)))
    vb = R.T @ v
    acc = R @ ca.vertcat(0, 0, T / m) - ca.vertcat(0, 0, g) + R @ (-ca.DM(c) * vb) / m
    qdot = 0.5 * ca.vertcat(-qx * w[0] - qy * w[1] - qz * w[2],
                            qw * w[0] + qy * w[2] - qz * w[1],
                            qw * w[1] - qx * w[2] + qz * w[0],
                            qw * w[2] + qx * w[1] - qy * w[0])
    wdot = (u[1:4] - w) / ca.DM(np.asarray(tau_w, dtype=float))
    Tdot = (u[0] - T) / tau_T
    return x, u, ca.vertcat(v, acc, qdot, wdot, Tdot)


def rk4(f, x, u, h, n=2):
    s = h / n
    for _ in range(n):
        k1 = f(x, u); k2 = f(x + s / 2 * k1, u); k3 = f(x + s / 2 * k2, u); k4 = f(x + s * k3, u)
        x = x + s / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


class _IpoptSolver:
    """CasADi Opti + IPOPT over the shooting problem (reference implementation)."""

    def __init__(self, ctl):
        p, N, h, w = ctl.p, ctl.Nh, ctl.dtm, ctl.wts
        x, u, xdot = model_rhs(p, ctl.tau_w, ctl.tau_T)
        f = ca.Function("f", [x, u], [xdot])
        xs, us = ca.SX.sym("xs", 14), ca.SX.sym("us", 4)
        F = ca.Function("F", [xs, us], [rk4(f, xs, us, h, 2)]).expand()
        opti = ca.Opti()
        X, U = opti.variable(14, N + 1), opti.variable(4, N)
        X0, REF, Uprev = opti.parameter(14), opti.parameter(14, N + 1), opti.parameter(4)
        Qp, Qv = ca.diag(ca.DM(w["p"])), ca.diag(ca.DM(w["v"]))
        cost = 0
        for k in range(N + 1):
            s = w["term"] if k == N else 1.0
            ep, ev = X[0:3, k] - REF[0:3, k], X[3:6, k] - REF[3:6, k]
            dq = ca.dot(X[6:10, k], REF[6:10, k])
            ew = X[10:13, k] - REF[10:13, k]
            cost += s * (ca.mtimes([ep.T, Qp, ep]) + ca.mtimes([ev.T, Qv, ev])
                         + w["att"] * (1 - dq ** 2) + w["w"] * ca.dot(ew, ew))
        mr = np.asarray(p.max_rate, dtype=float)
        for k in range(N):
            du = U[:, k] - (Uprev if k == 0 else U[:, k - 1])
            cost += (w["T"] * (U[0, k] - REF[13, k]) ** 2 + w["wc"] * ca.dot(U[1:4, k], U[1:4, k])
                     + w["du"] * ca.dot(du, du))
            opti.subject_to(X[:, k + 1] == F(X[:, k], U[:, k]))
            opti.subject_to(opti.bounded(0.1 * p.mass * p.gravity, U[0, k], p.max_total_thrust()))
            for i in range(3):
                opti.subject_to(opti.bounded(-mr[i], U[1 + i, k], mr[i]))
        opti.subject_to(X[:, 0] == X0)
        opti.minimize(cost)
        opti.solver("ipopt", {"expand": True, "print_time": 0},
                    {"print_level": 0, "max_iter": 30, "tol": 1e-3, "acceptable_tol": 1e-2,
                     "acceptable_iter": 1, "warm_start_init_point": "yes", "mu_init": 1e-3, "sb": "yes"})
        self.opti, self.X, self.U, self.X0, self.REF, self.Uprev = opti, X, U, X0, REF, Uprev
        self.Xg = None
        self.Ug = None
        self.n_failures = 0

    def reset(self, u_hover, N):
        self.Xg, self.Ug = None, np.tile(u_hover.reshape(4, 1), (1, N))

    def solve(self, x0, ref, u_prev):
        o = self.opti
        o.set_value(self.X0, x0); o.set_value(self.REF, ref); o.set_value(self.Uprev, u_prev)
        o.set_initial(self.X, self.Xg if self.Xg is not None else np.tile(x0.reshape(14, 1), (1, ref.shape[1])))
        o.set_initial(self.U, self.Ug)
        try:
            sol = o.solve(); Xs, Us = sol.value(self.X), sol.value(self.U)
        except RuntimeError:
            self.n_failures += 1
            Xs, Us = o.debug.value(self.X), o.debug.value(self.U)
        self.Xg, self.Ug = np.asarray(Xs).reshape(14, -1), np.asarray(Us).reshape(4, -1)
        return self.Ug[:, 0].copy()


class RateAwareMpc(MpcController):
    command = "ctbr"

    def __init__(self, params, course, v_max=4.0, a_max=6.0, *, N=50, dt_mpc=0.05,
                 tau_w=TAU_W_ISAAC, tau_T=None, delay_s=0.0, weights=None, backend="ipopt"):
        super().__init__(params, course, v_max=v_max, a_max=a_max, horizon=N)
        self.Nh, self.dtm = N, dt_mpc
        self.tau_w = np.asarray(tau_w, dtype=float)
        self.tau_T = float(tau_T if tau_T is not None else max(params.tau_motor, 0.01))
        self.delay_s = float(delay_s)
        self.wts = dict(WEIGHTS)
        self.wts.update(weights or {})
        self.backend = backend
        self.ctrl_dt = params.control_dt
        self.solver = None
        x, u, xdot = model_rhs(params, self.tau_w, self.tau_T)
        f = ca.Function("f", [x, u], [xdot])
        self._f_fine = ca.Function("Ff", [x, u], [rk4(f, x, u, params.physics_dt, 1)]).expand()

    def _make_solver(self):
        if self.backend == "ipopt":
            return _IpoptSolver(self)
        if self.backend == "acados":
            from mpc_rate_acados import AcadosRateSolver
            return AcadosRateSolver(self)
        raise ValueError(f"unknown backend {self.backend!r}")

    @property
    def n_failures_total(self):
        return getattr(self.solver, "n_failures", 0)

    def reset(self, n_envs, device="cpu", dtype=torch.float32):
        if n_envs != 1:
            raise ValueError("RateAwareMpc flies one env")
        super().reset(n_envs, device, dtype)           # plan + _ref (also builds the torque NMPC; unused)
        if self.solver is None:
            self.solver = self._make_solver()
        p = self.p
        self.T_est = p.mass * p.gravity
        self.u_last = np.array([self.T_est, 0.0, 0.0, 0.0])
        self.u_active = self.u_last.copy()
        self.pending = []                              # (time it lands, u) of commands sent, not yet active
        self.solver.reset(self.u_last, self.Nh)
        self.step = 0
        self.solve_times = []

    def _ref_horizon(self, t):
        r = self._ref.build_reference_horizon(t, self.Nh, self.dtm)        # 12 x N+1
        q = quat_from_euler_zyx_np(r[3], r[4], r[5])
        Tff = np.array([self._ref.feedforward_thrust(t + k * self.dtm) for k in range(self.Nh + 1)])
        return np.vstack([r[0:3], r[6:9], q, r[9:12], Tff[None, :]])         # 14 x N+1

    def _predict(self, x, t_now):
        """State at t_now -> state at t_now + delay_s, flying the commands active then."""
        if self.delay_s <= 0:
            return x
        cmd, dt, t = self.u_active, self.p.physics_dt, 0.0
        while t < self.delay_s - 1e-9:
            for ta, u in self.pending:
                if ta <= t_now + t + 1e-9:
                    cmd = u
            x = np.asarray(self._f_fine(x, cmd)).reshape(-1)
            x[6:10] /= np.linalg.norm(x[6:10])
            t += dt
        return x

    def act(self, pos, vel, quat, gate_idx, dt, omega=None):
        if omega is None:
            raise ValueError("RateAwareMpc needs omega (body rates)")
        t = self.step * self.ctrl_dt
        keep = []
        for ta, u in self.pending:                     # commands that have landed by now
            if ta <= t + 1e-9:
                self.u_active = u
            else:
                keep.append((ta, u))
        self.pending = keep
        a = 1.0 - math.exp(-self.ctrl_dt / self.tau_T)
        self.T_est += a * (self.u_active[0] - self.T_est)
        x = np.concatenate([pos[0].double().cpu().numpy(), vel[0].double().cpu().numpy(),
                            quat[0].double().cpu().numpy(), omega[0].double().cpu().numpy(), [self.T_est]])
        x = self._predict(x, t)
        ref = self._ref_horizon(t + self.delay_s)
        if np.dot(ref[6:10, 0], x[6:10]) < 0:           # quaternion sign continuity
            ref[6:10] *= -1
        for k in range(1, ref.shape[1]):
            if np.dot(ref[6:10, k], ref[6:10, k - 1]) < 0:
                ref[6:10, k] *= -1
        t0 = time.perf_counter()
        u = self.solver.solve(x, ref, self.u_last)
        self.solve_times.append(time.perf_counter() - t0)
        mr = np.asarray(self.p.max_rate, dtype=float)
        u[0] = float(np.clip(u[0], 0.0, self.p.max_total_thrust()))
        u[1:4] = np.clip(u[1:4], -mr, mr)
        self.u_last = u
        self.pending.append((t + self.delay_s, u))
        if self.delay_s <= 0:
            self.u_active = u
        self.step += 1
        return (torch.tensor([u[0]], dtype=self.dtype, device=self.device),
                torch.tensor(u[1:4], dtype=self.dtype, device=self.device).reshape(1, 3))
