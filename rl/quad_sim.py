"""
Batched rigid-body quadrotor simulator for RL training and controller tests.

CONVENTIONS (see drone_params.py / rotations.py)
    World ENU (z up, gravity -z). Body FLU (x fwd, y left, z up). Quaternions
    WXYZ body -> world. Body rates in the body frame. Everything is a torch
    tensor with a leading env dimension N.

WHY this exists: Isaac Sim is too slow and too heavy to iterate on policies,
so this is a minimal model of the same airframe (Pegasus Iris dynamics) that
runs thousands of envs on a CPU. It deliberately mirrors Pegasus: quadratic
thrust per rotor, yaw moment k_m*w^2*rot_dir, linear body-frame drag, 250 Hz
physics. No aerodynamics beyond that, no ground contact (the runner flags
ground hits instead).

MIXER SIGN DERIVATION
    Rotor i sits at (x_i, y_i, z) in body FLU and pushes along body +z with
    thrust f_i = k_f w_i^2. Torque about the CG is r_i x f_i with
    r_i = (x_i, y_i, z) and f_i = (0, 0, f_i):
        r x f = (y_i f_i,  -x_i f_i,  0)
    so  roll  torque (about +x) =  sum  y_i f_i
        pitch torque (about +y) = -sum  x_i f_i
    Yaw (about +z) comes from the rotor drag moment, which Pegasus applies as
    k_m w_i^2 rot_dir_i = (k_m/k_f) rot_dir_i f_i. Collective thrust is
    sum f_i. Stacking gives the 4x4 allocation matrix A with [T, tx, ty, tz]
    = A f; the mixer solves f = A^-1 [T, tx, ty, tz], clips each f_i to
    [k_f w_min^2, k_f w_max^2] (the physical limits; no desaturation priority
    scheme -- a deliberately simple choice), and converts to rotor speed
    w_i = sqrt(f_i / k_f).
    Sanity check (roll): a positive roll torque needs more thrust on the
    rotors with y > 0 (left side), lifting the left side. Right-hand rotation
    about forward x takes y(left) -> z(up), so the left side rises and the
    thrust vector tilts toward -y (right side dips): the drone accelerates
    to the right (world -y at yaw 0).

RATE CONTROLLER (CTBR inner loop)
    The policy (or PID) commands collective thrust [N] and body rates
    [rad/s]; like PX4's rate loop this is tracked by a PID on body-rate error
    whose output is a torque. Gains are in physical units:
        torque = I * (Kp*e + Ki*integral(e) - Kd * d(omega)/dt),  e = w_ref - w
    On the ideal plant I*dw/dt = torque this is a first-order lag with time
    constant 1/Kp (Kp = 40 1/s -> 25 ms; PX4's typical rate loop is in the
    20-30 ms range). D acts on the measurement, not the error, to avoid
    kicks on setpoint steps. Gyroscopic coupling and rotor inertia torques are
    not cancelled (the integrator absorbs them), as in PX4.

DOMAIN RANDOMISATION (randomize / DynRand)
    The plant can differ per env from the nominal QuadParams: mass, thrust
    coefficient, inertia, drag, yaw-moment coefficient, motor lag and the
    rate-loop gain. Only the PLANT changes; the controller side (rate-loop
    torque I_nominal * Kp * e, mixer A built from nominal k_m / k_f, thrust ->
    rotor speed with nominal k_f) keeps the nominal numbers, as PX4 does not
    know the true airframe. With no randomisation every env is nominal and the
    simulation is the same as before.

DYNAMICS
    Forces: sum rotor thrust along body +z, rotated to world, plus linear drag
    (-coeff * v_body, body frame) and gravity. Torques: sum r_i x f_i + yaw
    moments - omega x (I omega). Integration is semi-implicit Euler: velocity
    then position (using the NEW velocity); omega then attitude (quaternion
    right-multiplied by the exact exponential of omega*dt), then renormalise.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch

from drone_params import QuadParams
from rotations import quat_to_rotmat, quat_mul, quat_from_rotvec, quat_normalize


def build_allocation(params: QuadParams, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """4x4 matrix A with [T, tau_x, tau_y, tau_z] = A @ f (see module docstring)."""
    xy = torch.tensor(params.rotor_xy, dtype=dtype, device=device)       # (4,2)
    rot_dir = torch.tensor(params.rot_dir, dtype=dtype, device=device)   # (4,)
    return torch.stack([
        torch.ones(4, dtype=dtype, device=device),
        xy[:, 1],
        -xy[:, 0],
        rot_dir * params.k_m / params.k_f,
    ], dim=0)


@dataclass
class DynRand:
    """Per-env plant ranges, sampled uniformly at each reset. Scales multiply the
    nominal value; tau_motor is absolute [s]. None leaves that parameter nominal.
    Defaults (2026-11-07) bracket the Starling 2 Max's estimates and PX4's
    slower rate loop. On the Max, the corners of rate_kp x tau_motor x inertia
    give 63 % rise times of 26-106 ms roll/pitch (median 52) and 48-172 ms yaw
    (median 93), overshoot up to 40 %; PX4 on the Iris measured 58-64 ms and
    110-120 ms with 40 % yaw overshoot (rl/isaac/RESULTS_2026-10-24.md). The
    Max's own PX4 response has not been measured yet (rl/isaac/calibrate.py rates)."""
    mass: Optional[tuple] = (0.95, 1.10)
    thrust: Optional[tuple] = (0.90, 1.10)       # k_f
    inertia: Optional[tuple] = (0.70, 1.40)      # per axis
    drag: Optional[tuple] = (0.50, 1.50)         # per axis
    k_m: Optional[tuple] = (0.60, 1.60)
    tau_motor: Optional[tuple] = (0.020, 0.050)  # s
    rate_kp: Optional[tuple] = (0.40, 1.00)      # per axis


class RateController:
    """Batched body-rate PID producing a body torque [N m]. See module docstring."""

    def __init__(self, params: QuadParams, n_envs: int, device="cpu", dtype=torch.float32):
        self.params = params
        self.inertia = torch.tensor(params.inertia_diag, dtype=dtype, device=device)
        self.kp_nominal = torch.tensor(params.rate_kp, dtype=dtype, device=device)
        self.kp = self.kp_nominal.expand(n_envs, 3).clone()     # per env (DynRand.rate_kp)
        self.ki = torch.tensor(params.rate_ki, dtype=dtype, device=device)
        self.kd = torch.tensor(params.rate_kd, dtype=dtype, device=device)
        self.integral = torch.zeros(n_envs, 3, dtype=dtype, device=device)
        self.prev_omega = torch.zeros(n_envs, 3, dtype=dtype, device=device)
        self.has_prev = torch.zeros(n_envs, 1, dtype=torch.bool, device=device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.integral[env_ids] = 0.0
        self.has_prev[env_ids] = False

    def __call__(self, omega: torch.Tensor, omega_ref: torch.Tensor, dt: float) -> torch.Tensor:
        err = omega_ref - omega
        lim = self.params.rate_int_limit
        self.integral = (self.integral + err * dt).clamp(-lim, lim)   # anti-windup clamp
        domega = torch.where(self.has_prev, (omega - self.prev_omega) / dt,
                             torch.zeros_like(omega))
        self.prev_omega = omega.clone()
        self.has_prev = torch.ones_like(self.has_prev)
        ang_acc = self.kp * err + self.ki * self.integral - self.kd * domega
        return self.inertia * ang_acc


class QuadSim:
    """N independent quadrotors stepped in lockstep."""

    def __init__(self, n_envs: int, params: Optional[QuadParams] = None,
                 device="cpu", dtype=torch.float32):
        self.n = n_envs
        self.p = params if params is not None else QuadParams()
        self.device, self.dtype = device, dtype
        kw = dict(dtype=dtype, device=device)
        p = self.p

        self.inertia = torch.tensor(p.inertia_diag, **kw)
        self.rotor_pos = torch.tensor([[x, y, p.rotor_z] for x, y in p.rotor_xy], **kw)  # (4,3)
        self.rot_dir = torch.tensor(p.rot_dir, **kw)
        self.drag = torch.tensor(p.drag_coeff, **kw)
        self.g_world = torch.tensor([0.0, 0.0, -p.gravity], **kw)
        self.A_inv = torch.linalg.inv(build_allocation(p, device, dtype))
        self.f_max = p.k_f * p.w_max ** 2
        self.f_min = p.k_f * p.w_min ** 2

        self.pos = torch.zeros(n_envs, 3, **kw)
        self.vel = torch.zeros(n_envs, 3, **kw)
        self.quat = torch.zeros(n_envs, 4, **kw)
        self.quat[:, 0] = 1.0
        self.omega = torch.zeros(n_envs, 3, **kw)
        self.rotor_w = torch.full((n_envs, 4), p.hover_rotor_speed(), **kw)
        self.rate_ctrl = RateController(p, n_envs, device, dtype)

        # per-env TRUE plant (see DOMAIN RANDOMISATION); nominal until randomize()
        self.mass_true = torch.full((n_envs, 1), p.mass, **kw)
        self.inertia_true = self.inertia.expand(n_envs, 3).clone()
        self.kf_true = torch.full((n_envs, 1), p.k_f, **kw)
        self.km_true = torch.full((n_envs, 1), p.k_m, **kw)
        self.drag_true = self.drag.expand(n_envs, 3).clone()
        self.tau_true = torch.full((n_envs, 1), p.tau_motor, **kw)

    # ------------------------------------------------------------------
    def nominal_dynamics(self, env_ids=None):
        """Put the plant of `env_ids` (default all) back to the nominal QuadParams."""
        ids = slice(None) if env_ids is None else torch.as_tensor(env_ids, device=self.device).reshape(-1)
        p = self.p
        self.mass_true[ids] = p.mass
        self.inertia_true[ids] = self.inertia
        self.kf_true[ids] = p.k_f
        self.km_true[ids] = p.k_m
        self.drag_true[ids] = self.drag
        self.tau_true[ids] = p.tau_motor
        self.rate_ctrl.kp[ids] = self.rate_ctrl.kp_nominal

    def randomize(self, env_ids, rand: "DynRand", generator: Optional[torch.Generator] = None):
        """Sample a new plant for `env_ids` from `rand` (see DynRand)."""
        ids = torch.as_tensor(env_ids, device=self.device).reshape(-1)
        k = ids.numel()
        self.nominal_dynamics(ids)
        if k == 0:
            return
        kw = dict(dtype=self.dtype, device=self.device)

        def u(rng, n):
            lo, hi = rng
            return lo + (hi - lo) * torch.rand(k, n, generator=generator, **kw)

        p = self.p
        if rand.mass is not None:
            self.mass_true[ids] = p.mass * u(rand.mass, 1)
        if rand.thrust is not None:
            self.kf_true[ids] = p.k_f * u(rand.thrust, 1)
        if rand.inertia is not None:
            self.inertia_true[ids] = self.inertia * u(rand.inertia, 3)
        if rand.drag is not None:
            self.drag_true[ids] = self.drag * u(rand.drag, 3)
        if rand.k_m is not None:
            self.km_true[ids] = p.k_m * u(rand.k_m, 1)
        if rand.tau_motor is not None:
            self.tau_true[ids] = u(rand.tau_motor, 1)
        if rand.rate_kp is not None:
            self.rate_ctrl.kp[ids] = self.rate_ctrl.kp_nominal * u(rand.rate_kp, 3)

    # ------------------------------------------------------------------
    def reset(self, env_ids=None, pos=None, vel=None, quat=None, omega=None,
              rotor_w=None):
        """Reset the given envs (default all). Unspecified fields go to rest at
        the origin with identity attitude and hover rotor speed."""
        if env_ids is None:
            env_ids = torch.arange(self.n, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device).reshape(-1)
        k = env_ids.numel()
        kw = dict(dtype=self.dtype, device=self.device)

        def given(x, default):
            return default if x is None else torch.as_tensor(x, **kw).expand(k, -1)

        ident = torch.tensor([1.0, 0.0, 0.0, 0.0], **kw).expand(k, 4)
        self.pos[env_ids] = given(pos, torch.zeros(k, 3, **kw))
        self.vel[env_ids] = given(vel, torch.zeros(k, 3, **kw))
        self.quat[env_ids] = quat_normalize(given(quat, ident))
        self.omega[env_ids] = given(omega, torch.zeros(k, 3, **kw))
        hover = torch.full((k, 4), self.p.hover_rotor_speed(), **kw)
        self.rotor_w[env_ids] = given(rotor_w, hover)
        self.rate_ctrl.reset(env_ids)

    # ------------------------------------------------------------------
    def mix(self, thrust_total: torch.Tensor, torque_body: torch.Tensor) -> torch.Tensor:
        """Collective thrust (N,) [N] + body torque (N,3) [N m] -> per-rotor
        thrust (N,4) [N], clipped to the rotor limits."""
        wrench = torch.cat([thrust_total.reshape(-1, 1), torque_body], dim=-1)   # (N,4)
        f = wrench @ self.A_inv.T
        return f.clamp(self.f_min, self.f_max)

    def thrust_to_rotor_speed(self, f: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(f.clamp_min(0.0) / self.p.k_f)

    # ------------------------------------------------------------------
    def step_rotor_speeds(self, w_ref: torch.Tensor, dt: Optional[float] = None):
        """Advance ONE physics step (default physics_dt) given commanded rotor
        speeds (N,4) [rad/s]: first-order motor lag, then rigid-body dynamics."""
        p = self.p
        dt = p.physics_dt if dt is None else dt
        w_ref = w_ref.clamp(p.w_min, p.w_max)
        tau = self.tau_true
        alpha = torch.where(tau > 0.0, 1.0 - torch.exp(-dt / tau.clamp_min(1e-9)),
                            torch.ones_like(tau))                  # (N,1); 1 = no lag
        self.rotor_w = self.rotor_w + alpha * (w_ref - self.rotor_w)

        w2 = self.rotor_w ** 2
        f = self.kf_true * w2                                      # (N,4) rotor thrusts
        R = quat_to_rotmat(self.quat)                              # (N,3,3)

        # forces
        thrust_body = torch.zeros_like(self.pos)
        thrust_body[:, 2] = f.sum(-1)
        v_body = torch.einsum("nji,nj->ni", R, self.vel)           # R^T v
        drag_body = -self.drag_true * v_body
        acc = torch.einsum("nij,nj->ni", R, thrust_body + drag_body) / self.mass_true \
            + self.g_world

        # torques: r_i x f_i (f along +z) + yaw reaction moments - gyroscopic
        tx = (self.rotor_pos[:, 1] * f).sum(-1)
        ty = -(self.rotor_pos[:, 0] * f).sum(-1)
        tz = (self.km_true * w2 * self.rot_dir).sum(-1)
        torque = torch.stack([tx, ty, tz], dim=-1)
        gyro = torch.cross(self.omega, self.inertia_true * self.omega, dim=-1)
        domega = (torque - gyro) / self.inertia_true

        # semi-implicit Euler
        self.vel = self.vel + acc * dt
        self.pos = self.pos + self.vel * dt
        self.omega = self.omega + domega * dt
        self.quat = quat_normalize(quat_mul(self.quat, quat_from_rotvec(self.omega * dt)))

    def step_ctbr(self, thrust_total_N: torch.Tensor, body_rate_ref: torch.Tensor):
        """Advance one CONTROL step (control_dt) with collective thrust [N] and
        body-rate setpoint [rad/s] held constant. The rate loop, mixer and
        motors run at physics_dt inside."""
        p = self.p
        n_sub = max(1, int(round(p.control_dt / p.physics_dt)))
        dt = p.control_dt / n_sub
        thrust_total_N = thrust_total_N.reshape(-1).to(self.dtype)
        body_rate_ref = body_rate_ref.to(self.dtype)
        for _ in range(n_sub):
            torque = self.rate_ctrl(self.omega, body_rate_ref, dt)
            f = self.mix(thrust_total_N, torque)
            self.step_rotor_speeds(self.thrust_to_rotor_speed(f), dt)

    def step_wrench(self, thrust_total_N: torch.Tensor, torque_body: torch.Tensor):
        """Advance one CONTROL step (control_dt) with collective thrust [N] and
        body torque [N m] (N,3) held constant (zero-order hold). The rate
        loop is bypassed: the wrench goes straight through the mixer and motors
        at physics_dt. This is the interface of a torque-level controller such
        as the NMPC in control/, which replaces the rate loop."""
        p = self.p
        n_sub = max(1, int(round(p.control_dt / p.physics_dt)))
        dt = p.control_dt / n_sub
        thrust_total_N = thrust_total_N.reshape(-1).to(self.dtype)
        torque_body = torque_body.to(self.dtype)
        w_ref = self.thrust_to_rotor_speed(self.mix(thrust_total_N, torque_body))
        for _ in range(n_sub):
            self.step_rotor_speeds(w_ref, dt)
