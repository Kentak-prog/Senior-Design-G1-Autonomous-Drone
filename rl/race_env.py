"""
Batched, torch-native drone-racing environment for PPO, plus a Stable-Baselines3
VecEnv adapter.

CONVENTIONS (same as the rest of rl/, see drone_params.py / course.py)
    World ENU (z up, gravity -z). Body FLU (x fwd, y left, z up). Quaternions
    WXYZ body -> world. Gate frame +X right / +Y down / +Z flight direction.
    Everything is a torch tensor with a leading env dimension N.

WHY this exists: the policy is a STANDALONE controller compared against the PID
    and the team's MPC, so it speaks the same command as the PID -- collective
    thrust [N] + body rates [rad/s] (CTBR) -- and flies the same QuadSim
    physics. The env is a thin task layer (spawn, observation, reward, reset)
    on top of QuadSim + course.check_gate; nothing is re-implemented.

COURSES
    A pool of courses is packed into padded tensors (pad = repeat the last
    gate, so a clamped index is always valid). Every env owns a course id and a
    target gate index. On reset an env picks a random course AND a random START
    gate, then spawns 2-6 m behind that gate with noise. Randomising the start
    gate is what trains every segment of a track (including the fast ones deep
    into a lap) instead of only the first, as in UZH's Swift.

OBSERVATION (build_obs, shared with rl_policy.RlPolicyController so the two
    cannot diverge). Everything is relative / body-frame, so the policy does
    not depend on where the course is in the world:
        body-frame velocity / SPEED_SCALE (3), world up (+z) expressed in the
        body frame (3; the gravity direction -- deliberately NOT the full
        rotation matrix, whose heading part is irrelevant because every other
        entry is body-relative and courses face random directions), body rates
        / max_rate (3), previous action (4), then for
        the CURRENT and the NEXT gate: centre relative to the drone in the body
        frame / DIST_SCALE (3) and gate normal in the body frame (3); finally
        altitude / DIST_SCALE (1). Last gate: the "next" gate repeats the
        current one. Scales: distances 10 m, speeds 10 m/s, rates by the
        per-axis rate limit, so all entries are O(1).

ACTION (N,4) in [-1,1], clipped. thrust = m g (1 + a0) clipped to
    [0, max_total_thrust] (a0 = 0 hovers), body rates = a[1:4] * max_rate.
    Note a0 <= 1 caps the commanded thrust at 2 g; the rotors could give ~2.8 g.

REWARD / TERMINATION: see RaceEnvConfig (every weight is documented there).
    Crash (gate collision / ground / out of bounds / tilt) and success
    TERMINATE; hitting max_episode_time TRUNCATES (the value function must
    bootstrap through a time limit, not treat it as a failure).

STEP returns (obs, reward, terminated, truncated, infos). infos is a list of N
    dicts, empty except for envs whose episode just ended. Finished envs are
    reset immediately; obs holds the FRESH observation and the old one is in
    infos[i]["terminal_observation"].
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import torch

from course import Course, GateBatch, check_gate, random_course, FIXED_TRACKS
from drone_params import QuadParams
from quad_sim import QuadSim
from rotations import quat_to_rotmat, quat_from_euler_zyx

DIST_SCALE = 10.0     # m   -> distances in the observation are divided by this
SPEED_SCALE = 10.0    # m/s -> velocities in the observation are divided by this
ACT_DIM = 4

CRASH_NONE, CRASH_GATE, CRASH_GROUND, CRASH_OOB, CRASH_TILT = 0, 1, 2, 3, 4
CRASH_NAMES = ["none", "gate", "ground", "oob", "tilt"]


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
@dataclass
class RaceEnvConfig:
    # --- course pool ---
    n_courses: int = 256            # random courses in the training pool
    min_gates: int = 3              # gates per random course ~ U{min..max}
    max_gates: int = 8
    course_seed: int = 0            # seeds the pool (separate from the env seed)
    course_yaw_jitter_deg: float = 0.0   # +- random gate yaw off the travel direction
    # (course.random_course); > 0 teaches turning into gates that face away
    include_fixed_tracks: bool = False   # also train on straight3/oval6/slalom5.
    # Default False so the fixed tracks are a HELD-OUT evaluation set.

    # --- spawn (all relative to the randomly chosen start gate) ---
    spawn_back: tuple = (2.0, 6.0)         # m behind the gate along -normal
    spawn_lateral_std: float = 0.5         # m, along the gate's right axis
    spawn_vertical_std: float = 0.5        # m, world z
    spawn_min_z: float = 0.5               # m, spawn height floor
    spawn_yaw_noise_deg: float = 20.0      # +-, around facing along the normal
    spawn_tilt_noise_deg: float = 10.0     # +-, roll and pitch
    spawn_speed_along: tuple = (0.0, 2.0)  # m/s along the gate normal
    spawn_vel_std: float = 0.3             # m/s, extra Gaussian per axis
    spawn_rate_std: float = 0.2            # rad/s, Gaussian per axis

    # --- observation ---
    obs_noise_std: float = 0.01   # Gaussian noise on the scaled obs (env only;
    # the deployed controller adds none). 0 disables it.

    # --- reward weights (per control step unless noted) ---
    w_progress: float = 1.0       # * (d_prev - d_now), d = distance to the target gate
    #                               centre (Song et al. 2021)
    w_gate: float = 10.0          # bonus for passing the target gate
    w_finish: float = 10.0        # extra bonus for passing the last gate (success)
    w_crash: float = 10.0         # penalty (subtracted) on any crash
    w_rate: float = 0.01          # * ||omega|| [rad/s], body-rate penalty
    w_smooth: float = 0.001       # * ||a - a_prev||^2, action smoothness
    w_perception: float = 0.0     # * exp(-angle^4), angle = body +x vs direction to
    # the target gate centre. OFF until the camera mount direction is pinned down
    # (the camera may not look along body +x).

    # --- termination ---
    max_episode_time: float = 20.0   # s; truncation
    ground_z: float = 0.05           # m; below this = ground crash
    max_gate_dist: float = 30.0      # m; farther from the target gate = out of bounds
    max_tilt_deg: float = 80.0       # deg between body z and world z


# ----------------------------------------------------------------------
# Packed courses
# ----------------------------------------------------------------------
class CoursePool:
    """C courses padded to Gmax gates as tensors (see module docstring)."""

    def __init__(self, courses: Sequence[Course], device="cpu", dtype=torch.float32):
        kw = dict(dtype=dtype, device=device)
        C = len(courses)
        G = max(len(c) for c in courses)
        centers = np.zeros((C, G, 3))
        R = np.zeros((C, G, 3, 3))
        side = np.zeros((C, G))
        bar = np.zeros((C, G))
        for ci, c in enumerate(courses):
            for gi in range(G):
                g = c.gates[min(gi, len(c) - 1)]      # pad with the last gate
                T = g.T_world_gate()
                centers[ci, gi], R[ci, gi] = T[:3, 3], T[:3, :3]
                side[ci, gi], bar[ci, gi] = g.side, g.bar
        self.centers = torch.as_tensor(centers, **kw)      # (C,G,3)
        self.R = torch.as_tensor(R, **kw)                  # (C,G,3,3) gate -> world
        self.side = torch.as_tensor(side, **kw)            # (C,G)
        self.bar = torch.as_tensor(bar, **kw)              # (C,G)
        self.n_gates = torch.tensor([len(c) for c in courses], dtype=torch.long,
                                    device=device)         # (C,)
        self.names = [c.name for c in courses]
        self.n_courses = C

    def gates(self, course_id: torch.Tensor, gate_idx: torch.Tensor) -> GateBatch:
        """GateBatch for (course_id (N,), gate_idx (N,)); the index is clamped
        into [0, n_gates-1], so idx = n_gates (finished) and "next of last" are
        safe. Equivalent to course.gate_batch for a single course."""
        i = torch.minimum(gate_idx.clamp_min(0), self.n_gates[course_id] - 1)
        return GateBatch(center=self.centers[course_id, i], R=self.R[course_id, i],
                         side=self.side[course_id, i], bar=self.bar[course_id, i])


def make_course_pool(cfg: RaceEnvConfig) -> List[Course]:
    """Seeded random courses (+ the fixed tracks if configured)."""
    rng = np.random.default_rng(cfg.course_seed)
    courses: List[Course] = []
    while len(courses) < cfg.n_courses:
        n = int(rng.integers(cfg.min_gates, cfg.max_gates + 1))
        try:
            courses.append(random_course(rng, n, name=f"rand_{len(courses):03d}",
                                         yaw_jitter_deg=cfg.course_yaw_jitter_deg))
        except RuntimeError:       # extremely rare: could not place gates
            continue
    if cfg.include_fixed_tracks:
        courses += [make() for make in FIXED_TRACKS.values()]
    return courses


# ----------------------------------------------------------------------
# Observation
# ----------------------------------------------------------------------
def obs_dim() -> int:
    return 3 + 3 + 3 + 4 + 2 * (3 + 3) + 1


def build_obs(pos, vel, quat, omega, prev_action, gate1: GateBatch, gate2: GateBatch,
              params: QuadParams) -> torch.Tensor:
    """(N, obs_dim) observation; layout and scales in the module docstring.
    Used by RaceEnv AND rl_policy.RlPolicyController."""
    R = quat_to_rotmat(quat)                                    # body -> world
    max_rate = torch.tensor(params.max_rate, dtype=pos.dtype, device=pos.device)

    def to_body(v_world):                                       # R^T v
        return torch.einsum("nji,nj->ni", R, v_world)

    parts = [to_body(vel) / SPEED_SCALE, R[:, 2, :],                 # R^T e_z = row 2 of R
             omega / max_rate, prev_action]
    for g in (gate1, gate2):
        parts.append(to_body(g.center - pos) / DIST_SCALE)
        parts.append(to_body(g.R[:, :, 2]))                     # gate +Z = normal
    parts.append(pos[:, 2:3] / DIST_SCALE)
    return torch.cat(parts, dim=-1)


# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------
class RaceEnv:
    render_mode = None

    def __init__(self, n_envs: int, cfg: Optional[RaceEnvConfig] = None,
                 params: Optional[QuadParams] = None,
                 courses: Optional[Sequence[Course]] = None, seed: int = 0,
                 device="cpu", dtype=torch.float32):
        self.n = n_envs
        self.cfg = cfg if cfg is not None else RaceEnvConfig()
        self.p = params if params is not None else QuadParams()
        self.device, self.dtype = device, dtype
        if courses is None:
            courses = make_course_pool(self.cfg)
        self.pool = CoursePool(courses, device, dtype)
        self.sim = QuadSim(n_envs, self.p, device=device, dtype=dtype)
        self.obs_dim = obs_dim()
        self.max_steps = int(round(self.cfg.max_episode_time / self.p.control_dt))
        self.max_rate = torch.tensor(self.p.max_rate, dtype=dtype, device=device)
        self.cos_tilt_max = math.cos(math.radians(self.cfg.max_tilt_deg))

        kw = dict(device=device)
        self.course_id = torch.zeros(n_envs, dtype=torch.long, **kw)
        self.gate_idx = torch.zeros(n_envs, dtype=torch.long, **kw)
        self.start_idx = torch.zeros(n_envs, dtype=torch.long, **kw)
        self.ep_len = torch.zeros(n_envs, dtype=torch.long, **kw)
        self.ep_return = torch.zeros(n_envs, dtype=dtype, **kw)
        self.prev_action = torch.zeros(n_envs, ACT_DIM, dtype=dtype, **kw)
        self.gen = torch.Generator(device=device)
        self.seed(seed)
        self.reset()

    # ------------------------------------------------------------------
    def seed(self, seed: int):
        self.gen.manual_seed(int(seed))

    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, device=self.device, dtype=self.dtype)

    def _randn(self, *shape):
        return torch.randn(*shape, generator=self.gen, device=self.device, dtype=self.dtype)

    def _uniform(self, k, lo, hi, n=None):
        shape = (k,) if n is None else (k, n)
        return lo + (hi - lo) * self._rand(*shape)

    # ------------------------------------------------------------------
    def reset(self) -> torch.Tensor:
        """Reset every env; returns the observation (N, obs_dim)."""
        self._reset_envs(torch.arange(self.n, device=self.device))
        return self._observe()

    def _reset_envs(self, ids: torch.Tensor):
        """Respawn the envs `ids` behind a random gate of a random course."""
        cfg = self.cfg
        k = ids.numel()
        cid = (self._rand(k) * self.pool.n_courses).long().clamp_max(self.pool.n_courses - 1)
        ng = self.pool.n_gates[cid]
        gi = (self._rand(k) * ng).long()
        gi = torch.minimum(gi, ng - 1)
        self.course_id[ids], self.gate_idx[ids], self.start_idx[ids] = cid, gi, gi

        c = self.pool.centers[cid, gi]
        Rg = self.pool.R[cid, gi]
        normal, right = Rg[:, :, 2], Rg[:, :, 0]
        back = self._uniform(k, *cfg.spawn_back)
        lat = cfg.spawn_lateral_std * self._randn(k)
        vert = cfg.spawn_vertical_std * self._randn(k)
        pos = c - back.unsqueeze(-1) * normal + lat.unsqueeze(-1) * right
        pos[:, 2] = (pos[:, 2] + vert).clamp_min(cfg.spawn_min_z)

        d2r = math.pi / 180.0
        heading = torch.atan2(normal[:, 1], normal[:, 0]) \
            + self._uniform(k, -1, 1) * cfg.spawn_yaw_noise_deg * d2r
        roll = self._uniform(k, -1, 1) * cfg.spawn_tilt_noise_deg * d2r
        pitch = self._uniform(k, -1, 1) * cfg.spawn_tilt_noise_deg * d2r
        quat = quat_from_euler_zyx(roll, pitch, heading)
        speed = self._uniform(k, *cfg.spawn_speed_along)
        vel = speed.unsqueeze(-1) * normal + cfg.spawn_vel_std * self._randn(k, 3)
        omega = cfg.spawn_rate_std * self._randn(k, 3)

        self.sim.reset(ids, pos=pos, vel=vel, quat=quat, omega=omega)
        self.ep_len[ids] = 0
        self.ep_return[ids] = 0.0
        self.prev_action[ids] = 0.0

    def set_state(self, pos, vel=None, quat=None, omega=None, course_id=0, gate_idx=0,
                  env_ids=None):
        """Place envs in a chosen state (tests, debugging). Resets counters and
        the previous action."""
        ids = torch.arange(self.n, device=self.device) if env_ids is None \
            else torch.as_tensor(env_ids, device=self.device).reshape(-1)
        k = ids.numel()
        self.sim.reset(ids, pos=pos, vel=vel, quat=quat, omega=omega)
        self.course_id[ids] = torch.as_tensor(course_id, device=self.device).expand(k)
        g = torch.as_tensor(gate_idx, device=self.device).expand(k)
        self.gate_idx[ids], self.start_idx[ids] = g, g
        self.ep_len[ids] = 0
        self.ep_return[ids] = 0.0
        self.prev_action[ids] = 0.0

    # ------------------------------------------------------------------
    def _observe(self, ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        s = self.sim
        if ids is None:
            ids = slice(None)
        cid, gi = self.course_id[ids], self.gate_idx[ids]
        obs = build_obs(s.pos[ids], s.vel[ids], s.quat[ids], s.omega[ids],
                        self.prev_action[ids], self.pool.gates(cid, gi),
                        self.pool.gates(cid, gi + 1), self.p)
        if self.cfg.obs_noise_std > 0.0:
            obs = obs + self.cfg.obs_noise_std * self._randn(*obs.shape)
        return obs

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, action: torch.Tensor):
        cfg, p, s = self.cfg, self.p, self.sim
        a = action.to(device=self.device, dtype=self.dtype).clamp(-1.0, 1.0)
        thrust = (p.mass * p.gravity * (1.0 + a[:, 0])).clamp(0.0, p.max_total_thrust())
        rates = a[:, 1:] * self.max_rate

        gates = self.pool.gates(self.course_id, self.gate_idx)
        p_prev = s.pos.clone()
        d_prev = (p_prev - gates.center).norm(dim=-1)
        s.step_ctbr(thrust, rates)
        p_now = s.pos
        d_now = (p_now - gates.center).norm(dim=-1)

        passed, hit = check_gate(p_prev, p_now, gates, p.collision_radius)
        R = quat_to_rotmat(s.quat)
        ground = p_now[:, 2] < cfg.ground_z
        oob = d_now > cfg.max_gate_dist
        tilt = R[:, 2, 2] < self.cos_tilt_max          # cos(angle(body z, world z))
        crash = hit | ground | oob | tilt
        reason = torch.zeros_like(self.gate_idx)       # priority gate > ground > oob > tilt
        reason = torch.where(tilt, torch.full_like(reason, CRASH_TILT), reason)
        reason = torch.where(oob, torch.full_like(reason, CRASH_OOB), reason)
        reason = torch.where(ground, torch.full_like(reason, CRASH_GROUND), reason)
        reason = torch.where(hit, torch.full_like(reason, CRASH_GATE), reason)

        n_pass = passed & ~crash
        self.gate_idx = self.gate_idx + n_pass.long()
        success = n_pass & (self.gate_idx >= self.pool.n_gates[self.course_id])

        # --- reward ---
        reward = cfg.w_progress * (d_prev - d_now)
        reward = reward + cfg.w_gate * n_pass.to(self.dtype) + cfg.w_finish * success.to(self.dtype)
        reward = reward - cfg.w_crash * crash.to(self.dtype)
        reward = reward - cfg.w_rate * s.omega.norm(dim=-1)
        reward = reward - cfg.w_smooth * (a - self.prev_action).pow(2).sum(-1)
        if cfg.w_perception != 0.0:
            to_gate = gates.center - p_now
            cosang = (R[:, :, 0] * to_gate).sum(-1) / to_gate.norm(dim=-1).clamp_min(1e-6)
            ang = torch.acos(cosang.clamp(-1.0, 1.0))
            reward = reward + cfg.w_perception * torch.exp(-ang ** 4)

        self.prev_action = a
        self.ep_len = self.ep_len + 1
        self.ep_return = self.ep_return + reward

        terminated = crash | success
        truncated = (self.ep_len >= self.max_steps) & ~terminated
        done = terminated | truncated

        obs = self._observe()
        infos = [{} for _ in range(self.n)]
        ids = torch.nonzero(done).flatten()
        if ids.numel() > 0:
            term_obs = obs[ids].clone()
            ep_r = self.ep_return[ids].tolist()
            ep_l = self.ep_len[ids].tolist()
            passed_n = (torch.minimum(self.gate_idx[ids], self.pool.n_gates[self.course_id[ids]])
                        - self.start_idx[ids]).tolist()
            succ = success[ids].tolist()
            why = reason[ids].tolist()
            for j, i in enumerate(ids.tolist()):
                infos[i] = {"episode": {"r": ep_r[j], "l": ep_l[j]},
                            "gates_passed": passed_n[j], "success": succ[j],
                            "crash_reason": CRASH_NAMES[why[j]],
                            "terminal_observation": term_obs[j]}
            self._reset_envs(ids)
            obs[ids] = self._observe(ids)
        return obs, reward, terminated, truncated, infos


# ----------------------------------------------------------------------
# Stable-Baselines3 adapter
# ----------------------------------------------------------------------
from gymnasium import spaces                                    # noqa: E402
from stable_baselines3.common.vec_env import VecEnv             # noqa: E402


class RaceVecEnv(VecEnv):
    """Wraps RaceEnv for SB3: numpy float32 in/out, infos as SB3 expects
    (terminal_observation, TimeLimit.truncated, episode {r,l}, is_success)."""

    def __init__(self, n_envs: int, cfg: Optional[RaceEnvConfig] = None,
                 params: Optional[QuadParams] = None,
                 courses: Optional[Sequence[Course]] = None, seed: int = 0,
                 device="cpu"):
        self.env = RaceEnv(n_envs, cfg, params, courses, seed, device)
        obs_space = spaces.Box(-np.inf, np.inf, (self.env.obs_dim,), dtype=np.float32)
        act_space = spaces.Box(-1.0, 1.0, (ACT_DIM,), dtype=np.float32)
        super().__init__(n_envs, obs_space, act_space)
        self._actions = None

    def reset(self):
        self.env.reset()
        self._reset_seeds()
        self._reset_options()
        return self.env._observe().numpy().astype(np.float32)

    def step_async(self, actions: np.ndarray):
        self._actions = actions

    def step_wait(self):
        act = torch.as_tensor(np.asarray(self._actions), dtype=self.env.dtype)
        obs, rew, term, trunc, infos = self.env.step(act)
        term, trunc = term.numpy(), trunc.numpy()
        for i, info in enumerate(infos):
            if info:                                   # episode ended in env i
                info["terminal_observation"] = info["terminal_observation"].numpy()
                info["TimeLimit.truncated"] = bool(trunc[i])
                info["is_success"] = bool(info["success"])
        return (obs.numpy().astype(np.float32), rew.numpy().astype(np.float32),
                term | trunc, infos)

    def close(self):
        pass

    def seed(self, seed=None):
        if seed is None:
            seed = int(np.random.randint(0, 2 ** 31 - 1))
        self.env.seed(seed)
        return [seed + i for i in range(self.num_envs)]

    # --- attribute plumbing: one shared env, so every index sees the same value ---
    def get_attr(self, attr_name, indices=None):
        src = self.env if hasattr(self.env, attr_name) else self
        return [getattr(src, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.env, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        fn = getattr(self.env, method_name)
        return [fn(*method_args, **method_kwargs) for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]
