"""
RlPolicyController: a trained PPO policy behind the same controller interface
as pid_controller.PidController, so fly_course.run_controller (and the PID-vs-RL
comparison) treats them identically:

    reset(n_envs, device, dtype)
    act(pos, vel, quat, gate_idx, dt, omega=...) -> (thrust [N] (N,), body-rate ref [rad/s] (N,3))

CONVENTIONS as in race_env.py: world ENU, body FLU, quaternions WXYZ body ->
world, gate frame +X right / +Y down / +Z flight direction.

WHY it needs `omega` and keeps state: the observation contains the body rates
and the previous action, which the old act(pos, vel, quat, gate_idx, dt)
signature does not carry. fly_course.run_controller now also passes
omega=sim.omega, and the controller remembers its own last action.

The observation is built by race_env.build_obs -- the very function the
training env calls -- from the course's gates (current and next, found from
gate_idx), then normalised with the VecNormalize statistics saved at training
time, then fed to the policy deterministically. The action maps to a command
exactly as RaceEnv.step does: thrust = m g (1 + a0), rates = a[1:] * max_rate.

Loading: a run directory containing model.zip and vecnormalize.pkl. For use
inside training (periodic evaluation) pass `policy` and `vecnorm` objects
instead; the normaliser statistics are then read live.
"""

import os
import pickle

import numpy as np
import torch

from course import Course
from drone_params import QuadParams
from race_env import CoursePool, build_obs, ACT_DIM


class RlPolicyController:
    def __init__(self, params: QuadParams, course: Course, model_dir: str = None,
                 policy=None, vecnorm=None):
        self.p = params
        self.course = course
        if model_dir is not None:
            from stable_baselines3 import PPO
            model_dir = model_dir if os.path.isdir(model_dir) else os.path.dirname(model_dir)
            policy = PPO.load(os.path.join(model_dir, "model.zip"), device="cpu").policy
            with open(os.path.join(model_dir, "vecnormalize.pkl"), "rb") as f:
                vecnorm = pickle.load(f)
        if policy is None or vecnorm is None:
            raise ValueError("give model_dir, or both policy and vecnorm")
        self.policy, self.vecnorm = policy, vecnorm

    def reset(self, n_envs: int, device="cpu", dtype=torch.float32):
        self.n, self.device, self.dtype = n_envs, device, dtype
        self.pool = CoursePool([self.course], device, dtype)
        self.course_id = torch.zeros(n_envs, dtype=torch.long, device=device)
        self.prev_action = torch.zeros(n_envs, ACT_DIM, dtype=dtype, device=device)

    def observe(self, pos, vel, quat, omega, gate_idx) -> torch.Tensor:
        """Raw (un-normalised) observation, identical to RaceEnv's (without noise)."""
        return build_obs(pos, vel, quat, omega, self.prev_action,
                         self.pool.gates(self.course_id, gate_idx),
                         self.pool.gates(self.course_id, gate_idx + 1), self.p)

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        rms = self.vecnorm.obs_rms
        out = (obs - rms.mean) / np.sqrt(rms.var + self.vecnorm.epsilon)
        return np.clip(out, -self.vecnorm.clip_obs, self.vecnorm.clip_obs).astype(np.float32)

    def act(self, pos, vel, quat, gate_idx, dt, omega=None):
        if omega is None:
            raise ValueError("RlPolicyController needs omega (body rates)")
        obs = self.observe(pos, vel, quat, omega, gate_idx)
        a_np, _ = self.policy.predict(self.normalize(obs.numpy()), deterministic=True)
        a = torch.as_tensor(a_np, dtype=self.dtype, device=self.device).clamp(-1.0, 1.0)
        self.prev_action = a
        p = self.p
        thrust = (p.mass * p.gravity * (1.0 + a[:, 0])).clamp(0.0, p.max_total_thrust())
        rates = a[:, 1:] * torch.tensor(p.max_rate, dtype=self.dtype, device=self.device)
        return thrust, rates
