"""
Classical cascaded PID baseline that outputs the SAME command as the future RL
policy: collective thrust [N] + body-rate setpoint [rad/s] (CTBR). Sharing the
command interface (and the QuadSim rate loop underneath) is what makes the
PID-vs-RL comparison fair.

CONVENTIONS
    World ENU (z up), body FLU, quaternions WXYZ body -> world, body rates in
    the body frame, batched over a leading env dimension (see drone_params.py).
    Gate frame / flight direction as in course.py: the world-frame flight
    direction of a gate is its +Z axis.

CASCADE
    waypoint -> position P -> desired velocity (clamped to v_max)
             -> velocity PID -> desired acceleration a
             -> desired thrust vector F = m (a + g z_world), tilt-limited
             -> collective thrust = F . (body z axis in world)
             -> desired attitude Rd (body z along F, nose toward the target yaw)
             -> attitude P on the rotation error -> body-rate setpoint.

    Rotation error: e_R = vee(0.5 (Rd^T R - R^T Rd)). If the body is rotated by
    a small angle th (body-frame axis) away from the desired attitude,
    R = Rd Exp(th), Rd^T R = Exp(th), and 0.5(Exp - Exp^T) = [th]x, so e_R = th.
    The corrective rate is therefore -Kp * e_R (the sign that is easy to get wrong).

WAYPOINT LOGIC (mirrors vision/gate_pose.approach_waypoints, implemented here
    so rl/ does not import vision/): for the current target gate fly
        stage 0: pre-gate point  = center - standoff * normal
        stage 1: gate center
        stage 2: post-gate point = center + standoff * normal
    until the runner reports the gate passed (gate_idx advances), which resets
    to stage 0 of the next gate. Flying the normal line keeps the approach
    square to the opening, which is what stops the drone clipping a frame.
"""

import math
from dataclasses import dataclass

import torch

from course import Course
from drone_params import QuadParams
from rotations import quat_to_rotmat


@dataclass
class PidGains:
    standoff: float = 1.5          # m, pre/post point distance from the gate plane
    pre_tol: float = 0.5           # m, switch pre -> center within this distance of the pre point
    center_tol: float = 0.4        # m, switch center -> post within this distance of the center
    v_max: float = 4.0             # m/s, desired-velocity clamp
    v_max_gate: float = 3.0        # m/s, clamp while lined up on the gate (stages 1, 2)
    kp_pos: float = 1.5            # 1/s, position error -> desired velocity
    kp_vel: float = 4.0            # 1/s, velocity error -> acceleration
    ki_vel: float = 0.5            # 1/s^2, integral term (absorbs drag bias)
    kd_vel: float = 0.05           # D on velocity measurement (dimensionless)
    vel_int_limit: float = 2.0     # m/s (integral of velocity error), anti-windup
    a_max: float = 10.0            # m/s^2, desired-acceleration norm clamp
    max_tilt_deg: float = 35.0     # tilt limit of the desired thrust vector
    kp_att: tuple = (10.0, 10.0, 3.0)   # 1/s on rotation error (roll, pitch, yaw)


class PidController:
    """Batched. Call reset() once, then act() every control step."""

    def __init__(self, params: QuadParams, course: Course, gains: PidGains = None):
        self.p = params
        self.course = course
        self.g = gains if gains is not None else PidGains()
        self._gate_pos = None

    def reset(self, n_envs: int, device="cpu", dtype=torch.float32):
        kw = dict(device=device, dtype=dtype)
        self.n, self.device, self.dtype = n_envs, device, dtype
        import numpy as np
        centers = torch.as_tensor(np.stack([g.center for g in self.course.gates]), **kw)
        normals = torch.as_tensor(np.stack([g.normal for g in self.course.gates]), **kw)
        self.centers, self.normals = centers, normals
        self.stage = torch.zeros(n_envs, dtype=torch.long, device=device)
        self.last_idx = torch.zeros(n_envs, dtype=torch.long, device=device)
        self.vel_int = torch.zeros(n_envs, 3, **kw)
        self.prev_vel = None
        self.kp_att = torch.tensor(self.g.kp_att, **kw)
        self.max_rate = torch.tensor(self.p.max_rate, **kw)

    # ------------------------------------------------------------------
    def _target(self, pos, gate_idx):
        """Update the per-env stage and return (target point, gate center, normal)."""
        g = self.g
        idx = gate_idx.clamp(0, len(self.course) - 1)
        c, n = self.centers[idx], self.normals[idx]
        changed = idx != self.last_idx
        self.stage = torch.where(changed, torch.zeros_like(self.stage), self.stage)
        self.last_idx = idx

        pre, post = c - g.standoff * n, c + g.standoff * n
        d_pre = (pos - pre).norm(dim=-1)
        d_ctr = (pos - c).norm(dim=-1)
        self.stage = torch.where((self.stage == 0) & (d_pre < g.pre_tol),
                                 torch.ones_like(self.stage), self.stage)
        self.stage = torch.where((self.stage == 1) & (d_ctr < g.center_tol),
                                 torch.full_like(self.stage, 2), self.stage)
        st = self.stage.unsqueeze(-1)
        target = torch.where(st == 0, pre, torch.where(st == 1, c, post))
        return target, c, n

    def act(self, pos, vel, quat, gate_idx, dt):
        """pos, vel (N,3), quat (N,4) wxyz, gate_idx (N,) long -> (thrust (N,) [N],
        body-rate setpoint (N,3) [rad/s])."""
        g, p = self.g, self.p
        target, c, n = self._target(pos, gate_idx)

        # position P -> desired velocity
        v_cap = torch.where(self.stage == 0,
                            torch.full_like(pos[:, 0], g.v_max),
                            torch.full_like(pos[:, 0], g.v_max_gate)).unsqueeze(-1)
        v_des = g.kp_pos * (target - pos)
        sp = v_des.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        v_des = v_des * torch.minimum(torch.ones_like(sp), v_cap / sp)

        # velocity PID -> desired acceleration
        v_err = v_des - vel
        self.vel_int = (self.vel_int + v_err * dt).clamp(-g.vel_int_limit, g.vel_int_limit)
        dv = torch.zeros_like(vel) if self.prev_vel is None else (vel - self.prev_vel) / dt
        self.prev_vel = vel.clone()
        a = g.kp_vel * v_err + g.ki_vel * self.vel_int - g.kd_vel * dv
        an = a.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        a = a * torch.minimum(torch.ones_like(an), g.a_max / an)

        # desired thrust vector, tilt limited
        F = p.mass * a
        F[:, 2] = F[:, 2] + p.mass * p.gravity
        F[:, 2] = F[:, 2].clamp_min(0.3 * p.mass * p.gravity)
        tan_max = math.tan(math.radians(g.max_tilt_deg))
        Fh = F[:, :2]
        Fh_n = Fh.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        Fh = Fh * torch.minimum(torch.ones_like(Fh_n), tan_max * F[:, 2:3] / Fh_n)
        F = torch.cat([Fh, F[:, 2:3]], dim=-1)

        # collective thrust: projection of F onto the current body z axis
        R = quat_to_rotmat(quat)
        thrust = (F * R[:, :, 2]).sum(-1).clamp(0.0, p.max_total_thrust())

        # desired attitude: body z along F, yaw toward the gate
        b3 = F / F.norm(dim=-1, keepdim=True)
        to_gate = c[:, :2] - pos[:, :2]
        far = to_gate.norm(dim=-1) > 1.0          # direction to the gate is unstable up close
        yaw_dir = torch.where((self.stage == 0).unsqueeze(-1) & far.unsqueeze(-1),
                              to_gate, n[:, :2])
        yaw = torch.atan2(yaw_dir[:, 1], yaw_dir[:, 0])
        heading = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=-1)
        b2 = torch.cross(b3, heading, dim=-1)
        b2 = b2 / b2.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        b1 = torch.cross(b2, b3, dim=-1)
        Rd = torch.stack([b1, b2, b3], dim=-1)    # columns

        # attitude P on rotation error -> body-rate setpoint
        M = Rd.transpose(-1, -2) @ R
        S = 0.5 * (M - M.transpose(-1, -2))
        e_R = torch.stack([S[:, 2, 1], S[:, 0, 2], S[:, 1, 0]], dim=-1)
        rates = -self.kp_att * e_R
        rates = torch.maximum(torch.minimum(rates, self.max_rate), -self.max_rate)
        return thrust, rates
