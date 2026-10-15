"""
Small batched rotation utilities (torch) for the RL simulator.

CONVENTIONS
    Quaternions are WXYZ, shape (N, 4), and map BODY -> WORLD (same component
    order as vision/camera.quat_to_rot; ROS is XYZW, convert at the boundary).
    Rotation matrices are (N, 3, 3) with R @ v_body = v_world.
    Euler angles are intrinsic ZYX (yaw, then pitch, then roll):
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll). With the body in FLU, positive yaw
    turns the nose counter-clockwise seen from above, positive pitch is
    rotation about +y (left), which points the nose DOWN, positive roll is
    rotation about +x (forward), which dips the RIGHT side.
"""

import torch


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product a * b (apply b first, then a), WXYZ."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(N,4) wxyz -> (N,3,3). Normalises first, so slightly off-unit input is safe."""
    q = quat_normalize(q)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
        2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y),
    ], dim=-1)
    return R.reshape(q.shape[:-1] + (3, 3))


def quat_from_rotvec(v: torch.Tensor) -> torch.Tensor:
    """Exact exponential map: rotation vector (axis * angle) -> quaternion wxyz."""
    angle = v.norm(dim=-1, keepdim=True)
    half = 0.5 * angle
    small = angle < 1e-6
    # sin(half)/angle, with a Taylor expansion where angle -> 0
    k = torch.where(small, 0.5 - angle ** 2 / 48.0,
                    torch.sin(half) / angle.clamp_min(1e-12))
    return torch.cat([torch.cos(half), v * k], dim=-1)


def quat_from_euler_zyx(roll: torch.Tensor, pitch: torch.Tensor,
                        yaw: torch.Tensor) -> torch.Tensor:
    """Euler angles (rad, each shape (N,)) -> quaternion wxyz, R = Rz Ry Rx."""
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    return torch.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dim=-1)


def rotmat_to_euler_zyx(R: torch.Tensor):
    """(N,3,3) -> (roll, pitch, yaw), each (N,). For logging; singular at pitch = +-90 deg."""
    roll = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    pitch = -torch.asin(R[..., 2, 0].clamp(-1.0, 1.0))
    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    return roll, pitch, yaw
