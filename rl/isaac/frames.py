"""
Frame and unit conversions between PX4 (MAVLink) and the rl/ stack. Pure
numpy: no pymavlink, no torch, importable anywhere.

CONVENTIONS
    rl/ side (see HANDOFF.md): world ENU (x east, y north, z up), body FLU
        (x forward, y left, z up), quaternions WXYZ body -> world, yaw
        counter-clockwise from east.
    PX4 side: local frame NED (x north, y east, z down) with its origin at the
        EKF origin; body FRD (x forward, y right, z down); the attitude
        quaternion (MAVLink ATTITUDE_QUATERNION q1..q4 = w, x, y, z) rotates
        body FRD -> local NED; body rates are FRD; yaw is measured clockwise
        from north.
    "Rotation of frame b expressed in frame a" is written R_a_b (maps b -> a).

DERIVATION (why the quaternion formula is what it is)
    A vector in FLU body coordinates v_flu becomes v_frd = F v_flu with
    F = diag(1, -1, -1) (x stays, y flips left->right, z flips up->down).
    PX4's attitude rotates it into NED: v_ned = R_ned_frd v_frd. NED -> ENU is
    v_enu = M v_ned with M = [[0,1,0],[1,0,0],[0,0,-1]] (east = NED y, north =
    NED x, up = -down). Chaining,
        R_enu_flu = M R_ned_frd F.
    Both M and F are proper rotations (det +1): F is a half turn about x, M a
    half turn about the axis (1,1,0)/sqrt(2). Their quaternions are
        q_F = (0, 1, 0, 0),  q_M = (0, s, s, 0),  s = 1/sqrt(2),
    so q_enu_flu = q_M * q_ned_frd * q_F (Hamilton product). The inverse uses
    the conjugates. Quaternions q and -q are the same rotation; results are
    returned with w >= 0.
    Check: level, nose north (NED/FRD identity) -> q_M * q_F = (-s,0,0,-s) ->
    (s,0,0,s) = +90 deg about up = nose along +y (north) in ENU. Correct.

YAW: yaw_ned (clockwise from north) = pi/2 - yaw_enu (counter-clockwise
    from east). Facing north: yaw_enu = pi/2 -> yaw_ned = 0.

THRUST: PX4 SET_ATTITUDE_TARGET takes a normalised collective thrust in [0, 1].
    ThrustMap converts to/from Newtons; see the class docstring for the two
    modes and the assumption they rest on.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # rl/
from drone_params import QuadParams  # noqa: E402

_S = math.sqrt(0.5)
Q_M = np.array([0.0, _S, _S, 0.0])     # NED -> ENU as a rotation (half turn about (1,1,0))
Q_F = np.array([0.0, 1.0, 0.0, 0.0])   # FLU -> FRD as a rotation (half turn about x)
M_NED_TO_ENU = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
F_FLU_TO_FRD = np.diag([1.0, -1.0, -1.0])


# --------------------------------------------------------------------------
# vectors
# --------------------------------------------------------------------------
def ned_to_enu(v):
    """(3,) or (N,3): (n, e, d) -> (e, n, -d)."""
    v = np.asarray(v, dtype=np.float64)
    return np.stack([v[..., 1], v[..., 0], -v[..., 2]], axis=-1)


def enu_to_ned(v):
    """(3,) or (N,3): (e, n, u) -> (n, e, -u)."""
    v = np.asarray(v, dtype=np.float64)
    return np.stack([v[..., 1], v[..., 0], -v[..., 2]], axis=-1)


def rates_frd_to_flu(w):
    """Body rates FRD -> FLU: (p, q, r) -> (p, -q, -r). Pitch and yaw rate flip
    because FLU y and z point opposite to FRD y and z."""
    w = np.asarray(w, dtype=np.float64)
    return np.stack([w[..., 0], -w[..., 1], -w[..., 2]], axis=-1)


def rates_flu_to_frd(w):
    """Same sign flips; the map is its own inverse."""
    return rates_frd_to_flu(w)


# --------------------------------------------------------------------------
# quaternions (WXYZ, numpy, batched over leading dims)
# --------------------------------------------------------------------------
def _qmul(a, b):
    aw, ax, ay, az = (a[..., i] for i in range(4))
    bw, bx, by, bz = (b[..., i] for i in range(4))
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def _qconj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def _canon(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(q[..., :1] < 0.0, -q, q)


def quat_ned_frd_to_enu_flu(q_wxyz):
    """PX4 attitude (body FRD -> NED) -> rl/ attitude (body FLU -> ENU)."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    return _canon(_qmul(_qmul(Q_M, q), Q_F))


def quat_enu_flu_to_ned_frd(q_wxyz):
    """Inverse of quat_ned_frd_to_enu_flu."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    return _canon(_qmul(_qmul(_qconj(Q_M), q), _qconj(Q_F)))


def quat_to_rotmat(q_wxyz):
    """(4,) or (N,4) WXYZ -> (3,3) or (N,3,3), R @ v_body = v_world."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = (q[..., i] for i in range(4))
    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
        2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y),
    ], axis=-1)
    return R.reshape(q.shape[:-1] + (3, 3))


def quat_from_yaw(yaw):
    """Rotation about +z by `yaw` (rad) as WXYZ (works for ENU/FLU and NED/FRD
    level attitudes alike)."""
    yaw = np.asarray(yaw, dtype=np.float64)
    return np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], axis=-1)


def yaw_of_quat_enu(q_wxyz):
    """Heading (rad, counter-clockwise from east) of the body x axis, from an
    ENU/FLU quaternion."""
    R = quat_to_rotmat(q_wxyz)
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def tilt_of_quat(q_wxyz):
    """Angle (rad) between body z and world z (valid for ENU/FLU or NED/FRD:
    both have body z / world z antiparallel-or-parallel consistently)."""
    R = quat_to_rotmat(q_wxyz)
    return np.arccos(np.clip(R[..., 2, 2], -1.0, 1.0))


# --------------------------------------------------------------------------
# yaw
# --------------------------------------------------------------------------
def wrap_pi(a):
    return (np.asarray(a, dtype=np.float64) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_enu_to_ned(yaw_enu):
    return wrap_pi(math.pi / 2.0 - np.asarray(yaw_enu, dtype=np.float64))


def yaw_ned_to_enu(yaw_ned):
    return wrap_pi(math.pi / 2.0 - np.asarray(yaw_ned, dtype=np.float64))


# --------------------------------------------------------------------------
# thrust
# --------------------------------------------------------------------------
class ThrustMap:
    """Collective thrust [N] <-> PX4 normalised thrust u in [0, 1].

    ASSUMPTION both modes rest on: PX4 THR_MDL_FAC = 0 (so normalised thrust
    is passed to the motors as-is, u_motor = u) and Pegasus turns a motor
    output u into rotor speed w = 1000 u + 100 with thrust 4 k_f w^2. If either
    is false the mapping is wrong; the hover calibration measures `scale`.

    mode "model": thrust(u) = scale * QuadParams.px4_output_to_thrust(u). The
        exact quadratic model; hover is u ~ 0.556 for the default Iris.
    mode "linear": thrust(u) = scale * m g * u / u_hover, i.e. thrust
        proportional to u, calibrated so that u_hover carries the weight. Use
        it when a measured hover throttle is all you trust.
    `scale` is the factor by which the REAL thrust exceeds the model's, fitted
    by `calibrate.py hover`. to_px4 therefore asks the model for T / scale.
    Both directions clip to [0, 1] (to_px4 on u; from_px4 on its input).
    """

    def __init__(self, mode="model", params=None, u_hover=None, scale=1.0):
        if mode not in ("model", "linear"):
            raise ValueError(f"unknown thrust map mode {mode!r}")
        if scale <= 0.0:
            raise ValueError("scale must be positive")
        self.mode = mode
        self.p = params if params is not None else QuadParams()
        self.u_hover = float(u_hover) if u_hover is not None else self.p.hover_motor_output()
        if not 0.0 < self.u_hover <= 1.0:
            raise ValueError("u_hover must be in (0, 1]")
        self.scale = float(scale)

    def to_px4(self, T_newton):
        T = float(T_newton) / self.scale
        if self.mode == "model":
            return self.p.thrust_to_px4_output(T)          # already clipped to [0, 1]
        u = self.u_hover * T / self.p.hover_thrust()
        return min(max(u, 0.0), 1.0)

    def from_px4(self, u):
        u = min(max(float(u), 0.0), 1.0)
        if self.mode == "model":
            return self.scale * self.p.px4_output_to_thrust(u)
        return self.scale * self.p.hover_thrust() * u / self.u_hover

    def __repr__(self):
        return (f"ThrustMap(mode={self.mode!r}, u_hover={self.u_hover:.4f}, "
                f"scale={self.scale:.4f})")
