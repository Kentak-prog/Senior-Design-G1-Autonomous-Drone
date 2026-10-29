"""
A PX4-like plant for rehearsing the MPC the way Isaac flies it: the vehicle
takes thrust + an ATTITUDE setpoint (what mpc_offboard_bridge.py sends PX4),
not thrust + torques (what the NMPC's own model takes).

  attitude  follows the setpoint as a first-order lag with time constant tau
            (PX4's attitude + rate loops; ~0.05-0.1 s on the Iris)
  rates     the body rates (p, q, r) of that attitude change -- the gyro sees
            these, so they must be body rates, not Euler-angle rates
  thrust    applied directly along body z; gravity; no drag

Used by control/test_mpc_core.py and Trajectory/simulate_vio_mpc.py --plant px4.
State order as everywhere: [x, y, z, roll, pitch, yaw, vx, vy, vz, p, q, r]
(world ENU, body FLU, R = Rz(yaw) Ry(pitch) Rx(roll)).
"""

import numpy as np

G = 9.81


def _rot(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


class Px4AttitudePlant:
    def __init__(self, m, tau=0.08):
        self.m, self.tau = float(m), float(tau)

    def step(self, x, thrust, att_cmd, dt):
        """x: 12-vector (or 12x1), thrust (N), att_cmd (roll, pitch, yaw) rad -> next 12x1 state after dt."""
        x = np.asarray(x, dtype=float).reshape(12)
        att, vel, pos = x[3:6].copy(), x[6:9].copy(), x[0:3].copy()
        cmd = np.asarray(att_cmd, dtype=float).reshape(3).copy()
        cmd[2] = att[2] + (cmd[2] - att[2] + np.pi) % (2 * np.pi) - np.pi          # yaw the short way
        a = min(1.0, dt / self.tau)
        new_att = att + a * (cmd - att)
        dphi, dth, dpsi = (new_att - att) / dt                     # Euler-angle rates
        phi, th = 0.5 * (att[0] + new_att[0]), 0.5 * (att[1] + new_att[1])
        rates = np.array([dphi - np.sin(th) * dpsi,                 # -> body rates (ZYX Euler kinematics)
                          np.cos(phi) * dth + np.sin(phi) * np.cos(th) * dpsi,
                          -np.sin(phi) * dth + np.cos(phi) * np.cos(th) * dpsi])
        acc = _rot(*new_att) @ np.array([0.0, 0.0, float(thrust) / self.m]) - np.array([0.0, 0.0, G])
        new_vel = vel + acc * dt
        new_pos = pos + 0.5 * (vel + new_vel) * dt
        return np.r_[new_pos, new_att, new_vel, rates].reshape(12, 1)
