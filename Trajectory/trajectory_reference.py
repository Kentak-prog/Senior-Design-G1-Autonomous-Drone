"""
Min-snap trajectory -> the 12-state reference the NMPC in control/ tracks.

State order (control/New_Drone_State_Equations.py):
    X = [x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p, q, r]

control/Drone_Race_Simulation.py feeds the MPC level attitude (phi = theta = 0)
and zero body rates. A min-snap trajectory also gives acceleration, so this
module fills in the attitude the drone actually needs to produce that
acceleration (differential flatness), using the SAME translational equations
as the control model:

    x_ddot = (U1/m) (cphi sth cpsi + sphi spsi)
    y_ddot = (U1/m) (cphi sth spsi - sphi cpsi)
    z_ddot = (U1/m) cphi cth - g

=> thrust direction b = (a + g e3) / |a + g e3|, rotate by -psi, read off
phi and theta. Body rates p, q, r come from the Euler-rate relations in
control/Drone_State_Equations.py. U1 (feed-forward thrust) is exposed too,
for warm-starting the optimizer.

Heading (psi): starts at the given start heading, blends over the first part
of the climb-out to the direction of travel, and follows the velocity
heading from then on (holding the last heading whenever the drone is nearly
stopped). Near a gate, velocity is along the gate normal, so the forward
camera faces the gate as it goes through.

FRAME: world Z-up throughout (Isaac Sim, vision/, and the control model all
agree), so positions pass through unchanged.
"""

import os
import sys

import numpy as np

_CONTROL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "control")
if _CONTROL not in sys.path:
    sys.path.insert(0, _CONTROL)
from drone_constraints import drone_constraints   # noqa: E402

G = 9.81


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


class TrajectoryReference:
    """
    Pre-samples a MinSnapTrajectory on a fine grid and serves NMPC reference
    states by interpolation. Past the end of the trajectory it holds the final
    hover state.

    m is the drone mass (kg), defaulting to control/drone_constraints.py. It only scales the
    feed-forward thrust -- the attitude reference does not depend on mass.
    """

    def __init__(self, traj, start_yaw: float, dt: float = 0.005,
                 yaw_min_speed: float = 0.3, yaw_blend_speed: float = 1.0,
                 yaw_smooth_sigma: float = 0.3, m: float = drone_constraints["mass"]):  # kg
        self.traj = traj
        self.duration = traj.duration
        times, pos, vel, acc, _, _ = traj.sample(dt)
        self.times = times

        psi = self._heading(times, vel, start_yaw, yaw_min_speed, yaw_blend_speed, yaw_smooth_sigma)
        phi, theta, thrust = self._attitude_from_accel(acc, psi, m)

        # Euler rates -> body rates (inverse of the kinematics in Drone_State_Equations.py)
        dphi, dtheta, dpsi = (np.gradient(a, times) for a in (phi, theta, psi))
        p = dphi - np.sin(theta) * dpsi
        q = np.cos(phi) * dtheta + np.cos(theta) * np.sin(phi) * dpsi
        r = np.cos(theta) * np.cos(phi) * dpsi - np.sin(phi) * dtheta

        self.states = np.vstack([pos.T, phi, theta, psi, vel.T, p, q, r])   # 12 x M
        self.thrust = thrust

    @staticmethod
    def _heading(times, vel, start_yaw, min_speed, blend_speed, yaw_smooth_sigma):
        speed = np.linalg.norm(vel[:, :2], axis=1)
        raw = np.arctan2(vel[:, 1], vel[:, 0])

        # Hold the last good heading wherever the drone is (nearly) stopped
        moving = speed > min_speed
        if not np.any(moving):
            return np.full_like(times, start_yaw)
        idx = np.where(moving, np.arange(len(times)), 0)
        idx = np.maximum.accumulate(idx)
        idx[:np.argmax(moving)] = np.argmax(moving)     # before first motion: first good sample
        heading = np.unwrap(raw[idx])
        # Pick the 2*pi branch closest to the start heading
        heading += 2 * np.pi * np.round((start_yaw - heading[0]) / (2 * np.pi))

        # Speed-weighted Gaussian smoothing: in a tight hairpin the drone slows
        # down and the velocity heading whips around; weighting by speed lets
        # the slow part barely count, so yaw turns through the hairpin smoothly
        # instead of flicking ~130 deg in a second. Through a gate the path is
        # straight, so the smoothed heading still lines up with the gate normal.
        dt = times[1] - times[0]
        half = int(3 * yaw_smooth_sigma / dt)
        kernel = np.exp(-0.5 * (np.arange(-half, half + 1) * dt / yaw_smooth_sigma) ** 2)
        w = np.maximum(speed, 1e-3)
        pad = lambda a: np.pad(a, half, mode='edge')
        heading = (np.convolve(pad(w * heading), kernel, 'valid') /
                   np.convolve(pad(w), kernel, 'valid'))

        # Blend from the start heading to the travel heading while speeding up
        reached = speed >= blend_speed
        t_blend = times[np.argmax(reached)] if np.any(reached) else times[-1]
        w = _smoothstep(times / max(t_blend, 1e-6))
        return start_yaw + w * (heading - start_yaw)

    @staticmethod
    def _attitude_from_accel(acc, psi, m):
        f = np.column_stack((acc[:, 0], acc[:, 1], acc[:, 2] + G))     # = (U1/m) * b
        thrust = m * np.linalg.norm(f, axis=1)
        b = f / np.linalg.norm(f, axis=1, keepdims=True)
        c, s = np.cos(psi), np.sin(psi)
        bx = c * b[:, 0] + s * b[:, 1]          # Rz(-psi) @ b
        by = -s * b[:, 0] + c * b[:, 1]
        phi = np.arcsin(np.clip(-by, -1.0, 1.0))
        theta = np.arctan2(bx, b[:, 2])
        return phi, theta, thrust

    # ------------------------------------------------------------------ queries
    def reference_state(self, t: float) -> np.ndarray:
        """12-vector reference at time t (same role as Drone_Race_Simulation.reference_state)."""
        t = float(np.clip(t, 0.0, self.duration))
        return np.array([np.interp(t, self.times, row) for row in self.states])

    def build_reference_horizon(self, t0: float, N: int, dt: float) -> np.ndarray:
        """12 x (N+1) receding-horizon reference (same signature as Drone_Race_Simulation's)."""
        ts = np.clip(t0 + dt * np.arange(N + 1), 0.0, self.duration)
        return np.array([np.interp(ts, self.times, row) for row in self.states])

    def feedforward_thrust(self, t: float) -> float:
        return float(np.interp(np.clip(t, 0.0, self.duration), self.times, self.thrust))
