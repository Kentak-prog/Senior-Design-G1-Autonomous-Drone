"""
MPC core (no ROS): follows a min-snap reference with the repo NMPC.

  TrajectoryBuffer  a stamped min-snap reference (times relative to t0, the 12
                    reference states per sample), evaluated over the MPC
                    horizon at any time `now`
  MpcCore           the NMPC (MPC_Optimizer.create_drone_mpc_optimizer) with
                    warm starting and the previous command carried over
  hover_reference   a hold-this-point reference

Used by the ROS 2 node ros2_nodes/mpc_node.py and by the Python rehearsals
(Trajectory/simulate_vio_mpc.py). Tested in control/test_mpc_core.py.

Reference / state row order: [x, y, z, roll, pitch, yaw, vx, vy, vz, p, q, r]
(world ENU, body FLU, R = Rz(yaw) Ry(pitch) Rx(roll)).
"""

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from MPC_Optimizer import create_drone_mpc_optimizer   # noqa: E402
from drone_constraints import drone_constraints        # noqa: E402

G = 9.81
POINT_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz", "p", "q", "r")


def stamp_to_sec(stamp) -> float:
    """builtin_interfaces/Time (or anything with .sec / .nanosec) -> seconds."""
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


class TrajectoryBuffer:
    """
    A min-snap reference: times relative to t0 (seconds, ROS time of t = 0) and
    the 12 reference states per sample. Evaluated by linear interpolation;
    clamped to the first sample before t0 and to the last after the end (the
    plan ends at rest, so that is a hover).
    """

    def __init__(self, t0, times, states, thrust=None):
        self.t0 = float(t0)
        self.times = np.asarray(times, dtype=float)
        self.states = np.array(states, dtype=float)          # 12 x M
        self.states[5] = np.unwrap(self.states[5])           # continuous yaw along the plan
        self.thrust = None if thrust is None else np.asarray(thrust, dtype=float)
        if self.states.shape[0] != 12 or self.states.shape[1] != len(self.times) or len(self.times) < 2:
            raise ValueError("TrajectoryBuffer needs 12 x M states for M >= 2 times")

    @classmethod
    def from_msg(cls, msg):
        pts = msg.points
        times = [pt.t for pt in pts]
        states = [[getattr(pt, f) for pt in pts] for f in POINT_FIELDS]
        return cls(stamp_to_sec(msg.header.stamp), times, states, [pt.thrust for pt in pts])

    @property
    def duration(self):
        return float(self.times[-1])

    def ended(self, now):
        return now - self.t0 > self.duration

    def horizon(self, now, N, dt, yaw_now=None):
        """12 x (N+1) reference from `now`; yaw shifted by whole turns to sit next to yaw_now."""
        ts = np.clip(now - self.t0 + dt * np.arange(N + 1), 0.0, self.duration)
        ref = np.array([np.interp(ts, self.times, row) for row in self.states])
        if yaw_now is not None:
            ref[5] += 2 * np.pi * np.round((yaw_now - ref[5, 0]) / (2 * np.pi))
        return ref


def hover_reference(pos, yaw, N):
    ref = np.zeros((12, N + 1))
    ref[0:3] = np.asarray(pos, dtype=float).reshape(3, 1)
    ref[5] = yaw
    return ref


class MpcCore:
    """The repo NMPC with warm starting and the previous command carried over."""

    def __init__(self, N=10, dt=0.02, m=drone_constraints["mass"]):
        self.N, self.dt, self.m = N, dt, m
        (self.opti, self.X, self.U, self.X_init, self.U_prev, self.X_ref) = create_drone_mpc_optimizer(
            N=N, dt=dt, m=m)
        self.hover_u = np.array([[m * G], [0.0], [0.0], [0.0]])
        self.u_prev = self.hover_u.copy()
        self._X_guess = None
        self._U_guess = None

    def reset(self):
        self.u_prev = self.hover_u.copy()
        self._X_guess = self._U_guess = None

    def step(self, x, ref):
        """
        x: 12x1 state, ref: 12 x (N+1). Returns dict(u0 4x1, x1 12x1 predicted next
        state, ok bool, solve_ms float). On failure: u0 = hover thrust, x1 = level at
        the current yaw, ok = False.
        """
        x = np.asarray(x, dtype=float).reshape(12, 1)
        if self._X_guess is None:
            self._X_guess = np.tile(x, (1, self.N + 1))
            self._U_guess = np.tile(self.hover_u, (1, self.N))
        self._X_guess[:, 0:1] = x                       # the guess must start where the drone is
        self.opti.set_value(self.X_init, x)
        self.opti.set_value(self.U_prev, self.u_prev)
        self.opti.set_value(self.X_ref, ref)
        self.opti.set_initial(self.X, self._X_guess)
        self.opti.set_initial(self.U, self._U_guess)
        t0 = time.perf_counter()
        ok = True
        try:
            sol = self.opti.solve()
            X_sol, U_sol = np.asarray(sol.value(self.X)), np.asarray(sol.value(self.U))
        except RuntimeError:
            X_sol, U_sol = np.asarray(self.opti.debug.value(self.X)), np.asarray(self.opti.debug.value(self.U))
            ok = bool(np.all(np.isfinite(X_sol)) and np.all(np.isfinite(U_sol)))
        solve_ms = 1000.0 * (time.perf_counter() - t0)

        if not ok:
            self.reset()
            x1 = x.copy()
            x1[3:5] = 0.0
            x1[9:12] = 0.0
            return {"u0": self.hover_u.copy(), "x1": x1, "ok": False, "solve_ms": solve_ms}

        u0 = U_sol[:, 0].reshape(4, 1)
        self.u_prev = u0
        self._X_guess = np.hstack([X_sol[:, 1:], X_sol[:, -1:]])     # shift one step for the next solve
        self._U_guess = np.hstack([U_sol[:, 1:], U_sol[:, -1:]])
        return {"u0": u0, "x1": X_sol[:, 1].reshape(12, 1), "ok": True, "solve_ms": solve_ms}
