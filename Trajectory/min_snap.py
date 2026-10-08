"""
Aaron Arnold
ver: 10.8.26

minimum-snap trajectory solver

each segment between waypoints is a 7th-order polynomial per axis, the solver
minimizes the integral of squared snap (4th derivative of position) subject to:
  - passing through every waypoint (gate)
  - rest-to-rest boundary conditions (zero vel/acc/jerk at start and finish)
  - continuity of vel/acc/jerk/snap at interior waypoints (the optimum is
    then C6 smooth on its own)
  - optional velocity DIRECTION at chosen waypoints: velocity is forced
    parallel to a given vector (e.g. a gate normal, so the drone crosses the
    gate perpendicular to its plane) while the optimizer still picks the speed

The direction constraints couple x/y/z, so all three axes are solved together
in one closed-form KKT system. No external optimizer is needed, only numpy.

Pure math: knows nothing about gates, cameras or the controller. See
gate_path.py for the gate-specific planning on top of this.
"""

import numpy as np
from math import factorial

ORDER = 7                 # polynomial order -> 8 coefficients per segment
N_COEFF = ORDER + 1
SNAP = 4                  # derivative being minimized
CONTINUITY = 4            # derivatives 1-4 enforced continuous at interior waypoints

# D[r, k] = k! / (k - r)!  : multiplier on coefficient k in the r-th derivative
D = np.array([[factorial(k) / factorial(k - r) if k >= r else 0.0 for k in range(N_COEFF)]
              for r in range(N_COEFF)])


def poly_deriv_row(t: float, r: int) -> np.ndarray:
    """Row vector mapping polynomial coefficients -> r-th derivative at time t."""
    k = np.arange(N_COEFF)
    return D[r] * t ** np.maximum(k - r, 0)


def snap_cost_matrix(T: float) -> np.ndarray:
    """Q such that c^T Q c = integral_0^T (d^4 p / dt^4)^2 dt for one segment."""
    k = np.arange(N_COEFF)
    p = k[:, None] + k[None, :] - 2 * SNAP + 1
    safe_p = np.maximum(p, 1)
    return np.outer(D[SNAP], D[SNAP]) * np.where(p > 0, T ** safe_p / safe_p, 0.0)


def _perpendicular_basis(d: np.ndarray):
    """Two unit vectors perpendicular to d (and to each other)."""
    d = d / np.linalg.norm(d)
    helper = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u1 = np.cross(d, helper)
    u1 /= np.linalg.norm(u1)
    return u1, np.cross(d, u1)


class MinSnapTrajectory:
    """
    waypoints:     (M+1) x 3 positions, in flight order
    segment_times: M durations in seconds
    fixed_dir:     {waypoint_index: direction (3,)} -- velocity at that
                   (interior) waypoint is constrained parallel to direction
    """

    def __init__(self, waypoints, segment_times, fixed_dir=None):
        self.waypoints = np.asarray(waypoints, dtype=float)
        self.n_seg = len(self.waypoints) - 1
        if self.n_seg < 1:
            raise ValueError("Need at least two waypoints.")
        self.fixed_dir = dict(fixed_dir or {})
        self.T = np.asarray(segment_times, dtype=float)
        self.t_breaks = np.concatenate(([0.0], np.cumsum(self.T)))
        self.coeffs, self.snap_cost = self._solve()   # coeffs: (3 axes, M segments, 8)

    def _solve(self):
        n, M = N_COEFF, self.n_seg
        N = n * M                      # unknowns per axis
        NV = 3 * N                     # unknowns total

        Q_axis = np.zeros((N, N))
        for s in range(M):
            Q_axis[s*n:(s+1)*n, s*n:(s+1)*n] = snap_cost_matrix(self.T[s])
        Q = np.kron(np.eye(3), Q_axis)

        A_rows, b = [], []

        def add(blocks, value):
            """blocks: list of (axis, segment, coefficient row)."""
            row = np.zeros(NV)
            for ax, seg, vec in blocks:
                start = ax * N + seg * n
                row[start:start + n] += vec
            A_rows.append(row)
            b.append(value)

        for ax in range(3):
            pts = self.waypoints[:, ax]

            # Waypoint positions: each segment starts and ends on its waypoints
            for s in range(M):
                add([(ax, s, poly_deriv_row(0.0, 0))], pts[s])
                add([(ax, s, poly_deriv_row(self.T[s], 0))], pts[s + 1])

            # Rest-to-rest: zero vel/acc/jerk at start and finish
            for r in range(1, 4):
                add([(ax, 0, poly_deriv_row(0.0, r))], 0.0)
                add([(ax, M - 1, poly_deriv_row(self.T[-1], r))], 0.0)

            # Interior continuity
            for s in range(M - 1):
                for r in range(1, CONTINUITY + 1):
                    add([(ax, s, poly_deriv_row(self.T[s], r)),
                         (ax, s + 1, -poly_deriv_row(0.0, r))], 0.0)

        # Velocity direction at interior waypoints: v . u1 = 0 and v . u2 = 0
        for idx, direction in self.fixed_dir.items():
            if 0 < idx < self.n_seg:
                vel_row = poly_deriv_row(self.T[idx - 1], 1)
                for u in _perpendicular_basis(np.asarray(direction, dtype=float)):
                    add([(ax, idx - 1, u[ax] * vel_row) for ax in range(3) if abs(u[ax]) > 1e-12], 0.0)

        A = np.array(A_rows)
        b = np.array(b)

        # KKT system:  [2Q  A^T] [c]   [0]
        #              [A   0  ] [l] = [b]
        m = A.shape[0]
        KKT = np.zeros((NV + m, NV + m))
        KKT[:NV, :NV] = 2.0 * Q
        KKT[:NV, NV:] = A.T
        KKT[NV:, :NV] = A
        rhs = np.concatenate((np.zeros(NV), b))

        c = np.linalg.solve(KKT, rhs)[:NV]
        return c.reshape(3, M, n), float(c @ Q @ c)

    @property
    def duration(self) -> float:
        return float(self.t_breaks[-1])

    def segment_index(self, t):
        return np.clip(np.searchsorted(self.t_breaks, t, side='right') - 1, 0, self.n_seg - 1)

    def evaluate(self, t: float, derivative: int = 0) -> np.ndarray:
        """Derivative (0=pos, 1=vel, 2=acc, 3=jerk, 4=snap) at global time t (clamped to [0, duration])."""
        t = float(np.clip(t, 0.0, self.duration))
        seg = int(self.segment_index(t))
        return self.coeffs[:, seg, :] @ poly_deriv_row(t - self.t_breaks[seg], derivative)

    def sample(self, dt: float = 0.02, times=None):
        """Returns (times, pos, vel, acc, jerk, snap), each N x 3, vectorized."""
        if times is None:
            times = np.arange(0.0, self.duration + 1e-9, dt)
        times = np.clip(np.asarray(times, dtype=float), 0.0, self.duration)
        seg = self.segment_index(times)
        tau = times - self.t_breaks[seg]
        seg_coeffs = self.coeffs[:, seg, :]                  # (3, n_samples, n_coeff)

        k = np.arange(N_COEFF)
        out = []
        for d in range(5):
            powers = tau[:, None] ** np.maximum(k - d, 0) * D[d]   # (n_samples, n_coeff)
            out.append(np.einsum('asc,sc->sa', seg_coeffs, powers))
        return (times, *out)


def initial_segment_times(points, v_cruise: float) -> np.ndarray:
    dists = np.linalg.norm(np.diff(np.asarray(points), axis=0), axis=1)
    return np.maximum(dists / v_cruise, 0.2)


def fit_to_limits(points, fixed_dir=None, v_max: float = 4.0, a_max: float = 6.0,
                  v_cruise_frac: float = 0.6) -> MinSnapTrajectory:
    """
    Solve with distance-proportional segment times, then uniformly rescale all
    times so the peak speed or acceleration just reaches v_max / a_max.

    Direction constraints are homogeneous, so scaling every segment time by s
    scales velocity by exactly 1/s and acceleration by 1/s^2 -- the shape is
    unchanged. The loop only absorbs sampling error in the peak estimates.
    """
    T = initial_segment_times(points, v_cruise=v_cruise_frac * v_max)
    traj = None
    for _ in range(5):
        traj = MinSnapTrajectory(points, T, fixed_dir)
        _, _, vel, acc, _, _ = traj.sample(0.005)
        scale = max(np.max(np.linalg.norm(vel, axis=1)) / v_max,
                    np.sqrt(np.max(np.linalg.norm(acc, axis=1)) / a_max))
        if abs(scale - 1.0) < 1e-3:
            break
        T = T * scale
    return traj
