"""
Swift-style state estimation (Kaufmann et al., Nature 2023, Methods "VIO drift estimation") for the
simulator: a base VIO that drifts, and a Kalman filter that estimates that drift from gate sightings.
Plan and rationale: px4/SWIFT_VOXL2_PLAN.md ("The drift filter", "Order of work" step 2).

    DriftEmulator    the BASE VIO. Isaac's true pose plus a drift like a feature-tracking VIO's
                     (what the VOXL 2 runs on the real drone): an initial offset, a scale error
                     (the position error grows with the distance flown), and a velocity random walk.
                     Attitude and body rates are the true ones (a VIO's attitude is good; Swift takes
                     it straight from the VIO). Seeded, so a run can be repeated.
    DriftFilter      Swift's filter. State x = [p_d (3), v_d (3)]: the VIO's position error and its
                     rate, world frame. Predict: p_d += v_d dt, v_d a random walk. Update at each gate
                     sighting: the drone's position from the gate, put through the BASE VIO's attitude
                     at the image time (not the PnP rotation), gives z = p_vio(t_img) - p_from_gate, a
                     measurement of p_d at t_img. Chi-square gate against bad PnP solves and wrong
                     gate matches. Output: VIO position - p_d, VIO velocity - v_d; attitude and rates
                     straight from the VIO.

Frames: world ENU, body FLU, camera OpenCV for detections (vision/gate_pose.GateDetection.T_cam_gate),
mount T_body_cam in the USD camera convention as control/vio.py takes it.
"""

from collections import deque

import numpy as np

R_CV_FROM_USD = np.diag([1.0, -1.0, -1.0])     # vision/camera.py (as control/vio.py)


def _inv(T):
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = R.T, -R.T @ t
    return out


# ---------------------------------------------------------------------------------------------- base VIO
class DriftEmulator:
    """
    Base VIO = truth + drift.
        p_vio = p + p_d,   v_vio = v + v_d,   attitude = true attitude
        v_d   = scale (*) v  +  b,   elementwise scale ~ N(0, scale_std) per axis drawn once (a VIO's scale
                error), b a velocity bias: first-order Gauss-Markov, std bias_std, time constant bias_tau_s
                (bounded: a hovering VIO that sees static features does not run away), p_d' = v_d
        p_d(0) ~ N(0, init_offset_std) per axis (z: half of it)
    Defaults: 2 % scale error, the main term once flying (0.6-0.9 m over a 30-45 m lap; Swift's T265
    "drifts badly at racing speed"), and a 0.005 m/s, 5 s bias (~0.1 m per minute of hovering). A random
    walk in velocity (tried first: 0.05 and 0.005 m/s/sqrt(s)) integrates without bound: 5 m in 20 s and
    3 m in 60 s of hover in Isaac. Size them from a VOXL bag when we have one (SWIFT_VOXL2_PLAN.md step 1).
    """

    def __init__(self, scale_std=0.02, bias_std=0.005, bias_tau_s=5.0, init_offset_std=0.3, seed=0):
        self.rng = np.random.default_rng(seed)
        self.scale = self.rng.normal(0.0, scale_std, 3)
        self.bias_std, self.bias_tau = float(bias_std), float(bias_tau_s)
        self.p_d = self.rng.normal(0.0, init_offset_std, 3) * np.array([1.0, 1.0, 0.5])
        self.b = self.rng.normal(0.0, self.bias_std, 3)
        self.t = None

    def step(self, t, pos, vel):
        """True (pos, vel) at time t -> base VIO (pos, vel). Call in time order."""
        pos, vel = np.asarray(pos, dtype=float), np.asarray(vel, dtype=float)
        if self.t is not None:
            dt = min(max(t - self.t, 0.0), 0.1)
            if dt > 0:
                a = np.exp(-dt / self.bias_tau)
                self.b = a * self.b + self.rng.normal(0.0, self.bias_std * np.sqrt(1.0 - a * a), 3)
            self.p_d += (self.scale * vel + self.b) * dt
        self.t = t
        v_d = self.scale * vel + self.b
        return pos + self.p_d, vel + v_d

    @property
    def drift(self):
        return self.p_d.copy()


# ------------------------------------------------------------------------------------------ drift filter
class DriftFilter:
    """
    gates:           objects with .T_world_gate (4x4) -- Trajectory/gate_path.Gate from the course file
    T_body_cam_usd:  camera mount (control/vio.camera_mount_usd(pitch, translation))
    q_v:             drift-rate random walk, m/s per sqrt(s) (the emulator's vel_rw plus margin for
                     the scale error's changes with speed)
    p0_std, v0_std:  prior on the drift at start (the VIO's start offset is unknown)
    """

    def __init__(self, gates, T_body_cam_usd, q_v=0.15, p0_std=0.5, v0_std=0.1, history_s=1.5,
                 max_range=9.0, min_pos_std=0.05, assoc_tol=1.0, chi2=16.3, att_std_deg=1.0):
        self.gates = list(gates)
        T = np.asarray(T_body_cam_usd, dtype=float).copy()
        T[:3, :3] = T[:3, :3] @ R_CV_FROM_USD.T
        self.T_body_cam_cv = T
        self.q_v = float(q_v)
        self.x = np.zeros(6)
        self.P = np.diag([p0_std ** 2] * 3 + [v0_std ** 2] * 3)
        self.t = None
        self.hist = deque()                    # (t, p_vio, v_vio, R_wb) of the base VIO
        self.history_s = history_s
        self.max_range, self.min_pos_std, self.assoc_tol = max_range, min_pos_std, assoc_tol
        self.chi2 = chi2
        self.att_std = np.radians(att_std_deg)
        self.stats = dict(fixes=0, rejected=0, unmatched=0, out_of_range=0, no_history=0)
        self.events = []                       # (t_img, gate, accepted, chi2, |z - p_d|)

    # ---------------------------------------------------------------------------------------- VIO input
    def add_vio(self, t, p_vio, v_vio, R_wb):
        """One base-VIO sample (world position, world velocity, body->world rotation) at time t."""
        self._predict(t)
        self.hist.append((t, np.asarray(p_vio, float), np.asarray(v_vio, float), np.asarray(R_wb, float)))
        while self.hist and self.hist[0][0] < t - self.history_s:
            self.hist.popleft()

    def _predict(self, t):
        if self.t is None:
            self.t = t
            return
        dt = t - self.t
        if dt <= 0:
            return
        F = np.eye(6)
        F[:3, 3:] = dt * np.eye(3)
        Q = np.zeros((6, 6))
        q = self.q_v ** 2
        Q[:3, :3] = q * dt ** 3 / 3 * np.eye(3)
        Q[:3, 3:] = Q[3:, :3] = q * dt ** 2 / 2 * np.eye(3)
        Q[3:, 3:] = q * dt * np.eye(3)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = t

    def _vio_at(self, t):
        """Base VIO sample nearest in time to t (within 50 ms), or None."""
        if not self.hist:
            return None
        best = min(self.hist, key=lambda h: abs(h[0] - t))
        return best if abs(best[0] - t) <= 0.05 else None

    # ---------------------------------------------------------------------------------------- output
    def corrected(self, p_vio, v_vio):
        return np.asarray(p_vio, float) - self.x[:3], np.asarray(v_vio, float) - self.x[3:]

    def position_std(self):
        return np.sqrt(np.diag(self.P)[:3])

    # ---------------------------------------------------------------------------------------- camera
    def add_detections(self, t_img, dets):
        """Gate detections (vision/gate_pose.GateDetection) from the image captured at t_img.
        Returns the number of fixes applied."""
        h = self._vio_at(t_img)
        if h is None:
            self.stats["no_history"] += 1
            return 0
        _, p_vio, _, R_wb = h
        lag = (self.t - t_img) if self.t is not None else 0.0
        p_est = p_vio - (self.x[:3] - lag * self.x[3:])       # corrected position at t_img
        T_wb = np.eye(4)
        T_wb[:3, :3], T_wb[:3, 3] = R_wb, p_est
        T_wc = T_wb @ self.T_body_cam_cv
        T_cw = _inv(T_wc)
        n = 0
        used = set()
        for det in sorted(dets, key=lambda d: d.range_m):
            if det.range_m > self.max_range:
                self.stats["out_of_range"] += 1
                continue
            c = det.T_cam_gate[:3, 3]
            d = [np.linalg.norm((T_cw @ g.T_world_gate)[:3, 3] - c) if i not in used else np.inf
                 for i, g in enumerate(self.gates)]
            i = int(np.argmin(d))
            if not d[i] < self.assoc_tol + 0.1 * det.range_m:
                self.stats["unmatched"] += 1
                continue
            used.add(i)
            # Swift: the gate's position in the camera, put through the VIO's attitude (not PnP's rotation)
            R_wc = R_wb @ self.T_body_cam_cv[:3, :3]
            p_cam = self.gates[i].T_world_gate[:3, 3] - R_wc @ c
            p_fix = p_cam - R_wb @ self.T_body_cam_cv[:3, 3]
            z = p_vio - p_fix                                  # measured drift at t_img
            H = np.hstack([np.eye(3), -lag * np.eye(3)])       # p_d(t_img) = p_d(now) - lag v_d
            pos_std = max(self.min_pos_std, det.center_std_m if np.isfinite(det.center_std_m) else 0.0,
                          0.004 * det.range_m ** 2)            # PnP depth error grows ~ range^2
            ang = self.att_std * det.range_m                   # attitude error times the lever arm
            R = (pos_std ** 2 + ang ** 2) * np.eye(3)
            y = z - H @ self.x
            S = H @ self.P @ H.T + R
            m2 = float(y @ np.linalg.solve(S, y))
            ok = m2 <= self.chi2
            self.events.append((t_img, i, ok, m2, float(np.linalg.norm(y))))
            if not ok:
                self.stats["rejected"] += 1
                continue
            K = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = (np.eye(6) - K @ H) @ self.P
            self.P = 0.5 * (self.P + self.P.T)
            self.stats["fixes"] += 1
            n += 1
        return n
