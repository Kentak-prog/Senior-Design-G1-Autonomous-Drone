"""
Gate-based visual-inertial odometry (VIO): IMU + PnP fixes on a KNOWN gate map.

The drone's pose comes ONLY from the IMU and the camera. No GPS, no simulator
ground truth, no PX4 position. Built on Kalman_Filter_MPC.DroneCasadiEKF12D
(strapdown IMU prediction + partial-state updates).

  IMU (every sample, 250 Hz in Pegasus)
      FRD -> FLU, subtract the gyro bias, predict_imu at the FIXED sample
      period (1 / imu_rate of sim time -- never a timestamp difference, see
      below), update_gyro. The filter state after every sample is kept for
      history_s seconds.

  Camera (every frame with gates detected; vision/gate_pose.estimate_gate_pose)
      Each detection is matched to a map gate by where the current estimate
      expects that gate in the camera. Then:
        normal reliable   -> full pose fix: T_world_body = T_world_gate(map) *
                             inv(T_cam_gate) * inv(T_body_cam)   (position + roll/pitch/yaw)
        normal unreliable -> position-only fix, using the ESTIMATED attitude
      Innovations failing a chi-square gate are rejected (a mismatched or
      bad PnP solve cannot yank the estimate).

  Latency: a fix is applied at the image's capture time: rewind to the
      stored state at that time, update, then replay the IMU samples since.

  Start-up: (1) calibration -- the first calib_s seconds of IMU (drone
      hovering, e.g. after `commander takeoff`) give the gyro bias, which
      Pegasus draws up to ~0.5 deg/s at power-on and holds nearly constant;
      (2) initialization -- the first fix with a reliable normal sets the
      pose (velocity 0, hovering). The spawn pose from the gate manifest is
      only a hint for matching detections to map gates before that.

TIMES: add_imu / add_detections take timestamps on ONE clock (sim time: run
workspace/sim_clock.py in Isaac so Pegasus' IMU and the camera agree). The
timestamps only place a camera frame among the IMU samples; integration
always uses the fixed IMU period, because /clock ticks once per rendered
frame and several IMU samples can share a stamp.

Frames: world ENU, body FLU, camera USD (Isaac) / OpenCV (vision) as in
vision/camera.py. State order [x, y, z, roll, pitch, yaw, vx, vy, vz, p, q, r].
"""

from collections import deque

import numpy as np

from Kalman_Filter_MPC import DroneCasadiEKF12D, wrap_angle

R_CV_FROM_USD = np.diag([1.0, -1.0, -1.0])     # vision/camera.py
FRD_FROM_FLU = np.diag([1.0, -1.0, -1.0])      # same matrix both ways


def rot_rpy(roll, pitch, yaw):
    """R = Rz(yaw) Ry(pitch) Rx(roll), body -> world (the control model's convention)."""
    cf, sf, ct, st, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cp * ct, cp * st * sf - sp * cf, cp * st * cf + sp * sf],
                     [sp * ct, sp * st * sf + cp * cf, sp * st * cf - cp * sf],
                     [-st, ct * sf, ct * cf]])


def rpy_from_rot(R):
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def _T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def _inv(T):
    R, t = T[:3, :3], T[:3, 3]
    return _T(R.T, -R.T @ t)


def camera_mount_usd(pitch_deg=15.0, translation=(0.30, 0.0, 0.05)):
    """
    Body FLU -> camera USD for the front camera, as workspace/spawn_example.py
    sets it up (translate, then rotateXYZ (90 + pitch, 0, -90)). pitch_deg > 0
    tilts the camera UP. 15 deg = spawn_example.py today, which equals
    vision/eval_isaac_gates.T_body_cam_usd_from_mount().
    """
    from eval_isaac_gates import rotate_xyz_deg_to_matrix
    return _T(rotate_xyz_deg_to_matrix(90.0 + pitch_deg, 0.0, -90.0), np.asarray(translation, dtype=float))


class GateVIO:
    def __init__(self, gates, T_body_cam_usd, imu_rate=250.0, accel_noise=0.07, gyro_noise=0.006,
                 calib_s=2.0, history_s=1.5, max_range=15.0, min_pos_std=0.05, min_ang_std_deg=1.0,
                 assoc_tol=1.0, chi2_full=22.5, chi2_pos=16.3, initial_guess=None, delay_compensation=True,
                 accel_rw=0.1, relock_n=5, relock_window_s=0.5, relock_spread_m=0.3, relock_max_range=10.0,
                 pos_only_max_range=8.0, relock_blind_s=1.0):
        """
        gates:          map gates (gate_path.Gate, world ENU), e.g. gates_from_manifest(test_gates.json)
        T_body_cam_usd: static camera mount, body FLU -> camera USD (eval_isaac_gates.T_body_cam_usd_from_mount)
        initial_guess:  (pos ENU, yaw) used ONLY to match the first detections to map gates
        chi2_full/pos:  innovation gates (99.9 % for 6 / 3 degrees of freedom)
        accel_rw:       unmodelled acceleration, m/s^2 per sqrt(s) (Kalman_Filter_MPC). In Isaac
                        (2026-11-06) with white accel noise only, the reported std was 3.5x too small
                        after 4.8 s without fixes (0.30 m against 1.06 m), so good gate_2 fixes were
                        rejected. The drift was a 0.11 m/s velocity error plus about 0.04 m/s^2 (tilt).
        pos_only_max_range: position-only fixes (gate normal unreliable) from farther than this are
                        dropped. Such a fix goes through the ESTIMATED attitude, so its error grows with
                        range x attitude error, and the remaining ones carry that in their noise.
                        Isaac 2026-11-06, attempt 2: 91 position-only fixes of gate_2 at 9-11 m moved
                        the estimate 0.64 m sideways before gate_0 (frame margin 0.11 m).
        relock_*:       RE-LOCK. relock_n full fixes of the same gate, all rejected within
                        relock_window_s and agreeing within relock_spread_m (range under
                        relock_max_range), mean the estimate is wrong, not the camera: widen the
                        covariance to the innovation and accept the fix. relock_n=0 disables.
                        Only after a BLIND SPELL (no fix accepted for relock_blind_s): while fixes
                        are flowing, a gate that keeps disagreeing is the biased one. Isaac
                        2026-11-06, attempt 3: a re-lock at engage took the half-clipped gate_0's
                        biased fixes over a healthy estimate (0.09 m off), and the drone missed gate_0.
        """
        self.gates = list(gates)
        self.T_body_cam_usd = np.asarray(T_body_cam_usd, dtype=float)
        self.T_body_cam_cv = self.T_body_cam_usd.copy()
        self.T_body_cam_cv[:3, :3] = self.T_body_cam_usd[:3, :3] @ R_CV_FROM_USD.T
        self.dt = 1.0 / float(imu_rate)
        self.ekf = DroneCasadiEKF12D(dt=self.dt, accel_noise=accel_noise, gyro_noise=gyro_noise, accel_rw=accel_rw)
        self.relock_n, self.relock_window = relock_n, relock_window_s
        self.relock_spread, self.relock_max_range = relock_spread_m, relock_max_range
        self.pos_only_max_range = pos_only_max_range
        self.relock_blind = relock_blind_s
        self._last_fix_t = None                         # image time of the last accepted fix
        self._rejected = deque()                        # (t_img, gate index, z) of recently rejected full fixes
        self.calib_n = int(round(calib_s * imu_rate))
        self.max_range, self.assoc_tol = max_range, assoc_tol
        self.min_pos_std, self.min_ang_std = min_pos_std, np.radians(min_ang_std_deg)
        self.chi2_full, self.chi2_pos = chi2_full, chi2_pos
        self.initial_guess = initial_guess
        self.delay_compensation = delay_compensation

        self._calib_gyro, self._calib_acc = [], []
        self.gyro_bias = np.zeros(3)
        self.calibrated = self.calib_n == 0
        self.initialized = False
        self.t = None                                   # time of the latest IMU sample
        n_hist = int(np.ceil(history_s * imu_rate)) + 1
        self.hist = deque(maxlen=n_hist)                # (t, x, P) after each IMU sample
        self.imu_hist = deque(maxlen=n_hist)            # (t, acc_flu, gyro_flu_corrected)
        self.stats = {"imu": 0, "fixes_full": 0, "fixes_pos": 0, "rejected": 0, "unmatched": 0,
                      "too_old": 0, "relocks": 0, "pos_only_far": 0, "init_time": None}
        self.events = []                                # (t, kind, gate, innovation norm)

    # ------------------------------------------------------------------ IMU
    def add_imu(self, t, accel_frd, gyro_frd):
        """One IMU sample (Pegasus /drone00/sensors/imu: FRD body frame, specific force)."""
        acc = FRD_FROM_FLU @ np.asarray(accel_frd, dtype=float)
        gyro = FRD_FROM_FLU @ np.asarray(gyro_frd, dtype=float)
        self.stats["imu"] += 1
        if not self.calibrated:
            self._calib_gyro.append(gyro)
            self._calib_acc.append(acc)
            if len(self._calib_gyro) >= self.calib_n:
                self.gyro_bias = np.mean(self._calib_gyro, axis=0)
                self.calibrated = True
            self.t = t
            return
        g = gyro - self.gyro_bias
        self.t = t
        if not self.initialized:
            self.imu_hist.append((t, acc, g))           # kept so the init fix can replay from its time
            return
        self._propagate(acc, g)
        self.hist.append((t, self.ekf.x.copy(), self.ekf.P.copy()))
        self.imu_hist.append((t, acc, g))

    def _propagate(self, acc, g):
        self.ekf.predict_imu(acc, g, dt=self.dt)
        self.ekf.update_gyro(g)

    @property
    def calibration_summary(self):
        if not self._calib_acc:
            return None
        a = np.mean(self._calib_acc, axis=0)
        return {"gyro_bias_deg_s": np.degrees(self.gyro_bias), "accel_mean": a, "accel_norm": float(np.linalg.norm(a))}

    # ------------------------------------------------------------------ camera
    def _T_world_body_est(self, x=None):
        x = self.ekf.x[:, 0] if x is None else x
        return _T(rot_rpy(*x[3:6]), x[0:3])

    def _guess_T_world_body(self):
        if self.initialized:
            return self._T_world_body_est()
        if self.initial_guess is None:
            return None
        pos, yaw = self.initial_guess
        return _T(rot_rpy(0.0, 0.0, yaw), np.asarray(pos, dtype=float))

    def _associate(self, dets, T_world_body):
        """Greedy: each detection to the map gate whose expected camera-frame center is nearest."""
        T_cam_world = _inv(T_world_body @ self.T_body_cam_cv)
        expected = [T_cam_world @ g.T_world_gate for g in self.gates]
        pairs, used = [], set()
        for det in sorted(dets, key=lambda d: d.range_m):
            if det.range_m > self.max_range:
                continue
            c = det.T_cam_gate[:3, 3]
            d = [np.linalg.norm(Te[:3, 3] - c) if i not in used else np.inf for i, Te in enumerate(expected)]
            i = int(np.argmin(d))
            if d[i] < self.assoc_tol + 0.1 * det.range_m:
                pairs.append((det, i, expected[i]))
                used.add(i)
            else:
                self.stats["unmatched"] += 1
        return pairs

    def _body_pose_from(self, det, gate_idx, T_cam_gate_expected):
        """Full-pose measurement (T_world_body) from one detection of a known gate."""
        T_cg = det.T_cam_gate.copy()
        # A square gate looks the same from behind: if PnP's normal points the other way from
        # the map gate's (the drone is past it), turn the detection 180 deg about the gate's y.
        if T_cg[:3, 2] @ T_cam_gate_expected[:3, 2] < 0:
            T_cg[:3, 0] *= -1.0
            T_cg[:3, 2] *= -1.0
        T_world_cam_cv = self.gates[gate_idx].T_world_gate @ _inv(T_cg)
        return T_world_cam_cv @ _inv(self.T_body_cam_cv)

    def _stds(self, det):
        pos = max(self.min_pos_std, det.center_std_m if np.isfinite(det.center_std_m) else 0.0,
                  0.004 * det.range_m ** 2)            # PnP depth error grows ~ range^2
        ang = max(self.min_ang_std,
                  np.radians(det.normal_std_deg) if np.isfinite(det.normal_std_deg) else 0.0)
        return pos, ang

    def add_detections(self, t_img, dets):
        """
        Gate detections (GateDetection list) from the image captured at t_img.
        Returns the number of fixes applied.
        """
        if not dets or not self.calibrated:
            return 0
        T_wb = self._guess_T_world_body()
        if T_wb is None:
            return 0
        pairs = self._associate(dets, T_wb)
        if not pairs:
            return 0

        if not self.initialized:
            reliable = [p for p in pairs if p[0].normal_is_reliable]
            if not reliable:
                return 0
            det, i, Te = reliable[0]                    # nearest reliable gate
            T = self._body_pose_from(det, i, Te)
            pos_std, ang_std = self._stds(det)
            self.ekf.x = np.zeros((12, 1))
            self.ekf.x[0:3, 0] = T[:3, 3]
            self.ekf.x[3:6, 0] = rpy_from_rot(T[:3, :3])
            self.ekf.P = np.diag([pos_std ** 2] * 3 + [ang_std ** 2] * 3 + [0.05 ** 2] * 3 + [0.01] * 3)
            self.initialized = True
            self.stats["init_time"] = t_img
            self.events.append((t_img, "init", self.gates[i].name, 0.0))
            # replay the IMU recorded since the image
            later = [s for s in self.imu_hist if s[0] > t_img]
            self.imu_hist.clear()
            self.hist.clear()
            for (t, acc, g) in later:
                self._propagate(acc, g)
                self.hist.append((t, self.ekf.x.copy(), self.ekf.P.copy()))
                self.imu_hist.append((t, acc, g))
            return 1

        # rewind to the image time
        replay = []
        if self.delay_compensation and self.hist:
            if t_img < self.hist[0][0]:
                self.stats["too_old"] += 1
                return 0
            k = max(i for i, h in enumerate(self.hist) if h[0] <= t_img) if self.hist[0][0] <= t_img else None
            if k is not None and k < len(self.hist) - 1:
                _, x_k, P_k = self.hist[k]
                self.ekf.x, self.ekf.P = x_k.copy(), P_k.copy()
                replay = list(self.imu_hist)[k + 1:]
                hist_keep = list(self.hist)[:k + 1]
            else:
                hist_keep = None
        else:
            hist_keep = None

        applied = 0
        for det, i, _ in pairs:
            # re-match against the rewound estimate (it is what the camera saw)
            T_cam_world = _inv(self._T_world_body_est() @ self.T_body_cam_cv)
            Te = T_cam_world @ self.gates[i].T_world_gate
            pos_std, ang_std = self._stds(det)
            if det.normal_is_reliable:
                T = self._body_pose_from(det, i, Te)
                z = np.concatenate([T[:3, 3], rpy_from_rot(T[:3, :3])])
                idx, R = list(range(6)), np.array([pos_std ** 2] * 3 + [ang_std ** 2] * 3)
                kind, chi2 = "full", self.chi2_full
            else:
                # position only, through the ESTIMATED attitude
                if det.range_m > self.pos_only_max_range:
                    self.stats["pos_only_far"] += 1
                    continue
                x = self.ekf.x[:, 0]
                R_wb = rot_rpy(*x[3:6])
                R_wc = R_wb @ self.T_body_cam_cv[:3, :3]
                p_cam = self.gates[i].center - R_wc @ det.T_cam_gate[:3, 3]
                z = p_cam - R_wb @ self.T_body_cam_cv[:3, 3]
                # the attitude estimate's own error, lifted through the range, adds to the PnP error
                att_std = max(self.min_ang_std, float(np.sqrt(np.max(np.diag(self.ekf.P)[3:6]))))
                lift = det.range_m * att_std
                idx, R = [0, 1, 2], np.array([pos_std ** 2 + lift ** 2] * 3)
                kind, chi2 = "pos", self.chi2_pos
            # chi-square gate on the innovation
            H_x = self.ekf.x[idx, 0]
            y = z - H_x
            for r, s in enumerate(idx):
                if s in (3, 4, 5):
                    y[r] = wrap_angle(y[r])
            S = self.ekf.P[np.ix_(idx, idx)] + np.diag(R)
            d2 = float(y @ np.linalg.solve(S, y))
            if d2 > chi2:
                if kind == "full" and self._relock(t_img, i, z, y, det):
                    self.stats["relocks"] += 1
                    self.events.append((t_img, "relock", self.gates[i].name, d2))
                    self.ekf.update_measurement(z, idx, R)
                    self.stats["fixes_full"] += 1
                    applied += 1
                    continue
                self.stats["rejected"] += 1
                self.events.append((t_img, "reject", self.gates[i].name, d2))
                continue
            self._rejected.clear()
            self._last_fix_t = t_img if self._last_fix_t is None else max(self._last_fix_t, t_img)
            self.ekf.update_measurement(z, idx, R)
            self.stats["fixes_full" if kind == "full" else "fixes_pos"] += 1
            self.events.append((t_img, kind, self.gates[i].name, d2))
            applied += 1

        # replay forward to now
        if hist_keep is not None:
            self.hist = deque(hist_keep, maxlen=self.hist.maxlen)
            imu_keep = list(self.imu_hist)[:len(hist_keep)]
            self.imu_hist = deque(imu_keep, maxlen=self.imu_hist.maxlen)
            for (t, acc, g) in replay:
                self._propagate(acc, g)
                self.hist.append((t, self.ekf.x.copy(), self.ekf.P.copy()))
                self.imu_hist.append((t, acc, g))
        return applied

    def _relock(self, t_img, gate_idx, z, y, det):
        """A rejected full fix: re-lock (widen P to the innovation, return True) once relock_n recent
        rejected fixes of this gate agree with each other; otherwise remember it and return False."""
        if not self.relock_n or det.range_m > self.relock_max_range:
            return False
        if self._last_fix_t is not None and t_img - self._last_fix_t < self.relock_blind:
            return False                                  # fixes are flowing: this gate is the odd one out
        self._rejected.append((t_img, gate_idx, np.asarray(z, dtype=float).copy()))
        while self._rejected and (t_img - self._rejected[0][0] > self.relock_window
                                  or self._rejected[0][1] != gate_idx):
            self._rejected.popleft()
        if len(self._rejected) < self.relock_n:
            return False
        pts = np.array([r[2][0:3] for r in self._rejected])
        if np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)) > self.relock_spread:
            return False
        P = self.ekf.P
        for k in range(6):
            P[k, k] += float(y[k]) ** 2                   # position and angles: at least the innovation
        for k in range(6, 9):
            P[k, k] += 0.5 ** 2                           # velocity: unknown after drifting this far
        self._rejected.clear()
        return True

    # ------------------------------------------------------------------ output
    @property
    def state(self):
        """12x1 estimate at the latest IMU sample (None until initialized)."""
        return self.ekf.x.copy() if self.initialized else None

    def position_std(self):
        return np.sqrt(np.diag(self.ekf.P)[0:3]) if self.initialized else None
