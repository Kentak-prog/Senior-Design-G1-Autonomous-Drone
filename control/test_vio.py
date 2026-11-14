"""Self-tests for control/vio.py and the look-at yaw. Run: python3 test_vio.py

No ROS, no simulator. Detections are built from the TRUE geometry (exact
T_cam_gate), so each test isolates one piece of the VIO. The closed-loop
rehearsal with noisy PnP is Trajectory/simulate_vio_mpc.py.
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(REPO, "Trajectory"), os.path.join(REPO, "vision")]

from vio import GateVIO, camera_mount_usd, rot_rpy, rpy_from_rot, FRD_FROM_FLU, _T, _inv   # noqa: E402
from gate_pose import GateDetection                                                       # noqa: E402
from gate_path import Gate, gate_crossings                                                # noqa: E402
from eval_isaac_gates import T_world_gate_from_spec, T_body_cam_usd_from_mount           # noqa: E402

G = 9.81
GATES_3 = [(5.1644, 0.5666, 3.3667, 7.909), (8.6018, -3.2453, 3.6556, -22.091), (13.0455, -2.0747, 3.7681, -7.091)]
GATES = [Gate(f"gate_{i}", T_world_gate_from_spec(*g)) for i, g in enumerate(GATES_3)]
START = np.array([-0.04, 0.0, 2.56])
START_YAW = math.radians(-7.1)
MOUNT0 = camera_mount_usd(0.0)


def _det(T_world_body, gate, mount=MOUNT0, center_std=0.03, normal_std=2.0):
    """An exact detection of `gate` from a drone at T_world_body (OpenCV camera frame, like PnP)."""
    T_cv = mount.copy()
    T_cv[:3, :3] = mount[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    T_cam_gate = _inv(T_world_body @ T_cv) @ gate.T_world_gate
    return GateDetection(T_cam_gate=T_cam_gate, reprojection_error_px=0.1,
                         range_m=float(np.linalg.norm(T_cam_gate[:3, 3])), corners_px=np.zeros((4, 2)),
                         center_std_m=center_std, normal_std_deg=normal_std)


def _hover_imu(vio, n, t0=0.0, bias_flu=(0.0, 0.0, 0.0)):
    for k in range(n):
        vio.add_imu(t0 + (k + 1) * vio.dt, FRD_FROM_FLU @ np.array([0.0, 0.0, G]),
                    FRD_FROM_FLU @ np.asarray(bias_flu))
    return t0 + n * vio.dt


def test_mount_matches_spawn_example():
    assert np.allclose(camera_mount_usd(15.0), T_body_cam_usd_from_mount())
    print("  camera_mount_usd(15) == the mount the vision eval verified  OK")


def test_calibration_removes_gyro_bias_and_checks_gravity():
    vio = GateVIO(GATES, MOUNT0, calib_s=1.0)
    bias = np.radians([0.3, -0.4, 0.5])
    _hover_imu(vio, 250, bias_flu=bias)
    c = vio.calibration_summary
    assert vio.calibrated and np.allclose(np.radians(c["gyro_bias_deg_s"]), bias, atol=1e-9)
    assert abs(c["accel_norm"] - G) < 1e-9 and c["accel_mean"][2] > 0      # FRD z=-9.81 read as FLU +9.81
    print("  1 s of hover: gyro bias exact, accelerometer +9.81 up after FRD -> FLU  OK")


def test_pose_from_exact_detection_and_from_behind():
    vio = GateVIO(GATES, MOUNT0, calib_s=0.0)
    T_wb = _T(rot_rpy(0.05, -0.08, 0.2), np.array([1.0, -0.5, 2.9]))
    for i, g in enumerate(GATES):
        det = _det(T_wb, g)
        Te = _inv(T_wb @ vio.T_body_cam_cv) @ g.T_world_gate
        T = vio._body_pose_from(det, i, Te)
        assert np.allclose(T, T_wb, atol=1e-9)
        # the same gate seen from behind: PnP's normal points the other way
        flipped = det.T_cam_gate.copy()
        flipped[:3, 0] *= -1
        flipped[:3, 2] *= -1
        det.T_cam_gate = flipped
        assert np.allclose(vio._body_pose_from(det, i, Te), T_wb, atol=1e-9)
    print("  exact detection -> exact body pose (also with the normal flipped, gate seen from behind)  OK")


def test_initializes_from_hint_and_holds_with_fixes():
    vio = GateVIO(GATES, MOUNT0, calib_s=0.5, initial_guess=(START + [0.3, -0.2, 0.1], START_YAW + 0.05))
    T_wb = _T(rot_rpy(0.0, 0.0, START_YAW), START)
    t = _hover_imu(vio, 125)
    assert vio.calibrated and not vio.initialized
    for _ in range(20):                                  # 2 s of hover, a frame every 0.1 s
        vio.add_detections(t, [_det(T_wb, g) for g in GATES])
        t = _hover_imu(vio, 25, t0=t)
    x = vio.state[:, 0]
    assert vio.initialized and np.linalg.norm(x[0:3] - START) < 0.01 and abs(x[5] - START_YAW) < 1e-3, x[0:6]
    print(f"  hint 0.4 m off -> initialized from the gates, holds {np.linalg.norm(x[0:3] - START) * 100:.2f} cm "
          f"from the true hover pose  OK")


def test_rejects_an_outlier():
    vio = GateVIO(GATES, MOUNT0, calib_s=0.0, initial_guess=(START, START_YAW))
    T_wb = _T(rot_rpy(0.0, 0.0, START_YAW), START)
    t = _hover_imu(vio, 1)
    vio.add_detections(t, [_det(T_wb, GATES[0])])
    t = _hover_imu(vio, 25, t0=t)
    for _ in range(5):
        vio.add_detections(t, [_det(T_wb, g) for g in GATES])
        t = _hover_imu(vio, 25, t0=t)
    before = vio.state[:, 0].copy()
    bad = _det(_T(rot_rpy(0.0, 0.0, START_YAW), START + [0.0, 0.0, 0.8]), GATES[0])   # 0.8 m wrong
    vio.add_detections(t, [bad])
    assert vio.stats["rejected"] >= 1 and np.linalg.norm(vio.state[0:3, 0] - before[0:3]) < 0.02
    print("  a detection implying the drone is 0.8 m off is rejected by the chi-square gate  OK")


def _moving_run(delay_compensation, latency=0.15):
    """Constant 2 m/s along +x; images captured every 0.1 s, delivered `latency` late."""
    vio = GateVIO(GATES, MOUNT0, calib_s=0.0, initial_guess=(START, START_YAW),
                  delay_compensation=delay_compensation)
    v = np.array([2.0, 0.0, 0.0])
    pose = lambda t: _T(rot_rpy(0.0, 0.0, 0.0), START + v * max(t - 0.5, 0.0))   # noqa: E731
    pending, errs, t = [], [], 0.0
    for k in range(int(2.5 / vio.dt)):
        t = (k + 1) * vio.dt
        acc = [0.0, 0.0, G]
        if abs(t - 0.5) < vio.dt / 2:                  # the 2 m/s kick, as one IMU sample of acceleration
            acc = [2.0 / vio.dt, 0.0, G]
        vio.add_imu(t, FRD_FROM_FLU @ np.array(acc), np.zeros(3))
        if k % 25 == 0:
            pending.append((t + latency, t, [_det(pose(t), g) for g in GATES[:1]]))
        while pending and pending[0][0] <= t:
            _, tc, d = pending.pop(0)
            vio.add_detections(tc, d)
        if vio.initialized and t > 1.0:
            errs.append(np.linalg.norm(vio.state[0:3, 0] - pose(t)[:3, 3]))
    return float(np.mean(errs))


def test_delay_compensation():
    with_c, without = _moving_run(True), _moving_run(False)
    assert with_c < 0.05 and without > 2 * with_c, (with_c, without)
    print(f"  2 m/s with 150 ms camera latency: error {with_c * 100:.1f} cm with replay vs "
          f"{without * 100:.1f} cm applying fixes late  OK")


def test_look_at_yaw_keeps_gates_ahead_and_still_passes():
    import min_snap_planner as msn
    traj, wps, ref = msn.plan_course(GATES, START, START_YAW, yaw_mode="next_gate")
    _, _, ref_v = msn.plan_course(GATES, START, START_YAW, yaw_mode="velocity")
    ts = ref.times
    assert abs(ref.states[5, 0] - START_YAW) < 1e-6
    res = gate_crossings(ts, ref.states[0:3].T, ref.states[6:9].T, GATES)
    assert all(any(c["inside"] and c["forward"] for c in cs) for _, cs in res)

    def off_axis(r):
        """Worst angle between the nose and the next gate, while that gate is more than 3 m away."""
        t_c = [traj.t_breaks[i] for i, w in enumerate(wps) if w["type"] == "center"]
        worst = 0.0
        for k, t in enumerate(r.times):
            j = np.searchsorted(t_c, t, side="right")
            if j >= len(GATES):
                break
            d = GATES[j].center - r.states[0:3, k]
            if np.linalg.norm(d[:2]) < 3.0:
                continue
            ang = abs((math.atan2(d[1], d[0]) - r.states[5, k] + math.pi) % (2 * math.pi) - math.pi)
            worst = max(worst, math.degrees(ang))
        return worst
    a, b = off_axis(ref), off_axis(ref_v)
    # bound = the camera's half horizontal FOV (60 deg); the worst case is just after a target
    # switch, while the smoothing turns the nose (2026-10-28: 26 deg vs 61 with velocity yaw)
    assert a < 30.0 and b > 2 * a, (a, b)
    print(f"  next-gate yaw: nose within {a:.0f} deg of the next gate (velocity yaw: up to {b:.0f} deg), "
          f"all gates still passed  OK")


def _settled_vio(seconds=5.0, **kw):
    """A VIO hovering at START after `seconds` of good fixes from every gate at 60 Hz, as in Isaac
    before the gap (hundreds of fixes: the filter is then very sure of its tilt)."""
    vio = GateVIO(GATES, MOUNT0, calib_s=0.0, initial_guess=(START, START_YAW), **kw)
    T_wb = _T(rot_rpy(0.0, 0.0, START_YAW), START)
    t = _hover_imu(vio, 1)
    for _ in range(int(seconds * 60)):
        vio.add_detections(t, [_det(T_wb, g) for g in GATES])
        t = _hover_imu(vio, 4, t0=t)
    return vio, T_wb, t


def test_dead_reckoning_uncertainty_is_honest():
    # Isaac 2026-11-06: 4.8 s without a usable gate; at the start 0.11 m/s velocity error, then about
    # 0.04 m/s^2 of unmodelled acceleration (tilt) -> 1.06 m, while the filter claimed 0.30 m.
    ratio = {}
    for rw in (0.0, 0.1):
        vio, T_wb, t = _settled_vio(accel_rw=rw)
        vio.ekf.x[6, 0] += 0.11                                   # velocity error when the fixes stop
        for k in range(int(4.8 / vio.dt)):                        # no fixes; the IMU reads 0.04 m/s^2 too much
            t += vio.dt
            vio.add_imu(t, FRD_FROM_FLU @ np.array([0.04, 0.0, G]), np.zeros(3))
        err = float(np.linalg.norm(vio.state[0:3, 0] - START))
        ratio[rw] = err / float(np.linalg.norm(vio.position_std()))
    assert ratio[0.0] > 3.0 and ratio[0.1] < 1.5, ratio
    print(f"  4.8 s blind with Isaac's drift: error / reported std {ratio[0.0]:.1f} (white noise only) -> "
          f"{ratio[0.1]:.2f} with accel_rw 0.1  OK")


def test_relock_after_drift():
    def drifted(relock_n, spread=0.0):
        vio, T_wb, t = _settled_vio(relock_n=relock_n)
        vio.ekf.x[0, 0] += 1.0                                    # 1 m off, and sure of itself
        vio.ekf.P[0:3, 0:3] = np.eye(3) * 0.03 ** 2
        for k in range(20):                                       # 15 Hz of gate_0 at about 5 m (Isaac: 3.5-4.8 m)
            jitter = np.array([0.0, spread * (1 if k % 2 else -1), 0.0])
            vio.add_detections(t, [_det(_T(rot_rpy(0.0, 0.0, START_YAW), START + jitter), GATES[0])])
            t = _hover_imu(vio, 17, t0=t)
        return vio, float(np.linalg.norm(vio.state[0:3, 0] - START))
    vio, err = drifted(5)
    assert vio.stats["relocks"] == 1 and err < 0.1, (vio.stats, err)
    vio0, err0 = drifted(0)
    assert vio0.stats["relocks"] == 0 and err0 > 0.9, (vio0.stats, err0)
    vio_j, err_j = drifted(5, spread=0.5)                         # fixes that disagree with each other
    assert vio_j.stats["relocks"] == 0, vio_j.stats
    print(f"  1 m off and overconfident: 5 consistent rejected fixes -> re-lock, error {err:.3f} m "
          f"(without: {err0:.2f} m); disagreeing fixes (+-0.5 m) never re-lock  OK")



def test_no_relock_while_fixes_flow():
    # Isaac 2026-11-06, attempt 3: just after engage gate_0 (half clipped) gave consistent but biased
    # fixes while gate_1 / gate_2 kept fixing the estimate. Re-locking to gate_0 then was wrong.
    vio, T_wb, t = _settled_vio(seconds=1.0)
    biased = _T(rot_rpy(0.0, 0.0, START_YAW), START + np.array([0.0, 0.0, 0.6]))
    for _ in range(20):
        vio.add_detections(t, [_det(biased, GATES[0]), _det(T_wb, GATES[1]), _det(T_wb, GATES[2])])
        t = _hover_imu(vio, 4, t0=t)
    err = float(np.linalg.norm(vio.state[0:3, 0] - START))
    assert vio.stats["relocks"] == 0 and vio.stats["rejected"] >= 15 and err < 0.1, (vio.stats, err)
    print(f"  one gate consistently 0.6 m off while the others keep fixing: rejected, no re-lock "
          f"(error {err:.3f} m)  OK")


def test_far_position_only_fixes_are_dropped():
    # Isaac 2026-11-06: position-only fixes of gate_2 at 9-11 m (normal unreliable) pulled the estimate
    # 0.64 m sideways. A biased far one must be ignored; a near one is still used, and weighted by
    # range x attitude uncertainty.
    def run(gate, offset):
        vio, T_wb, t = _settled_vio(seconds=1.0)
        before = vio.state[0:3, 0].copy()
        wrong = _T(rot_rpy(0.0, 0.0, START_YAW), START + offset)       # what a biased fix implies
        for _ in range(30):
            vio.add_detections(t, [_det(wrong, gate, normal_std=40.0)])   # normal unreliable -> position-only
            t = _hover_imu(vio, 4, t0=t)
        return vio, float(np.linalg.norm(vio.state[0:3, 0] - before))
    far, moved_far = run(GATES[2], np.array([0.0, 0.6, 0.0]))         # gate_2, about 13 m away
    assert far.stats["pos_only_far"] == 30 and far.stats["fixes_pos"] == 0 and moved_far < 0.05, (far.stats, moved_far)
    near, moved_near = run(GATES[0], np.array([0.0, 0.3, 0.0]))       # gate_0, about 5 m away
    assert near.stats["fixes_pos"] > 0 and moved_near > 0.1, (near.stats, moved_near)
    print(f"  position-only fixes: gate_2 at 13 m dropped (estimate moved {moved_far:.3f} m); "
          f"gate_0 at 5 m still used (moved {moved_near:.2f} m toward it)  OK")


def _tilted_hover_run(swift, pnp_pitch_err_deg=3.0, blind_s=3.0):
    """Hover at a slight true tilt. The detections come from a pose whose PITCH is pnp_pitch_err_deg
    off (a biased PnP normal, as Isaac 2026-11-13). Init, a few fixes, then blind_s with no gates.
    Returns (vio, true roll/pitch, position drift over the blind stretch)."""
    true_rp = (0.02, -0.03)
    R_true = rot_rpy(true_rp[0], true_rp[1], START_YAW)
    T_true = _T(R_true, START)
    T_bad = _T(rot_rpy(true_rp[0], true_rp[1] + math.radians(pnp_pitch_err_deg), START_YAW), START)
    f_body = R_true.T @ np.array([0.0, 0.0, G])                        # specific force at a tilted hover
    kw = dict(attitude_updates="yaw", tilt_init="accel", init_fixes=3) if swift else {}
    vio = GateVIO(GATES, MOUNT0, calib_s=0.5, initial_guess=(START, START_YAW), **kw)
    t = 0.0

    def imu(n):
        nonlocal t
        for _ in range(n):
            t += vio.dt
            vio.add_imu(t, FRD_FROM_FLU @ f_body, np.zeros(3))

    imu(125)
    for _ in range(10):
        vio.add_detections(t, [_det(T_bad, GATES[0])])
        imu(25)
    p0 = vio.state[0:3, 0].copy()
    imu(int(blind_s / vio.dt))
    return vio, true_rp, float(np.linalg.norm(vio.state[0:3, 0] - p0)), T_true


def test_swift_tilt_from_gravity_not_pnp():
    # Isaac 2026-11-13: the first PnP fix left the pitch ~3 deg off; nothing corrected it and
    # g sin 3 deg leaked into the velocity. Swift mode takes roll/pitch from the accelerometer.
    old, rp, drift_old, _ = _tilted_hover_run(swift=False)
    new, rp, drift_new, _ = _tilted_hover_run(swift=True)
    err_old = math.degrees(abs(old.state[4, 0] - rp[1]))
    err_new = math.degrees(abs(new.state[4, 0] - rp[1]))
    assert err_old > 2.0, err_old                                       # the failure, reproduced
    assert err_new < 0.2 and math.degrees(abs(new.state[3, 0] - rp[0])) < 0.2, (err_new, new.state[3:6, 0])
    assert drift_old > 1.0 and drift_new < 0.1, (drift_old, drift_new)
    print(f"  PnP pitch 3 deg off: old init pitch error {err_old:.1f} deg -> {drift_old:.2f} m drift in 3 s blind; "
          f"Swift mode {err_new:.2f} deg -> {drift_new:.3f} m  OK")


def test_swift_holds_hover_with_fixes():
    vio = GateVIO(GATES, MOUNT0, calib_s=0.5, initial_guess=(START + [0.3, -0.2, 0.1], START_YAW + 0.05),
                  attitude_updates="yaw", tilt_init="accel", init_fixes=3)
    T_wb = _T(rot_rpy(0.0, 0.0, START_YAW), START)
    t = _hover_imu(vio, 125)
    for _ in range(20):
        vio.add_detections(t, [_det(T_wb, g) for g in GATES])
        t = _hover_imu(vio, 25, t0=t)
    x = vio.state[:, 0]
    assert vio.initialized and np.linalg.norm(x[0:3] - START) < 0.01 and abs(x[5] - START_YAW) < 1e-3, x[0:6]
    assert vio.stats["fixes_full"] > 10 and vio.stats["rejected"] == 0, vio.stats
    print(f"  Swift mode (position + yaw from gates, tilt from gravity): holds "
          f"{np.linalg.norm(x[0:3] - START) * 100:.2f} cm from the hover pose  OK")


def test_calibration_waits_until_still():
    # Isaac 2026-11-14: calibrated while still turning after the move to the start (5 deg/s yaw,
    # 0.7 deg/s pitch taken as bias). calib_still waits for a still window instead.
    bias = np.radians([0.3, -0.4, 0.5])
    turn = np.radians([0.0, 0.7, 5.0])
    for still, expect_bias in [(False, bias + turn), (True, bias)]:
        vio = GateVIO(GATES, MOUNT0, calib_s=1.0, calib_still=still)
        rng = np.random.default_rng(0)
        for k in range(250 + 250 + 400):                 # 1 s turning, 1 s settling to 0, then still
            w = turn if k < 250 else (turn * (1 - (k - 250) / 250) if k < 500 else np.zeros(3))
            vio.add_imu((k + 1) * vio.dt, FRD_FROM_FLU @ np.array([0.0, 0.0, G]),
                        FRD_FROM_FLU @ (bias + w + rng.normal(0, 0.0054, 3)))
            if vio.calibrated:
                break
        assert vio.calibrated, still
        err = np.degrees(np.abs(vio.gyro_bias - expect_bias)).max()
        assert err < 0.05 if still else err < 0.1, (still, np.degrees(vio.gyro_bias), err)
        if still:
            assert vio.stats["calib_waits"] > 0
            print(f"  calib_still: waited {vio.stats['calib_waits']} windows through the turn, bias error {err:.3f} deg/s  OK")


def test_slow_tilt_correction_bounds_gyro_bias():
    # Swift mode without gate tilt updates lets a residual gyro bias integrate; tilt_tau_s bounds it.
    def run(tau):
        vio = GateVIO(GATES, MOUNT0, calib_s=0.0, initial_guess=(START, START_YAW),
                      attitude_updates="yaw", tilt_init="accel", tilt_tau_s=tau)
        T_wb = _T(rot_rpy(0.0, 0.0, START_YAW), START)
        resid = np.radians([0.0, 0.3, 0.0])              # 0.3 deg/s pitch bias the calibration missed
        t = 0.0
        for k in range(int(20.0 / vio.dt)):
            t += vio.dt
            vio.add_imu(t, FRD_FROM_FLU @ np.array([0.0, 0.0, G]), FRD_FROM_FLU @ resid)
            if k % 8 == 0:
                vio.add_detections(t, [_det(T_wb, GATES[0])])
        return math.degrees(abs(vio.state[4, 0]))
    off, on = run(None), run(2.0)
    assert off > 2.0 and on < 0.8 and on < off / 4, (off, on)      # bounded near bias x tau = 0.6 deg
    print(f"  0.3 deg/s residual pitch bias for 20 s: tilt error {off:.1f} deg without correction, "
          f"{on:.2f} deg with tilt_tau_s=2  OK")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        t()
    print(f"\nAll tests passed. ({len(tests)})")
