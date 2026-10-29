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


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        t()
    print(f"\nAll tests passed. ({len(tests)})")
