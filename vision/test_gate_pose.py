"""Self-tests. Run: python test_gate_pose.py"""

import numpy as np

from camera import (CameraIntrinsics, make_transform, invert_transform,
                    usd_camera_pose_to_cv)
from gate_pose import (estimate_gate_pose, project_gate, order_corners,
                       gate_pose_world, approach_waypoints)

INTR = CameraIntrinsics.from_horizontal_fov(1280, 720, 90.0)
SIDE = 1.5


def test_zero_noise_roundtrip():
    """Perfect corners must recover the pose to solver precision."""
    R = np.diag([1.0, -1.0, -1.0])
    T_true = make_transform(R, [0.4, -0.2, 5.0])
    corners = project_gate(T_true, INTR, SIDE)
    det = estimate_gate_pose(corners, INTR, SIDE, assume_ordered=True)
    err = np.linalg.norm(det.T_cam_gate[:3, 3] - T_true[:3, 3])
    assert err < 1e-6, err
    assert det.reprojection_error_px < 1e-6
    print(f"  zero-noise position error: {err:.2e} m  OK")


def test_corner_ordering():
    """order_corners must recover TL,TR,BR,BL from a shuffled quad."""
    R = np.diag([1.0, -1.0, -1.0])
    T_true = make_transform(R, [0.0, 0.0, 4.0])
    corners = project_gate(T_true, INTR, SIDE)
    rng = np.random.default_rng(3)
    for _ in range(50):
        shuffled = corners[rng.permutation(4)]
        assert np.allclose(order_corners(shuffled), corners, atol=1e-6)
    print("  corner ordering survives 50 random permutations  OK")


def test_world_transform_and_waypoints():
    """Gate at a known world spot, camera elsewhere, must come back correct."""
    gate_world = np.array([8.0, 2.0, 2.5])
    # Gate normal (+Z of the gate frame) points along world -X, i.e. back at
    # the drone; gate "up" (+Y) is world +Z.
    R_world_gate = np.array([[0.0, 0.0, -1.0],
                             [-1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0]])
    T_world_gate_true = make_transform(R_world_gate, gate_world)

    # Camera at (0, 0, 2.5) in USD convention, looking along world +X,
    # up = world +Z. USD cameras look down their own -Z.
    R_world_usd = np.array([[0.0, 0.0, -1.0],
                            [-1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0]])
    T_world_cam = usd_camera_pose_to_cv(np.array([0.0, 0.0, 2.5]),
                                        rot_to_quat(R_world_usd))

    T_cam_gate = invert_transform(T_world_cam) @ T_world_gate_true
    assert T_cam_gate[2, 3] > 0, "gate ended up behind the camera"

    corners = project_gate(T_cam_gate, INTR, SIDE)
    det = estimate_gate_pose(corners, INTR, SIDE, assume_ordered=True)
    T_est = gate_pose_world(det, T_world_cam)

    err = np.linalg.norm(T_est[:3, 3] - gate_world)
    assert err < 1e-5, err

    wps = approach_waypoints(T_est, standoff=1.0)
    assert np.allclose(wps[1], gate_world, atol=1e-5)
    spacing = np.linalg.norm(wps[2] - wps[0])
    assert abs(spacing - 2.0) < 1e-6
    print(f"  world-frame position error: {err:.2e} m, "
          f"waypoint spacing {spacing:.3f} m  OK")


def rot_to_quat(R):
    """Rotation matrix -> quaternion (w, x, y, z), Shepperd's method."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S
    return np.array([w, x, y, z])


def test_bootstrap_uncertainty():
    """Predicted 1-sigma must track actual error, not just exist."""
    rng = np.random.default_rng(0)
    sigma = 1.0
    for rng_m in (4.0, 10.0):
        R = np.diag([1.0, -1.0, -1.0])
        T_true = make_transform(R, [0.0, 0.0, rng_m])
        clean = project_gate(T_true, INTR, SIDE)
        actual, predicted = [], []
        for _ in range(60):
            noisy = clean + rng.normal(0.0, sigma, clean.shape)
            det = estimate_gate_pose(noisy, INTR, SIDE, assume_ordered=True,
                                     corner_sigma_px=sigma, bootstrap_n=12,
                                     seed=int(rng.integers(1 << 30)))
            actual.append(np.linalg.norm(det.T_cam_gate[:3, 3] - T_true[:3, 3]))
            predicted.append(det.center_std_m)
        a, p = np.mean(actual), np.mean(predicted)
        ratio = p / a
        assert 0.5 < ratio < 2.0, (rng_m, a, p)
        print(f"  {rng_m:4.1f} m: actual {a:.3f} m, predicted {p:.3f} m "
              f"(ratio {ratio:.2f})  OK")


def test_scale_sensitivity():
    """A wrong gate-size assumption shows up as a proportional range error."""
    R = np.diag([1.0, -1.0, -1.0])
    T_true = make_transform(R, [0.0, 0.0, 6.0])
    corners = project_gate(T_true, INTR, SIDE)
    det = estimate_gate_pose(corners, INTR, SIDE * 1.05, assume_ordered=True)
    ratio = det.range_m / 6.0
    assert abs(ratio - 1.05) < 1e-3, ratio
    print(f"  5% gate-size error -> {(ratio - 1) * 100:.1f}% range error  OK")


if __name__ == "__main__":
    print("test_zero_noise_roundtrip");      test_zero_noise_roundtrip()
    print("test_corner_ordering");           test_corner_ordering()
    print("test_world_transform_and_waypoints"); test_world_transform_and_waypoints()
    print("test_bootstrap_uncertainty");     test_bootstrap_uncertainty()
    print("test_scale_sensitivity");         test_scale_sensitivity()
    print("\nAll tests passed.")
