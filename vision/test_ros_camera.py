"""Self-tests. Run: python3 test_ros_camera.py

Deliberately imports nothing rclpy/cv_bridge-related at module scope, and
never instantiates GatePoseNode -- this whole file must pass on a machine
with no ROS installed at all (see ros_camera.py's module docstring).
"""

import numpy as np

from camera import CameraIntrinsics, make_transform, quat_to_rot
from gate_pose import estimate_gate_pose, gate_pose_world, project_gate
from ros_camera import (intrinsics_from_camera_info, quat_xyzw_to_wxyz,
                        transform_to_matrix, T_world_cam_from_transform,
                        verify_optical_convention)

INTR = CameraIntrinsics.from_horizontal_fov(1280, 720, 90.0)
SIDE = 1.5


def _camera_info_dict(intr: CameraIntrinsics, upper: bool) -> dict:
    k = [intr.fx, 0.0, intr.cx, 0.0, intr.fy, intr.cy, 0.0, 0.0, 1.0]
    d = list(np.asarray(intr.distortion, dtype=float).ravel())
    p = [intr.fx, 0.0, intr.cx, 0.0, 0.0, intr.fy, intr.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    if upper:
        return {"K": k, "D": d, "P": p, "width": intr.width, "height": intr.height}
    return {"k": k, "d": d, "p": p, "width": intr.width, "height": intr.height}


def test_camera_info_roundtrip():
    """intrinsics_from_camera_info must recover fx/fy/cx/cy/width/height,
    for lowercase fields, uppercase fields, and the prefer_projection path."""
    for upper in (False, True):
        msg = _camera_info_dict(INTR, upper)
        got = intrinsics_from_camera_info(msg, prefer_projection=False)
        assert abs(got.fx - INTR.fx) < 1e-9, (upper, got.fx, INTR.fx)
        assert abs(got.fy - INTR.fy) < 1e-9
        assert abs(got.cx - INTR.cx) < 1e-9
        assert abs(got.cy - INTR.cy) < 1e-9
        assert got.width == INTR.width
        assert got.height == INTR.height

        got_p = intrinsics_from_camera_info(msg, prefer_projection=True)
        assert abs(got_p.fx - INTR.fx) < 1e-9
        assert abs(got_p.fy - INTR.fy) < 1e-9
        assert abs(got_p.cx - INTR.cx) < 1e-9
        assert abs(got_p.cy - INTR.cy) < 1e-9

        spelling = "UPPERCASE K/D/P" if upper else "lowercase k/d/p"
        print(f"  {spelling}: K-path and P-path both recover fx={got.fx:.2f} "
              f"fy={got.fy:.2f} cx={got.cx:.2f} cy={got.cy:.2f}  OK")

    # Distortion padding/truncation.
    short = dict(_camera_info_dict(INTR, False))
    short["d"] = [0.1, 0.2]
    got = intrinsics_from_camera_info(short)
    assert got.distortion.shape == (5,)
    assert np.allclose(got.distortion, [0.1, 0.2, 0.0, 0.0, 0.0])

    long = dict(_camera_info_dict(INTR, False))
    long["d"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    got = intrinsics_from_camera_info(long)
    assert got.distortion.shape == (5,)
    assert np.allclose(got.distortion, [0.1, 0.2, 0.3, 0.4, 0.5])
    print("  distortion padded/truncated to exactly 5 elements  OK")


class _FakeQuat:
    """Stand-in for geometry_msgs/Quaternion: attribute access, xyzw order."""
    def __init__(self, x, y, z, w):
        self.x, self.y, self.z, self.w = x, y, z, w


class _FakeVec3:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


def test_quaternion_ordering():
    """
    A rotation NOT symmetric under swapping xyzw<->wxyz must round-trip
    correctly through quat_xyzw_to_wxyz + transform_to_matrix, and feeding
    the SAME numbers in the wrong order must produce a materially different
    rotation -- otherwise this test could not actually catch the bug the
    ordering mismatch causes.
    """
    # 90 deg about an axis with three distinct-magnitude components, so the
    # quaternion (w, x, y, z) has four distinct nonzero values, and swapping
    # w with a component changes the rotation substantially rather than by
    # symmetry.
    axis = np.array([1.0, 2.0, 3.0])
    axis = axis / np.linalg.norm(axis)
    angle = np.radians(73.0)
    w = np.cos(angle / 2)
    x, y, z = axis * np.sin(angle / 2)
    assert len({round(v, 6) for v in (w, x, y, z)}) == 4, "need 4 distinct components"

    R_true = quat_to_rot(np.array([w, x, y, z]))

    t = _FakeVec3(1.0, -2.0, 3.5)
    q_ros_correct = _FakeQuat(x, y, z, w)  # correct ROS ordering: x,y,z,w
    T = transform_to_matrix(t, q_ros_correct)
    assert np.allclose(T[:3, :3], R_true, atol=1e-9)
    assert np.allclose(T[:3, 3], [1.0, -2.0, 3.5])
    print("  correct xyzw->wxyz conversion recovers the true rotation  OK")

    # Now deliberately feed the wrong order: pretend the ROS (x, y, z, w)
    # values were handed straight to quat_to_rot without going through
    # quat_xyzw_to_wxyz first -- i.e. (x, y, z, w) misread as (w, x, y, z).
    # This is exactly the bug quat_xyzw_to_wxyz exists to prevent.
    R_wrong = quat_to_rot(np.asarray([x, y, z, w]))
    diff = np.linalg.norm(R_wrong - R_true)
    assert diff > 0.5, f"wrong-order rotation too close to correct one: diff={diff}"
    print(f"  swapped-order rotation differs from the true one by {diff:.2f}  "
          f"(bug would be caught)  OK")


def test_full_pipeline_no_ros():
    """
    Gate at a known world pose, camera elsewhere, TF-like transform ->
    T_world_cam_from_transform -> project -> estimate_gate_pose ->
    gate_pose_world must recover the world position to < 1e-6, for BOTH
    optical_frame settings against the matching ground-truth transform.
    """
    gate_world = np.array([6.0, -1.0, 3.0])
    R_world_gate = np.array([[0.0, 0.0, 1.0],
                             [-1.0, 0.0, 0.0],
                             [0.0, -1.0, 0.0]])
    T_world_gate_true = make_transform(R_world_gate, gate_world)

    cam_pos = np.array([0.0, 0.0, 3.0])

    # --- optical_frame=True: TF already gives OpenCV/optical convention
    # directly (+X right, +Y down, +Z forward = world +X here). ---
    R_world_cv = np.array([[0.0, 0.0, 1.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0]])
    quat_xyzw = _rot_to_quat_xyzw_for_test(R_world_cv)
    transform = _FakeTransform(cam_pos, quat_xyzw)
    T_world_cam = T_world_cam_from_transform(transform, optical_frame=True)
    assert np.allclose(T_world_cam[:3, :3], R_world_cv, atol=1e-9)

    T_cam_gate = _to_cam(T_world_cam, T_world_gate_true)
    assert T_cam_gate[2, 3] > 0, "gate ended up behind the camera (optical_frame=True)"
    corners = project_gate(T_cam_gate, INTR, SIDE)
    det = estimate_gate_pose(corners, INTR, SIDE, assume_ordered=True)
    T_est = gate_pose_world(det, T_world_cam)
    err_true = np.linalg.norm(T_est[:3, 3] - gate_world)
    assert err_true < 1e-6, err_true
    print(f"  optical_frame=True:  world position error {err_true:.2e} m  OK")

    # --- optical_frame=False: TF gives the USD/body convention. Derive the
    # USD-frame ground truth algebraically from R_world_cv rather than by
    # hand: R_world_cv = R_world_usd @ R_CV_FROM_USD.T, and R_CV_FROM_USD is
    # its own inverse (diag(1,-1,-1)), so R_world_usd = R_world_cv @ R_CV_FROM_USD.
    from camera import R_CV_FROM_USD
    R_world_usd = R_world_cv @ R_CV_FROM_USD
    # Sanity: applying the same correction back must reproduce R_world_cv.
    assert np.allclose(R_world_usd @ R_CV_FROM_USD.T, R_world_cv, atol=1e-9), \
        "test setup bug: USD ground truth does not correspond to the CV one"

    quat_xyzw_usd = _rot_to_quat_xyzw_for_test(R_world_usd)
    transform_usd = _FakeTransform(cam_pos, quat_xyzw_usd)
    T_world_cam_f = T_world_cam_from_transform(transform_usd, optical_frame=False)
    assert np.allclose(T_world_cam_f[:3, :3], R_world_cv, atol=1e-9), \
        "optical_frame=False did not correct USD->CV as expected"

    T_cam_gate_f = _to_cam(T_world_cam_f, T_world_gate_true)
    assert T_cam_gate_f[2, 3] > 0, "gate ended up behind the camera (optical_frame=False)"
    corners_f = project_gate(T_cam_gate_f, INTR, SIDE)
    det_f = estimate_gate_pose(corners_f, INTR, SIDE, assume_ordered=True)
    T_est_f = gate_pose_world(det_f, T_world_cam_f)
    err_false = np.linalg.norm(T_est_f[:3, 3] - gate_world)
    assert err_false < 1e-6, err_false
    print(f"  optical_frame=False: world position error {err_false:.2e} m  OK")


def test_verify_optical_convention():
    """verify_optical_convention reports a sane, in-front, in-image pixel
    for a gate placed directly ahead of the camera."""
    cam_pos = np.array([0.0, 0.0, 3.0])
    R_world_cv = np.array([[0.0, 0.0, 1.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0]])
    T_world_cam = make_transform(R_world_cv, cam_pos)
    gate_world = np.array([6.0, -1.0, 3.0])

    result = verify_optical_convention(T_world_cam, INTR, gate_world)
    assert result["in_front_of_camera"] is True
    assert result["in_image"] is True
    assert np.all(np.isfinite(result["pixel"]))
    print(f"  verify_optical_convention: pixel {result['pixel']}, "
          f"in_front={result['in_front_of_camera']}, in_image={result['in_image']}  OK")

    # A point behind the camera must be flagged as such.
    behind = np.array([-6.0, -1.0, 3.0])
    result_behind = verify_optical_convention(T_world_cam, INTR, behind)
    assert result_behind["in_front_of_camera"] is False
    print("  point behind the camera correctly flagged  OK")


# --- small local helpers for building TF-like fixtures, kept out of
# ros_camera.py itself since they exist only to construct test inputs. ---

def _rot_to_quat_xyzw_for_test(R):
    from ros_camera import _rot_to_quat_xyzw
    return _rot_to_quat_xyzw(R)


class _FakeTransform:
    """Stand-in for geometry_msgs/Transform: .translation, .rotation."""
    def __init__(self, translation, quat_xyzw):
        self.translation = _FakeVec3(*translation)
        self.rotation = _FakeQuat(*quat_xyzw)


def _to_cam(T_world_cam, T_world_gate):
    from camera import invert_transform
    return invert_transform(T_world_cam) @ T_world_gate


def test_imports_cleanly_without_rclpy():
    """
    Module-level guard: ros_camera must already be imported (it is, at the
    top of this file) without rclpy/cv_bridge present, and none of the
    functions exercised above may have needed them. This test exists so a
    future edit that moves an rclpy import to module scope fails loudly.
    """
    import sys
    assert "rclpy" not in sys.modules, "rclpy got imported somewhere it shouldn't have"
    assert "cv_bridge" not in sys.modules, "cv_bridge got imported somewhere it shouldn't have"
    import ros_camera
    assert hasattr(ros_camera, "GatePoseNode"), "GatePoseNode class must still be defined"
    print("  ros_camera imported and used with no rclpy/cv_bridge present  OK")


if __name__ == "__main__":
    print("test_camera_info_roundtrip");         test_camera_info_roundtrip()
    print("test_quaternion_ordering");           test_quaternion_ordering()
    print("test_full_pipeline_no_ros");          test_full_pipeline_no_ros()
    print("test_verify_optical_convention");     test_verify_optical_convention()
    print("test_imports_cleanly_without_rclpy"); test_imports_cleanly_without_rclpy()
    print("\nAll tests passed.")
