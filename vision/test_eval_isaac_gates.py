"""Self-tests. Run: python3 test_eval_isaac_gates.py

Offline only -- numpy + cv2, no ROS, no sim. Exercises the ROS-free half of
eval_isaac_gates.py: the yaw-convention math, greedy matching, the
optical_frame decision rule, and (most importantly) a full offline
detect -> PnP -> lift -> decide pipeline against synthetic rendered frames,
proving the harness can actually tell optical_frame=True from
optical_frame=False rather than just agreeing with itself either way.
"""

import numpy as np

from camera import CameraIntrinsics, make_transform, invert_transform
from ros_camera import T_world_cam_from_transform
from gate_pose import DEFAULT_GATE_SIDE
from eval_isaac_gates import (T_world_gate_from_spec, match_detections,
                              decide_optical_frame, evaluate_frame,
                              rotate_xyz_deg_to_matrix, T_body_cam_usd_from_mount)
from test_color_gate_detector import render_synthetic_gate

# Higher resolution than test_color_gate_detector.py's 640x480 fixture: the
# end-to-end test below needs the synthetic renderer's pixel-grid rounding
# (inherent to compositing quads with cv2.fillPoly, not a detector defect)
# to stay a small fraction of a pixel relative to the ~9 m ranges involved,
# so the sub-0.2 m position-error assertions reflect the pipeline's real
# accuracy rather than rasterization noise.
INTR = CameraIntrinsics.from_horizontal_fov(1280, 960, 90.0)


# --- T_world_gate_from_spec ------------------------------------------------

def test_yaw0_axes_and_determinant():
    """At yaw=0: gate +Z = world +X, +Y = world -Z, +X = world -Y, and the
    rotation must be a proper (det=+1) rotation."""
    T = T_world_gate_from_spec(5.0, 1.0, 2.0, 0.0)
    R = T[:3, :3]
    assert abs(np.linalg.det(R) - 1.0) < 1e-9, np.linalg.det(R)
    assert np.allclose(R[:, 2], [1.0, 0.0, 0.0], atol=1e-9), "gate +Z must be world +X at yaw=0"
    assert np.allclose(R[:, 1], [0.0, 0.0, -1.0], atol=1e-9), "gate +Y must be world -Z at yaw=0"
    assert np.allclose(R[:, 0], [0.0, -1.0, 0.0], atol=1e-9), "gate +X must be world -Y at yaw=0"
    assert np.allclose(T[:3, 3], [5.0, 1.0, 2.0])
    print("  yaw=0: det(R)=+1, gate axes match the documented convention  OK")


def test_yaw90_normal_direction():
    """At yaw=90, positive (CCW about world +Z) rotation must swing gate +Z
    from world +X to world +Y."""
    T = T_world_gate_from_spec(0.0, 0.0, 0.0, 90.0)
    R = T[:3, :3]
    assert abs(np.linalg.det(R) - 1.0) < 1e-9
    assert np.allclose(R[:, 2], [0.0, 1.0, 0.0], atol=1e-9), f"gate +Z at yaw=90: {R[:, 2]}"
    print("  yaw=90: gate +Z now points along world +Y  OK")


def test_consistency_with_camera_frame():
    """
    A camera at (0,0,2.5) looking along world +X, in the OpenCV convention
    (cam +Z = world +X, cam +Y = world -Z, cam +X = world -Y), and a yaw=0
    gate at (5,0,2.5) directly ahead: T_cam_gate must be identity rotation
    with translation (0,0,5) exactly. This is the assertion that the
    manifest's yaw convention and gate_pose.py's gate-model frame agree --
    if it ever drifted, every downstream error number in this evaluator
    would be measuring the disagreement, not real detector/TF error.
    """
    R_world_cam = np.array([[0.0, 0.0, 1.0],
                            [-1.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0]])
    T_world_cam = make_transform(R_world_cam, [0.0, 0.0, 2.5])
    T_world_gate = T_world_gate_from_spec(5.0, 0.0, 2.5, 0.0)

    T_cam_gate = invert_transform(T_world_cam) @ T_world_gate
    assert np.allclose(T_cam_gate[:3, :3], np.eye(3), atol=1e-9), T_cam_gate[:3, :3]
    assert np.allclose(T_cam_gate[:3, 3], [0.0, 0.0, 5.0], atol=1e-9), T_cam_gate[:3, 3]
    print("  camera + yaw=0 gate: T_cam_gate is identity rotation, translation (0,0,5)  OK")


# --- rotate_xyz_deg_to_matrix / T_body_cam_usd_from_mount ------------------

def test_rotate_xyz_mount_axes():
    """
    rotateXYZ(90, 0, -90) is the "level, forward-looking" base mount
    spawn_example.py builds on (before the pitch_deg tilt is added): the
    USD camera's local -Z (forward) must land on body +X, local +Y (up) on
    body +Z, and local +X (right) on body -Y.
    """
    R = rotate_xyz_deg_to_matrix(90.0, 0.0, -90.0)
    assert abs(np.linalg.det(R) - 1.0) < 1e-9
    fwd = R @ np.array([0.0, 0.0, -1.0])
    up = R @ np.array([0.0, 1.0, 0.0])
    right = R @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(fwd, [1.0, 0.0, 0.0], atol=1e-9), fwd
    assert np.allclose(up, [0.0, 0.0, 1.0], atol=1e-9), up
    assert np.allclose(right, [0.0, -1.0, 0.0], atol=1e-9), right
    print("  rotateXYZ(90,0,-90): USD -Z->body +X, +Y->body +Z, +X->body -Y  OK")

    # With the actual spawn_example.py mount, rotateXYZ(105, 0, -90) --
    # 90 + pitch_deg=15 -- forward should be the same base forward rotated
    # 15 deg about the body Y axis: right stays -Y, forward keeps zero Y
    # component and |fwd.x| = cos(15), |fwd.z| = sin(15). Not asserting the
    # SIGN of the pitch (nose up vs down is a matter of rotation-direction
    # convention this test doesn't need to pin down) -- only that the tilt
    # magnitude and the unaffected axes are right, and that R2 is still a
    # proper rotation.
    R2 = rotate_xyz_deg_to_matrix(105.0, 0.0, -90.0)
    assert abs(np.linalg.det(R2) - 1.0) < 1e-9
    fwd2 = R2 @ np.array([0.0, 0.0, -1.0])
    right2 = R2 @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(right2, [0.0, -1.0, 0.0], atol=1e-9), right2
    assert abs(fwd2[1]) < 1e-9, fwd2
    assert abs(abs(fwd2[0]) - np.cos(np.radians(15.0))) < 1e-9, fwd2
    assert abs(abs(fwd2[2]) - np.sin(np.radians(15.0))) < 1e-9, fwd2
    print(f"  rotateXYZ(105,0,-90): right unchanged, forward tilted by 15 deg "
         f"(fwd={fwd2})  OK")

    T = T_body_cam_usd_from_mount()
    assert np.allclose(T[:3, :3], R2, atol=1e-9)
    assert np.allclose(T[:3, 3], [0.30, 0.0, 0.05], atol=1e-9)
    print("  T_body_cam_usd_from_mount(): translation + rotation match the mount constants  OK")


# --- match_detections --------------------------------------------------

def test_match_detections_correct_pairing_and_rejection():
    """Shuffled truth/est arrays must still pair up correctly; an extra
    unmatched estimate stays unmatched; a pair beyond max_dist_m is rejected."""
    truth = np.array([[0.0, 0.0, 5.0],
                      [3.0, 1.0, 8.0],
                      [-2.0, 0.5, 10.0]])
    # Same points, small noise, shuffled order, plus one extra estimate with
    # no corresponding truth gate.
    rng = np.random.default_rng(1)
    perm = [2, 0, 1]
    est = truth[perm] + rng.normal(0.0, 0.02, size=(3, 3))
    est = np.vstack([est, [50.0, 50.0, 50.0]])  # unmatchable extra

    matches = match_detections(est, truth, max_dist_m=1.0)
    assert len(matches) == 3, matches
    got_pairs = {(i, j) for i, j, _d in matches}
    expected_pairs = {(0, perm[0]), (1, perm[1]), (2, perm[2])}
    assert got_pairs == expected_pairs, (got_pairs, expected_pairs)
    matched_est_idx = {i for i, _j, _d in matches}
    assert 3 not in matched_est_idx, "the unmatchable extra estimate must not be matched"
    print("  shuffled truth/est pair up correctly; extra estimate stays unmatched  OK")

    # A truth/est pair further apart than max_dist_m must be rejected outright.
    far_est = np.array([[100.0, 100.0, 100.0]])
    far_truth = np.array([[0.0, 0.0, 0.0]])
    assert match_detections(far_est, far_truth, max_dist_m=1.0) == []
    print("  pair beyond max_dist_m is rejected  OK")


# --- decide_optical_frame -----------------------------------------------

def test_decide_optical_frame_clear_winner():
    """One convention ~0.1 m errors, the other ~8 m -> picks the accurate
    one with confident=True."""
    good = {"matches": [{"pos_err_m": 0.08}, {"pos_err_m": 0.12}, {"pos_err_m": 0.10}]}
    bad = {"matches": [{"pos_err_m": 7.5}, {"pos_err_m": 8.5}]}

    decision = decide_optical_frame(good, bad)
    assert decision["optical_frame"] is True
    assert decision["confident"] is True, decision
    print(f"  clear winner (optical_frame=True): {decision['reason']}")

    decision2 = decide_optical_frame(bad, good)
    assert decision2["optical_frame"] is False
    assert decision2["confident"] is True, decision2
    print(f"  clear winner (optical_frame=False): {decision2['reason']}")


def test_decide_optical_frame_both_bad():
    """Both conventions have large, comparable errors -> not confident."""
    bad_a = {"matches": [{"pos_err_m": 6.0}, {"pos_err_m": 7.0}]}
    bad_b = {"matches": [{"pos_err_m": 6.5}, {"pos_err_m": 5.5}]}
    decision = decide_optical_frame(bad_a, bad_b)
    assert decision["confident"] is False, decision
    print(f"  both bad: confident=False as expected ({decision['reason']})")

    no_matches = {"matches": []}
    decision2 = decide_optical_frame(no_matches, no_matches)
    assert decision2["optical_frame"] is None
    assert decision2["confident"] is False
    print("  neither convention matched anything: optical_frame=None, confident=False  OK")


# --- end-to-end offline: proves the harness can tell the conventions apart --

def _heading_yaw_deg(fwd: np.ndarray) -> float:
    """Horizontal heading (degrees) of a world-frame direction vector,
    matching spawn_test_gates.gate_world_pose's yaw_deg = atan2(h.y, h.x)
    -- duplicated here (a few lines) rather than imported, since this test
    lives in vision/ and that function lives in workspace/ (see
    test_gate_layout.py's own sys.path note for why the two directories
    don't import each other)."""
    h = np.array([fwd[0], fwd[1], 0.0])
    n = np.linalg.norm(h)
    h = h / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
    return float(np.degrees(np.arctan2(h[1], h[0])))


def _make_manifest_for_camera(T_world_cam_cv: np.ndarray):
    """A small manifest of gates in front of an ARBITRARY OpenCV-convention
    camera pose (not necessarily axis-aligned), spread apart along the
    camera's own right/up axes so their holes never touch in the image
    regardless of how the camera is tilted."""
    R_cv = T_world_cam_cv[:3, :3]
    cam_pos = T_world_cam_cv[:3, 3]
    right, down, fwd = R_cv[:, 0], R_cv[:, 1], R_cv[:, 2]
    up = -down
    base_yaw = _heading_yaw_deg(fwd)

    # (range_m, lateral_along_right, vertical_along_up, yaw_rel_deg, name)
    specs = [
        (5.0, -2.0, 0.0, 0.0, "g0"),
        (7.0, 0.0, 1.1, 20.0, "g1"),
        (9.0, 2.3, -1.6, -15.0, "g2"),
    ]
    manifest = []
    for rng, lat, vert, yaw_rel, name in specs:
        center = cam_pos + rng * fwd + lat * right + vert * up
        manifest.append({"name": name, "x": float(center[0]), "y": float(center[1]),
                         "z": float(center[2]), "yaw_deg": base_yaw + yaw_rel,
                         "side": DEFAULT_GATE_SIDE})
    return manifest


def _render_manifest(manifest, T_world_cam_cv):
    img = None
    for g in manifest:
        T_world_gate = T_world_gate_from_spec(g["x"], g["y"], g["z"], g["yaw_deg"])
        T_cam_gate = invert_transform(T_world_cam_cv) @ T_world_gate
        assert T_cam_gate[2, 3] > 0, f"{g['name']} ended up behind the camera in test setup"
        img = render_synthetic_gate(INTR, T_cam_gate, gate_side=g["side"],
                                    bar_thickness=0.10, img=img)
    return img


def test_end_to_end_offline_distinguishes_conventions():
    """
    Builds the pose exactly the way the real pipeline now has to (FACT 1):
    a map->body TF (body yawed 30 deg so it is not trivially identity) composed
    with the static body->camera mount, giving T_world_cam_usd. CV ground
    truth is T_world_cam_from_transform(T_world_cam_usd, optical_frame=False)
    -- the "expected" branch. Gates are rendered through that CV truth, so
    the test is honest about which pose actually produced the pixels.

    Feeding evaluate_frame the raw USD pose must pick optical_frame=False
    (that's the correction that recovers the CV truth); feeding it the CV
    pose directly must pick optical_frame=True (already-correct, no
    correction needed). This is what proves the harness distinguishes the
    two conventions rather than agreeing with whichever it's given.
    """
    def Rz(deg):
        a = np.radians(deg)
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    T_map_body = make_transform(Rz(30.0), [0.0, 0.0, 2.5])
    T_world_cam_usd = T_map_body @ T_body_cam_usd_from_mount()
    T_world_cam_cv = T_world_cam_from_transform(T_world_cam_usd, optical_frame=False)

    manifest = _make_manifest_for_camera(T_world_cam_cv)
    rgb = _render_manifest(manifest, T_world_cam_cv)

    # --- Case 1: raw pose IS the USD/body convention -- optical_frame=False
    # is the correction that recovers ground truth. ---
    result_usd = evaluate_frame(rgb, INTR, T_world_cam_usd, manifest,
                                gate_side=DEFAULT_GATE_SIDE, corner_sigma_px=1.0)
    decision_usd = result_usd["decision"]
    assert decision_usd["optical_frame"] is False, decision_usd
    assert decision_usd["confident"] is True, decision_usd
    for m in result_usd["per_convention"][False]["matches"]:
        assert m["pos_err_m"] < 0.2, m
    assert len(result_usd["per_convention"][False]["matches"]) == len(manifest)
    print(f"  raw USD pose: decision={decision_usd['optical_frame']}, "
         f"confident={decision_usd['confident']}, "
         f"mean err {decision_usd['mean_pos_err_false']:.4f} m  OK")

    # --- Case 2: raw pose IS ALREADY the CV/optical convention --
    # optical_frame=True (no correction) is correct. ---
    result_cv = evaluate_frame(rgb, INTR, T_world_cam_cv, manifest,
                               gate_side=DEFAULT_GATE_SIDE, corner_sigma_px=1.0)
    decision_cv = result_cv["decision"]
    assert decision_cv["optical_frame"] is True, decision_cv
    assert decision_cv["confident"] is True, decision_cv
    for m in result_cv["per_convention"][True]["matches"]:
        assert m["pos_err_m"] < 0.2, m
    assert len(result_cv["per_convention"][True]["matches"]) == len(manifest)
    print(f"  raw CV pose: decision={decision_cv['optical_frame']}, "
         f"confident={decision_cv['confident']}, "
         f"mean err {decision_cv['mean_pos_err_true']:.4f} m  OK")

    # And the wrong branch in each case really is bad -- this is the part
    # that proves the harness can tell the two conventions apart, not just
    # that the right one happens to work.
    wrong_matches_1 = result_usd["per_convention"][True]["matches"]
    wrong_matches_2 = result_cv["per_convention"][False]["matches"]
    assert (not wrong_matches_1) or all(m["pos_err_m"] > 1.0 for m in wrong_matches_1)
    assert (not wrong_matches_2) or all(m["pos_err_m"] > 1.0 for m in wrong_matches_2)
    print("  the wrong optical_frame branch is either unmatched or grossly wrong in both cases  OK")


if __name__ == "__main__":
    print("test_yaw0_axes_and_determinant");   test_yaw0_axes_and_determinant()
    print("test_yaw90_normal_direction");      test_yaw90_normal_direction()
    print("test_consistency_with_camera_frame"); test_consistency_with_camera_frame()
    print("test_rotate_xyz_mount_axes");       test_rotate_xyz_mount_axes()
    print("test_match_detections_correct_pairing_and_rejection")
    test_match_detections_correct_pairing_and_rejection()
    print("test_decide_optical_frame_clear_winner"); test_decide_optical_frame_clear_winner()
    print("test_decide_optical_frame_both_bad");      test_decide_optical_frame_both_bad()
    print("test_end_to_end_offline_distinguishes_conventions")
    test_end_to_end_offline_distinguishes_conventions()
    print("\nAll tests passed.")
