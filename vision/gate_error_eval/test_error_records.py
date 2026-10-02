"""Self-tests. Run: python3 test_error_records.py

Offline only -- numpy + cv2, no ROS, no matplotlib. Covers error_records.py
(pose interpolation, frame_rows, CSV, summarize) and the ROS-free half of
eval_gate_bag.py (image decode, time base, per-frame function).
"""

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # vision/
sys.path.insert(0, HERE)

from camera import CameraIntrinsics, invert_transform
from gate_pose import project_gate, DEFAULT_GATE_SIDE
from ros_camera import transform_to_matrix, T_world_cam_from_transform
from eval_isaac_gates import (T_world_gate_from_spec, T_body_cam_usd_from_mount,
                              evaluate_frame)
import error_records as er
import eval_gate_bag as egb  # must import cleanly with no ROS installed

INTR = CameraIntrinsics.from_horizontal_fov(1280, 960, 90.0)


def quat_z(deg):
    a = np.radians(deg) / 2.0
    return np.array([0.0, 0.0, np.sin(a), np.cos(a)])  # xyzw


def same_rotation(q1, q2, tol=1e-9):
    return abs(abs(float(np.dot(q1, q2))) - 1.0) < tol


# --- PoseBuffer -----------------------------------------------------------

def test_slerp_midpoint_and_flip():
    q0, q1 = quat_z(0.0), quat_z(90.0)
    mid = er.slerp_xyzw(q0, q1, 0.5)
    assert np.allclose(mid, quat_z(45.0), atol=1e-9), mid
    # q1 and -q1 are the same rotation: result must be identical (short arc).
    mid_flip = er.slerp_xyzw(q0, -q1, 0.5)
    assert same_rotation(mid_flip, quat_z(45.0)), mid_flip
    assert np.allclose(er.slerp_xyzw(q0, q1, 0.0), q0) and same_rotation(er.slerp_xyzw(q0, q1, 1.0), q1)
    print("  SLERP midpoint = 45 deg; q vs -q hemisphere flip gives the same rotation  OK")


def test_buffer_interpolation_and_range():
    buf = er.PoseBuffer(max_extrap_s=0.05)
    # Added out of order on purpose; second quaternion is stored as -q.
    buf.add(1.0, [2.0, 0.0, 1.0], -quat_z(90.0))
    buf.add(0.0, [0.0, 0.0, 1.0], quat_z(0.0))
    pos, q = buf.interpolate(0.5)
    assert np.allclose(pos, [1.0, 0.0, 1.0])
    assert same_rotation(q, quat_z(45.0)), q
    assert buf.interpolate(1.04) is not None and buf.interpolate(-0.04) is not None
    pos_c, _ = buf.interpolate(1.04)
    assert np.allclose(pos_c, [2.0, 0.0, 1.0]), "within tolerance must clamp to the end sample"
    assert buf.interpolate(1.06) is None and buf.interpolate(-0.06) is None
    assert er.PoseBuffer().interpolate(0.0) is None
    print("  position lerp, sorted insert, clamp within tolerance, None beyond it  OK")


def test_velocity_constant_track():
    buf = er.PoseBuffer()
    v = np.array([1.0, -0.5, 0.2])
    for t in np.arange(0.0, 2.0, 1.0 / 130.0):
        buf.add(t, v * t + [3.0, 0.0, 2.0], quat_z(0.0))
    for t in (0.0, 0.5, 1.0, 1.98):
        vel = buf.velocity(t)
        assert np.allclose(vel, v, atol=1e-9), (t, vel)
    assert buf.velocity(5.0) is None
    print("  velocity of a constant-velocity track recovered everywhere  OK")


def test_T_world_cam_raw_at_matches_manual():
    buf = er.PoseBuffer()
    buf.add(0.0, [0.0, 0.0, 2.5], quat_z(10.0))
    buf.add(1.0, [2.0, 1.0, 2.5], quat_z(50.0))
    T = er.T_world_cam_raw_at(buf, 0.25)
    pos = np.array([0.5, 0.25, 2.5])
    T_manual = transform_to_matrix(pos, quat_z(20.0)) @ T_body_cam_usd_from_mount()
    assert np.allclose(T, T_manual, atol=1e-9)
    assert er.T_world_cam_raw_at(buf, 3.0) is None
    # Mount override (eval_gate_bag --mount-xyz): only the translation changes.
    T_mount = T_body_cam_usd_from_mount().copy()
    T_mount[:3, 3] = [-0.04, 0.0, -0.041]
    T_o = er.T_world_cam_raw_at(buf, 0.25, T_mount)
    assert np.allclose(T_o, transform_to_matrix(pos, quat_z(20.0)) @ T_mount, atol=1e-9)
    assert np.allclose(T_o[:3, :3], T[:3, :3]) and not np.allclose(T_o[:3, 3], T[:3, 3])
    print("  T_world_cam_raw_at == transform_to_matrix(interp) @ mount; None out of range; "
          "mount override  OK")


# --- synthetic scene ------------------------------------------------------

def make_scene():
    """Camera + gates built the way the real pipeline builds them. Returns
    (T_world_cam_raw, T_world_cam_cv, manifest). g0/g1 visible, g2 visible
    but the fake detector will skip it, g3 is BEHIND the camera."""
    a = np.radians(30.0)
    c, s = np.cos(a), np.sin(a)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    T_map_body = np.eye(4)
    T_map_body[:3, :3] = Rz
    T_map_body[:3, 3] = [1.0, 2.0, 2.5]
    T_raw = T_map_body @ T_body_cam_usd_from_mount()
    T_cv = T_world_cam_from_transform(T_raw, False)
    R, pos = T_cv[:3, :3], T_cv[:3, 3]
    right, down, fwd = R[:, 0], R[:, 1], R[:, 2]
    yaw = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
    specs = [("g0", 5.0, -2.0, 0.0, 0.0), ("g1", 7.0, 0.0, 1.1, 20.0),
             ("g2", 9.0, 2.3, -1.6, -15.0), ("g3", -6.0, 0.0, 0.0, 0.0)]
    manifest = []
    for name, rng, lat, vert, yrel in specs:
        ctr = pos + rng * fwd + lat * right - vert * down
        manifest.append({"name": name, "x": float(ctr[0]), "y": float(ctr[1]),
                         "z": float(ctr[2]), "yaw_deg": yaw + yrel, "side": DEFAULT_GATE_SIDE})
    return T_raw, T_cv, manifest


def make_fake_detector(T_cv, manifest, skip=("g2", "g3")):
    """Detector that ignores pixels and returns the exact projected corners."""
    T_cam_world = invert_transform(T_cv)

    def detector(_rgb):
        out = []
        for g in manifest:
            if g["name"] in skip:
                continue
            T_cam_gate = T_cam_world @ T_world_gate_from_spec(g["x"], g["y"], g["z"], g["yaw_deg"])
            out.append(project_gate(T_cam_gate, INTR, g["side"]))
        return out
    return detector


BLANK = np.zeros((INTR.height, INTR.width, 3), dtype=np.uint8)


def test_frame_rows_synthetic():
    T_raw, T_cv, manifest = make_scene()
    result = evaluate_frame(BLANK, INTR, T_raw, manifest, detector=make_fake_detector(T_cv, manifest))
    rows = er.frame_rows(1.5, 7, INTR, T_raw, manifest, result, [1.0, 2.0, 2.5], 0.8)
    assert [r["gate"] for r in rows] == ["g0", "g1", "g2", "g3"]
    by = {r["gate"]: r for r in rows}
    for name in ("g0", "g1"):
        r = by[name]
        assert r["detected"] and r["in_view"], r
        assert r["pos_err_m"] < 1e-3 and abs(r["range_err_m"]) < 1e-3 and r["lateral_err_m"] < 1e-3, r
        est = np.array([r["est_x"], r["est_y"], r["est_z"]])
        tru = np.array([r["true_x"], r["true_y"], r["true_z"]])
        assert np.allclose(est, tru, atol=1e-3)
        assert r["reproj_rms_px"] is not None and r["normal_is_reliable"] in (True, False)
    assert abs(by["g0"]["range_true_m"] - 5.0) < 1e-6 and abs(by["g1"]["range_true_m"] - 7.0) < 1e-6
    assert by["g0"]["off_axis_deg"] > 0.0 and by["g1"]["off_axis_deg"] > 0.0
    assert np.allclose([by["g0"]["true_x"], by["g0"]["true_y"], by["g0"]["true_z"]],
                       [manifest[0]["x"], manifest[0]["y"], manifest[0]["z"]])
    # g2: in view but not detected -> truth columns filled, error columns blank.
    r = by["g2"]
    assert r["in_view"] and not r["detected"]
    for k in ("est_x", "est_y", "est_z", "pos_err_m", "range_err_m", "lateral_err_m",
              "normal_err_deg", "reproj_rms_px", "normal_is_reliable"):
        assert r[k] is None, (k, r[k])
    # g3: behind the camera.
    assert not by["g3"]["in_view"] and by["g3"]["range_true_m"] < 0
    assert by["g3"]["off_axis_deg"] > 90.0
    assert all(r["t_s"] == 1.5 and r["frame"] == 7 and r["drone_speed_mps"] == 0.8
               and r["drone_x"] == 1.0 for r in rows)
    print("  frame_rows: ~0 error for detected gates, blank errors for undetected, "
          "in_view/range/off_axis correct  OK")
    return rows


def test_in_view_requires_all_corners():
    """A gate whose CENTER is just inside the top edge but whose top bar is
    cut off must not count as in view -- the case the first moving run hit
    (gates above the drone climbing out of frame on approach)."""
    T_raw, T_cv, _ = make_scene()
    pos, down, fwd = T_cv[:3, 3], T_cv[:3, 1], T_cv[:3, 2]
    yaw = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
    rng = 5.0
    # Center ~20 px below the top edge: y_px = cy - fy * up / rng.
    up = (INTR.cy - 20.0) * rng / INTR.fy
    ctr = pos + rng * fwd - up * down
    manifest = [{"name": "cut", "x": float(ctr[0]), "y": float(ctr[1]), "z": float(ctr[2]),
                 "yaw_deg": yaw, "side": DEFAULT_GATE_SIDE}]
    result = evaluate_frame(BLANK, INTR, T_raw, manifest, detector=lambda _rgb: [])
    r = er.frame_rows(0.0, 0, INTR, T_raw, manifest, result, [0.0, 0.0, 0.0], None,
                      gate_side=DEFAULT_GATE_SIDE)[0]
    from camera import project_points, in_frame, transform_points
    c_px = project_points(transform_points(invert_transform(T_cv), ctr), INTR)
    assert in_frame(c_px, INTR)[0], c_px  # center really is inside
    assert not r["in_view"], r
    print("  in_view is False when the center is inside but the top bar is cut off  OK")


def test_csv_roundtrip():
    T_raw, T_cv, manifest = make_scene()
    result = evaluate_frame(BLANK, INTR, T_raw, manifest, detector=make_fake_detector(T_cv, manifest))
    rows = er.frame_rows(0.25, 3, INTR, T_raw, manifest, result, [1.0, 2.0, 2.5], None)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.csv")
        er.write_csv(p, rows)
        back = er.read_csv(p)
    assert len(back) == len(rows)
    for a, b in zip(rows, back):
        for k in er.CSV_FIELDS:
            if a[k] is None:
                assert b[k] is None, (k, b[k])
            elif isinstance(a[k], (bool, str)):
                assert a[k] == b[k], (k, a[k], b[k])
            else:
                assert abs(a[k] - b[k]) < 1e-12, (k, a[k], b[k])
    assert isinstance(back[0]["detected"], bool) and isinstance(back[0]["frame"], int)
    print("  CSV round trip preserves floats, bools, blanks, strings  OK")


def test_summarize_hand_built():
    def row(gate, det, in_view, pos=None, rng=None, lat=None, rt=5.0):
        return {"gate": gate, "detected": det, "in_view": in_view, "pos_err_m": pos,
                "range_err_m": rng, "lateral_err_m": lat, "range_true_m": rt}
    rows = [row("a", True, True, 0.1, 0.08, 0.02, 4.0), row("a", True, True, 0.3, -0.2, 0.04, 6.0),
            row("a", False, True), row("a", False, False),
            row("b", False, True), row("b", False, True)]
    s = er.summarize(rows)
    a = s["a"]
    assert a["n_frames_in_view"] == 3 and a["n_detected"] == 2
    assert abs(a["detection_rate"] - 2 / 3) < 1e-12
    assert abs(a["pos_err_mean"] - 0.2) < 1e-12 and abs(a["pos_err_std"] - 0.1) < 1e-12
    assert abs(a["pos_err_median"] - 0.2) < 1e-12
    assert abs(a["pos_err_p95"] - np.percentile([0.1, 0.3], 95)) < 1e-12
    assert abs(a["range_err_mean"] - (-0.06)) < 1e-12 and abs(a["range_err_abs_mean"] - 0.14) < 1e-12
    assert abs(a["lateral_err_mean"] - 0.03) < 1e-12 and abs(a["range_true_mean"] - 5.0) < 1e-12
    assert s["b"]["n_detected"] == 0 and s["b"]["detection_rate"] == 0.0
    assert np.isnan(s["b"]["pos_err_mean"])
    assert s["ALL"]["n_frames_in_view"] == 5 and s["ALL"]["n_detected"] == 2
    assert list(s)[-1] == "ALL"
    assert "ALL" in er.format_summary_table(s)
    print("  summarize: per-gate and ALL numbers match hand calculation  OK")


# --- eval_gate_bag (ROS-free half) ---------------------------------------

def test_decode_image_padded_step():
    h, w = 4, 5
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    pad = 7

    def pack(arr, extra):
        ch = arr.shape[2]
        buf = np.zeros((h, w * ch + extra), dtype=np.uint8)
        buf[:, :w * ch] = arr.reshape(h, w * ch)
        buf[:, w * ch:] = 255  # garbage in the padding
        return buf.tobytes(), w * ch + extra

    data, step = pack(rgb, pad)
    assert np.array_equal(egb.decode_image(data, w, h, step, "rgb8"), rgb)
    data, step = pack(rgb[:, :, ::-1], pad)
    assert np.array_equal(egb.decode_image(data, w, h, step, "bgr8"), rgb)
    rgba = np.concatenate([rgb, np.full((h, w, 1), 9, np.uint8)], axis=2)
    data, step = pack(rgba, pad)
    assert np.array_equal(egb.decode_image(data, w, h, step, "rgba8"), rgb)
    bgra = np.concatenate([rgb[:, :, ::-1], np.full((h, w, 1), 9, np.uint8)], axis=2)
    data, step = pack(bgra, 0)
    assert np.array_equal(egb.decode_image(data, w, h, step, "bgra8"), rgb)
    try:
        egb.decode_image(data, w, h, step, "mono8")
        raise AssertionError("mono8 should be rejected")
    except ValueError:
        pass
    print("  decode_image: rgb8/bgr8/rgba8/bgra8 with padded step; others rejected  OK")


def test_time_base_and_gap():
    name, it, pt, warn = egb.choose_time_base([10.0, 11.0], [100.0, 101.0],
                                              [9.5, 10.5, 11.5], [100.0, 100.5, 101.5], "auto")
    assert name == "header" and warn is None
    name, it, pt, warn = egb.choose_time_base([10.0, 11.0], [100.0, 101.0],
                                              [1.0, 2.0], [100.0, 100.5, 101.5], "auto")
    assert name == "bag" and warn and "WARNING" in warn and list(it) == [100.0, 101.0]
    assert egb.choose_time_base([1], [2], [1], [2], "bag")[0] == "bag"
    gap = egb.median_pose_gap_s([1.0, 2.0], [0.99, 1.5, 2.02])
    assert abs(gap - 0.015) < 1e-9, gap
    print("  time base: header on overlap, bag + WARNING otherwise; median gap  OK")


def test_build_intrinsics_stale_warning():
    stale = {"width": 320, "height": 240, "k": [763.5, 0, 160, 0, 763.5, 120, 0, 0, 1], "d": []}
    intr, msgs = egb.build_intrinsics(stale, 1280, 720, None)
    assert intr.width == 320 and any("WARNING" in m and "stale" in m for m in msgs)
    intr, msgs = egb.build_intrinsics(stale, 1280, 720, 60.0)
    assert intr.width == 1280 and abs(intr.fx - 640 / np.tan(np.radians(30))) < 1e-9
    ok = {"width": 1280, "height": 720, "k": [1108.5, 0, 640, 0, 1108.5, 360, 0, 0, 1], "d": []}
    assert egb.build_intrinsics(ok, 1280, 720, None)[1] == []
    print("  build_intrinsics: stale camera_info WARNING, --hfov-deg override  OK")


def test_evaluate_image_at():
    T_raw, T_cv, manifest = make_scene()
    # Constant-pose buffer reproducing T_raw: body pose = T_raw @ inv(mount).
    T_body = T_raw @ invert_transform(T_body_cam_usd_from_mount())
    from ros_camera import _rot_to_quat_xyzw
    q = _rot_to_quat_xyzw(T_body[:3, :3])
    buf = er.PoseBuffer()
    for t in np.arange(0.0, 1.0, 0.01):
        buf.add(t, T_body[:3, 3], q)
    det = make_fake_detector(T_cv, manifest)
    rows = egb.evaluate_image_at(BLANK, 0.5, 0.25, 4, INTR, buf, manifest, detector=det)
    assert rows is not None and len(rows) == len(manifest)
    r0 = rows[0]
    assert r0["detected"] and r0["pos_err_m"] < 1e-3 and r0["t_s"] == 0.25 and r0["frame"] == 4
    assert r0["drone_speed_mps"] is not None and r0["drone_speed_mps"] < 1e-6
    assert egb.evaluate_image_at(BLANK, 5.0, 4.75, 5, INTR, buf, manifest, detector=det) is None
    print("  evaluate_image_at: rows with ~0 error; None when no pose in range  OK")


def test_import_without_matplotlib():
    assert "matplotlib" not in sys.modules, "error_records/eval_gate_bag must not import matplotlib"
    assert "rclpy" not in sys.modules
    print("  neither matplotlib nor rclpy imported  OK")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("\nAll tests passed.")
