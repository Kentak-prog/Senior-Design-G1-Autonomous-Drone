"""Offline tests for live_gate_test.py (no drone, no feed, no CNN). Run: python3 test_live_gate_test.py"""

import os
import sys
import tempfile

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from live_gate_test import Lens, load_calib, quad_from_contour  # noqa: E402
from gate_pose import estimate_gate_pose  # noqa: E402

APPROX = os.path.join(_HERE, "calib", "hires_front_small_approx.yml")


def test_load_approx_calib():
    K, D, fisheye, size = load_calib(APPROX)
    assert size == (1024, 768), size
    assert not fisheye
    assert np.allclose(K, [[504.9, 0, 512], [0, 504.9, 384], [0, 0, 1]])
    assert np.allclose(D, 0)


def test_lens_rescales_to_frame_size():
    K, D, fe, size = load_calib(APPROX)
    lens = Lens(K, D, fe, size, (512, 384), calibrated=True)
    assert np.isclose(lens.K[0, 0], 504.9 / 2) and np.isclose(lens.K[0, 2], 256)
    assert np.isclose(K[0, 0], 504.9), "Lens must not modify the caller's K"


def test_fisheye_detected_from_4_coeffs():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "fe.yml")
        fs = cv2.FileStorage(p, cv2.FILE_STORAGE_WRITE)
        fs.write("M", np.array([[400., 0, 320], [0, 400, 240], [0, 0, 1]]))
        fs.write("D", np.array([[0.1], [0.01], [0.0], [0.0]]))
        fs.release()
        _, D, fisheye, _ = load_calib(p)
        assert fisheye and D.size == 4
        lens = Lens(*load_calib(p)[:3], (0, 0), (640, 480), calibrated=True)
        px = np.array([[100., 80.], [500., 400.]])
        assert np.allclose(lens.to_image(lens.to_pinhole(px)), px, atol=0.05)


def test_quad_from_square_contour():
    m = np.zeros((200, 200), np.uint8)
    cv2.rectangle(m, (50, 60), (150, 160), 255, -1)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    q = quad_from_contour(cnts[0])
    assert q.shape == (4, 2)
    assert np.allclose(sorted(q[:, 0]), [50, 50, 150, 150], atol=1)


def test_real_gate_label_range():
    """Hand-labelled frame 720 of real_gate_01 (0.914 m gate, approx intrinsics) solves to ~1.75 m."""
    K, D, fe, size = load_calib(APPROX)
    lens = Lens(K, D, fe, size, size, calibrated=True)
    c = np.array([[458.96, 85.25], [723.17, 87.52], [750.03, 347.32], [452.68, 347.01]])
    g = estimate_gate_pose(lens.to_pinhole(c), lens.intr, gate_side=0.9144, assume_ordered=True)
    assert abs(g.range_m - 1.75) < 0.03, g.range_m
    assert g.reprojection_error_px < 2.0


def test_hybrid_finds_rolled_opening():
    """A 45 deg rolled coral frame with a bump on the opening: the hybrid detector returns the
    opening's 4 corners (not the box), and a box with no frame in it falls back to "bbox"."""
    from live_gate_test import HybridDetector
    img = np.full((768, 1024, 3), 40, np.uint8)
    c, r_out, r_in = np.array([512.0, 384.0]), 200.0, 175.0
    diamond = lambda r: np.array([c + r * np.array([np.cos(a), np.sin(a)])  # noqa: E731
                                  for a in np.radians([0, 90, 180, 270])])
    cv2.fillPoly(img, [diamond(r_out).astype(np.int32)], (80, 110, 230))       # coral (BGR)
    cv2.fillPoly(img, [diamond(r_in).astype(np.int32)], (40, 40, 40))
    cv2.circle(img, (512 + 80, 384 - 80), 12, (80, 110, 230), -1)              # "hand" bump on the inner edge
    gate_box = np.array([[312, 184], [712, 184], [712, 584], [312, 584]], float)
    empty_box = np.array([[20, 20], [120, 20], [120, 120], [20, 120]], float)
    fake_cnn = lambda bgr: [(gate_box, 0.9, "bbox"), (empty_box, 0.5, "bbox")]  # noqa: E731
    out = HybridDetector(fake_cnn, (0, 80, 50), (12, 255, 255))(img)
    (q, _, src), (_, _, src2) = out
    assert src == "yolo+color", src
    want = diamond(r_in)
    err = max(np.linalg.norm(want - p, axis=1).min() for p in q)
    assert err < 6.0, (q, err)
    assert src2 == "bbox", src2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("All tests passed.")
