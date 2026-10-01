"""Self-tests. Run: python3 test_fly_gate_approach.py

Pure geometry only: no pymavlink, no ROS, no vehicle. The MAVLink flow in
fly_gate_approach.py is NOT covered here (untested against the cluster).
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fly_gate_approach as fga


def test_import_without_pymavlink():
    assert "pymavlink" not in sys.modules
    assert fga.TYPE_MASK_POS_VEL_YAW == 2496 and fga.TYPE_MASK_POS_YAW == 2552
    print("  imports without pymavlink; type masks 2496 / 2552  OK")


def test_carrot_start_mid_end():
    start = [1.0, 2.0, -2.5]
    s0 = fga.carrot_state(start, 0.0, 3.0, 1.0, 0.0)
    assert np.allclose(s0["pos"], start) and np.allclose(s0["vel"], [1, 0, 0]) and not s0["done"]
    s1 = fga.carrot_state(start, 0.0, 3.0, 1.0, 1.5)
    assert np.allclose(s1["pos"], [2.5, 2.0, -2.5]) and not s1["done"]
    s2 = fga.carrot_state(start, 0.0, 3.0, 1.0, 10.0)
    assert np.allclose(s2["pos"], [4.0, 2.0, -2.5]) and np.allclose(s2["vel"], 0) and s2["done"]
    assert fga.flight_duration_s(3.0, 1.5) == 2.0
    print("  carrot: start, midpoint, clamp at end with zero feed-forward velocity  OK")


def test_heading_rotation():
    # NED yaw 90 deg = east: +y, altitude (z) untouched.
    s = fga.carrot_state([0, 0, -2.0], math.pi / 2, 2.0, 1.0, 1.0)
    assert np.allclose(s["pos"], [0.0, 1.0, -2.0], atol=1e-12)
    assert np.allclose(s["vel"], [0.0, 1.0, 0.0], atol=1e-12)
    s = fga.carrot_state([0, 0, -2.0], math.radians(45), 2.0, 1.0, 100.0)
    assert np.allclose(s["pos"], [math.sqrt(2.0), math.sqrt(2.0), -2.0])
    # NED <-> ENU
    assert np.allclose(fga.ned_to_enu([1.0, 2.0, -3.0]), [2.0, 1.0, 3.0])
    assert np.allclose(fga.enu_to_ned(fga.ned_to_enu([1.0, 2.0, -3.0])), [1.0, 2.0, -3.0])
    # East-pointing ENU heading is NED yaw +90 deg; north is 0.
    assert abs(fga.yaw_ned_from_heading_enu(np.array([1.0, 0.0, 0.0])) - math.pi / 2) < 1e-12
    assert abs(fga.yaw_ned_from_heading_enu(np.array([0.0, 1.0, 0.0]))) < 1e-12
    print("  heading rotation, altitude fixed, NED/ENU and yaw conversion  OK")


def test_camera_forward_horizontal():
    # USD camera looking along world +X, level: forward = -Z_cam = +X world.
    # Columns of R are camera axes in world: X_cam=-Y world, Y_cam=+Z, Z_cam=-X world.
    R = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert np.allclose(fga.camera_forward_horizontal_enu(R), [1.0, 0.0, 0.0])
    # Pitched 15 deg down: horizontal part must still be +X, unit length.
    a = math.radians(15.0)
    Rp = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]) @ R
    f = fga.camera_forward_horizontal_enu(Rp)
    assert np.allclose(f, [1.0, 0.0, 0.0], atol=1e-9) and abs(np.linalg.norm(f) - 1) < 1e-12
    print("  camera horizontal forward from R_world_cam_usd  OK")


GATES = [{"name": "g0", "x": 6.0, "y": 0.0, "z": 2.5},
         {"name": "g1", "x": 10.0, "y": 3.0, "z": 2.5}]


def test_clearance_refusal():
    start, d = np.array([0.0, 0.0, 2.5]), np.array([1.0, 0.0, 0.0])
    ok, rep = fga.check_clearance(start, d, 3.0, GATES, 2.0)
    assert ok and abs(rep[0]["min_dist_m"] - 3.0) < 1e-9 and abs(rep[0]["end_dist_m"] - 3.0) < 1e-9
    ok, rep = fga.check_clearance(start, d, 4.5, GATES, 2.0)
    assert not ok and abs(rep[0]["min_dist_m"] - 1.5) < 1e-9
    # Exactly at the limit is allowed.
    assert fga.check_clearance(start, d, 4.0, GATES, 2.0)[0]
    # Flying through a gate (past it) is refused with closest approach 0.
    ok, rep = fga.check_clearance(start, d, 8.0, GATES, 2.0)
    assert not ok and rep[0]["min_dist_m"] < 1e-9 and abs(rep[0]["s_closest_m"] - 6.0) < 1e-9
    # Off-path gate (g1, 3 m to the side): closest approach is 3 m at x=10.
    ok, rep = fga.check_clearance(start, d, 12.0, [GATES[1]], 2.0)
    assert ok and abs(rep[0]["min_dist_m"] - 3.0) < 1e-9
    safe = fga.max_safe_distance(start, d, GATES, 2.0)
    assert abs(safe - 4.0) < 1e-6, safe
    print("  clearance: allowed at the limit, refused beyond, through-gate, max safe distance  OK")


def test_manifest_start_and_heading():
    raw = {"gates": GATES, "camera_pose_usd": {
        "position": [0.0, 0.0, 2.5],
        "R_world_cam_usd": [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]}}
    start, d, gates = fga.manifest_start_and_heading(raw)
    assert np.allclose(start, [0, 0, 2.5]) and np.allclose(d, [1, 0, 0]) and len(gates) == 2
    try:
        fga.manifest_start_and_heading({"gates": GATES})
        raise AssertionError("expected ValueError without camera_pose_usd")
    except ValueError:
        pass
    print("  manifest start/heading; missing camera_pose_usd rejected  OK")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("\nAll tests passed.")
