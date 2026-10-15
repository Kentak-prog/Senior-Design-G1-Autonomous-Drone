"""Self-tests for course geometry and gate crossing. Run: python test_course.py

This file deliberately reaches across the rl//workspace boundary, exactly like
vision/test_gate_layout.py does: workspace/spawn_test_gates.py has to live in
workspace/ to run inside Isaac's interpreter, but its pure geometry
(_T_world_gate, the manifest format) is the ground truth rl/course.py must
match, so sys.path is extended once, here, to import it.

Expectations are hand-written world-frame vectors, not round trips.
"""

import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "workspace"))

import numpy as np
import torch

from spawn_test_gates import _T_world_gate
from course import (Course, Gate, GateBatch, gate_batch, check_gate, random_course,
                    FIXED_TRACKS, TRACKS_DIR)

RADIUS = 0.30


def test_T_world_gate_matches_spawn():
    for yaw in [0.0, 30.0, 90.0, -45.0, 180.0, 270.5]:
        g = Gate("g", 3.0, -2.0, 2.5, yaw)
        assert np.allclose(g.T_world_gate(), _T_world_gate(3.0, -2.0, 2.5, yaw), atol=1e-12), yaw
    print("  Gate.T_world_gate == spawn_test_gates._T_world_gate  OK")


def test_gate_axes_hand_vectors():
    T0 = Gate("g", 1.0, 2.0, 3.0, 0.0).T_world_gate()
    assert np.allclose(T0[:3, 0], [0, -1, 0]), T0   # gate +X (right) = world -Y
    assert np.allclose(T0[:3, 1], [0, 0, -1])       # gate +Y (down) = world -Z
    assert np.allclose(T0[:3, 2], [1, 0, 0])        # flight direction = world +X
    assert np.allclose(T0[:3, 3], [1, 2, 3])
    T90 = Gate("g", 0, 0, 0, 90.0).T_world_gate()
    assert np.allclose(T90[:3, 2], [0, 1, 0], atol=1e-12), T90[:3, 2]   # flies north
    assert np.allclose(T90[:3, 0], [1, 0, 0], atol=1e-12)   # right of a north-facing gate is east
    print("  yaw 0: flight +x; yaw 90: flight +y, right = +x  OK")


def _batch(course, n):
    return gate_batch(course, torch.zeros(n, dtype=torch.long))


def _cross(course, p_prev, p_now):
    p0 = torch.tensor(p_prev, dtype=torch.float32)
    p1 = torch.tensor(p_now, dtype=torch.float32)
    return check_gate(p0, p1, _batch(course, len(p_prev)), RADIUS)


def test_crossings():
    # Gate at (10, 2, 3) facing world +y (yaw 90): flight direction +y,
    # right = world +x, down = world -z. Opening 1.5 m, bars 0.1 m.
    course = Course([Gate("g", 10.0, 2.0, 3.0, 90.0)])
    cases = [  # (dx from center, dz from center, name, expect_pass, expect_collide)
        (0.0, 0.0, "center", True, False),
        (0.4, 0.0, "inside shrunk opening", True, False),
        (0.0, -0.4, "low but inside", True, False),
        (0.5, 0.0, "in the radius margin", False, True),
        (0.7, 0.0, "frame hit (side)", False, True),
        (0.0, 0.7, "frame hit (top)", False, True),
        (1.2, 0.0, "outside outer frame", False, False),
        (0.0, 1.5, "far above", False, False),
    ]
    p_prev = [[10.0 + dx, 1.95, 3.0 + dz] for dx, dz, *_ in cases]
    p_now = [[10.0 + dx, 2.05, 3.0 + dz] for dx, dz, *_ in cases]
    passed, hit = _cross(course, p_prev, p_now)
    for i, (dx, dz, name, ep, ec) in enumerate(cases):
        assert bool(passed[i]) == ep and bool(hit[i]) == ec, (name, passed[i], hit[i])
    # backwards through the center: never a pass, and not a collision either
    passed, hit = _cross(course, [[10.0, 2.05, 3.0]], [[10.0, 1.95, 3.0]])
    assert not passed[0] and not hit[0]
    # moving parallel to the gate plane, nowhere near it: nothing
    passed, hit = _cross(course, [[5.0, 0.0, 3.0]], [[5.0, 0.1, 3.0]])
    assert not passed[0] and not hit[0]
    # sitting inside a bar without crossing: collision by proximity
    passed, hit = _cross(course, [[10.85, 2.0, 3.0]], [[10.85, 2.0, 3.0]])
    assert not passed[0] and hit[0]
    # 0.2 m from the right bar's outer face (bar spans 0.75..0.85 off center): too close;
    # 0.4 m: clear
    passed, hit = _cross(course, [[11.05, 2.0, 3.0], [11.25, 2.0, 3.0]],
                         [[11.05, 2.0, 3.0], [11.25, 2.0, 3.0]])
    assert hit[0] and not hit[1], hit
    print("  pass / frame hit / miss / backwards / proximity cases  OK")


def test_json_roundtrip_and_manifest_format():
    c = Course([Gate("a", 1.0, 2.0, 3.0, 45.0), Gate("b", -4.0, 0.5, 2.0, -90.0)], "t")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c.json")
        c.save(path)
        c2 = Course.load(path)
        raw = json.load(open(path))
    assert c2.name == "t" and len(c2) == 2
    for g1, g2 in zip(c.gates, c2.gates):
        assert (g1.name, g1.x, g1.y, g1.z, g1.yaw_deg, g1.side) == \
               (g2.name, g2.x, g2.y, g2.z, g2.yaw_deg, g2.side)
    assert "yaw_convention" in raw and raw["course_name"] == "t"
    assert set(raw["gates"][0]) == {"name", "x", "y", "z", "yaw_deg", "side"}

    # hand-written dict in the exact format spawn_test_gates.main() writes
    manifest = {
        "yaw_convention": "yaw_deg=0 means gate +Z (flight direction) is world +X; ...",
        "gates": [{"name": "gate_0", "x": 6.0, "y": -0.5, "z": 1.8, "yaw_deg": 12.0, "side": 1.5},
                  {"name": "gate_1", "x": 9.0, "y": 1.0, "z": 2.0, "yaw_deg": -8.0, "side": 1.5}],
        "camera_pose_usd": {"position": [0, 0, 1], "R_world_cam_usd": [[1, 0, 0]] * 3,
                            "hfov_deg": 60.0, "vfov_deg": 36.0, "resolution": [1280, 720]},
        "dropped": [],
    }
    c3 = Course.from_dict(manifest)
    assert len(c3) == 2 and c3.gates[1].yaw_deg == -8.0 and c3.gates[0].z == 1.8
    assert c3.gates[0].side == 1.5
    print("  JSON round trip + spawn_test_gates manifest loads  OK")


def test_fixed_tracks_on_disk():
    for name, make in FIXED_TRACKS.items():
        loaded = Course.load(os.path.join(TRACKS_DIR, f"{name}.json"))
        ref = make()
        assert len(loaded) == len(ref)
        for a, b in zip(loaded.gates, ref.gates):
            assert abs(a.x - b.x) < 1e-9 and abs(a.yaw_deg - b.yaw_deg) < 1e-9
    s3 = Course.load(os.path.join(TRACKS_DIR, "straight3.json"))
    assert [g.x for g in s3.gates] == [8.0, 16.0, 24.0] and all(g.z == 2.5 for g in s3.gates)
    print("  rl/tracks/*.json match the generators  OK")


def test_random_course_constraints():
    for seed in range(60):
        n = 5 + seed % 8
        c = random_course(np.random.default_rng(seed), n)
        P = np.array([g.center for g in c.gates])
        assert len(c) == n
        assert (P[:, 2] >= 1.5 - 1e-9).all() and (P[:, 2] <= 4.0 + 1e-9).all()
        legs = P[1:, :2] - P[:-1, :2]
        d = np.linalg.norm(legs, axis=1)
        assert (d >= 5.0 - 1e-9).all() and (d <= 12.0 + 1e-9).all(), (seed, d)
        hd = np.arctan2(legs[:, 1], legs[:, 0])
        turn = np.degrees(np.abs(np.angle(np.exp(1j * (hd[1:] - hd[:-1])))))
        assert (turn <= 60.0 + 1e-6).all(), (seed, turn)
        # gate yaw ~ direction of travel (within 30 deg of each adjacent leg)
        for i, g in enumerate(c.gates):
            yaw = math.radians(g.yaw_deg)
            for j in ([i - 1] if i > 0 else []) + ([i] if i < n - 1 else []):
                err = abs(np.angle(np.exp(1j * (yaw - hd[j]))))
                assert math.degrees(err) <= 30.0 + 1e-6, (seed, i, j, math.degrees(err))
        D = np.linalg.norm(P[:, None] - P[None], axis=-1) + np.eye(n) * 99
        assert D.min() >= 3.0 - 1e-9, (seed, D.min())
    print("  random_course spacing/heading/height/overlap over 60 seeds  OK")


if __name__ == "__main__":
    print("test_T_world_gate_matches_spawn");   test_T_world_gate_matches_spawn()
    print("test_gate_axes_hand_vectors");       test_gate_axes_hand_vectors()
    print("test_crossings");                    test_crossings()
    print("test_json_roundtrip_and_manifest_format"); test_json_roundtrip_and_manifest_format()
    print("test_fixed_tracks_on_disk");         test_fixed_tracks_on_disk()
    print("test_random_course_constraints");    test_random_course_constraints()
    print("\nAll tests passed.")
