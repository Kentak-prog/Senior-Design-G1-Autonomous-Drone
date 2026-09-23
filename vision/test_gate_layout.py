"""Self-tests. Run: python3 test_gate_layout.py

Offline only -- numpy, no omni/pxr, no ROS, no sim. Exercises the pure
geometry in workspace/spawn_test_gates.py: gate_world_pose's camera-relative
placement, check_layout's FOV/overlap/nesting detection, and plan_layout's
drop-the-farthest-offender loop.

THIS IS THE ONE PLACE IN THIS REPO THAT REACHES ACROSS THE vision//workspace
BOUNDARY: spawn_test_gates.py has to live in workspace/ so it can be pasted
into Isaac Sim's own script editor (which does not have vision/ on its
path), but its pure-geometry functions (no omni/pxr at module scope -- see
that file's docstring) are worth unit testing like anything else in this
directory. sys.path is extended here, once, for exactly that reason.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "workspace"))

import numpy as np

from spawn_test_gates import (gate_world_pose, check_layout, plan_layout,
                              GATE_SPECS, GATE_SIDE, BAR_THICKNESS,
                              _hfov_vfov_from_aspect, RESOLUTION)


def _axis_cam(position=(0.0, 0.0, 2.5), fwd=(1.0, 0.0, 0.0), hfov_deg=None):
    """A camera dict with right/up derived from a purely horizontal `fwd`
    (fine for these tests, which never point the camera up/down)."""
    fwd = np.asarray(fwd, dtype=float)
    fwd = fwd / np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)  # fwd x world-up: right-handed camera right
    right = right / np.linalg.norm(right)
    cam = {"position": np.asarray(position, dtype=float), "right": right,
          "up": up, "fwd": fwd}
    if hfov_deg is not None:
        cam["hfov_deg"], cam["vfov_deg"] = _hfov_vfov_from_aspect(hfov_deg, *RESOLUTION)
    return cam


# --- gate_world_pose ---------------------------------------------------

def test_gate_world_pose_axis_aligned_camera():
    """Camera at (0,0,2.5) looking along world +X: fwd=+X, right=-Y (check
    the helper below matches the brief's convention), up=+Z."""
    cam = _axis_cam(position=(0.0, 0.0, 2.5), fwd=(1.0, 0.0, 0.0))
    assert np.allclose(cam["right"], [0.0, -1.0, 0.0], atol=1e-9), cam["right"]
    assert np.allclose(cam["up"], [0.0, 0.0, 1.0], atol=1e-9)

    g = gate_world_pose((5.0, 0.0, 0.0, 0.0), cam)
    assert abs(g["x"] - 5.0) < 1e-9 and abs(g["y"]) < 1e-9 and abs(g["z"] - 2.5) < 1e-9, g
    assert abs(g["yaw_deg"] - 0.0) < 1e-9, g
    print(f"  camera fwd=+X, spec (5,0,0,0) -> {g}  OK")


def test_gate_world_pose_yawed_camera():
    """Camera yawed so fwd=+Y (world) -> yaw_deg must come out as 90."""
    cam = _axis_cam(position=(0.0, 0.0, 2.5), fwd=(0.0, 1.0, 0.0))
    g = gate_world_pose((5.0, 0.0, 0.0, 0.0), cam)
    assert abs(g["x"]) < 1e-9 and abs(g["y"] - 5.0) < 1e-9, g
    assert abs(g["yaw_deg"] - 90.0) < 1e-9, g
    print(f"  camera fwd=+Y (yawed 90 deg), spec (5,0,0,0) -> yaw_deg={g['yaw_deg']}  OK")


def test_gate_world_pose_elevation_raises_z():
    """el=+10 deg must raise the gate center's z by range*sin(10 deg)."""
    cam = _axis_cam(position=(0.0, 0.0, 2.5), fwd=(1.0, 0.0, 0.0))
    g = gate_world_pose((5.0, 0.0, 10.0, 0.0), cam)
    expected_z = 2.5 + 5.0 * np.sin(np.radians(10.0))
    assert abs(g["z"] - expected_z) < 1e-9, (g["z"], expected_z)
    print(f"  el=+10 deg: z rose by {g['z'] - 2.5:.4f} m, expected {5.0*np.sin(np.radians(10.0)):.4f} m  OK")


# --- check_layout --------------------------------------------------------

def test_check_layout_default_specs_no_problems_and_nested():
    """The default GATE_SPECS nested chain, seen through a generous 60 deg
    hfov camera, must have zero problems."""
    cam = _axis_cam(hfov_deg=60.0)
    gates = [gate_world_pose(spec, cam, name=f"gate_{i}") for i, spec in enumerate(GATE_SPECS)]
    problems = check_layout(gates, cam, GATE_SIDE, BAR_THICKNESS)
    assert problems == [], problems
    print(f"  default GATE_SPECS, 60 deg hfov: no problems ({len(gates)} nested gates)  OK")


def test_check_layout_old_layout_reports_overlaps():
    """The OLD (pre-revision) fixed-world-coordinate layout -- x in
    3/5/7/9/12, y in 0/1.3/-1.5/0.8/-0.6, z clamped to >=0.9, camera at
    (0.3,0,2.55) pitched 15 deg down along +X -- must report overlaps."""
    pitch = np.radians(15.0)
    fwd = np.array([np.cos(pitch), 0.0, -np.sin(pitch)])
    right = np.array([0.0, -1.0, 0.0])
    up = np.cross(right, fwd) * -1.0
    up = up / np.linalg.norm(up)
    hfov, vfov = _hfov_vfov_from_aspect(60.0, *RESOLUTION)
    cam = {"position": np.array([0.3, 0.0, 2.55]), "right": right, "up": up, "fwd": fwd,
          "hfov_deg": hfov, "vfov_deg": vfov}

    def old_z(x):
        return max(2.55 - x * np.tan(pitch), 0.9)

    old_specs = [(3.0, 0.0), (5.0, 1.3), (7.0, -1.5), (9.0, 0.8), (12.0, -0.6)]
    old_gates = [{"name": f"old_{i}", "x": x, "y": y, "z": old_z(x), "yaw_deg": 0.0}
                for i, (x, y) in enumerate(old_specs)]

    problems = check_layout(old_gates, cam, GATE_SIDE, BAR_THICKNESS)
    assert problems, "expected the old fixed layout to report overlaps, got none"
    print(f"  OLD layout: {len(problems)} problem(s) reported, as expected:")
    for p in problems:
        print(f"    {p}")


# --- plan_layout -----------------------------------------------------------

def test_plan_layout_narrow_fov_drops_close_gate_keeps_nested_remainder():
    """A narrow 24 deg hfov camera can't fit the 3 m gate_0 (it subtends
    too wide an angle up close) -- plan_layout must drop it and leave a
    remainder with no further problems."""
    cam = _axis_cam(hfov_deg=24.0)
    plan = plan_layout(GATE_SPECS, cam, GATE_SIDE, BAR_THICKNESS)
    kept_names = {g["name"] for g in plan["gates"]}
    dropped_names = {note.split(":")[0].removeprefix("dropped ") for note in plan["dropped"]}

    assert "gate_0" in dropped_names, plan["dropped"]
    assert "gate_0" not in kept_names
    assert plan["gates"], "expected at least one gate to survive"

    remaining_problems = check_layout(plan["gates"], cam, GATE_SIDE, BAR_THICKNESS)
    assert remaining_problems == [], remaining_problems
    print(f"  24 deg hfov: dropped {sorted(dropped_names)}, kept {sorted(kept_names)}, "
         f"remainder has no problems  OK")


if __name__ == "__main__":
    print("test_gate_world_pose_axis_aligned_camera")
    test_gate_world_pose_axis_aligned_camera()
    print("test_gate_world_pose_yawed_camera");        test_gate_world_pose_yawed_camera()
    print("test_gate_world_pose_elevation_raises_z");  test_gate_world_pose_elevation_raises_z()
    print("test_check_layout_default_specs_no_problems_and_nested")
    test_check_layout_default_specs_no_problems_and_nested()
    print("test_check_layout_old_layout_reports_overlaps")
    test_check_layout_old_layout_reports_overlaps()
    print("test_plan_layout_narrow_fov_drops_close_gate_keeps_nested_remainder")
    test_plan_layout_narrow_fov_drops_close_gate_keeps_nested_remainder()
    print("\nAll tests passed.")
