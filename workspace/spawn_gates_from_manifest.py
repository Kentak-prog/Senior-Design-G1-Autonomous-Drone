"""
Spawn gates at the ABSOLUTE poses in a saved manifest (instead of relative to
the live camera like spawn_test_gates.py), so a course can be reproduced
exactly -- e.g. the 3-gate course of 2026-10-23 after an Isaac restart or a
camera change.

Run in the Isaac Sim Script Editor (any time after the stage exists):
    exec(open("/workspace/Senior-Design-G1-Autonomous-Drone/workspace/spawn_gates_from_manifest.py").read())
Another manifest:
    import os; os.environ["GATES_MANIFEST"] = "/path/to/test_gates.json"; exec(open(...).read())
(an os.environ variable, not a Script Editor variable: see spawn_test_gates.main's note on SPAWN_LAYOUT.)

It uses spawn_test_gates.py's own builders (same magenta bars, same gate
convention), clears /World/TestGates first, and copies the manifest to
/workspace/test_gates.json so every other script reads the same course.
Safe to re-run.
"""

import json
import os
import shutil

_REPO = "/workspace/Senior-Design-G1-Autonomous-Drone"
_manifest = os.environ.pop(
    "GATES_MANIFEST", f"{_REPO}/Trajectory/isaac_results/3_gates/course/pod_test_gates.json")

_sg = {"__name__": "spawn_test_gates_lib"}   # import without running its main()
exec(open(f"{_REPO}/workspace/spawn_test_gates.py").read(), _sg)


def _spawn_from_manifest(path):
    import omni.usd
    from pxr import UsdGeom
    with open(path) as f:
        gates = json.load(f)["gates"]
    stage = omni.usd.get_context().get_stage()
    _sg["_clear_root"](stage)
    UsdGeom.Xform.Define(stage, _sg["ROOT_PATH"])
    material = _sg["_make_material"](stage)
    for g in gates:
        _sg["_build_gate"](stage, g, material)
        print(f"[spawn_gates_from_manifest] {g['name']}: ({g['x']:.3f}, {g['y']:.3f}, {g['z']:.3f}) "
              f"yaw {g['yaw_deg']:+.2f} deg")
    _sg["_ensure_light"](stage)
    if os.path.abspath(path) != os.path.abspath(_sg["MANIFEST_PATH"]):
        shutil.copyfile(path, _sg["MANIFEST_PATH"])
    print(f"[spawn_gates_from_manifest] {len(gates)} gate(s) from {path}; "
          f"{_sg['MANIFEST_PATH']} now holds the same course")


_spawn_from_manifest(_manifest)
