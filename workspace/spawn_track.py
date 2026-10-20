"""
Spawns a whole race track (rl/tracks/*.json) in Isaac Sim as solid-magenta gates
and writes /workspace/track_gates.json so the vision tools can read the ground
truth later.

RUN (Isaac Sim Script Editor, after spawn_example.py; the drone may be anywhere):
    import os; os.environ["SPAWN_TRACK"] = "straight3"
    exec(open("/workspace/scripts/spawn_track.py").read())
SPAWN_TRACK is a path to a track json, or a bare name resolved to
/workspace/race/rl/tracks/<name>.json. It is a ONE-SHOT environment variable
(popped here), for the same reason spawn_test_gates.py uses one: a plain
Script Editor variable was not visible to main() on 2026-10-01. Unset = FATAL
(no silent default track).

WHAT IS REUSED, NOT COPIED: the gate geometry (four bars, GATE_SIDE x
GATE_SIDE opening, BAR_THICKNESS), the magenta material and the fallback light
are spawn_test_gates's own `_build_gate`, `_make_material`, `_ensure_light`,
imported with importlib from this file's directory (falling back to
/workspace/scripts). Importing it under another name means its
`if __name__ == "__main__"` block does NOT run. Those helpers read the module
global ROOT_PATH when they build prim paths, so main() temporarily points it at
TRACK_ROOT (/World/RaceTrack, distinct from spawn_test_gates's
/World/TestGates, so both can coexist) and restores it afterwards. The root
is cleared on every rerun.

CONVENTIONS: gate x, y, z in world ENU (= Pegasus world), yaw_deg CCW about +z,
yaw 0 = flight direction world +X, gate frame +X right / +Y down / +Z flight
direction (the manifest's yaw_convention text says the same).

COLLIDERS: none. spawn_test_gates._add_bar defines plain UsdGeom.Cube prims with
a scale and a material and applies no UsdPhysics.CollisionAPI / RigidBodyAPI,
so the Iris passes through the gates and a gate "collision" exists only in
rl/course.check_gate. That is deliberate (do not add colliders here): the
runner scores gates geometrically.

Everything except main() is pure Python/numpy and importable without Isaac.
"""

import importlib.util
import json
import os
import sys

TRACK_ROOT = "/World/RaceTrack"
MANIFEST_PATH = "/workspace/track_gates.json"
TRACKS_DIR_CLUSTER = "/workspace/race/rl/tracks"
FALLBACK_SCRIPT_DIRS = ("/workspace/scripts",)


def _here():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:       # Script Editor exec() has no __file__
        return None


def load_spawn_test_gates():
    """Import spawn_test_gates.py as a module (its main() does not run)."""
    dirs = [d for d in [_here(), *FALLBACK_SCRIPT_DIRS] if d]
    for d in dirs:
        path = os.path.join(d, "spawn_test_gates.py")
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("spawn_test_gates", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["spawn_test_gates"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise RuntimeError(f"spawn_test_gates.py not found in {dirs}")


def resolve_track(value):
    """Path to a track json from SPAWN_TRACK: an existing path, else a bare name
    under the cluster tracks dir, else (local runs) <repo>/rl/tracks."""
    if os.path.isfile(value):
        return value
    cands = [os.path.join(TRACKS_DIR_CLUSTER, value + ".json")]
    here = _here()
    if here:
        cands.append(os.path.join(os.path.dirname(here), "rl", "tracks", value + ".json"))
    for c in cands:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"SPAWN_TRACK={value!r}: not a file and not found as {cands}")


def track_gate_specs(course_dict):
    """Track dict (rl/course.Course.to_dict format) -> list of gate dicts with
    name, x, y, z, yaw_deg, side, bar -- the spec format _build_gate takes."""
    specs = []
    for g in course_dict["gates"]:
        specs.append({"name": g["name"], "x": float(g["x"]), "y": float(g["y"]),
                      "z": float(g["z"]), "yaw_deg": float(g["yaw_deg"]),
                      "side": float(g.get("side", 1.5)), "bar": float(g.get("bar", 0.10))})
    return specs


def build_manifest(course_dict, specs, yaw_convention):
    """Same schema spawn_test_gates writes (yaw_convention, gates[name, x, y, z,
    yaw_deg, side]) plus course_name; no camera_pose_usd (a track has none)."""
    return {
        "yaw_convention": yaw_convention,
        "course_name": course_dict.get("course_name", "track"),
        "gates": [{"name": s["name"], "x": s["x"], "y": s["y"], "z": s["z"],
                   "yaw_deg": s["yaw_deg"], "side": s["side"]} for s in specs],
    }


def main():
    value = os.environ.pop("SPAWN_TRACK", None)
    if not value:
        msg = ("[spawn_track] FATAL: SPAWN_TRACK is not set. Run "
               "os.environ['SPAWN_TRACK'] = 'straight3' (or a path) first.")
        print(msg)
        raise RuntimeError(msg)
    sgt = load_spawn_test_gates()
    path = resolve_track(value)
    with open(path) as f:
        course_dict = json.load(f)
    specs = track_gate_specs(course_dict)
    for s in specs:
        # the shared builder uses the module constants, not the spec's side/bar
        if abs(s["side"] - sgt.GATE_SIDE) > 1e-9 or abs(s["bar"] - sgt.BAR_THICKNESS) > 1e-9:
            msg = (f"[spawn_track] FATAL: {s['name']} has side/bar {s['side']}/{s['bar']} but "
                   f"spawn_test_gates builds {sgt.GATE_SIDE}/{sgt.BAR_THICKNESS}")
            print(msg)
            raise RuntimeError(msg)

    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    old_root = sgt.ROOT_PATH
    sgt.ROOT_PATH = TRACK_ROOT          # the shared helpers build paths from this global
    try:
        sgt._clear_root(stage)
        UsdGeom.Xform.Define(stage, TRACK_ROOT)
        material = sgt._make_material(stage)
        for s in specs:
            sgt._build_gate(stage, s, material)
        sgt._ensure_light(stage)
    finally:
        sgt.ROOT_PATH = old_root

    manifest = build_manifest(course_dict, specs,
                              course_dict.get("yaw_convention", ""))
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[spawn_track] built {len(specs)} gate(s) of {manifest['course_name']} under "
          f"{TRACK_ROOT} from {path}; wrote {MANIFEST_PATH}")


# Isaac's Script Editor runs pasted scripts with __name__ == "__main__" (see the
# note at the bottom of spawn_test_gates.py); a plain import must not spawn.
if __name__ == "__main__":
    main()
