"""
Spawns solid-magenta test gates in Isaac Sim for vision/eval_isaac_gates.py
to detect, and writes their ground-truth poses (plus the camera pose used to
place them) to ~/test_gates.json.

RUN ORDER (this changed -- read before running):
    1. (Isaac Sim script editor) spawn_example.py.
    2. Press Play, and take off / hover if you want the drone airborne --
       under physics the drone otherwise just falls to the ground. The
       camera's live pose is read directly off the USD stage (see
       read_camera() below), so gates land in front of wherever the camera
       ACTUALLY looks, whatever that turns out to be.
    3. (Isaac Sim script editor, same session) spawn_test_gates.py  <- this file.
    4. (ROS 2 sourced) python3 vision/eval_isaac_gates.py --gates ~/test_gates.json

    If the drone moves after step 3 (gates are spawned once, relative to
    the camera pose at that instant), eval_isaac_gates.py's TF-based camera
    pose still works fine -- gate detection and PnP don't care where the
    camera currently is, only where it was when the picture was taken.
    What can break is the LAYOUT: the nested chain built at spawn time (see
    below) assumed a particular camera pose, so if the drone has since
    drifted far enough, farther gates may no longer sit inside nearer ones'
    openings in the new view.

WHY THE LAYOUT IS A NESTED CHAIN, AND WHY IT'S COMPUTED FROM THE LIVE
CAMERA, NOT HARDCODED:
    Pegasus's ROS2CameraGraph camera (an `omni.isaac.sensor.Camera` with no
    focal/aperture explicitly set -- see the module read_camera() reads
    from) has an FOV this script cannot know in advance. A fixed grid of
    gates picked without knowing the FOV can and did (see git history)
    overlap in the image: two same-colored gate frames that cross on-screen
    merge into one contour and break the hole-detector in
    vision/color_gate_detector.py. The fix is to lay gates out as a nested
    chain along the camera's actual optical axis -- each farther gate seen
    THROUGH the nearer one's opening -- which color_gate_detector.py's
    RETR_CCOMP-based contour search already handles correctly (a contour
    inside a hole goes back to being a top-level contour; see
    vision/test_color_gate_detector.py's test_nested_gate_through_opening).
    read_camera() reads the camera's live world pose and FOV straight off
    the USD stage so the layout in GATE_SPECS (angular offsets, not fixed
    world coordinates) actually lands where the camera can resolve it,
    whatever the FOV or the drone's current pose happens to be.

YAW CONVENTION (also written into the manifest's "yaw_convention" field --
eval_isaac_gates.py's T_world_gate_from_spec() must match this exactly):
    yaw_deg = 0 means the gate's flight direction (gate +Z, in
    gate_pose.py's gate-model frame) points along world +X. Positive yaw
    rotates the gate counter-clockwise about world +Z (right-hand rule),
    i.e. the same sense as a standard Rz(yaw) rotation matrix.

    At yaw=0 the gate frame in world is: gate +X = world -Y, gate +Y =
    world -Z, gate +Z = world +X. Check this is right-handed:
    (-Y) x (-Z) = Y x Z = +X. Matches. This mirrors gate_pose.py's gate
    model frame (+X right, +Y DOWN, +Z flight direction) exactly.

MODULE STRUCTURE: everything through plan_layout() below is pure
numpy/math geometry with NO omni/pxr imports, so it is importable and unit
testable on a machine with no Isaac Sim installed at all -- see
vision/test_gate_layout.py, which imports this module directly (the ONE
place in this repo that reaches across the vision//workspace boundary,
because this script has to live in workspace/ to run inside Isaac's own
interpreter, but its pure geometry is worth testing like anything else).
Only read_camera() and main() (and the small Usd-building helpers they use)
touch omni.usd / pxr, and those imports are all lazy, inside the functions
that need them -- `import omni.usd`/`from pxr import ...` never happens at
module scope.
"""

import json
import math
import os

import numpy as np

ROOT_PATH = "/World/TestGates"
MANIFEST_PATH = os.path.expanduser("~/test_gates.json")

GATE_SIDE = 1.5           # opening, inner edge to inner edge, meters
BAR_THICKNESS = 0.10       # meters
GATE_COLOR = (1.0, 0.0, 1.0)  # magenta, linear RGB

# Mirrored from spawn_example.py's ROS2CameraGraph config
# ("resolution": [320, 240]) -- KEEP IN SYNC WITH IT. Used only to derive
# vfov from hfov (Isaac derives the vertical FOV from the render aspect
# ratio, not a separate verticalAperture attribute -- see read_camera()).
RESOLUTION = (320, 240)

CAMERA_PRIM_PATH = "/World/Iris/body/front_cam"

# Used only if the camera prim's focal length / horizontal aperture can't
# be read (missing or non-positive) -- see read_camera().
FALLBACK_HFOV_DEG = 60.0

# Ranges follow a ~1.4x ratio so each farther gate's outer square (side +
# 2*bar_thickness) fits inside the nearer gate's opening (side) with
# roughly a degree of angular margin, for a wide range of plausible FOVs --
# see plan_layout(), which drops whichever gate is farthest out of a
# problem if a given camera's actual FOV is narrower than expected anyway.
# (range_m, az_deg, el_deg, yaw_rel_deg) -- az/el are camera-relative
# angular offsets (az positive = toward camera right, el positive = toward
# camera up); yaw_rel_deg is added to the gate's heading yaw (see
# gate_world_pose). All on-axis (az=el=0) here, which is what makes them
# nest: only yaw varies, to still exercise the normal-direction estimate.
GATE_SPECS = [
    (3.0, 0.0, 0.0, 0.0),
    (4.2, 0.0, 0.0, 20.0),
    (5.9, 0.0, 0.0, 0.0),
    (8.2, 0.0, 0.0, -25.0),
    (11.5, 0.0, 0.0, 0.0),
]


# ---------------------------------------------------------------------------
# Pure geometry -- no omni/pxr imports below this point until read_camera().
# ---------------------------------------------------------------------------

def _hfov_vfov_from_aspect(hfov_deg: float, width: int, height: int):
    """vfov implied by a given hfov and the render aspect ratio (Isaac
    derives the vertical extent this way, not from a separate
    verticalAperture attribute -- see read_camera())."""
    vfov_deg = math.degrees(2.0 * math.atan(math.tan(math.radians(hfov_deg) / 2.0)
                                            * (height / width)))
    return hfov_deg, vfov_deg


def gate_world_pose(spec, cam: dict, name: str = "gate") -> dict:
    """
    (range_m, az_deg, el_deg, yaw_rel_deg) + a camera dict (with "position",
    "right", "up", "fwd" world-frame vectors) -> {"name", "x", "y", "z",
    "yaw_deg"} in absolute world coordinates, using the SAME yaw convention
    as eval_isaac_gates.T_world_gate_from_spec (see module docstring).

    az/el are angular offsets from the camera's optical axis (az positive
    toward camera right, el positive toward camera up); yaw_rel_deg is
    added on top of the gate's "face the camera" heading yaw, so yaw_rel=0
    always produces a gate pointed straight back down its own line of
    sight regardless of where az/el put it.
    """
    range_m, az_deg, el_deg, yaw_rel_deg = spec
    az, el = math.radians(az_deg), math.radians(el_deg)
    position = np.asarray(cam["position"], dtype=float)
    right = np.asarray(cam["right"], dtype=float)
    up = np.asarray(cam["up"], dtype=float)
    fwd = np.asarray(cam["fwd"], dtype=float)

    direction = (math.cos(el) * math.cos(az) * fwd
                + math.cos(el) * math.sin(az) * right
                + math.sin(el) * up)
    center = position + range_m * direction

    h = np.array([fwd[0], fwd[1], 0.0])
    if np.linalg.norm(h) < 1e-6:
        # Camera looking (near-)straight up or down: fall back to a
        # horizontal direction derived from "right" instead of a
        # degenerate/undefined heading.
        h = np.cross(right, np.array([0.0, 0.0, 1.0]))
    h = h / np.linalg.norm(h)
    yaw_deg = math.degrees(math.atan2(h[1], h[0])) + yaw_rel_deg

    return {"name": name, "x": float(center[0]), "y": float(center[1]),
           "z": float(center[2]), "yaw_deg": float(yaw_deg)}


def clamp_to_ground(gate: dict, gate_side: float = GATE_SIDE,
                    bar_thickness: float = BAR_THICKNESS,
                    min_clearance: float = 0.05):
    """
    If the gate's bottom bar would end up less than min_clearance above
    z=0, raise the gate's z just enough to clear it. Returns (gate, note)
    where note is a human-readable string if a clamp happened, else None.
    Does not mutate the input dict.
    """
    half_outer = gate_side / 2.0 + bar_thickness
    bottom = gate["z"] - half_outer
    if bottom >= min_clearance:
        return gate, None
    new_z = min_clearance + half_outer
    note = (f"{gate['name']}: raised z from {gate['z']:.2f} m to {new_z:.2f} m "
           f"to keep the bottom bar {min_clearance:.2f} m above the ground")
    clamped = dict(gate)
    clamped["z"] = new_z
    return clamped, note


def _T_world_gate(x: float, y: float, z: float, yaw_deg: float) -> np.ndarray:
    """
    Same construction as eval_isaac_gates.T_world_gate_from_spec, DUPLICATED
    rather than imported -- this file has to run inside Isaac Sim's own
    interpreter (which does not have vision/ on its path), and its pure
    geometry has to be testable without vision/ installed either. If this
    convention ever changes, change it in both places (each file's
    docstring names the other).
    """
    R_yaw0 = np.array([[0.0, 0.0, 1.0],
                       [-1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0]])
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    R = Rz @ R_yaw0
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def _gate_local_corners(half: float) -> np.ndarray:
    """4x3 corners (TL, TR, BR, BL) of a half-extent square in the gate's
    local frame (+X right, +Y down, +Z flight direction, z=0 plane) --
    matches gate_pose.gate_model_points()'s layout, duplicated for the same
    reason as _T_world_gate."""
    h = half
    return np.array([[-h, -h, 0.0], [h, -h, 0.0], [h, h, 0.0], [-h, h, 0.0]])


def _transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.atleast_2d(pts)
    return (T[:3, :3] @ p.T).T + T[:3, 3]


def _az_el(vec_from_cam: np.ndarray, cam: dict):
    """Camera-relative (az_deg, el_deg) of a world-frame direction vector
    (need not be unit length) -- az about cam["up"] via cam["right"], el
    about cam["right"] via cam["up"], both measured from cam["fwd"]."""
    d = np.asarray(vec_from_cam, dtype=float)
    fwd = np.asarray(cam["fwd"], dtype=float)
    right = np.asarray(cam["right"], dtype=float)
    up = np.asarray(cam["up"], dtype=float)
    az = math.degrees(math.atan2(float(np.dot(d, right)), float(np.dot(d, fwd))))
    el = math.degrees(math.atan2(float(np.dot(d, up)), float(np.dot(d, fwd))))
    return az, el


def _gate_bboxes(gate: dict, cam: dict, gate_side: float, bar_thickness: float):
    """(outer_bbox, opening_bbox), each (min_az, max_az, min_el, max_el) in
    degrees, of a gate's outer silhouette and its opening, as seen from
    cam."""
    T = _T_world_gate(gate["x"], gate["y"], gate["z"], gate["yaw_deg"])
    cam_pos = np.asarray(cam["position"], dtype=float)

    def _bbox(half):
        world = _transform_points(T, _gate_local_corners(half))
        azels = [_az_el(p - cam_pos, cam) for p in world]
        azs = [a for a, _e in azels]
        els = [e for _a, e in azels]
        return (min(azs), max(azs), min(els), max(els))

    return _bbox(gate_side / 2.0 + bar_thickness), _bbox(gate_side / 2.0)


def _check_layout_detailed(gates: list, cam: dict, gate_side: float = GATE_SIDE,
                           bar_thickness: float = BAR_THICKNESS,
                           margin_deg: float = 0.3):
    """check_layout()'s implementation, returning (message, [gate_names])
    pairs so plan_layout() knows which gate(s) each problem implicates."""
    cam_pos = np.asarray(cam["position"], dtype=float)
    info = {}
    for g in gates:
        outer, opening = _gate_bboxes(g, cam, gate_side, bar_thickness)
        rng = float(np.linalg.norm(np.array([g["x"], g["y"], g["z"]]) - cam_pos))
        info[g["name"]] = {"outer": outer, "opening": opening, "range": rng}

    problems = []
    hfov, vfov = cam["hfov_deg"], cam["vfov_deg"]
    limit_az, limit_el = hfov / 2.0 - margin_deg, vfov / 2.0 - margin_deg
    for g in gates:
        o = info[g["name"]]["outer"]
        if o[0] < -limit_az or o[1] > limit_az or o[2] < -limit_el or o[3] > limit_el:
            problems.append((
                f"{g['name']}: out of FOV (outer az [{o[0]:.1f}, {o[1]:.1f}] deg, "
                f"el [{o[2]:.1f}, {o[3]:.1f}] deg, vs limit az +-{limit_az:.1f}, "
                f"el +-{limit_el:.1f} deg)",
                [g["name"]]))

    names = [g["name"] for g in gates]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ni, nj = names[i], names[j]
            oi, oj = info[ni]["outer"], info[nj]["outer"]
            disjoint = (oi[1] < oj[0] - margin_deg or oj[1] < oi[0] - margin_deg
                       or oi[3] < oj[2] - margin_deg or oj[3] < oi[2] - margin_deg)
            if disjoint:
                continue

            near, far = (ni, nj) if info[ni]["range"] <= info[nj]["range"] else (nj, ni)
            near_opening = info[near]["opening"]
            far_outer = info[far]["outer"]
            nested = (near_opening[0] + margin_deg <= far_outer[0]
                     and far_outer[1] <= near_opening[1] - margin_deg
                     and near_opening[2] + margin_deg <= far_outer[2]
                     and far_outer[3] <= near_opening[3] - margin_deg)
            if not nested:
                problems.append((
                    f"{ni} x {nj}: overlap -- bars will cross in the image and merge "
                    f"in the color mask", [ni, nj]))

    return problems


def check_layout(gates: list, cam: dict, gate_side: float = GATE_SIDE,
                 bar_thickness: float = BAR_THICKNESS,
                 margin_deg: float = 0.3) -> list:
    """
    Human-readable list of layout problems for `gates` as seen from `cam`:
    any gate whose outer silhouette falls outside the camera's FOV (minus
    margin_deg), and any pair of gates that is neither disjoint nor cleanly
    nested (one entirely inside the other's opening, by margin_deg) in the
    image -- same-color bars that cross on-screen merge into one contour
    and break color_gate_detector.py's hole search. Empty list = no
    problems.
    """
    return [msg for msg, _names in _check_layout_detailed(
        gates, cam, gate_side, bar_thickness, margin_deg)]


def camera_low_warning(cam: dict, min_z: float = 1.0):
    """Not a layout problem (doesn't trigger a drop) -- just a heads-up
    that the drone may still be on the ground."""
    z = float(np.asarray(cam["position"], dtype=float)[2])
    if z < min_z:
        return (f"drone appears to be on the ground (camera z={z:.2f} m < {min_z:.1f} m) "
               f"-- take off / hover before spawning gates, or expect a floor-dominated view")
    return None


def plan_layout(specs: list, cam: dict, gate_side: float = GATE_SIDE,
                bar_thickness: float = BAR_THICKNESS, margin_deg: float = 0.3,
                ground_clearance: float = 0.05) -> dict:
    """
    Build gates from `specs` relative to `cam`, ground-clamp each, then
    repeatedly check_layout() and drop the FARTHEST gate involved in any
    problem until none remain (or nothing is left). Prints the final az/el
    table for every kept gate and a line for every drop.

    Returns {"gates": [...surviving gate dicts...], "dropped": ["dropped
    <name>: <reason>", ...], "clamp_notes": [...]}.
    """
    gates = []
    clamp_notes = []
    for i, spec in enumerate(specs):
        g = gate_world_pose(spec, cam, name=f"gate_{i}")
        g, note = clamp_to_ground(g, gate_side, bar_thickness, ground_clearance)
        if note:
            clamp_notes.append(note)
        gates.append(g)

    cam_pos = np.asarray(cam["position"], dtype=float)
    dropped = []
    while gates:
        detailed = _check_layout_detailed(gates, cam, gate_side, bar_thickness, margin_deg)
        if not detailed:
            break
        involved = set()
        for _msg, names in detailed:
            involved.update(names)
        if not involved:
            break

        def _range(name):
            g = next(x for x in gates if x["name"] == name)
            return float(np.linalg.norm(np.array([g["x"], g["y"], g["z"]]) - cam_pos))

        farthest = max(involved, key=_range)
        reason = next((m for m, names in detailed if farthest in names), "layout problem")
        dropped.append(f"dropped {farthest}: {reason}")
        gates = [g for g in gates if g["name"] != farthest]

    print("\n[spawn_test_gates] layout (az/el relative to camera, range in meters):")
    for g in gates:
        vec = np.array([g["x"], g["y"], g["z"]]) - cam_pos
        az, el = _az_el(vec, cam)
        rng = float(np.linalg.norm(vec))
        print(f"  {g['name']}: range={rng:.2f} m az={az:+.1f} deg el={el:+.1f} deg "
             f"yaw={g['yaw_deg']:+.1f} deg")
    for note in dropped:
        print(f"  {note}")
    for note in clamp_notes:
        print(f"  {note}")

    return {"gates": gates, "dropped": dropped, "clamp_notes": clamp_notes}


# ---------------------------------------------------------------------------
# Everything below touches omni/pxr -- all imports lazy, inside the
# functions that need them, so everything above remains importable (and
# unit tested, see vision/test_gate_layout.py) without Isaac Sim installed.
# ---------------------------------------------------------------------------

def read_camera(stage) -> dict:
    """
    Live camera world pose + FOV, read straight off the USD stage, so
    plan_layout() can place gates in front of wherever the camera actually
    looks (see module docstring for why this can't be hardcoded).

    Returns {"position", "right", "up", "fwd" (world-frame numpy vectors,
    USD camera convention: +X right, +Y up, -Z forward), "R_world_cam_usd"
    (3x3, columns = world-frame right/up/back -- for the manifest),
    "hfov_deg", "vfov_deg"}.
    """
    from pxr import UsdGeom, Usd, Gf

    prim = stage.GetPrimAtPath(CAMERA_PRIM_PATH)
    if not prim.IsValid():
        raise RuntimeError(f"no prim at {CAMERA_PRIM_PATH} -- run spawn_example.py first "
                           f"(and press Play)")

    xf = UsdGeom.Xformable(prim)
    M = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    position = np.array(M.ExtractTranslation(), dtype=float)

    def _world_dir(local):
        d = M.TransformDir(Gf.Vec3d(*local))
        d = np.array([d[0], d[1], d[2]], dtype=float)
        n = np.linalg.norm(d)
        return d / n if n > 1e-9 else d

    right = _world_dir((1.0, 0.0, 0.0))
    up = _world_dir((0.0, 1.0, 0.0))
    back = _world_dir((0.0, 0.0, 1.0))   # USD camera +Z is backward
    fwd = -back

    cam_schema = UsdGeom.Camera(prim)
    focal = h_ap = None
    try:
        focal = cam_schema.GetFocalLengthAttr().Get()
        h_ap = cam_schema.GetHorizontalApertureAttr().Get()
    except Exception:
        pass

    width, height = RESOLUTION
    if focal and h_ap and focal > 0 and h_ap > 0:
        hfov_deg = math.degrees(2.0 * math.atan(h_ap / (2.0 * focal)))
        vfov_deg = math.degrees(2.0 * math.atan(h_ap * (height / width) / (2.0 * focal)))
    else:
        print(f"[spawn_test_gates] WARNING: camera focal length / horizontal aperture "
             f"missing or non-positive (focal={focal}, h_ap={h_ap}) -- falling back to "
             f"FALLBACK_HFOV_DEG={FALLBACK_HFOV_DEG}")
        hfov_deg, vfov_deg = _hfov_vfov_from_aspect(FALLBACK_HFOV_DEG, width, height)

    cam = {
        "position": position, "right": right, "up": up, "fwd": fwd,
        "R_world_cam_usd": np.column_stack([right, up, back]),
        "hfov_deg": hfov_deg, "vfov_deg": vfov_deg,
    }
    print(f"[spawn_test_gates] camera: position={position.tolist()}, forward={fwd.tolist()}, "
         f"hfov={hfov_deg:.1f} deg, vfov={vfov_deg:.1f} deg")
    return cam


def _clear_root(stage):
    if stage.GetPrimAtPath(ROOT_PATH).IsValid():
        stage.RemovePrim(ROOT_PATH)


def _make_material(stage):
    from pxr import UsdShade, Sdf, Gf

    mat_path = f"{ROOT_PATH}/Looks/GateMat"
    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*GATE_COLOR))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _has_light(stage) -> bool:
    from pxr import UsdLux

    for prim in stage.Traverse():
        try:
            if prim.IsA(UsdLux.DistantLight) or prim.IsA(UsdLux.DomeLight):
                return True
        except Exception:
            # Lux API can differ across Isaac/USD versions -- if IsA() isn't
            # available for some reason, don't block gate spawning over it.
            pass
    return False


def _ensure_light(stage):
    from pxr import UsdLux

    if _has_light(stage):
        return
    light_path = f"{ROOT_PATH}/Light"
    try:
        light = UsdLux.DistantLight.Define(stage, light_path)
        light.CreateIntensityAttr(3000.0)
    except Exception as exc:
        print(f"[spawn_test_gates] WARNING: could not create fallback DistantLight "
             f"({exc}); gates may render black")


def _add_bar(stage, gate_path: str, name: str, material, center: tuple, size: tuple):
    """One bar of the gate frame: a unit Cube scaled to `size` (a 3-tuple of
    full extents in meters) and translated to `center`, in the gate's local
    (Xform) frame."""
    from pxr import UsdGeom, Gf, UsdShade

    cube_path = f"{gate_path}/{name}"
    cube = UsdGeom.Cube.Define(stage, cube_path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddScaleOp().Set(Gf.Vec3d(*size))

    cube.CreateDisplayColorAttr([Gf.Vec3f(*GATE_COLOR)])
    UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(material)
    return cube


def _build_gate(stage, spec: dict, material):
    """
    Build one gate as ROOT_PATH/<name>: an Xform at (x,y,z) rotated rotateZ
    by yaw_deg, containing four Cube bars arranged so the OPENING is exactly
    GATE_SIDE x GATE_SIDE. At yaw=0 the gate's horizontal axis is world Y
    and its vertical axis is world Z (see module docstring); bars are
    BAR_THICKNESS deep along world X at yaw=0, and rotateZ carries that
    local frame to any other yaw.
    """
    from pxr import UsdGeom, Gf

    gate_path = f"{ROOT_PATH}/{spec['name']}"
    xform = UsdGeom.Xform.Define(stage, gate_path)
    xf = UsdGeom.Xformable(xform.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(spec["x"], spec["y"], spec["z"]))
    xf.AddRotateZOp().Set(float(spec["yaw_deg"]))

    half_outer = GATE_SIDE / 2.0 + BAR_THICKNESS / 2.0
    outer_len = GATE_SIDE + 2.0 * BAR_THICKNESS

    _add_bar(stage, gate_path, "bar_top", material,
             center=(0.0, 0.0, half_outer),
             size=(BAR_THICKNESS, outer_len, BAR_THICKNESS))
    _add_bar(stage, gate_path, "bar_bottom", material,
             center=(0.0, 0.0, -half_outer),
             size=(BAR_THICKNESS, outer_len, BAR_THICKNESS))
    _add_bar(stage, gate_path, "bar_left", material,
             center=(0.0, -half_outer, 0.0),
             size=(BAR_THICKNESS, BAR_THICKNESS, GATE_SIDE))
    _add_bar(stage, gate_path, "bar_right", material,
             center=(0.0, half_outer, 0.0),
             size=(BAR_THICKNESS, BAR_THICKNESS, GATE_SIDE))


def main():
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()

    try:
        cam = read_camera(stage)
    except RuntimeError as exc:
        print(f"[spawn_test_gates] FATAL: {exc}")
        return

    warn = camera_low_warning(cam)
    if warn:
        print(f"[spawn_test_gates] WARNING: {warn}")

    plan = plan_layout(GATE_SPECS, cam, GATE_SIDE, BAR_THICKNESS)
    gates = plan["gates"]
    if not gates:
        print("[spawn_test_gates] FATAL: no gates survived layout planning -- "
             "check the camera's pose/FOV printed above")
        return

    _clear_root(stage)
    UsdGeom.Xform.Define(stage, ROOT_PATH)

    material = _make_material(stage)
    for g in gates:
        _build_gate(stage, g, material)

    _ensure_light(stage)

    manifest = {
        "yaw_convention": (
            "yaw_deg=0 means gate +Z (flight direction) is world +X; positive "
            "yaw rotates the gate counter-clockwise about world +Z (right-hand "
            "rule); at yaw=0, gate +X = world -Y and gate +Y = world -Z."
        ),
        "gates": [
            {"name": g["name"], "x": g["x"], "y": g["y"], "z": g["z"],
             "yaw_deg": g["yaw_deg"], "side": GATE_SIDE}
            for g in gates
        ],
        "camera_pose_usd": {
            "position": cam["position"].tolist(),
            "R_world_cam_usd": cam["R_world_cam_usd"].tolist(),
            "hfov_deg": cam["hfov_deg"],
            "vfov_deg": cam["vfov_deg"],
            "resolution": list(RESOLUTION),
        },
        "dropped": plan["dropped"],
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[spawn_test_gates] wrote {MANIFEST_PATH} ({len(gates)} gate(s) kept, "
         f"{len(plan['dropped'])} dropped)")
    print(f"[spawn_test_gates] next: on the machine with ROS 2 sourced, run")
    print(f"    python3 eval_isaac_gates.py --gates {MANIFEST_PATH}")


# Guarded (not called unconditionally like spawn_example.py's
# asyncio.ensure_future(main())) because this module must also be plain
# `import`-able with no omni/pxr present at all, for the pure-geometry
# tests in vision/test_gate_layout.py. Isaac Sim's script editor executes
# pasted scripts with __name__ == "__main__", so pasting this file in still
# runs main() exactly as before.
if __name__ == "__main__":
    main()
