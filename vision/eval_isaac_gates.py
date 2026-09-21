"""
ROS 2 evaluator: settles gate-pose accuracy AND the `optical_frame` question
against a live Isaac Sim, in one run.

WHAT IT SETTLES:
    1. How accurate the perception pipeline actually is (position, range,
       lateral, and normal error) against sim ground truth, instead of the
       synthetic corner noise eval_pnp.py uses.
    2. Which value of `optical_frame` (see ros_camera.T_world_cam_from_transform's
       docstring and vision/HANDOFF.md's OPEN BLOCKING QUESTION) is correct --
       automatically, by computing BOTH and comparing each against ground
       truth. Whichever convention's gate poses land near the known gate
       positions is the correct one.

WHERE THE CAMERA POSE ACTUALLY COMES FROM (read this before touching
--world-frame/--body-frame):
    Pegasus's `ROS2CameraGraph` (the node graph spawn_example.py builds)
    publishes rgb/depth/camera_info and sets each message's `frame_id` to
    `tf_frame_id` ("front_cam") -- but there is no ROS2PublishTransformTree
    node in that graph, so a `front_cam` (or any camera) TF is NEVER
    published. `verify_isaac_pipeline.py`'s TF wait for `front_cam` cannot
    succeed against this sim; treat it as superseded (see HANDOFF.md).

    The only dynamic TF Pegasus publishes comes from `ROS2Backend`
    (ros2_backend.py): a `TransformBroadcaster` on `/tf`, `map ->
    {namespace}_base_link` carrying the vehicle's position and attitude
    (namespace defaults to "drone" + vehicle_id, so `drone0_base_link` for
    spawn_example.py's `vehicle_id: 0`). The camera pose has to be built by
    hand from that plus the STATIC body->camera mount spawn_example.py
    applies to the camera prim:

        T_world_cam_usd = T_map_body(from TF) @ T_body_cam_usd(static mount)

    T_body_cam_usd_from_mount() below reconstructs that static mount
    (translate (0.30, 0, 0.05), rotateXYZ (105, 0, -90) on
    /World/Iris/body/front_cam) exactly as spawn_example.py builds it --
    see its KEEP-IN-SYNC comment. T_world_cam_usd is then still in the
    USD/body camera convention (+X right, +Y up, -Z forward), so it goes
    through the SAME T_world_cam_from_transform(..., optical_frame) this
    module always used; optical_frame is expected to resolve to False
    (USD convention, needing the R_CV_FROM_USD correction) since nothing
    downstream of ROS2Backend ever re-expresses the pose in the ROS
    optical convention -- but that is exactly the thing this script proves
    empirically rather than assumes, so both branches are still computed.

HOW TO RUN IT (three steps, in order -- see HANDOFF.md's "Running the sim
test" section for the full run order including taking off/hovering):
    1. In the Isaac Sim script editor, run workspace/spawn_example.py, then
       press Play (and take off/hover if you want the drone airborne).
    2. In the SAME script editor session, run workspace/spawn_test_gates.py
       to spawn the magenta test gates and write ~/test_gates.json.
    3. On the machine with ROS 2 sourced:
           python3 eval_isaac_gates.py --gates ~/test_gates.json
       (all other arguments have sim-matching defaults; see --help). If
       `map -> drone0_base_link` TF isn't available for some reason, pass
       `--camera-pose manifest` to use the pose spawn_test_gates.py
       recorded when it read the live camera instead.

HOW TO READ THE OUTPUT:
    A per-gate table is printed under EACH optical_frame value, followed by
    a decision block. The decision names the winning optical_frame value,
    whether it is `confident` (low absolute error AND a large margin over
    the other convention), and a one-line `reason`. `annotated_frame.png`
    (green = detected corners, red = ground truth projected under the
    winning convention), `gate_mask.png` (the color segmentation the
    detector used), and `eval_isaac_gates.csv` (one row per gate per frame
    per convention) are written to --out-dir for a closer look.

WHY THIS IS NOT A ROUND-TRIP TEST:
    verify_isaac_pipeline.py checks convention by projecting a KNOWN point
    and looking at where it lands -- useful, but it never touches the
    detector or solvePnP (and, per the fact above, cannot even get that far
    against this sim, since the TF it waits for is never published).
    This script instead detects corners from the actual rendered image with
    color_gate_detector.py and solves a real PnP problem, exactly the path
    production perception will take. HANDOFF.md and gate_pose.py's
    test_waypoint_ordering make the same point about round-trip tests: "a
    round-trip test cannot catch a convention bug like this one (projecting
    and solving with the same wrong model stays perfectly self-consistent)."
    Ground truth here comes from the manifest spawn_test_gates.py writes,
    independent of anything this pipeline computed -- corners come from the
    image, not from re-projecting the truth through TF, so a convention bug
    that is self-consistent between projection and solving cannot hide
    from it.

ALL LOGIC BELOW THIS POINT IS ROS-FREE AND IMPORTABLE OFFLINE (see
test_eval_isaac_gates.py) -- only main() touches rclpy/sensor_msgs/tf2_ros/
cv_bridge, and even there the imports are lazy, mirroring
verify_isaac_pipeline.py.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import cv2

from camera import (make_transform, invert_transform, transform_points,
                    project_points, in_frame)
from gate_pose import (estimate_gate_pose, gate_pose_world, project_gate,
                       DEFAULT_GATE_SIDE)
import ros_camera
from ros_camera import T_world_cam_from_transform, intrinsics_from_camera_info
from color_gate_detector import detect_gates_color, detect_gates_color_debug

# --- Static body -> camera mount, duplicated from workspace/spawn_example.py
# -- KEEP IN SYNC WITH IT. spawn_example.py builds this on the
# /World/Iris/body/front_cam prim as:
#     xf.AddTranslateOp().Set(Gf.Vec3d(0.30, 0.0, 0.05))
#     xf.AddRotateXYZOp().Set(Gf.Vec3f(90.0 + pitch_deg, 0.0, -90.0))  # pitch_deg = 15.0
# If either changes there, change it here too -- there is no code sharing
# the two files (spawn_example.py runs inside Isaac Sim's interpreter,
# eval_isaac_gates.py runs under plain ROS 2 python3).
CAMERA_MOUNT_TRANSLATION = (0.30, 0.0, 0.05)
CAMERA_MOUNT_ROTATE_XYZ_DEG = (105.0, 0.0, -90.0)  # (90 + pitch_deg=15, 0, -90)


def rotate_xyz_deg_to_matrix(x_deg: float, y_deg: float, z_deg: float) -> np.ndarray:
    """
    USD's `UsdGeom.XformOp` rotateXYZ, given (x, y, z) in degrees, composes
    -- in column-vector convention, applied to a point as R @ p -- as
    R = Rz(z) @ Ry(y) @ Rx(x): rotate about the (fixed, parent-frame) X axis
    first, then Y, then Z. This matches how spawn_example.py builds the
    camera mount with a single `AddRotateXYZOp().Set(Gf.Vec3f(x, y, z))`.
    """
    x, y, z = np.radians([x_deg, y_deg, z_deg])
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ Ry @ Rx


def T_body_cam_usd_from_mount() -> np.ndarray:
    """Static body->camera mount as a 4x4, in the USD/body camera convention
    (the camera prim's own axes are never touched by this function -- see
    the module docstring for how this composes with the TF-sourced
    map->body pose)."""
    R = rotate_xyz_deg_to_matrix(*CAMERA_MOUNT_ROTATE_XYZ_DEG)
    return make_transform(R, CAMERA_MOUNT_TRANSLATION)


def load_manifest(path) -> list:
    """
    ~/test_gates.json (written by spawn_test_gates.py) -> list of
    {"name", "x", "y", "z", "yaw_deg", ...} dicts. Accepts both the
    {"yaw_convention": ..., "gates": [...]} wrapper spawn_test_gates.py
    writes and a bare list, so hand-built test fixtures don't need the
    wrapper.
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "gates" in data:
        return data["gates"]
    return data


def T_world_cam_raw_from_manifest(manifest_raw: dict) -> np.ndarray:
    """
    manifest_raw["camera_pose_usd"] (written by spawn_test_gates.py's
    read_camera()) -> T_world_cam, 4x4, still in the raw USD/body camera
    convention -- i.e. exactly the same kind of object
    T_body_cam_usd_from_mount()-composed-with-TF produces, so it can be fed
    to T_world_cam_from_transform() the same way. Used by --camera-pose
    manifest as a TF-free fallback.
    """
    cam = manifest_raw["camera_pose_usd"]
    R = np.asarray(cam["R_world_cam_usd"], dtype=float).reshape(3, 3)
    t = np.asarray(cam["position"], dtype=float)
    return make_transform(R, t)


def pose_delta(T1: np.ndarray, T2: np.ndarray):
    """(position delta in meters, rotation delta in degrees) between two
    4x4 poses -- rotation delta via the trace of R1^T @ R2."""
    pos_delta_m = float(np.linalg.norm(T1[:3, 3] - T2[:3, 3]))
    R1, R2 = T1[:3, :3], T2[:3, :3]
    cos_theta = (np.trace(R1.T @ R2) - 1.0) / 2.0
    rot_delta_deg = float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))
    return pos_delta_m, rot_delta_deg


def T_world_gate_from_spec(x: float, y: float, z: float, yaw_deg: float) -> np.ndarray:
    """
    Manifest gate spec -> T_world_gate, 4x4.

    Matches spawn_test_gates.py's yaw convention exactly (see that module's
    docstring and vision/HANDOFF.md): yaw_deg=0 means the gate's flight
    direction (gate +Z) is world +X, and positive yaw rotates the gate
    counter-clockwise about world +Z (right-hand rule).

    At yaw=0, gate +X = world -Y, gate +Y = world -Z, gate +Z = world +X --
    the same right-handed layout test_gate_pose.py's
    test_world_transform_and_waypoints uses, written out explicitly as
    columns (each column is one gate axis expressed in world coordinates):
    check (-Y) x (-Z) = +X, confirming this is right-handed. Yaw is then
    applied as a left-multiplication by Rz(yaw): rotation about world Z is
    what "yaw" means for a gate planted in a Z-up world.
    """
    R_world_gate_yaw0 = np.array([[0.0, 0.0, 1.0],
                                  [-1.0, 0.0, 0.0],
                                  [0.0, -1.0, 0.0]])
    yaw = np.radians(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.0],
                  [s, c, 0.0],
                  [0.0, 0.0, 1.0]])
    R = Rz @ R_world_gate_yaw0
    return make_transform(R, [x, y, z])


def match_detections(est_centers, truth_centers, max_dist_m: float) -> list:
    """
    Greedy nearest-pair matching between Nx3 estimated centers and Mx3
    truth centers: repeatedly take the closest unmatched (est, truth) pair
    under max_dist_m, each side used at most once. Returns a list of
    (est_idx, truth_idx, dist_m) tuples, sorted by distance ascending (the
    order they were accepted in).

    Greedy nearest-pair rather than a full assignment solver (e.g.
    Hungarian) because gates here are spaced meters apart and max_dist_m is
    a hard gate -- ambiguous near-tie assignments that would need an
    optimal solver are not expected to occur at that separation, and greedy
    is simple enough to be obviously correct in a test.
    """
    est = np.atleast_2d(np.asarray(est_centers, dtype=float)) if len(est_centers) else np.zeros((0, 3))
    truth = np.atleast_2d(np.asarray(truth_centers, dtype=float)) if len(truth_centers) else np.zeros((0, 3))
    if est.shape[0] == 0 or truth.shape[0] == 0:
        return []

    candidates = []
    for i in range(est.shape[0]):
        for j in range(truth.shape[0]):
            d = float(np.linalg.norm(est[i] - truth[j]))
            if d <= max_dist_m:
                candidates.append((d, i, j))
    candidates.sort(key=lambda c: c[0])

    used_est, used_truth = set(), set()
    matches = []
    for d, i, j in candidates:
        if i in used_est or j in used_truth:
            continue
        used_est.add(i)
        used_truth.add(j)
        matches.append((i, j, d))
    return matches


def evaluate_frame(rgb: np.ndarray, intr, T_world_cam_raw: np.ndarray, manifest: list,
                   gate_side: float = DEFAULT_GATE_SIDE,
                   corner_sigma_px: float = 1.0,
                   detector=detect_gates_color,
                   max_match_dist_m: float = 2.0) -> dict:
    """
    One rgb frame + one RAW camera pose (4x4, not yet corrected for
    optical_frame -- see T_world_cam_from_transform) + the ground-truth
    manifest -> per-gate errors under BOTH optical_frame interpretations,
    plus a decision about which one is correct.

    Detection and PnP solving (detector -> estimate_gate_pose) happen
    exactly once -- T_cam_gate is convention-independent, so there is no
    reason to re-detect or re-solve per optical_frame value. Only the
    camera-to-world lift (gate_pose_world, via T_world_cam_from_transform)
    depends on optical_frame, so that is the only part computed twice.

    Returns {"per_convention": {True: {...}, False: {...}}, "decision": {...},
    "n_detections": int, "detections": [GateDetection, ...]}. "detections"
    is the same list used for both conventions (convention-independent),
    handed back so callers (e.g. main()'s annotated-frame output) don't
    need to re-detect. Each per_convention entry is {"matches": [...],
    "unmatched_truth": [names...], "n_detections": int, "n_truth": int};
    each match dict has "gate", "est_idx", "truth_idx", "pos_err_m",
    "range_err_m", "lateral_err_m", "normal_err_deg",
    "normal_err_deg_folded", "reproj_rms_px", "normal_is_reliable",
    "sigma_pos_m".
    """
    detections_px = detector(rgb)
    dets = []
    for corners in detections_px:
        det = estimate_gate_pose(corners, intr, gate_side, corner_sigma_px=corner_sigma_px)
        if det is not None:
            dets.append(det)

    truth_T = [T_world_gate_from_spec(g["x"], g["y"], g["z"], g["yaw_deg"]) for g in manifest]
    truth_centers = np.array([T[:3, 3] for T in truth_T]) if truth_T else np.zeros((0, 3))
    truth_normals = [T[:3, 2] for T in truth_T]
    truth_names = [g["name"] for g in manifest]

    per_convention = {}
    for optical_frame in (True, False):
        T_world_cam = T_world_cam_from_transform(T_world_cam_raw, optical_frame)
        T_cam_world = invert_transform(T_world_cam)

        est_T_world = [gate_pose_world(det, T_world_cam) for det in dets]
        est_centers = (np.array([T[:3, 3] for T in est_T_world])
                       if est_T_world else np.zeros((0, 3)))

        pairs = match_detections(est_centers, truth_centers, max_match_dist_m)

        matches = []
        matched_truth_idx = set()
        for i, j, dist in pairs:
            det = dets[i]
            est_T = est_T_world[i]
            truth_center = truth_centers[j]
            truth_normal_world = truth_normals[j]
            matched_truth_idx.add(j)

            # T_cam_gate is convention-independent, so the estimated center
            # in the camera frame is just det.T_cam_gate's translation --
            # no need to re-transform est_center back through T_cam_world.
            est_cam = det.T_cam_gate[:3, 3]
            truth_cam = transform_points(T_cam_world, truth_center)[0]
            diff_cam = est_cam - truth_cam
            range_err_m = float(diff_cam[2])              # along the optical (+Z) axis
            lateral_err_m = float(np.linalg.norm(diff_cam[:2]))  # perpendicular to it

            est_normal_world = est_T[:3, 2]
            cosang = float(np.clip(np.dot(est_normal_world, truth_normal_world), -1.0, 1.0))
            normal_err_deg = float(np.degrees(np.arccos(cosang)))
            # The gate-frame sign bug this harness exists to catch (see
            # HANDOFF.md) flips the normal's sign, which maps this angle
            # theta to 180-theta. Report the raw angle as the primary
            # number, but also fold it so a 179 deg (flipped-but-otherwise-
            # correct) normal doesn't read as "completely wrong" at a glance.
            normal_err_deg_folded = float(min(normal_err_deg, 180.0 - normal_err_deg))

            matches.append({
                "gate": truth_names[j],
                "est_idx": i,
                "truth_idx": j,
                "pos_err_m": float(dist),
                "range_err_m": range_err_m,
                "lateral_err_m": lateral_err_m,
                "normal_err_deg": normal_err_deg,
                "normal_err_deg_folded": normal_err_deg_folded,
                "reproj_rms_px": float(det.reprojection_error_px),
                "normal_is_reliable": bool(det.normal_is_reliable),
                "sigma_pos_m": (None if np.isnan(det.center_std_m)
                               else float(det.center_std_m)),
            })

        unmatched_truth = [truth_names[j] for j in range(len(truth_names))
                           if j not in matched_truth_idx]

        per_convention[optical_frame] = {
            "matches": matches,
            "unmatched_truth": unmatched_truth,
            "n_detections": len(dets),
            "n_truth": len(manifest),
        }

    decision = decide_optical_frame(per_convention[True], per_convention[False])
    return {"per_convention": per_convention, "decision": decision,
           "n_detections": len(dets), "detections": dets}


def _match_stats(results: dict):
    """(mean pos_err_m, n_matches) over a per_convention result's matches, or
    (None, 0) if there were none."""
    matches = results.get("matches", [])
    if not matches:
        return None, 0
    return float(np.mean([m["pos_err_m"] for m in matches])), len(matches)


def decide_optical_frame(results_true: dict, results_false: dict) -> dict:
    """
    Pick the correct optical_frame value from the per-convention evaluate_frame
    results: lower mean matched position error wins, provided it has at
    least one match.

    confident=True requires BOTH a low absolute error (winner mean error
    < 0.5 m) AND a wide margin over the loser (loser has zero matches, or
    loser mean error > 4x the winner's) -- a "winner" that is merely the
    less-bad of two bad options should not be reported as settled.
    """
    mean_true, n_true = _match_stats(results_true)
    mean_false, n_false = _match_stats(results_false)

    if n_true == 0 and n_false == 0:
        return {
            "optical_frame": None, "confident": False,
            "mean_pos_err_true": None, "mean_pos_err_false": None,
            "n_matches_true": 0, "n_matches_false": 0,
            "reason": ("neither optical_frame value matched a single detection to a "
                      "truth gate -- check the detector, --gates path, and "
                      "--max-match-dist before trusting either branch"),
        }

    if n_true > 0 and (n_false == 0 or mean_true <= mean_false):
        winner, winner_mean = True, mean_true
        winner_n, loser_mean, loser_n = n_true, mean_false, n_false
    else:
        winner, winner_mean = False, mean_false
        winner_n, loser_mean, loser_n = n_false, mean_true, n_true

    confident = bool(winner_mean is not None and winner_mean < 0.5
                    and (loser_n == 0 or (loser_mean is not None and loser_mean > 4.0 * winner_mean)))

    loser_desc = "no matches" if loser_n == 0 else f"{loser_mean:.3f} m over {loser_n} match(es)"
    reason = (f"optical_frame={winner} wins: mean matched position error "
             f"{winner_mean:.3f} m over {winner_n} match(es), vs {loser_desc} "
             f"for optical_frame={not winner}."
             + ("" if confident else " Margin is not decisive -- do not hardcode this yet; "
                                     "collect more frames or investigate the detector."))

    return {
        "optical_frame": winner, "confident": confident,
        "mean_pos_err_true": mean_true, "mean_pos_err_false": mean_false,
        "n_matches_true": n_true, "n_matches_false": n_false,
        "reason": reason,
    }


def render_annotated(rgb: np.ndarray, intr, T_world_cam_winner: np.ndarray,
                     detections: list, manifest: list,
                     gate_side: float = DEFAULT_GATE_SIDE) -> np.ndarray:
    """
    Debug frame: detected corner polygons in green, ground-truth gate
    corners (projected under the winning optical_frame convention) in red
    with name labels. Returns a BGR image ready for cv2.imwrite.
    """
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    for det in detections:
        pts = np.round(det.corners_px).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    T_cam_world = invert_transform(T_world_cam_winner)
    for g in manifest:
        T_world_gate = T_world_gate_from_spec(g["x"], g["y"], g["z"], g["yaw_deg"])
        T_cam_gate = T_cam_world @ T_world_gate
        corners = project_gate(T_cam_gate, intr, gate_side)
        if np.isnan(corners).any() or not in_frame(corners, intr).all():
            continue  # behind the camera or off-image -- nothing sane to draw
        pts = np.round(corners).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        label_xy = tuple(np.round(corners.mean(axis=0)).astype(int))
        cv2.putText(frame, g["name"], label_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (0, 0, 255), 1, cv2.LINE_AA)
    return frame


CSV_FIELDS = ["frame", "gate", "convention", "detected", "range_true_m",
             "pos_err_m", "range_err_m", "lateral_err_m", "normal_err_deg",
             "reproj_rms_px", "normal_is_reliable"]


def write_csv(path, rows) -> None:
    """rows: iterable of dicts keyed by (a subset of) CSV_FIELDS. Column
    names reuse eval_pnp.py's where the meaning is the same (pos/range/
    lateral/normal err, reproj px) so the two remain easy to compare."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


_CONVENTION_LABELS = {
    True: "raw pose treated as already optical (control)",
    False: "USD camera pose, converted with R_CV_FROM_USD (expected)",
}


def _print_convention_table(optical_frame: bool, result: dict) -> None:
    conv = result["per_convention"][optical_frame]
    print(f"\n--- optical_frame={optical_frame}  [{_CONVENTION_LABELS[optical_frame]}] ---")
    hdr = f"{'gate':<10} {'pos_err_m':>10} {'range_err_m':>12} {'lateral_err_m':>14} {'normal_err_deg':>15} {'reproj_px':>10}"
    print(hdr)
    for m in conv["matches"]:
        print(f"{m['gate']:<10} {m['pos_err_m']:10.3f} {m['range_err_m']:12.3f} "
              f"{m['lateral_err_m']:14.3f} {m['normal_err_deg']:15.1f} {m['reproj_rms_px']:10.2f}")
    for name in conv["unmatched_truth"]:
        print(f"{name:<10} {'--- not detected / not matched ---':>10}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--gates", default=os.path.expanduser("~/test_gates.json"),
                    help="path to the manifest spawn_test_gates.py wrote (default: ~/test_gates.json)")
    ap.add_argument("--namespace", default="/iris_0/front_cam",
                    help="topic namespace for rgb/camera_info (default: /iris_0/front_cam)")
    ap.add_argument("--world-frame", default="map",
                    help="TF frame the vehicle pose is published in (default: map, per "
                        "ROS2Backend's TransformBroadcaster -- see module docstring)")
    ap.add_argument("--body-frame", default="drone0_base_link",
                    help="TF frame for the vehicle body (default: drone0_base_link, i.e. "
                        "'drone' + vehicle_id=0 from spawn_example.py's PX4MavlinkBackendConfig)")
    ap.add_argument("--camera-pose", choices=["tf", "manifest"], default="tf",
                    help="'tf' (default): build the camera pose from live map->body TF plus "
                        "the static mount. 'manifest': use the camera pose "
                        "spawn_test_gates.py recorded from the live USD stage when it spawned "
                        "the gates -- use this if map->body TF isn't available")
    ap.add_argument("--timeout", type=float, default=10.0,
                    help="seconds to wait for camera_info/image/TF readiness (default: 10)")
    ap.add_argument("--frames", type=int, default=1,
                    help="number of distinct rgb frames to collect and evaluate (default: 1)")
    ap.add_argument("--out-dir", default=".",
                    help="directory for annotated_frame.png / gate_mask.png / eval_isaac_gates.csv")
    ap.add_argument("--gate-side", type=float, default=None,
                    help="override gate opening side (m); default: read from the manifest, "
                        "falling back to gate_pose.DEFAULT_GATE_SIDE")
    ap.add_argument("--corner-sigma-px", type=float, default=1.0)
    ap.add_argument("--max-match-dist", type=float, default=2.0,
                    help="max distance (m) between an estimated and a truth gate center to "
                        "count as the same gate (default: 2.0)")
    args = ap.parse_args()

    with open(args.gates) as f:
        manifest_raw = json.load(f)
    manifest = load_manifest(args.gates)

    gate_side = args.gate_side
    if gate_side is None:
        if manifest and "side" in manifest[0]:
            gate_side = float(manifest[0]["side"])
        else:
            gate_side = DEFAULT_GATE_SIDE

    if isinstance(manifest_raw, dict) and "yaw_convention" in manifest_raw:
        print(f"manifest yaw convention: {manifest_raw['yaw_convention']}")
    print(f"loaded {len(manifest)} gate(s) from {args.gates}, gate_side={gate_side} m")
    if isinstance(manifest_raw, dict) and manifest_raw.get("dropped"):
        print(f"manifest notes {len(manifest_raw['dropped'])} gate(s) dropped by spawn_test_gates.py:")
        for note in manifest_raw["dropped"]:
            print(f"  {note}")

    manifest_camera_T = None
    if isinstance(manifest_raw, dict) and "camera_pose_usd" in manifest_raw:
        manifest_camera_T = T_world_cam_raw_from_manifest(manifest_raw)

    if args.camera_pose == "manifest" and manifest_camera_T is None:
        sys.exit(f"--camera-pose manifest requested but {args.gates} has no "
                 f"'camera_pose_usd' entry -- re-run spawn_test_gates.py, or use --camera-pose tf")

    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image, CameraInfo
        from tf2_ros import Buffer, TransformListener
        from cv_bridge import CvBridge
    except ImportError as e:
        sys.exit(f"this script needs ROS 2 sourced (rclpy/tf2_ros/cv_bridge): {e}")

    rclpy.init()
    node = Node("eval_isaac_gates")
    bridge = CvBridge()
    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)

    state = {"info": None, "rgb": None}
    node.create_subscription(CameraInfo, f"{args.namespace}/camera_info",
                             lambda m: state.__setitem__("info", m), 10)
    node.create_subscription(Image, f"{args.namespace}/rgb",
                             lambda m: state.__setitem__("rgb", m), 10)

    need_tf = args.camera_pose == "tf"
    deadline = time.time() + args.timeout

    def ready():
        base_ready = state["info"] is not None and state["rgb"] is not None
        if not need_tf:
            return base_ready
        return base_ready and tf_buffer.can_transform(args.world_frame, args.body_frame, rclpy.time.Time())

    while rclpy.ok() and time.time() < deadline and not ready():
        rclpy.spin_once(node, timeout_sec=0.2)

    if state["info"] is None:
        sys.exit(f"no camera_info on {args.namespace}/camera_info in {args.timeout}s -- "
                 f"is spawn_example.py running with PX4 SITL connected?")
    if state["rgb"] is None:
        sys.exit(f"no image on {args.namespace}/rgb in {args.timeout}s")
    if need_tf and not tf_buffer.can_transform(args.world_frame, args.body_frame, rclpy.time.Time()):
        sys.exit(f"no TF from {args.world_frame} to {args.body_frame} in {args.timeout}s -- "
                 f"known frames:\n{tf_buffer.all_frames_as_string()}\n"
                 f"try --body-frame <name> (spawn_example.py's vehicle_id changes the default "
                 f"'drone0_base_link'), or --camera-pose manifest to skip TF entirely")

    intr = intrinsics_from_camera_info(state["info"])
    print(f"\ncamera_info: {intr.width}x{intr.height}, fx={intr.fx:.1f} fy={intr.fy:.1f}")
    if intr.width <= 320:
        print(f"  NOTE: {intr.width}x{intr.height} is low resolution for range accuracy "
              f"at distance -- a 1.5 m gate at 12 m spans only ~{intr.fx*1.5/12:.0f} px here")

    # Collect --frames distinct rgb messages (by header stamp), each paired
    # with the camera pose at (as close as possible to) that same time.
    collected = []
    seen_stamp = None
    warned_tf_fallback = False
    cross_checked = False
    deadline2 = time.time() + args.timeout
    while len(collected) < args.frames and rclpy.ok() and time.time() < deadline2:
        rclpy.spin_once(node, timeout_sec=0.2)
        msg = state["rgb"]
        if msg is None:
            continue
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        if stamp == seen_stamp:
            continue
        seen_stamp = stamp

        if args.camera_pose == "manifest":
            T_world_cam_raw = manifest_camera_T
        else:
            try:
                t = tf_buffer.lookup_transform(args.world_frame, args.body_frame,
                                               msg.header.stamp).transform
            except Exception as exc1:
                # Pegasus's TF and Isaac's image stamps may not share a
                # clock -- fall back to the latest available TF once, with
                # a note, rather than dropping every frame over a stamp
                # mismatch.
                try:
                    t = tf_buffer.lookup_transform(args.world_frame, args.body_frame,
                                                   rclpy.time.Time()).transform
                    if not warned_tf_fallback:
                        print(f"NOTE: TF at the image stamp was unavailable ({exc1}); "
                             f"using the latest available TF instead for this and any "
                             f"further frames")
                        warned_tf_fallback = True
                except Exception as exc2:
                    print(f"TF lookup failed at both the image stamp and latest: {exc2}")
                    print(f"known frames:\n{tf_buffer.all_frames_as_string()}")
                    print("try --body-frame <name> or --camera-pose manifest")
                    continue
            T_map_body = ros_camera.transform_to_matrix(t.translation, t.rotation)
            T_world_cam_raw = T_map_body @ T_body_cam_usd_from_mount()

            if manifest_camera_T is not None and not cross_checked:
                # This is what separates "mount wrong" from "convention
                # wrong": if the TF+mount pose disagrees with the pose
                # spawn_test_gates.py read straight off the USD stage, the
                # optical_frame decision below is not trustworthy no matter
                # what it says -- either the drone moved, or CAMERA_MOUNT_*
                # / rotate_xyz_deg_to_matrix disagree with spawn_example.py.
                pos_delta_m, rot_delta_deg = pose_delta(T_world_cam_raw, manifest_camera_T)
                print(f"\ncross-check vs. manifest camera_pose_usd: "
                     f"position delta {pos_delta_m:.3f} m, rotation delta {rot_delta_deg:.2f} deg")
                if pos_delta_m > 0.10 or rot_delta_deg > 2.0:
                    print("  WARNING: this is bigger than TF/mount noise should produce. "
                         "Either the drone moved since spawn_test_gates.py ran, or "
                         "CAMERA_MOUNT_TRANSLATION/CAMERA_MOUNT_ROTATE_XYZ_DEG/"
                         "rotate_xyz_deg_to_matrix() disagree with spawn_example.py's actual "
                         "mount -- in the latter case, do NOT trust the optical_frame decision "
                         "below until this is resolved.")
                cross_checked = True

        rgb = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        collected.append((rgb, T_world_cam_raw))

    if not collected:
        sys.exit(f"never got a distinct rgb frame with a usable camera pose within {args.timeout}s")
    if len(collected) < args.frames:
        print(f"WARNING: only collected {len(collected)}/{args.frames} distinct frames "
              f"within {args.timeout}s -- proceeding with what we have")

    os.makedirs(args.out_dir, exist_ok=True)

    per_frame_results = []
    csv_rows = []
    last_rgb, last_T_world_cam_raw, last_result, last_mask = None, None, None, None

    for fi, (rgb, T_world_cam_raw) in enumerate(collected):
        _corners_debug, mask = detect_gates_color_debug(rgb)
        result = evaluate_frame(rgb, intr, T_world_cam_raw, manifest, gate_side=gate_side,
                                corner_sigma_px=args.corner_sigma_px,
                                max_match_dist_m=args.max_match_dist)
        per_frame_results.append(result)
        last_rgb, last_T_world_cam_raw, last_result, last_mask = rgb, T_world_cam_raw, result, mask

        for optical_frame in (True, False):
            conv = result["per_convention"][optical_frame]
            by_gate = {m["gate"]: m for m in conv["matches"]}
            T_world_cam = T_world_cam_from_transform(T_world_cam_raw, optical_frame)
            T_cam_world = invert_transform(T_world_cam)
            for g in manifest:
                truth_world = np.array([g["x"], g["y"], g["z"]])
                range_true_m = float(transform_points(T_cam_world, truth_world)[0, 2])
                m = by_gate.get(g["name"])
                csv_rows.append({
                    "frame": fi,
                    "gate": g["name"],
                    "convention": optical_frame,
                    "detected": m is not None,
                    "range_true_m": range_true_m,
                    "pos_err_m": m["pos_err_m"] if m else "",
                    "range_err_m": m["range_err_m"] if m else "",
                    "lateral_err_m": m["lateral_err_m"] if m else "",
                    "normal_err_deg": m["normal_err_deg"] if m else "",
                    "reproj_rms_px": m["reproj_rms_px"] if m else "",
                    "normal_is_reliable": m["normal_is_reliable"] if m else "",
                })

    n_gates = len(manifest)
    n_det_last = last_result["n_detections"]
    print(f"\ndetector found {n_det_last} of {n_gates} gate(s) in the most recent frame "
         f"({len(collected)} frame(s) evaluated total)")

    for optical_frame in (True, False):
        _print_convention_table(optical_frame, last_result)
        errs = [m["pos_err_m"] for r in per_frame_results
               for m in r["per_convention"][optical_frame]["matches"]]
        if errs:
            print(f"  mean pos_err over {len(collected)} frame(s): "
                 f"{np.mean(errs):.3f} m (std {np.std(errs):.3f} m, n={len(errs)})")
        else:
            print(f"  no matches across {len(collected)} frame(s)")

    # Decide from the pooled per-frame decisions rather than only the last
    # frame's: aggregate every frame's matches per convention and re-run
    # decide_optical_frame on the pooled result so a single unlucky frame
    # cannot flip the verdict.
    pooled = {True: {"matches": [], "unmatched_truth": []},
             False: {"matches": [], "unmatched_truth": []}}
    for r in per_frame_results:
        for optical_frame in (True, False):
            pooled[optical_frame]["matches"].extend(r["per_convention"][optical_frame]["matches"])
    decision = decide_optical_frame(pooled[True], pooled[False])

    print("\n--- decision ---")
    print(f"  optical_frame = {decision['optical_frame']}")
    print(f"  confident     = {decision['confident']}")
    print(f"  reason: {decision['reason']}")

    annotated_path = os.path.join(args.out_dir, "annotated_frame.png")
    mask_path = os.path.join(args.out_dir, "gate_mask.png")
    csv_path = os.path.join(args.out_dir, "eval_isaac_gates.csv")

    if decision["optical_frame"] is not None:
        T_world_cam_winner = T_world_cam_from_transform(last_T_world_cam_raw, decision["optical_frame"])
        annotated = render_annotated(last_rgb, intr, T_world_cam_winner, last_result["detections"],
                                     manifest, gate_side)
        cv2.imwrite(annotated_path, annotated)
        print(f"\nsaved {annotated_path} (green = detected, red = truth under the winning convention)")
    else:
        cv2.imwrite(annotated_path, cv2.cvtColor(last_rgb, cv2.COLOR_RGB2BGR))
        print(f"\nsaved {annotated_path} (no confident convention to overlay truth with)")

    cv2.imwrite(mask_path, last_mask)
    print(f"saved {mask_path}")
    write_csv(csv_path, csv_rows)
    print(f"saved {csv_path}")

    if decision["confident"]:
        print(f"\nNext step: hardcode optical_frame={decision['optical_frame']} wherever "
             f"T_world_cam_from_transform() or GatePoseNode is instantiated, remove the "
             f"'UNVERIFIED' warnings from ros_camera.py's docstrings, and update "
             f"HANDOFF.md's status table (see its OPEN BLOCKING QUESTION section).")
    else:
        print(f"\nNot confident enough to hardcode optical_frame yet -- collect more frames "
             f"(--frames), check gate_mask.png for a bad color threshold, and re-run before "
             f"touching ros_camera.py's UNVERIFIED warnings.")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
