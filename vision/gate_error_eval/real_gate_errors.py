"""
Tony's gate-error figures (plot_gate_errors.py) for the REAL drone: turns live_gate_test.py runs
into gate_errors.csv rows (error_records.CSV_FIELDS) and plots them with plot_gate_errors.plot_all,
so the figures match the Isaac ones (fig1 error vs range, fig2 vs time, fig3 top-down, fig4
detection rate).

Two kinds of truth, and the figure titles say which:

  still   Drone held still / on a mount, gate at a taped distance (one "gate" per run, e.g. "2.0 m").
          Truth is the tape, along the measured bearing: range error only (lateral was not taped,
          so it is left blank). World frame = camera at the origin, x forward, y left.
              python real_gate_errors.py still --out-dir ../live_runs/figs_still_tape

  vio     A moving run logged with the drone's own VIO (voxl-logger -v qvio) and a clock stamp
          (python3 -c 'import time;print(time.time(),time.monotonic())' > clock.txt). Each frame's
          camera pose comes from the VIO, so each estimate lands in the VIO world frame. There is no
          tape truth: the reference gate position is the MEDIAN of the corner-fix world estimates,
          so "error" is scatter about that reference (it includes VIO drift, and a constant bias
          would not show). in_view = the reference gate's center projects into the image.
              python real_gate_errors.py vio ../live_runs/onboard_hybrid_handheld_motion_03

  px4     A moving run recorded with live_gate_test.py --mavlink (px4_pose.csv): each frame's camera pose
          comes from PX4's own estimate (LOCAL_POSITION_NED + ATTITUDE_QUATERNION over MAVLink), so the
          gates land in PX4's local NED frame, the frame PX4 flies in. Same "median estimate" reference
          as vio. No clock stamp needed: the poses and the frames share the camera test's clock.
              python real_gate_errors.py px4 ../live_runs/<run>

The VIO's Euler convention and the image latency (Portal receive time minus capture time) are not
documented for this setup, so both are picked from the data: the pair that gives the tightest
cluster of world estimates of the (fixed) gate. The choice is printed and written to the summary.
"""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_VISION = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _VISION)
import plot_gate_errors as pge  # noqa: E402
import error_records as er  # noqa: E402

LIVE_RUNS = os.path.join(_VISION, "live_runs")

# Still runs with a taped distance (2026-11-23). kind "range" = taped straight to the gate
# (gate at camera height), "horizontal" = taped along the floor (gate on a stand above the camera).
STILL_RUNS = [
    ("onboard_hybrid_1m", 1.00, "range"),
    ("onboard_hybrid_2m", 2.00, "range"),
    ("onboard_hybrid_3m", 3.00, "range"),
    ("onboard_hybrid_02", 3.50, "range"),          # backlit by the windows: mostly box-only fixes
    ("onboard_hybrid_4m", 4.00, "range"),
    ("onboard_hybrid_5m", 5.00, "range"),          # corners never found: box-only fixes
    ("onboard_hybrid_fixed_0deg", 2.54, "horizontal"),  # drone on a mount, gate on its stand
]

# hires_front with respect to imu_apps, /etc/modalai/extrinsics.conf (D0012 Starling 2 Max C29):
# T [0.0408, 0.006, 0.0186] m, RPY_parent_to_child [0, 90, 90] deg, intrinsic XYZ.
T_IMU_CAM_T = np.array([0.0408, 0.006, 0.0186])
IMAGE_WH = (1024, 768)


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


R_IMU_CAM = _rx(0.0) @ _ry(math.radians(90)) @ _rz(math.radians(90))   # camera axes in the imu frame
# imu_apps with respect to body: T [0.0303, -0.0057, -0.0116] m, no rotation (same extrinsics file). PX4's
# position and attitude are the body's (FRD), so the camera is R_IMU_CAM / this offset from it.
R_BODY_CAM = R_IMU_CAM
T_BODY_CAM = np.array([0.0303, -0.0057, -0.0116]) + T_IMU_CAM_T

EULER = {
    "zyx": lambda r, p, y: _rz(y) @ _ry(p) @ _rx(r),     # aerospace yaw-pitch-roll
    "xyz": lambda r, p, y: _rx(r) @ _ry(p) @ _rz(y),     # intrinsic XYZ (ModalAI extrinsics style)
}


def read_all_detections(run_dir):
    """{frame: [row with a pose, ...]} -- every detection of a frame, not only det_idx 0."""
    out = {}
    for r in csv.DictReader(open(os.path.join(run_dir, "detections.csv"))):
        if r["range_m"]:
            out.setdefault(int(r["frame"]), []).append(r)
    return out


def read_detections(run_dir):
    """{frame: [t_wall, det_idx-0 row with a pose, or None]} for EVERY frame of the run.

    detections.csv only has frames where the detector returned something, so the frame count comes
    from raw_jpg/ (one JPEG per processed frame) and the missing frames' times are interpolated."""
    rows = list(csv.DictReader(open(os.path.join(run_dir, "detections.csv"))))
    frames = {}
    for r in rows:
        f = int(r["frame"])
        frames.setdefault(f, [float(r["t_wall"]), None])
        if r["det_idx"] == "0" and r["range_m"]:
            frames[f][1] = r
    raw = os.path.join(run_dir, "raw_jpg")
    n = len([x for x in os.listdir(raw) if x.endswith(".jpg")]) if os.path.isdir(raw) else 0
    known = sorted(frames)
    if n > 0 and known:
        tk = np.array([frames[f][0] for f in known])
        for f in range(n):
            if f not in frames:
                frames[f] = [float(np.interp(f, known, tk)), None]
    return frames


def _fix_xyz(r):
    return np.array([float(r["x_m"]), float(r["y_m"]), float(r["z_m"])])


def _row(**kw):
    out = {k: None for k in er.CSV_FIELDS}
    out.update(kw)
    return out


# ------------------------------------------------------------------------------------ still

def still_rows(runs=STILL_RUNS):
    rows = []
    for name, taped, kind in runs:
        run_dir = os.path.join(LIVE_RUNS, name)
        frames = read_detections(run_dir)
        label = f"{taped:.2f} m" + (" floor" if kind == "horizontal" else "")
        t0 = min(t for t, _ in frames.values())
        for f in sorted(frames):
            t, r = frames[f]
            base = dict(t_s=t - t0, frame=f, gate=label, in_view=True, drone_x=0.0, drone_y=0.0,
                        drone_z=0.0, drone_speed_mps=0.0, off_axis_deg=None, normal_err_deg=None)
            if r is None:
                rows.append(_row(**base, detected=False, range_true_m=taped,
                                 true_x=taped, true_y=0.0, true_z=0.0))
                continue
            c = _fix_xyz(r)                                   # camera frame: x right, y down, z fwd
            if kind == "range":
                est = float(np.linalg.norm(c))
            else:
                est = float(math.hypot(c[0], c[2]))
            u = c / np.linalg.norm(c)
            truth_c = u * (np.linalg.norm(c) * taped / est)   # the estimate's bearing, taped distance
            w = lambda v: (float(v[2]), float(-v[0]), float(-v[1]))   # world: x fwd, y left, z up
            ex, ey, ez = w(c)
            tx, ty, tz = w(truth_c)
            rows.append(_row(**base, detected=True, est_x=ex, est_y=ey, est_z=ez,
                             true_x=tx, true_y=ty, true_z=tz, range_true_m=taped,
                             pos_err_m=abs(est - taped), range_err_m=est - taped, lateral_err_m=None,
                             reproj_rms_px=float(r["reproj_px"]), normal_is_reliable=r["src"] == "yolo+color"))
    return rows


# --------------------------------------------------------------------------- moving runs: pose sources
#
# A pose source gives, for a camera frame's receive time t (detections.csv t_wall) and an assumed
# image latency, the camera's rotation and position in a fixed world frame plus the drone's speed.
# Several "variants" can be tried (the VIO's unknown Euler convention); fit_sync keeps the variant and
# latency with the tightest gate clusters.

class VioSource:
    """The drone's own VIO, logged with voxl-logger -v qvio (qvio.csv) and a clock stamp (clock.txt).
    World frame: the VIO's (gravity-aligned, z down)."""

    frame_label = "VIO world"

    def __init__(self, run_dir):
        q = list(csv.DictReader(open(os.path.join(run_dir, "qvio.csv"))))
        self.t = np.array([int(r["timestamp(ns)"]) * 1e-9 for r in q])
        self.p = np.array([[float(r[f"T_imu_wrt_vio_{a}(m)"]) for a in "xyz"] for r in q])
        self.rpy = np.array([[float(r["roll(rad)"]), float(r["pitch(rad)"]), float(r["yaw(rad)"])] for r in q])
        self.v = np.array([[float(r[f"vel_imu_wrt_vio_{a}(m/s)"]) for a in "xyz"] for r in q])
        wall0, mono0 = map(float, open(os.path.join(run_dir, "clock.txt")).read().split())
        self.offset = wall0 - mono0              # frames are on time.time(), the VIO on CLOCK_MONOTONIC
        self.variants = {e: (lambda t, lat, e=e: self.pose(t, lat, e)) for e in EULER}
        self.n = len(self.t)

    def pose(self, t_frame, lat, euler):
        tm = t_frame - self.offset - lat
        i = int(np.searchsorted(self.t, tm))
        if i <= 0 or i >= len(self.t):
            return None
        a = (tm - self.t[i - 1]) / (self.t[i] - self.t[i - 1])
        pos = self.p[i - 1] + a * (self.p[i] - self.p[i - 1])
        j = i - 1 if a < 0.5 else i
        R_w_imu = EULER[euler](*self.rpy[j])
        return R_w_imu @ R_IMU_CAM, pos + R_w_imu @ T_IMU_CAM_T, float(np.linalg.norm(self.v[j]))


def quat_to_R(qw, qx, qy, qz):
    """Rotation matrix of a unit quaternion (w, x, y, z): body -> world (MAVLink ATTITUDE_QUATERNION)."""
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]])


class Px4Source:
    """PX4's own estimate over MAVLink, recorded by live_gate_test.py --mavlink (px4_pose.csv:
    LOCAL_POSITION_NED + ATTITUDE_QUATERNION, on the same time.time() clock as the camera frames).
    World frame: PX4's local NED (x north, y east, z down), the frame PX4 flies in. Whatever PX4 fuses
    (optical flow + barometer since 2026-11-25, VIO before) is what this frame is."""

    frame_label = "PX4 local NED"

    def __init__(self, run_dir):
        rows = [r for r in csv.DictReader(open(os.path.join(run_dir, "px4_pose.csv"))) if r["qw"] not in ("", "nan")]
        if len(rows) < 10:
            raise SystemExit(f"{run_dir}/px4_pose.csv has {len(rows)} usable poses: was the run recorded with "
                             f"--mavlink and mavlink_tee.py running?")
        self.t = np.array([float(r["t_wall"]) for r in rows])
        self.p = np.array([[float(r[k]) for k in ("x", "y", "z")] for r in rows])
        self.v = np.array([[float(r[k]) for k in ("vx", "vy", "vz")] for r in rows])
        self.R = [quat_to_R(*(float(r[k]) for k in ("qw", "qx", "qy", "qz"))) for r in rows]
        self.variants = {"px4": self.pose}
        self.n = len(self.t)

    def pose(self, t_frame, lat):
        tm = t_frame - lat                       # same clock: only the image latency
        i = int(np.searchsorted(self.t, tm))
        if i <= 0 or i >= len(self.t):
            return None
        a = (tm - self.t[i - 1]) / (self.t[i] - self.t[i - 1])
        pos = self.p[i - 1] + a * (self.p[i] - self.p[i - 1])
        j = i - 1 if a < 0.5 else i
        R_w_body = self.R[j]
        return R_w_body @ R_BODY_CAM, pos + R_w_body @ T_BODY_CAM, float(np.linalg.norm(self.v[j]))


def world_estimates(frames, alldet, pose_fn, latency, corners_only=True):
    """(frame, world xyz) of every detection (all det_idx) of the run."""
    out = []
    for f in sorted(alldet):
        if f not in frames:
            continue
        cp = pose_fn(frames[f][0], latency)
        if cp is None:
            continue
        R, c, _ = cp
        for r in alldet[f]:
            if corners_only and r["src"] != "yolo+color":
                continue
            out.append((f, R @ _fix_xyz(r) + c))
    return out


def _spread(pts):
    """Median distance of the estimates to their nearest gate cluster, and the cluster centers."""
    P = np.array([x for _, x in pts])
    refs, _ = cluster_gates(P)
    d = np.min([np.linalg.norm(P - r, axis=1) for r in refs], axis=0)
    return float(np.median(d)), refs


def cluster_gates(points, sep=1.0, min_pts=30):
    """Reference positions of the fixed gates: medians of clusters of world estimates.
    Greedy: start from the overall median, add a cluster for >= min_pts points farther than sep from
    every reference, re-center each on the median of its own points, repeat."""
    P = np.asarray(points)
    refs = [np.median(P, axis=0)]
    for _ in range(10):
        d = np.min([np.linalg.norm(P - r, axis=1) for r in refs], axis=0)
        far = P[d > sep]
        if len(far) >= min_pts:
            refs.append(np.median(far, axis=0))
        lab = np.argmin([np.linalg.norm(P - r, axis=1) for r in refs], axis=0)
        new = []
        for k, r in enumerate(refs):
            mine = P[(lab == k) & (np.linalg.norm(P - r, axis=1) < sep)]
            if len(mine) >= min_pts:
                new.append(np.median(mine, axis=0))
        refs = new or refs[:1]
    lab = np.argmin([np.linalg.norm(P - r, axis=1) for r in refs], axis=0)
    counts = [int(np.sum(lab == k)) for k in range(len(refs))]
    order = np.argsort(counts)[::-1]                      # gate_A = the most-seen gate
    return [refs[k] for k in order], [counts[k] for k in order]


def fit_sync(frames, alldet, source):
    """Variant (VIO Euler convention, or PX4's single one) and image latency with the tightest gate clusters."""
    best, table = None, []
    for name, fn in source.variants.items():
        for lat in np.arange(0.0, 0.801, 0.05):
            pts = world_estimates(frames, alldet, fn, lat)
            if len(pts) < 20:
                continue
            s, refs = _spread(pts)
            table.append((name, round(float(lat), 2), round(s, 4)))
            if best is None or s < best[0]:
                best = (s, name, float(lat), refs)
    if best is None:
        raise SystemExit("fewer than 20 corner fixes with a pose: nothing to place the gates from")
    return best, table


def _project(lens_K, lens_D, fisheye, Xc):
    import cv2
    pts = Xc.reshape(1, 1, 3).astype(np.float64)
    if fisheye:
        uv, _ = cv2.fisheye.projectPoints(pts, np.zeros(3), np.zeros(3), lens_K, lens_D.reshape(-1)[:4])
    else:
        uv, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), lens_K, lens_D)
    return uv.reshape(2)


def moving_rows(run_dir, calib, source):
    """gate_errors rows for a moving run, every detection placed in the source's world frame."""
    from live_gate_test import load_calib
    K, D, fisheye, csize = load_calib(calib)
    if csize and tuple(csize) != IMAGE_WH and csize[0]:
        s = IMAGE_WH[0] / csize[0]
        K = K.copy()
        K[:2] *= s
    frames = read_detections(run_dir)
    alldet = read_all_detections(run_dir)
    (spread, variant, lat, refs), table = fit_sync(frames, alldet, source)
    pose_fn = source.variants[variant]
    names = [f"gate_{chr(ord('A') + k)}" for k in range(len(refs))]
    t0 = min(t for t, _ in frames.values())
    rows, n_unmatched, n_dets = [], 0, 0
    for f in sorted(frames):
        t = frames[f][0]
        cp = pose_fn(t, lat)
        if cp is None:
            continue
        R, c, speed = cp
        # each detection to its nearest gate (world distance under 0.5 m + 10 % of range), else unmatched
        best = {}
        for r in alldet.get(f, []):
            n_dets += 1
            est_c = _fix_xyz(r)
            est_w = R @ est_c + c
            d = [np.linalg.norm(est_w - ref) for ref in refs]
            k = int(np.argmin(d))
            if d[k] > 0.5 + 0.1 * float(r["range_m"]):
                n_unmatched += 1
                continue
            if k not in best or d[k] < best[k][0]:
                best[k] = (d[k], r, est_c, est_w)
        for k, ref in enumerate(refs):
            ref_c = R.T @ (ref - c)                                   # this gate in the camera frame
            rng_true = float(np.linalg.norm(ref_c))
            in_view = False
            if ref_c[2] > 0.3:
                u, w_ = _project(K, D, fisheye, ref_c)
                in_view = bool(0 <= u < IMAGE_WH[0] and 0 <= w_ < IMAGE_WH[1])
            base = dict(t_s=t - t0, frame=f, gate=names[k], true_x=float(ref[0]), true_y=float(ref[1]),
                        true_z=float(ref[2]), drone_x=float(c[0]), drone_y=float(c[1]), drone_z=float(c[2]),
                        drone_speed_mps=speed, range_true_m=rng_true,
                        off_axis_deg=float(math.degrees(math.atan2(math.hypot(ref_c[0], ref_c[1]), ref_c[2]))),
                        in_view=in_view, normal_err_deg=None)
            if k not in best:
                rows.append(_row(**base, detected=False))
                continue
            _, r, est_c, est_w = best[k]
            los = ref_c / rng_true
            d = est_c - ref_c
            rng_err = float(d @ los)
            rows.append(_row(**base, detected=True, est_x=float(est_w[0]), est_y=float(est_w[1]),
                             est_z=float(est_w[2]), pos_err_m=float(np.linalg.norm(d)), range_err_m=rng_err,
                             lateral_err_m=float(np.linalg.norm(d - rng_err * los)),
                             reproj_rms_px=float(r["reproj_px"]), normal_is_reliable=r["src"] == "yolo+color"))
    # A gate may be set up or carried in partway through (gate_B in run 3): count it as in view only
    # between its first and last detection, so detection rates don't include time it wasn't there.
    windows = {}
    for n in names:
        ts = [r["t_s"] for r in rows if r["gate"] == n and r["detected"]]
        windows[n] = (min(ts), max(ts)) if ts else (0.0, -1.0)
        for r in rows:
            if r["gate"] == n and not (windows[n][0] <= r["t_s"] <= windows[n][1]):
                r["in_view"] = False
    info = dict(pose_source=source.frame_label, variant=variant, latency_s=lat,
                median_scatter_to_nearest_gate_m=spread,
                gate_windows_s={n: list(w) for n, w in windows.items()},
                gates={n: ref.tolist() for n, ref in zip(names, refs)}, detections=n_dets,
                unmatched_detections=n_unmatched, n_poses=source.n,
                sync_search=[dict(variant=e, latency_s=l, scatter_m=s) for e, l, s in table])
    return rows, info


def vio_rows(run_dir, calib):
    return moving_rows(run_dir, calib, VioSource(run_dir))


def px4_rows(run_dir, calib):
    return moving_rows(run_dir, calib, Px4Source(run_dir))


# ------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("still")
    s.add_argument("--out-dir", default=os.path.join(LIVE_RUNS, "figs_still_tape"))
    for mode in ("vio", "px4"):
        v = sub.add_parser(mode)
        v.add_argument("run_dir")
        v.add_argument("--calib", default=os.path.join(_VISION, "calib", "hires_front_1024x768.yml"))
        v.add_argument("--out-dir", default=None, help="default: <run_dir>/figs")
        v.add_argument("--title", default=None)
    a = ap.parse_args()

    if a.mode == "still":
        rows = still_rows()
        suffix = "(real drone, still, tape truth)"
        out = a.out_dir
        info = {"runs": [dict(run=n, taped_m=d, kind=k) for n, d, k in STILL_RUNS]}
    else:
        rows, info = (vio_rows if a.mode == "vio" else px4_rows)(a.run_dir, a.calib)
        name = os.path.basename(os.path.normpath(a.run_dir))
        suffix = a.title or ("(real drone, hand-held, ref = median est.)" if a.mode == "vio"
                             else "(real drone, PX4 frame, ref = median est.)")
        out = a.out_dir or os.path.join(a.run_dir, "figs" if a.mode == "vio" else "figs_px4")
        print(f"[{name}] poses: {info['pose_source']} ({info['n_poses']}), variant {info['variant']}, "
              f"image latency {info['latency_s']:.2f} s, "
              f"median scatter {info['median_scatter_to_nearest_gate_m']:.3f} m, gates "
              + ", ".join(f"{n} {np.round(x, 2).tolist()}" for n, x in info["gates"].items())
              + f"; {info['unmatched_detections']} of {info['detections']} detections matched no gate")
    os.makedirs(out, exist_ok=True)
    er.write_csv(os.path.join(out, "gate_errors.csv"), rows)
    json.dump(info, open(os.path.join(out, "info.json"), "w"), indent=2)
    for path in pge.plot_all(rows, out, suffix):
        print(f"saved {path}")


if __name__ == "__main__":
    main()
