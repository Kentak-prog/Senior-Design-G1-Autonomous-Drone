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


# -------------------------------------------------------------------------------------- vio

def load_vio(run_dir):
    q = list(csv.DictReader(open(os.path.join(run_dir, "qvio.csv"))))
    t = np.array([int(r["timestamp(ns)"]) * 1e-9 for r in q])
    p = np.array([[float(r[f"T_imu_wrt_vio_{a}(m)"]) for a in "xyz"] for r in q])
    rpy = np.array([[float(r["roll(rad)"]), float(r["pitch(rad)"]), float(r["yaw(rad)"])] for r in q])
    v = np.array([[float(r[f"vel_imu_wrt_vio_{a}(m/s)"]) for a in "xyz"] for r in q])
    return t, p, rpy, v


def _interp_pose(t_vio, p, rpy, t, euler):
    """Pose at time t: position linear, rotation of the nearer sample (VIO at ~30 Hz)."""
    i = int(np.searchsorted(t_vio, t))
    if i <= 0 or i >= len(t_vio):
        return None
    a = (t - t_vio[i - 1]) / (t_vio[i] - t_vio[i - 1])
    pos = p[i - 1] + a * (p[i] - p[i - 1])
    j = i - 1 if a < 0.5 else i
    R = EULER[euler](*rpy[j])
    return pos, R


def cam_pose(t_vio, p, rpy, t, euler):
    ip = _interp_pose(t_vio, p, rpy, t, euler)
    if ip is None:
        return None
    pos, R_w_imu = ip
    return R_w_imu @ R_IMU_CAM, pos + R_w_imu @ T_IMU_CAM_T            # R_w_cam, camera center


def world_estimates(frames, alldet, vio, offset, latency, euler, corners_only=True):
    """(frame, world xyz) of every detection (all det_idx) of the run."""
    t_vio, p, rpy, _ = vio
    out = []
    for f in sorted(alldet):
        if f not in frames:
            continue
        cp = cam_pose(t_vio, p, rpy, frames[f][0] - offset - latency, euler)
        if cp is None:
            continue
        R, c = cp
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
        refs = new
    lab = np.argmin([np.linalg.norm(P - r, axis=1) for r in refs], axis=0)
    counts = [int(np.sum(lab == k)) for k in range(len(refs))]
    order = np.argsort(counts)[::-1]                      # gate_A = the most-seen gate
    return [refs[k] for k in order], [counts[k] for k in order]


def fit_sync(frames, alldet, vio, offset):
    best, table = None, []
    for euler in EULER:
        for lat in np.arange(0.0, 0.801, 0.05):
            pts = world_estimates(frames, alldet, vio, offset, lat, euler)
            if len(pts) < 20:
                continue
            s, refs = _spread(pts)
            table.append((euler, round(float(lat), 2), round(s, 4)))
            if best is None or s < best[0]:
                best = (s, euler, float(lat), refs)
    return best, table


def _project(lens_K, lens_D, fisheye, Xc):
    import cv2
    pts = Xc.reshape(1, 1, 3).astype(np.float64)
    if fisheye:
        uv, _ = cv2.fisheye.projectPoints(pts, np.zeros(3), np.zeros(3), lens_K, lens_D.reshape(-1)[:4])
    else:
        uv, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), lens_K, lens_D)
    return uv.reshape(2)


def vio_rows(run_dir, calib):
    from live_gate_test import load_calib
    K, D, fisheye, csize = load_calib(calib)
    if csize and tuple(csize) != IMAGE_WH and csize[0]:
        s = IMAGE_WH[0] / csize[0]
        K = K.copy()
        K[:2] *= s
    wall0, mono0 = map(float, open(os.path.join(run_dir, "clock.txt")).read().split())
    offset = wall0 - mono0
    frames = read_detections(run_dir)
    alldet = read_all_detections(run_dir)
    vio = load_vio(run_dir)
    (spread, euler, lat, refs), table = fit_sync(frames, alldet, vio, offset)
    names = [f"gate_{chr(ord('A') + k)}" for k in range(len(refs))]
    t_vio, p, rpy, v = vio
    t0 = min(t for t, _ in frames.values())
    rows, n_unmatched, n_dets = [], 0, 0
    for f in sorted(frames):
        t = frames[f][0]
        tm = t - offset - lat
        cp = cam_pose(t_vio, p, rpy, tm, euler)
        if cp is None:
            continue
        R, c = cp
        i = min(max(int(np.searchsorted(t_vio, tm)), 0), len(t_vio) - 1)
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
                        drone_speed_mps=float(np.linalg.norm(v[i])), range_true_m=rng_true,
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
    info = dict(euler=euler, latency_s=lat, median_scatter_to_nearest_gate_m=spread,
                gate_windows_s={n: list(w) for n, w in windows.items()},
                gates={n: ref.tolist() for n, ref in zip(names, refs)}, detections=n_dets,
                unmatched_detections=n_unmatched, clock_offset_s=offset, n_vio=len(t_vio),
                sync_search=[dict(euler=e, latency_s=l, scatter_m=s) for e, l, s in table])
    return rows, info


# ------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("still")
    s.add_argument("--out-dir", default=os.path.join(LIVE_RUNS, "figs_still_tape"))
    v = sub.add_parser("vio")
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
        rows, info = vio_rows(a.run_dir, a.calib)
        name = os.path.basename(os.path.normpath(a.run_dir))
        suffix = a.title or "(real drone, hand-held, ref = median est.)"
        out = a.out_dir or os.path.join(a.run_dir, "figs")
        print(f"[{name}] VIO Euler {info['euler']}, image latency {info['latency_s']:.2f} s, "
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
