"""
Per-frame gate-error records: pose interpolation, row building, CSV I/O and
summary statistics for the offline bag evaluation.

WHY THIS MODULE EXISTS (and why it is split out):
    The bag evaluator (eval_gate_bag.py) needs ROS; the plotting script
    (plot_gate_errors.py) must run on a Mac with only numpy + matplotlib.
    Everything that both sides share -- the CSV schema, the pose buffer, the
    row builder, the summary table -- lives here with NO ROS and NO
    matplotlib. The vision-heavy imports (cv2 via gate_pose / ros_camera /
    eval_isaac_gates) are done lazily inside the two functions that need them,
    so read_csv() / summarize() work on a machine without cv2 as well.

FRAME CONVENTIONS (see vision/HANDOFF.md -- unchanged here):
    * /drone00/state/pose is a PoseStamped in `map`, quaternion XYZW.
    * T_world_cam_raw = T_map_body @ T_body_cam_usd_from_mount() is the raw
      USD-convention camera pose. optical_frame=False (R_CV_FROM_USD
      correction) is the settled convention and the ONLY one used here, via
      the `per_convention[False]` result of eval_isaac_gates.evaluate_frame.
    * Gates are static, so ground truth is the manifest alone; only the
      camera pose varies from frame to frame.
    * Transform naming: T_a_b maps frame b -> frame a.
"""

import csv
import math
import os
import sys

import numpy as np

# vision/ is the parent directory; make it importable so `camera`,
# `eval_isaac_gates` etc. resolve when this file is run from gate_error_eval/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera import (CameraIntrinsics, invert_transform, transform_points,  # noqa: E402
                    project_points, in_frame)

CSV_FIELDS = [
    "t_s", "frame", "gate", "detected",
    "est_x", "est_y", "est_z",
    "true_x", "true_y", "true_z",
    "drone_x", "drone_y", "drone_z", "drone_speed_mps",
    "range_true_m", "off_axis_deg", "in_view",
    "pos_err_m", "range_err_m", "lateral_err_m", "normal_err_deg",
    "reproj_rms_px", "normal_is_reliable",
]
_BOOL_FIELDS = {"detected", "in_view", "normal_is_reliable"}
_STR_FIELDS = {"gate"}
_INT_FIELDS = {"frame"}


# --------------------------------------------------------------------------
# Pose interpolation
# --------------------------------------------------------------------------

def _normalize_quat(q) -> np.ndarray:
    q = np.asarray(q, dtype=float).ravel()
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("zero-norm quaternion")
    return q / n


def slerp_xyzw(q0, q1, u: float) -> np.ndarray:
    """
    Spherical linear interpolation between two XYZW quaternions, u in [0, 1].

    q and -q are the same rotation; if dot(q0, q1) < 0 the shorter arc is
    found by negating q1 first, so a hemisphere flip between consecutive
    pose messages never makes the interpolated attitude swing the long way
    round.
    """
    q0 = _normalize_quat(q0)
    q1 = _normalize_quat(q1)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        # Nearly parallel: sin(theta) -> 0, fall back to a normalized lerp.
        return _normalize_quat(q0 + u * (q1 - q0))
    theta = math.acos(min(1.0, d))
    s = math.sin(theta)
    return (math.sin((1.0 - u) * theta) * q0 + math.sin(u * theta) * q1) / s


class PoseBuffer:
    """
    Time-sorted (t, position(3), quat_xyzw(4)) samples with interpolation.

    t is in seconds on whatever clock the caller chose (header stamp or bag
    receive time -- see eval_gate_bag.py); the buffer does not care.
    """

    def __init__(self, max_extrap_s: float = 0.05):
        self.max_extrap_s = float(max_extrap_s)
        self._t, self._p, self._q = [], [], []
        self._arr = None  # cached sorted numpy arrays, rebuilt after add()

    def add(self, t: float, position, quat_xyzw) -> None:
        self._t.append(float(t))
        self._p.append(np.asarray(position, dtype=float).ravel())
        self._q.append(_normalize_quat(quat_xyzw))
        self._arr = None

    def __len__(self) -> int:
        return len(self._t)

    def _sorted(self):
        if self._arr is None:
            order = np.argsort(np.asarray(self._t), kind="stable")
            self._arr = (np.asarray(self._t)[order],
                         np.asarray(self._p).reshape(-1, 3)[order],
                         np.asarray(self._q).reshape(-1, 4)[order])
        return self._arr

    @property
    def t_first(self) -> float:
        return float(self._sorted()[0][0])

    @property
    def t_last(self) -> float:
        return float(self._sorted()[0][-1])

    def times(self) -> np.ndarray:
        return self._sorted()[0]

    def interpolate(self, t: float):
        """(position, quat_xyzw) at time t, or None if t is more than
        max_extrap_s outside [first, last] (or the buffer is empty).
        Inside the tolerance the query is clamped to the end sample."""
        if not self._t:
            return None
        ts, ps, qs = self._sorted()
        if t < ts[0] - self.max_extrap_s or t > ts[-1] + self.max_extrap_s:
            return None
        t = min(max(t, ts[0]), ts[-1])
        if len(ts) == 1:
            return ps[0].copy(), qs[0].copy()
        i = int(np.searchsorted(ts, t, side="right")) - 1
        i = min(max(i, 0), len(ts) - 2)
        dt = ts[i + 1] - ts[i]
        u = 0.0 if dt <= 0.0 else (t - ts[i]) / dt
        pos = ps[i] + u * (ps[i + 1] - ps[i])
        return pos, slerp_xyzw(qs[i], qs[i + 1], u)

    def velocity(self, t: float, window_s: float = 0.1):
        """
        Velocity vector (3,) in m/s from a finite difference of the samples
        within +-window_s/2 of t (at least the two neighbours bracketing t).
        A window rather than the single bracketing pair because 130 Hz pose
        samples are quantized in time and a 7 ms difference is noisy. Returns
        None outside the interpolation range or with fewer than 2 samples.
        """
        if len(self._t) < 2 or self.interpolate(t) is None:
            return None
        ts, ps, _ = self._sorted()
        lo = int(np.searchsorted(ts, t - window_s / 2.0, side="left"))
        hi = int(np.searchsorted(ts, t + window_s / 2.0, side="right")) - 1
        lo = min(max(lo, 0), len(ts) - 2)
        hi = min(max(hi, lo + 1), len(ts) - 1)
        i = int(np.searchsorted(ts, t, side="right")) - 1
        i = min(max(i, 0), len(ts) - 2)
        lo, hi = min(lo, i), max(hi, i + 1)
        dt = ts[hi] - ts[lo]
        if dt <= 0.0:
            return None
        return (ps[hi] - ps[lo]) / dt


def T_world_cam_raw_at(pose_buffer: PoseBuffer, t: float):
    """
    4x4 raw (USD-convention) camera pose at time t: the interpolated
    /drone00/state/pose composed with the static body->camera mount, exactly
    as eval_isaac_gates.py --camera-pose pose does. None if t is outside the
    buffer (beyond its extrapolation tolerance).
    """
    # Lazy: pulls in cv2 via eval_isaac_gates -> gate_pose; see module docstring.
    from eval_isaac_gates import T_body_cam_usd_from_mount
    from ros_camera import transform_to_matrix
    got = pose_buffer.interpolate(t)
    if got is None:
        return None
    pos, quat_xyzw = got
    # transform_to_matrix takes plain length-3 / length-4 (xyzw) sequences.
    T_map_body = transform_to_matrix(pos, quat_xyzw)
    return T_map_body @ T_body_cam_usd_from_mount()


# --------------------------------------------------------------------------
# Row building
# --------------------------------------------------------------------------

def frame_rows(t_s, frame_idx, intr: CameraIntrinsics, T_world_cam_raw,
               manifest: list, eval_result: dict, drone_pos, drone_speed,
               gate_side: float = None, in_view_margin_px: int = 3) -> list:
    """
    One row dict per manifest gate for one evaluated frame.

    Uses ONLY the optical_frame=False branch of eval_result (the settled
    convention). Truth-side columns (range_true_m, off_axis_deg, in_view) are
    computed from the manifest and the camera pose alone, so they are
    filled for undetected gates too -- that is what makes detection-rate vs
    range plots possible. Estimate/error columns are None when the gate was
    not matched.

    range_true_m is truth depth along camera +Z (OpenCV frame); off_axis_deg
    is the angle between the optical axis and the ray to the true gate
    center; in_view means all four true corners of the gate opening project
    inside the image (by in_view_margin_px) in front of the camera. Not just
    the center: in the first moving run (2026-10-01) gates above the drone
    climbed out of the top of the frame on approach, center still in view,
    and a center-only test counted those cut-off gates as detector misses.
    """
    from ros_camera import T_world_cam_from_transform
    from eval_isaac_gates import T_world_gate_from_spec
    from gate_pose import gate_model_points, DEFAULT_GATE_SIDE
    model = gate_model_points(DEFAULT_GATE_SIDE if gate_side is None else gate_side)
    T_world_cam = T_world_cam_from_transform(np.asarray(T_world_cam_raw, dtype=float), False)
    T_cam_world = invert_transform(T_world_cam)
    matches = {m["gate"]: m for m in eval_result["per_convention"][False]["matches"]}
    drone_pos = np.asarray(drone_pos, dtype=float).ravel()

    rows = []
    for g in manifest:
        truth = np.array([g["x"], g["y"], g["z"]], dtype=float)
        p_cam = transform_points(T_cam_world, truth)[0]
        dist = float(np.linalg.norm(p_cam))
        off_axis = (float(np.degrees(np.arccos(np.clip(p_cam[2] / dist, -1.0, 1.0))))
                    if dist > 1e-9 else 0.0)
        T_world_gate = T_world_gate_from_spec(g["x"], g["y"], g["z"], g["yaw_deg"])
        corners_cam = transform_points(T_cam_world @ T_world_gate, model)
        visible = bool(np.all(corners_cam[:, 2] > 0.0) and
                       np.all(in_frame(project_points(corners_cam, intr), intr,
                                       margin=in_view_margin_px)))

        m = matches.get(g["name"])
        row = {
            "t_s": float(t_s), "frame": int(frame_idx), "gate": g["name"],
            "detected": m is not None,
            "est_x": None, "est_y": None, "est_z": None,
            "true_x": float(truth[0]), "true_y": float(truth[1]), "true_z": float(truth[2]),
            "drone_x": float(drone_pos[0]), "drone_y": float(drone_pos[1]),
            "drone_z": float(drone_pos[2]),
            "drone_speed_mps": None if drone_speed is None else float(drone_speed),
            "range_true_m": float(p_cam[2]), "off_axis_deg": off_axis,
            "in_view": visible,
            "pos_err_m": None, "range_err_m": None, "lateral_err_m": None,
            "normal_err_deg": None, "reproj_rms_px": None, "normal_is_reliable": None,
        }
        if m is not None:
            est = m["est_center_world"]
            row.update({
                "est_x": float(est[0]), "est_y": float(est[1]), "est_z": float(est[2]),
                "pos_err_m": float(m["pos_err_m"]),
                "range_err_m": float(m["range_err_m"]),
                "lateral_err_m": float(m["lateral_err_m"]),
                "normal_err_deg": float(m["normal_err_deg"]),
                "reproj_rms_px": float(m["reproj_rms_px"]),
                "normal_is_reliable": bool(m["normal_is_reliable"]),
            })
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------

def write_csv(path, rows) -> None:
    """Rows are dicts keyed by CSV_FIELDS; None is written as a blank cell."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in CSV_FIELDS})


def read_csv(path) -> list:
    """Inverse of write_csv: numerics -> float (frame -> int), 'True'/'False'
    -> bool, blank -> None, gate stays a string."""
    rows = []
    with open(path, newline="") as f:
        for rec in csv.DictReader(f):
            row = {}
            for k, v in rec.items():
                if v is None or v == "":
                    row[k] = None
                elif k in _STR_FIELDS:
                    row[k] = v
                elif k in _BOOL_FIELDS:
                    row[k] = (v == "True")
                elif k in _INT_FIELDS:
                    row[k] = int(float(v))
                else:
                    row[k] = float(v)
            rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

SUMMARY_FIELDS = ["gate", "n_frames_in_view", "n_detected", "detection_rate",
                  "pos_err_mean", "pos_err_std", "pos_err_median", "pos_err_p95",
                  "range_err_mean", "range_err_abs_mean", "lateral_err_mean",
                  "range_true_mean"]


def _stats(rows) -> dict:
    """Stats for one group of rows. n_detected / detection_rate count only
    rows where the gate was in view; error statistics are over those same
    detected in-view rows, so every number refers to the same population."""
    in_view = [r for r in rows if r.get("in_view")]
    det = [r for r in in_view if r.get("detected") and r.get("pos_err_m") is not None
           and math.isfinite(r["pos_err_m"])]
    nan = float("nan")

    def col(key):
        return np.array([r[key] for r in det if r.get(key) is not None], dtype=float)

    pos, rng_e, lat = col("pos_err_m"), col("range_err_m"), col("lateral_err_m")
    rt = np.array([r["range_true_m"] for r in det if r.get("range_true_m") is not None],
                  dtype=float)
    return {
        "n_frames_in_view": len(in_view),
        "n_detected": len(det),
        "detection_rate": (len(det) / len(in_view)) if in_view else nan,
        "pos_err_mean": float(pos.mean()) if pos.size else nan,
        "pos_err_std": float(pos.std()) if pos.size else nan,
        "pos_err_median": float(np.median(pos)) if pos.size else nan,
        "pos_err_p95": float(np.percentile(pos, 95)) if pos.size else nan,
        "range_err_mean": float(rng_e.mean()) if rng_e.size else nan,
        "range_err_abs_mean": float(np.abs(rng_e).mean()) if rng_e.size else nan,
        "lateral_err_mean": float(lat.mean()) if lat.size else nan,
        "range_true_mean": float(rt.mean()) if rt.size else nan,
    }


def summarize(rows) -> dict:
    """{gate_name: stats, ..., "ALL": stats pooled over every gate}."""
    out = {}
    for name in sorted({r["gate"] for r in rows}):
        out[name] = _stats([r for r in rows if r["gate"] == name])
    out["ALL"] = _stats(list(rows))
    return out


def write_summary_csv(path, summary: dict) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for gate, st in summary.items():
            w.writerow(dict(gate=gate, **st))


def format_summary_table(summary: dict) -> str:
    """Fixed-width text table of summarize()'s output."""
    cols = SUMMARY_FIELDS
    head = ["gate", "in_view", "det", "rate", "err_mean", "err_std", "err_med",
            "err_p95", "rng_err", "|rng_err|", "lat_err", "range_m"]
    lines = [" ".join(f"{h:>9}" if i else f"{h:<9}" for i, h in enumerate(head))]
    for gate, st in summary.items():
        cells = [f"{gate:<9}", f"{st['n_frames_in_view']:>9d}", f"{st['n_detected']:>9d}"]
        for k in cols[3:]:
            v = st[k]
            cells.append(f"{'--':>9}" if v != v else f"{v:>9.3f}")
        lines.append(" ".join(cells))
    return "\n".join(lines)
