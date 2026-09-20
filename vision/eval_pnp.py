"""
Task 2.2 — Perception accuracy characterization.

Monte Carlo evaluation of the gate pose estimator. Generates gates at known
poses, projects them through the camera model, corrupts the corner pixels with
Gaussian noise standing in for detector error, then measures how badly the
recovered pose misses.

This runs with no simulator and no detector, so the accuracy numbers exist
before either of those is finished. Once Tye's detector is up, replace the
synthetic noise with his measured corner-localization error and rerun.

    python eval_pnp.py            # summary table
    python eval_pnp.py --csv out.csv
"""

import argparse
import numpy as np

from camera import CameraIntrinsics, make_transform
from gate_pose import (estimate_gate_pose, project_gate, DEFAULT_GATE_SIDE)
from camera import in_frame


def random_rotation(max_angle_deg: float, rng: np.random.Generator) -> np.ndarray:
    """Random rotation about a random axis, angle uniform in [0, max]."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    ang = np.radians(rng.uniform(0.0, max_angle_deg))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


# Gate facing the camera head-on: gate +Z (normal) maps to -Z_cam, gate +Y
# (up) maps to -Y_cam, which is up in the image since image Y grows downward.
R_FACING = np.diag([1.0, -1.0, -1.0])


def sample_gate_pose(range_m: float, tilt_deg: float, lateral_frac: float,
                     rng: np.random.Generator) -> np.ndarray:
    """A gate at the given range, tilted and offset off the optical axis."""
    R = R_FACING @ random_rotation(tilt_deg, rng)
    lateral = rng.uniform(-lateral_frac, lateral_frac, size=2) * range_m
    t = np.array([lateral[0], lateral[1], range_m])
    return make_transform(R, t)


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(abs(a @ b), -1.0, 1.0))))


def run_trials(intr, ranges, noise_px, gate_side, tilt_deg, n_trials, seed=0):
    rng = np.random.default_rng(seed)
    rows = []

    for rng_m in ranges:
        for sigma in noise_px:
            pos_err, depth_err, lat_err, norm_err, reproj = [], [], [], [], []
            pred_std, unreliable = [], []
            attempts = 0
            while len(pos_err) < n_trials and attempts < n_trials * 20:
                attempts += 1
                T_true = sample_gate_pose(rng_m, tilt_deg, 0.15, rng)
                corners = project_gate(T_true, intr, gate_side)
                if not in_frame(corners, intr).all():
                    continue
                noisy = corners + rng.normal(0.0, sigma, size=corners.shape)

                det = estimate_gate_pose(noisy, intr, gate_side,
                                         assume_ordered=True,
                                         corner_sigma_px=sigma,
                                         bootstrap_n=12,
                                         seed=int(rng.integers(1 << 30)))
                if det is None:
                    continue

                t_true = T_true[:3, 3]
                t_est = det.T_cam_gate[:3, 3]
                err = t_est - t_true
                pos_err.append(np.linalg.norm(err))
                depth_err.append(abs(err[2]))
                lat_err.append(np.linalg.norm(err[:2]))
                norm_err.append(angle_between(T_true[:3, 2], det.T_cam_gate[:3, 2]))
                reproj.append(det.reprojection_error_px)
                pred_std.append(det.center_std_m)
                unreliable.append(0.0 if det.normal_is_reliable else 1.0)

            if not pos_err:
                continue
            rows.append({
                "range_m": rng_m,
                "noise_px": sigma,
                "n": len(pos_err),
                "pos_err_median_m": float(np.median(pos_err)),
                "pos_err_p95_m": float(np.percentile(pos_err, 95)),
                "depth_err_median_m": float(np.median(depth_err)),
                "lateral_err_median_m": float(np.median(lat_err)),
                "normal_err_median_deg": float(np.median(norm_err)),
                "predicted_std_median_m": float(np.median(pred_std)),
                "reproj_median_px": float(np.median(reproj)),
                "normal_unreliable_frac": float(np.mean(unreliable)),
            })
    return rows


def print_table(rows):
    hdr = (f"{'range':>6} {'noise':>6} {'pos p50':>9} {'pos p95':>9} "
           f"{'pred sd':>9} {'depth p50':>10} {'lat p50':>9} {'normal':>8} "
           f"{'bad n':>7}")
    print(hdr)
    print(f"{'(m)':>6} {'(px)':>6} {'(m)':>9} {'(m)':>9} {'(m)':>9} "
          f"{'(m)':>10} {'(m)':>9} {'(deg)':>8} {'(frac)':>7}")
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['range_m']:6.1f} {r['noise_px']:6.1f} "
              f"{r['pos_err_median_m']:9.3f} {r['pos_err_p95_m']:9.3f} "
              f"{r['predicted_std_median_m']:9.3f} "
              f"{r['depth_err_median_m']:10.3f} {r['lateral_err_median_m']:9.3f} "
              f"{r['normal_err_median_deg']:8.1f} "
              f"{r['normal_unreliable_frac']:7.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--hfov", type=float, default=90.0)
    ap.add_argument("--gate-side", type=float, default=DEFAULT_GATE_SIDE)
    ap.add_argument("--tilt-deg", type=float, default=25.0)
    ap.add_argument("--trials", type=int, default=250)
    ap.add_argument("--csv", type=str, default=None)
    args = ap.parse_args()

    intr = CameraIntrinsics.from_horizontal_fov(args.width, args.height, args.hfov)
    print(f"Camera: {args.width}x{args.height}, hFOV {args.hfov} deg, "
          f"fx = {intr.fx:.1f} px")
    print(f"Gate side: {args.gate_side} m, tilt up to {args.tilt_deg} deg, "
          f"{args.trials} trials per cell\n")

    ranges = [2.0, 4.0, 6.0, 8.0, 12.0, 16.0]
    noise = [0.5, 1.0, 2.0, 4.0]
    rows = run_trials(intr, ranges, noise, args.gate_side, args.tilt_deg,
                      args.trials)
    print_table(rows)

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
