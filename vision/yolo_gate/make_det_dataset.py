"""Auto-label real pink-gate frames for a YOLO DETECT model (1 class "gate", bbox only).

Labels come from the color detector's CLEAN fixes (make_dataset.labels_for: reprojection
<= 3 px, range < 4 m; far false quads stay unlabelled = background). The box is the gate's
OUTER silhouette: the opening corners give the pose (PnP, hires_front calibration), and the
outer square (opening + 2 pipe diameters, 0.914 m for the 36 in gates) is projected back
through the fisheye lens, sampled along each edge so the barrel-curved sides are enclosed.

Each run is split by TIME (30 fps neighbours are near-duplicates): first part train, then
val, then test, with a 30-frame gap between parts.

    python make_det_dataset.py --runs pink_gate_01:2 pink_gate_02:1 pink_gate_03:1 --out datasets/pink_det_v1
      (run:step keeps every step-th labelled frame of that run)
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VISION = os.path.dirname(HERE)
sys.path.insert(0, VISION)
sys.path.insert(0, HERE)
from camera import project_points, transform_points  # noqa: E402
from gate_pose import estimate_gate_pose, gate_model_points  # noqa: E402
from live_gate_test import load_calib, Lens  # noqa: E402
from make_dataset import labels_for, RUNS  # noqa: E402


def outer_box(lens, corners, opening, outer, n=24):
    g = estimate_gate_pose(lens.to_pinhole(corners), lens.intr, gate_side=opening, bootstrap_n=0)
    if g is None:
        return None
    c = gate_model_points(outer)
    pts = np.vstack([c[i] + (c[(i + 1) % 4] - c[i]) * t for i in range(4) for t in np.linspace(0, 1, n)])
    px = lens.to_image(project_points(transform_points(g.T_cam_gate, pts), lens.intr))
    return px.min(0), px.max(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run[:step]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", default=os.path.join(VISION, "calib", "hires_front_1024x768.yml"))
    ap.add_argument("--opening", type=float, default=0.8636, help="inner opening side, m")
    ap.add_argument("--outer", type=float, default=0.9144, help="outer side incl. pipe, m")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.20)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    K, D, fe, cs = load_calib(a.calib)
    lens = Lens(K, D, fe, cs, (1024, 768), True)

    counts = {"train": 0, "val": 0, "test": 0}
    for spec in a.runs:
        run, step = (spec.split(":") + ["1"])[:2]
        lab = labels_for(run, 3.0, 4.0)
        fr = sorted(lab)
        n = len(fr)
        c_val, c_test = fr[int(n * (1 - a.val_frac - a.test_frac))], fr[int(n * (1 - a.test_frac))]
        splits = {"train": [f for f in fr if f < c_val - 30][::int(step)],
                  "val": [f for f in fr if c_val <= f < c_test - 30],
                  "test": [f for f in fr if f >= c_test]}
        for split, frames in splits.items():
            img_dir, lab_dir = os.path.join(out, "images", split), os.path.join(out, "labels", split)
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lab_dir, exist_ok=True)
            for f in frames:
                src = os.path.join(RUNS, run, "raw_jpg", f"{f:05d}.jpg")
                if not os.path.exists(src):
                    continue
                w, h = 1024, 768
                rows = []
                for c in lab[f]:
                    b = outer_box(lens, c, a.opening, a.outer)
                    if b is None:
                        rows = None
                        break
                    (x0, y0), (x1, y1) = np.clip(b[0], 0, [w - 1, h - 1]), np.clip(b[1], 0, [w - 1, h - 1])
                    rows.append(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                                f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
                if not rows:
                    continue
                name = f"{run}_{f:05d}"
                dst = os.path.join(img_dir, name + ".jpg")
                if not os.path.exists(dst):
                    os.symlink(src, dst)
                open(os.path.join(lab_dir, name + ".txt"), "w").write("\n".join(rows) + "\n")
                counts[split] += 1
        print(run, {k: len(v) for k, v in splits.items()})
    open(os.path.join(out, "data.yaml"), "w").write(
        f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: gate\n")
    print(counts, "->", out)


if __name__ == "__main__":
    main()
