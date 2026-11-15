"""Auto-label real pink-gate frames for a YOLO pose model (1 class, 4 keypoints = opening corners).

Labels come from the color detector's CLEAN fixes in vision/live_runs/<run>/detections.csv
(reprojection <= --max-reproj px, range < --max-range m). A frame is used only if every
near (< max-range) detection in it is clean; far detections (the 11-13 m false quads) are
left unlabeled, i.e. taught as background. Frames with no detection are NOT used as
negatives: the color detector misses real gates (shadow, occlusion), so those frames
would teach the CNN to miss them too.

Keypoints are ordered with gate_pose.order_corners (same order PnP uses). The box is the
corner quad's bbox grown by --pad of its size (covers the pipe around the opening).

    python make_dataset.py --train pink_gate_01 --test pink_gate_03 --out datasets/pink_v1
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
VISION = os.path.dirname(HERE)
sys.path.insert(0, VISION)
from gate_pose import order_corners  # noqa: E402

RUNS = os.path.join(VISION, "live_runs")


def labels_for(run, max_reproj, max_range):
    d = pd.read_csv(os.path.join(RUNS, run, "detections.csv"))
    near = d[d.range_m < max_range]
    out = {}
    for f, g in near.groupby("frame"):
        if (g.reproj_px > max_reproj).any():
            continue
        out[int(f)] = [order_corners(r[["c0x", "c0y", "c1x", "c1y", "c2x", "c2y", "c3x", "c3y"]]
                                     .to_numpy(float).reshape(4, 2)) for _, r in g.iterrows()]
    return out


def write(split, run, labels, out, step, pad):
    img_dir, lab_dir = os.path.join(out, "images", split), os.path.join(out, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)
    n = 0
    for f in sorted(labels)[::step]:
        src = os.path.join(RUNS, run, "raw_jpg", f"{f:05d}.jpg")
        img = cv2.imread(src)
        if img is None:
            continue
        h, w = img.shape[:2]
        name = f"{run}_{f:05d}"
        dst = os.path.join(img_dir, name + ".jpg")
        if not os.path.exists(dst):
            os.symlink(src, dst)
        rows = []
        for c in labels[f]:
            (x0, y0), (x1, y1) = c.min(0), c.max(0)
            px, py = pad * (x1 - x0), pad * (y1 - y0)
            x0, y0, x1, y1 = max(x0 - px, 0), max(y0 - py, 0), min(x1 + px, w - 1), min(y1 + py, h - 1)
            kp = " ".join(f"{x / w:.6f} {y / h:.6f} 2" for x, y in c)
            rows.append(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                        f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f} {kp}")
        open(os.path.join(lab_dir, name + ".txt"), "w").write("\n".join(rows) + "\n")
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-reproj", type=float, default=3.0)
    ap.add_argument("--max-range", type=float, default=4.0)
    ap.add_argument("--step", type=int, default=2, help="keep every Nth labelled train frame (30 fps is redundant)")
    ap.add_argument("--val-frac", type=float, default=0.15, help="last fraction of each train run -> val")
    ap.add_argument("--pad", type=float, default=0.12)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    counts = {"train": 0, "val": 0, "test": 0}
    for run in a.train:
        lab = labels_for(run, a.max_reproj, a.max_range)
        fr = sorted(lab)
        cut = fr[int(len(fr) * (1 - a.val_frac))]       # time-split, not random: neighbours are near-duplicates
        counts["train"] += write("train", run, {f: lab[f] for f in fr if f < cut - 30}, out, a.step, a.pad)
        counts["val"] += write("val", run, {f: lab[f] for f in fr if f >= cut}, out, a.step, a.pad)
    for run in a.test:
        counts["test"] += write("test", run, labels_for(run, a.max_reproj, a.max_range), out, 1, a.pad)
    open(os.path.join(out, "data.yaml"), "w").write(
        f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: gate\nkpt_shape: [4, 3]\nflip_idx: [1, 0, 3, 2]\n")
    print(counts, "->", out)


if __name__ == "__main__":
    main()
