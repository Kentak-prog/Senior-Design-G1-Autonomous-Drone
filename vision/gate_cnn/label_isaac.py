"""Turn real Isaac Sim camera frames into CNN training data, with almost no hand labelling.

Idea: in clean, single-gate Isaac frames the colour detector is already accurate, so it can
label them automatically. You then glance through and reject bad ones. The CNN is fine-tuned
on these real frames mixed with synthetic ones, and learns the real look of the Isaac gates
(lighting, edges, camera) while keeping its robustness training.

Usage:
    1. Save front-camera frames from Isaac as PNGs into isaac_raw/
       (any resolution with a 60:36 aspect; a few hundred from different spots is plenty)
    2. python label_isaac.py isaac_raw/ isaac_frames/
       -> a window per frame: press  y = keep,  n = skip,  q = quit
       (add --auto to keep every frame without the review window)
    3. python train.py --real isaac_frames/ --init gate_cnn.npz --steps 1500 --lr 5e-4
"""
import argparse
import glob
import os

import cv2
import numpy as np

import baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--auto", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    kept = 0
    for path in sorted(glob.glob(os.path.join(a.src, "*.png"))):
        img = cv2.imread(path)
        quad, mask = baseline.detect(img)
        if quad is None:
            continue
        # skip frames where the gate touches the border (corners would be clipped, so wrong)
        h, w = img.shape[:2]
        if (quad[:, 0] < 3).any() or (quad[:, 0] > w - 4).any() or (quad[:, 1] < 3).any() or (quad[:, 1] > h - 4).any():
            continue
        if not a.auto:
            vis = img.copy()
            cv2.polylines(vis, [np.round(quad).astype(np.int32)], True, (0, 255, 0), 1, cv2.LINE_AA)
            for k, p in enumerate(quad):
                cv2.putText(vis, "TL TR BR BL".split()[k], tuple(np.round(p).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.imshow("y = keep, n = skip, q = quit", cv2.resize(vis, None, fx=2, fy=2))
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break
            if key != ord("y"):
                continue
        name = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(a.dst, name + ".png"), img)
        np.savetxt(os.path.join(a.dst, name + ".txt"), quad.reshape(1, 8), fmt="%.2f")
        kept += 1
    cv2.destroyAllWindows()
    print(f"kept {kept} labelled frames in {a.dst}")


if __name__ == "__main__":
    main()
