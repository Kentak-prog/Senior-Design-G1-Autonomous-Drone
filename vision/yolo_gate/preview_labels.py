"""Low-res JPEG previews of every labelled image in a YOLO pose dataset, to eyeball label accuracy.

Draws the box (cyan), the 4 opening corners (green, TL/TR/BR/BL) and the opening outline,
plus a contact sheet per split (6x5 thumbnails, one sheet per 30 images).

    python preview_labels.py datasets/pink_v1          # -> datasets/pink_v1/previews/{train,val,test,sheets}
"""
import glob
import os
import sys

import cv2
import numpy as np

ds = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "datasets/pink_v1")
W = int(sys.argv[2]) if len(sys.argv) > 2 else 512
out = os.path.join(ds, "previews")
NAMES = ("TL", "TR", "BR", "BL")

for split in ("train", "val", "test"):
    imgs = sorted(glob.glob(os.path.join(ds, "images", split, "*.jpg")))
    os.makedirs(os.path.join(out, split), exist_ok=True)
    thumbs = []
    for p in imgs:
        img = cv2.imread(p)
        h, w = img.shape[:2]
        s = W / w
        img = cv2.resize(img, (W, int(h * s)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        lab = p.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)[:-4] + ".txt"
        for row in open(lab).read().split("\n"):
            if not row.strip():
                continue
            v = list(map(float, row.split()))
            cx, cy, bw, bh = v[1:5]
            cv2.rectangle(img, (int((cx - bw / 2) * w), int((cy - bh / 2) * h)),
                          (int((cx + bw / 2) * w), int((cy + bh / 2) * h)), (255, 255, 0), 1)
            if len(v) < 17:                    # detect labels: box only
                continue
            kp =(np.array(v[5:]).reshape(4, 3)[:, :2] * (w, h))
            cv2.polylines(img, [kp.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 1)
            for i, (x, y) in enumerate(kp):
                cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)
                cv2.putText(img, NAMES[i], (int(x) + 4, int(y) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (0, 255, 0), 1)
        name = os.path.basename(p)
        cv2.putText(img, name[:-4], (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(img, name[:-4], (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(out, split, name), img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        thumbs.append(cv2.resize(img, (256, 192), interpolation=cv2.INTER_AREA))
    os.makedirs(os.path.join(out, "sheets"), exist_ok=True)
    for k in range(0, len(thumbs), 30):
        t = thumbs[k:k + 30]
        t += [np.zeros_like(thumbs[0])] * (30 - len(t))
        sheet = np.vstack([np.hstack(t[r * 6:(r + 1) * 6]) for r in range(5)])
        cv2.imwrite(os.path.join(out, "sheets", f"{split}_{k // 30:03d}.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 75])
    print(f"{split}: {len(imgs)} previews")
print("->", out)
