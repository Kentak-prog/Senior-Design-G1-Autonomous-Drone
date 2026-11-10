"""Run the CNN (and the colour detector, for comparison) on a recorded Isaac video or a folder of frames.

    python run_on_video.py isaac_stage4_course_truth.webm --crop 70,58,635,345 --out isaac_test.mp4
    python run_on_video.py isaac_raw/ --out isaac_test.mp4

--crop x,y,w,h cuts the drone camera viewport out of a screen recording of the Isaac window.
(Raw camera frames saved straight from the sensor don't need it.)

Green = CNN + colour refine, red = color threshold alone. A detection is drawn only when the
CNN is confident and the whole gate is in view; otherwise the frame says "no full gate".
Pass --save-frames DIR to also save the cropped frames for label_isaac.py.
"""
import argparse
import glob
import os

import cv2
import numpy as np

import baseline
from detector import GateCNN, refine_with_colour, good_detection


def frames(src, crop, step):
    if os.path.isdir(src):
        for i, p in enumerate(sorted(glob.glob(os.path.join(src, "*.png")) + glob.glob(os.path.join(src, "*.jpg")))):
            if i % step == 0:
                yield cv2.imread(p)
        return
    cap = cv2.VideoCapture(src)
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            if crop:
                x, y, w, h = crop
                f = f[y:y + h, x:x + w]
            yield f
        i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--crop", default=None)
    ap.add_argument("--out", default="isaac_test.mp4")
    ap.add_argument("--step", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--fps", type=float, default=20)
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--save-frames", default=None)
    a = ap.parse_args()
    crop = [int(v) for v in a.crop.split(",")] if a.crop else None
    cnn = GateCNN(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gate_cnn.npz"))
    if a.save_frames:
        os.makedirs(a.save_frames, exist_ok=True)
    writer, n, n_det, n_col = None, 0, 0, 0
    for f in frames(a.src, crop, a.step):
        if a.save_frames:
            cv2.imwrite(os.path.join(a.save_frames, f"frame_{n:05d}.png"), f)
        c, conf = cnn.detect(f)
        good = good_detection(f, c, conf, a.conf)
        if good:
            c = refine_with_colour(f, c)
        b, _ = baseline.detect(f)
        v = f.copy()
        lw = max(2, f.shape[1] // 300)
        if b is not None:
            n_col += 1
            cv2.polylines(v, [np.round(b).astype(np.int32)], True, (60, 60, 255), lw, cv2.LINE_AA)
        if good:
            n_det += 1
            pts = np.round(c).astype(np.int32)
            cv2.polylines(v, [pts], True, (80, 255, 80), lw, cv2.LINE_AA)
            for k, p in enumerate(pts):
                cv2.circle(v, tuple(p), lw * 3, (80, 255, 80), -1, cv2.LINE_AA)
        label = f"CNN conf {conf.min():.2f}" + ("" if good else "  (no full gate)")
        cv2.putText(v, label, (10, f.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
        strip = np.zeros((30, v.shape[1], 3), np.uint8)
        cv2.putText(strip, "green = CNN detection (passed all checks)    red = color threshold alone",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        v = np.vstack([v, strip])
        if writer is None:
            writer = cv2.VideoWriter("_tmp_raw.mp4", cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (v.shape[1], v.shape[0]))
        writer.write(v)
        n += 1
    writer.release()
    import shutil, subprocess
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "_tmp_raw.mp4", "-c:v", "libx264", "-preset",
                        "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", a.out], check=True)
        os.remove("_tmp_raw.mp4")
    else:
        os.replace("_tmp_raw.mp4", a.out)
    print(f"{n} frames -> {a.out}.  CNN full-gate detections: {n_det}.  colour blobs found: {n_col}")


if __name__ == "__main__":
    main()
