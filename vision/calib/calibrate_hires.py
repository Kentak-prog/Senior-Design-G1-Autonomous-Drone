"""Calibrate hires_front from the VOXL Portal MJPEG feed with the printed checkerboard.

Board: checkerboard_9x6_25mm_letter.pdf (9x6 inner corners, 25 mm squares; pass
--square-mm with what you actually measured on the print).

  capture: reads the feed headless, auto-saves a frame when the board is found, held
           still, and in a new place/size/tilt compared with the frames already saved.
           Writes <out>/coverage.png (all saved corners drawn on one image) and prints
           progress, so whoever moves the board knows where to put it next.
  solve:   cv2.fisheye calibration on the saved frames, drops outlier frames, writes a
           VOXL/OpenCV yaml (M, D, width, height, distortion_model: fisheye).

    adb forward tcp:8080 tcp:80
    python3 calibrate_hires.py capture --target 40
    python3 calibrate_hires.py solve
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from live_gate_test import MjpegReader  # noqa: E402

URL = "http://localhost:8080/video_raw/hires_front_small_color"
PATTERN = (9, 6)
SB_FLAGS = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY


def detect(gray):
    ok, c = cv2.findChessboardCornersSB(gray, PATTERN, flags=SB_FLAGS)
    return c.reshape(-1, 2).astype(np.float64) if ok else None


def pose_feature(c, w, h):
    """Where the board is (centroid), how big (scale) and how tilted (side ratios)."""
    g = c.reshape(PATTERN[1], PATTERN[0], 2)
    tl, tr, bl, br = g[0, 0], g[0, -1], g[-1, 0], g[-1, -1]
    top, bot = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    lef, rig = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
    cx, cy = c.mean(0) / (w, h)
    scale = np.sqrt(cv2.contourArea(np.float32([tl, tr, br, bl]))) / np.hypot(w, h)
    return np.array([cx, cy, 2 * scale, np.log(top / bot), np.log(lef / rig)])


def coverage_image(frame, saved_corners, live=None):
    img = (frame * 0.5).astype(np.uint8)
    for c in saved_corners:
        hull = cv2.convexHull(c.astype(np.float32)).astype(np.int32)
        cv2.polylines(img, [hull], True, (0, 255, 0), 1)
        for p in c:
            cv2.circle(img, tuple(int(v) for v in p), 2, (0, 255, 0), -1)
    if live is not None:
        cv2.polylines(img, [cv2.convexHull(live.astype(np.float32)).astype(np.int32)],
                      True, (0, 0, 255), 2)
    h, w = img.shape[:2]
    for i in (1, 2):                                   # 3x3 grid to judge coverage
        cv2.line(img, (w * i // 3, 0), (w * i // 3, h), (255, 255, 0), 1)
        cv2.line(img, (0, h * i // 3), (w, h * i // 3), (255, 255, 0), 1)
    return img


def grid_counts(saved_corners, w, h):
    n = np.zeros((3, 3), int)
    for c in saved_corners:
        for p in c:
            n[min(int(p[1] * 3 / h), 2), min(int(p[0] * 3 / w), 2)] += 1
    return n


def capture(a):
    os.makedirs(a.out, exist_ok=True)
    rd = MjpegReader(a.url)
    saved, feats = [], []
    for f in sorted(glob.glob(os.path.join(a.out, "frame_*.png"))):   # resume
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        c = detect(g)
        if c is not None:
            saved.append(c)
            feats.append(pose_feature(c, g.shape[1], g.shape[0]))
    print(f"[capture] {len(saved)} frames already in {a.out}; target {a.target}", flush=True)
    seq, prev, t_end, last_status = 0, None, time.time() + a.duration, 0.0
    while len(saved) < a.target and time.time() < t_end:
        if rd.error:
            sys.exit(f"[capture] feed error: {rd.error}")
        seq, _, frame = rd.latest(seq)
        if frame is None:
            time.sleep(0.01)
            continue
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        c = detect(gray)
        status = "no board"
        if c is not None:
            still = prev is not None and np.abs(c - prev).max() < a.still_px
            f = pose_feature(c, w, h)
            d = min((np.linalg.norm(f - g) for g in feats), default=np.inf)
            status = f"board, {'still' if still else 'moving'}, novelty {d:.2f}"
            if still and d > a.novelty:
                name = os.path.join(a.out, f"frame_{len(saved):03d}.png")
                cv2.imwrite(name, frame)
                saved.append(c)
                feats.append(f)
                cv2.imwrite(os.path.join(a.out, "coverage.png"), coverage_image(frame, saved))
                print(f"[capture] saved {len(saved)}/{a.target}  centre ({f[0]:.2f},{f[1]:.2f}) "
                      f"size {f[2]:.2f}  grid corners per cell:\n{grid_counts(saved, w, h)}", flush=True)
        prev = c
        if time.time() - last_status > 5:
            print(f"[capture] {status}", flush=True)
            cv2.imwrite(os.path.join(a.out, "coverage.png"), coverage_image(frame, saved, c))
            last_status = time.time()
    rd.stop()
    print(f"[capture] done: {len(saved)} frames", flush=True)


def fisheye_calib(objp, imgs, size):
    # seed K (fx ~ EXIF 3.1 mm scaled to this width); identity K makes InitExtrinsics fail
    f0 = 3.1e-3 / 1.55e-6 * size[0] / 4056
    K = np.array([[f0, 0, size[0] / 2], [0, f0, size[1] / 2], [0, 0, 1]], float)
    D = np.zeros((4, 1))
    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
             | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS)
    crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-9)
    rms, K, D, rv, tv = cv2.fisheye.calibrate([objp] * len(imgs), imgs, size, K, D,
                                              flags=flags, criteria=crit)
    per = []
    for i, c in enumerate(imgs):
        p, _ = cv2.fisheye.projectPoints(objp, rv[i], tv[i], K, D)
        per.append(np.sqrt(np.mean(np.sum((p - c) ** 2, -1))))
    return rms, K, D, np.array(per)


def solve(a):
    files = sorted(glob.glob(os.path.join(a.out, "frame_*.png")))
    objp = np.zeros((1, PATTERN[0] * PATTERN[1], 3), np.float64)
    objp[0, :, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * a.square_mm / 1000
    imgs, names, size = [], [], None
    for f in files:
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        size = (g.shape[1], g.shape[0])
        c = detect(g)
        if c is not None:
            imgs.append(c.reshape(1, -1, 2))
            names.append(os.path.basename(f))
    print(f"[solve] {len(imgs)}/{len(files)} frames with the board, image {size}")
    if len(imgs) < 10:
        sys.exit("[solve] need at least 10 frames")
    while True:
        rms, K, D, per = fisheye_calib(objp, imgs, size)
        bad = per > max(3 * np.median(per), 1.0)
        if not bad.any() or len(imgs) - bad.sum() < 10:
            break
        print(f"[solve] dropping {[n for n, b in zip(names, bad) if b]} (rms {per[bad].round(2)} px)")
        imgs = [c for c, b in zip(imgs, bad) if not b]
        names = [n for n, b in zip(names, bad) if not b]

    w, h = size
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def theta(r):                              # distorted radius (normalised) -> angle
        t = r
        for _ in range(50):
            f = t * (1 + D[0, 0] * t**2 + D[1, 0] * t**4 + D[2, 0] * t**6 + D[3, 0] * t**8) - r
            df = 1 + 3 * D[0, 0] * t**2 + 5 * D[1, 0] * t**4 + 7 * D[2, 0] * t**6 + 9 * D[3, 0] * t**8
            t -= f / df
        return np.degrees(t)
    hfov = theta(cx / fx) + theta((w - cx) / fx)
    vfov = theta(cy / fy) + theta((h - cy) / fy)
    print(f"[solve] {len(imgs)} frames, RMS {rms:.3f} px (per-frame max {per.max():.2f})")
    print(f"[solve] fx {fx:.2f}  fy {fy:.2f}  cx {cx:.2f}  cy {cy:.2f}")
    print(f"[solve] D {D.ravel().round(5).tolist()}")
    print(f"[solve] FOV {hfov:.1f} x {vfov:.1f} deg; implied focal length "
          f"{fx * 4056 / w * 1.55e-3:.2f} mm (if the 1024 stream is the full 4056 sensor width)")

    fs = cv2.FileStorage(a.yml, cv2.FILE_STORAGE_WRITE)
    fs.writeComment(f"hires_front (IMX412) small_color stream {w}x{h}, VOXL Portal MJPEG. "
                    f"{len(imgs)} frames, {PATTERN[0]}x{PATTERN[1]} corners, {a.square_mm} mm squares.")
    fs.write("M", K)
    fs.write("D", D)
    fs.write("reprojection_error", float(rms))
    fs.write("width", w)
    fs.write("height", h)
    fs.write("distortion_model", "fisheye")
    fs.write("camera", "hires_front")
    fs.write("calibration_time", time.strftime("%Y-%m-%d %H:%M:%S"))
    fs.release()
    print(f"[solve] wrote {a.yml}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["capture", "solve"])
    ap.add_argument("--url", default=URL)
    ap.add_argument("--out", default=os.path.join(_HERE, "hires_front_frames"))
    ap.add_argument("--yml", default=os.path.join(_HERE, "hires_front_1024x768.yml"))
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--target", type=int, default=40)
    ap.add_argument("--duration", type=float, default=900, help="capture time limit, s")
    ap.add_argument("--still-px", type=float, default=1.5, help="max corner motion between frames")
    ap.add_argument("--novelty", type=float, default=0.12, help="min pose-feature distance to keep")
    a = ap.parse_args()
    capture(a) if a.mode == "capture" else solve(a)


if __name__ == "__main__":
    main()
