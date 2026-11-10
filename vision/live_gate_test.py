"""
Live real-camera gate test: VOXL Portal MJPEG feed -> gate detector -> PnP ->
overlay video + per-frame CSV.

    adb forward tcp:8080 tcp:80          # VOXL Portal on localhost:8080
    python3 live_gate_test.py --gate-side 0.914 --no-color-checks \
        --calib calib/hires_front_small_approx.yml --duration 60 --out-dir live_runs/real_gate_02

(36 in white PVC lab gate; --no-color-checks because Tye's CNN checks assume
purple. Ultralytics weights instead: --detector cnn --model best.pt.)

Feed: http://localhost:8080/video_raw/<pipe> is multipart MJPEG (boundary
"--foo"). It is parsed here by JPEG SOI/EOI markers, not by OpenCV's
VideoCapture, which is unreliable on multipart streams. Only the newest frame
is processed; older ones are dropped ("never queue").

Detector (--detector):
  cnn    Ultralytics YOLO weights (--model). Uses, in order of preference:
         4 keypoints (pose model, corners in model keypoint order, re-ordered
         by gate_pose.order_corners), else a segmentation mask (largest
         contour -> 4-gon), else the bounding box (axis-aligned: gives range
         and bearing but NO gate orientation; flagged "bbox" in the CSV).
  color  vision/color_gate_detector.py with --hsv-lo/--hsv-hi (default magenta).

Intrinsics: --calib takes an OpenCV FileStorage yaml with M (or K /
camera_matrix) and D (or distortion_coefficients), optional "distortion_model:
fisheye" (VOXL's voxl-calibrate-camera writes this format). Fisheye corners
are undistorted to the pinhole K before PnP. Without --calib, --hfov-deg
builds an ideal pinhole: on this lens (heavy barrel distortion) that is only a
plumbing check, and every overlay frame says UNCALIBRATED.

Pose is in the OpenCV camera frame (+X right, +Y down, +Z forward). No world
frame: this test has no ground truth, so compare range against a tape measure.
"""

import argparse
import csv
import os
import sys
import threading
import time
import urllib.request

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from camera import CameraIntrinsics  # noqa: E402
from gate_pose import estimate_gate_pose, project_gate, order_corners  # noqa: E402


# --------------------------------------------------------------------------- feed

class MjpegReader:
    """Background thread keeping only the newest decoded BGR frame."""

    def __init__(self, url: str, timeout: float = 5.0):
        self.url, self.timeout = url, timeout
        self._lock = threading.Lock()
        self._frame, self._seq, self._t = None, 0, 0.0
        self._stop = threading.Event()
        self.error = None
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _run(self):
        try:
            resp = urllib.request.urlopen(self.url, timeout=self.timeout)
            buf = b""
            while not self._stop.is_set():
                chunk = resp.read(65536)
                if not chunk:
                    raise IOError("feed closed")
                buf += chunk
                # newest complete JPEG in the buffer
                end = buf.rfind(b"\xff\xd9")
                if end < 0:
                    continue
                start = buf.rfind(b"\xff\xd8", 0, end)
                if start < 0:
                    buf = buf[end + 2:]
                    continue
                jpg, buf = buf[start:end + 2], buf[end + 2:]
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                with self._lock:
                    self._frame, self._t = img, time.time()
                    self._seq += 1
        except Exception as e:  # surfaced by the main loop
            self.error = e

    def latest(self, after_seq: int):
        with self._lock:
            if self._seq > after_seq:
                return self._seq, self._t, self._frame
        return after_seq, None, None

    def stop(self):
        self._stop.set()


# -------------------------------------------------------------------- intrinsics

def load_calib(path: str):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"cannot open {path}")

    def mat(*names):
        for n in names:
            node = fs.getNode(n)
            if not node.empty():
                return node.mat()
        return None

    K = mat("M", "K", "camera_matrix")
    D = mat("D", "distortion_coefficients", "dist_coeffs")
    model_node = fs.getNode("distortion_model")
    model = model_node.string() if not model_node.empty() else ""
    w = fs.getNode("width").real() or fs.getNode("image_width").real()
    h = fs.getNode("height").real() or fs.getNode("image_height").real()
    fs.release()
    if K is None:
        raise ValueError(f"{path}: no camera matrix (M/K/camera_matrix)")
    D = np.zeros(5) if D is None else D.ravel()
    fisheye = model.lower().startswith(("fisheye", "equidistant")) or D.size == 4
    return K.astype(float), D.astype(float), fisheye, (int(w), int(h))


class Lens:
    """Maps detected (distorted) pixel corners to the pinhole intrinsics PnP uses."""

    def __init__(self, K, D, fisheye, calib_size, frame_size, calibrated: bool):
        fw, fh = frame_size
        K = K.copy()
        if calib_size[0] and calib_size[0] != fw:   # calibrated at another res
            s = fw / calib_size[0]
            K[:2] *= s
        self.K, self.D, self.fisheye, self.calibrated = K, D, fisheye, calibrated
        self.intr = CameraIntrinsics(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
                                     width=fw, height=fh,
                                     distortion=np.zeros(5) if fisheye else D)

    def to_pinhole(self, px: np.ndarray) -> np.ndarray:
        if not self.fisheye:
            return px                       # solvePnP handles k1..k3 itself
        pts = px.reshape(-1, 1, 2).astype(np.float64)
        und = cv2.fisheye.undistortPoints(pts, self.K, self.D.reshape(4, 1), P=self.K)
        return und.reshape(-1, 2)

    def to_image(self, px_pinhole: np.ndarray) -> np.ndarray:
        """Pinhole pixels -> distorted image pixels (for drawing reprojections)."""
        if not self.fisheye:
            return px_pinhole
        Kinv = np.linalg.inv(self.K)
        h = np.c_[px_pinhole, np.ones(len(px_pinhole))] @ Kinv.T
        d = cv2.fisheye.distortPoints(h[:, None, :2].astype(np.float64),
                                      self.K, self.D.reshape(4, 1))
        return d.reshape(-1, 2)


# ---------------------------------------------------------------------- detectors

def quad_from_contour(cnt):
    peri = cv2.arcLength(cnt, True)
    for eps in (0.02, 0.03, 0.05, 0.08):
        ap = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(ap) == 4:
            return ap.reshape(4, 2).astype(np.float64)
    box = cv2.boxPoints(cv2.minAreaRect(cnt))
    return box.astype(np.float64)


class CnnDetector:
    def __init__(self, weights: str, conf: float, imgsz: int):
        try:
            from ultralytics import YOLO
        except ImportError:
            sys.exit("ultralytics not installed: pip install ultralytics "
                     "(or export the model and add a loader here)")
        self.model = YOLO(weights)
        self.conf, self.imgsz = conf, imgsz
        print(f"[cnn] {weights} task={self.model.task} classes={self.model.names}")

    def __call__(self, bgr):
        r = self.model.predict(bgr, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        out = []
        n = 0 if r.boxes is None else len(r.boxes)
        for i in range(n):
            score = float(r.boxes.conf[i])
            if r.keypoints is not None and r.keypoints.xy.shape[1] >= 4:
                kp = r.keypoints.xy[i].cpu().numpy()[:4].astype(np.float64)
                out.append((kp, score, "kpts"))
            elif r.masks is not None:
                m = (r.masks.data[i].cpu().numpy() > 0.5).astype(np.uint8)
                m = cv2.resize(m, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    out.append((quad_from_contour(max(cnts, key=cv2.contourArea)), score, "mask"))
            else:
                x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy().astype(np.float64)
                out.append((np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]), score, "bbox"))
        return out


class TyeGateCNN:
    """Tye's NumPy gate CNN (EGN1/gate_cnn). Nearest gate only, OUTER corners
    (1.9 m on the course gates -> pass --gate-side 1.9).

    The net expects a ~5:3 frame; the 4:3 VOXL frame is centre-cropped to 5:3
    and corners shifted back. Every raw detection is returned; `src` says
    whether it passed Tye's good_detection() ("cnn") or not ("cnn-rej"), so
    the overlay shows rejects in red and only accepted ones get PnP.
    good_detection's colour checks assume the purple gate colour."""

    def __init__(self, cnn_dir, conf_min, refine, color_checks):
        sys.path.insert(0, cnn_dir)
        import detector as D
        self.D = D
        self.cnn = D.GateCNN(os.path.join(cnn_dir, "gate_cnn.npz"))
        self.conf_min, self.refine, self.color_checks = conf_min, refine, color_checks
        print(f"[cnn] Tye gate CNN from {cnn_dir} (conf_min {conf_min}, refine {refine}, "
              f"colour checks {color_checks})")

    def __call__(self, bgr):
        H, W = bgr.shape[:2]
        ch = min(H, int(round(W * 3 / 5)))
        y0 = (H - ch) // 2
        crop = bgr[y0:y0 + ch]
        c, p = self.cnn.detect(crop)
        D = self.D
        if self.color_checks:
            ok = D.good_detection(crop, c, p, conf_min=self.conf_min)
        else:
            sides = np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1)
            ok = bool(p.min() > self.conf_min and sides.min() > 0.04 * W
                      and sides.min() / sides.max() > 0.3 and D.fully_in_view(c, crop.shape))
        if ok and self.refine:
            c = D.refine_with_colour(crop, c)
        if not ok and p.min() < 0.5 * self.conf_min:
            return []                              # nothing there: don't clutter the overlay
        c = c + [0, y0]
        return [(c, float(p.min()), "cnn" if ok else "cnn-rej")]


class ColorDetector:
    def __init__(self, lo, hi, min_area):
        from color_gate_detector import detect_gates_color
        self.fn, self.lo, self.hi, self.min_area = detect_gates_color, lo, hi, min_area

    def __call__(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return [(c, 1.0, "color") for c in self.fn(rgb, self.lo, self.hi, self.min_area)]


# ------------------------------------------------------------------------ overlay

def draw(img, dets, lens, gate_side, fps, uncalibrated, label):
    for d in dets:
        c = d["corners_img"].astype(int)
        col = (0, 0, 255) if d["src"].endswith("-rej") else (0, 255, 255)
        cv2.polylines(img, [c.reshape(-1, 1, 2)], True, col, 2)
        if d["det"] is None:
            cv2.putText(img, f"{d['src']} conf {d['score']:.2f}", tuple(c.min(axis=0) + [0, -8]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        for k, p in enumerate(c):
            cv2.circle(img, tuple(p), 4, (0, 255, 255), -1)
            cv2.putText(img, str(k), tuple(p + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        g = d["det"]
        if g is None:
            continue
        rep = lens.to_image(project_gate(g.T_cam_gate, lens.intr, gate_side)).astype(int)
        cv2.polylines(img, [rep.reshape(-1, 1, 2)], True, (0, 255, 0), 1)
        # gate normal from centre, 0.5 m long
        T = g.T_cam_gate
        ctr = T[:3, 3]
        pts3 = np.array([ctr, ctr + 0.5 * T[:3, 2]])
        pp, _ = cv2.projectPoints(pts3, np.zeros(3), np.zeros(3), lens.intr.K, np.zeros(5))
        pp = lens.to_image(pp.reshape(-1, 2)).astype(int)
        cv2.arrowedLine(img, tuple(pp[0]), tuple(pp[1]), (255, 0, 255), 2)
        x, y, z = ctr
        txt = [f"{d['src']} {d['score']:.2f}  range {g.range_m:.2f} m",
               f"xyz {x:+.2f} {y:+.2f} {z:+.2f}  reproj {g.reprojection_error_px:.1f}px"]
        org = c.min(axis=0)
        for k, t in enumerate(txt):
            yy = max(20, org[1] - 28 + 18 * k)
            cv2.putText(img, t, (org[0], yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(img, t, (org[0], yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    hud = f"{label}  {fps:4.1f} fps  gates {sum(d['det'] is not None for d in dets)}"
    cv2.putText(img, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(img, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
    if uncalibrated:
        cv2.putText(img, "UNCALIBRATED - range not trustworthy", (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return img


# --------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8080/video_raw/hires_front_small_color")
    ap.add_argument("--detector", choices=["tye", "cnn", "color"], default="tye")
    ap.add_argument("--model", help="Ultralytics weights (.pt) for --detector cnn")
    ap.add_argument("--cnn-dir", default=os.path.join(os.path.dirname(os.path.dirname(_HERE)), "gate_cnn"),
                    help="Tye's gate_cnn folder for --detector tye")
    ap.add_argument("--cnn-conf", type=float, default=0.45)
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--no-color-checks", action="store_true",
                    help="skip good_detection's purple-colour checks (non-purple real gate)")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--hsv-lo", type=int, nargs=3, default=None)
    ap.add_argument("--hsv-hi", type=int, nargs=3, default=None)
    ap.add_argument("--min-area", type=float, default=400)
    ap.add_argument("--gate-side", type=float, required=True,
                    help="inner opening side length in metres (the 4 corners PnP sees)")
    ap.add_argument("--calib", help="OpenCV yaml intrinsics for this camera")
    ap.add_argument("--hfov-deg", type=float, default=100.0,
                    help="pinhole fallback when --calib is absent (plumbing only)")
    ap.add_argument("--duration", type=float, default=60.0, help="seconds; 0 = until Ctrl-C")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--show", action="store_true", help="also open a live preview window")
    ap.add_argument("--dry-run", action="store_true", help="no detector: test feed + video only")
    a = ap.parse_args(argv)

    os.makedirs(a.out_dir, exist_ok=True)
    reader = MjpegReader(a.url)

    seq, frame = 0, None
    t_wait = time.time()
    while frame is None:
        if reader.error:
            sys.exit(f"feed error: {reader.error}  (is `adb forward tcp:8080 tcp:80` up?)")
        if time.time() - t_wait > 10:
            sys.exit("no frame in 10 s")
        seq, _, frame = reader.latest(seq)
        time.sleep(0.01)
    H, W = frame.shape[:2]
    print(f"[feed] {a.url}  {W}x{H}")

    if a.calib:
        K, D, fe, csize = load_calib(a.calib)
        lens = Lens(K, D, fe, csize, (W, H), calibrated=True)
        print(f"[calib] {a.calib} fisheye={fe} fx={lens.K[0,0]:.1f} D={D.round(4).tolist()}")
    else:
        i = CameraIntrinsics.from_horizontal_fov(W, H, a.hfov_deg)
        lens = Lens(i.K, np.zeros(5), False, (0, 0), (W, H), calibrated=False)
        print(f"[calib] NONE: ideal pinhole at hFOV {a.hfov_deg} deg -- ranges are NOT trustworthy")

    if a.dry_run:
        detector, label = (lambda bgr: []), "dry-run"
    elif a.detector == "tye":
        detector = TyeGateCNN(a.cnn_dir, a.cnn_conf, not a.no_refine, not a.no_color_checks)
        label = "Tye CNN"
    elif a.detector == "cnn":
        if not a.model:
            sys.exit("--detector cnn needs --model <weights>")
        detector, label = CnnDetector(a.model, a.conf, a.imgsz), f"CNN {os.path.basename(a.model)}"
    else:
        from color_gate_detector import MAGENTA_HSV_LO, MAGENTA_HSV_HI
        detector = ColorDetector(tuple(a.hsv_lo or MAGENTA_HSV_LO),
                                 tuple(a.hsv_hi or MAGENTA_HSV_HI), a.min_area)
        label = "color"

    vid_path = os.path.join(a.out_dir, "overlay.mp4")
    raw_dir = os.path.join(a.out_dir, "raw_jpg")
    os.makedirs(raw_dir, exist_ok=True)
    # (t, JPEG bytes of the annotated frame) -- written at the measured rate at the end.
    # Kept JPEG-encoded: raw BGR at 1024x768 is ~2.4 MB/frame (~6 GB for 2400 frames).
    frames = []
    csv_f = open(os.path.join(a.out_dir, "detections.csv"), "w", newline="")
    wr = csv.writer(csv_f)
    wr.writerow(["frame", "t_wall", "det_idx", "src", "score", "range_m", "x_m", "y_m", "z_m",
                 "nx", "ny", "nz", "reproj_px", "c0x", "c0y", "c1x", "c1y", "c2x", "c2y", "c3x", "c3y"])

    t0, n, last_t, fps, n_det = time.time(), 0, None, 0.0, 0
    try:
        while a.duration <= 0 or time.time() - t0 < a.duration:
            if reader.error:
                print(f"[feed] error: {reader.error}")
                break
            seq, tf, frame = reader.latest(seq)
            if frame is None:
                time.sleep(0.003)
                continue
            raw = frame.copy()
            dets = []
            for corners, score, src in detector(frame):
                corners_img = order_corners(np.asarray(corners, float).reshape(4, 2))
                g = None if src.endswith("-rej") else estimate_gate_pose(
                    lens.to_pinhole(corners_img), lens.intr, gate_side=a.gate_side, assume_ordered=True)
                dets.append({"corners_img": corners_img, "det": g, "score": score, "src": src})
                row = [n, f"{tf:.3f}", len(dets) - 1, src, f"{score:.3f}"]
                if g is not None:
                    T = g.T_cam_gate
                    row += [f"{g.range_m:.3f}", *[f"{v:.3f}" for v in T[:3, 3]],
                            *[f"{v:.3f}" for v in T[:3, 2]], f"{g.reprojection_error_px:.2f}"]
                else:
                    row += [""] * 8
                row += [f"{v:.1f}" for v in corners_img.ravel()]
                wr.writerow(row)
            n_det += any(d["det"] is not None for d in dets)
            if last_t is not None:
                fps = 0.9 * fps + 0.1 / max(tf - last_t, 1e-3) if fps else 1.0 / max(tf - last_t, 1e-3)
            last_t = tf
            out = draw(frame, dets, lens, a.gate_side, fps, not lens.calibrated, label)
            frames.append((tf, cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]))
            cv2.imwrite(os.path.join(raw_dir, f"{n:05d}.jpg"), raw)   # every frame: re-run offline
            if a.show:
                cv2.imshow("gate test", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            n += 1
            if n % 30 == 0:
                best = min((d["det"].range_m for d in dets if d["det"]), default=None)
                print(f"[run] {n} frames  {fps:.1f} fps  with-gate {n_det}"
                      + (f"  nearest {best:.2f} m" if best else ""), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        csv_f.close()
        if frames:
            span = frames[-1][0] - frames[0][0]
            rate = (len(frames) - 1) / span if span > 0 else 15.0
            vw = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"), rate, (W, H))
            for _, jpg in frames:
                vw.write(cv2.imdecode(jpg, cv2.IMREAD_COLOR))
            vw.release()
            print(f"[done] {len(frames)} frames @ {rate:.1f} fps, {n_det} with a gate -> {vid_path}")
        if a.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
