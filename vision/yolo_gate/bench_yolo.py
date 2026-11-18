#!/usr/bin/env python3
"""
Time the gate YOLO model on the device you run this on, without Ultralytics / PyTorch:
an .onnx through onnxruntime (CPU), or a .tflite through libgate_tflite (CPU or the VOXL 2's GPU,
docker/voxl2-tflite-gpu/). Only numpy + OpenCV + that runtime are needed, so it runs in the
slim containers on the drone.

Export (anywhere Ultralytics is installed; the .onnx lands next to the .pt, same name, so
rename it before exporting another size):
    yolo export model=vision/yolo_gate/weights/pink_det_v1_yolo11n.pt format=onnx imgsz=480,640
480x640 (H, W) fits the 1024x768 hires frame with no padding, the same input live_gate_test.py's
.pt gets, and is ~25 % less work than 640x640. imgsz=240,320 is ~4x less again.
.tflite: export_tflite.py (ONNX -> TFLite float32 + float16).

    python3 bench_yolo.py --model pink_det_v1_480x640.onnx --images ../live_runs/hybrid_live_01/raw_jpg --n 200
    python3 bench_yolo.py --model pink_det_v1_480x640_float16.tflite --delegate gpu --n 200
    python3 bench_yolo.py --model pink_det_v1_480x640.onnx --threads 3          # synthetic frames

--images takes a folder of JPEGs (or one file); frames are used in order, cycling. Without it,
random 1024x768 frames are used: fine for timing, but no detections.

Prints per-frame timing (preprocess, inference, decode) and, with real images, the share of
frames with a box and the median score. --compare-pt (needs Ultralytics) checks the boxes against
the .pt model on the same frames; --compare-onnx against an .onnx (e.g. a .tflite against the
.onnx it was converted from, on the drone, where Ultralytics isn't installed).
"""

import argparse
import glob
import os
import statistics
import time

import cv2
import numpy as np


def letterbox(bgr, size):
    """Resize keeping aspect ratio, pad to the model input (H, W) with grey (Ultralytics' default)."""
    H, W = size
    h, w = bgr.shape[:2]
    r = min(H / h, W / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (H - nh) // 2
    left = (W - nw) // 2
    out = np.full((H, W, 3), 114, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = img
    blob = out[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0  # BGR->RGB, NCHW
    return np.ascontiguousarray(blob), r, left, top


def decode(out, r, left, top, conf, iou=0.5):
    """YOLOv8/11 detect head, output (1, 4 + n_classes, N): cx, cy, w, h in input pixels.
    Returns [(x1, y1, x2, y2, score, cls)] in original-image pixels after NMS."""
    p = out[0].T                        # (N, 4 + nc)
    scores_all = p[:, 4:]
    cls = scores_all.argmax(1)
    scores = scores_all[np.arange(len(p)), cls]
    keep = scores >= conf
    if not keep.any():
        return []
    p, scores, cls = p[keep], scores[keep], cls[keep]
    cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    x1 = (cx - w / 2 - left) / r
    y1 = (cy - h / 2 - top) / r
    boxes_xywh = np.stack([x1, y1, w / r, h / r], 1)
    idx = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), conf, iou)
    idx = np.array(idx).reshape(-1)
    return [(float(x1[i]), float(y1[i]), float(x1[i] + w[i] / r), float(y1[i] + h[i] / r),
             float(scores[i]), int(cls[i])) for i in idx]


def load_frames(path, limit):
    if path is None:
        rng = np.random.default_rng(0)
        return [rng.integers(0, 255, (768, 1024, 3), dtype=np.uint8) for _ in range(4)], False
    files = [path] if os.path.isfile(path) else sorted(glob.glob(os.path.join(path, "*.jpg")))
    if not files:
        raise SystemExit(f"no .jpg found at {path}")
    step = max(1, len(files) // limit) if len(files) > limit else 1
    files = files[::step][:limit]
    return [cv2.imread(f) for f in files], True


class OnnxBackend:
    def __init__(self, path, threads):
        import onnxruntime as ort
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0]
        self.size = (int(self.inp.shape[2]), int(self.inp.shape[3]))   # NCHW
        self.desc = f"onnxruntime CPU, input {self.inp.shape}, output {self.sess.get_outputs()[0].shape}"

    def __call__(self, blob):                       # blob: NCHW float32
        return self.sess.run(None, {self.inp.name: blob})[0]


class TFLiteBackend:
    def __init__(self, path, threads, delegate, fp16):
        from tflite_gpu import TFLiteModel
        self.m = TFLiteModel(path, delegate=delegate, threads=threads, fp16=fp16)
        self.size = (self.m.in_shape[1], self.m.in_shape[2])             # NHWC
        where = "GPU (OpenCL" + (", fp16)" if fp16 else ", fp32)") if self.m.gpu else "CPU (XNNPACK)"
        part = "" if not self.m.gpu else (", whole graph on the GPU" if self.m.plan_nodes == 1 else
                                          f", GPU took only part of the graph: {self.m.plan_nodes} plan nodes")
        self.desc = f"TFLite {where}{part}, input {self.m.in_shape}, output {self.m.out_shape}"

    def __call__(self, blob):
        out = self.m.run(blob.transpose(0, 2, 3, 1))
        return out if out.shape[1] < out.shape[2] else out.transpose(0, 2, 1)   # -> (1, 4+nc, N)


def make_backend(path, threads, delegate="cpu", fp16=True):
    if path.endswith(".tflite"):
        return TFLiteBackend(path, threads, delegate, fp16)
    return OnnxBackend(path, threads)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help=".onnx file")
    ap.add_argument("--images", help="folder of .jpg (or one file); default: synthetic frames")
    ap.add_argument("--n", type=int, default=100, help="frames to time (after warm-up)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads; 0 = the runtime's default")
    ap.add_argument("--delegate", choices=["cpu", "gpu"], default="cpu", help=".tflite only")
    ap.add_argument("--fp32-gpu", action="store_true", help=".tflite on the GPU: compute in fp32, not fp16")
    ap.add_argument("--compare-pt", help="also run this .pt with Ultralytics on the same frames and compare boxes")
    ap.add_argument("--compare-onnx", help="also run this .onnx on the same frames and compare boxes")
    a = ap.parse_args(argv)

    run = make_backend(a.model, a.threads, a.delegate, not a.fp32_gpu)
    size = run.size
    print(f"model {os.path.basename(a.model)}: {run.desc}, threads {a.threads or 'default'}, "
          f"cpu cores {os.cpu_count()}")

    frames, real = load_frames(a.images, a.n)
    for i in range(a.warmup):
        blob, *_ = letterbox(frames[i % len(frames)], size)
        run(blob)

    t_pre, t_inf, t_dec, results = [], [], [], []
    for i in range(a.n):
        bgr = frames[i % len(frames)]
        t0 = time.perf_counter()
        blob, r, left, top = letterbox(bgr, size)
        t1 = time.perf_counter()
        out = run(blob)
        t2 = time.perf_counter()
        dets = decode(out, r, left, top, a.conf)
        t3 = time.perf_counter()
        t_pre.append((t1 - t0) * 1e3); t_inf.append((t2 - t1) * 1e3); t_dec.append((t3 - t2) * 1e3)
        results.append(dets)

    tot = [p + q + s for p, q, s in zip(t_pre, t_inf, t_dec)]
    print(f"{a.n} frames  ms median / p90:  preprocess {statistics.median(t_pre):.1f} / {pct(t_pre, .9):.1f}   "
          f"inference {statistics.median(t_inf):.1f} / {pct(t_inf, .9):.1f}   "
          f"decode {statistics.median(t_dec):.1f} / {pct(t_dec, .9):.1f}")
    print(f"total {statistics.median(tot):.1f} ms median -> {1000 / statistics.median(tot):.1f} fps "
          f"(YOLO only; the color corners + PnP come on top)")

    # The same input must give the same output. On a Mac the TFLite GPU delegate (Apple's
    # deprecated OpenCL) gave outputs up to ~2000 px apart run to run (2026-11-18), while the CPU
    # was exact: a GPU result only counts once this and --compare-onnx pass on the device itself.
    blob, *_ = letterbox(frames[0], size)
    outs = [run(blob) for _ in range(5)]
    spread = max(float(np.abs(outs[0] - o).max()) for o in outs[1:])
    print(f"repeatability (same frame 5x): max output difference {spread:.4g}"
          + ("  ** NOT REPEATABLE: do not trust this backend's results **" if spread > 1e-2 else "  OK"))

    if real:
        n_hit = sum(1 for d in results if d)
        best = [max(x[4] for x in d) for d in results if d]
        print(f"frames with a box: {n_hit}/{len(results)} ({100 * n_hit / len(results):.0f} %)"
              + (f", median best score {statistics.median(best):.3f}" if best else ""))

    if a.compare_pt and real:
        from ultralytics import YOLO
        pt = YOLO(a.compare_pt)
        diffs, agree = [], 0
        for bgr, d in zip(frames, results):
            # Feed Ultralytics the same padded square the ONNX model saw: on a raw 1024x768 frame it
            # pads only to 640x480 for a .pt, a different input, and the boxes differ by several px.
            blob, r, left, top = letterbox(bgr, size)
            sq = np.ascontiguousarray((blob[0].transpose(1, 2, 0)[:, :, ::-1] * 255).round().astype(np.uint8))
            rb = pt.predict(sq, conf=a.conf, imgsz=list(size), device="cpu", verbose=False)[0].boxes
            has_pt = len(rb) > 0
            if has_pt == bool(d):
                agree += 1
            if has_pt and d:
                bp = rb.xyxy[rb.conf.argmax()].cpu().numpy()
                bp = (bp - np.array([left, top, left, top])) / r
                bo = np.array(max(d, key=lambda x: x[4])[:4])
                diffs.append(float(np.abs(bp - bo).max()))
        print(f".pt vs .onnx: box / no-box agree on {agree}/{len(results)} frames"
              + (f"; best-box corner difference median {statistics.median(diffs):.2f} px, "
                 f"max {max(diffs):.2f} px" if diffs else ""))


    if a.compare_onnx and real:
        ref = OnnxBackend(a.compare_onnx, a.threads)
        if ref.size != size:
            raise SystemExit(f"--compare-onnx input {ref.size} differs from the model's {size}")
        diffs, agree = [], 0
        for bgr, d in zip(frames, results):
            blob, r, left, top = letterbox(bgr, size)
            dr = decode(ref(blob), r, left, top, a.conf)
            agree += bool(dr) == bool(d)
            if dr and d:
                bo = np.array(max(dr, key=lambda x: x[4])[:4])
                bt = np.array(max(d, key=lambda x: x[4])[:4])
                diffs.append(float(np.abs(bo - bt).max()))
        print(f"{os.path.basename(a.compare_onnx)} vs this model: box / no-box agree on {agree}/{len(results)} frames"
              + (f"; best-box corner difference median {statistics.median(diffs):.2f} px, "
                 f"max {max(diffs):.2f} px" if diffs else ""))


if __name__ == "__main__":
    main()
