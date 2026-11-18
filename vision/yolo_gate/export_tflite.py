#!/usr/bin/env python3
"""
Convert the gate YOLO .onnx to TFLite (float32 + float16 weights) for the VOXL 2's GPU delegate,
with onnx2tf. Ultralytics' own `format=tflite` export pulled in packages that broke its venv
(2026-11-18) and no longer writes float16, so it goes through the .onnx instead.

    # a separate venv (TensorFlow is large; keep it out of the YOLO venv):
    python -m venv tfvenv
    tfvenv/bin/pip install "tensorflow==2.19.*" tf_keras onnx2tf onnx onnx_graphsurgeon sng4onnx psutil opencv-python-headless
    tfvenv/bin/python export_tflite.py --onnx weights/pink_det_v1_480x640.onnx --images ../live_runs/hybrid_live_01/raw_jpg

Writes <out>/<name>_float32.tflite and <name>_float16.tflite (float16 weights, float32 input and
output: what ModalAI recommends for the GPU delegate). Input NHWC float32 RGB in [0, 1]; output
(1, 4 + classes, N), boxes in input pixels, the same as the .onnx: bench_yolo.py decodes both.

onnx2tf checks the conversion on sample images, which it downloads from GitHub as a pickled .npy
that newer numpy refuses to load. This script writes that file itself, from --images (real gate
frames) or random data, so no download happens.
"""

import argparse
import glob
import os
import shutil
import sys

import numpy as np

CALIB_NAME = "calibration_image_sample_data_20x128x128x3_float32.npy"   # the name onnx2tf looks for


def calib_data(images):
    if images:
        import cv2
        files = sorted(glob.glob(os.path.join(images, "*.jpg")))
        if not files:
            sys.exit(f"no .jpg in {images}")
        files = files[:: max(1, len(files) // 20)][:20]
        a = np.stack([cv2.resize(cv2.imread(f), (128, 128))[:, :, ::-1] for f in files])
        return a.astype(np.float32) / 255.0
    return np.random.default_rng(0).random((20, 128, 128, 3), dtype=np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", help="output folder (default: next to the .onnx, <name>_tflite/)")
    ap.add_argument("--images", help="folder of real frames for onnx2tf's check (default: random)")
    a = ap.parse_args(argv)

    import onnx2tf

    onnx_path = os.path.abspath(a.onnx)
    name = os.path.splitext(os.path.basename(onnx_path))[0]
    out = os.path.abspath(a.out or os.path.join(os.path.dirname(onnx_path), name + "_tflite"))
    os.makedirs(out, exist_ok=True)

    cwd = os.getcwd()
    os.chdir(out)                       # onnx2tf reads the sample file from the current folder
    try:
        np.save(CALIB_NAME, calib_data(os.path.join(cwd, a.images) if a.images else None))
        onnx2tf.convert(input_onnx_file_path=onnx_path, output_folder_path=out, non_verbose=True,
                        not_use_onnxsim=True, output_signaturedefs=True)  # else op names break the saved_model step
    finally:
        if os.path.exists(CALIB_NAME):
            os.remove(CALIB_NAME)
        os.chdir(cwd)

    for prec in ("float32", "float16"):
        src = os.path.join(out, f"{name}_{prec}.tflite")
        if not os.path.isfile(src):
            sys.exit(f"onnx2tf did not write {src}")
        print(f"{prec}: {src} ({os.path.getsize(src) / 1e6:.1f} MB)")
    shutil.rmtree(os.path.join(out, "saved_model"), ignore_errors=True)


if __name__ == "__main__":
    main()
