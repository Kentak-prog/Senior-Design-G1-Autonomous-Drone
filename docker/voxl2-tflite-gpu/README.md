# voxl2-tflite-gpu: OpenCL + TensorFlow Lite on the VOXL 2's GPU, in Docker

A container for the Starling 2 Max's VOXL 2 that can run our gate YOLO on the **Adreno 650 GPU**:
OpenCL from the drone's own GPU driver (ModalAI's
[OpenCL in Docker](https://docs.modalai.com/voxl-2-opencl-in-docker/) recipe), and TensorFlow Lite
2.17.1 built from source with its OpenCL GPU delegate, callable from Python.

**Status (2026-11-18):** nothing here has run on the drone yet; everything marked **Check** below is
still to do there. Tested on a Mac (M-series, TFLite 2.17.1 built with this `CMakeLists.txt`):

| What | Result on the Mac |
|---|---|
| `.onnx` → `.tflite` export (`export_tflite.py`) | float32 and float16 written |
| `libgate_tflite` + `tflite_gpu.py`, **CPU** | boxes identical to the `.onnx` on 100/100 real frames (float16 within 0.12 px), ~29 fps, repeatable |
| `libgate_tflite` + `tflite_gpu.py`, **GPU** | the GPU delegate loads OpenCL and takes **the whole graph** (`BATCH_MATMUL` included), ~54 fps, **but its outputs are not repeatable**: the same frame run 5 times differs by thousands of px, in fp32 too |
| `opencl_check.c` | compiles against the Khronos OpenCL headers (not run: the Mac's OpenCL is 1.2) |

The Mac's GPU result is most likely Apple's deprecated OpenCL driver (the CPU path, same library and
model, is exact). It proves the plumbing; it says nothing about accuracy on the Adreno 650. Step 7's
repeatability and `--compare-onnx` checks decide that on the drone.

Mac-only build issues, not expected on the drone's Ubuntu/GCC: Apple clang rejects `std::abs<float>`
in TFLite 2.17's `kernels/elementwise.cc` (patched locally for the test), and `benchmark_model` wants
Apple's CoreML delegate at link time.

## What's in the image, and what isn't

| In the image | Not in the image |
|---|---|
| Ubuntu 22.04 (arm64), Python 3, numpy, OpenCV (headless), onnxruntime | **no repo files**: the build only sees this folder |
| The drone's Adreno GPU driver (`libOpenCL.so` + the Qualcomm libraries it needs), repacked by `repack_deps.sh` | no models, no calibration |
| `libgate_tflite.so`: TFLite 2.17.1 + GPU delegate + a small C API (`gate_tflite.cpp`) | no TensorFlow, no PyTorch, no Ultralytics |
| `benchmark_model` (TFLite's benchmark tool, with `--use_gpu`), `opencl_check` | |

`run.sh` mounts the repo (`/data/repo`) at `/repo`, so code and models change without a rebuild.
For flight, a later image will `COPY` only an explicit list (the vision modules, the calibration, the
`.tflite` model, the ROS nodes).

```
drone (VOXL 2)                                    container
  /dev/kgsl-3d0, /dev/ion  (GPU device nodes) ──►  passed in by run.sh (not --privileged)
  qti-adreno + 4 libs  ──repack_deps.sh──► deps/*.deb ──► installed in the image
                                                   libgate_tflite.so ──► vision/yolo_gate/tflite_gpu.py (ctypes)
  /data/repo ────────────────────────────────────►  /repo
```

Why build TFLite instead of using the drone's: the drone's `qrb5165-tflite` is 2.8.0 on SDK 1.5.1
(ModalAI's current builds use 2.17.1), C++ only, and our YOLO11n uses ops (`BATCH_MATMUL` in its
attention block) that are newer than 2.8's GPU delegate. TFLite's Python packages don't include the
GPU delegate, hence the small C library.

## Files

| File | What it does |
|---|---|
| `Dockerfile` | two stages: build TFLite + `libgate_tflite.so`, then a slim runtime image |
| `CMakeLists.txt`, `gate_tflite.cpp` | the C API: load a `.tflite`, run it on the GPU (OpenCL only) or CPU |
| `repack_deps.sh` | run on the drone: the GPU driver packages → `deps/*.deb` (git-ignored: Qualcomm binaries) |
| `run.sh` | start the container with the GPU, host network and shared memory, the repo at `/repo` |
| `opencl_check.c` | lists the OpenCL devices and runs a test kernel on the GPU |
| `../../vision/yolo_gate/tflite_gpu.py` | Python side: `TFLiteModel(path, delegate="gpu").run(x)` |
| `../../vision/yolo_gate/bench_yolo.py` | times the model (`.onnx` or `.tflite`, CPU or GPU) |
| `../../vision/yolo_gate/export_tflite.py` | `.onnx` → `.tflite` (float32 + float16) |

## Steps

Before: the drone on Wi‑Fi in **station mode** (it needs internet to build), SSH working, the repo at
`/data/repo` (`px4/VOXL2_BRINGUP_WINDOWS.md`, Steps 1–6). Commands are on the drone unless noted.

### 1. Docker on the drone can do multi-stage builds
```bash
docker --version
docker info | grep -i "root dir"      # on /data: the build needs several GB
```
**Check:** Docker 17.05 or newer, root dir on `/data`. If it's older, ModalAI's page installs a
current Docker with the official `get-docker.sh` script.

### 2. Repack the GPU driver
```bash
cd /data/repo/docker/voxl2-tflite-gpu
./repack_deps.sh
```
**Check:** `deps/` has five `.deb` files (`qti-adreno`, `libcutils0`, `libsync`, `qti-libion`,
`liblog0`) and the script lists `/dev/kgsl-3d0` and `/dev/ion`. Re-run this after every VOXL SDK
upgrade: the driver in the image has to match the drone's kernel.

### 3. Build the image
```bash
docker build -t voxl2-tflite-gpu .                       # BUILD_JOBS=6 by default
docker build --build-arg BUILD_JOBS=4 -t voxl2-tflite-gpu .   # if the build runs out of memory
```
The first build clones TensorFlow (~460 MB) and compiles TFLite: expect 30–90 minutes. Later builds
reuse Docker's cache unless `gate_tflite.cpp` or `CMakeLists.txt` change.

**Check:** ends with `Successfully tagged voxl2-tflite-gpu:latest`. `docker images` shows it.

### 4. The GPU works inside the container
```bash
./run.sh opencl_check
```
**Check:** one platform, one device `QUALCOMM Adreno(TM)`, `OpenCL 2.0 Adreno(TM) 650`, and the last
line `OK: the GPU ran a ... kernel correctly`. `FAIL: no OpenCL platform` = the device nodes or the
driver are missing (see Troubleshooting).

### 5. The model, as TFLite (laptop)
`vision/yolo_gate/export_tflite.py` converts the `.onnx` (see `bench_yolo.py`'s docstring for the
ONNX export) to `<name>_float32.tflite` and `<name>_float16.tflite`. Use a separate venv: TensorFlow
is large, and Ultralytics' own TFLite export broke the YOLO venv's packages when tried (2026-11-18).
```bash
python -m venv tfvenv
tfvenv/bin/pip install "tensorflow==2.19.*" tf_keras onnx2tf onnx onnx_graphsurgeon sng4onnx psutil opencv-python-headless
cd vision/yolo_gate
../../tfvenv/bin/python export_tflite.py --onnx weights/pink_det_v1_480x640.onnx --images ../live_runs/hybrid_live_01/raw_jpg
```
(On Windows: `tfvenv\Scripts\pip` and `..\..\tfvenv\Scripts\python`.) Copy both `.tflite` files and
the `.onnx` to `/data/repo/vision/yolo_gate/weights/` on the drone.

**Check (Mac, 2026-11-18):** 100 real frames, against the `.onnx`: float32 boxes identical (largest
raw-output difference 0.004 px); float16 within 0.9 px median, 5.5 px worst, same box / no-box decision
on every frame. `tf.lite.experimental.Analyzer` reports the model GPU-compatible.

### 6. TFLite's own benchmark, GPU vs CPU
```bash
./run.sh benchmark_model --graph=vision/yolo_gate/weights/pink_det_v1_480x640_float16.tflite \
    --use_gpu=true --gpu_precision_loss_allowed=true --num_runs=100
./run.sh benchmark_model --graph=vision/yolo_gate/weights/pink_det_v1_480x640_float16.tflite \
    --use_gpu=false --num_threads=3 --num_runs=100
```
**Check:** the GPU run's log says the GPU delegate was created and took the graph (look for
`Created TensorFlow Lite delegate for GPU` and a line on how many nodes it replaced; all of them is
best). Write down the average inference time of both runs.

### 7. From Python, on real frames
```bash
./run.sh python3 vision/yolo_gate/bench_yolo.py --model vision/yolo_gate/weights/pink_det_v1_480x640_float16.tflite \
    --delegate gpu --n 200 --images vision/live_runs/<a run>/raw_jpg \
    --compare-onnx vision/yolo_gate/weights/pink_det_v1_480x640.onnx
```
(`raw_jpg/` folders stay off git: copy one to the drone first, or leave out `--images` for timing only.)

**Check:**
- the first line says `GPU (OpenCL, fp16), whole graph on the GPU`. `GPU took only part of the graph`
  means some ops fell back to the CPU: still usable, but slower, so compare against `--delegate cpu`;
- `repeatability (same frame 5x): ... OK`. `NOT REPEATABLE` (what the Mac's GPU gave) = don't use
  the GPU on this setup, whatever its speed;
- the GPU's boxes agree with the `.onnx` on (nearly) every frame, within a few px (fp16: a few px
  more than fp32; `--fp32-gpu` to compare);
- the fps, with the VIO running too. About 10 fps or more is the bar for YOLO on the drone
  (`px4/VOXL2_BRINGUP_WINDOWS.md`, Step 8). Keep whichever of GPU / CPU is faster.

### 8. Live gate vision on the drone (hybrid: YOLO box + color corners)
VOXL Portal's hires stream, read on the drone itself. `--delegate gpu` if step 7 passed on the GPU,
else `--delegate cpu` (or the `.onnx`):
```bash
./run.sh python3 vision/live_gate_test.py --detector hybrid \
    --model vision/yolo_gate/weights/pink_det_v1_480x640_float16.tflite --delegate gpu \
    --url http://localhost/video_raw/hires_front_small_color --gate-side 0.8636 \
    --calib vision/calib/hires_front_1024x768.yml --duration 120 --out-dir vision/live_runs/onboard_hybrid_01
```
Do the tape-measure test while it runs (`px4/VOXL2_BRINGUP_WINDOWS.md`, Step 9). If it stops at
once with `output changes run to run`, the GPU failed the repeatability check: re-run with
`--delegate cpu`.

**Check (Mac, 2026-11-18, recorded frames served as a fake Portal feed, no Ultralytics installed):**
the hybrid detector with the `.onnx` and with the float16 `.tflite` on the CPU gave the same
detections as the `.pt` through Ultralytics on 150/150 frames (corners within 0.13 px); the whole
script ran at the feed's rate. On the drone: the fps it prints at the end, ranges against the tape.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `repack_deps.sh`: package X is not installed | the name differs on this SDK: `dpkg -l \| grep -i adreno` (and the others), fix `PKGS` |
| `docker build` fails at `FROM ... AS build` | Docker older than 17.05 (step 1) |
| build killed / `c++: fatal error: Killed` | out of memory: `--build-arg BUILD_JOBS=3` |
| `opencl_check`: no OpenCL platform | container started without `--device /dev/kgsl-3d0 --device /dev/ion` (use `run.sh`), or deps not installed: rebuild after step 2 |
| GPU fine in `opencl_check`, `gate_tflite: GPU delegate could not take the graph` | an op the GPU delegate rejects: `benchmark_model --use_gpu=true` prints which; try the float32 model, or `--delegate cpu` |
| `libgate_tflite not found` | outside the container: set `GATE_TFLITE_LIB` to the `.so` |
| Works once, then the GPU errors after an SDK upgrade | the image's driver no longer matches the kernel: redo step 2 and rebuild |
