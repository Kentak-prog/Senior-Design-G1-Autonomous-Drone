# onboard_hybrid_gpu_01: first gate vision run ON the drone, YOLO on the GPU (2026-11-18)

Everything ran on the Starling 2 Max's VOXL 2, in the `docker/voxl2-tflite-gpu` container: VOXL Portal's
`hires_front_small_color` (1024×768) read on `localhost`, YOLO11n box (480×640 float16 `.tflite`) on the
Adreno 650 through TFLite's OpenCL delegate, color corners inside the box, PnP with the hires calibration.
The drone sat on the bench on USB‑C; the pink 36 in gate was on a stand, **about 2 m away (not
tape-measured)**. The planned 1/2/3/4 m and tilt schedule was not followed.

```bash
./run.sh python3 vision/live_gate_test.py --detector hybrid \
    --model vision/yolo_gate/weights/pink_det_v1_480x640_float16.tflite --delegate gpu \
    --url http://localhost/video_raw/hires_front_small_color --gate-side 0.8636 \
    --calib vision/calib/hires_front_1024x768.yml --duration 150 --out-dir vision/live_runs/onboard_hybrid_gpu_01
```
(Stopped with Ctrl‑C at ~140 s.)

- 1,349 frames at **9.3 fps** overall (10–15 fps in the progress lines). That is the whole loop: JPEG
  decode, YOLO on the GPU (~27 ms), color corners, PnP, overlay, and writing every raw frame. YOLO alone
  benchmarked at 27 fps on the GPU.
- A YOLO box in 1,001 frames (74 %). But **real corners (`yolo+color`) in only 161**; 788 fell back to the
  box (`bbox`), 83 box fixes rejected.
- `yolo+color` range: median **1.84 m** (p25 1.75, p75 2.11), reprojection median 2.6 px; consecutive
  fixes differ by 2 cm median. Box-only range: median 1.92 m.

## Why so few corners: the backlit pipe

The gate's right side faced bright windows. Backlit, its pink loses saturation, so the color mask
(`S >= 80`) has gaps on that bar, the loop doesn't close, no opening is found, and the hybrid falls back
to the box. Same gate, same box, a few frames apart: frame 603 got corners, 611 didn't.

Re-run offline on all 1,349 recorded frames (YOLO `.onnx` on a Mac, so slightly different boxes than the
GPU's):

| Corner step | Frames with corners (of 1,000 with a box) | Range median [p10, p90] | Reprojection median |
|---|---|---|---|
| as run: `S >= 80`, close 0.08 × box (max 31 px) | 119 | 1.85 m [1.54, 2.23] | 2.7 px |
| close 0.12 × box (max 41 px) | 560 | 1.95 m [1.72, 2.24] | 2.8 px |
| **`S >= 50`** (`--hsv-lo 0 50 50`) | **826** | **2.02 m** [1.76, 2.38] | 4.3 px |
| `S >= 50` + close 0.12 | 889 | 2.03 m [1.78, 2.44] | 4.8 px |

`--hsv-lo 0 50 50` needs no code change and gets corners on 7× more frames; its median matches the
~2 m stand-off. The higher reprojection error is the washed-out pixels it now accepts. Without a tape
measure, none of these ranges is checked against the truth.

## Next run
- `--hsv-lo 0 50 50`, gate at **measured** 1, 2, 3, 4 m, then turned ~20° and ~40°.
- Avoid having the windows directly behind the gate, or note when they are.

`detections.csv` is tracked; `raw_jpg/` (every frame) and `overlay.mp4` stay local (git-ignored).
