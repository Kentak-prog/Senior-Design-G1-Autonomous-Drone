# real_gate_01: first real-camera gate test, 2026-11-09

Starling 2 Max on the bench, `hires_front_small_color` (1024x768, strong barrel/fisheye distortion) via VOXL Portal
MJPEG (`adb forward tcp:8080 tcp:80`), `vision/live_gate_test.py --detector tye`. Gate: thin white PVC square,
36 in (0.914 m) side, on a pole, white-walled lab. About 240 s, 2429 frames at ~10 fps.

## Result 1: Tye's CNN accepted no frames (0 of 2429)

- The net put a raw quad on 405 frames. All 405 were rejected (`cnn-rej` in `detections.csv`); mean conf 0.26,
  max 0.41, below `--cnn-conf 0.45`. So PnP never ran.
- Expected. The net was trained only on thick purple 1.9 m outer gates in Isaac, and `good_detection()` also checks
  for purple. It needs fine-tuning on labelled white-PVC frames before it can be used on the real gate.

## Result 2: PnP from hand-labelled corners is consistent

`handlabel/label_pnp.py` takes eyeballed corners for 10 frames and snaps each side to the pipe centreline
(top-hat ridge, then a Huber line fit). It solves PnP with the **approximate** intrinsics in
`vision/calib/hires_front_small_approx.yml` (EXIF focal length, no distortion). Corners are in `handlabel/labels.json`
and overlays are in `handlabel/label_*.jpg` / `montage.jpg`.

| frame | range m | x / y / z m (OpenCV cam) | normal yaw deg | reproj px |
|---|---|---|---|---|
| 360 | 1.65 | +0.32 -0.31 +1.58 | +17.6 | 6.7 |
| 420 | 1.58 | +0.22 -0.36 +1.52 | +7.7 | 3.0 |
| 540 | 1.75 | +0.32 -0.58 +1.62 | +5.5 | 4.0 |
| 600 | 1.77 | +0.40 -0.56 +1.62 | -5.7 | 1.4 |
| 720 | 1.75 | +0.27 -0.56 +1.64 | -1.2 | 1.2 |
| 900 | 1.79 | +0.29 -0.56 +1.68 | -2.0 | 1.2 |
| 1200 | 1.79 | +0.33 -0.56 +1.67 | -7.2 | 1.5 |
| 1500 | 1.76 | +0.32 -0.56 +1.64 | -5.7 | 1.2 |
| 1800 | 1.71 | +0.24 -0.56 +1.60 | -0.8 | 0.9 |
| 2100 | 1.77 | +0.28 -0.56 +1.66 | +3.3 | 4.0 |

Frames 540-2100 (camera static) agree to within about ±0.04 m in range. 360 and 420 were taken while the camera was
still moving. **These ranges are not validated.** No tape-measured distance was recorded, and with zero distortion
the barrel lens biases range. Next run: tape-measure the camera-to-gate distance and calibrate the hires camera
with a checkerboard first.

## Files

| file | in git | |
|---|---|---|
| `detections.csv` | yes | one row per raw detection (schema in `live_gate_test.py`) |
| `handlabel/` | yes | label script, labels, overlays, montage. Regenerate with `python3 handlabel/label_pnp.py handlabel` |
| `raw_jpg/` (270 MB) | no | every raw frame, needed to re-run detectors offline. Kept locally only |
| `overlay.mp4` (60 MB) | no | annotated video. Kept locally only |
