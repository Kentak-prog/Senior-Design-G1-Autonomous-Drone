# yolo_live_02: YOLO11n box detector live, epoch-32 checkpoint (2026-11-13)

Starling 2 Max `hires_front_small_color` (1024×768) via VOXL Portal (`adb forward tcp:8080 tcp:80`), run from `vision/`:

```bash
~/venvs/yolo/bin/python live_gate_test.py --detector cnn --model <runs>/weights/epoch32.pt --conf 0.4 --device mps --gate-side 0.8636 --calib calib/hires_front_1024x768.yml --duration 300 --out-dir live_runs/yolo_live_02 --show
```

- 7,639 frames, 7,296 with a box (96 %). From about frame 1,000 on, the gate leaned beside the drone with one pink corner at the image edge, and this checkpoint boxed that partial view the whole time (reprojection p50 41 px, so the ranges are meaningless). The epoch-14 model (`pink_det_v1_yolo11n.pt`) is the one kept.
- Frames 0–1,000: a gate rolled into a diamond. Box range said 1.17 m; corners found inside the box give 1.55 m (2.3 px; 1.63 m before the line-fit corner fix). That is the case that motivated `--detector hybrid`.
- Offline test set for the hybrid detector.

`detections.csv` is tracked. `raw_jpg/` (every frame, for re-running offline) and `overlay.mp4` stay local.
