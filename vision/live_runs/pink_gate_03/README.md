# pink_gate_03: Pink gate, color detector, lights on, brightness floor 50 (2026-11-13)

Starling 2 Max `hires_front_small_color` (1024×768) via VOXL Portal (`adb forward tcp:8080 tcp:80`), run from `vision/`:

```bash
python3 live_gate_test.py --detector color --hsv-lo 0 80 50 --hsv-hi 12 255 255 --close-px 31 --gate-side 0.8636 --calib calib/hires_front_1024x768.yml --duration 300 --out-dir live_runs/pink_gate_03 --show
```

- 1,987 frames, 829 with a fix (42 %). Reprojection p50 1.1 px, p90 6.6 px.
- Steady one-second holds: 1.808 m ± 0.9 cm, 1.812 m ± 0.5 cm, 1.784 m ± 2.0 cm (each under 1 px reprojection). **Not checked against a tape measure.**
- 107 false fixes at 11–13 m: a small reddish square in the background fits a perfect quad (reprojection ≈ 0.4 px), so a reprojection cutoff does not remove them.
- Training and test source for `vision/yolo_gate` (lights on).

`detections.csv` is tracked. `raw_jpg/` (every frame, for re-running offline) and `overlay.mp4` stay local.
