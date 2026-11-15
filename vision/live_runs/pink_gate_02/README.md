# pink_gate_02: Pink gate, color detector, lights on, old brightness floor (2026-11-13)

Starling 2 Max `hires_front_small_color` (1024×768) via VOXL Portal (`adb forward tcp:8080 tcp:80`), run from `vision/`:

```bash
python3 live_gate_test.py --detector color --hsv-lo 0 80 70 --hsv-hi 12 255 255 --close-px 31 --gate-side 0.8636 --calib calib/hires_front_1024x768.yml --duration 300 --out-dir live_runs/pink_gate_02 --show
```

- Stopped early: 1,510 frames, 617 with a fix (41 %).
- With the lights on, the lower-right bar sat in shadow at V ≈ 65, below the floor of 70, so the loop stayed open and detection dropped to 0 % in places. Fixed in `pink_gate_03` with V ≥ 50.
- Training source for `vision/yolo_gate` (lights on).

`detections.csv` is tracked. `raw_jpg/` (every frame, for re-running offline) and `overlay.mp4` stay local.
