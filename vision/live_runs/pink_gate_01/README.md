# pink_gate_01: Pink 36 in gate, color detector, lights off (2026-11-13)

Starling 2 Max `hires_front_small_color` (1024×768) via VOXL Portal (`adb forward tcp:8080 tcp:80`), run from `vision/`:

```bash
python3 live_gate_test.py --detector color --hsv-lo 0 80 70 --hsv-hi 12 255 255 --close-px 31 --gate-side 0.8636 --calib calib/hires_front_1024x768.yml --duration 300 --out-dir live_runs/pink_gate_01 --show
```

- 8,238 frames at 27.5 fps; 3,608 with a fix (44 %, including time with no gate in view).
- Range (reprojection ≤ 10 px) p5/p50/p95: 0.77 / 2.01 / 2.95 m. Reprojection p50 3.1 px, p90 7.0 px.
- Under about 0.9 m the error climbs to 10–17 px: the opening fills the most distorted part of the lens, and its rounded corners don't fit a quad.
- 69 fixes beyond 4 m are false quads on reddish background objects.
- The gate shows up coral/salmon (hue 2–6), not the sim's magenta. The white tape at the joints breaks the pink loop, hence `--close-px 31`.
- Training source for `vision/yolo_gate` (lights off).

`detections.csv` is tracked. `raw_jpg/` (every frame, for re-running offline) and `overlay.mp4` stay local.
