# hybrid_live_01: Hybrid detector live: YOLO box, then color corners inside it (2026-11-13)

Starling 2 Max `hires_front_small_color` (1024×768) via VOXL Portal (`adb forward tcp:8080 tcp:80`), run from `vision/`:

```bash
~/venvs/yolo/bin/python live_gate_test.py --detector hybrid --model yolo_gate/weights/pink_det_v1_yolo11n.pt --conf 0.4 --device mps --gate-side 0.8636 --calib calib/hires_front_1024x768.yml --duration 300 --out-dir live_runs/hybrid_live_01 --show
```

- 2,760 frames at 26.8 fps. YOLO box in 1,999 frames; **real corners (`yolo+color`) in 1,523**. 60 frames used the box range, and 416 box-only fixes were rejected (`bbox-rej`, mostly the partial gate near the lens).
- Against the color detector run offline on every 3rd frame: 512 hybrid fixes vs 341 color (389 vs 213 with reprojection ≤ 10 px). Where both fire, range agrees: median difference 5 mm, p90 2.7 cm.
- Steady holds such as 1.425 m ± 0.8 cm (1.7 px). **Not checked against a tape measure.**
- Recorded before the corner line-fit fix in `HybridDetector` (the close kernel was rounding the opening's corners). Re-run offline on every 3rd frame with the fix: reprojection p50 4.3 px (was 4.8), p90 16.5 px (was 18.2), and ranges 1.9 cm closer at the median.
- Corner error vs gate tilt (with the fix): 2.3 px (0–20°), 3.8 px (20–40°), 6.3 px (40–60°), 23.1 px (60–90°). Past about 50° the near pipe hides part of the opening, and those fixes are wrong. No reprojection cutoff is applied yet.

`detections.csv` is tracked. `raw_jpg/` (every frame, for re-running offline) and `overlay.mp4` stay local.
