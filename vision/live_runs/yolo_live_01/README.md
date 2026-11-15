# yolo_live_01: YOLO11n box detector live, epoch-14 model (2026-11-13)

Starling 2 Max `hires_front_small_color` (1024×768) via VOXL Portal (`adb forward tcp:8080 tcp:80`), run from `vision/`:

```bash
~/venvs/yolo/bin/python live_gate_test.py --detector cnn --model yolo_gate/weights/pink_det_v1_yolo11n.pt --conf 0.4 --device mps --gate-side 0.8636 --calib calib/hires_front_1024x768.yml --duration 300 --out-dir live_runs/yolo_live_01 --show
```

- 4,669 frames at 27.5 fps, 3,416 with a box (73 %). Confidence p50 0.94.
- Color detector run offline on every 3rd frame: 255 clean color fixes vs 1,138 YOLO frames. YOLO caught 885 the color detector missed (tilted gates, hands on the pipe, gates out to about 5.4 m); color caught 2 that YOLO missed.
- Range from the box alone, treating it as the 0.914 m outer square, reads **11 cm (5 %) short** of the color range, at every distance. The fisheye bulges the box outward. 90 % of frames are within 27 cm.
- Box range is wrong for rolled or edge-on gates. 40 frames have an extra box spanning the gate and a person in front of it.
- Led to the hybrid detector (`hybrid_live_01`).

`detections.csv` is tracked. `raw_jpg/` (every frame, for re-running offline) and `overlay.mp4` stay local.
