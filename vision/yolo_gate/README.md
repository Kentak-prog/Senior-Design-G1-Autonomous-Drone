# YOLO gate detector (real pink 36 in gates)

A YOLO11n **box** detector for the lab's pink 36 in gates, trained on real `hires_front` frames that were labelled
automatically from the color detector. Live, it runs as the first stage of `live_gate_test.py --detector hybrid`:
YOLO finds the box, then the color mask inside that box gives the gate opening's four corners for PnP.

| File | What it is |
|---|---|
| `weights/pink_det_v1_yolo11n.pt` | The trained model (epoch 14 of 34; training stopped early with patience 20), 5.5 MB |
| `results/pink_det_v1/` | Training curves, PR curve, confusion matrix, `args.yaml`, `results.csv` |
| `make_det_dataset.py` | Builds the box dataset from `vision/live_runs/<run>/detections.csv` + `raw_jpg/` |
| `make_dataset.py` | Same for a 4-keypoint pose model (the `labels_for` helper is shared) |
| `preview_labels.py` | Low-res JPEG previews and contact sheets of every label |
| `datasets/`, `runs/` | Local only (gitignored): symlinked images, labels, previews, Ultralytics run folders |

## How the labels are made

- Only the color detector's **clean** fixes are used: reprojection ≤ 3 px and range < 4 m. Far false quads (the 11–13 m ones)
  stay unlabelled, so the model learns them as background.
- The box is the gate's **outer silhouette**. The opening corners give the pose; then the 0.914 m outer square is projected
  back through the fisheye calibration, sampled along each edge so the curved sides fit inside the box.
- Each run is split by **time**: train, then val, then test, with a 30-frame gap between parts. Neighbouring frames at 30 fps
  are near-duplicates, so a random split would leak.

`pink_det_v1`: runs `pink_gate_01` (lights off, every 2nd frame) + `pink_gate_02` + `pink_gate_03` (lights on). Train 1,056 images (549 lights off,
507 lights on), val 189, test 476.

```bash
python3 make_det_dataset.py --runs pink_gate_01:2 pink_gate_02:1 pink_gate_03:1 --out datasets/pink_det_v1
python3 preview_labels.py datasets/pink_det_v1
~/venvs/yolo/bin/yolo detect train model=yolo11n.pt data=datasets/pink_det_v1/data.yaml epochs=60 imgsz=640 \
    batch=16 device=mps project=runs name=pink_det_v1 patience=20
```

## Results

**Held-out test** (476 frames): precision 0.979, recall 1.000, mAP50 0.995, mAP50-95 0.959. Validation mAP50-95 is 0.985.
The test frames come from the same three recordings (same room, gate and people) and only cover cases the color detector
handled, so these numbers are a best case.

**Live** (details in each run's README):

| Run | Detector | Frames with a fix | Notes |
|---|---|---|---|
| `live_runs/yolo_live_01` | box only, epoch 14 | 73 % | 3.5× the clean color fixes on the same frames. Box range reads 11 cm (5 %) short. A rolled gate breaks box range |
| `live_runs/yolo_live_02` | box only, epoch 32 | 96 % | Mostly a partial gate leaning beside the drone; the box-based ranges there are meaningless |
| `live_runs/hybrid_live_01` | **hybrid**, epoch 14 | 55 % with real corners | 1.8× the color detector's fixes; range matches color within 7 mm (p50) |

The hybrid's corners are line fits to the middle of each side of the opening (RANSAC), so the corner rounding from
the close kernel and hands on the pipe don't bias them. Corner error grows with gate tilt: 2.3 px at 0–20°, 3.8 px at
20–40°, 6.3 px at 40–60°, 23 px past 60°. Past about 50° the near pipe hides part of the opening (offline re-run of
`hybrid_live_01`).

Ultralytics runs from its own venv: `python3.12 -m venv ~/venvs/yolo && ~/venvs/yolo/bin/pip install ultralytics pandas`.
