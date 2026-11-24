# Onboard hybrid gate test vs tape measure (2026-11-23)

Live gate vision on the drone (VOXL 2, `docker/voxl2-tflite-gpu` container, README step 8), real
gates in the lab, drone held by hand. Command for every run (only `--duration` / `--out-dir` change):

```bash
./run.sh python3 vision/live_gate_test.py --detector hybrid \
    --model vision/yolo_gate/weights/pink_det_v1_480x640_float16.tflite --delegate gpu \
    --url http://localhost/video_raw/hires_front_small_color --gate-side 0.8636 \
    --calib vision/calib/hires_front_1024x768.yml --duration 30 --out-dir vision/live_runs/<run>
```

Before the runs: `opencl_check` OK (Adreno 650, OpenCL 2.0); `bench_yolo.py` fp16 on the GPU, whole
graph on the GPU, 22.5 fps YOLO only, repeatability max difference 0.

## Results

| Run | Taped | Frames | Corners found (`yolo+color`) | Corner range, median (p10–p90) | Box-only range, median | Reproj (corners) | fps |
|---|---|---|---|---|---|---|---|
| `onboard_hybrid_1m` | 1.0 m | 208 | 208 (100%) | 0.998 m (0.99–1.13) | — | 12.7 px | 6.9 |
| `onboard_hybrid_2m` | 2.0 m | 432 | 432 (100%) | 1.990 m (1.98–2.00) | — | 1.6 px | 14.4 |
| `onboard_hybrid_3m` | 3.0 m | 388 | 254 (65%) | 2.984 m (2.95–3.02) | 2.937 m (−2.1%) | 1.0 px | 12.9 |
| `onboard_hybrid_4m` | 4.0 m | 501 | 463 (92%) | 4.054 m (4.03–4.07) | 3.966 m (−0.9%) | 0.47 px | 16.7 |
| `onboard_hybrid_5m` | 5.0 m | 550 | 0 (0%) | — | 5.038 m (+0.8%, p10–p90 5.01–5.05) | 0.87 px (box) | 18.3 |
| `onboard_hybrid_02` | 3.5 m | 1214 | 10 (1%) | 3.556 m (10 frames) | 3.408 m (−2.6%) | 1.2 px (box) | 10.1 |

`onboard_hybrid_02` was 120 s, gate backlit by the windows (looked dark red, not magenta); 145 more
fixes were `bbox-rej`.

## Findings

- **Corner (yolo+color) range is within ~1–2 cm from 1 to 3 m.**
- **Corner detection drops with distance and backlight:** 100% at 1–2 m, 65% at 3 m, 1% at 3.5 m
  facing the windows. Light the gate from the front; keep windows out of the background.
- **Box-only range reads ~2.3% short, proportional to range** → the `--bbox-side` default
  (0.9144 m outer) is likely too small; ~0.935 m would fit. Tape the gate's outer side and pass it.
- **At 1 m the corner fit is loose (12.7 px reproj)**: the gate reaches the fisheye's edges; range
  still right. 2 m is the sweet spot (1.6 px, 14 fps).
- **fps** 7–14, lowest when the gate fills the frame (1 m).

## Not done yet

The 4 m run was after moving to a better-lit spot in the lab (gate not backlit): corners found in
92% of frames, +5 cm (+1.4%), the loosest corner range of the series. Its box-only range was only
−0.9%, which weakens the "bbox-side too small" explanation: tape the outer side to settle it.

At 5 m (same spot) the corner finder found nothing: the gate's pipe is ~2–3 px wide and dark
(HSV V ~50, S often < 60). An offline sweep on 55 of the frames (red HSV ranges up to 0–20 / S≥50 /
V≥30, and 170–180, close kernel 0.08–0.16 × box) also found 0. The box-only range stayed good
(+0.8%), but with no gate orientation. So: corner fixes up to ~4–4.5 m, box-only to at least 5 m.
Box-only bias across runs (−2.1%, −2.6%, −0.9%, +0.8%) isn't proportional to range, so it's not
just `--bbox-side`; it looks like hand-held pose noise.

**Turned gate (`onboard_hybrid_angled_1m`, despite the name: 2.60 m taped, gate turned ~26.6° about
its vertical axis, mounted on a stand above the camera).** 60 s, 544 frames, corners found in 543,
9.1 fps (falling from ~12 to ~8 during the run). Camera frame, median: forward z = 2.58 m (taped 2.60:
−2 cm), 0.78 m above, 0.13 m left; straight-line range 2.70 m. Gate normal (−0.526, 0.071, 0.848):
**32.0° turned, vs ~26.6° set (+5.4°)**, reproj 1.6 px. The position holds up when the gate is turned;
the angle reads ~5° high (for the drift filter only position is used; the VIO's full fixes use the
normal, with `normal_std_deg` sizing how much to trust it).

**Turned ~40° (`onboard_hybrid_angled_40deg`), same stand.** 60 s, 512 frames, corners in all 512,
~10 fps. Forward z = 2.50 m, 0.79 m above, 0.07 m left; range 2.63 m. Normal (−0.806, −0.087, 0.585):
**54.2° turned (p10–p90 54.1–54.3), vs ~40° set (+14°)**, reproj 3.5 px (looser than at 26.6°).
Between the two runs the reading changed by 22° for a 13.4° change in the gate, so this is not just a
fixed offset from the drone's heading. Open: how the angles were set, whether the opening is really
square (PnP assumes 0.8636 × 0.8636 m), and the drone's heading relative to the gate's 0°.

Measured opening: 0.84 m wide × 0.85 m tall. Not square enough to matter (worth ~1° at 40°). The
0.8636 m `--gate-side` is ~2% larger than the tape, yet it is what made ranges match the tape at 1–4 m,
so keep it as the effective size (the color mask's hole likely sits ~1 cm outside the tube's inner
edge on each side).

Likely cause of the angle error: **the tube's thickness**. Seen at an angle, the frame's depth (tube
diameter d) hides part of the opening, so it looks narrower by about d·sin θ and PnP reads a bigger
turn. That predicts the 26.6° run's 32.0° with d ≈ 9 cm (a plausible tube), and grows with angle. It
does not fully explain 54° at 40° (it predicts ~45°), so the 40° setting is worth re-measuring: with
d = 9 cm, a 54.2° reading means ~48.5° true. Fix options: model the opening's depth in PnP, or trust the
gate normal less (VIO `normal_std_deg`) at large angles.

**Re-measured 30° (`onboard_hybrid_angled_30deg`), angle set with a floor reference line.** Frames
0–229: corners in 214, forward 2.50 m, **turn 41.9° (p10–p90 41.8–42.1) vs 30° set (+12°)**, reproj
1.9 px. From ~frame 230 the gate appears turned the other way (top bar slopes down to the right, right
post not found) and only box fixes came through (296 `bbox`, 2.52 m): something moved mid-run.
A 9 cm tube would only explain ~+3° here (it would need d ≈ 20 cm), so the tube is not the main cause.
The face-on distance runs also read 11–21° of turn on gates meant to face the camera. Remaining
suspects: the drone's heading not square to the gate's 0° line (hand-held), or a calibration bias in
the upper image where the gate sits. Next: a 0° run with the drone squared to the gate.

**0° (`onboard_hybrid_angled_0deg`), gate at 2.54 m taped.** Corners in 297 of 489 frames (box-only in
most of the rest). Horizontal distance 2.48 m (−6 cm, −2.3%), straight-line 2.60 m. **Yaw −13.6°
(p10–p90 −14.2 to −11.9) for a gate set at 0°.**

Signed yaw (camera frame) across the turned-gate runs, minus the 0° run's −13.6° offset:

| Run | Set | Camera yaw | Minus offset |
|---|---|---|---|
| `onboard_hybrid_angled_0deg` | 0° | −13.6° | 0° |
| `onboard_hybrid_angled_30deg` (frames < 230) | 30° (reference line) | −41.9° | **28.3°** (−1.7°) |
| `onboard_hybrid_angled_40deg` | ~40° (atan method) | −54.1° | **40.5°** (+0.5°) |
| `onboard_hybrid_angled_1m` | ~26.6° (atan method) | −31.8° | 18.2° (−8.4°, drone likely held differently) |

So **changes in gate angle are measured to within ~2°**; the error is a near-constant ~13° offset.
Most likely the hand-held drone's heading wasn't square to the reference line (a constant offset in
the gate normal is what a yawed camera gives). A camera mount yaw or calibration bias would look the
same here; a run with the drone in a fixed jig squared to the line would tell them apart.

**0°, drone on a fixed mount squared to the reference line (`onboard_hybrid_fixed_0deg`).** Corners in
549 of 552 frames, reproj 1.4 px, 9.2 fps. Horizontal distance 2.57 m (taped 2.54: +3 cm), straight-
line 2.69 m. **Yaw −6.8° (p10–p90 −7.5 to −6.2)**, pitch +1.8°. The gate appears 3.9° left of the
camera axis (bearing), which a drone yawed ~4° off the line also produces: a yawed camera shifts the
bearing and the normal by the same amount, so the gate's angle relative to the line of sight is
−6.8 − (−3.9) ≈ −3°. Hand-holding was half of the 13.6°; the rest is a few degrees of drone heading
plus ≤3° that could be the gate itself, the mount, or calibration.

Gate sag (top leaning forward) would show as pitch, not yaw: pitch here is +1.8°, so any sag is ~2°
and does not explain the yaw offset.

**Conclusion for the angle:** the camera measures the gate's angle to within ~2–3°; the large offsets
in the hand-held runs came from how the drone was held.

## Hand-held motion (`onboard_hybrid_handheld_motion_01`)

Started over SSH with `nohup` (log in `run.log`), drone carried by hand around the lab and through
the gate for 300 s; no tape ground truth. 2779 frames at 9.3 fps (11–16 fps at the end).

- YOLO saw a gate in 1385 frames (the gate was often out of view): corners in 786 (57%), box-only
  139 (10%), rejected 460 (33%). **81% of the rejected boxes touch the image edge**: gates cut off by
  the frame, correctly dropped.
- **Gate fully in frame (958 frames): corners found in 79%.** By image sharpness (Laplacian variance,
  quartiles): blurriest 69%, middle two 87% / 90%, sharpest 69% (the sharpest quarter is mostly
  close-ups / odd views, not a blur effect). Motion blur costs ~20 points of corner detection.
- Range 0.50–2.90 m. Consecutive corner fixes differ by median 2 cm, p90 14 cm, max 39 cm (includes
  real motion: at ~1 m/s and 10 fps the drone moves ~10 cm per frame).
- Detection by 30 s block ranged 5% (gate mostly out of view) to 81%.

The turned-gate test is therefore mostly settled; several gates in view at once, the measured `--bbox-side`.

## Drone incident

The drone lost power between the 1 m and 2 m runs. On reboot `docker.service` failed with
`panic: invalid page type` in BuildKit's cache (`/var/lib/docker/buildkit/cache.db`, corrupted by the
power cut). Fix: `mv` that file to `cache.db.corrupt-bak`, `systemctl reset-failed docker.service
docker.socket && systemctl start docker`. Images unaffected; only the build cache is lost. Shut the
drone down (`poweroff`) before pulling the battery.

## Hand-held runs 2 and 3, and error figures in Tony's format

- **Run 2** (`onboard_hybrid_handheld_motion_02`): started with run 1's command, so on the drone it
  overwrote run 1's CSV/video and frames 0–2126 in `onboard_hybrid_handheld_motion_01/` (the laptop
  copy of run 1 is the original). Copied out as run 2: 2127 frames, 7.1 fps, 638 with a gate. No VIO.
- **Run 3** (`onboard_hybrid_handheld_motion_03`): 1681 frames at 5.6 fps (live overlay streamed with
  `--serve-port 8091` for part of it), with the drone's VIO logged (`qvio.csv`, `voxl-logger -v qvio`,
  ~30 Hz, state OKAY throughout) and a wall/monotonic clock stamp (`clock.txt`). Two gates were in
  the lab: gate_A on its stand (whole run) and gate_B (from ~208 s, at times held by a person).

Figures by `vision/gate_error_eval/real_gate_errors.py`, which writes Tony's `gate_errors.csv` rows
and plots them with his `plot_gate_errors.py` (same four figures as the Isaac runs):

- `figs_still_tape/`: the still runs against the tape (1–5 m, 3.5 m backlit, 2.54 m fixed mount).
  Range error only (lateral wasn't taped). Mean |error| 5.0 cm over 3690 fixes; per run 1.2 cm (2 m)
  to 8.8 cm (3.5 m, backlit, box-only).
- `onboard_hybrid_handheld_motion_03/figs/`: every detection put in the VIO world frame through the
  logged pose and the hires_front extrinsics (`/etc/modalai/extrinsics.conf`). **No ground truth**:
  each gate's reference is the median of its corner-fix estimates, so the "error" is scatter around
  that (it includes VIO drift and frame-to-pose timing; a constant bias wouldn't show).
  - gate_A: detected in 68% of the frames it was in view, median error 5.6 cm, mean 9.5 cm, p95
    32 cm. gate_B: 54%, median 19 cm (hand-held at times, so it partly moved).
  - Drone still (45–115 s): 1–5 cm. Moving: spikes to 0.3–0.6 m, partly timing: the image latency
    (Portal receive vs capture) is fitted from the data (0.45 s) but the fit is flat from 0.2–0.6 s,
    and at a few rad/s of hand rotation a 0.1 s timing error moves a 2 m gate by tens of cm.
  - Detection under 1 m drops to 21% (gate larger than the frame / cut off); 1–4 m 77–90%.
  - 17 of 960 detections matched neither gate (false boxes).
