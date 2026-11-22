# Swift-style VIO in Isaac: an emulated drifting VIO + a gate-based drift filter (2026-11-17)

Step 2 of `px4/SWIFT_VOXL2_PLAN.md`: emulate the base VIO the VOXL 2 will provide, write the drift
filter, and fly `mpc_rate` on its output.

## What was built

- **`control/drift_filter.py`**
  - `DriftEmulator`: the base VIO = Isaac's true pose + drift. A start offset (0.3 m per axis, z half),
    a scale error (2 % per axis, drawn once: the main term in flight) and a bounded velocity bias
    (0.005 m/s, 5 s correlation). Attitude and rates are the true ones (a VIO's attitude is good; Swift
    takes it straight from the VIO). Seeded.
  - `DriftFilter`: Swift's filter. State = the VIO's position error and its rate (6). Each gate
    sighting: the gate's position in the camera (PnP) put through the **base VIO's attitude** at the
    image time gives the drone's position, so `p_vio - p_fix` measures the drift; image latency is
    handled (`p_d(t_img) = p_d(now) - lag v_d`), bad matches fail association or a chi-square gate.
    Output: base VIO − estimated drift; attitude straight from the VIO.
- **`ros2_nodes/swift_vio_node.py`**: runs both in the pod (true pose from `/drone00/state/pose` and
  `/state/twist_inertial`, used only by the emulator; front camera; colour detector + SQPnP) and
  publishes the corrected estimate on `/vio/odometry`, so `runner.py --state vio` flies `mpc_rate` on it
  unchanged. `correct:=false` publishes the base VIO instead (the baseline). Logs truth, base VIO,
  estimate, true and estimated drift, and every gate sighting.
- **`control/test_drift_filter.py`** (3 tests): the 3-gate course's min-snap path at 8/10 as truth,
  synthetic sightings from the C29 camera (90.7° × 74.4°, ≤ 9 m, noise ∝ range²), 30 Hz, 60 ms latency.
  Base VIO 0.15–0.92 m off on average; corrected 0.03–0.05 m mean, ≤ 0.15 m max; injected bad solves
  rejected; without fixes the output is the base VIO.

## Live in Isaac (PX4 SITL, final C29 model, 8/10, `--mpc-yaw next_gate`)

Errors are against Isaac's **true** pose, while flying (> 1 m/s, including the braking after the lap,
where no gate is in view and the drift estimate coasts).

| Track | Drift seed | Estimate | Result | Base VIO error mean / max | Flown-on error mean / max |
|---|---|---|---|---|---|
| slalom3_pink36 (0.864 m gates) | 1 | **Swift** | **3/3, 5.46 s**, margin 0.29 m | 0.44 / 0.48 m | **0.11 / 0.33 m** |
| | 1 | no filter | **hit gate 0** | 0.47 / 0.49 m | 0.47 / 0.49 m |
| | 2 | **Swift** | **3/3, 5.44 s**, margin 0.33 m | 0.92 / 0.94 m | **0.11 / 0.35 m** |
| | 2 | no filter | **hit gate 0** | 0.89 / 0.90 m | 0.89 / 0.90 m |
| | 3 | **Swift** | **3/3, 5.46 s**, margin 0.31 m | 0.18 / 0.41 m | **0.13 / 0.34 m** |
| three_gate_isaac (1.5 m gates) | 1 | **Swift** | **3/3, 5.44 s**, margin 0.64 m | 0.44 / 0.49 m | **0.18 / 0.36 m** |
| | 1 | no filter | 3/3, 5.36 s, margin 0.38 m | 0.43 / 0.48 m | 0.43 / 0.48 m |

- **Swift VIO: 4/4 laps, all 5.44–5.46 s**, the same as flying on PX4's GPS estimate (5.50 s): the
  filter removes a drift of up to 0.9 m and the controller does not notice it.
- **Without the filter the same drift hits gate 0 on the pink gates (0/2)**; on the 1.5 m gates it
  passes, but 0.26 m closer to the frame (margin 0.38 m vs 0.64 m).
- The no-filter runs needed the runner's VIO safety checks loosened (`--vio-start-err 1.0
  --vio-max-err 2.0`): with the defaults (start within 0.3 m of PX4) they correctly refused to start.
- Pink-gate margins with Swift are 0.29–0.33 m against the Starling's 0.25 m collision radius (on GPS:
  about 0.38 m). The remaining error is the gate-fix error (PnP + camera) and the coasting between
  sightings.
- Live estimate error (0.11–0.18 m mean) is larger than offline (0.03–0.05 m): the real images, PnP
  bias, the pose/clock pairing in the node, and the braking after the lap (no gates in view) count here.

GIF: `../../graphics/gifs/2026-11-18_slalom3-pink_swift-vio-vs-no-filter_v8a10.gif` (Swift seeds 2 and 1, no filter
seed 2). Logs (runner `log.csv` / `summary.json`, `vio/swift_vio_log.csv`, `vio/swift_vio_events.csv`) in
the same folder. Videos (chase camera, Downloads): `isaac_mpc_rate_swiftvio_v8a10_*` and
`isaac_mpc_rate_nofilter_v8a10_*`.

## Follow-up (2026-11-18): circle course with the pink gates, 3/4 up to 10/10

New track `rl/tracks/circle3_pink36.json`: `circle3_isaac`'s gate positions with Tony's 0.864 m pink gates.
Drift seed 2 throughout: the base VIO was **0.93–0.96 m** off the truth for the whole flight.

**First try (gate fixes from up to 9 m, as above): 3/4 passed with a 0.25 m margin (the collision radius),
5/7 hit gate 1.** The trace showed the estimate's error growing *while* fixes came in during the turns:

- **Image delay:** fitting the error along the velocity gave **16–17 ms** (one render frame: Isaac's image
  shows the scene a frame before its stamp), about 5 cm at 4 m/s. Now compensated (`image_delay_s`).
- **Far, oblique fixes (the main term):** 0.1–0.2 m cross-track error. A pink gate at 8–9 m is ~50 px wide
  and one pixel of corner error is ~0.15 m in depth; on the circle's turns the gates are mostly seen far
  and at an angle. The base VIO drifts slowly, so **fewer, closer fixes** are better: `max_range` 9 → **6 m**.
- (Pose timestamps: the node now uses Pegasus' sim-time stamps; the measured arrival lag was ~0, so this
  was not the cause.)

**With fixes from up to 6 m:**

| Limits | Result | Smallest margin | Flown-on error vs truth, mean / max |
|---|---|---|---|
| 3/4 | **3/3, 13.60 s** | 0.32 m (was 0.25) | 0.04 / 0.20 m (was 0.09 / 0.22) |
| 5/7 | **3/3, 10.20 s** (was: hit gate 1) | 0.32 m | 0.04 / 0.09 m (was 0.19 / 0.68) |
| 8/10 | **3/3, 8.52 s** | 0.33 m | 0.04 / 0.11 m |
| 10/10 | **3/3, 8.92 s** | 0.31 m | 0.06 / 0.14 m |

- **4/4 up the ladder on the pink gates with 0.94 m of drift removed**; 8/10 equals the GPS lap on this
  layout (8.50 s, 1.5 m gates, 2026-11-13).
- **10/10 is not faster than 8/10**: peak speed only 4.8 m/s, so on this short, curvy course the lap is
  set by the 10 m/s² acceleration limit, not by v_max.
- Margins 0.31–0.33 m against the Starling's 0.25 m collision radius (6–8 cm to spare).
- The slalom3 / 3-gate runs above used the 9 m limit; not re-flown with 6 m yet.

GIF: `flight_realtime_circle_pink.gif` (5/7 with 9 m fixes, 5/7 and 10/10 with 6 m). Logs:
`mpc_rate_swift_*circle3_pink36*` (9 m) and `mpc_rate_swift6m_*` (6 m). Videos (Downloads):
`isaac_mpc_rate_swiftvio_v{3a4,5a7}_circle3_pink36_s2.webm` (9 m) and
`isaac_mpc_rate_swiftvio6m_v{3a4,5a7,8a10,10a10}_circle3_pink36_s2.webm` (6 m).

## Bring-up lesson (Isaac)

A fresh Isaac stage has no ground plane. Adding it by creating Isaac's `World` **before**
`spawn_starling.py` made Pegasus reuse that world at Isaac's default 60 Hz physics instead of 250 Hz: the
drone shook (body rates 1–2.6 rad/s at hover, hover thrust 0.151 vs 0.138) and its yaw wobbled ±10°, so
`runner.goto` never settled. Spawn first, then add the ground plane to `World.instance()`. Changing the
step on the live world did not help; it took a fresh Isaac.

## Next

- Size the drift from a real VOXL 2 VIO bag (`SWIFT_VOXL2_PLAN.md` step 1) and rerun with it.
- The circle course and 3/4, 5/7 for the full ladder.
- Swap the emulated base VIO for the VOXL's own on the drone; the filter and node stay.
