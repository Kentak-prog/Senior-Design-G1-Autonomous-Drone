# Starling 2 Max (C29) on the circle course: Stages 0–4, 2026-11-07

First run of `px4/STARLING_QUICK_RUN.md` in Isaac Sim + PX4 SITL with
`workspace/spawn_starling.py` (C29 front camera, our dynamics numbers) on the
circle course (`../circle_course.json`). **Stages 0–4 passed**, after three
fixes to the Starling sim found on the way (below). Pod run folder:
`/workspace/vio_runs/s29_1/` (CSVs stay on the pod).

| File | |
|---|---|
| `s3/flight_report.png`, `s3/flight_summary.txt` | Stage 3: 15 s hover hold on our MPC (ground truth) |
| `s4/flight_report.png`, `s4/flight_summary.txt` | Stage 4: the course gate_0 → gate_1 → gate_2 on ground truth, VIO as a preview |
| (pod only) | `/workspace/vio_runs/s29_1/`: every flight_log / bridge_log CSV, `vio/` (Stages 2–4) and `s4r_vio/` (4R) with `vio_log.csv` and `vio_events.csv`, and the PX4 logs in `/workspace/PX4-Autopilot/log/2026-11-07/` |
| `s4r2/`, `s4r3/` | Stage 4R attempts 1 and 2 (below): analyzer figure and summary, plus `replan_events.txt` (every plan, pass, failure) and `gate_map.txt` (the map at each plan), the pod's CSVs renamed so git keeps them |

## Results

| Stage | Result | Numbers |
|---|---|---|
| 0 pod tests | GO | all pass; Starling MPC solve median 7.7 ms |
| 1 spawn + PX4 hover | GO | 2.51 m within 5 mm, body rates within 0.25°/s; thrust 0.139 = `hover_thrust_estimate` 0.139 (motor output 0.341, 5,407 RPM); camera 1024×768, fx 439.6; clock, IMU and camera stamps agree |
| 2 dry run | GO except VIO yaw | rates and lags OK, MPC p95 11.7 ms, VIO position error max 0.16–0.18 m; **VIO yaw error a steady +3.7°** (std 0.07°) against the 3.0° limit |
| 3 hover hold | GO | max horizontal 0.01 m, max vertical 0.06 m, completed 15 s, thrust conversion 5.53 N → 0.135 with no `--hover-thrust` |
| 4 course | **ALL GATES PASSED** | below |

Stage 4, against the offline rehearsal (Starling + C29):

| Gate | Time (offline) | Offset | Frame margin | Speed |
|---|---|---|---|---|
| gate_0 | 3.96 s (3.9) | (−0.00, +0.06) m | 0.69 m | 1.57 m/s |
| gate_1 | 10.48 s (10.4) | (+0.07, +0.06) m | 0.68 m | 1.55 m/s |
| gate_2 | 17.11 s (16.9) | (+0.06, +0.06) m | 0.69 m | 1.59 m/s |

Tracking (true drone vs reference): mean 0.08 m, max 0.21 m. Altitude
within 0.06 m. Watchdog never tripped.

## Fixes needed to get here (all in `workspace/spawn_starling.py`, committed)

1. **Rotor anchors in the wrong frame.** The rotor joint anchors
   (`localPos0`) are in the Iris mesh prim's frame, 90° from the body frame;
   writing the Starling positions unconverted put every rotor 90° off, so
   PX4's roll and pitch were swapped through the motors. Flipped on takeoff
   twice. Checked offline by `workspace/test_starling_geometry.py`.
2. **Overlapping prop colliders.** The Iris' 10-inch props overlap front to
   back at the Starling spacing (hubs 0.19 m apart, props 0.247 m); the
   animated blades hit each other and spun the drone at −200°/s on the
   ground as soon as it armed. Prop colliders disabled, rotor mass set to
   5 g each (totals unchanged).
3. **Iris rate gains.** They act 3.4× (roll) and 5× (pitch) harder on the
   Starling (angular acceleration per unit of PX4 torque: 1041 / 1067 rad/s²
   against 307 / 215). Pitch oscillated at ±270°/s at liftoff and it climbed
   to 86 m. Scaled gains (roll P/I/D 0.044/0.06/0.0009, pitch
   0.03/0.04/0.0006) are now in `PX4_SITL_PARAMS` and
   `px4/starling2max_sitl_params.sh`.

## Open items

- **VIO yaw offset.** +3.7° steady at hover (Stage 2), ~1° in flight. The
  C29's wide lens (fx ≈ 440 against ≈ 1100 for the Iris camera) makes the
  gate's yaw from PnP about 2.5× coarser; a mount-model mismatch is also
  possible. Look at it before Stage 6 (flying on the VIO).
- **VIO after the last gate.** Through the course the VIO stayed within
  0.08–0.45 m of the truth (offline 0.21–0.40 m). After gate_2 the plan ends
  with the camera facing away from every gate, so the VIO dead-reckons on the
  IMU and drifts (58 m by the end of the log, hundreds of metres later). It
  matters for the closed lap (4R) and Stages 6–7.
- **Yaw loop lightly damped.** PX4 still has the Iris yaw gains; the
  Starling's yaw response per unit is ~0.74× the Iris'. A disturbance at
  hover set off a ±7° yaw swing (period ~2.5 s) that took about a minute to
  die out, and the Stage 4 plot shows a ±2.5° yaw wobble in flight.
  `MC_YAWRATE_P` 0.27 (= 0.2 / 0.74) would restore the Iris loop; untested.
- **MPC solve time** rose over the session: ~9.5 ms (Stage 2), 15–17 ms
  (Stage 3), median 24 / p95 30 ms (Stage 4). Under the ~48 ms hard limit,
  over the 15 ms GO. `ros2-eval` sat at its 2-CPU cap (MPC + vio_node +
  monitor ≈ 160 %) and was throttled.

## Stage 4R (closed lap, replanning from the camera): not passed yet

Three attempts the same day, each stopped by a different bug, each fixed:

| Attempt | What stopped it | Fix |
|---|---|---|
| setup | `ros_camera_pose_fallback.py` never got a pose: it subscribed reliable to Pegasus' best-effort `/drone00/state/pose` | best-effort subscription |
| setup | gate_1 seen from 12 m came in up to ~1 m and ~100° off; every replan failed and the CPU overload pushed the MPC p95 past 50 ms | `replan_node -p max_gate_range_m:=9.0` |
| 1 (`s4r2`) | aborted at once: the first plan started at yaw 1210 rad (the pre-plan search yaw counted from t = 0 on a ~2420 s sim clock); the MPC asked for 53° | `replanner.HoldReference` starts its clock at first use |
| 2 (`s4r3`) | passed gate_0 ("through"), corrected gate_1 to 0.12 m, then clipped close views of gate_1 (0.7–2.6 m off, often two per frame) became 5 phantom gates; the route flew at them; aborted at a 31° tilt | `replan_node -p map_new_gates:=false` (prior gates only) |

Each fix has a regression test in `Trajectory/test_replan_session.py` that
fails on the old code with the live numbers.

Attempt 2 in numbers (`s4r3/`): gate_0 passed at t = 3.89 s, offset
(−0.02, +0.05) m, frame margin 0.70 m, 1.98 m/s; the bridge ran 42.8 s wall
before the abort. Its `gate_map.txt` shows gate_1 corrected from 0.57 m to
0.12–0.13 m off and the phantom gates map_3–map_7 0.66–2.57 m from it. The
VIO preview held ~0.1 m through gate_0, then lost track in the swerving
around gate_1 (yaw error up to 28°) and dead-reckoned. The summaries'
whole-log numbers (tracking error ~9–12 m, solve p95 79–100 ms) are
dominated by the minutes after each abort, when `mpc_node` kept chasing the
stale plan with the bridge off; read them with that in mind. Video of attempt 2:
`isaac_stage4r_starling_c29_abort.webm` (user's Downloads, not in git).
After an aborted run, restart `mpc_node`: it keeps tracking the old plan's
end point (9 m error, 47° setpoint tilt at hover). The camera's close-range
gate poses are noisy with the C29 lens; the detector may need work before
Stages 6–7 fly on the VIO.

## Pod notes from the day

- `ros2-eval` sees Isaac's topics only over UDP:
  `export FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/fastdds_udp.xml`.
- A stale `/tmp/px4-sock-0` stops a new PX4 from starting its server
  (`PX4 server not running`): remove it before starting PX4.
- If PX4's `pxh>` is lost (dropped terminal), `python3 Tools/mavlink_shell.py
  0.0.0.0:14550` in `px4-sitl` gives a shell over MAVLink.
- A flipped drone can be put back upright without restarting Isaac:
  `workspace/reset_drone.py` in the Script Editor, then restart PX4.
