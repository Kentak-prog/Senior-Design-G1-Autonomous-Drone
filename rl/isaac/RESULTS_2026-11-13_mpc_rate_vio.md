# Rate-loop-aware MPC on the VIO, live in Isaac, 2026-11-13

First closed-loop runs of `mpc_rate` with its state from the VIO (`ros2_nodes/vio_node.py`: IMU + front
camera on the known gate map) instead of PX4's GPS-based estimate. Same Isaac + PX4 SITL setup and
Starling 2 Max as `RESULTS_2026-11-13_mpc_rate_v8a10.md`. Gates from the track file (race-day map).

**Result: 1 of 18 runs completed** (our 3-gate course at 3 m/s / 4 m/s²). Every other run ended in a
safety abort (VIO more than 1 m from PX4's estimate), a gate hit, or never started because the VIO did
not settle. No crash: each abort handed PX4 back to hold. The controller was not the problem (1.9–2.1 ms
per tick, no oscillation); the VIO was.

## What was run

- `runner.py --state vio` (new): the controller flies on `/vio/odometry` (position, velocity, attitude;
  body rates from PX4's gyro). PX4's estimate (sim GPS, ~truth) only scores the lap and guards it:
  start only when the VIO is within 0.3 m of it for 1 s (20 s max), abort when it is more than 1.0 m
  off for 10 ticks (0.2 s) or older than 0.5 s.
- `vio_node.py` (changed): the gate size for PnP now comes from the track file (it assumed 1.5 m; Tony's
  pink openings are 0.864 m, which would have made every range 1.7× too long), the gate color range is a
  parameter (`gate_hsv_lo/hi`; `[135,60,40]`–`[179,255,255]` sees the pink gates near and far and the
  magenta ones), and `initial_pose` gives the start pose for matching the first detections (track files
  have no camera pose, so the VIO never initialized without it).
- Per run: fly to the track's start facing gate 1, start a fresh `vio_node` there (gyro calibration,
  first fix), then race. Three runs per track with the plan limits stepped up: 3/4, 5/7, 8/10
  (`--mpc-vmax/--mpc-amax`).

## Results

| Track | 3 m/s, 4 m/s² | 5 m/s, 7 m/s² | 8 m/s, 10 m/s² |
|---|---|---|---|
| three_gate_isaac (1.5 m magenta) | **3/3, 8.58 s**, margin 0.67 m, VIO err mean 0.19 / max 0.31 m | 1/3, VIO abort (1.24 m) | 1/3, gate 2 hit (VIO max 0.81 m) |
| straight3_pink36 | 0/3, VIO never settled (init 0.8 m off, diverged) | 1/3, VIO abort (1.13 m) | 0/3, VIO never settled |
| oval6_pink36 | 0/6, VIO abort (1.23 m) | 0/6, gate 1 hit (VIO max 0.58 m) | 0/6, VIO never settled |
| slalom3_pink36 | 1/3, VIO abort (1.19 m) | 0/3, VIO never settled | 1/3, gate 2 hit (VIO max **0.28 m**) |
| slalom5_pink36 | 0/5, VIO abort (1.23 m) | 0/5, gate 1 hit (VIO max 0.47 m) | 1/5, VIO abort (1.25 m) |
| circle3_isaac (1.5 m magenta) | 1/3, VIO abort (1.19 m) | 1/3, VIO abort (1.19 m) | 1/3, VIO abort (1.19 m) |

"VIO err" is the distance from PX4's estimate. Controller time 1.9–2.1 ms mean, ≤ 6.3 ms max.

## Why it failed

**1. The VIO falls behind along the direction of travel.** Splitting its error into along-track and
cross-track: cross-track stays 0.0–0.2 m, along-track grows with the VIO *behind* the drone. In the
completed 3-gate run, the VIO's speed was 96 % of the truth and it lagged about 70 ms (0.18 m at 3 m/s).
On the circle course, once the fixes stopped near gate 1, the VIO's speed decayed from 2.6 to 0.25 m/s in
2 s while the drone held about 2.6 m/s, and the error grew from 0.2 to 1.2 m.

**2. The cause is a tilt error that only gate fixes can correct.** The VIO is a strapdown filter: the IMU
pushes velocity through its attitude estimate, and nothing but a gate fix observes roll and pitch. Its
first fix sets the attitude from one PnP solve. On the circle run that left the pitch about 3° off from
the start (−2.6° against PX4's +0.1° at the hover) and it stayed there through the run: g·sin 3° ≈
0.5 m/s² of gravity leaked into the velocity, enough to bleed it off in every blind stretch. On the
completed 3-gate run the initial tilt was within 1° and the lap held. The circle's VIO also started with
a 4° yaw error (0.3° on the 3-gate run).

**3. Tony's pink gates give the VIO poor fixes.** Their 1 in pipe is 2–3 px wide at 4–5 m: many
detections are rejected (over a whole VIO session: straight3 run 1, 23 fixes vs 109 rejected; oval6
run 1, 318 vs 661; slalom3 run 1, 207 vs 685), the first fix can be 0.7–1.0 m off (4 of 12 pink runs
never got within 0.3 m to start), and the opening leaves only 0.13 m beyond the 0.30 m collision radius,
so even a VIO 0.28 m off hit a gate (slalom3, 8/10). The 1.5 m magenta gates had 9–20 fixes per
rejection.

**4. Faster = longer blind gaps.** At 5/7 and 8/10 the camera tilts further toward the ground and the
gates leave the view sooner; every faster run on the magenta courses aborted after gate 1.

## What would fix it (next steps for the VIO)

1. **Initialize roll and pitch from the accelerometer, not the PnP fix.** The VIO already averages the
   accelerometer for 2 s at the hover (the gyro calibration); gravity in that average gives roll and
   pitch to a fraction of a degree. Use PnP only for position and yaw at init. This removes the 0.5 m/s²
   leak found here.
2. **Keep tilt observable in flight**: an accelerometer gravity update when the measured specific force
   is near 1 g, or tighter tilt from the full gate fixes.
3. **Pink gates**: either a detector that refines thin pipe edges, or fall back to position-only fixes
   from them; the VIO was tuned on 1.5 m gates with 10 cm bars.
4. Then repeat this ladder starting at 3/4 on the 3-gate course before stepping up.

## Files

- Videos (main view + drone camera, one per track; not in git), user's Downloads:
  `isaac_mpc_rate_vio_<track>.webm` for three_gate_isaac, straight3, slalom3, slalom5, circle3. The oval6
  video was lost when the noVNC page reloaded (the recorder now also keeps its chunks in IndexedDB).
- Logs on the pod, `/workspace/runs_mpc_rate_vio/mpc_rate_vio_v<V>a<A>_<track>_r<n>/`: `log.csv` (with
  `vio_px/py/pz`, `vio_err`, `vio_age_ms`), `summary.json` (`vio` block), `start.txt`, `vio_node.txt`,
  `vio/vio_log.csv` (VIO vs truth every IMU step), `vio/vio_events.csv` (every fix and rejection).

## Pod setup used

- The system `python3` (ROS) lost its SciPy with a container restart; `vio_node` runs with
  `PYTHONPATH=/workspace/pylib_acados` (SciPy 1.13.1; it warns about NumPy 1.21.5, and `control/test_vio.py`
  passes 11/11 with it).
- `workspace/sim_clock.py` must be running in Isaac (it was not after the last Isaac restart).
- Every ROS shell: `FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/fastdds_udp.xml`, or nothing is received.
- One run (ros2-eval):
  ```bash
  source /opt/ros/humble/setup.bash && source /workspace/ros_ws/install/setup.bash
  export FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/fastdds_udp.xml DRONE_VEHICLE=starling2max OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  cd /workspace/Senior-Design-G1-Autonomous-Drone
  PYTHONPATH=/workspace/pylib_acados:$PYTHONPATH python3 ros2_nodes/vio_node.py --ros-args -p use_sim_time:=true \
    -p initial_pose:="[1.202,0.016,3.367,7.91]" -p gates_file:=$PWD/rl/tracks/three_gate_isaac.json \
    -p camera_pitch_deg:=0.0 -p "camera_pos:=[0.0711,-0.0003,-0.0070]" \
    -p "gate_hsv_lo:=[135,60,40]" -p "gate_hsv_hi:=[179,255,255]" -p log_dir:=/workspace/runs_mpc_rate_vio/<name>/vio &
  cd rl/isaac && PYTHONPATH=/workspace/pylib_acados:$PYTHONPATH ACADOS_SOURCE_DIR=/workspace/acados \
    LD_LIBRARY_PATH=/workspace/acados/lib:$LD_LIBRARY_PATH ACADOS_CODEGEN_DIR=/workspace/acados_codegen \
    /workspace/venvs/race/bin/python runner.py --track ../tracks/three_gate_isaac.json --controller mpc_rate \
    --state vio --mpc-vmax 3 --mpc-amax 4 --origin-offset 0 0 0.05 --thrust-scale 1.0 --max-time 90 \
    --out-dir /workspace/runs_mpc_rate_vio/<name>
  ```
  `initial_pose` is the track's start pose (`Course.start_pose()`, yaw in degrees); fly there first.

## Follow-up 2026-11-14: Swift-style VIO (tilt from gravity, gates correct position + yaw only)

After *Champion-level drone racing* (Kaufmann et al., Nature 2023): Swift's gates correct only the
translational drift of a VIO whose attitude comes from elsewhere (an Intel T265's own VIO). `control/vio.py`
got that as an option (SWIFT MODE: `attitude_updates="yaw"`, `tilt_init="accel"`, `init_fixes=5`; old
behaviour stays the default). Offline it does what it should: `control/test_vio.py` 13/13, including the
reproduction of 2026-11-13 (a PnP fix 3° off in pitch drifts 3.1 m in 3 s blind; Swift mode 0.00° and
0.000 m), and `Trajectory/simulate_vio_mpc.py` 24/24 on both magenta courses at 3/4 and 5/7, mean VIO error
0.04–0.08 m (old 0.05–0.10 m; that rehearsal has no PnP attitude bias, so the old mode passes it too).

**Live on the 3-gate course (fresh Isaac + PX4 after a pod restart): 0/3.**

| Run | Result | VIO error mean / max |
|---|---|---|
| 3/4 | gate 1, then VIO abort (1.11 m) | 0.38 / 1.11 m |
| 5/7 | VIO never settled at the start | – |
| 8/10 | gate 1, then VIO abort (1.14 m) | 0.48 / 1.14 m |

What happened:
- **The tilt fix worked.** VIO roll and pitch stayed within 1–2° of PX4's for the whole of runs 1 and 3
  (2026-11-13: a 3° bias that never went away).
- **Blind stretches still sink it.** In run 1 the gate fixes stop at 3.7 s, close to gate 2 (out of
  view), and in 1.7 s of IMU-only flight the error grows 0.4 → 1.1 m. The remaining ~1° tilt error
  (about 0.7° already at the hover) is enough.
- **New failure: nothing corrects roll/pitch any more.** In run 2 the gyro bias was calibrated while the
  drone was still turning after the move to the start (5.1°/s yaw, 0.7°/s pitch taken as "bias"). With
  gate attitude updates off, the 0.7°/s pitch error integrated without bound: −13° within ~20 s, and the
  VIO diverged before the race. Swift can trust its inertial attitude because the T265's visual odometry
  keeps it right; ours has no second source.

Conclusion: removing gate attitude updates is not enough on its own, and it needs (a) gyro calibration
only when the drone is provably still and (b) some slow, robust tilt correction (heavily down-weighted
gate attitude, or gravity when the specific force is near 1 g). The bigger gap is the blind stretch: our
VIO has nothing between gates, where Swift's base VIO tracks features of the room. On the Starling 2 Max
that role belongs to ModalAI's own VIO on the VOXL 2 (to be confirmed on the drone); in Isaac it needs a
feature-tracking VIO (e.g. OpenVINS on the front camera) or a camera yaw that keeps gates in view.

Video (main view + drone camera, not in git): `isaac_mpc_rate_vioswift_three_gate_isaac.webm`. Logs on the
pod: `/workspace/runs_mpc_rate_vio/mpc_rate_vioswift_*`. The live runs used a dev copy of `control/vio.py`
and `ros2_nodes/vio_node.py` in `/workspace/vio_dev` (VIO params `attitude_updates:=yaw tilt_init:=accel
init_fixes:=5`).

### Follow-up 2: still-only gyro calibration + slow tilt correction (2026-11-14)

`control/vio.py` options (off by default): `calib_still=True` takes the gyro bias only from a 2 s
window whose quarter means agree (gyro within 0.15°/s, accel within 0.1 m/s²), with |mean gyro| <
1.5°/s and |accel| within 0.15 m/s² of g; otherwise the window slides on. `tilt_tau_s=T` moves roll /
pitch toward each reliable fix's PnP tilt with time constant T, so a residual bias b leaves at most
about b·T. Tests 15/15 (a 5°/s turn during calibration is waited out, bias error 0.027°/s; a 0.3°/s
residual pitch bias stays at 0.5° tilt error with T = 2 s, 2.8° without). Live with
`attitude_updates:=yaw tilt_init:=accel init_fixes:=5 calib_still:=true tilt_tau_s:=5.0`:

| Run | Result | VIO error mean / max |
|---|---|---|
| 3/4 | gate 1, then VIO abort (1.10 m) | 0.39 / 1.10 m |
| 5/7 | gate 1, then hit gate 2 | 0.36 / 0.74 m |
| 8/10 | **gates 1 and 2**, then hit gate 3 | 0.32 / 0.79 m |

- **The start is fixed.** Every run started with the VIO 0.09–0.11 m from PX4's estimate; the calibration
  waited out the hover wobble (11–130 windows, up to about a minute in the pod).
- **Flight is not.** The 3/4 run lost its fixes for 1.5 s from 3.7 s, closing on gate 2, and drifted
  0.4 → 1.1 m, exactly as before. Mean tilt error is now 0.9° (1.9° at 8/10, with 40–50° tilts).
- 8/10 passed two gates for the first time on the VIO.

So the remaining failure is the blind stretch itself, not the start or a bad bias. Next: keep the
next gate in the camera (perception-aware yaw in the MPC reference) and/or a feature-tracking VIO
between gates.

Also found while running: a VIO from the previous run that is still alive (its 4–5 min timeout)
keeps publishing on `/vio/odometry`, and a runner listening then mixes two estimates (one run saw a
247 m error from a stale, drifted VIO). The run script now waits until no `vio_node` is running.
Video: `isaac_mpc_rate_vioswift2_three_gate_isaac.webm` (Downloads). Logs: `/workspace/runs_mpc_rate_vio/mpc_rate_vioswift2_*`.

### Follow-up 3: keep the next gate in the camera (2026-11-14): 5 / 6

`mpc_rate.py` has `yaw_mode="next_gate"` (runner `--mpc-yaw next_gate`): the reference yaw points the
nose and front camera at the next gate (`Trajectory/min_snap_planner.look_at_yaw_profile`, already used
by the ROS stack) instead of along the velocity. The planned path is unchanged. The bench
(`bench_mpc_delay.py --yaw`) now reports how often SOME gate is whole in the C29 camera (all four opening
corners inside 98.7° × 82.3°, within 12 m), which is what a VIO fix needs:

| Course, limits | velocity yaw: whole gate in view / longest blind | next_gate yaw |
|---|---|---|
| 3-gate, 3/4 | 74 % / 1.28 s (live: 1.5 s without fixes) | 91 % / 0.80 s |
| 3-gate, 8/10 | 66 % / 0.76 s | 77 % / 0.58 s |
| circle, 3/4 | 50 % / 3.30 s | 91 % / 0.84 s |
| circle, 8/10 | 42 % / 2.12 s | 87 % / 0.64 s |

Laps, tilt and margins are identical on the bench (only the yaw changes). Live, with the Swift-mode VIO of
follow-up 2 (`attitude_updates:=yaw tilt_init:=accel init_fixes:=5 calib_still:=true tilt_tau_s:=5.0`):

| Track | 3/4 | 5/7 | 8/10 |
|---|---|---|---|
| three_gate_isaac | **3/3, 8.68 s**, VIO max 0.37 m, margin 0.65 m | **3/3, 6.54 s**, VIO max 0.53 m, margin 0.60 m | **3/3, 5.48 s**, VIO max 0.30 m, margin 0.61 m |
| circle3_isaac | **3/3, 13.66 s**, VIO max 0.56 m ¹ | gate 1 hit, VIO 0.75 m off ¹ | **3/3, 8.48 s**, VIO max 0.76 m, margin 0.50 m |

¹ with `max_range:=9.0`: on the first two circle tries the VIO started 0.31 m from the truth (fixes of
the gate 11–12 m away, put through a ~1.5° yaw error), just over the runner's 0.30 m start threshold, so
the race never started; the cap (as in Stage 4R) fixed the start.

- **First full laps on the VIO at every speed step on the 3-gate course**, the 8/10 lap as fast as on
  PX4's GPS estimate (5.48 s against 5.50 s), and the first circle laps.
- The controller is `mpc_rate` with the team's new `N=50` (2.5 s horizon): 3.8 ms mean, ≤ 12 ms per tick
  on the pod (1.8 ms at `N=20`), inside the 20 ms budget.
- The circle 5/7 failure: the VIO error grew to 0.75 m before gate 1 (not analysed yet).

Pod: these ran from a dev copy in `/workspace/vio_dev` (the checkout plus this session's `rl/` and
`control/` changes). Videos: `isaac_mpc_rate_vioyaw_three_gate_isaac.webm`,
`isaac_mpc_rate_vioyaw_circle3_isaac.webm`. Logs: `/workspace/runs_mpc_rate_vio/mpc_rate_vioyaw_*`.
