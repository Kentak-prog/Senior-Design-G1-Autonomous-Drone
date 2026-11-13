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
