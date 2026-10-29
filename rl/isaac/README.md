# rl/isaac: fly our controllers in Isaac Sim (Phase 1)

Runs the PID baseline and the trained RL policy (the same `controller.act`
objects `fly_course.py` flies in QuadSim) on **PX4 SITL inside Isaac Sim /
Pegasus**, over MAVLink, and ships the offline stand-in and calibration tools
needed to trust the result. **Phase 1 is written and unit-tested offline only;
nothing here has been run against the live sim** (see the last section). The MPC
(`command = "wrench"`) is refused with `NotImplementedError("MPC wrench->CTBR comes in Phase 4")`.

| File | Role |
|---|---|
| `frames.py` | NED/FRD <-> ENU/FLU vectors, quaternions, rates, yaw; `ThrustMap` (N <-> PX4 u). Pure numpy. |
| `vehicle.py` | `MavlinkVehicle` (pymavlink, lazy import), `SimVehicle` (offline PX4 stand-in on QuadSim), `VehicleState` |
| `runner.py` | `run_in_vehicle(...)` + CLI: go to the start pose, race the track, abort to loiter |
| `calibrate.py` | `params`, `origin`, `hover`, `rates` |
| `vision_gate_map.py`, `gate_vision.py`, `course_update.py`, `test_vision_bridge.py` | vision mode: gate map, camera front ends, pushing a new course into a controller (see "Vision mode") |
| `test_isaac_bridge.py` | tests (plain functions, `All tests passed.`) |
| `../../workspace/spawn_track.py` | Isaac Script Editor script: build a whole track from `rl/tracks/*.json` |

## Runbook (cluster)

1. Isaac Script Editor: `spawn_example.py` once (spawns the Iris at [0, 0, 2.5]; it falls on Play).
2. Isaac Script Editor: `import os; os.environ["SPAWN_TRACK"] = "straight3"` then run
   `workspace/spawn_track.py` (`exec(open("/workspace/scripts/spawn_track.py").read())`).
   `SPAWN_TRACK` is a bare track name (resolved to `/workspace/race/rl/tracks/<name>.json`) or a path.
   It is one-shot and the script fails loudly if it is unset. Gates go under `/World/RaceTrack`
   (spawn_test_gates uses `/World/TestGates`), the manifest to `/workspace/track_gates.json`.
3. Press Play. Start PX4 SITL (as for the vision flights). Take off: `commander takeoff`.
4. In the controller container: `source /workspace/venvs/race/bin/activate`, then
   `cd /workspace/race/rl/isaac`. The venv needs numpy + torch (+ stable-baselines3 for `--controller rl`) + pymavlink.
5. Calibrate, in this order (all take `--connection`, default `udpin:0.0.0.0:14540`; each ends with AUTO.LOITER):
   ```
   python calibrate.py params
   python calibrate.py origin                    # then compare with ros2 topic echo /drone00/state/pose --once
   python calibrate.py hover --origin-offset X Y Z [--alt 3]
   python calibrate.py rates --origin-offset X Y Z --thrust-scale S
   ```
6. First flight (the geofence defaults to the track's box + 8 m, z 0.5 .. top gate + 4 m):
   ```
   python runner.py --track ../tracks/straight3.json --controller pid \
       --origin-offset X Y Z --thrust-scale S --out-dir ../runs/isaac/pid_straight3
   python runner.py --track ../tracks/oval6.json --controller rl --model ../runs/best ...
   ```
   Dry-run anything offline first with `--sim` (SimVehicle, no PX4): `python runner.py --track ../tracks/oval6.json --controller pid --sim`.

Ctrl-C at any time commands AUTO.LOITER. Run from `rl/isaac/` or `rl/`; every module adds `rl/` to `sys.path` itself.

## CLI

`runner.py`: `--track` (required), `--controller pid|rl|mpc_ctbr`, `--model DIR`, `--connection`,
`--origin-offset X Y Z` (default 0 0 0), `--thrust-map model|linear`, `--u-hover`,
`--thrust-scale`, `--geofence XMIN YMIN ZMIN XMAX YMAX ZMAX` (ENU; default `auto_geofence`:
track box + 8 m), `--max-time 60`, `--ground-z 0.15`, `--out-dir` (writes `log.csv`, `summary.json`),
`--sim`, `--sim-start X Y Z` (with `--sim`: where the stand-in starts hovering).

`calibrate.py {params,origin,hover,rates}`: the vehicle options above plus `--out-dir`;
`origin --seconds 5`; `hover --alt 3 [--u u1 u2 ...]`; `rates --alt 3 --rate-rp 1.5 --rate-yaw 1.0`.

### Runner phases and aborts
A: stream position setpoints > 1 s, enter OFFBOARD (up to 5 tries), fly a 2 m/s carrot to
`course.start_pose()`; done when |pos err| < 0.25 m, |vel| < 0.25 m/s, |yaw err| < 0.15 rad
for 1 s of sim time (timeout 60 s). B: `controller.reset(1)`, one tick per `control_dt` of **sim**
time, gate logic identical to `fly_course.run_controller`. After the last gate: hold the position
1 s, then loiter. `summary.json` `crash_kind`: `gate`, `ground` (z < 0.15), `tilt` (> 80 deg),
`oob` (> 30 m from the target gate), `geofence`, `timeout`, `interrupt`, `link` (no PX4 state for 3 s wall),
`offboard`. `crashed` is true only for gate/ground/tilt/oob; `aborted` for any. Other keys: the
`fly_course` summary keys, `min_margin` (distance from the gate-plane crossing to the opening edge,
as in `compare.py`), `ctrl_ms_mean/max`, `real_time_factor` (sim s / wall s). `log.csv` has a `phase` column
(0 start, 1 race, 2 final hold).

## What each calibration output means

- **params** prints PX4 values. `THR_MDL_FAC` must be 0 (ThrustMap assumes normalised thrust = motor output).
  `MC_*RATE_MAX` (deg/s) are for reference only: in the cluster's PX4 v1.18 offboard body rates go straight to
  `vehicle_rates_setpoint` and only manual acro input is scaled by the max rates, so the policy's 6/6/3 rad/s are
  not clamped (checked in the source, 2026-10-20). Nothing here writes PX4 params.
  `COM_OF_LOSS_T` / `COM_OBL_RC_ACT` are the offboard-loss failsafe. A name that prints `no answer` is not known to this PX4.
- **origin** prints ENU positions computed with zero offset. Compare with the ROS pose at the same moment:
  `origin_offset = ros_position - printed mean`. Yaw is printed too; if it differs from the ROS pose yaw,
  PX4's heading origin is not Pegasus' axes and a pure offset is not enough (not handled).
- **hover** steps u and fits `T_real = scale * T_model(u)`. Use `scale` as `--thrust-scale`. The `u where az = 0`
  line is an independent hover-throttle estimate (`--thrust-map linear --u-hover`). If the per-step `scale_i`
  disagree strongly, the mapping is not a pure scale. Steps that reach 0.7 m of the 1 m / 8 m altitude limits are
  cut short; reaching the limit aborts. Start with `--alt 4` if you suspect the thrust is far off.
- **rates** doublets per axis, vehicle vs QuadSim: rise time to 63% and peak rate. A longer vehicle rise time
  means PX4's rate loop is softer than `QuadParams.rate_kp` (25 ms roll/pitch, 50 ms yaw); a peak below the
  command means PX4 limits or saturates. The doublet is 13 + 13 + 25 ticks (0.26 / 0.26 / 0.5 s).

## Frames (derivations are in `frames.py`)

World ENU, body FLU, quaternions WXYZ body -> world (rl/); PX4 local NED, body FRD, attitude quaternion
FRD -> NED, rates FRD. `world_enu = ned_to_enu(local_ned) + origin_offset_enu`.
`R_enu_flu = M R_ned_frd F`, `M = [[0,1,0],[1,0,0],[0,0,-1]]`, `F = diag(1,-1,-1)`, as quaternions
`q_enu_flu = q_M * q_ned_frd * q_F` with `q_M = (0,s,s,0)`, `q_F = (0,1,0,0)`, `s = 1/sqrt 2`.
`yaw_ned = pi/2 - yaw_enu`. Rates FRD (p,q,r) -> FLU (p,-q,-r). Level and facing north is NED/FRD identity and
ENU yaw +90 deg.

Thrust: `SET_ATTITUDE_TARGET` with `type_mask = 128` (attitude ignored), body rates FRD, thrust in [0,1]. `ThrustMap("model")`:
`u = (sqrt(T / 4 k_f) - 100) / 1000` (default Iris hover u ~ 0.556), assuming `THR_MDL_FAC = 0` and Pegasus `w = 1000 u + 100`.
`SET_POSITION_TARGET_LOCAL_NED` uses type mask 2552 (position + yaw). Custom modes: OFFBOARD main 6, AUTO.LOITER main 4 sub 3.

## Tests

```
cd rl/isaac && /usr/local/bin/python3.12 test_isaac_bridge.py     # ~20 s
```
The end-to-end tests fly PID (straight3, oval6 with a non-zero origin offset) and the RL policy (`runs/best`, straight3
and oval6; skipped if missing) through `run_in_vehicle` on `SimVehicle`, which stores its state as PX4 would (NED, FRD)
so every conversion is exercised, and assert the lap time is within 0.5 s of `fly_course.run_controller`. Deliberately
wrong conversions (outbound pitch rate, attitude, yaw) must make them fail.

## Vision mode (gates from the camera, 2026-10-24)

The controller does not get the track file. It starts from a ROUGH PRIOR (the track's gates perturbed by
`--prior-pos-std` m per axis and `--prior-yaw-std` deg, seeded) with the gate count and order known, and camera
detections correct a world-frame gate map. **Scoring stays against the TRUE track** (gate pass / collision / `oob`
safety / summary margins): the runner is the referee and sees truth, the controller sees only the map.

```
# offline, no ROS/Isaac: SimVehicle + a geometric camera model
python runner.py --track ../tracks/oval6.json --controller pid --sim --gates vision --sim-vision \
    --prior-pos-std 0.5 --prior-yaw-std 10 --prior-seed 0 --out-dir ../runs/isaac/vis_pid_oval6
# live (cluster pod: ROS 2 Humble sourced, rclpy + cv2 importable by the venv)
python runner.py --track ../tracks/straight3.json --controller pid --gates vision \
    --origin-offset X Y Z --thrust-scale S --mount-xyz 0.30 0 0.05 [--hfov-override 60]
```
Options: `--gates truth|vision` (default truth: unchanged behaviour), `--prior-pos-std 0.5`, `--prior-yaw-std 10`,
`--prior-seed 0`, `--mount-xyz X Y Z` (camera in the body frame, default 0.30 0 0.05; it was once found at
-0.040 0 -0.041, so MEASURE IT, see vision/HANDOFF.md), `--mount-rot-xyz-deg` (default 105 0 -90), `--hfov-override`
(rebuild the intrinsics from the image size, for a stale `camera_info`; 60 for spawn_example.py), `--sim-vision`
(with `--sim`), `--cam-namespace /iris_0`, `--pose-topic /drone00/state/pose`. `--controller` also takes `mpc_ctbr`.
The auto geofence is built around the PRIOR gates (+8 m).

| Piece | What it does |
|---|---|
| `gate_map.GateMap` | per gate: position by a recursive weighted mean (variance shrinks with detections, measurement variance grows with camera range: `(0.10 + 0.02 r)^2`, floor 0.08 m), yaw only from detections with `normal_reliable`, the 180-deg normal ambiguity resolved so gate +Z points along the PRIOR's flight direction, yaw further than 60 deg from the prior rejected. Association: nearest gate centre within 2.5 m, one detection per gate, only gates from the current target on (`set_progress`), 3 gates lookahead, ties go to the earlier gate. Full description in the module docstring. |
| `gate_vision.process_frame` | pure: RGB image + intrinsics + body pose + mount -> world-frame detections, using vision/ (`detect_gates_color`, `estimate_gate_pose`, `gate_pose_world`, `optical_frame=False`). Drops reprojection RMS > 4 px, range > 25 m, openings touching the image border. |
| `gate_vision.RosGateVision` | rclpy node in the RUNNER process: `<ns>/front_cam/rgb`, `camera_info`, `/drone00/state/pose` (all `qos_profile_sensor_data`), a worker thread that processes only the newest image (frames are dropped, never queued). `drain()`, `stats()`, `wait_ready()`, `close()`. |
| `gate_vision.SimGateVision` | offline stand-in: a gate is detected when all 4 opening corners are inside the 60 x 36 deg FOV and within 15 m of the real camera pose; noise grows with range, +1% range bias, 15 Hz, `normal_reliable` only when the view is >= 12 deg off the gate normal. |
| `course_update.update_course` | pushes the map into the controller without a reset: the controller's own `update_course(course, pos, vel)` if it has one (MPC), else adapters that rebuild PID `centers`/`normals` (stage, integrator kept) and the RL `CoursePool` (previous action kept). The runner pushes only when the map changed, and once right after `controller.reset`. |

**Logged.** `log.csv` gets `n_det` (detections drained at this state) and `map_err` (3-D error of the CURRENT target
gate's estimate vs truth, logging only). `summary.json["vision"]`: detections total / associated, ticks with detections,
per-gate map error AT THE MOMENT THE GATE WAS CROSSED (3-D, in the gate plane, yaw), their means, final and prior errors,
per-gate filter stats, and the camera front end's stats (frames in / processed / dropped, ms per frame, detections per frame,
rejections). `format_summary` prints it.

**Phase A** flies to the PRIOR start pose and feeds the map as it goes. With the camera mounted 15 deg up and a 36 deg vFOV,
a level hover 4 m in front of a gate at the same height cannot see the gate's lower corners, so gate 0 is usually NOT
corrected before the race starts (offline: 0.23-0.25 m in-plane error at the crossing of oval6's gate 0). It is
corrected as the drone pitches down while accelerating.

**Offline results** (SimVehicle + SimGateVision, prior 0.5 m / 10 deg, seeds 0-2, in-plane map error at crossing is what
decides passing; passing needs it below about 0.45 m):

| Controller | Track | Truth gates | Vision, seeds 0 / 1 / 2 | Prior only (no detections), seeds 0 / 1 / 2 |
|---|---|---|---|---|
| PID | straight3 | 8.26 s | 8.20 / 8.26 / 8.16 s | all hit a gate |
| PID | oval6 | 18.70 s | 18.72 / 18.48 / 18.94 s | all hit a gate |
| RL | straight3 | 3.06 s | 3.06 / 3.14 / 3.12 s | all hit a gate |
| RL | oval6 | 7.46 s | **fails at gate 4 (3/6) on all 3 seeds** | hits a gate |
| MPC (`mpc_ctbr`) | straight3 / oval6 | see mpc_ctbr | 9.42 s / 24.82 s (seed 0) | hits a gate |

RL on oval6 fails even with noise-free detections and with a prior that is only 0.1 m wrong (a controller flown on a
0.1 m-perturbed gate map misses a TRUE gate in 4 of 5 seeds): the policy's oval6 lap has an edge margin of 0.31 m against
the 0.30 m the pass test needs. This is the policy's margin, not the vision path.

**Assumptions.** Gates are upright, 1.5 m opening, magenta, count and order known; the pose topic is the body frame
(FLU in ENU); the camera pose is the latest pose at image arrival (stamps are on different clocks: sim vs wall), so a moving
drone adds `speed x (image transport delay)` to the world error (0.04 m per 30 ms at 1.5 m/s, 0.3 m at 10 m/s); detections are
stamped with the image (sim) time.

**Untested live.** `RosGateVision` has only run against a fake rclpy (QoS arguments, threading, dropping, stale-camera_info
handling, pairing logic); nothing with real topics: that the three topics deliver, `Image.encoding` (rgb8/rgba8 handled),
the real frame rate and ms per frame on the pod's CPU (the worker thread shares the GIL with the control loop), image-to-pose
latency, camera_info after the refresh helper, and the mount value. Not checked against the live sim: that detections of
the real gates associate within 2.5 m of a perturbed prior, the 4 px reprojection and 3 px border thresholds, the
`normal_reliable` rate in flight (SimGateVision assumes "12 deg off-axis"), partial or occluded gates, other gates in
the background, the ~1% range bias (modelled but not corrected), motion blur at speed. The MPC hook (`update_course`) is the MPC
author's code; only the straight3 / oval6 seed 0 runs above exercised it.

## Untested against the live sim

Checked against the cluster's sources on 2026-10-20 (not flown): mask 128 with roll+pitch rates and throttle set makes
PX4 v1.18 publish `vehicle_rates_setpoint` in OFFBOARD (`mavlink_receiver.cpp`), and the cluster's pymavlink
`set_attitude_target_send` takes exactly the arguments we pass (no `thrust_body`).

- Everything MAVLink: only a fake pymavlink master (call arguments, conversions) is tested. Not verified: that
  PX4 sends `ATTITUDE_QUATERNION` (31) at the requested 100 Hz, that `MAV_CMD_SET_MESSAGE_INTERVAL` is honoured for
  HEARTBEAT, and that PX4 stays in OFFBOARD when the stream switches from position to attitude-target messages.
- INT32 parameter encoding in `PARAM_VALUE`/`PARAM_SET` (bytewise assumed), and the existence of `COM_OBL_RC_ACT`.
- That PX4's rate-loop tuning and `THR_MDL_FAC` let the CTBR commands through as modelled; that Pegasus
  `w = 1000 u + 100`, mass, k_f and the Iris USD match `QuadParams` (HANDOFF TODOs); the thrust-to-u mapping at all.
- The origin offset is a pure translation; the ENU/NED heading alignment is assumed, not measured.
- PX4 time (`time_boot_ms`) is 1 ms resolution; the 0.5 / 1.5 s waits and 3 s wall `link` timeout are guesses; control
  ticks assume the controller process keeps up with Isaac's real-time factor.
- `SimVehicle`'s "PX4" is QuadSim plus a simple position hold: it has no EKF noise, latency, motor lag, or rate limits.
- `spawn_track.py` was never run inside Isaac (omni/pxr code paths: `_clear_root`, `_make_material`, `_build_gate`,
  `_ensure_light` called with a patched `ROOT_PATH`). The gates have no colliders (`spawn_test_gates._add_bar`
  makes plain cubes); gate hits are scored geometrically by `course.check_gate`, not by physics.
- The cluster venv's packages (torch, SB3, pymavlink) were not checked.
