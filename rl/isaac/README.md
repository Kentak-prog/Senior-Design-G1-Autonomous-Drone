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

`runner.py`: `--track` (required), `--controller pid|rl`, `--model DIR`, `--connection`,
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
