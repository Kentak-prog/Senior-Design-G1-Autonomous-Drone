# px4/ — flying the min-snap course in Isaac Sim through PX4

Two ways to fly the gate course in Isaac Sim through PX4 OFFBOARD:

- **Stage 1, PX4 position control** (`fly_min_snap_offboard.py`). PX4's own
  position controller tracks the min-snap reference, streamed as position /
  velocity / acceleration / yaw setpoints. **Flown successfully on 2026-10-23**
  (`Trajectory/isaac_results/`).
- **Stage 2, the repo NMPC in control** (`ros2_nodes/mpc_offboard_bridge.py`, with
  `ros2_nodes/mpc_node.py` and `ros2_nodes/min_snap_node.py`). The NMPC sends
  attitude + thrust setpoints (`SET_ATTITUDE_TARGET`), and PX4 only runs its
  attitude and rate loops. **Untested against the cluster**; see "Stage 2" below.

| File | What |
|---|---|
| `fly_min_snap_offboard.py` | stage 1: plan from the live PX4 pose through the manifest gates, then fly it in OFFBOARD |
| `test_fly_min_snap_offboard.py` | self-tests, including the full OFFBOARD sequence against a fake PX4 (`python3 test_fly_min_snap_offboard.py`) |
| `attitude_bridge.py` | stage 2: MPC attitude/thrust setpoints → PX4 `SET_ATTITUDE_TARGET` (conversions + OFFBOARD sequence; its ROS 2 front end is `ros2_nodes/mpc_offboard_bridge.py`) |
| `test_attitude_bridge.py` | frame + thrust conversions and the OFFBOARD sequence against a fake PX4 |

## Running it

Everything runs **inside the Isaac pod**. PX4 SITL only accepts MAVLink on
localhost, and the pod's containers share localhost. Use the `ros2-eval`
container: it has numpy and OpenCV, and sees the repo on `/workspace`.

1. **Isaac is set up** (see `vision/HANDOFF.md`, "Running the sim test"):
   - `spawn_example.py` has run once.
   - PX4 is connected and the drone is hovering (`commander takeoff`).
   - The gates are spawned, so `/workspace/test_gates.json` is current.
     Spawn the gates while hovering, from the same pose you'll fly from.
2. **Install pymavlink** in the container, once per container:
   ```
   pip install pymavlink
   ```
   If pip tries to install or upgrade numpy, abort: NumPy 2 breaks ROS
   Humble's `cv_bridge`.
3. **Check the plan without flying.** This needs no PX4; use the hover pose
   (ENU, read from `/World/Iris/body`):
   ```
   python3 px4/fly_min_snap_offboard.py --plan-only --start X Y Z --heading DEG
   ```
4. **Fly:**
   ```
   python3 px4/fly_min_snap_offboard.py
   ```
   It prints the live pose and the plan, including the planned gate passes,
   asks `Fly ...? [y/N]`, and then:
   1. streams hover setpoints, then switches to OFFBOARD
   2. hovers at the start for `--hover-s` seconds (default 3)
   3. flies the trajectory
   4. holds the end point
   5. switches to AUTO.LOITER

   **Ctrl-C at any time hands the drone back to AUTO.LOITER.**
5. **Results** go to `/workspace/px4_runs/min_snap_<time>/`:
   - `plan_reference.csv`: the planned path in ENU: t, position, velocity, acceleration, yaw.
   - `flight_log.csv`: every tick's setpoint and the measured state (ENU), with the PX4 mode.
   - `summary.json`: status, gate passes of the flown path, tracking error.
   - `test_gates.json`: a copy of the manifest that was flown.

## Sim slower than real time: `--time-scale`

The first cluster run (2026-10-23, 15:07) aborted 2.2 s into the trajectory
with 1.52 m tracking error, for two reasons:

- **Isaac ran at 0.42x real time.** The drone covered only 0.42 of the
  distance its own reported velocity implies.
- **Everything else used the wall clock.** PX4 lockstep is off in
  `spawn_example.py`, so PX4 and this script run on wall time, and the
  setpoints outran the drone.

`--time-scale S` advances the trajectory at S x wall time, so the setpoints
move at the sim's pace. Every run prints the measured sim speed and, if it
differs, the value to use next:

```
[flight] measured sim speed: 0.42x real time (this run used --time-scale 1)
[flight] -> next run: --time-scale 0.42
```

The ratio depends on Isaac's frame rate (~28 FPS that day), so re-measure it
whenever the scene or the rendering load changes.

## Starting from the spawn pose: `--reposition`

The script plans from wherever the drone is. After an abort, PX4's loiter
can leave it far from the start. After the 2026-10-23 aborts it was 1.35 m
in front of gate_0, and the plan from there crept through gate_0 at 0.12 m/s.

`--reposition` flies the drone back to where it hovered when the gates were
spawned, then flies the course from there:

1. **Find the start.** It's recovered from the manifest's `camera_pose_usd`
   through the static camera mount. On the cluster manifest it agreed with
   Isaac's `/World/Iris/body` to within 4 cm.
2. **Check the path.** The straight line from the live pose to the start is
   checked against every gate's four frame bars. It refuses before touching
   OFFBOARD if the line comes within `--reposition-clearance` (default 0.6 m).
3. **Enter OFFBOARD where the drone is.** Holding at the distant start
   instead would make PX4 jump straight at it.
4. **Fly back.** It moves in a straight line at `--reposition-speed` (default
   0.5 m/s in sim units, scaled by `--time-scale`), turning to the start
   heading on the way.
5. **Settle.** It waits until the drone is within 0.15 m and slower than
   0.2 m/s (up to 20 s), then does the normal hover and the course.

The course is planned from the start pose and shown before you confirm.
All the usual aborts apply during the move.

```
python3 px4/fly_min_snap_offboard.py --time-scale 0.44 --reposition
```

## Defaults and safety

- **Speed limits:** `--v-max 2 --a-max 2` makes a slow, ~17 s run for the
  3-gate course. The repo sim used 4 m/s and 6 m/s², but PX4's default
  horizontal acceleration limit (`MPC_ACC_HOR`) is about 3 m/s². Go faster
  only after a clean slow run.
- **Refusals:** it won't fly if the drone isn't airborne (`--min-alt`, default
  1 m) or if the plan doesn't pass every gate.
- **Warning:** it warns if the drone has moved more than 1.5 m from where
  the gates were spawned.
- **In-flight aborts:** each of these holds the drone at its measured
  position for 1 s, then switches to AUTO.LOITER:
  - tracking error over `--max-error` (default 1.5 m)
  - PX4 leaving OFFBOARD mode (e.g. someone switched modes)
  - no `LOCAL_POSITION_NED` for `--stale-s` (default 0.5 s)
  - Ctrl-C
- **Gate colliders:** it's not known whether the spawned gates have
  colliders, so treat a miss as a possible crash.

## Frames

- The manifest and the planner use Pegasus world = **ENU**: x east, y north, z up.
- PX4 local = **NED**: x north, y east, z down; yaw clockwise from north.
- They share an origin (measured 2026-10-01; see
  `vision/gate_error_eval/fly_gate_approach.py`), so the conversion is:
  - position: `(e, n, u) -> (n, e, -u)`
  - yaw: `yaw_ned = 90° - yaw_enu`

Before the first flight, check this by comparing the PX4 start pose printed
by the script with `/World/Iris/body` in Isaac.

## Stage 2: the NMPC flies (untested on the cluster)

```
vision gate detector ──/iris_0/front_cam/gate_poses──▶ ros2_nodes/min_snap_node.py
                                                          │ /min_snap/trajectory (latched, stamped t=0)
state (/drone00/state/* or an Odometry) ─────────────────▶ ros2_nodes/mpc_node.py   (NMPC, 50 Hz)
                                                          │ /mpc/attitude_setpoint
                                                          ▼
                                     ros2_nodes/mpc_offboard_bridge.py ──MAVLink SET_ATTITUDE_TARGET──▶ PX4
```

**Status.** The planner, the NMPC (closed loop on the 3-gate course: all gates,
2.6 cm max error, ~7 ms per solve) and the bridge (conversions + OFFBOARD
sequence against a fake PX4) are tested without ROS. None of it has run on the
cluster, and the ROS wiring itself has not been run anywhere.

**State source.** Both nodes default to `state_source:=pegasus`, which reads
the simulator's **ground truth** (`/drone00/state/pose` + `/drone00/state/twist`).
That's fine for bringing the controller up, but not for a vision + IMU test.
`state_source:=odometry` reads a `nav_msgs/Odometry` instead, e.g. from an
estimator built on `control/Kalman_Filter_MPC.py`'s IMU + pose updates.

**Clock.** Run every node with `use_sim_time:=true`, and have Isaac publish
`/clock`. The trajectory's time base, the MPC's 50 Hz timer and all staleness
checks then follow sim time, so no `--time-scale` is needed. The bridge
measures setpoint staleness by arrival time, so it doesn't depend on which
clock the stamps use.

**Run** (in `ros2-eval`, ROS sourced):

```bash
# once: build the message package (it had no package.xml / CMakeLists.txt before)
cd /workspace && mkdir -p ros_ws/src && ln -sfn /workspace/Senior-Design-G1-Autonomous-Drone/drone_msgs ros_ws/src/drone_msgs
cd ros_ws && colcon build --packages-select drone_msgs && source install/setup.bash

cd /workspace/Senior-Design-G1-Autonomous-Drone
python3 control/test_mpc_core.py && python3 px4/test_attitude_bridge.py     # sanity, no ROS needed
python3 ros2_nodes/ros_camera_pose_fallback.py &                                    # gate detections
python3 ros2_nodes/min_snap_node.py --ros-args -p use_sim_time:=true &
python3 ros2_nodes/mpc_node.py --ros-args -p use_sim_time:=true &
python3 ros2_nodes/mpc_offboard_bridge.py --ros-args -p use_sim_time:=true             # asks before engaging
```

**Order.** Drone hovering (`commander takeoff`) → gates spawned →
`min_snap_node` logs `published plan through N gate(s)`. Until then
`mpc_node` holds the hover point. → The bridge prints the MPC's hover thrust
(about 0.556 normalized) and asks before engaging. If the drone sinks or
climbs at hover, pass `--hover-thrust` with PX4's hover throttle
(`MPC_THR_HOVER` or the hover thrust estimate). Ctrl-C hands back to
AUTO.LOITER.

Bridge aborts (to AUTO.LOITER):
- no new setpoint for 0.3 s
- the MPC solver failing for 0.5 s
- a setpoint tilted more than 45°

If someone switches PX4 out of OFFBOARD, the bridge stops without fighting them.

## Stage 3: VIO + MPC on the 3-gate course (untested on the cluster)

The NMPC flies the 3-gate course of 2026-10-23 using **only the IMU and the
camera**: no GPS, no simulator ground truth, no PX4 position.

```
/drone00/sensors/imu ─┐
/iris_0/front_cam/rgb ┴─▶ ros2_nodes/vio_node.py ──/vio/odometry──▶ min_snap_node (map + look-at yaw)
       (gate map: /workspace/test_gates.json)        └──────────▶ mpc_node ──▶ mpc_offboard_bridge ──▶ PX4
```

PX4 keeps running its own estimator (with Pegasus' GPS), but in this mode
it only uses it for its attitude loop. Its position estimate matters only
if the bridge aborts to AUTO.LOITER: that's the safety net.

### Rehearsed offline first (`Trajectory/simulate_vio_mpc.py`)

The rehearsal reproduces the same pipeline with Pegasus-like sensors: a 250 Hz
FRD IMU with Pegasus' noise and a random gyro bias, and a 1280×720 camera
through the real PnP, with latency and corner noise. Results:

- **All gates passed.** This holds for 4 random seeds and for a degraded
  camera (10 Hz, 150 ms latency, 1.5 px noise).
- **VIO error at the gates:** 0.08–0.23 m (`python_sim/vio_mpc_rehearsal.png`).

It also found three things the Isaac run needs:

1. **Level camera.** With the 15° up-tilt used for the vision tests, the
   gates' bottom edges leave the 36° vertical field of view once the drone
   climbs to gate height. The VIO then sees nothing after takeoff and dead
   reckons: about 3 m of error.
2. **Look-at yaw** (`yaw_mode:=next_gate`). Facing along the velocity turns
   the camera up to 60° away from the next gate on this slalom.

   Only **both together** fix the problem: VIO error mean 0.70 → 0.25 m, max
   2.98 → 0.94 m (and the max comes after the last gate).
3. **One clock.** Without latency compensation, 150 ms of camera latency makes
   the VIO diverge to 11 m and the drone misses gates. Compensation needs the
   IMU and the images on the same clock. Pegasus stamps the IMU with **wall**
   time and Isaac stamps images with **sim** time, so `workspace/sim_clock.py`
   is required.
4. **Speed.** The planner now defaults to 80 % of the drone's limits (9.6 m/s,
   12 m/s²) with time-optimized segment timing. The MPC alone (perfect state)
   flies that plan through all gates. **The VIO can't keep up:** at 12 m/s²
   the drone tilts about 52°, the level camera points at the ground, and the
   gates drop out of view. Rehearsal, one seed each:

   | Limits (v / a) | Plan | VIO error mean / max | Result |
   |---|---|---|---|
   | 2 / 2 | 16.0 s | 0.21 / 0.61 m | all gates |
   | 3 / 3 | 11.7 s | 0.28 / 0.85 m | all gates |
   | 4 / 4 | 10.1 s | 0.31 / 0.86 m | all gates |
   | 5 / 6 | 8.2 s | 1.07 / 2.86 m | all gates, marginal |
   | 9.6 / 12 (default) | 5.8 s | 1.68 / 4.42 m | **gate_2 missed** |

   So the runbook passes `v_max:=2.0 a_max:=2.0`. Step up to 3 and then 4
   once 2 is clean.

### Runbook

**`px4/VIO_MPC_TEST_PLAN.md`** is the runbook: eight stages, each adding
one unknown, each with a GO criterion. In brief:
- pod prep: casadi, scipy, `drone_msgs`;
- Isaac with a level camera and `/clock`;
- a dry run until `ros2_nodes/flight_monitor.py` prints `PREFLIGHT: GO`;
- hover and course on ground truth;
- the VIO open-loop on the proven position flier;
- hover and course on the VIO.

Two things differ from running the nodes by hand:
- **The plan starts on engage.** `min_snap_node` with
  `start_topic:=/mpc_offboard_bridge/engaged` publishes its plan (t = 0) only
  once the bridge confirms OFFBOARD. Otherwise the reference runs on while
  PX4 hovers by itself.
- **`flight_monitor` is the scorer and the sim-only watchdog**: VIO vs
  truth, truth vs reference, altitude. When it trips, it publishes on
  `/flight_monitor/abort`, and the bridge hands PX4 to AUTO.LOITER.
  `px4/analyze_vio_mpc_flight.py <log_dir>` scores the run the way the
  rehearsal does.
