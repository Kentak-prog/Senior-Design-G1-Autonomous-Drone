# Starling 2 Max: quick run of Stages 0–4 (circle course)

A one-page checklist of `px4/VIO_MPC_TEST_PLAN.md` Stages 0–4 for the
Starling 2 Max (C29 camera) on the circle course. The background and every
GO criterion are in the test plan: its Stage sections, "The circle course",
and "The Starling 2 Max (C29) on the circle course".

The sim uses our numbers: `workspace/spawn_starling.py`, the `starling2max`
profile and `px4/starling2max_sitl_params.sh`, which `control/test_vehicles.py`
checks against each other. Go in order and stop at the first failed GO.

**First run, 2026-11-07: Stages 0–4 passed** (Stage 4: all gates on ground
truth, margins 0.68–0.69 m). Results, the three sim fixes it took, and the
open items: `Trajectory/isaac_results/circle_course/starling_c29_2026-11-07/README.md`.

**Before:** commit and push the working tree, then pull in the pod (Stage 0).

---

## Stage 0: pod prep

```bash
cd /workspace/Senior-Design-G1-Autonomous-Drone && git fetch && git checkout main && git pull
```
If the container restarted since the last run, redo the installs, the
one-thread BLAS line and the `drone_msgs` build (test plan, Stage 0).

Tests, default vehicle (Iris), then the Starling:
```bash
for t in control/test_mpc_core.py control/test_vio.py control/test_vehicles.py Trajectory/test_min_snap_planner.py \
         px4/test_attitude_bridge.py px4/test_flight_checks.py px4/test_analyze_vio_mpc_flight.py; do
  python3 $t | tail -1; done
DRONE_VEHICLE=starling2max python3 px4/test_attitude_bridge.py | tail -1
DRONE_VEHICLE=starling2max python3 control/test_mpc_core.py 2>&1 | grep -E "solve median|Error"
```
**GO:**
- Every test in the loop prints `All tests passed`.
- The Starling bridge test passes. Its thrust line shows hover → about 0.139.
- The Starling `test_mpc_core` prints a `solve median` line, then fails only
  at `test_hover_hold_through_px4_attitude_loop` with
  `AssertionError: (0.05, ...)`. That is known: an Iris-only check that hover
  without the attitude lead must drift. Write down the solve median. Under
  15 ms is the test plan's GO; the hard limit at a 0.42× sim is about 48 ms.
  On a laptop it ranged 13–25 ms, against 7–15 ms for the Iris.

**Every ROS shell** from here on:
```bash
source /opt/ros/humble/setup.bash && source /workspace/ros_ws/install/setup.bash \
  && cd /workspace/Senior-Design-G1-Autonomous-Drone && export DRONE_VEHICLE=starling2max \
  && export FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/fastdds_udp.xml
```
`DRONE_VEHICLE` sets the MPC's model and the bridge's thrust conversion.
Forgetting it in one shell flies the Starling with Iris numbers.
`FASTRTPS_DEFAULT_PROFILES_FILE`: since the 2026-11-06 pod reset, Isaac's
topics reach `ros2-eval` only over UDP (shared memory between the containers
does not work). Without it, `ros2 topic echo` and the nodes see nothing.

**PX4 parameters from a shell** (instead of pasting them at `pxh>`): in a
second `px4-sitl` shell, with PX4 already running,
`PX4_PARAM=/workspace/PX4-Autopilot/build/px4_sitl_default/bin/px4-param sh px4/starling2max_sitl_params.sh`.
If a client says `PX4 server not running` while PX4 is up, a stale
`/tmp/px4-sock-0` from an earlier PX4 is in the way: stop PX4,
`rm -f /tmp/px4-sock-0 /tmp/px4_lock-0`, start it again. PX4's logs are in
`/workspace/PX4-Autopilot/log/<date>/`.

**Isaac geometry check** (no Isaac needed; `isaac-sim` container), after
any change to `workspace/spawn_starling.py`:
```bash
cd /workspace/Senior-Design-G1-Autonomous-Drone && L=$(ls -d /isaac-sim/extscache/omni.usd.libs-*) \
  && PYTHONPATH=$L LD_LIBRARY_PATH=$L/bin /isaac-sim/kit/python/bin/python3 workspace/test_starling_geometry.py
```
It must end with `ALL OK`.

## Stage 1: Isaac + PX4

Stop any PX4 left over from an earlier spawn first (`vision/HANDOFF.md`).
Use a fresh Isaac process with the Flat Grid. In the Script Editor, once
each, in this order:
```python
exec(open("/workspace/Senior-Design-G1-Autonomous-Drone/workspace/spawn_starling.py").read())
```
```python
exec(open("/workspace/Senior-Design-G1-Autonomous-Drone/workspace/sim_clock.py").read())
```
```python
import os; os.environ["GATES_MANIFEST"] = "/workspace/Senior-Design-G1-Autonomous-Drone/Trajectory/isaac_results/circle_course/circle_course.json"; exec(open("/workspace/Senior-Design-G1-Autonomous-Drone/workspace/spawn_gates_from_manifest.py").read())
```
The spawn log must show:
- `[starling] camera 1024x768, hFOV 98.7 deg, at (0.0711, -0.0003, -0.007) m (body FLU), pitch 0.0 deg up`
- four `[starling] rotorN anchor in body frame (...)` lines: rotor0
  (+0.095, −0.130), rotor1 (−0.095, +0.130), rotor2 (+0.095, +0.130),
  rotor3 (−0.095, −0.130). Any other order means the rotors are not where
  PX4 mixes them, and it will flip on takeoff (what happened on 2026-11-07,
  before the fix in `apply_starling_geometry`).
- `[starling] rotors moved to x=+-0.095 m, y=+-0.13 m (4 rotor joint anchors updated)`
- four `[starling] rotorN: 1 prop collider(s) disabled, mass 0.005 kg`. The
  Iris' 10-inch props overlap at the Starling spacing; with their colliders
  on, the blades hit each other and the drone spun at −200°/s on the ground
  as soon as it armed (2026-11-07).
- `[starling] body mass 0.5530 kg + rotors 0.0200 kg = 0.573 kg`
- `[init] ... hover 5407 RPM = motor output 0.341 = hover thrust 0.139`, then
  a block of `param set` lines. Copy them.

The gate spawner prints gate_0 (4.911, −0.667, 2.600) yaw −7.20, gate_1
(8.267, −8.650) yaw −127.20, and gate_2 (−0.325, −7.565) yaw +112.80.

Press **Play**, then start PX4 (Isaac's sim waits for it):
```bash
cd /workspace/PX4-Autopilot && PX4_SIM_MODEL=none_iris ./build/px4_sitl_default/bin/px4 ./ROMFS/px4fmu_common/ -s ./ROMFS/px4fmu_common/init.d-posix/rcS -i 0
```
At `pxh>`, once `Ready for takeoff!` appears:
1. **Paste the `param set` lines** from the spawn log (the same set as
   `px4/starling2max_sitl_params.sh`): thrust model, Starling rotor geometry
   and yaw ratio, `MPC_USE_HTE`, two sim-only settings, and the Starling's
   roll and pitch rate gains. `MPC_USE_HTE` prints "not found" on the pod's
   PX4 build (its hover-thrust estimator always runs); that is harmless.
2. Run `commander takeoff`.
3. After about 20 s of hover, check the motors:
   ```
   listener actuator_motors
   listener hover_thrust_estimate
   ```

**GO:**
- **Hover:** steady at about (0, 0, 2.5), with no growing wobble.
  2026-11-07: 2.51 m within 5 mm, body rates within 0.25°/s.
- **Thrust:** `actuator_motors` all about **0.139**, and `hover_thrust`
  about **0.139**. `actuator_motors` is the thrust command; PX4's thrust
  model (`THR_MDL_FAC` 0.9) turns 0.139 into motor output 0.341, which is
  5,407 RPM, the real drone's hover. Neither readout shows the 0.341.
- **Clock:** these stamps agree within a few tenths of a second:
  ```bash
  ros2 topic echo --once /clock
  ros2 topic echo --once /drone00/sensors/imu --field header.stamp
  ros2 topic echo --once /iris_0/front_cam/camera_info --field header.stamp
  ```
- **Camera:** `ros2 topic echo --once /iris_0/front_cam/camera_info` shows
  `width: 1024`, `height: 768`, and `k[0]` about 439.8.
- **View:** in the `/iris_0/front_cam` viewport, gate_0 is whole and
  centered, and gate_1 is whole on the right.

If it fails:
- **Oscillates and climbs away:** the rate gains weren't set, so SITL kept
  the Iris gains. They act 3.4× (roll) and 5× (pitch) harder on the
  Starling: it went into a ±270°/s pitch oscillation and climbed to 86 m
  (2026-11-07). Check `param show MC_PITCHRATE_P` reads 0.03.
- **Spins on the ground as soon as it arms:** the prop colliders are on.
  Check the four `rotorN: 1 prop collider(s) disabled` spawn lines.
- **Flips at liftoff, roll and pitch mixed up:** the rotors aren't where PX4
  mixes them. Check the four `rotorN anchor in body frame` spawn lines.
- **Thrust far from 0.139:** the mass or k_f didn't apply. Check the
  `[starling]` lines.
- **`hover_thrust` far from 0.14:** the `param set` lines weren't pasted
  (`THR_MDL_FAC` must be 0.9).

## Stage 2: dry run (nothing engaged)

```bash
export RUN=/workspace/vio_runs/s29_$(date +%H%M%S) && mkdir -p $RUN
```
Each in its own shell:
```bash
python3 ros2_nodes/vio_node.py --ros-args -p use_sim_time:=true -p camera_pitch_deg:=0.0 \
    -p "camera_pos:=[0.0711,-0.0003,-0.0070]" -p ground_truth_topic:=/drone00/state/pose -p log_dir:=$RUN/vio
python3 ros2_nodes/flight_monitor.py --ros-args -p use_sim_time:=true -p log_dir:=$RUN
python3 ros2_nodes/mpc_node.py --ros-args -p use_sim_time:=true -p state_source:=odometry -p odom_topic:=/vio/odometry
python3 ros2_nodes/min_snap_node.py --ros-args -p use_sim_time:=true -p state_source:=odometry \
    -p odom_topic:=/vio/odometry -p gates_file:=/workspace/test_gates.json -p yaw_mode:=next_gate \
    -p settle_s:=3.0 -p v_max:=2.0 -p a_max:=2.0 -p start_topic:=/mpc_offboard_bridge/engaged
```
Leave `manifest_camera_pitch_deg` at its default of 15: the course file's
camera pose was recorded with the Iris.

**GO:** `flight_monitor` prints `PREFLIGHT: GO`.
- `vio_node`: `|accel| at hover ~9.81` and a camera line for 1024×768.
  Its error against the truth should be about 0.1 m. It sees gate_0, and
  maybe gate_1.
- `mpc_node`: `holding [...]`, then solve stats. Compare them with Stage 0.
- `min_snap_node`: `plan ready, waiting for /mpc_offboard_bridge/engaged`.

Then stop `mpc_node` and `min_snap_node` (Ctrl-C). Leave `vio_node`
running: in Stages 3–4 it gives a free VIO preview.

## Stage 3: hover hold on ground truth (15 s)

```bash
python3 ros2_nodes/mpc_node.py --ros-args -p use_sim_time:=true -p state_source:=pegasus
python3 ros2_nodes/flight_monitor.py --ros-args -p use_sim_time:=true -p log_dir:=$RUN/s3 -p max_vio_err_m:=99.0
python3 ros2_nodes/mpc_offboard_bridge.py --max-duration 15 --max-tilt-deg 30 --log-dir $RUN/s3 --ros-args -p use_sim_time:=true
```
There is **no `--hover-thrust`**. With `DRONE_VEHICLE=starling2max`, the
bridge inverts the Starling's thrust chain itself: 0.139 at hover, PX4's
`THR_MDL_FAC` included. Restart `flight_monitor` (stop the Stage 2 one
first): the watchdog latches.

**GO:** run `python3 px4/analyze_vio_mpc_flight.py $RUN/s3`. Its `[bridge]`
line must show:
- `max horizontal` and `max vertical` both under 0.3 m;
- `completed: reached --max-duration`, with no abort;
- the drone neither sinks nor climbs.

If it fails:
- **Steady altitude error:** the hover thrust is off. Read the thrust in
  `bridge_log.csv` and give `--hover-thrust`, as was done for the Iris
  (0.58).
- **Flips, or drifts sideways at once:** stop. Send `bridge_log.csv`.
- **Wobbles:** check the rate gains are the Starling's (Stage 1); if they
  are, lower `MC_PITCHRATE_P` / `MC_ROLLRATE_P` by a quarter.

Between runs at `pxh>`, if it moved away from the hover point:
`commander mode loiter`, then let it settle.

## Stage 4: the course on ground truth (gate_0 → gate_1 → gate_2)

`mpc_node` (pegasus) keeps running. Use a new `flight_monitor`, add the
planner, then engage:
```bash
python3 ros2_nodes/flight_monitor.py --ros-args -p use_sim_time:=true -p log_dir:=$RUN/s4 -p max_vio_err_m:=99.0
python3 ros2_nodes/min_snap_node.py --ros-args -p use_sim_time:=true -p state_source:=pegasus \
    -p gates_file:=/workspace/test_gates.json -p yaw_mode:=next_gate -p settle_s:=3.0 \
    -p v_max:=2.0 -p a_max:=2.0 -p start_topic:=/mpc_offboard_bridge/engaged
python3 ros2_nodes/mpc_offboard_bridge.py --max-duration 60 --max-tilt-deg 30 --log-dir $RUN/s4 --ros-args -p use_sim_time:=true
```
Wait for `plan ready` and `PREFLIGHT: GO` before the bridge.

**Expect** (offline rehearsal, Starling + C29, 2/2):
- gate passes at about +3.9 s, +10.4 s and +16.9 s;
- frame margins about 0.7 m;
- about 17 s of sim time, roughly 40 s wall.

**GO:** `analyze_vio_mpc_flight.py $RUN/s4` reports `ALL GATES PASSED`, and
tracking max under 0.6 m. Also note `vio flying position error`, the free
preview. The offline rehearsal flew on the VIO with a max error of
0.21–0.40 m.

**Between attempts** at `pxh>`:
```
param set RTL_RETURN_ALT 5
param set RTL_DESCEND_ALT 5
commander mode auto:rtl
```
then `commander takeoff`. Gate tops are at 3.45 m, so 5 m clears them. Restart
`flight_monitor` (and `min_snap_node`) for each attempt.

## If there is time: Stage 4R (one closed lap, replanning from the camera)

As the test plan's Stage 4R, with these replacing `min_snap_node`:
```bash
python3 ros2_nodes/ros_camera_pose_fallback.py --camera-pitch-deg 0 --camera-pos 0.0711 -0.0003 -0.0070
python3 ros2_nodes/replan_node.py --ros-args -p use_sim_time:=true -p state_source:=pegasus -p gates_file:=/workspace/test_gates.json -p prior_sigma:=0.5 -p prior_seed:=0 -p laps:=1 -p close_loop:=true -p yaw_mode:=next_gate -p v_max:=2.0 -p a_max:=2.0 -p start_topic:=/mpc_offboard_bridge/engaged -p log_dir:=$RUN/s4r
```
Run the bridge with `--max-duration 120 --log-dir $RUN/s4r`. Score with
`analyze_vio_mpc_flight.py $RUN/s4r --route gate_0,gate_1,gate_2,gate_0`.

## After

- Stop `vio_node` with **Ctrl-C** (SIGINT) so it writes `vio_events.csv`.
- Copy `$RUN` back.
- CSVs stay out of git; figures and summaries can go in.

**Recovering mid-session:**
- Lost the `pxh>` prompt (a dropped terminal)? In `px4-sitl`,
  `cd /workspace/PX4-Autopilot && python3 Tools/mavlink_shell.py 0.0.0.0:14550`
  gives the same shell over MAVLink. Don't start a second PX4.
- Drone crashed or flipped? Disarm, run `workspace/reset_drone.py` in the
  Script Editor (back upright at the takeoff spot), then restart PX4.
