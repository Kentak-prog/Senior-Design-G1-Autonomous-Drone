# Isaac Sim test plan: VIO + MPC on the 3-gate course

**Goal.** The NMPC flies the 3-gate course of 2026-10-23 in Isaac Sim, with
its state coming **only** from the VIO (IMU + front camera on the known gate
map). PX4 is used for its attitude loop only.

**Where we are.** Every piece is tested without ROS, including closed-loop
rehearsals (`Trajectory/simulate_vio_mpc.py`: all gates at 2/2 to 4/4 m/s,
m/s²). **No node of this stack has run on ROS yet**, and the attitude
bridge has never driven the real PX4. So the plan adds one unknown at a time.
Every stage has a GO criterion. Don't skip ahead when a stage fails: its log
tells you what broke.

| Stage | What's new | State the MPC uses | Engaged? | GO to the next stage when |
|---|---|---|---|---|
| 0. Pod prep | deps, `drone_msgs`, checkout | — | no | the pod tests pass, MPC solve < 15 ms |
| 1. Isaac + PX4 | level camera, `/clock`, gates | — | no | the clock check agrees, hovering at ~(0, 0, 2.5) |
| 2. Dry run | every node on ROS | VIO | **no** | `flight_monitor` prints **PREFLIGHT: GO** |
| 3. Hover hold, truth | bridge → PX4 attitude | ground truth | 15 s | holds within 0.3 m, no abort |
| 4. Course, truth | MPC + min-snap through PX4 | ground truth | course | all gates, tracking max < 0.6 m |
| 5. VIO open-loop | VIO on the real flight | — (PX4 position flier) | no MPC | VIO max error < 0.6 m, every gate seen |
| 6. Hover hold, VIO | VIO in the loop | VIO | 15 s | holds within 0.3 m, no abort |
| **7. Course, VIO** | **the goal** | VIO | course | **all gates**, watchdog never trips |
| 8. Faster | 3/3, then 4/4 | VIO | course | as 7 |

Stages 3 and 4 separate "the bridge / PX4 attitude mapping is wrong" from
"the VIO is wrong". Stage 5 measures the VIO on the real images and IMU
while the proven position flier (`fly_min_snap_offboard.py`, the flight
of 2026-10-23) flies.

---

## Safety net (every engaged stage)

Each layer hands PX4 to **AUTO.LOITER**, which flies on PX4's own GPS
estimate and is independent of the VIO:

| Layer | Trips on | Where |
|---|---|---|
| `flight_monitor` watchdog (sim only) | VIO > 1.0 m from the truth for 0.2 s; the true drone > 2.0 m from the reference for 0.2 s; true altitude < 0.5 m; no VIO for 0.5 s | `px4/flight_checks.Watchdog` → `/flight_monitor/abort` |
| bridge | no setpoint for 0.3 s; solver invalid for 0.5 s; setpoint tilt > `--max-tilt-deg` (use **30** at 2/2: the plan needs about 12°) | `px4/attitude_bridge.py` |
| operator | anything that looks wrong | Ctrl-C in the bridge, or `commander mode loiter` at `pxh>` |

The watchdog arms only after the first **GO**. Once it trips, it stays
tripped: restart `flight_monitor` before the next run. The bridge refuses to
start if the monitor has already tripped. Land with `commander land`.

**The plan starts on engage.** `min_snap_node` with
`start_topic:=/mpc_offboard_bridge/engaged` computes the plan as soon as the
VIO has settled, but publishes it (t = 0) only when the bridge confirms
OFFBOARD. Without this, the reference would run on while PX4 still hovers by
itself, and at engage the MPC would chase a point seconds down the course.
Until the plan arrives, `mpc_node` holds the hover point.

---

## Stage 0: pod prep (once per container)

`ros2-eval` restarted 14 h ago. As of 2026-10-28 it has **no casadi and no
scipy**, an **old `drone_msgs` build** (no `AttitudeThrustSetpoint`), and the
repo on `px4-offboard-min-snap` at `7146eb0`.

```bash
cd /workspace/Senior-Design-G1-Autonomous-Drone && git fetch && git checkout main && git pull
pip install --user "casadi==3.6.7" "scipy==1.10.1" pymavlink
python3 -c "import numpy; print(numpy.__version__)"       # must still be 1.21.5 (NumPy 2 breaks cv_bridge)
cd /workspace/ros_ws && rm -rf build/drone_msgs install/drone_msgs \
  && colcon build --packages-select drone_msgs && source install/setup.bash
ros2 interface show drone_msgs/msg/AttitudeThrustSetpoint  # must print the message
cd /workspace/Senior-Design-G1-Autonomous-Drone
for t in control/test_mpc_core.py control/test_vio.py Trajectory/test_min_snap_planner.py \
         px4/test_attitude_bridge.py px4/test_flight_checks.py px4/test_analyze_vio_mpc_flight.py; do
  python3 $t | tail -1; done
```

**GO:** every test prints `All tests passed`, and `test_mpc_core` reports a
median solve under 15 ms. The pod CPU is shared with Isaac. At 50 Hz sim
time and a ~0.42× sim, the MPC has about 48 ms of wall time per step.

Each shell that runs a node needs:
`source /opt/ros/humble/setup.bash && source /workspace/ros_ws/install/setup.bash && cd /workspace/Senior-Design-G1-Autonomous-Drone`.

## Stage 1: Isaac + PX4

Fresh Isaac process with the Flat Grid. In the Script Editor, once each:

```python
import os; os.environ["CAMERA_PITCH_DEG"] = "0"; exec(open("/workspace/Senior-Design-G1-Autonomous-Drone/workspace/spawn_example.py").read())
```
```python
exec(open("/workspace/Senior-Design-G1-Autonomous-Drone/workspace/sim_clock.py").read())
```
```python
exec(open("/workspace/Senior-Design-G1-Autonomous-Drone/workspace/spawn_gates_from_manifest.py").read())
```

Expect `[init] camera pitch 0.0 deg up`, `[sim_clock] ... now uses sim time`,
and three `gate_N` lines matching the course. Start PX4 (`vision/HANDOFF.md`),
wait for `Ready for takeoff!`, then `commander takeoff`.

**GO:** these three stamps agree within a few tenths of a second (all sim time):
```bash
ros2 topic echo --once /clock
ros2 topic echo --once /drone00/sensors/imu --field header.stamp
ros2 topic echo --once /iris_0/front_cam/camera_info --field header.stamp
```

## Stage 2: dry run (nothing engaged)

One run directory for all the logs:
```bash
export RUN=/workspace/vio_runs/run_$(date +%H%M%S) && mkdir -p $RUN
```

Then, each in its own shell (or `&` with `| tee $RUN/<name>.txt`):
```bash
python3 ros2_nodes/vio_node.py --ros-args -p use_sim_time:=true -p camera_pitch_deg:=0.0 \
    -p ground_truth_topic:=/drone00/state/pose -p log_dir:=$RUN/vio
python3 ros2_nodes/flight_monitor.py --ros-args -p use_sim_time:=true -p log_dir:=$RUN
python3 ros2_nodes/mpc_node.py --ros-args -p use_sim_time:=true -p state_source:=odometry -p odom_topic:=/vio/odometry
python3 ros2_nodes/min_snap_node.py --ros-args -p use_sim_time:=true -p state_source:=odometry \
    -p odom_topic:=/vio/odometry -p gates_file:=/workspace/test_gates.json -p yaw_mode:=next_gate \
    -p settle_s:=3.0 -p v_max:=2.0 -p a_max:=2.0 -p start_topic:=/mpc_offboard_bridge/engaged
```

Expect:
- `vio_node`: `|accel| at hover ~9.81`, and an error vs ground truth under about 0.1 m.
- `mpc_node`: `no trajectory yet: holding [...]`, then solve stats every 2 s.
- `min_snap_node`: `plan ready, waiting for /mpc_offboard_bridge/engaged`.
  This takes several seconds: the time optimization.
- `flight_monitor`: a preflight table every 5 s.

**GO:** `PREFLIGHT: GO`. Each `NO` line names what to fix. The usual suspects:
- the IMU stamp lag is seconds off: `sim_clock.py` was not run;
- the VIO error is over 0.3 m;
- the camera rate is 0: wrong topic.

Leave the nodes running for stage 6 and stage 7. Stages 3–4 need different
state sources, so stop `mpc_node` and `min_snap_node` (Ctrl-C) before them.

## Stage 3: hover hold on ground truth

First engage ever of `mpc_offboard_bridge` on the real PX4. The MPC holds the
hover point and nothing else. Don't run `min_snap_node`, so there's no plan.
```bash
python3 ros2_nodes/mpc_node.py --ros-args -p use_sim_time:=true -p state_source:=pegasus
python3 ros2_nodes/flight_monitor.py --ros-args -p use_sim_time:=true -p log_dir:=$RUN/s3 \
    -p max_vio_err_m:=99.0 -p plan_on_engage:=true
python3 ros2_nodes/mpc_offboard_bridge.py --max-duration 15 --max-tilt-deg 30 --log-dir $RUN/s3 \
    --ros-args -p use_sim_time:=true
```
`max_vio_err_m:=99` keeps a misbehaving VIO from aborting a flight that doesn't use it.

**GO:** after the hand back, `python3 px4/analyze_vio_mpc_flight.py $RUN/s3`
prints a `[bridge]` line (from `bridge_log.csv`) with these numbers:
- `max horizontal` and `max vertical` both under 0.3 m;
- `completed: reached --max-duration`, with no abort;
- the drone neither sinks nor climbs.

If it does sink or climb, rerun with `--hover-thrust` set to PX4's hover
throttle. If it flips or drifts sideways at once, the attitude frame mapping
is wrong (`px4/attitude_bridge.py`): stop and debug there, with the log.

## Stage 4: the course on ground truth

As stage 3, plus the plan from the true state:
```bash
python3 ros2_nodes/min_snap_node.py --ros-args -p use_sim_time:=true -p state_source:=pegasus \
    -p gates_file:=/workspace/test_gates.json -p yaw_mode:=next_gate -p settle_s:=3.0 \
    -p v_max:=2.0 -p a_max:=2.0 -p start_topic:=/mpc_offboard_bridge/engaged
python3 ros2_nodes/mpc_offboard_bridge.py --max-duration 60 --max-tilt-deg 30 --log-dir $RUN/s4 \
    --ros-args -p use_sim_time:=true
```
Use a new `flight_monitor` with `log_dir:=$RUN/s4` and `max_vio_err_m:=99.0`.

**GO:** `analyze_vio_mpc_flight.py $RUN/s4` reports `ALL GATES PASSED` and
tracking max under 0.6 m. The rehearsal had 0.29 mean and 0.66 max *with*
VIO, so perfect state should do better. Also compare:
- the solve times with stage 0;
- `vio flying position error`: the VIO ran alongside, so this is a free
  stage-5 preview.

## Stage 5: VIO open-loop on the proven flier

The flight of 2026-10-23, now with look-at yaw, and the VIO watching:
```bash
python3 ros2_nodes/vio_node.py --ros-args -p use_sim_time:=true -p camera_pitch_deg:=0.0 \
    -p ground_truth_topic:=/drone00/state/pose -p log_dir:=$RUN/s5/vio
python3 ros2_nodes/flight_monitor.py --ros-args -p use_sim_time:=true -p log_dir:=$RUN/s5 -p watchdog:=false
python3 px4/fly_min_snap_offboard.py --yaw-mode next_gate --reposition --time-scale <from the last run> \
    --out-dir $RUN/s5/px4
```
The monitor's preflight shows NO on the MPC lines here: there is no MPC, so
ignore them.

**GO:** `analyze_vio_mpc_flight.py $RUN/s5` gives a VIO position error
(`whole log`) max under 0.6 m and mean under 0.3 m. Also check
`$RUN/s5/vio/vio_events.csv`: it should show full fixes on every gate and
few rejections. If the error grows between gates, the IMU propagation is the
problem (bias, frame). If it jumps at fixes, the camera mount or the latency
is.

## Stage 6: hover hold on the VIO

Stage 2's nodes, with `min_snap_node` stopped (no plan) and a fresh
`flight_monitor` (`log_dir:=$RUN/s6`, default watchdog):
```bash
python3 ros2_nodes/mpc_offboard_bridge.py --max-duration 15 --max-tilt-deg 30 --log-dir $RUN/s6 \
    --ros-args -p use_sim_time:=true
```
**GO:** same as stage 3. A slow drift here, which stage 3 didn't have, is VIO
drift at hover.

## Stage 7: the course on the VIO (the goal)

Stage 2's full set, fresh `flight_monitor` (`log_dir:=$RUN/s7`), then start
recording (`Trajectory/isaac_results/3_gates/scripts/record_novnc.js`:
`startRec()`) and engage:
```bash
python3 ros2_nodes/mpc_offboard_bridge.py --max-duration 60 --max-tilt-deg 30 --log-dir $RUN/s7 \
    --ros-args -p use_sim_time:=true
```
At engage the bridge logs `OFFBOARD confirmed -> True on
/mpc_offboard_bridge/engaged`, then `min_snap_node` logs `published plan`.
The plan is 16.0 s of sim time, about 38 s wall at 0.42×. Afterwards:
`analyze_vio_mpc_flight.py $RUN/s7`.

**Pass:** `ALL GATES PASSED`, the watchdog never trips. The rehearsal's
numbers to compare against:
- VIO error mean 0.21 m, max 0.61 m;
- tracking mean 0.29 m, max 0.66 m;
- frame margins 0.60–0.68 m.

## Stage 8: faster

Repeat stage 7 with `v_max/a_max` 3/3, then 4/4. Raise `--max-tilt-deg` to
35 and 40. The rehearsal predicts 11.7 s and 10.1 s. 5/6 was marginal
(VIO max 2.9 m), and the 80 % default (9.6/12) missed gate_2 because the
camera tilts to the ground.

---

## What every run leaves (`$RUN/<stage>/`)

| File | From | Content |
|---|---|---|
| `flight_log.csv` | `flight_monitor` | 50 Hz: truth, VIO (+ std, age), reference, setpoint, solve time, valid, watchdog |
| `monitor_events.txt` | `flight_monitor` | the preflight tables, arming, trajectory start/end, the trip reason |
| `flight_summary.txt`, `flight_report.png` | `analyze_vio_mpc_flight.py` | the score: VIO error, tracking, solves, gate passes with margins |
| `bridge_log.csv` | `mpc_offboard_bridge --log-dir` | what PX4 was sent, and PX4's own position, every bridge step; final status line |
| `vio/vio_log.csv`, `vio/vio_events.csv` | `vio_node` | the estimate next to the truth every IMU step; every fix and rejection |
| `<node>.txt` | `tee` | console output |

Copy a run back (from the laptop, with the pod name from `kubectl get pods -n fake-isaac`):
```bash
kubectl cp -n fake-isaac -c ros2-eval <pod>:/workspace/vio_runs/run_<time> Trajectory/isaac_results/3_gates/vio_mpc/run_<time>
```
Commit only the summaries and plots. Leave the large CSVs and the recordings
out of git (link them in the doc), as with the 2026-10-23 flight video.

## Known limits of this test

- The watchdog uses the simulator's truth. On hardware, the only checks left
  are the bridge's checks and PX4's own failsafes.
- No in-flight replanning. `min_snap_node` plans once.
  `Trajectory/replanner.py` (laps, rolling window) is not wired into ROS yet.
- One course, one seed per speed, so a pass is evidence, not proof. Repeat
  stage 7 at least three times before stepping up.
