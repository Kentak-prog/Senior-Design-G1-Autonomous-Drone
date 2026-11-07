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

Done on 2026-10-29: `main`, casadi 3.6.7, scipy 1.10.1, matplotlib 3.7.5,
numpy 1.26.4 with one BLAS thread, and `drone_msgs` rebuilt. All tests pass
in the pod. A container restart loses the `--user` installs, so redo this
after one.

```bash
cd /workspace/Senior-Design-G1-Autonomous-Drone && git fetch && git checkout main && git pull
pip install --user "casadi==3.6.7" "scipy==1.10.1" "matplotlib==3.7.5" pymavlink
python3 -c "import numpy; print(numpy.__version__)"       # 1.x only (NumPy 2 breaks cv_bridge)
# matplotlib's contourpy pulls numpy 1.26.4 into ~/.local. That works (cv_bridge + cv2 checked
# 2026-10-29), BUT pip's numpy bundles OpenBLAS, which starts a thread per core: test_vio went
# from seconds to over 11 minutes. Use one BLAS thread, in every shell:
echo 'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1' >> ~/.bashrc && source ~/.bashrc
# an old hand-made /workspace/ros_ws/drone_msgs (2026-10-22) duplicates the package: move it out
[ -d /workspace/ros_ws/drone_msgs ] && mv /workspace/ros_ws/drone_msgs /workspace/drone_msgs_old_20261022
mkdir -p /workspace/ros_ws/src && ln -sfn /workspace/Senior-Design-G1-Autonomous-Drone/drone_msgs /workspace/ros_ws/src/drone_msgs
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
python3 ros2_nodes/mpc_offboard_bridge.py --hover-thrust 0.58 --max-duration 15 --max-tilt-deg 30 --log-dir $RUN/s3 \
    --ros-args -p use_sim_time:=true
```
`max_vio_err_m:=99` keeps a misbehaving VIO from aborting a flight that doesn't use it.

**GO:** after the hand back, `python3 px4/analyze_vio_mpc_flight.py $RUN/s3`
prints a `[bridge]` line (from `bridge_log.csv`) with these numbers:
- `max horizontal` and `max vertical` both under 0.3 m;
- `completed: reached --max-duration`, with no abort;
- the drone neither sinks nor climbs.

`--hover-thrust 0.58` is measured (see below). If the drone still sinks or
climbs, adjust it. If it flips or drifts sideways at once, the attitude frame mapping
is wrong (`px4/attitude_bridge.py`): stop and debug there, with the log.

### Stage 3 results (2026-10-29)

Three runs, three findings, each fixed before the next:

1. **Bridge aborted before OFFBOARD** ("no new MPC setpoint for more than
   0.3 s"). It handled ROS only inside its loop, so setpoints went stale
   while its `[y/N]` prompt blocked. Fixed: ROS callbacks run on a background
   thread (`ros2_nodes/mpc_offboard_bridge.py`).
2. **0.12 m sag at engage.** To hold altitude the MPC needed 15.87 N
   against m·g = 14.7 N: the Pegasus Iris thrust curve says hover is at
   0.556 throttle, but PX4 really hovers at about 0.58. Fixed:
   `--hover-thrust 0.58`. The second run held 2.565 m against a 2.57 m
   hold point, at 14.75 N.
3. **x offset never corrected.** The drone sat 0.30 m from its hold point
   for 36 s, while the MPC itself reported the error.
   - Cause: the NMPC's controls are thrust + torques. `mpc_node` sent PX4
     the attitude predicted one step (0.02 s) ahead, which is the current
     attitude, so PX4 just held it.
   - Fixed: the attitude setpoint is now the MPC's prediction 0.1 s ahead
     (`attitude_lead_s`, `control/mpc_core.py` "ATTITUDE LEAD").
   - Checked offline on a PX4-like plant (`control/px4_attitude_plant.py`,
     `control/test_mpc_core.py`): a 0.3 m offset drops below 0.05 m within
     3 s; with the old one-step lead it is still 0.7–1.2 m off after 10 s.
   - Cost: PX4's attitude lag isn't in the MPC's model, so course tracking
     at 2/2 is about 0.3 m mean / 0.55 m max (0.03 m with torques fed
     straight in). All gates still pass.
   - 0.1 s is the best lead, or within 0.05 m of it, for a PX4 lag of
     0.05–0.12 s.

**Rerun Stage 3 after pulling these fixes and restarting `mpc_node`.**

## Stage 4: the course on ground truth

As stage 3, plus the plan from the true state:
```bash
python3 ros2_nodes/min_snap_node.py --ros-args -p use_sim_time:=true -p state_source:=pegasus \
    -p gates_file:=/workspace/test_gates.json -p yaw_mode:=next_gate -p settle_s:=3.0 \
    -p v_max:=2.0 -p a_max:=2.0 -p start_topic:=/mpc_offboard_bridge/engaged
python3 ros2_nodes/mpc_offboard_bridge.py --hover-thrust 0.58 --max-duration 60 --max-tilt-deg 30 --log-dir $RUN/s4 \
    --ros-args -p use_sim_time:=true
```
Use a new `flight_monitor` with `log_dir:=$RUN/s4` and `max_vio_err_m:=99.0`.

**GO:** `analyze_vio_mpc_flight.py $RUN/s4` reports `ALL GATES PASSED` and
tracking max under 0.6 m. The rehearsal had 0.29 mean and 0.66 max *with*
VIO, so perfect state should do better. Also compare:
- the solve times with stage 0;
- `vio flying position error`: the VIO ran alongside, so this is a free
  stage-5 preview.

## Stage 4R: the course on ground truth, replanning from the camera (one closed lap)

As Stage 4, but `ros2_nodes/replan_node.py` replaces `min_snap_node`. It starts
from a gate map **off by about 0.5 m** (the manifest perturbed with a fixed
seed). The camera's gate poses correct it, and it **replans in flight** when a
gate it is heading for moves. The route is **one lap, and only one**: gate_0,
gate_1, gate_2, then back through gate_0 to close it (`laps:=1`,
`close_loop:=true`, which are also `replan_node`'s defaults). Do not raise
`laps` for this stage: a second lap only adds flight time and the 2-lap plan
can fail to converge (`px4/NEXT_TESTS.md` section 2). It still flies on ground
truth; the camera poses are lifted into the world with the true pose.

Same Isaac and PX4 bring-up. In the `ros2-eval` shells (step 3 of the
runbook in each, with `DRONE_VEHICLE=iris`):

```bash
export RUN=/workspace/vio_runs/s4r_$(date +%H%M%S) && mkdir -p $RUN
python3 ros2_nodes/flight_monitor.py --ros-args -p use_sim_time:=true -p log_dir:=$RUN -p max_vio_err_m:=99.0
python3 ros2_nodes/mpc_node.py --ros-args -p use_sim_time:=true -p state_source:=pegasus
python3 ros2_nodes/ros_camera_pose_fallback.py --camera-pitch-deg 0
python3 ros2_nodes/replan_node.py --ros-args -p use_sim_time:=true -p state_source:=pegasus     -p gates_file:=/workspace/test_gates.json -p prior_sigma:=0.5 -p prior_seed:=0 -p laps:=1     -p close_loop:=true -p v_max:=2.0 -p a_max:=2.0 -p start_topic:=/mpc_offboard_bridge/engaged -p log_dir:=$RUN
```

`ros_camera_pose_fallback.py` takes plain flags, no `--ros-args`, and needs
`--camera-pitch-deg 0` (the spawn's level camera). With the default 15° every
gate lands about 2.6 m low at 10 m range.

Expect:
- `replan_node` logs `prior gate_N: ... (x m from the manifest)` and then
  `plan ready (first plan) ... through gate_0, gate_1, gate_2, gate_0`. While
  hovering, it re-plans whenever the camera moves a gate.
- The monitor reaches `PREFLIGHT: GO`.

Then:

```bash
python3 ros2_nodes/mpc_offboard_bridge.py --hover-thrust 0.58 --max-duration 120 --max-tilt-deg 30 --log-dir $RUN --ros-args -p use_sim_time:=true
```

Offline (`Trajectory/test_replan_session.py`, same course and prior seed): 3 to 4
replans, gate_0 to gate_0 in about 19.7 s, so the flight is about 27 s of sim
time, roughly 65 s wall. `--max-duration 120` leaves room.

During the flight `replan_node` logs every `passed gate_N` and every
`published plan #k (reason)`. Score it with the route order:

```bash
python3 px4/analyze_vio_mpc_flight.py $RUN --route gate_0,gate_1,gate_2,gate_0
```

**Pass:** `[route] ... OK` (gate_0 twice) and the watchdog never trips. Also
keep `replan_events.csv` and `gate_map.csv` from `$RUN`: they show each plan,
its reason, and how far the camera moved each gate.

## The circle course (from 2026-11-06): Stage 4R, then Stage 7

**Why.** On the 2026-10-23 course no gate fits wholly in the 60° × 36°
camera from +5.5 to +10 s and again on the way back to gate_0, so the color
detector + PnP gives no fixes for up to 4.7 s and the VIO drifts
(`px4/NEXT_TESTS.md`). The circle course puts the 3 gates 120° apart on a
5 m circle. Each gate faces along the flight direction and sits at camera
height. gate_0 is 5 m straight ahead of the takeoff hover. With
`yaw_mode next_gate`, the next gate is whole in frame from each pass until
about 3 m before the next gate.

The last ~2 s before each gate stays blind on ANY layout. Closer than about
3 m, a 1.7 m gate does not fit in the 36° vertical view.

| Offline, 2/2, level camera | 2026-10-23 course | circle |
|---|---|---|
| Hover: gates whole in view | gate_1, gate_2 (gate_0 clipped) | gate_0 |
| Longest gap with no fix, engage → last pass | 4.7 s (three gaps over 4 s) | 3.5 s from hover, then 2.1–2.5 s before each gate |
| Biased fixes (≥ 0.3 m off) | 6 | 0 |
| VIO rehearsal max error, 3 seeds | 0.46–0.87 m | 0.16–0.33 m |
| Gate frame margins, 3 seeds | 0.44–0.70 m | 0.66–0.75 m |
| Lap time (closed lap) | 28 s | 28 s exact map, 36 s with 4R's prior |

Sources:
- the gap and fix rows: `Trajectory/check_course_camera.py`, which renders
  the camera along a simulated closed lap and runs the real color detector
  + PnP;
- the VIO rehearsal rows: `Trajectory/simulate_vio_mpc.py`, an open run
  gate_0 → gate_2, seeds 0–2.

Course file: `Trajectory/isaac_results/circle_course/circle_course.json`.
It is written by `Trajectory/make_circle_course.py`:

| Gate | x | y | z | yaw |
|---|---|---|---|---|
| gate_0 | 4.911 | −0.667 | 2.6 | −7.2° |
| gate_1 | 8.267 | −8.650 | 2.6 | −127.2° |
| gate_2 | −0.325 | −7.565 | 2.6 | +112.8° |

It keeps the old `camera_pose_usd` (same takeoff spot), so `vio_node`'s
spawn hint still holds.

**Spawn it** in Stage 1 instead of the default course. The spawner copies it
to `/workspace/test_gates.json`, which every node and
`analyze_vio_mpc_flight.py` read, so no other command changes:
```python
import os; os.environ["GATES_MANIFEST"] = "/workspace/Senior-Design-G1-Autonomous-Drone/Trajectory/isaac_results/circle_course/circle_course.json"; exec(open("/workspace/Senior-Design-G1-Autonomous-Drone/workspace/spawn_gates_from_manifest.py").read())
```
Expect gate_0 at (4.911, −0.667, 2.600) yaw −7.20, gate_1 at (8.267,
−8.650) yaw −127.20, gate_2 at (−0.325, −7.565) yaw +112.80. In the
`/iris_0/front_cam` view at the hover, **gate_0 is whole in the frame**, and
the other two are off to the right.

**Then Stage 4R** exactly as written above, with `prior_seed:=0`. It passed
offline on this course (seed 0: 4 passes, 0 biased fixes). Add
`-p yaw_mode:=next_gate` to `replan_node`, so 4R flies the same nose
direction Stage 7 will.

Other seeds work too since 2026-11-06. Before that fix, the gate map
matched a sighting to a map gate only within `max(1 m, 10 % of range)`, but
`prior_sigma 0.5` is per axis, so some seeds put a gate more than 1 m off.
With seed 1 (gate_1 1.08 m off), the camera's gate_1 became a 4th gate. The
replanner's min-snap fit then ran away (a 7×10¹² s plan, MemoryError), and
the lap never closed. Now:
- `Trajectory/gate_map.py` also matches within the gate's 99.7 %
  uncertainty;
- `Trajectory/min_snap.fit_to_limits` gives up (keeping the current plan)
  instead of stretching forever.

Offline, seeds 0–3 all close the lap with 0 biased fixes. With the prior
off, the last approach to gate_0 has a longer blind stretch (3–5.7 s
instead of 2.1 s). That is harmless here, since 4R flies on ground truth.

**Then Stage 7** with the run set of `px4/NEXT_TESTS.md` section 3:
- `vio_node` + `flight_monitor` first;
- then `mpc_node` (odometry) and `replan_node` (odometry, `prior_sigma 0`,
  `use_camera false`, `yaw_mode next_gate`, `laps 1`, `close_loop true`);
- then the bridge.

Check `python3 Trajectory/check_course_camera.py` first if anything about
the course changes.

## The Starling 2 Max (C29) on the circle course

The same stages, on our real airframe and its front camera instead of the
Pegasus Iris.

**Camera.** The C29 front hires camera is the IMX412 (ModalAI M0161) with a
2.7 mm lens.
- **Mount**, from the drone's `/etc/modalai/extrinsics.conf` (copy:
  `workspace/starling2max_c29_extrinsics.conf`): hires_front is **level**,
  at (0.0711, −0.0003, −0.0070) m in body FLU.
- **Field of view**: the datasheet row is mislabeled. Read as H × V × D it
  is 120.4° × 93.5° × 146°, a distorted wide lens. Isaac renders an ideal
  pinhole, so we use the undistorted image with the lens's central focal
  length: **fx = fy = 439.8 px at 1024 × 768, 98.7° × 82.3°**.
- The sensor itself is confirmed by Sony's IMX412-AACK flyer: 4056 × 3040
  active pixels, 1.55 µm square pixels.
- 1024 × 768 (the 4:3 full-sensor mode) is assumed. Check the drone's
  `voxl-camera-server.conf` and its calibration (`/data/modalai/`).
- The sensor's 16:9 modes crop rows (4K2K 4056 × 2288; 1080p binned from
  4056 × 2224): same 98.7° across, about 66° vertical. For those, spawn with
  `os.environ["CAMERA_RESOLUTION"] = "1024x578"`. Offline on the circle
  course that gives a longest gap of 2.0 s and 82 % of frames with a fix,
  against 1.2 s and 89 % at 4:3.

These are `spawn_starling.py`'s defaults.

**Offline** (`DRONE_VEHICLE=starling2max`, this camera, 2/2, closed lap,
`check_course_camera.py` and `simulate_vio_mpc.py`):

| | Iris camera, circle | **C29 camera, circle** | C29 camera, 2026-10-23 course |
|---|---|---|---|
| Gates whole in view at the hover | gate_0 | gate_0, gate_1 | all three |
| Longest gap with no fix, engage → last pass | 3.5 s | **1.2 s** (0.8–1.2 s before each gate) | 2.9 s |
| Frames with a good fix | 59 % | **89 %** | 65 % |
| Fixes ≥ 0.3 m off | 0 | 6 (gate_1 at 11–12 m) | 13 |
| VIO rehearsal max error | 0.16–0.33 m (Iris) | **0.21–0.40 m**, 3 seeds, all gates | 0.47–0.69 m (2 seeds) |
| Stage 4R, map 0.5 m off, seed 0 | passes | passes, longest gap 1.3 s | |

About those columns:
- The 82° vertical view keeps a gate whole until about 1 m away, not 3 m.
- The 6 off fixes are far-range noise: a 1.7 m gate at 12 m is about 60 px
  wide at 1024 px. The VIO weights them by range² and still passed every
  seed. Capping `max_range` at 10 m did not help in the rehearsal. If the
  bag (V1) shows far fixes biased, cap it there, or stream wider.

**What changes in the runbook:**

1. **Every `ros2-eval` shell:** `export DRONE_VEHICLE=starling2max`
   (instead of `iris`).
2. **Stage 1, Script Editor:** run `spawn_starling.py` instead of
   `spawn_example.py`. It sets the C29 camera itself, with no
   `CAMERA_PITCH_DEG`:
   ```python
   exec(open("/workspace/Senior-Design-G1-Autonomous-Drone/workspace/spawn_starling.py").read())
   ```
   Then `sim_clock.py`, then the circle course (`GATES_MANIFEST` line in
   "The circle course").
3. **Check the spawn printed:**
   - `[starling] camera 1024x768, hFOV 98.7 deg, at (0.0711, -0.0003, -0.007) m (body FLU), pitch 0.0 deg up`
   - `[init] camera ... expected fx=439.8`
   - `hover 5407 RPM = motor output 0.341 = hover thrust 0.139`, then a
     block of `param set` lines.
4. **Before `commander takeoff`, in `pxh>`**, paste the `param set` lines the
   spawn printed (or run `sh px4/starling2max_sitl_params.sh` in the PX4
   container; same set). They cover:
   - PX4's thrust model, as on the drone: `THR_MDL_FAC` 0.9,
     `MPC_THR_HOVER` 0.14, `MPC_THR_MAX` 0.6, `MPC_THR_MIN` 0.04;
   - `MPC_USE_HTE` 1;
   - the Max's rotor geometry and yaw ratio (`CA_ROTOR0..3_PX/PY/KM`);
     without them SITL allocates torque for the Iris;
   - two sim-only settings (`COM_LOW_BAT_ACT` 0, `MC_AIRMODE` 1).

   SITL keeps the Iris rate gains. If the hover wobbles, halve
   `MC_ROLLRATE_P` / `MC_PITCHRATE_P` and their `_D`.
5. **GO check:** `ros2 topic echo --once /iris_0/front_cam/camera_info`
   shows `width: 1024`, `height: 768`, and `k[0]` about 439.8. (The ROS
   names keep `iris_0` on purpose; see `spawn_starling.py`.)
6. **Camera mount on every node that uses the camera:**
   - `vio_node`:
     `-p camera_pitch_deg:=0.0 -p "camera_pos:=[0.0711,-0.0003,-0.0070]"`.
     Leave `manifest_camera_pitch_deg` at 15: the manifest's camera pose was
     recorded with the Iris.
   - `ros_camera_pose_fallback.py` (4R):
     `--camera-pitch-deg 0 --camera-pos 0.0711 -0.0003 -0.0070`.
7. **Bridge:** leave out `--hover-thrust`. With
   `DRONE_VEHICLE=starling2max` its default inverts the Starling's whole
   thrust chain, PX4's `THR_MDL_FAC` included (`px4/attitude_bridge.py`,
   since 2026-11-06): about 0.139 at hover, PX4's `MPC_THR_HOVER`. Before
   that change, the default sent the motor output (0.341) instead, which
   would have climbed hard. The Iris needed its measured 0.58, so prove the
   hover first.

**Order** (Stages 0–4 as a copy-paste checklist: `px4/STARLING_QUICK_RUN.md`):
- **Stage 3, hover hold on ground truth.** A steady altitude error means
  the hover thrust is off: set `--hover-thrust` from the bridge log, as was
  done for the Iris.
- **Stage 4**, ground truth, gate_0 → gate_1 → gate_2 (`min_snap_node`).
- **Stage 4R**, ground truth, one closed lap.
- **Stage 6**, hover on the VIO.
- **Stage 7**, closed lap on the VIO.

No Isaac run of these stages with the Starling is logged yet, so do not skip Stage 3.

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
python3 ros2_nodes/mpc_offboard_bridge.py --hover-thrust 0.58 --max-duration 15 --max-tilt-deg 30 --log-dir $RUN/s6 \
    --ros-args -p use_sim_time:=true
```
**GO:** same as stage 3. A slow drift here, which stage 3 didn't have, is VIO
drift at hover.

## Stage 7: the course on the VIO (the goal)

Stage 2's full set, fresh `flight_monitor` (`log_dir:=$RUN/s7`), then start
recording (`Trajectory/isaac_results/3_gates/scripts/record_novnc.js`:
`startRec()`) and engage:
```bash
python3 ros2_nodes/mpc_offboard_bridge.py --hover-thrust 0.58 --max-duration 60 --max-tilt-deg 30 --log-dir $RUN/s7 \
    --ros-args -p use_sim_time:=true
```
At engage the bridge logs `OFFBOARD confirmed -> True on
/mpc_offboard_bridge/engaged`, then `min_snap_node` logs `published plan`.
The plan is 16.0 s of sim time, about 38 s wall at 0.42×. Afterwards:
`analyze_vio_mpc_flight.py $RUN/s7`.

**Pass:** `ALL GATES PASSED`, the watchdog never trips. The rehearsal's
numbers to compare against (`Trajectory/simulate_vio_mpc.py
--camera-pitch-deg 0 --v-max 2 --a-max 2`, through the PX4-like plant with
a 0.08 s lag and a 0.1 s lead, 2026-10-29):
- VIO error mean 0.24 m, max 0.98 m;
- tracking mean 0.52 m, max 0.88 m;
- frame margins 0.44–0.65 m.

With torques fed straight to the plant (`--plant torque`, the rehearsal
before 2026-10-29), tracking was 0.29 m mean and 0.66 m max, with margins of
0.60–0.68 m. That plant does not show PX4's lag.

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
- `min_snap_node` plans once. In-flight replanning is `ros2_nodes/replan_node.py`
  (Stage 4R), flown for one closed lap only.
- One course, one seed per speed, so a pass is evidence, not proof. Repeat
  stage 7 at least three times before stepping up.
