# ros2_nodes/: every ROS 2 node of the flight stack

Each file here is a **thin ROS 2 wrapper**: subscriptions, publishers,
parameters, logging. The logic each one runs lives with its subsystem and
imports **without ROS**, so the Python sims and the tests use exactly the same
code:

| Node | What it does | ROS-free logic (tested there) |
|---|---|---|
| `vio_node.py` | IMU + front camera → pose estimate (VIO) on the known gate map | `control/vio.py` (`control/test_vio.py`) |
| `min_snap_node.py` | gates (detections, or the course map) + current pose → min-snap reference | `Trajectory/min_snap_planner.py` (`Trajectory/test_min_snap_planner.py`) |
| `mpc_node.py` | NMPC tracking the reference → attitude + thrust setpoint, 50 Hz | `control/mpc_core.py` (`control/test_mpc_core.py`) |
| `mpc_offboard_bridge.py` | setpoints → PX4 OFFBOARD (`SET_ATTITUDE_TARGET`), with aborts | `px4/attitude_bridge.py` (`px4/test_attitude_bridge.py`) |
| `flight_monitor.py` | preflight GO / NO-GO, the flight log, the sim-only watchdog → `/flight_monitor/abort` | `px4/flight_checks.py` (`px4/test_flight_checks.py`); scored by `px4/analyze_vio_mpc_flight.py` |
| `ros_camera_pose_fallback.py` | camera → gate poses in the world, using the drone's pose topic | `vision/` (detector, PnP) |
| `ros_state.py` | not a node: ROS messages → the 12-state, shared by the nodes | `test_ros_state.py` |

The messages (`MinSnapTrajectory`, `AttitudeThrustSetpoint`) are in the
`drone_msgs/` package at the repo root. Build it once per container
(`px4/README.md`, "Stage 2").

Not here, on purpose:
- `vision/eval_isaac_gates.py`, `vision/gate_error_eval/eval_gate_bag.py`,
  `vision/verify_isaac_pipeline.py` and `vision/ros_camera.py` use ROS but are
  offline evaluation tools or libraries that other code imports, not part of
  the flight stack.
- `workspace/sim_clock.py` runs inside Isaac Sim's Script Editor.

## Topics

```
/drone00/sensors/imu ─────┐
/iris_0/front_cam/rgb ────┴─▶ vio_node ──/vio/odometry──┬──▶ min_snap_node ──/min_snap/trajectory──┐
                                                       └──────────────────────────────────────▶ mpc_node
                                                                                                    │ /mpc/attitude_setpoint
                                                                                                    ▼
                                                                      mpc_offboard_bridge ──MAVLink──▶ PX4
```

| Topic | Type | From → to |
|---|---|---|
| `/drone00/sensors/imu` | `sensor_msgs/Imu` (FRD, 250 Hz sim) | Pegasus → `vio_node` |
| `/iris_0/front_cam/rgb`, `/camera_info` | `sensor_msgs/Image`, `CameraInfo` | Isaac → `vio_node`, `ros_camera_pose_fallback` |
| `/vio/odometry` | `nav_msgs/Odometry` (map ENU, body FLU) | `vio_node` → `min_snap_node`, `mpc_node` |
| `/min_snap/trajectory` | `drone_msgs/MinSnapTrajectory` (latched, stamp = t 0) | `min_snap_node` → `mpc_node` |
| `/mpc/attitude_setpoint` | `drone_msgs/AttitudeThrustSetpoint` | `mpc_node` → `mpc_offboard_bridge` |
| `/mpc_offboard_bridge/engaged` | `std_msgs/Bool` (latched) | bridge → `min_snap_node` (`start_topic`): the plan starts at OFFBOARD |
| `/flight_monitor/abort` | `std_msgs/String` (the reason) | `flight_monitor` → bridge (aborts to AUTO.LOITER) |
| `/iris_0/front_cam/gate_poses` | `geometry_msgs/PoseArray` | `ros_camera_pose_fallback` → `min_snap_node` (detection mode) |
| `/drone00/state/pose`, `/twist` | ground truth | `state_source:=pegasus` (bring-up only); `vio_node` logs it for scoring |

## Running

From the repo root, with ROS sourced and every node on `use_sim_time:=true`
(Isaac must publish `/clock`: `workspace/sim_clock.py`). The full VIO + MPC
runbook, staged with GO criteria, is `px4/VIO_MPC_TEST_PLAN.md`:

```bash
python3 ros2_nodes/vio_node.py --ros-args -p use_sim_time:=true -p camera_pitch_deg:=0.0
python3 ros2_nodes/min_snap_node.py --ros-args -p use_sim_time:=true -p state_source:=odometry \
    -p odom_topic:=/vio/odometry -p gates_file:=/workspace/test_gates.json -p yaw_mode:=next_gate -p settle_s:=3.0 \
    -p v_max:=2.0 -p a_max:=2.0 -p start_topic:=/mpc_offboard_bridge/engaged
python3 ros2_nodes/flight_monitor.py --ros-args -p use_sim_time:=true
python3 ros2_nodes/mpc_node.py --ros-args -p use_sim_time:=true -p state_source:=odometry -p odom_topic:=/vio/odometry
python3 ros2_nodes/mpc_offboard_bridge.py --ros-args -p use_sim_time:=true
```

**Status:** none of these nodes has run on ROS yet. Their logic is tested
without ROS, including closed-loop rehearsals (`Trajectory/simulate_vio_mpc.py`).
