#!/usr/bin/env python3
"""
ROS 2 node for min-snap trajectory generation.

Input:
  /iris_0/front_cam/gate_poses  geometry_msgs/PoseArray (ros2_nodes/ros_camera_pose_fallback.py), world ENU.
                                Each pose's orientation is the FULL gate frame
                                (+X right, +Y down, +Z flight direction); an identity
                                quaternion means the gate's normal was unreliable.
  state                         same sources as ros2_nodes/mpc_node.py (pegasus / odometry)

Output:
  /min_snap/trajectory  drone_msgs/MinSnapTrajectory, latched (transient local).
                        header.stamp = ROS time of t = 0 (sim time with use_sim_time);
                        every point carries the full 12-state reference + feed-forward thrust.

What it does:
  1. Latches the per-frame detections into a stable set of gates (GateLatch):
     a detection within match_dist of a known gate refines it, anything else
     starts a new one; a gate needs min_hits sightings to count. Gates are
     ordered by their distance from the drone when first seen.
  2. Once at least min_gates gates are confirmed and no new gate has appeared
     for settle_s, plans ONCE from the drone's live position and heading
     and publishes it.
  3. Keeps watching: if a planned gate's estimate later moves more than
     replan_warn_m it logs a warning. In-flight replanning is not done here
     (Trajectory/replanner.py is the place for it).

  gates_file:=<test_gates.json> skips 1 and plans through the KNOWN course map
  once the state has been arriving for settle_s (the VIO + MPC run: the VIO
  localizes against the same map). yaw_mode:=next_gate points the nose at the
  next gate instead of along the velocity, which keeps the gates in the camera.

The planning lives in Trajectory/min_snap_planner.py (no ROS; tested there).

Run (ros2-eval, ROS sourced, drone_msgs built):
  python3 ros2_nodes/min_snap_node.py --ros-args -p use_sim_time:=true
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_REPO, "Trajectory"), os.path.join(_REPO, "control")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from min_snap_planner import (gate_from_pose, GateLatch, plan_course, trajectory_points,   # noqa: E402
                              POINT_FIELDS, DEFAULT_GATE_SIDE)
from ros_state import StateTracker, subscribe_state                                         # noqa: E402


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
    from geometry_msgs.msg import PoseArray
    from drone_msgs.msg import MinSnapPoint, MinSnapTrajectory

    class MinSnapRosNode(Node):
        def __init__(self):
            super().__init__("min_snap_ros_node")
            self.declare_parameter("gate_topic", "/iris_0/front_cam/gate_poses")
            self.declare_parameter("trajectory_topic", "/min_snap/trajectory")
            self.declare_parameter("state_source", "pegasus")
            self.declare_parameter("pose_topic", "/drone00/state/pose")
            self.declare_parameter("twist_topic", "/drone00/state/twist")
            self.declare_parameter("odom_topic", "/state_estimate/odometry")
            # 0 = the planner default: 80 % of the drone's limits (9.6 m/s, 12 m/s^2 for the Iris)
            self.declare_parameter("v_max", 0.0)
            self.declare_parameter("a_max", 0.0)
            self.declare_parameter("optimize_times", True)
            self.declare_parameter("min_gates", 1)
            self.declare_parameter("min_hits", 3)
            self.declare_parameter("match_dist", 1.0)
            self.declare_parameter("settle_s", 1.0)
            self.declare_parameter("gate_side", DEFAULT_GATE_SIDE)
            self.declare_parameter("replan_warn_m", 0.3)
            # gates_file set: plan through the KNOWN course map (the VIO localizes against the same
            # map) instead of latching live detections. yaw_mode next_gate: look at the next gate.
            self.declare_parameter("gates_file", "")
            self.declare_parameter("yaw_mode", "velocity")
            gp = lambda n: self.get_parameter(n).value   # noqa: E731
            self.gp = gp

            self.tracker = StateTracker()
            subscribe_state(self, self.tracker, gp("state_source"), gp("pose_topic"), gp("twist_topic"),
                            gp("odom_topic"))
            self.latch = GateLatch(match_dist=float(gp("match_dist")), min_hits=int(gp("min_hits")))
            self.map_gates = None
            self.first_state_time = None
            if gp("gates_file"):
                from gate_path import gates_from_manifest
                self.map_gates = gates_from_manifest(gp("gates_file"))
                self.get_logger().info(f"planning through {len(self.map_gates)} map gates from {gp('gates_file')} "
                                       f"(yaw_mode {gp('yaw_mode')})")
            else:
                self.create_subscription(PoseArray, gp("gate_topic"), self.gate_cb, qos_profile_sensor_data)
            latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.pub = self.create_publisher(MinSnapTrajectory, gp("trajectory_topic"), latched)
            self.create_timer(0.1, self.tick)
            self.planned_gates = None
            self.MinSnapPoint, self.MinSnapTrajectory = MinSnapPoint, MinSnapTrajectory

        def _now(self):
            return self.get_clock().now().nanoseconds * 1e-9

        def gate_cb(self, msg):
            x = self.tracker.state
            if x is None:
                return
            viewer = x[0:3, 0]
            dets = [gate_from_pose(p.position, (p.orientation.x, p.orientation.y, p.orientation.z,
                                                p.orientation.w),
                                   side=float(self.gp("gate_side")), viewer_pos=viewer) for p in msg.poses]
            self.latch.update(dets, self._now(), viewer)

        def tick(self):
            x = self.tracker.state
            if x is None:
                return
            if self.map_gates is not None:
                # known map: plan once the state has been coming in for settle_s (estimator settled)
                if self.planned_gates is not None:
                    return
                if self.first_state_time is None:
                    self.first_state_time = self._now()
                if self._now() - self.first_state_time >= float(self.gp("settle_s")):
                    self.publish_plan(self.map_gates, x)
                return
            gates = self.latch.confirmed(viewer_pos=x[0:3, 0])
            if self.planned_gates is not None:
                for g, g0 in zip(gates, self.planned_gates):
                    if np.linalg.norm(g.center - g0.center) > float(self.gp("replan_warn_m")):
                        self.get_logger().warn(f"{g.name} moved {np.linalg.norm(g.center - g0.center):.2f} m since "
                                               f"the plan (not replanning in flight)", throttle_duration_sec=5.0)
                return
            settled = (self.latch.last_new_gate_time is not None
                       and self._now() - self.latch.last_new_gate_time >= float(self.gp("settle_s")))
            if len(gates) < int(self.gp("min_gates")) or not settled:
                return
            self.publish_plan(gates, x)

        def publish_plan(self, gates, x):
            start, yaw = x[0:3, 0].copy(), float(x[5, 0])
            try:
                traj, wps, ref = plan_course(gates, start, yaw, float(self.gp("v_max")) or None,
                                             float(self.gp("a_max")) or None, yaw_mode=str(self.gp("yaw_mode")),
                                             optimize_times=bool(self.gp("optimize_times")))
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"min-snap planning failed: {exc}", throttle_duration_sec=2.0)
                return
            ts, states, thrust = trajectory_points(ref)
            msg = self.MinSnapTrajectory()
            msg.header.stamp = self.get_clock().now().to_msg()     # t = 0 of this plan
            msg.header.frame_id = "map"
            for k, t in enumerate(ts):
                pt = self.MinSnapPoint()
                pt.t = float(t)
                for i, f in enumerate(POINT_FIELDS):
                    setattr(pt, f, float(states[i, k]))
                pt.thrust = float(thrust[k])
                msg.points.append(pt)
            self.pub.publish(msg)
            self.planned_gates = gates
            self.get_logger().info(
                f"published plan through {len(gates)} gate(s) from {np.round(start, 2)}: {traj.duration:.2f} s, "
                f"{len(wps)} waypoints, {len(ts)} points; "
                + ", ".join(f"{g.name} {np.round(g.center, 2)}{'' if g.normal_reliable else ' (center only)'}"
                            for g in gates))

    rclpy.init()
    node = MinSnapRosNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
