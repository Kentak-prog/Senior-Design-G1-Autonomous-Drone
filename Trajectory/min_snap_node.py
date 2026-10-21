#!/usr/bin/env python3
"""
ROS 2 node for min-snap trajectory generation.

Input:
  /gate_poses        geometry_msgs/PoseArray

Output:
  /min_snap/trajectory  custom message: drone_msgs/MinSnapTrajectory

This wraps the repo's trajectory generator:
  - gate_path.plan_gate_trajectory(...)
  - trajectory_reference.TrajectoryReference(...)

It does NOT change the trajectory math.
"""

import os
import sys
from typing import List, Dict, Any

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("control", "Trajectory", "vision"):
    p = os.path.join(REPO, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from gate_path import Gate, plan_gate_trajectory
from trajectory_reference import TrajectoryReference

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose

try:
    from drone_msgs.msg import MinSnapPoint, MinSnapTrajectory
except Exception:
    # fallback minimal message class if not generated yet
    # For a real project, generate this with ros2 interface packages.
    from rclpy.serialization import serialize_message
    # This block is only a placeholder; in practice generate custom msg.
    class MinSnapPoint:
        pass
    class MinSnapTrajectory:
        pass


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

def pose_array_to_gate_poses(msg: PoseArray) -> List[Dict[str, Any]]:
    gates = []
    for i, pose in enumerate(msg.poses):
        yaw = quat_to_yaw(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        gates.append({
            "name": f"gate_{i}",
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "z": float(pose.position.z),
            "yaw": yaw,
            "side": 1.5,
        })
    return gates

def gate_dict_to_gate(g: Dict[str, Any]) -> Gate:
    x = float(g["x"])
    y = float(g["y"])
    z = float(g["z"])
    yaw = float(g.get("yaw", 0.0))
    side = float(g.get("side", 1.5))

    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw),  np.cos(yaw), 0.0],
        [0.0,         0.0,        1.0],
    ], dtype=float)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return Gate(name=str(g.get("name", "gate")), T_world_gate=T, side=side)

class MinSnapRosNode(Node):
    def __init__(self):
        super().__init__("min_snap_ros_node")

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(PoseArray, "/gate_poses", self.gate_cb, qos)

        self.pub = self.create_publisher(MinSnapTrajectory, "/min_snap/trajectory", 10)
        self.timer = self.create_timer(0.05, self.publish_trajectory)

        self.start_pose = np.zeros(3)
        self.latest_gate_poses: List[Dict[str, Any]] = []

    def gate_cb(self, msg: PoseArray):
        self.latest_gate_poses = pose_array_to_gate_poses(msg)

    def publish_trajectory(self):
        if len(self.latest_gate_poses) == 0:
            return

        gates = [gate_dict_to_gate(g) for g in self.latest_gate_poses]
        start = self.start_pose.copy()

        try:
            traj, waypoints = plan_gate_trajectory(
                gates=gates,
                start=start,
                v_max=4.0,
                a_max=6.0,
                standoff=1.0,
                finish_dist=4.0,
                clearance=0.5,
                max_iters=10,
                verbose=False,
            )

            ref = TrajectoryReference(
                traj,
                start_yaw=0.0,
                dt=0.005,
                m=1.0,
            )

            ts = np.arange(0.0, ref.duration + 1e-9, 0.05)
            points = []
            for t in ts:
                s = ref.reference_state(float(t))
                p = MinSnapPoint()
                p.t = float(t)
                p.x = float(s[0])
                p.y = float(s[1])
                p.z = float(s[2])
                p.vx = float(s[6])
                p.vy = float(s[7])
                p.vz = float(s[8])
                p.yaw = float(s[5])
                points.append(p)

            msg = MinSnapTrajectory()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.points = points
            self.pub.publish(msg)

        except Exception as exc:
            self.get_logger().warn(f"min_snap generation failed: {exc}")

def main():
    rclpy.init()
    node = MinSnapRosNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()