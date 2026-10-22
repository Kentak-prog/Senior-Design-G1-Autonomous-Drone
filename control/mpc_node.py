#!/usr/bin/env python3
"""
MPC bridge node.

Input topics:
  /isaac/odometry   nav_msgs/Odometry
  /min_snap/trajectory  custom message from min_snap node

Output:
  /mpc/command      geometry_msgs/Twist

Uses the repo's existing optimizer:
  control/MPC_Optimizer.py
"""

import os
import sys
from typing import List, Dict, Any, Optional

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("control", "Trajectory", "vision"):
    p = os.path.join(REPO, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from MPC_Optimizer import create_drone_mpc_optimizer
from drone_constraints import drone_constraints
from trajectory_reference import TrajectoryReference

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from drone_msgs.msg import MinSnapPoint, MinSnapTrajectory


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class IsaacMpcBridge(Node):
    def __init__(self):
        super().__init__("isaac_mpc_bridge")

        self.dt = 0.02
        self.N = 10
        self.mass = drone_constraints["mass"]

        self.state = np.zeros((12, 1), dtype=float)
        self.min_snap_points: List[Dict[str, float]] = []

        self.opti, self.X, self.U, self.X_init, self.U_prev, self.X_ref = create_drone_mpc_optimizer(
            N=self.N, dt=self.dt, m=self.mass
        )

        self.u_prev = np.zeros((4, 1))

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(Odometry, "/isaac/odometry", self.odom_cb, qos)
        self.create_subscription(MinSnapTrajectory, "/min_snap/trajectory", self.min_snap_cb, qos)

        self.cmd_pub = self.create_publisher(Twist, "/mpc/command", 10)
        self.control_timer = self.create_timer(0.02, self.control_loop)

    def odom_cb(self, msg: Odometry):
        pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z], dtype=float)
        vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z], dtype=float)
        ang = np.array([msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z], dtype=float)

        q = msg.pose.pose.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)

        self.state = np.array([
            [pos[0]], [pos[1]], [pos[2]],
            [0.0], [0.0], [yaw],
            [vel[0]], [vel[1]], [vel[2]],
            [ang[0]], [ang[1]], [ang[2]],
        ], dtype=float)

    def min_snap_cb(self, msg):
        self.min_snap_points = []
        for p in msg.points:
            self.min_snap_points.append({
                "t": float(p.t),
                "x": float(p.x),
                "y": float(p.y),
                "z": float(p.z),
                "vx": float(p.vx),
                "vy": float(p.vy),
                "vz": float(p.vz),
                "yaw": float(p.yaw),
            })

    def _reference_from_min_snap(self, t0: float) -> Optional[np.ndarray]:
        if len(self.min_snap_points) < 2:
            return None

        pts = np.array([
            [p["t"], p["x"], p["y"], p["z"], p["vx"], p["vy"], p["vz"], p["yaw"]]
            for p in self.min_snap_points
        ], dtype=float)

        ref = np.zeros((12, self.N + 1), dtype=float)
        ts = t0 + np.arange(self.N + 1) * self.dt
        times = pts[:, 0]

        for k, t in enumerate(ts):
            x = np.interp(t, times, pts[:, 1])
            y = np.interp(t, times, pts[:, 2])
            z = np.interp(t, times, pts[:, 3])
            vx = np.interp(t, times, pts[:, 4])
            vy = np.interp(t, times, pts[:, 5])
            vz = np.interp(t, times, pts[:, 6])
            yaw = np.interp(t, times, pts[:, 7], period=2*np.pi)

            ref[0, k] = x
            ref[1, k] = y
            ref[2, k] = z

            ref[3, k] = 0.0
            ref[4, k] = 0.0
            ref[5, k] = yaw

            ref[6, k] = vx
            ref[7, k] = vy
            ref[8, k] = vz

            ref[9, k] = 0.0
            ref[10, k] = 0.0
            ref[11, k] = 0.0

        return ref

    def build_reference_horizon(self, t0: float) -> np.ndarray:
        if len(self.min_snap_points) >= 2:
            ref = self._reference_from_min_snap(t0)
            if ref is not None:
                return ref

        hover = np.tile(self.state.reshape(12), (1, self.N + 1))
        return hover

    def control_loop(self):
        try:
            ref_horizon = self.build_reference_horizon(0.0)

            self.opti.set_value(self.X_init, self.state)
            self.opti.set_value(self.U_prev, np.zeros((4, 1)))
            self.opti.set_value(self.X_ref, ref_horizon)

            try:
                self.opti.set_initial(self.X, np.tile(self.state, (1, self.N + 1)))
                self.opti.set_initial(self.U, np.tile(np.array([[0.0],[0.0],[0.0],[0.0]]), (1, self.N)))
            except Exception:
                pass

            sol = self.opti.solve()
            U_sol = sol.value(self.U)
            u0 = np.asarray(U_sol[:, 0]).reshape(4, 1)

        except Exception:
            u0 = np.array([[self.mass * 9.81], [0.0], [0.0], [0.0]])

        cmd = Twist()
        cmd.linear.x = float(u0[0, 0] / 10.0)
        cmd.linear.y = float(u0[1, 0] / 10.0)
        cmd.linear.z = float(u0[2, 0] / 10.0)
        cmd.angular.z = float(u0[3, 0] / 10.0)

        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = IsaacMpcBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
