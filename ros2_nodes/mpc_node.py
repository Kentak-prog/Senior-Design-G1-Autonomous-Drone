#!/usr/bin/env python3
"""
MPC node: tracks the min-snap reference with the repo NMPC and publishes an
attitude + thrust setpoint for the flight controller.

Inputs:
  state         --state_source pegasus (default): /drone00/state/pose + /drone00/state/twist
                (the simulator's ground truth -- for bringing the controller up)
                --state_source odometry: nav_msgs/Odometry on --odom_topic (e.g. /vio/odometry)
  /min_snap/trajectory   drone_msgs/MinSnapTrajectory (ros2_nodes/min_snap_node.py)

Output:
  /mpc/attitude_setpoint drone_msgs/AttitudeThrustSetpoint, at --rate_hz
                         (ros2_nodes/mpc_offboard_bridge.py sends it to PX4)

Every tick (ROS clock -- sim time when the node runs with use_sim_time:=true):
  1. state = latest full 12-state (roll and pitch included); skip the tick if
     it is older than state_timeout
  2. reference = the trajectory evaluated at (now - trajectory.header.stamp)
     over the MPC horizon; before any trajectory arrives, hold the first
     state's position and yaw (hover)
  3. solve the NMPC, warm-started from the previous solution shifted one step,
     with U_prev = the command applied last tick
  4. publish the predicted attitude and body rates one step ahead plus the
     first thrust command (valid=false and a level hover if the solve failed)

The MPC itself lives in control/mpc_core.py (no ROS; tested there).

Run (ros2-eval, ROS sourced, drone_msgs built):
  python3 ros2_nodes/mpc_node.py --ros-args -p use_sim_time:=true
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_REPO, "control")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mpc_core import TrajectoryBuffer, MpcCore, hover_reference    # noqa: E402
from drone_constraints import drone_constraints                    # noqa: E402
from ros_state import StateTracker, subscribe_state, quat_xyzw_from_rpy   # noqa: E402


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from drone_msgs.msg import MinSnapTrajectory, AttitudeThrustSetpoint

    class MpcNode(Node):
        def __init__(self):
            super().__init__("mpc_node")
            self.declare_parameter("state_source", "pegasus")
            self.declare_parameter("pose_topic", "/drone00/state/pose")
            self.declare_parameter("twist_topic", "/drone00/state/twist")
            self.declare_parameter("odom_topic", "/state_estimate/odometry")
            self.declare_parameter("trajectory_topic", "/min_snap/trajectory")
            self.declare_parameter("setpoint_topic", "/mpc/attitude_setpoint")
            self.declare_parameter("rate_hz", 50.0)
            self.declare_parameter("horizon", 10)
            self.declare_parameter("state_timeout", 0.5)
            # attitude setpoint = the MPC's predicted attitude this far ahead (control/mpc_core.py
            # "ATTITUDE LEAD"); 0.02 = one step, which leaves PX4 holding its current attitude
            self.declare_parameter("attitude_lead_s", 0.1)
            gp = lambda n: self.get_parameter(n).value   # noqa: E731

            self.dt = 1.0 / float(gp("rate_hz"))
            self.N = int(gp("horizon"))
            self.state_timeout = float(gp("state_timeout"))
            self.core = MpcCore(N=self.N, dt=self.dt, m=drone_constraints["mass"],
                                attitude_lead_s=float(gp("attitude_lead_s")))
            self.tracker = StateTracker()
            self.traj = None
            self.hold = None                 # (pos, yaw) hover point while there is no trajectory
            self.last_state_rx = None        # node-clock time the state last arrived
            self.stats = {"n": 0, "fail": 0, "ms": 0.0, "err": 0.0, "t": None}

            # count arrivals on the node clock, so staleness never compares two different clocks
            tracker = self.tracker

            def stamp_rx(fn):
                def wrapped(msg):
                    fn(msg)
                    self.last_state_rx = self._now()
                return wrapped
            tracker.on_pose, tracker.on_twist, tracker.on_odometry = (
                stamp_rx(tracker.on_pose), stamp_rx(tracker.on_twist), stamp_rx(tracker.on_odometry))
            subscribe_state(self, tracker, gp("state_source"), gp("pose_topic"), gp("twist_topic"),
                            gp("odom_topic"))

            latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(MinSnapTrajectory, gp("trajectory_topic"), self.on_trajectory, latched)
            self.pub = self.create_publisher(AttitudeThrustSetpoint, gp("setpoint_topic"), 10)
            self.create_timer(self.dt, self.control_loop)
            self.get_logger().info(f"MPC at {1 / self.dt:.0f} Hz, horizon {self.N}, attitude lead "
                                   f"{self.core.lead_idx * self.dt:.2f} s, mass "
                                   f"{drone_constraints['mass']} kg; "
                                   f"use_sim_time={self.get_parameter('use_sim_time').value}")

        def _now(self):
            return self.get_clock().now().nanoseconds * 1e-9

        def on_trajectory(self, msg):
            try:
                buf = TrajectoryBuffer.from_msg(msg)
            except ValueError as e:
                self.get_logger().warn(f"ignoring trajectory: {e}")
                return
            if self.traj is None or abs(buf.t0 - self.traj.t0) > 1e-6:
                self.get_logger().info(f"new trajectory: {len(buf.times)} points, {buf.duration:.2f} s, "
                                       f"starts at t={buf.t0:.2f}")
            self.traj = buf

        def control_loop(self):
            now = self._now()
            x = self.tracker.state
            if x is None or self.last_state_rx is None or now - self.last_state_rx > self.state_timeout:
                self.get_logger().warn("no fresh state; not publishing", throttle_duration_sec=2.0)
                return

            if self.traj is not None:
                ref = self.traj.horizon(now, self.N, self.dt, yaw_now=float(x[5, 0]))
                self.hold = None
            else:
                if self.hold is None:
                    self.hold = (x[0:3, 0].copy(), float(x[5, 0]))
                    self.get_logger().info(f"no trajectory yet: holding {np.round(self.hold[0], 2)}")
                ref = hover_reference(self.hold[0], self.hold[1], self.N)

            out = self.core.step(x, ref)
            xc, u0 = out["x_cmd"], out["u0"]          # the attitude / rates PX4 should track
            msg = AttitudeThrustSetpoint()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "map"
            qx, qy, qz, qw = quat_xyzw_from_rpy(float(xc[3, 0]), float(xc[4, 0]), float(xc[5, 0]))
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = qx, qy, qz, qw
            msg.body_rate.x, msg.body_rate.y, msg.body_rate.z = (float(v) for v in xc[9:12, 0])
            msg.thrust = float(u0[0, 0])
            msg.solve_time_ms = float(out["solve_ms"])
            msg.valid = bool(out["ok"])
            self.pub.publish(msg)

            s = self.stats
            s["n"] += 1
            s["fail"] += 0 if out["ok"] else 1
            s["ms"] += out["solve_ms"]
            s["err"] += float(np.linalg.norm(x[0:3, 0] - ref[0:3, 0]))
            if s["t"] is None:
                s["t"] = now
            if now - s["t"] >= 2.0:
                self.get_logger().info(f"{s['n']} solves: mean {s['ms'] / s['n']:.1f} ms, {s['fail']} failed, "
                                       f"mean position error {s['err'] / s['n']:.2f} m")
                self.stats = {"n": 0, "fail": 0, "ms": 0.0, "err": 0.0, "t": now}

    rclpy.init()
    node = MpcNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
