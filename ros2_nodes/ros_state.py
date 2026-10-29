"""
ROS message -> the 12-state the MPC and EKF use, shared by the ROS 2 nodes in ros2_nodes/ (mpc_node,
min_snap_node, vio_node).

State order: [x, y, z, roll, pitch, yaw, vx, vy, vz, p, q, r]
World ENU (z up), body FLU, R = Rz(yaw) Ry(pitch) Rx(roll), velocity in the
WORLD frame, body rates in the BODY frame (control/New_Drone_State_Equations.py).

THIS MODULE MUST IMPORT WITHOUT ROS. Messages are read by attribute only, so
the helpers are testable with plain stand-in objects.

State sources (StateTracker):
  "pegasus":  geometry_msgs/PoseStamped  (default /drone00/state/pose, frame "map")
              + geometry_msgs/TwistStamped (default /drone00/state/twist)
              The twist's linear part is in its header.frame_id: a world frame
              ("map", "world", "odom") is used as is, anything else (e.g. "base_link")
              is treated as body FLU and rotated into the world. Angular is body rates.
              This is Pegasus' GROUND-TRUTH state: fine for bringing the controller
              up, not for a vision + IMU test.
  "odometry": nav_msgs/Odometry (pose in header.frame_id, twist in child_frame_id,
              REP-105: child frame = body, so linear is rotated into the world)
              -- e.g. the output of an estimator node.
"""

import numpy as np

WORLD_FRAMES = ("map", "world", "odom", "enu")


def rpy_from_quat_xyzw(qx, qy, qz, qw):
    """(roll, pitch, yaw) for R = Rz(yaw) Ry(pitch) Rx(roll) from a ROS (x, y, z, w) quaternion."""
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return float(roll), float(pitch), float(yaw)


def quat_xyzw_from_rpy(roll, pitch, yaw):
    """ROS (x, y, z, w) quaternion of R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return (float(sr * cp * cy - cr * sp * sy),
            float(cr * sp * cy + sr * cp * sy),
            float(cr * cp * sy - sr * sp * cy),
            float(cr * cp * cy + sr * sp * sy))


def rotation_from_rpy(roll, pitch, yaw):
    """R = Rz(yaw) Ry(pitch) Rx(roll), body -> world."""
    cf, sf, ct, st, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cp * ct, cp * st * sf - sp * cf, cp * st * cf + sp * sf],
                     [sp * ct, sp * st * sf + cp * cf, sp * st * cf - cp * sf],
                     [-st, ct * sf, ct * cf]])


def rotation_from_quat_xyzw(qx, qy, qz, qw):
    return rotation_from_rpy(*rpy_from_quat_xyzw(qx, qy, qz, qw))


def is_world_frame(frame_id: str) -> bool:
    return (frame_id or "").strip("/").lower() in WORLD_FRAMES


def state_from_pose_twist(position, orientation, linear, angular, linear_in_world: bool) -> np.ndarray:
    """
    12x1 state from a pose (geometry_msgs Point + Quaternion) and a twist
    (Vector3 linear + angular). linear_in_world=False rotates a body-frame
    linear velocity into the world frame. angular is taken as body rates.
    """
    rpy = rpy_from_quat_xyzw(orientation.x, orientation.y, orientation.z, orientation.w)
    v = np.array([linear.x, linear.y, linear.z], dtype=float)
    if not linear_in_world:
        v = rotation_from_rpy(*rpy) @ v
    return np.array([[position.x], [position.y], [position.z],
                     [rpy[0]], [rpy[1]], [rpy[2]],
                     [v[0]], [v[1]], [v[2]],
                     [angular.x], [angular.y], [angular.z]], dtype=float)


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


class StateTracker:
    """
    Keeps the latest 12-state from either source. Feed it messages with
    on_pose / on_twist (pegasus) or on_odometry; read .state (12x1 or None)
    and .stamp (seconds, the message's header time).
    """

    def __init__(self):
        self._pose = None
        self._pose_stamp = None
        self._twist = None
        self._twist_world = True
        self.state = None
        self.stamp = None

    def on_pose(self, msg):           # geometry_msgs/PoseStamped
        self._pose = msg.pose
        self._pose_stamp = stamp_to_sec(msg.header.stamp)
        self._assemble()

    def on_twist(self, msg):          # geometry_msgs/TwistStamped
        self._twist = msg.twist
        self._twist_world = is_world_frame(msg.header.frame_id)
        self._assemble()

    def on_odometry(self, msg):       # nav_msgs/Odometry
        p, t = msg.pose.pose, msg.twist.twist
        # REP-105: twist is in child_frame_id (the body), unless that is empty or a world frame
        linear_in_world = (not msg.child_frame_id) or is_world_frame(msg.child_frame_id)
        self.state = state_from_pose_twist(p.position, p.orientation, t.linear, t.angular, linear_in_world)
        self.stamp = stamp_to_sec(msg.header.stamp)

    def _assemble(self):
        if self._pose is None:
            return
        if self._twist is None:
            zero = type("V", (), {"x": 0.0, "y": 0.0, "z": 0.0})()
            linear = angular = zero
        else:
            linear, angular = self._twist.linear, self._twist.angular
        self.state = state_from_pose_twist(self._pose.position, self._pose.orientation, linear, angular,
                                           self._twist_world if self._twist is not None else True)
        self.stamp = self._pose_stamp


def subscribe_state(node, tracker: StateTracker, source: str, pose_topic: str, twist_topic: str,
                    odom_topic: str):
    """Create the subscriptions for `source` ("pegasus" or "odometry") on an rclpy node."""
    from rclpy.qos import qos_profile_sensor_data
    if source == "pegasus":
        from geometry_msgs.msg import PoseStamped, TwistStamped
        node.create_subscription(PoseStamped, pose_topic, tracker.on_pose, qos_profile_sensor_data)
        node.create_subscription(TwistStamped, twist_topic, tracker.on_twist, qos_profile_sensor_data)
        node.get_logger().warn(f"state source 'pegasus' ({pose_topic} + {twist_topic}) is the simulator's "
                               f"GROUND TRUTH -- use it to bring the controller up, not for a vision + IMU test")
    elif source == "odometry":
        from nav_msgs.msg import Odometry
        node.create_subscription(Odometry, odom_topic, tracker.on_odometry, qos_profile_sensor_data)
    else:
        raise ValueError(f"unknown state source {source!r} (use 'pegasus' or 'odometry')")
