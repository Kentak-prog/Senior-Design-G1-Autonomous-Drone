#!/usr/bin/env python3
"""
ROS 2 gate-pose publisher for Isaac Sim environments that do not publish TF.

This is a no-TF fallback for the main ros_camera.py pipeline. Instead of
requiring a world->camera TF lookup, it consumes the vehicle/world pose topic
and composes it with a known static body->camera transform.

Expected runtime:
- a camera image topic, e.g. /iris_0/front_cam/rgb
- camera_info topic, e.g. /iris_0/front_cam/camera_info
- PoseStamped topic for the drone body/world pose, e.g. /drone00/state/pose
- optional body->camera static transform in the USD/body convention

This script publishes:
- /iris_0/front_cam/gate_poses : geometry_msgs/PoseArray

It intentionally mirrors the gate-detection and pose-estimation logic used by
vision/ros_camera.py so the downstream trajectory node sees the same type of
output.
"""

import argparse

import numpy as np

from camera import make_transform
from gate_pose import estimate_gate_pose, gate_pose_world, approach_waypoints, DEFAULT_GATE_SIDE
from ros_camera import intrinsics_from_camera_info, T_world_cam_from_transform


def body_to_camera_mount_from_params(
    translation=(0.30, 0.0, 0.05),
    rotate_xyz_deg=(105.0, 0.0, -90.0),
):
    """Construct the USD/body body->camera mount transform used by Isaac Sim.

    This matches the same static transform described in vision/eval_isaac_gates.py.
    """
    x_deg, y_deg, z_deg = rotate_xyz_deg
    x, y, z = np.radians([x_deg, y_deg, z_deg])

    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)

    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    R = Rz @ Ry @ Rx
    return make_transform(R, np.asarray(translation, dtype=float))


def _rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> quaternion in ROS (x, y, z, w) ordering."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x, y, z = (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x = (R[2, 1] - R[1, 2]) / S, 0.25 * S
        y, z = (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, y = (R[0, 2] - R[2, 0]) / S, 0.25 * S
        x, z = (R[0, 1] + R[1, 0]) / S, (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, z = (R[1, 0] - R[0, 1]) / S, 0.25 * S
        x, y = (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S
    return np.array([x, y, z, w])


class PoseFallbackGatePoseNode:
    """ROS 2 node using a pose topic instead of TF."""

    def __init__(
        self,
        detector,
        namespace: str = "/iris_0",
        pose_topic: str = "/drone00/state/pose",
        body_to_camera: np.ndarray | None = None,
        optical_frame: bool = False,
        gate_side: float = DEFAULT_GATE_SIDE,
        corner_sigma_px: float = 1.5,
        max_reprojection_error_px: float = 15.0,
    ):
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image, CameraInfo
        from geometry_msgs.msg import PoseArray, Pose, PoseStamped

        self._detector = detector
        self._namespace = namespace
        self._pose_topic = pose_topic
        self._optical_frame = optical_frame
        self._gate_side = gate_side
        self._corner_sigma_px = corner_sigma_px
        self._max_reprojection_error_px = max_reprojection_error_px

        self._body_to_camera = (
            np.asarray(body_to_camera, dtype=float)
            if body_to_camera is not None
            else body_to_camera_mount_from_params()
        )
        self._last_pose = None

        self._node = Node("gate_pose_pose_fallback_node")
        self._intr = None
        self._bridge = None

        self._node.create_subscription(PoseStamped, self._pose_topic,
                                       self._on_pose, 10)

        topic = f"{namespace}/front_cam"
        self._pub = self._node.create_publisher(PoseArray, f"{topic}/gate_poses", 10)
        self._node.create_subscription(CameraInfo, f"{topic}/camera_info",
                                       self._on_camera_info, 10)
        self._node.create_subscription(Image, f"{topic}/rgb", self._on_image, 10)

    @property
    def node(self):
        return self._node

    def _on_pose(self, msg):
        self._last_pose = msg.pose

    def _on_camera_info(self, msg):
        self._intr = intrinsics_from_camera_info(msg)

    def _camera_pose_to_world_camera_matrix(self):
        if self._last_pose is None:
            return None

        pose = self._last_pose
        T_world_body = np.asarray(
            make_transform(
                np.asarray(
                    __import__('camera').quat_to_rot(
                        np.array([pose.orientation.w, pose.orientation.x,
                                  pose.orientation.y, pose.orientation.z], dtype=float)
                    ),
                    dtype=float
                ),
                np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float),
            ),
            dtype=float,
        )

        # This is the world->camera transform for the no-TF fallback path.
        return T_world_body @ self._body_to_camera

    def _on_image(self, msg):
        if self._intr is None:
            self._node.get_logger().warn("no camera_info yet; skipping frame")
            return

        if self._last_pose is None:
            self._node.get_logger().warn("no pose yet; skipping frame")
            return

        if self._bridge is None:
            from cv_bridge import CvBridge
            self._bridge = CvBridge()

        rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        T_world_cam = self._camera_pose_to_world_camera_matrix()
        if T_world_cam is None:
            return

        # Match ros_camera.py's convention handling for the camera pose.
        # When this is generated from a USD/body pose, the default is optical_frame=False.
        if self._optical_frame:
            T_world_cam = T_world_cam
        else:
            from camera import R_CV_FROM_USD
            R_world_usd = T_world_cam[:3, :3]
            R_world_cv = R_world_usd @ R_CV_FROM_USD.T
            T_world_cam = make_transform(R_world_cv, T_world_cam[:3, 3])

        detections = self._detector(rgb)

        from geometry_msgs.msg import PoseArray, Pose
        out = PoseArray()
        out.header = msg.header
        out.header.frame_id = "world"

        for corners in detections:
            det = estimate_gate_pose(corners, self._intr, self._gate_side,
                                     corner_sigma_px=self._corner_sigma_px)
            if det is None:
                continue
            if det.reprojection_error_px > self._max_reprojection_error_px:
                continue

            T_world_gate = gate_pose_world(det, T_world_cam)
            center = T_world_gate[:3, 3]
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = center.tolist()

            if det.normal_is_reliable:
                qx, qy, qz, qw = _rot_to_quat_xyzw(T_world_gate[:3, :3])
                pose.orientation.x, pose.orientation.y = qx, qy
                pose.orientation.z, pose.orientation.w = qz, qw
                approach_waypoints(T_world_gate)
            else:
                pose.orientation.w = 1.0

            out.poses.append(pose)

        self._pub.publish(out)


def main():
    ap = argparse.ArgumentParser(description="No-TF fallback for the gate-pose pipeline")
    ap.add_argument("--namespace", default="/iris_0")
    ap.add_argument("--pose-topic", default="/drone00/state/pose")
    ap.add_argument("--optical-frame", action="store_true")
    ap.add_argument("--gate-side", type=float, default=DEFAULT_GATE_SIDE)
    ap.add_argument("--corner-sigma-px", type=float, default=1.5)
    ap.add_argument("--max-reprojection-error-px", type=float, default=15.0)
    args = ap.parse_args()

    try:
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
    except ImportError as exc:
        raise SystemExit(f"this script needs a ROS 2 environment: {exc}")

    # Use the repo's exact detector implementation if available.
    try:
        from color_gate_detector import detect_gates_color
        detector = detect_gates_color
    except Exception:
        raise SystemExit("could not import color_gate_detector; ensure the repo is in PYTHONPATH")

    rclpy.init()
    node = PoseFallbackGatePoseNode(
        detector=detector,
        namespace=args.namespace,
        pose_topic=args.pose_topic,
        optical_frame=args.optical_frame,
        gate_side=args.gate_side,
        corner_sigma_px=args.corner_sigma_px,
        max_reprojection_error_px=args.max_reprojection_error_px,
    )

    try:
        rclpy.spin(node.node)
    finally:
        node.node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
