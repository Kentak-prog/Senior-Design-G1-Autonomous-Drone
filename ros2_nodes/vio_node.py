#!/usr/bin/env python3
"""
VIO node: IMU + front camera -> nav_msgs/Odometry, on a known gate map
(control/vio.py). The pose comes ONLY from the IMU and the camera.

Inputs:
  /drone00/sensors/imu           sensor_msgs/Imu (Pegasus: FRD body, specific force, 250 Hz sim)
  /iris_0/front_cam/rgb          sensor_msgs/Image
  /iris_0/front_cam/camera_info  sensor_msgs/CameraInfo
  gates_file                     the course manifest (test_gates.json); the gates are spawned from it
Output:
  /vio/odometry                  nav_msgs/Odometry, frame "map" (ENU), child "base_link" (FLU),
                                 twist in the body frame (REP-105). mpc_node / min_snap_node read
                                 it with state_source:=odometry.

Logging only (never fed to the estimator): if ground_truth_topic is set, the
simulator's pose is recorded next to the estimate in <log_dir>/vio_log.csv.

TIMES: run Isaac's workspace/sim_clock.py and this node with use_sim_time:=true
so the IMU and image stamps share sim time (see control/vio.py "TIMES").

Run (ros2-eval, ROS sourced, drone hovering):
  python3 ros2_nodes/vio_node.py --ros-args -p use_sim_time:=true -p camera_pitch_deg:=0.0
"""

import csv
import json
import math
import os
import sys
import threading
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("ros2_nodes", "control", "Trajectory", "vision"):
    p = os.path.join(REPO, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from vio import GateVIO, camera_mount_usd, rot_rpy, _inv                  # noqa: E402
from ros_state import quat_xyzw_from_rpy, stamp_to_sec                    # noqa: E402


def spawn_hint(manifest_raw, recorded_pitch_deg=15.0):
    """(pos, yaw) of the body when the gates were spawned, from the manifest's camera pose
    (recorded with the camera mount of that time) -- only a hint for matching the first detections."""
    cam = manifest_raw.get("camera_pose_usd") if isinstance(manifest_raw, dict) else None
    if not cam or "R_world_cam_usd" not in cam:
        return None
    T_wc = np.eye(4)
    T_wc[:3, :3] = np.asarray(cam["R_world_cam_usd"], dtype=float)
    T_wc[:3, 3] = cam["position"]
    T_wb = T_wc @ _inv(camera_mount_usd(recorded_pitch_deg))
    return T_wb[:3, 3].copy(), math.atan2(T_wb[1, 0], T_wb[0, 0])


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Imu, Image, CameraInfo
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PoseStamped
    from gate_path import gates_from_manifest
    from gate_pose import estimate_gate_pose
    from color_gate_detector import detect_gates_color
    from ros_camera import intrinsics_from_camera_info

    class VioNode(Node):
        def __init__(self):
            super().__init__("vio_node")
            dp = self.declare_parameter
            dp("gates_file", "/workspace/test_gates.json")
            dp("imu_topic", "/drone00/sensors/imu")
            dp("image_topic", "/iris_0/front_cam/rgb")
            dp("camera_info_topic", "/iris_0/front_cam/camera_info")
            dp("odom_topic", "/vio/odometry")
            dp("camera_pitch_deg", 15.0)          # MUST match the camera spawn_example.py set up
            dp("manifest_camera_pitch_deg", 15.0)  # the mount when the manifest was recorded (2026-10-23: 15)
            dp("imu_rate", 250.0)
            dp("calib_s", 2.0)
            dp("corner_sigma_px", 1.0)
            dp("bootstrap_n", 8)
            dp("max_range", 15.0)
            dp("delay_compensation", True)
            dp("publish_every", 2)                # publish odometry every N IMU samples (250 Hz / 2)
            dp("ground_truth_topic", "/drone00/state/pose")   # logging only; "" disables
            dp("log_dir", "")
            gp = lambda n: self.get_parameter(n).value   # noqa: E731
            self.gp = gp

            with open(gp("gates_file")) as f:
                manifest = json.load(f)
            gates = gates_from_manifest(gp("gates_file"))
            hint = spawn_hint(manifest, float(gp("manifest_camera_pitch_deg")))
            self.mount = camera_mount_usd(float(gp("camera_pitch_deg")))
            self.vio = GateVIO(gates, self.mount, imu_rate=float(gp("imu_rate")), calib_s=float(gp("calib_s")),
                               max_range=float(gp("max_range")), initial_guess=hint,
                               delay_compensation=bool(gp("delay_compensation")))
            self.lock = threading.Lock()
            self.intr = None
            self.bridge = None
            self.n_imu = 0
            self.gt = None
            self.last_status = time.monotonic()
            self.cam_ms = []
            self.reported_calibration = False

            imu_group = MutuallyExclusiveCallbackGroup()
            cam_group = MutuallyExclusiveCallbackGroup()
            self.create_subscription(Imu, gp("imu_topic"), self.on_imu, qos_profile_sensor_data,
                                     callback_group=imu_group)
            self.create_subscription(Image, gp("image_topic"), self.on_image, qos_profile_sensor_data,
                                     callback_group=cam_group)
            self.create_subscription(CameraInfo, gp("camera_info_topic"), self.on_info, qos_profile_sensor_data,
                                     callback_group=cam_group)
            if gp("ground_truth_topic"):
                self.create_subscription(PoseStamped, gp("ground_truth_topic"), self.on_gt, qos_profile_sensor_data,
                                         callback_group=imu_group)
            self.pub = self.create_publisher(Odometry, gp("odom_topic"), 10)

            log_dir = gp("log_dir") or os.path.join(
                "/workspace/vio_runs" if os.path.isdir("/workspace") else os.path.join(REPO, "vio_runs"),
                time.strftime("vio_%Y%m%d_%H%M%S"))
            os.makedirs(log_dir, exist_ok=True)
            self.log_dir = log_dir
            self.log_f = open(os.path.join(log_dir, "vio_log.csv"), "w", newline="")
            self.log = csv.writer(self.log_f)
            self.log.writerow(["t", "x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz",
                               "sx", "sy", "sz", "gt_x", "gt_y", "gt_z", "gt_yaw", "fixes", "rejected"])
            self.get_logger().info(f"{len(gates)} map gates from {gp('gates_file')}; camera pitch "
                                   f"{gp('camera_pitch_deg')} deg; hint {None if hint is None else np.round(hint[0], 2)}; "
                                   f"log {log_dir}; use_sim_time={self.get_parameter('use_sim_time').value}")
            self.get_logger().info(f"calibrating the gyro for {gp('calib_s')} s -- keep the drone hovering")

        # ---------------- callbacks
        def on_info(self, msg):
            if self.intr is None:
                self.intr = intrinsics_from_camera_info(msg)
                self.get_logger().info(f"camera {msg.width}x{msg.height}, fx {self.intr.fx:.1f}")

        def on_gt(self, msg):
            p, q = msg.pose.position, msg.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            self.gt = (p.x, p.y, p.z, yaw)

        def on_imu(self, msg):
            t = stamp_to_sec(msg.header.stamp)
            a, w = msg.linear_acceleration, msg.angular_velocity
            with self.lock:
                self.vio.add_imu(t, [a.x, a.y, a.z], [w.x, w.y, w.z])
                self.n_imu += 1
                x = self.vio.state
                std = self.vio.position_std()
                stats = dict(self.vio.stats)
                calib = self.vio.calibration_summary if self.vio.calibrated else None
            if calib is not None and not self.reported_calibration:
                self.reported_calibration = True
                self.get_logger().info(f"gyro bias {np.round(calib['gyro_bias_deg_s'], 3)} deg/s; |accel| at hover "
                                       f"{calib['accel_norm']:.3f} m/s^2 (expect ~9.81 -- else the drone was not hovering "
                                       f"or the IMU frame is not what vio.py assumes)")
            if x is None or self.n_imu % int(self.gp("publish_every")):
                self._status(x, std, stats)
                return
            self.publish(msg.header.stamp, x, std)
            gt = self.gt if self.gt is not None else (np.nan,) * 4
            self.log.writerow([f"{t:.4f}"] + [f"{v:.5f}" for v in x[:, 0][:9]] + [f"{v:.4f}" for v in std]
                              + [f"{v:.5f}" for v in gt] + [stats["fixes_full"] + stats["fixes_pos"], stats["rejected"]])
            self._status(x, std, stats)

        def on_image(self, msg):
            if self.intr is None:
                return
            if self.bridge is None:
                from cv_bridge import CvBridge
                self.bridge = CvBridge()
            t0 = time.perf_counter()
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            dets = []
            for corners in detect_gates_color(rgb):
                det = estimate_gate_pose(corners, self.intr, corner_sigma_px=float(self.gp("corner_sigma_px")),
                                         bootstrap_n=int(self.gp("bootstrap_n")))
                if det is not None:
                    dets.append(det)
            with self.lock:
                self.vio.add_detections(stamp_to_sec(msg.header.stamp), dets)
            self.cam_ms.append(1000 * (time.perf_counter() - t0))

        # ---------------- output
        def publish(self, stamp, x, std):
            od = Odometry()
            od.header.stamp = stamp
            od.header.frame_id = "map"
            od.child_frame_id = "base_link"
            s = x[:, 0]
            od.pose.pose.position.x, od.pose.pose.position.y, od.pose.pose.position.z = (float(v) for v in s[0:3])
            q = quat_xyzw_from_rpy(*s[3:6])
            o = od.pose.pose.orientation
            o.x, o.y, o.z, o.w = q
            v_body = rot_rpy(*s[3:6]).T @ s[6:9]                       # REP-105: twist in the child (body) frame
            od.twist.twist.linear.x, od.twist.twist.linear.y, od.twist.twist.linear.z = (float(v) for v in v_body)
            od.twist.twist.angular.x, od.twist.twist.angular.y, od.twist.twist.angular.z = (float(v) for v in s[9:12])
            for i in range(3):
                od.pose.covariance[i * 7] = float(std[i] ** 2)
            self.pub.publish(od)

        def _status(self, x, std, stats):
            now = time.monotonic()
            if now - self.last_status < 2.0:
                return
            self.last_status = now
            cam = f", camera {np.median(self.cam_ms):.0f} ms/frame" if self.cam_ms else ""
            self.cam_ms = self.cam_ms[-50:]
            if x is None:
                state = "calibrating" if not self.vio.calibrated else "waiting for a reliable gate fix to initialize"
                self.get_logger().info(f"{state} ({self.n_imu} IMU samples{cam})")
                return
            gt = ""
            if self.gt is not None:
                gt = f", error vs ground truth {np.linalg.norm(np.array(self.gt[:3]) - x[0:3, 0]):.2f} m"
            self.get_logger().info(f"pos {np.round(x[0:3, 0], 2)} +-{np.round(std, 2)}, fixes {stats['fixes_full']} full "
                                   f"/ {stats['fixes_pos']} pos, {stats['rejected']} rejected{gt}{cam}")

        def close(self):
            self.log_f.close()
            with open(os.path.join(self.log_dir, "vio_events.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "kind", "gate", "chi2"])
                w.writerows(self.vio.events)
            self.get_logger().info(f"logs in {self.log_dir}")

    rclpy.init()
    node = VioNode()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
