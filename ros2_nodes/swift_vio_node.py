#!/usr/bin/env python3
"""
Swift-style VIO in Isaac (control/drift_filter.py): an EMULATED base VIO (the simulator's true pose plus
a feature-tracking VIO's drift) corrected by a drift Kalman filter from gate sightings, published where
runner.py --state vio reads it. px4/SWIFT_VOXL2_PLAN.md, "Order of work" step 2.

Inputs:
  /drone00/state/pose            geometry_msgs/PoseStamped, Pegasus' true pose (ENU / FLU). Only the
                                 EMULATOR reads it, to make the base VIO; the filter never sees it.
  /drone00/state/twist_inertial  geometry_msgs/TwistStamped, true world-frame velocity (if absent, the
                                 pose is differentiated)
  /iris_0/front_cam/rgb, /camera_info   the C29 front camera
  gates_file                     the course (rl/tracks/*.json)
Outputs:
  /vio/odometry                  nav_msgs/Odometry, the CORRECTED estimate (base VIO - drift), frame
                                 "map" (ENU), child "base_link" (FLU), twist in the body frame (REP-105)
                                 -- what runner.py --state vio flies on. With correct:=false the base VIO
                                 itself goes out here (the "no drift filter" baseline).
  /vio_base/odometry             the base VIO (logging)
  <log_dir>/swift_vio_log.csv    t, truth, base VIO, corrected, estimated and true drift, fixes
  <log_dir>/swift_vio_events.csv every gate sighting: t_img, gate, accepted, chi2, |innovation|

Run (ros2-eval, ROS sourced, sim_clock.py running in Isaac, drone hovering):
  python3 ros2_nodes/swift_vio_node.py --ros-args -p use_sim_time:=true \\
      -p gates_file:=rl/tracks/three_gate_isaac.json -p "camera_pos:=[0.0711,-0.0003,-0.0070]" \\
      -p camera_pitch_deg:=0.0 -p seed:=1
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

from drift_filter import DriftEmulator, DriftFilter      # noqa: E402
from vio import camera_mount_usd                         # noqa: E402


def quat_to_R(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, CameraInfo
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PoseStamped, TwistStamped
    from gate_path import gates_from_manifest
    from gate_pose import estimate_gate_pose
    from color_gate_detector import detect_gates_color
    from ros_camera import intrinsics_from_camera_info

    class SwiftVioNode(Node):
        def __init__(self):
            super().__init__("swift_vio_node")
            dp = self.declare_parameter
            dp("gates_file", "/workspace/Senior-Design-G1-Autonomous-Drone/rl/tracks/three_gate_isaac.json")
            dp("pose_topic", "/drone00/state/pose")
            dp("vel_topic", "/drone00/state/twist_inertial")
            dp("image_topic", "/iris_0/front_cam/rgb")
            dp("camera_info_topic", "/iris_0/front_cam/camera_info")
            dp("odom_topic", "/vio/odometry")
            dp("base_topic", "/vio_base/odometry")
            dp("camera_pitch_deg", 0.0)
            dp("camera_pos", [0.0711, -0.0003, -0.0070])     # spawn_starling CAMERA_POS (C29 front camera)
            dp("gate_hsv_lo", [135, 60, 40])                 # magenta and Tony's pink gates
            dp("gate_hsv_hi", [179, 255, 255])
            dp("corner_sigma_px", 1.0)
            dp("bootstrap_n", 8)
            dp("correct", True)                              # false: publish the base VIO (baseline)
            dp("use_stamps", True)                           # pose time = its sim-time stamp (else arrival)
            # emulated drift (control/drift_filter.DriftEmulator)
            dp("seed", 0)
            dp("scale_std", 0.02)
            dp("bias_std", 0.005)
            dp("bias_tau_s", 5.0)
            dp("init_offset_std", 0.3)
            # filter (control/drift_filter.DriftFilter)
            dp("q_v", 0.15)
            dp("p0_std", 0.5)
            # Gate fixes from at most this far. A 0.864 m pink gate at 8-9 m is ~50 px wide and one pixel of
            # corner error is ~0.15 m in depth; the base VIO drifts slowly, so fewer, closer fixes are better
            # (9 m: 0.1-0.2 m cross-track error on the circle's turns, gate 1 hit at 5/7, 2026-11-18).
            dp("max_range", 6.0)
            dp("image_delay_s", 0.017)
            dp("log_dir", "")
            gp = lambda n: self.get_parameter(n).value   # noqa: E731
            self.gp = gp

            with open(gp("gates_file")) as f:
                manifest = json.load(f)
            gates = gates_from_manifest(gp("gates_file"))
            sides = {round(float(g.get("side", 1.5)), 4) for g in
                     (manifest["gates"] if isinstance(manifest, dict) else manifest)}
            if len(sides) != 1:
                raise ValueError(f"gates of different sizes {sorted(sides)}: one gate_side per course")
            self.gate_side = sides.pop()
            mount = camera_mount_usd(float(gp("camera_pitch_deg")), tuple(float(v) for v in gp("camera_pos")))
            self.emu = DriftEmulator(scale_std=float(gp("scale_std")), bias_std=float(gp("bias_std")),
                                     bias_tau_s=float(gp("bias_tau_s")),
                                     init_offset_std=float(gp("init_offset_std")), seed=int(gp("seed")))
            self.flt = DriftFilter(gates, mount, q_v=float(gp("q_v")), p0_std=float(gp("p0_std")),
                                   max_range=float(gp("max_range")))
            self.correct = bool(gp("correct"))
            self.lock = threading.Lock()
            self.intr = None
            self.bridge = None
            self.v_true = None
            self.v_t = -1.0
            self.prev = None
            self.n = 0
            self.cam_ms = []
            self.lags = []                                   # pose arrival - stamp, s (diagnostics)
            self.img_lags = []                               # image arrival - capture stamp, s
            self.last_status = time.monotonic()

            st, cam = MutuallyExclusiveCallbackGroup(), MutuallyExclusiveCallbackGroup()
            self.create_subscription(PoseStamped, gp("pose_topic"), self.on_pose, qos_profile_sensor_data,
                                     callback_group=st)
            self.create_subscription(TwistStamped, gp("vel_topic"), self.on_vel, qos_profile_sensor_data,
                                     callback_group=st)
            self.create_subscription(Image, gp("image_topic"), self.on_image, qos_profile_sensor_data,
                                     callback_group=cam)
            self.create_subscription(CameraInfo, gp("camera_info_topic"), self.on_info, qos_profile_sensor_data,
                                     callback_group=cam)
            self.pub = self.create_publisher(Odometry, gp("odom_topic"), 10)
            self.pub_base = self.create_publisher(Odometry, gp("base_topic"), 10)

            log_dir = gp("log_dir") or os.path.join("/workspace/swift_vio_runs" if os.path.isdir("/workspace")
                                                    else os.path.join(REPO, "swift_vio_runs"),
                                                    time.strftime("swift_%Y%m%d_%H%M%S"))
            os.makedirs(log_dir, exist_ok=True)
            self.log_dir = log_dir
            self.log_f = open(os.path.join(log_dir, "swift_vio_log.csv"), "w", newline="")
            self.log = csv.writer(self.log_f)
            self.log.writerow(["t", "x", "y", "z", "base_x", "base_y", "base_z", "est_x", "est_y", "est_z",
                               "drift_est_x", "drift_est_y", "drift_est_z", "drift_x", "drift_y", "drift_z",
                               "std_x", "std_y", "std_z", "fixes", "rejected"])
            self.get_logger().info(
                f"{len(gates)} gates (side {self.gate_side} m) from {gp('gates_file')}; drift: scale "
                f"{np.round(self.emu.scale * 100, 2)} %, velocity bias {gp('bias_std')} m/s ({gp('bias_tau_s')} s), start offset "
                f"{np.round(self.emu.p_d, 2)} m (seed {gp('seed')}); correct={self.correct}; log {log_dir}; "
                f"use_sim_time={self.get_parameter('use_sim_time').value}")

        def now(self):
            return self.get_clock().now().nanoseconds * 1e-9

        def on_info(self, msg):
            if self.intr is None:
                self.intr = intrinsics_from_camera_info(msg)
                self.get_logger().info(f"camera {msg.width}x{msg.height}, fx {self.intr.fx:.1f}")

        def stamp_time(self, msg):
            """The message's own sim-time stamp (Pegasus stamps its state on /clock once sim_clock.py has
            run), else its arrival time on this node's sim clock. Arrival time lags the pose's real time
            by the transport delay: paired with an image's capture stamp, that put every gate fix ~0.2 m
            behind at 4 m/s (2026-11-18, circle3_pink36 5/7: hit gate 1)."""
            now = self.now()
            st = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
            if bool(self.gp("use_stamps")) and st > 0 and abs(now - st) < 5.0:
                self.lags.append(now - st)
                return st
            return now

        def on_vel(self, msg):
            v = msg.twist.linear
            self.v_true, self.v_t = np.array([v.x, v.y, v.z]), self.stamp_time(msg)

        def on_pose(self, msg):
            t = self.stamp_time(msg)
            p, q = msg.pose.position, msg.pose.orientation
            pos = np.array([p.x, p.y, p.z])
            R = quat_to_R(q.x, q.y, q.z, q.w)
            # /clock advances once per rendered frame, so several poses can share t: all are published (the
            # controller wants the newest pose), the drift advances at the clock ticks.
            if self.v_true is not None and t - self.v_t < 0.1:
                vel = self.v_true
            elif self.prev is not None and t > self.prev[0]:
                vel = (pos - self.prev[1]) / (t - self.prev[0])
            else:
                vel = self.prev[2] if self.prev is not None else np.zeros(3)
            if self.prev is None or t > self.prev[0]:
                self.prev = (t, pos, vel)
            with self.lock:
                p_vio, v_vio = self.emu.step(t, pos, vel)
                self.flt.add_vio(t, p_vio, v_vio, R)
                p_c, v_c = self.flt.corrected(p_vio, v_vio)
                d_est, std = self.flt.x[:3].copy(), self.flt.position_std()
                stats = dict(self.flt.stats)
            self.n += 1
            self._publish(self.pub_base, msg.pose.orientation, R, p_vio, v_vio, None)
            if self.correct:
                self._publish(self.pub, msg.pose.orientation, R, p_c, v_c, std)
            else:
                self._publish(self.pub, msg.pose.orientation, R, p_vio, v_vio, None)
            self.log.writerow([f"{t:.4f}"] + [f"{v:.4f}" for v in (*pos, *p_vio, *p_c, *d_est, *self.emu.drift, *std)]
                              + [stats["fixes"], stats["rejected"]])
            self._status(pos, p_vio, p_c, std, stats)

        def _publish(self, pub, q, R, p, v, std):
            od = Odometry()
            od.header.stamp = self.get_clock().now().to_msg()
            od.header.frame_id, od.child_frame_id = "map", "base_link"
            od.pose.pose.position.x, od.pose.pose.position.y, od.pose.pose.position.z = (float(x) for x in p)
            od.pose.pose.orientation = q
            vb = R.T @ v                                     # REP-105: twist in the body frame
            od.twist.twist.linear.x, od.twist.twist.linear.y, od.twist.twist.linear.z = (float(x) for x in vb)
            if std is not None:
                for i in range(3):
                    od.pose.covariance[i * 7] = float(std[i] ** 2)
            pub.publish(od)

        def on_image(self, msg):
            if self.intr is None:
                return
            if self.bridge is None:
                from cv_bridge import CvBridge
                self.bridge = CvBridge()
            t0 = time.perf_counter()
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            dets = []
            for corners in detect_gates_color(rgb, tuple(int(v) for v in self.gp("gate_hsv_lo")),
                                              tuple(int(v) for v in self.gp("gate_hsv_hi"))):
                det = estimate_gate_pose(corners, self.intr, gate_side=self.gate_side,
                                         corner_sigma_px=float(self.gp("corner_sigma_px")),
                                         bootstrap_n=int(self.gp("bootstrap_n")))
                if det is not None:
                    dets.append(det)
            t_img = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
            self.img_lags.append(self.now() - t_img)
            # Isaac's rendered image shows the scene ~one render frame before its stamp: fitted 16-17 ms from
            # the error along the velocity on three live runs (2026-11-18)
            t_img -= float(self.gp("image_delay_s"))
            with self.lock:
                self.flt.add_detections(t_img, dets)
            self.cam_ms.append(1000 * (time.perf_counter() - t0))

        def _status(self, pos, p_vio, p_c, std, stats):
            now = time.monotonic()
            if now - self.last_status < 2.0:
                return
            self.last_status = now
            cam = f", camera {np.median(self.cam_ms):.0f} ms/frame" if self.cam_ms else ""
            self.cam_ms = self.cam_ms[-50:]
            if self.lags or self.img_lags:
                cam += (f", arrival lag pose {1e3 * np.median(self.lags[-200:]) if self.lags else float('nan'):.0f} ms"
                        f" image {1e3 * np.median(self.img_lags[-50:]) if self.img_lags else float('nan'):.0f} ms")
            self.get_logger().info(
                f"base VIO error {np.linalg.norm(p_vio - pos):.2f} m, corrected {np.linalg.norm(p_c - pos):.2f} m "
                f"(+-{np.round(std, 2)}), fixes {stats['fixes']}, rejected {stats['rejected']}, unmatched "
                f"{stats['unmatched']}{cam}")

        def close(self):
            self.log_f.close()
            with open(os.path.join(self.log_dir, "swift_vio_events.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t_img", "gate", "accepted", "chi2", "innovation_m"])
                w.writerows(self.flt.events)
            self.get_logger().info(f"logs in {self.log_dir}")

    rclpy.init()
    node = SwiftVioNode()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()                         # SIGINT may already have shut the context down


if __name__ == "__main__":
    main()
