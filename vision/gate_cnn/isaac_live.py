"""Run the gate CNN live on the Isaac Sim front camera.

Two ways to hook it up, depending on how your Isaac setup delivers camera images:

A) Inside the team's existing vision code (simplest, recommended)
   Wherever gate_vision.py currently gets a frame and calls the colour detector, call
   process_frame(frame_bgr, K) instead. It returns the gate position in the camera frame,
   which feeds GateMap exactly like before.

B) As a separate ROS 2 node, if the camera is published on a ROS 2 topic
       python isaac_live.py --topic /front_cam/rgb --save isaac_raw/
   Shows the camera with detections, prints gate positions, and (with --save) stores
   raw frames for label_isaac.py. Change --topic to whatever `ros2 topic list` shows.

K (camera matrix) must match the Isaac camera. If you know the image size and the
horizontal FOV, config.intrinsics(w, h, (hfov, vfov)) builds it.
"""
import argparse
import os

import cv2
import numpy as np

import config as C
from detector import GateCNN, refine_with_colour, estimate_pose, good_detection

_cnn = None


def process_frame(frame_bgr, K, conf_min=0.45):
    """One camera frame -> (found, t_cam, corners).
    t_cam: gate centre in the camera frame (OpenCV axes x right, y down, z forward), metres."""
    global _cnn
    if _cnn is None:
        _cnn = GateCNN(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gate_cnn.npz"))
    corners, conf = _cnn.detect(frame_bgr)
    if not good_detection(frame_bgr, corners, conf, conf_min):
        return False, None, corners
    corners = refine_with_colour(frame_bgr, corners)
    ok, t, _ = estimate_pose(corners, K)
    return bool(ok), t, corners


def ros_main():
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge

    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/front_cam/rgb")
    ap.add_argument("--save", default=None, help="folder to save raw frames (every 5th)")
    a = ap.parse_args()
    if a.save:
        os.makedirs(a.save, exist_ok=True)

    class GateNode(Node):
        def __init__(self):
            super().__init__("gate_cnn")
            self.bridge = CvBridge()
            self.n = 0
            self.create_subscription(Image, a.topic, self.cb, 10)

        def cb(self, msg):
            f = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            K = C.intrinsics(f.shape[1], f.shape[0])
            if a.save and self.n % 5 == 0:
                cv2.imwrite(os.path.join(a.save, f"frame_{self.n:06d}.png"), f)
            found, t, c = process_frame(f, K)
            v = f.copy()
            if found:
                cv2.polylines(v, [np.round(c).astype(np.int32)], True, (80, 255, 80), 2, cv2.LINE_AA)
                self.get_logger().info(f"gate at x={t[0]:+.2f} y={t[1]:+.2f} z={t[2]:.2f} m (camera frame)")
            cv2.imshow("gate CNN", v)
            cv2.waitKey(1)
            self.n += 1

    rclpy.init()
    rclpy.spin(GateNode())


if __name__ == "__main__":
    ros_main()
