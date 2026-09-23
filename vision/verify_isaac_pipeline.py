"""
Run this on the machine where ROS 2 is sourced, AFTER spawn_example.py has
spawned the drone in Isaac Sim and Pegasus's ROS2CameraGraph + PX4 SITL
bridge are both up and publishing.

It is a plain ROS 2 node, not something to paste into the Isaac Sim script
editor: it only needs the topics and TF Pegasus is already publishing, and
rclpy is not guaranteed to be importable from inside Isaac Sim's own
interpreter on every container image anyway.

WHAT IT SETTLES:
    Whether the `front_cam` TF frame Pegasus publishes uses the ROS optical
    convention (+X right, +Y down, +Z forward -- same as OpenCV) or the
    USD/body convention (+Y up, -Z forward). ros_camera.py cannot answer
    this offline; it needs a live camera pose and a point of known world
    position, which is exactly what this script supplies.

HOW TO USE IT:
    1. Put an object at a KNOWN position in the Isaac Sim stage -- a gate,
       a cube, anything whose world-frame XYZ you can read off the Stage
       properties panel.
    2. Find the parent frame `front_cam` is published relative to:
           ros2 run tf2_tools view_frames
       (or try one with `ros2 run tf2_ros tf2_echo <candidate> front_cam`.)
    3. Run:
           python3 verify_isaac_pipeline.py --world-point X Y Z --world-frame world
    4. It prints BOTH interpretations and writes annotated_optical.png /
       annotated_usd.png -- the live RGB frame with the projected pixel
       marked, under each convention. Open both; the one where the mark
       lands on the actual object is correct.

EXPECTED RESULT -- read this before running it:
    The wrong convention usually does not just mis-place the pixel, it
    puts the point BEHIND the camera (p_cam.z <= 0), because the two
    conventions disagree on which axis is forward. So most of the time you
    can tell which is right from the printed in_front_of_camera flag
    alone, before opening either image:
        - exactly one of the two runs should print in_front_of_camera=True
          and in_image=True
        - that is the correct optical_frame value -- hardcode it into
          every future call to T_world_cam_from_transform() / GatePoseNode
          and delete the "unverified" warnings in ros_camera.py's docstrings
    If BOTH come back in_front_of_camera=True (possible if the object is
    also symmetric-ish relative to the camera), fall back to the annotated
    images: one mark will land on the object, the other will be clearly
    mirrored or offset well past plausible TF/detector noise.

WHAT A CLEAN RUN LOOKS LIKE, roughly:

    --- optical_frame=True  [OPTICAL (ROS) convention] ---
      point_cam: [ 0.05 -0.10  6.80]
      pixel: [652.3 301.1]
      in_front_of_camera: True
      in_image: True
      saved annotated_optical.png -- open it and check whether the mark is on the object

    --- optical_frame=False  [USD/body convention] ---
      point_cam: [ 0.05  0.10 -6.80]
      pixel: [nan nan]
      in_front_of_camera: False
      in_image: False
      saved annotated_usd.png -- open it and check whether the mark is on the object

That pattern -- one branch in front with a plausible on-image pixel, the
other behind the camera with a NaN pixel -- is what a correct camera.py /
ros_camera.py chain should produce; it is not a foregone conclusion, it is
the thing you are checking for.
"""

import argparse
import sys
import time

import numpy as np
import cv2

from camera import project_points
from ros_camera import (T_world_cam_from_transform, intrinsics_from_camera_info,
                        verify_optical_convention)


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--world-point", type=float, nargs=3, required=True,
                    metavar=("X", "Y", "Z"),
                    help="World-frame XYZ (meters) of a known object, e.g. a gate center")
    ap.add_argument("--world-frame", default="world",
                    help="TF frame the world point is expressed in (default: world)")
    ap.add_argument("--camera-frame", default="front_cam",
                    help="TF frame published for the camera (default: front_cam)")
    ap.add_argument("--namespace", default="/iris_0/front_cam",
                    help="Topic namespace for rgb/camera_info (default: /iris_0/front_cam)")
    ap.add_argument("--timeout", type=float, default=10.0,
                    help="Seconds to wait for camera_info, image and TF (default: 10)")
    args = ap.parse_args()

    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image, CameraInfo
        from tf2_ros import Buffer, TransformListener
        from cv_bridge import CvBridge
    except ImportError as e:
        sys.exit(f"this script needs ROS 2 sourced (rclpy/tf2_ros/cv_bridge): {e}")

    rclpy.init()
    node = Node("verify_isaac_pipeline")
    bridge = CvBridge()
    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)

    state = {"info": None, "rgb": None}
    node.create_subscription(CameraInfo, f"{args.namespace}/camera_info",
                             lambda m: state.__setitem__("info", m), 10)
    node.create_subscription(Image, f"{args.namespace}/rgb",
                             lambda m: state.__setitem__("rgb", m), 10)

    deadline = time.time() + args.timeout

    def ready():
        return (state["info"] is not None and state["rgb"] is not None
                and tf_buffer.can_transform(args.world_frame, args.camera_frame,
                                            rclpy.time.Time()))

    while rclpy.ok() and time.time() < deadline and not ready():
        rclpy.spin_once(node, timeout_sec=0.2)

    if state["info"] is None:
        sys.exit(f"no camera_info on {args.namespace}/camera_info in {args.timeout}s -- "
                 f"is spawn_example.py running and PX4 SITL connected?")
    if state["rgb"] is None:
        sys.exit(f"no image on {args.namespace}/rgb in {args.timeout}s")
    if not tf_buffer.can_transform(args.world_frame, args.camera_frame, rclpy.time.Time()):
        sys.exit(f"no TF from {args.world_frame} to {args.camera_frame} in {args.timeout}s -- "
                 f"run `ros2 run tf2_tools view_frames` to find the real frame names and "
                 f"pass --world-frame/--camera-frame")

    intr = intrinsics_from_camera_info(state["info"])
    tf = tf_buffer.lookup_transform(args.world_frame, args.camera_frame,
                                    rclpy.time.Time()).transform
    rgb = bridge.imgmsg_to_cv2(state["rgb"], desired_encoding="rgb8")
    world_point = np.array(args.world_point)

    print(f"camera_info: {intr.width}x{intr.height}, fx={intr.fx:.1f} fy={intr.fy:.1f}")
    if intr.width <= 320:
        print(f"  NOTE: {intr.width}x{intr.height} is low resolution for range accuracy "
              f"at distance -- a 1.5 m gate at 12 m spans only ~{intr.fx*1.5/12:.0f} px here")

    for optical_frame, out_path, label in [
        (True, "annotated_optical.png", "OPTICAL (ROS) convention"),
        (False, "annotated_usd.png", "USD/body convention"),
    ]:
        T_world_cam = T_world_cam_from_transform(tf, optical_frame)
        result = verify_optical_convention(T_world_cam, intr, world_point)
        print(f"\n--- optical_frame={optical_frame}  [{label}] ---")
        for k, v in result.items():
            print(f"  {k}: {v}")

        frame = rgb.copy()
        if result["in_image"]:
            u, v = int(round(result["pixel"][0])), int(round(result["pixel"][1]))
            cv2.drawMarker(frame, (u, v), (255, 0, 0), cv2.MARKER_CROSS, 24, 2)
        cv2.imwrite(out_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"  saved {out_path} -- open it and check whether the mark is on the object")

    print("\nWhichever run above showed in_front_of_camera=True and in_image=True "
          "(with a mark that visually lands on the object) is the correct optical_frame "
          "value. Hardcode it into every future T_world_cam_from_transform()/GatePoseNode "
          "call and remove the 'unverified' warnings from ros_camera.py's docstrings.")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
