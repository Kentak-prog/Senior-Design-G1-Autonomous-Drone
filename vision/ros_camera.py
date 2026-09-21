"""
ROS 2 adapter: sensor_msgs/CameraInfo and tf2 Transform -> the camera.py /
gate_pose.py convention, plus the rclpy node that wires gate detection into
a published PoseArray.

THIS MODULE MUST IMPORT WITHOUT ROS INSTALLED. Every rclpy / cv_bridge name
is imported lazily, inside the function or method that needs it, so that
intrinsics_from_camera_info(), quat_xyzw_to_wxyz(), transform_to_matrix(),
T_world_cam_from_transform() and verify_optical_convention() are all plain
numpy/cv2 and fully testable on a machine with no ROS whatsoever (see
test_ros_camera.py). Only instantiating GatePoseNode requires rclpy.

TWO THINGS IN THIS FILE ARE UNVERIFIED IN THE ACTUAL SIMULATOR AND MUST BE
CHECKED BEFORE TRUSTING DOWNSTREAM OUTPUT:

  1. Quaternion component order. ROS messages (geometry_msgs/Quaternion,
     tf2) are always (x, y, z, w). camera.quat_to_rot expects (w, x, y, z).
     Feeding one straight into the other silently produces a wrong-but-
     plausible-looking rotation -- see quat_xyzw_to_wxyz()'s docstring.

  2. Which frame convention the published TF transform actually uses. See
     T_world_cam_from_transform()'s docstring and verify_optical_convention()
     for the empirical check.
"""

from typing import Callable, Sequence

import numpy as np

from camera import (CameraIntrinsics, make_transform, invert_transform,
                    transform_points, project_points, in_frame,
                    quat_to_rot)
from gate_pose import (estimate_gate_pose, gate_pose_world, approach_waypoints,
                       DEFAULT_GATE_SIDE)


def _get(msg, lower: str, upper: str):
    """
    Fetch a field off msg by either spelling, duck-typed.

    Accepts a real sensor_msgs/CameraInfo, a plain dict, or any object with
    the attribute. ROS 2's rclpy generates lowercase field names (k, d, p,
    width, height); a lot of ROS 1 code, tutorials and hand-built test
    fixtures use uppercase (K, D, P). Rather than force callers to know
    which, try both.
    """
    if isinstance(msg, dict):
        if lower in msg:
            return msg[lower]
        if upper in msg:
            return msg[upper]
        raise KeyError(f"neither {lower!r} nor {upper!r} in camera_info dict")
    if hasattr(msg, lower):
        return getattr(msg, lower)
    if hasattr(msg, upper):
        return getattr(msg, upper)
    raise AttributeError(f"camera_info has neither .{lower} nor .{upper}")


def intrinsics_from_camera_info(msg, prefer_projection: bool = False) -> CameraIntrinsics:
    """
    sensor_msgs/CameraInfo (or a dict/duck-typed stand-in) -> CameraIntrinsics.

    msg.k (or .K) is the flat row-major 3x3 intrinsic matrix
    [fx, 0, cx, 0, fy, cy, 0, 0, 1]. msg.d (or .D) is the distortion vector,
    padded with zeros or truncated to exactly 5 elements (k1, k2, p1, p2, k3)
    to match CameraIntrinsics.distortion / cv2's expected layout.

    prefer_projection reads fx, fy, cx, cy out of msg.p (or .P) instead --
    the flat 3x4 row-major projection matrix [fx,0,cx,Tx, 0,fy,cy,Ty, 0,0,1,0].
    P is what to use once the image has already been rectified (e.g. by
    image_proc or an equivalent step upstream): K describes the raw,
    unrectified lens, and projecting with K against a rectified image is
    subtly wrong -- it ignores the rectification's re-centering and any
    stereo baseline term. Isaac's ROS2CameraGraph publishes an ideal pinhole
    with no distortion, so K and P should already agree in simulation; keep
    prefer_projection=False (the default) unless something downstream is
    rectifying the image, and re-check this once real camera hardware with
    real lens distortion is in the loop.
    """
    width = int(_get(msg, "width", "width"))
    height = int(_get(msg, "height", "height"))

    if prefer_projection:
        p = np.asarray(_get(msg, "p", "P"), dtype=float).reshape(3, 4)
        fx, cx = p[0, 0], p[0, 2]
        fy, cy = p[1, 1], p[1, 2]
    else:
        k = np.asarray(_get(msg, "k", "K"), dtype=float).reshape(3, 3)
        fx, cx = k[0, 0], k[0, 2]
        fy, cy = k[1, 1], k[1, 2]

    d = np.asarray(_get(msg, "d", "D"), dtype=float).ravel()
    dist = np.zeros(5)
    n = min(5, d.size)
    dist[:n] = d[:n]

    return CameraIntrinsics(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
                            width=width, height=height, distortion=dist)


def quat_xyzw_to_wxyz(q) -> np.ndarray:
    """
    ROS quaternion ordering (x, y, z, w) -> the (w, x, y, z) ordering that
    camera.quat_to_rot expects.

    THIS IS THE SINGLE EASIEST SILENT BUG IN THIS WHOLE PIPELINE. Every ROS
    message that carries a quaternion (geometry_msgs/Quaternion, tf2
    TransformStamped.transform.rotation, ...) is (x, y, z, w). Isaac Sim's
    own get_world_pose() and everything in camera.py/gate_pose.py is
    (w, x, y, z). Swap them and quat_to_rot() still returns *a* valid
    rotation matrix -- unit norm, orthogonal, no exception anywhere -- it is
    just the wrong rotation, and it can look plausible enough (e.g. for
    quaternions near identity) to pass a casual glance. Always go through
    this function, never hand a ROS quaternion to quat_to_rot() directly.

    Accepts anything with .x/.y/.z/.w attributes (a geometry_msgs.Quaternion)
    or a length-4 sequence in (x, y, z, w) order.
    """
    if hasattr(q, "x") and hasattr(q, "w"):
        x, y, z, w = q.x, q.y, q.z, q.w
    else:
        x, y, z, w = np.asarray(q, dtype=float).ravel()
    return np.array([w, x, y, z], dtype=float)


def transform_to_matrix(translation, rotation_xyzw) -> np.ndarray:
    """
    A ROS-style rigid transform -> 4x4 homogeneous matrix, via camera.make_transform.

    translation accepts a geometry_msgs/Vector3-like object (.x/.y/.z) or a
    length-3 sequence. rotation_xyzw accepts a geometry_msgs/Quaternion-like
    object (.x/.y/.z/.w) or a length-4 (x, y, z, w) sequence -- ROS ordering,
    converted internally via quat_xyzw_to_wxyz before it ever reaches
    quat_to_rot. Typical caller: a geometry_msgs/TransformStamped's
    `.transform.translation` and `.transform.rotation` fields.
    """
    if hasattr(translation, "x"):
        t = np.array([translation.x, translation.y, translation.z], dtype=float)
    else:
        t = np.asarray(translation, dtype=float).ravel()
    R = quat_to_rot(quat_xyzw_to_wxyz(rotation_xyzw))
    return make_transform(R, t)


def _rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """
    3x3 rotation -> quaternion in ROS (x, y, z, w) ordering, via Shepperd's
    method. The mirror image of camera.quat_to_rot: this module only ever
    hands ROS messages a quaternion in ROS ordering, never camera.py's
    (w, x, y, z) internal one -- see quat_xyzw_to_wxyz()'s docstring for why
    that boundary matters.
    """
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


def T_world_cam_from_transform(transform, optical_frame: bool) -> np.ndarray:
    """
    A world->camera TF lookup -> T_world_cam in the OpenCV convention that
    gate_pose.py expects (see camera.py's module docstring).

    optical_frame has NO default on purpose: which convention the published
    TF frame actually uses is not something to silently guess at, and
    getting it wrong flips the sign of the camera's forward/up axes without
    raising any error downstream.

    optical_frame=True:  the TF frame already follows the ROS optical-frame
        convention (+X right, +Y down, +Z forward) -- which IS the OpenCV
        convention camera.py uses -- so `transform` is returned unchanged
        (just converted to a 4x4).

    optical_frame=False: the TF frame follows the USD/Isaac body convention
        (+X right, +Y up, -Z forward), and this applies the same correction
        camera.usd_camera_pose_to_cv() applies to a USD camera pose
        (R_world_cv = R_world_usd @ R_CV_FROM_USD.T).

    WHICH ONE IS CORRECT HERE IS UNVERIFIED. ROS image pipelines
    conventionally publish camera TF under a frame_id ending in
    `_optical_frame` specifically to flag "this one is already OpenCV/REP-103
    optical convention"; every other TF frame in a ROS system is expected to
    follow the REP-103 body convention (+X forward, +Y left, +Z up) or, for
    USD/Isaac-native prims, the USD camera convention this module treats
    optical_frame=False as. But workspace/spawn_example.py configures
    ROS2CameraGraph with `"tf_frame_id": "front_cam"` -- no `_optical_frame`
    suffix -- so Pegasus's ROS2CameraGraph may or may not be following that
    naming convention faithfully. DO NOT TRUST either branch until you have
    run verify_optical_convention() (or isaac_camera.debug_check_projection
    equivalent) against a live simulator with a gate at a known world
    position, with both optical_frame=True and optical_frame=False, and
    confirmed which one puts the gate at a plausible pixel.
    """
    T = transform if isinstance(transform, np.ndarray) else transform_to_matrix(
        transform.translation, transform.rotation)
    T = np.asarray(T, dtype=float)
    if optical_frame:
        return T
    R_world_usd = T[:3, :3]
    position = T[:3, 3]
    # Re-derive via usd_camera_pose_to_cv so this stays byte-for-byte
    # consistent with how isaac_camera.py converts a USD camera pose --
    # quat_to_rot round-trips a rotation matrix back to a quaternion-free
    # correction by applying the same R_CV_FROM_USD directly.
    from camera import R_CV_FROM_USD
    R_world_cv = R_world_usd @ R_CV_FROM_USD.T
    return make_transform(R_world_cv, position)


def verify_optical_convention(T_world_cam: np.ndarray, intr: CameraIntrinsics,
                              world_point) -> dict:
    """
    Empirical check for T_world_cam_from_transform()'s optical_frame guess.

    Not a comment, a runnable test: point the drone at a gate whose world
    position you know, look up the camera TF, build T_world_cam with
    optical_frame=True, call this, then rebuild it with optical_frame=False
    and call this again. Exactly one of the two runs should report
    in_front_of_camera=True and in_image=True with a pixel that visually
    lands on the gate in the corresponding saved frame; the other will
    either put the gate behind the camera or project it somewhere absurd.
    Whichever one looks right is the convention Pegasus is actually
    publishing -- hardcode that as the optical_frame argument everywhere
    else and delete the other branch's uncertainty from the docstrings above.

    Returns a dict: point_cam (3,), pixel (2,) [nan, nan] if undefined,
    in_front_of_camera (bool, z_cam > 0), in_image (bool).
    """
    p_world = np.asarray(world_point, dtype=float).reshape(1, 3)
    T_cam_world = invert_transform(T_world_cam)
    p_cam = transform_points(T_cam_world, p_world)
    pixel = project_points(p_cam, intr)
    in_front = bool(p_cam[0, 2] > 0)
    ok = in_frame(pixel, intr)
    return {
        "point_cam": p_cam[0],
        "pixel": pixel[0],
        "in_front_of_camera": in_front,
        "in_image": bool(ok[0]),
    }


class GatePoseNode:
    """
    rclpy node: rgb image + camera_info + tf2 camera pose -> published gate
    poses.

    ALL rclpy / cv_bridge / tf2_ros imports happen lazily inside __init__ and
    the callbacks below, so importing this module (and every function above
    it) never requires ROS to be installed -- see the module docstring and
    test_ros_camera.py, which exercises everything except this class with no
    ROS present at all.

    detector: Callable[[np.ndarray], Sequence[np.ndarray]] mapping one HxWx3
        RGB image to a sequence of per-gate corner arrays, each 4x2 pixel
        coordinates -- the CORNERS of the gate opening, explicitly NOT a
        bounding box. This is the contract with gate detection (Tye's
        module): estimate_gate_pose() needs the four physical corners to
        solve a PnP problem, and a bounding box cannot recover the gate's
        orientation at all, only a rough range from apparent size. Corners
        may be in any order; order_corners() (called inside
        estimate_gate_pose) sorts them.
    namespace: ROS namespace the camera publishes under, e.g. "/iris_0"
        (matches workspace/spawn_example.py's ROS2CameraGraph "namespace"
        config). Subscribes under f"{namespace}/front_cam/...".
    optical_frame: forwarded verbatim to T_world_cam_from_transform() --
        see that function's docstring for why this has no default and why
        it is UNVERIFIED against the live simulator.
    gate_side: physical gate opening side length in meters, forwarded to
        estimate_gate_pose / gate_model_points.
    corner_sigma_px: detector corner-localization uncertainty in pixels,
        forwarded to estimate_gate_pose for its bootstrap uncertainty
        estimate (see gate_pose.py).
    """

    def __init__(self, detector: Callable[[np.ndarray], Sequence[np.ndarray]],
                namespace: str = "/iris_0",
                optical_frame: bool = True,
                gate_side: float = DEFAULT_GATE_SIDE,
                corner_sigma_px: float = 1.5,
                camera_frame: str = "front_cam",
                world_frame: str = "world",
                max_reprojection_error_px: float = 15.0):
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image, CameraInfo
        from geometry_msgs.msg import PoseArray
        import tf2_ros

        self._detector = detector
        self._namespace = namespace
        self._optical_frame = optical_frame
        self._gate_side = gate_side
        self._corner_sigma_px = corner_sigma_px
        self._camera_frame = camera_frame
        self._world_frame = world_frame
        self._max_reprojection_error_px = max_reprojection_error_px

        self._node = Node("gate_pose_node")
        self._intr = None
        self._bridge = None  # built lazily on first image callback

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self._node)

        topic = f"{namespace}/front_cam"
        self._pub = self._node.create_publisher(PoseArray, f"{topic}/gate_poses", 10)
        self._node.create_subscription(CameraInfo, f"{topic}/camera_info",
                                       self._on_camera_info, 10)
        self._node.create_subscription(Image, f"{topic}/rgb", self._on_image, 10)

    @property
    def node(self):
        """The underlying rclpy Node, for spin()/spin_once() by the caller."""
        return self._node

    def _on_camera_info(self, msg) -> None:
        self._intr = intrinsics_from_camera_info(msg)

    def _on_image(self, msg) -> None:
        """
        Image callback. Kept lean: convert, look up TF, detect, solve per
        detection, publish. Any per-detection failure is logged and skipped
        rather than raised, so one bad gate never drops the whole frame.
        """
        if self._intr is None:
            self._node.get_logger().warn("gate_pose_node: no camera_info yet, skipping frame")
            return

        if self._bridge is None:
            from cv_bridge import CvBridge
            self._bridge = CvBridge()
        rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

        try:
            transform = self._tf_buffer.lookup_transform(
                self._world_frame, self._camera_frame, msg.header.stamp)
        except Exception as exc:  # tf2 raises several distinct exception types
            self._node.get_logger().warn(f"gate_pose_node: tf lookup failed: {exc}")
            return
        T_world_cam = T_world_cam_from_transform(transform.transform, self._optical_frame)

        detections = self._detector(rgb)

        from geometry_msgs.msg import PoseArray, Pose
        out = PoseArray()
        out.header = msg.header
        out.header.frame_id = self._world_frame

        for corners in detections:
            det = estimate_gate_pose(corners, self._intr, self._gate_side,
                                     corner_sigma_px=self._corner_sigma_px)
            if det is None:
                self._node.get_logger().debug("gate_pose_node: solvePnP failed, skipping")
                continue
            if det.reprojection_error_px > self._max_reprojection_error_px:
                self._node.get_logger().debug(
                    f"gate_pose_node: reprojection error {det.reprojection_error_px:.1f}px "
                    f"> {self._max_reprojection_error_px}px, skipping")
                continue

            T_world_gate = gate_pose_world(det, T_world_cam)
            center = T_world_gate[:3, 3]

            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = center.tolist()
            # Orientation only carries meaning when the normal is reliable;
            # otherwise publish identity rather than a noisy rotation --
            # see GateDetection.normal_is_reliable in gate_pose.py for why
            # the normal can be unreliable near head-on even when the
            # center is fine.
            if det.normal_is_reliable:
                qx, qy, qz, qw = _rot_to_quat_xyzw(T_world_gate[:3, :3])
                pose.orientation.x, pose.orientation.y = qx, qy
                pose.orientation.z, pose.orientation.w = qz, qw
                # approach_waypoints() is available to callers that want the
                # standoff points too; not part of PoseArray's Pose shape,
                # so left to whoever consumes this node's other outputs.
                approach_waypoints(T_world_gate)
            else:
                pose.orientation.w = 1.0
                self._node.get_logger().debug(
                    "gate_pose_node: normal unreliable, publishing center only")

            out.poses.append(pose)

        self._pub.publish(out)
