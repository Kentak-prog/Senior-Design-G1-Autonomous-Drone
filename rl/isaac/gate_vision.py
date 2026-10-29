"""
Camera -> world-frame gate detections for the runner (vision mode).

    process_frame(rgb, intr, T_world_body, mount)   pure: one image -> detections
    RosGateVision                                   live: rclpy subscriptions + worker thread
    SimGateVision                                   offline stand-in (no images, geometry only)
    GateSource                                      GateMap + one of the above, as the runner uses it

Everything heavy is REUSED from vision/ (this file only wires it together):
detect_gates_color (magenta opening corners), estimate_gate_pose (SQPNP + LM,
bootstrap for normal_is_reliable), gate_pose_world, T_world_cam_from_transform
(optical_frame=False, settled live 2026-09-28), intrinsics_from_camera_info,
eval_isaac_gates.rotate_xyz_deg_to_matrix. rclpy is imported lazily inside
RosGateVision, so the rest imports (and is tested) without ROS.

CAMERA POSE. This sim publishes no /tf. Camera pose =
    T_world_body (from /drone00/state/pose, PoseStamped, frame map = ENU,
    quaternion xyzw) @ T_body_cam_usd (the static USD-convention mount from
    spawn_example.py: translate (0.30, 0, 0.05), rotateXYZ (105, 0, -90)),
then T_world_cam_from_transform(.., optical_frame=False). The mount is a
parameter because the live camera prim was once found at (-0.040, 0, -0.041)
(vision/HANDOFF.md, "Camera mount moved"): check it before trusting a run.

TIME / LATENCY (RosGateVision). Image stamps are SIM time, pose stamps are WALL
time, so they cannot be paired by stamp. Each image is paired with the LATEST
pose received when the image callback runs. The pose is therefore newer than
the image by the image transport + queueing delay (tens of ms; unmeasured
here). A camera that moved d metres in that time puts the gate d off, in the
world frame: about 0.04 m per 30 ms at 1.5 m/s and 0.3 m per 30 ms at 10 m/s,
plus the same delay in attitude (0.2 deg per 10 ms at 20 deg/s, which is
~4 cm at 10 m range). Detections are stamped with the IMAGE header time (sim
clock), which is the clock the runner's states use.

PROCESSING. The executor thread only stores the newest image (+ the pose at
that moment); a worker thread processes the newest pending image and drops
the rest ("never queue"), so the pose callbacks are never blocked by PnP.

STALE camera_info (vision/HANDOFF.md, "RESOLVED 2026-10-01"): if the
camera_info size differs from the image size the intrinsics are wrong; frames
are SKIPPED with a loud one-time warning (counted in stats) unless
hfov_override is given (intrinsics rebuilt from the IMAGE size, like
eval_gate_bag --hfov-deg) or allow_stale_info=True.
"""

import math
import os
import sys
import threading
import time
from collections import deque

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RL = os.path.dirname(_HERE)
_VISION = os.path.join(os.path.dirname(_RL), "vision")
for _p in (_HERE, _RL):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if _VISION not in sys.path:
    sys.path.append(_VISION)

from camera import CameraIntrinsics, make_transform, invert_transform  # noqa: E402
from gate_pose import estimate_gate_pose, gate_pose_world, gate_model_points  # noqa: E402
from color_gate_detector import detect_gates_color  # noqa: E402
from ros_camera import (T_world_cam_from_transform, intrinsics_from_camera_info,  # noqa: E402
                        transform_to_matrix)
from eval_isaac_gates import rotate_xyz_deg_to_matrix  # noqa: E402
from gate_map import Detection  # noqa: E402

DEFAULT_MOUNT_XYZ = (0.30, 0.0, 0.05)
DEFAULT_MOUNT_ROT_XYZ_DEG = (105.0, 0.0, -90.0)
GATE_SIDE = 1.5


def mount_matrix(xyz=DEFAULT_MOUNT_XYZ, rot_xyz_deg=DEFAULT_MOUNT_ROT_XYZ_DEG):
    """T_body_cam in the USD camera convention (as eval_isaac_gates builds it)."""
    return make_transform(rotate_xyz_deg_to_matrix(*rot_xyz_deg), np.asarray(xyz, float))


def T_world_body_from_state(s):
    """VehicleState (ENU pos, WXYZ FLU->world quat) -> 4x4."""
    from camera import quat_to_rot
    return make_transform(quat_to_rot(s.quat), s.pos)


def T_world_cam_cv(T_world_body, mount):
    """World -> OpenCV camera frame (what gate_pose_world wants)."""
    return T_world_cam_from_transform(np.asarray(T_world_body) @ np.asarray(mount),
                                      optical_frame=False)


# --------------------------------------------------------------------------
# the pure per-frame function
# --------------------------------------------------------------------------
def process_frame(rgb, intr, T_world_body, mount, *, t=0.0, gate_side=GATE_SIDE,
                  max_reproj_px=4.0, max_range=25.0, border_px=3, corner_sigma_px=1.5,
                  bootstrap_n=12, min_area_px=64, detector=None, info=None):
    """One RGB image (HxWx3 uint8, RGB order) -> list of Detection (world frame).

    Per detected gate opening: PnP -> reprojection RMS must be <= max_reproj_px
    (else dropped: occluded / clipped / false positive), range <= max_range,
    and no corner within border_px of the image edge (a clipped opening gives a
    biased pose). normal_reliable comes from the PnP bootstrap
    (corner_sigma_px > 0 is required for it, else it is always False).
    `info`, if a dict, receives counts: n_corners, rej_reproj, rej_range,
    rej_border, rej_pnp."""
    detector = detector if detector is not None else (
        lambda im: detect_gates_color(im, min_area_px=min_area_px))
    cnt = dict(n_corners=0, rej_reproj=0, rej_range=0, rej_border=0, rej_pnp=0)
    T_world_cam = T_world_cam_cv(T_world_body, mount)
    out = []
    for corners in detector(rgb):
        cnt["n_corners"] += 1
        c = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        if (c[:, 0].min() < border_px or c[:, 1].min() < border_px
                or c[:, 0].max() > intr.width - 1 - border_px
                or c[:, 1].max() > intr.height - 1 - border_px):
            cnt["rej_border"] += 1
            continue
        det = estimate_gate_pose(c, intr, gate_side, corner_sigma_px=corner_sigma_px,
                                 bootstrap_n=bootstrap_n)
        if det is None:
            cnt["rej_pnp"] += 1
            continue
        if det.reprojection_error_px > max_reproj_px:
            cnt["rej_reproj"] += 1
            continue
        if det.range_m > max_range:
            cnt["rej_range"] += 1
            continue
        out.append(Detection(gate_pose_world(det, T_world_cam), bool(det.normal_is_reliable),
                             float(t), float(det.range_m)))
    if info is not None:
        info.update(cnt)
    return out


def image_to_rgb(msg):
    """sensor_msgs/Image -> HxWx3 uint8 RGB without cv_bridge for the common
    encodings (rgb8, bgr8, rgba8, bgra8); cv_bridge for anything else."""
    enc = msg.encoding.lower()
    ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(enc)
    if ch is None:
        from cv_bridge import CvBridge
        return CvBridge().imgmsg_to_cv2(msg, desired_encoding="rgb8")
    h, w = int(msg.height), int(msg.width)
    a = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, int(msg.step))[:, :w * ch]
    a = a.reshape(h, w, ch)
    if enc.startswith("bgr"):
        a = a[..., [2, 1, 0, *range(3, ch)]]
    return np.ascontiguousarray(a[..., :3])


# --------------------------------------------------------------------------
# live path
# --------------------------------------------------------------------------
class RosGateVision:
    """rclpy subscriptions + worker thread; see the module docstring.

    drain() returns the Detections produced since the last call. stats()
    returns counters. close() stops everything (and shuts rclpy down only if
    this object initialised it). Topics: <namespace>/<cam_name>/{rgb,
    camera_info}, pose_topic. ALL subscriptions use qos_profile_sensor_data
    (best-effort): the pose publisher is best-effort (a default RELIABLE
    subscription silently receives nothing); for the camera topics
    eval_isaac_gates uses the default reliable depth-10 profile, and a
    best-effort subscriber also connects to a reliable publisher, so this is
    the safe superset."""

    def __init__(self, namespace="/iris_0", pose_topic="/drone00/state/pose", cam_name="front_cam",
                 mount_xyz=DEFAULT_MOUNT_XYZ, mount_rot_xyz_deg=DEFAULT_MOUNT_ROT_XYZ_DEG,
                 hfov_override=None, allow_stale_info=False, gate_side=GATE_SIDE,
                 max_reproj_px=4.0, max_range=25.0, border_px=3, corner_sigma_px=1.5,
                 min_area_px=64, node_name="rl_gate_vision", say=print):
        self.mount = mount_matrix(mount_xyz, mount_rot_xyz_deg)
        self.hfov_override = hfov_override
        import cv2
        cv2.setNumThreads(1)        # 2-CPU pod quota: see runner.py (thread limits)
        self.allow_stale_info = allow_stale_info
        self.kw = dict(gate_side=gate_side, max_reproj_px=max_reproj_px, max_range=max_range,
                       border_px=border_px, corner_sigma_px=corner_sigma_px,
                       min_area_px=min_area_px)
        self.say = say
        self._lock = threading.Lock()
        self._evt = threading.Event()
        self._run = True
        self._pose_T = None          # latest T_world_body
        self._info = None            # latest CameraInfo message
        self._pending = None         # (Image msg, T_world_body at arrival)
        self._out = []
        self._intr_cache = {}
        self._warned = set()
        self._proc_ms = deque(maxlen=2000)
        self.counters = dict(frames_in=0, frames_processed=0, frames_dropped=0,
                             frames_skipped_no_pose=0, frames_skipped_no_info=0,
                             frames_skipped_stale_info=0, frames_with_dets=0, dets_total=0,
                             rej_reproj=0, rej_range=0, rej_border=0, rej_pnp=0, n_corners=0)

        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, CameraInfo
        from geometry_msgs.msg import PoseStamped
        self._rclpy = rclpy
        self._own_init = not rclpy.ok()
        if self._own_init:
            rclpy.init()
        self.node = Node(node_name)
        base = f"{namespace}/{cam_name}"
        self.node.create_subscription(PoseStamped, pose_topic, self._on_pose, qos_profile_sensor_data)
        self.node.create_subscription(CameraInfo, f"{base}/camera_info", self._on_info,
                                      qos_profile_sensor_data)
        self.node.create_subscription(Image, f"{base}/rgb", self._on_image, qos_profile_sensor_data)
        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self.node)
        self._spin_thread = threading.Thread(target=self._spin, name="gate_vision_spin", daemon=True)
        self._work_thread = threading.Thread(target=self._work, name="gate_vision_work", daemon=True)
        self._spin_thread.start()
        self._work_thread.start()

    # --- ROS callbacks (executor thread; must stay cheap) ---
    def _on_pose(self, msg):
        self._pose_T = transform_to_matrix(msg.pose.position, msg.pose.orientation)

    def _on_info(self, msg):
        self._info = msg

    def _on_image(self, msg):
        with self._lock:
            self.counters["frames_in"] += 1
            if self._pending is not None:
                self.counters["frames_dropped"] += 1     # worker busy: newest wins
            self._pending = (msg, self._pose_T)
        self._evt.set()

    def _spin(self):
        while self._run:
            try:
                self._exec.spin_once(timeout_sec=0.1)
            except Exception as e:   # shutdown races
                if self._run:
                    self.say(f"[gate_vision] spin error: {e}")
                    time.sleep(0.1)

    # --- worker ---
    def _warn_once(self, key, msg):
        if key not in self._warned:
            self._warned.add(key)
            self.say(f"[gate_vision] WARNING: {msg}")

    def _intrinsics(self, w, h):
        key = (w, h)
        if key in self._intr_cache:
            return self._intr_cache[key]
        intr = None
        if self.hfov_override is not None:
            intr = CameraIntrinsics.from_horizontal_fov(w, h, self.hfov_override)
            self.say(f"[gate_vision] hfov_override {self.hfov_override} deg: intrinsics from the "
                     f"image size {w}x{h} (fx={intr.fx:.1f}), camera_info ignored")
        elif self._info is not None:
            intr = intrinsics_from_camera_info(self._info)
            if (intr.width, intr.height) != (w, h):
                self._warn_once(("stale", w, h),
                                f"camera_info says {intr.width}x{intr.height} but the image is "
                                f"{w}x{h}: camera_info is STALE (fx/cx/cy do not describe this "
                                f"image). Refresh it in Isaac or pass hfov_override (60 for "
                                f"spawn_example.py). " + ("Using it anyway (allow_stale_info)."
                                                          if self.allow_stale_info
                                                          else "SKIPPING frames."))
                if not self.allow_stale_info:
                    return "stale"
        else:
            return None
        self._intr_cache[key] = intr
        return intr

    def _work(self):
        while self._run:
            self._evt.wait(0.1)
            with self._lock:
                item, self._pending = self._pending, None
                self._evt.clear()
            if item is None:
                continue
            msg, T_body = item
            c = self.counters
            try:
                if T_body is None:
                    c["frames_skipped_no_pose"] += 1
                    continue
                intr = self._intrinsics(int(msg.width), int(msg.height))
                if intr is None:
                    c["frames_skipped_no_info"] += 1
                    continue
                if isinstance(intr, str):
                    c["frames_skipped_stale_info"] += 1
                    continue
                t_img = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
                t0 = time.perf_counter()
                info = {}
                dets = process_frame(image_to_rgb(msg), intr, T_body, self.mount, t=t_img,
                                     info=info, **self.kw)
                self._proc_ms.append(1e3 * (time.perf_counter() - t0))
                with self._lock:
                    self._out.extend(dets)
                c["frames_processed"] += 1
                c["dets_total"] += len(dets)
                c["frames_with_dets"] += 1 if dets else 0
                for k in ("n_corners", "rej_reproj", "rej_range", "rej_border", "rej_pnp"):
                    c[k] += info[k]
            except Exception as e:   # one bad frame must not kill the thread
                self._warn_once(("err", type(e).__name__), f"frame processing failed: {e!r}")

    # --- public ---
    def wait_ready(self, timeout=10.0):
        """Wait for a pose and (camera_info or hfov_override) and an image.
        Returns True when frames are being processed, else prints what is missing."""
        t_end = time.time() + timeout
        while time.time() < t_end:
            if self.counters["frames_processed"] > 0:
                return True
            time.sleep(0.1)
        c = self.counters
        self.say(f"[gate_vision] NOT READY after {timeout:.0f} s: pose={'yes' if self._pose_T is not None else 'NO'} "
                 f"camera_info={'yes' if self._info is not None else 'NO'} frames_in={c['frames_in']} "
                 f"skipped(no pose/no info/stale)={c['frames_skipped_no_pose']}/"
                 f"{c['frames_skipped_no_info']}/{c['frames_skipped_stale_info']}")
        return False

    def drain(self):
        with self._lock:
            out, self._out = self._out, []
        return out

    def stats(self):
        c = dict(self.counters)
        ms = list(self._proc_ms)
        c["proc_ms_mean"] = float(np.mean(ms)) if ms else float("nan")
        c["proc_ms_max"] = float(np.max(ms)) if ms else float("nan")
        c["dets_per_frame"] = c["dets_total"] / c["frames_processed"] if c["frames_processed"] else float("nan")
        return c

    def close(self):
        self._run = False
        self._evt.set()
        for th in (self._spin_thread, self._work_thread):
            th.join(timeout=1.0)
        try:
            self._exec.shutdown()
            self.node.destroy_node()
            if self._own_init:
                self._rclpy.shutdown()
        except Exception:
            pass


# --------------------------------------------------------------------------
# offline stand-in
# --------------------------------------------------------------------------
class SimGateVision:
    """Geometry-only camera model for offline runs. observe(state) (the runner
    calls it every tick) looks at the TRUE gates from the drone's camera: a
    gate is detected iff all 4 opening corners are in front of the camera and
    inside the FOV (hfov_deg x vfov_deg) and its centre is within max_range.
    Output, at most rate_hz per sim second, per visible gate:
      * centre noise N(0, sigma) per world axis, sigma = pos_noise_std *
        (0.5 + range / 10) (grows with range: 1.0x at 5 m, 2x at 15 m), plus
        range_bias * range along the line of sight, AWAY from the camera (the
        ~1% bias seen live, estimates sit just behind the gate);
      * yaw noise N(0, yaw_noise_std deg) about world Z;
      * normal_reliable only when the angle between the gate normal and the
        line of sight is >= normal_min_deg (near head-on the PnP normal is
        noise; the 12 deg threshold stands in for normal_std_deg < 15).
    NOT modelled: image latency, motion blur, pose latency, false positives,
    occlusion by other gates, 180-deg normal flips. drain() returns the
    detections made since the last call."""

    def __init__(self, truth_course, mount_xyz=DEFAULT_MOUNT_XYZ,
                 mount_rot_xyz_deg=DEFAULT_MOUNT_ROT_XYZ_DEG, hfov_deg=60.0, vfov_deg=36.0,
                 max_range=15.0, pos_noise_std=0.10, yaw_noise_std=5.0, range_bias=0.01,
                 rate_hz=15.0, normal_min_deg=12.0, seed=0):
        self.course = truth_course
        self.mount = mount_matrix(mount_xyz, mount_rot_xyz_deg)
        self.th, self.tv = math.tan(math.radians(hfov_deg) / 2), math.tan(math.radians(vfov_deg) / 2)
        self.max_range, self.pos_noise_std, self.range_bias = max_range, pos_noise_std, range_bias
        self.yaw_noise = math.radians(yaw_noise_std)
        self.period = 1.0 / rate_hz if rate_hz else 0.0
        self.normal_min = math.radians(normal_min_deg)
        self.rng = np.random.default_rng(seed)
        self._t_last = -1e18
        self._out = []
        self.counters = dict(frames_processed=0, dets_total=0, frames_with_dets=0)

    def visible(self, T_world_cam, gate):
        """(is_visible, range, T_cam_gate) for a TRUE gate."""
        T_cam_gate = invert_transform(T_world_cam) @ gate.T_world_gate()
        P = (T_cam_gate[:3, :3] @ gate_model_points(gate.side).T).T + T_cam_gate[:3, 3]
        rng = float(np.linalg.norm(T_cam_gate[:3, 3]))
        z = P[:, 2]
        ok = (bool(np.all(z > 0.3)) and bool(np.all(np.abs(P[:, 0]) <= self.th * z))
              and bool(np.all(np.abs(P[:, 1]) <= self.tv * z)) and rng <= self.max_range)
        return ok, rng, T_cam_gate

    def observe(self, state):
        if state.t_sim - self._t_last < self.period - 1e-9:
            return
        self._t_last = state.t_sim
        self.counters["frames_processed"] += 1
        T_world_cam = T_world_cam_cv(T_world_body_from_state(state), self.mount)
        cam_pos = T_world_cam[:3, 3]
        n_before = len(self._out)
        for g in self.course.gates:
            ok, rng, _ = self.visible(T_world_cam, g)
            if not ok:
                continue
            T = g.T_world_gate()
            los = (T[:3, 3] - cam_pos) / max(np.linalg.norm(T[:3, 3] - cam_pos), 1e-9)
            sigma = self.pos_noise_std * (0.5 + rng / 10.0)
            Tm = T.copy()
            Tm[:3, 3] = T[:3, 3] + self.range_bias * rng * los + self.rng.normal(0.0, sigma, 3)
            d = self.rng.normal(0.0, self.yaw_noise)
            c, s = math.cos(d), math.sin(d)
            Tm[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]) @ T[:3, :3]
            angle = math.acos(max(-1.0, min(1.0, float(T[:3, 2] @ los))))
            self._out.append(Detection(Tm, bool(angle >= self.normal_min), float(state.t_sim), rng))
        n_new = len(self._out) - n_before
        self.counters["dets_total"] += n_new
        self.counters["frames_with_dets"] += 1 if n_new else 0

    def drain(self):
        out, self._out = self._out, []
        return out

    def stats(self):
        c = dict(self.counters)
        c["dets_per_frame"] = c["dets_total"] / c["frames_processed"] if c["frames_processed"] else float("nan")
        return c

    def close(self):
        pass


# --------------------------------------------------------------------------
# what the runner holds
# --------------------------------------------------------------------------
class GateSource:
    """GateMap + a vision front end (RosGateVision / SimGateVision) + the truth
    course for LOGGING errors only. The runner calls step() every state,
    record_crossing(k) when gate k is passed, and summary() at the end."""

    def __init__(self, gate_map, vision, truth_course=None):
        self.map, self.vision = gate_map, vision
        self.truth = truth_course if truth_course is not None else gate_map.truth
        if self.truth is not None and gate_map.truth is None:
            gate_map.truth = self.truth
        self._pushed_version = -1
        self.ticks = 0
        self.ticks_with_dets = 0
        self.dets_total = 0
        self.assoc_total = 0
        self.crossings = []      # (gate, pos_err, lat_err, yaw_err_deg) when crossed

    def step(self, state, gate_idx):
        """Observe (sim), drain, update the map. Returns n detections drained."""
        if hasattr(self.vision, "observe"):
            self.vision.observe(state)
        dets = self.vision.drain()
        self.map.set_progress(gate_idx)
        assoc = self.map.update(dets)
        self.ticks += 1
        self.ticks_with_dets += 1 if dets else 0
        self.dets_total += len(dets)
        self.assoc_total += len(assoc)
        return len(dets)

    def changed(self):
        return self.map.version != self._pushed_version

    def current_course(self):
        """Map course; marks it as pushed."""
        self._pushed_version = self.map.version
        return self.map.course()

    def target_error(self, gate_idx):
        return self.map.error_of(min(gate_idx, len(self.map) - 1))[0]

    def record_crossing(self, gate):
        self.crossings.append((gate, *self.map.error_of(gate)))

    def summary(self):
        cross = np.array([c[1:] for c in self.crossings]) if self.crossings else np.zeros((0, 3))
        out = dict(
            ticks=self.ticks, ticks_with_detections=self.ticks_with_dets,
            detections_total=self.dets_total, detections_associated=self.assoc_total,
            crossing_gates=[c[0] for c in self.crossings],
            crossing_pos_err=[float(x) for x in cross[:, 0]],
            crossing_lat_err=[float(x) for x in cross[:, 1]],
            crossing_yaw_err_deg=[float(x) for x in cross[:, 2]],
            crossing_pos_err_mean=float(cross[:, 0].mean()) if len(cross) else float("nan"),
            crossing_lat_err_mean=float(cross[:, 1].mean()) if len(cross) else float("nan"),
            per_gate=self.map.stats(),
            vision=self.vision.stats() if hasattr(self.vision, "stats") else {})
        if self.truth is not None:
            e = self.map.errors(self.truth)
            out["final_pos_err"] = [float(x) for x in e["pos"]]
            out["prior_pos_err"] = _prior_errors(self.map, self.truth)
        return out


def _prior_errors(gate_map, truth):
    """Position error of the PRIOR gates vs truth (logging)."""
    return [float(np.linalg.norm(p.center - t.center))
            for p, t in zip(gate_map.prior.gates, truth.gates)]
