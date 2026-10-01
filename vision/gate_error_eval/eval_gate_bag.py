"""
Offline gate-pose error evaluation of a recorded ROS 2 bag -> gate_errors.csv.

WORKFLOW (this is step 2 of 3):
    1. fly_gate_approach.py flies a straight line while, in another shell,
           ros2 bag record -o gate_run1 /iris_0/front_cam/rgb \
               /iris_0/front_cam/camera_info /drone00/state/pose
    2. THIS SCRIPT (needs ROS 2 Humble sourced; run in the `ros2-eval`
       container):
           python3 eval_gate_bag.py --bag gate_run1 \
               --gates /workspace/test_gates.json --out-dir run1_out
    3. plot_gate_errors.py run1_out/gate_errors.csv   (on a Mac, no ROS)

WHY OFFLINE: the evaluator does detection + PnP per frame and the bag lets
the same flight be re-evaluated with different detector settings, and with
the pose interpolated to each image's own timestamp (the live evaluator just
uses the latest pose, which is only right when the drone is stationary).

TIMESTAMPS. Each message has a header stamp and a bag receive time. In this
sim the two publishers (camera graph and PX4/Pegasus backend) may be on
different clocks (sim time vs wall time). `--stamp-source auto` uses header
stamps when the image and pose header-stamp ranges overlap, and otherwise
prints a loud WARNING and falls back to bag receive time for both. The
median gap between each image time and the nearest pose sample is printed:
at ~130 Hz pose it should be a few ms; seconds means the clocks disagree.

IMAGE DECODE does not use cv_bridge (NumPy-version trouble on the cluster:
see vision/HANDOFF.md). decode_image() handles rgb8/bgr8/rgba8/bgra8 with
np.frombuffer and respects `step`.

Everything except main()'s bag I/O is ROS-free and importable offline (see
test_error_records.py); ROS imports are lazy inside main().
"""

import argparse
import os
import re
import sys

import numpy as np

# vision/ is the parent directory; make it importable so `eval_isaac_gates`,
# `camera`, etc. resolve when this file is run from gate_error_eval/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera import CameraIntrinsics  # noqa: E402
from gate_pose import DEFAULT_GATE_SIDE  # noqa: E402
from color_gate_detector import detect_gates_color  # noqa: E402
from eval_isaac_gates import evaluate_frame, load_manifest  # noqa: E402
from ros_camera import intrinsics_from_camera_info  # noqa: E402
import error_records as er  # noqa: E402


# --------------------------------------------------------------------------
# Image decode (no cv_bridge)
# --------------------------------------------------------------------------

_CHANNELS = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}


def decode_image(data, width: int, height: int, step: int, encoding: str) -> np.ndarray:
    """
    sensor_msgs/Image fields -> HxWx3 uint8 RGB array.

    `step` is the row stride in bytes and may exceed width*channels (row
    padding), so rows are sliced out of a (height, step) view first.
    """
    if encoding not in _CHANNELS:
        raise ValueError(f"unsupported image encoding {encoding!r}; this decoder handles "
                         f"{sorted(_CHANNELS)} (no cv_bridge on the cluster)")
    ch = _CHANNELS[encoding]
    row_bytes = width * ch
    if step < row_bytes:
        raise ValueError(f"step {step} < width*channels {row_bytes}")
    buf = np.frombuffer(bytes(data), dtype=np.uint8)
    if buf.size < step * height:
        raise ValueError(f"image buffer too small: {buf.size} < step*height {step * height}")
    img = buf[:step * height].reshape(height, step)[:, :row_bytes].reshape(height, width, ch)
    if encoding in ("bgr8", "bgra8"):
        img = img[:, :, [2, 1, 0]]
    else:
        img = img[:, :, :3]
    return np.ascontiguousarray(img)


# --------------------------------------------------------------------------
# Intrinsics / time base (ROS-free, testable)
# --------------------------------------------------------------------------

def build_intrinsics(info_msg, img_w: int, img_h: int, hfov_deg=None):
    """
    (CameraIntrinsics, [message strings]) with the same semantics as
    eval_isaac_gates.py: the real image size is the truth for the pixel grid;
    a stale camera_info (size != image) produces the same WARNING, and
    --hfov-deg rebuilds intrinsics from image size + FOV.
    """
    msgs = []
    intr = intrinsics_from_camera_info(info_msg) if info_msg is not None else None
    if intr is not None and (intr.width, intr.height) != (img_w, img_h):
        msgs.append(f"WARNING: camera_info says {intr.width}x{intr.height} but the rgb image "
                    f"is {img_w}x{img_h}. camera_info is stale, so its fx/cx/cy do not "
                    f"describe this image and PnP results from it will be wrong. Pass "
                    f"--hfov-deg <the hFOV spawn_example.py set> to build the intrinsics "
                    f"from the real image size.")
    if hfov_deg is not None:
        if intr is not None:
            msgs.append(f"NOTE: --hfov-deg {hfov_deg} given: IGNORING camera_info's "
                        f"fx={intr.fx:.1f} and size {intr.width}x{intr.height}; using the "
                        f"rgb image size {img_w}x{img_h} and fx from the FOV instead")
        intr = CameraIntrinsics.from_horizontal_fov(img_w, img_h, hfov_deg)
    if intr is None:
        raise ValueError("no camera_info in the bag and no --hfov-deg given")
    return intr, msgs


def choose_time_base(img_hdr_s, img_bag_s, pose_hdr_s, pose_bag_s, mode: str = "auto"):
    """
    Decide which clock to use. Returns (name, img_times, pose_times, warning)
    where name is "header" or "bag" and warning is a string or None.
    All inputs are sequences of seconds.
    """
    if mode == "header":
        return "header", np.asarray(img_hdr_s), np.asarray(pose_hdr_s), None
    if mode == "bag":
        return "bag", np.asarray(img_bag_s), np.asarray(pose_bag_s), None
    ih, ph = np.asarray(img_hdr_s, dtype=float), np.asarray(pose_hdr_s, dtype=float)
    if ih.size and ph.size and ih.min() <= ph.max() and ph.min() <= ih.max():
        return "header", ih, ph, None
    warn = ("WARNING: image header stamps "
            f"[{ih.min() if ih.size else float('nan'):.3f}, {ih.max() if ih.size else float('nan'):.3f}] "
            f"and pose header stamps [{ph.min() if ph.size else float('nan'):.3f}, "
            f"{ph.max() if ph.size else float('nan'):.3f}] do NOT overlap -- the publishers "
            "are probably on different clocks (sim vs wall). Falling back to bag RECEIVE time "
            "for both; this adds the transport latency to the pose/image pairing.")
    return "bag", np.asarray(img_bag_s), np.asarray(pose_bag_s), warn


def median_pose_gap_s(img_times, pose_times) -> float:
    """Median over images of |t_image - nearest pose sample time|."""
    pt = np.sort(np.asarray(pose_times, dtype=float))
    it = np.asarray(img_times, dtype=float)
    if pt.size == 0 or it.size == 0:
        return float("nan")
    idx = np.clip(np.searchsorted(pt, it), 1, pt.size - 1) if pt.size > 1 else np.zeros(it.size, int)
    if pt.size > 1:
        gap = np.minimum(np.abs(it - pt[idx]), np.abs(it - pt[idx - 1]))
    else:
        gap = np.abs(it - pt[0])
    return float(np.median(gap))


# --------------------------------------------------------------------------
# Per-frame logic (ROS-free)
# --------------------------------------------------------------------------

def evaluate_image_at(rgb, t_query: float, t_s: float, frame_idx: int, intr,
                      pose_buffer: "er.PoseBuffer", manifest: list,
                      gate_side: float = DEFAULT_GATE_SIDE,
                      corner_sigma_px: float = 1.0, max_match_dist_m: float = 2.0,
                      detector=detect_gates_color):
    """
    One image + the pose buffer -> list of CSV row dicts (one per manifest
    gate), or None if no pose is available within the buffer's tolerance
    at t_query (the caller counts and skips such frames).

    t_query: image time on the same clock as pose_buffer; t_s: the same time
    expressed as seconds since the first frame, for the CSV.
    """
    T_world_cam_raw = er.T_world_cam_raw_at(pose_buffer, t_query)
    if T_world_cam_raw is None:
        return None
    result = evaluate_frame(rgb, intr, T_world_cam_raw, manifest, gate_side=gate_side,
                            corner_sigma_px=corner_sigma_px, detector=detector,
                            max_match_dist_m=max_match_dist_m)
    pos, _quat = pose_buffer.interpolate(t_query)
    vel = pose_buffer.velocity(t_query)
    speed = float(np.linalg.norm(vel)) if vel is not None else None
    return er.frame_rows(t_s, frame_idx, intr, T_world_cam_raw, manifest, result, pos, speed,
                         gate_side=gate_side)


# --------------------------------------------------------------------------
# Bag I/O (ROS)
# --------------------------------------------------------------------------

def _storage_id(bag_dir: str) -> str:
    """Read storage_identifier from metadata.yaml (regex, no yaml dependency);
    default sqlite3 (the Humble default)."""
    meta = os.path.join(bag_dir, "metadata.yaml")
    if os.path.isfile(meta):
        with open(meta) as f:
            m = re.search(r"storage_identifier:\s*['\"]?([A-Za-z0-9_]+)", f.read())
        if m:
            return m.group(1)
    return "sqlite3"


def _open_reader(bag: str):
    import rosbag2_py
    reader = rosbag2_py.SequentialReader()
    storage = rosbag2_py.StorageOptions(uri=bag, storage_id=_storage_id(bag))
    converter = rosbag2_py.ConverterOptions(input_serialization_format="cdr",
                                            output_serialization_format="cdr")
    reader.open(storage, converter)
    return reader


def _stamp_s(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--bag", required=True, help="rosbag2 directory (contains metadata.yaml)")
    ap.add_argument("--gates", default="/workspace/test_gates.json")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--rgb-topic", default="/iris_0/front_cam/rgb")
    ap.add_argument("--info-topic", default="/iris_0/front_cam/camera_info")
    ap.add_argument("--pose-topic", default="/drone00/state/pose")
    ap.add_argument("--hfov-deg", type=float, default=None,
                    help="build intrinsics from this hFOV + the real image size, ignoring "
                         "camera_info's fx/cx/cy (use when camera_info is stale)")
    ap.add_argument("--stride", type=int, default=1, help="evaluate every Nth image (default 1)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--stamp-source", choices=["auto", "header", "bag"], default="auto")
    ap.add_argument("--max-match-dist", type=float, default=2.0)
    ap.add_argument("--corner-sigma-px", type=float, default=1.0)
    ap.add_argument("--gate-side", type=float, default=None,
                    help="gate opening side (m); default: from the manifest, else "
                         "gate_pose.DEFAULT_GATE_SIDE")
    args = ap.parse_args()

    try:
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
        import rosbag2_py  # noqa: F401
    except ImportError as e:
        sys.exit(f"this script needs ROS 2 Humble sourced (rclpy/rosbag2_py): {e}")

    manifest = load_manifest(args.gates)
    gate_side = args.gate_side
    if gate_side is None:
        gate_side = float(manifest[0]["side"]) if manifest and "side" in manifest[0] \
            else DEFAULT_GATE_SIDE
    print(f"loaded {len(manifest)} gate(s) from {args.gates}, gate_side={gate_side} m")

    # ---- pass 1: poses, first camera_info, image headers (no pixel data kept)
    reader = _open_reader(args.bag)
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    for topic in (args.rgb_topic, args.pose_topic):
        if topic not in type_map:
            sys.exit(f"topic {topic} not in bag; has: {sorted(type_map)}")
    msg_types = {topic: get_message(type_map[topic])
                 for topic in (args.rgb_topic, args.pose_topic, args.info_topic)
                 if topic in type_map}

    pose_samples = []     # (hdr_s, bag_s, pos, quat_xyzw)
    info_msg = None
    img_hdr_s, img_bag_s = [], []
    img_w = img_h = None
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic == args.pose_topic:
            m = deserialize_message(data, msg_types[topic])
            p, q = m.pose.position, m.pose.orientation
            pose_samples.append((_stamp_s(m.header), t_ns * 1e-9,
                                 (p.x, p.y, p.z), (q.x, q.y, q.z, q.w)))
        elif topic == args.info_topic and info_msg is None:
            info_msg = deserialize_message(data, msg_types[topic])
        elif topic == args.rgb_topic:
            m = deserialize_message(data, msg_types[topic])
            img_hdr_s.append(_stamp_s(m.header))
            img_bag_s.append(t_ns * 1e-9)
            if img_w is None:
                img_w, img_h = int(m.width), int(m.height)
    del reader
    if not img_hdr_s:
        sys.exit(f"no images on {args.rgb_topic}")
    if not pose_samples:
        sys.exit(f"no poses on {args.pose_topic}")
    print(f"bag: {len(img_hdr_s)} image(s), {len(pose_samples)} pose sample(s)")

    intr, notes = build_intrinsics(info_msg, img_w, img_h, args.hfov_deg)
    for n in notes:
        print(f"\n{n}")
    print(f"intrinsics: {intr.width}x{intr.height}, fx={intr.fx:.1f} fy={intr.fy:.1f}")

    name, img_t, pose_t, warn = choose_time_base(
        img_hdr_s, img_bag_s, [s[0] for s in pose_samples], [s[1] for s in pose_samples],
        args.stamp_source)
    if warn:
        print("\n" + "!" * 78 + f"\n{warn}\n" + "!" * 78)
    print(f"time base used: {name} stamps; median |image - nearest pose| gap = "
          f"{median_pose_gap_s(img_t, pose_t) * 1000:.1f} ms")

    buf = er.PoseBuffer()
    use_hdr = name == "header"
    for hdr, bag, pos, quat in pose_samples:
        buf.add(hdr if use_hdr else bag, pos, quat)
    t0 = float(img_t[0])

    # ---- pass 2: evaluate images
    reader = _open_reader(args.bag)
    rows = []
    idx = -1
    n_eval = n_nopose = 0
    while reader.has_next():
        topic, data, _t_ns = reader.read_next()
        if topic != args.rgb_topic:
            continue
        idx += 1
        if idx % max(1, args.stride) != 0:
            continue
        if args.max_frames is not None and n_eval + n_nopose >= args.max_frames:
            break
        m = deserialize_message(data, msg_types[topic])
        try:
            rgb = decode_image(m.data, m.width, m.height, m.step, m.encoding)
        except ValueError as e:
            sys.exit(f"image decode failed: {e}")
        t_query = float(img_t[idx])
        frame_rows = evaluate_image_at(rgb, t_query, t_query - t0, idx, intr, buf, manifest,
                                       gate_side=gate_side,
                                       corner_sigma_px=args.corner_sigma_px,
                                       max_match_dist_m=args.max_match_dist)
        if frame_rows is None:
            n_nopose += 1
            continue
        rows.extend(frame_rows)
        n_eval += 1
        if n_eval % 50 == 0:
            print(f"  evaluated {n_eval} frame(s) (image {idx + 1}/{len(img_t)})")
    del reader

    print(f"\nevaluated {n_eval} frame(s); skipped {n_nopose} with no pose within "
          f"{buf.max_extrap_s * 1000:.0f} ms")
    if not rows:
        sys.exit("no frames evaluated -- check the stamp source / pose topic")

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "gate_errors.csv")
    er.write_csv(csv_path, rows)
    summary = er.summarize(rows)
    er.write_summary_csv(os.path.join(args.out_dir, "summary.csv"), summary)
    print(f"saved {csv_path} ({len(rows)} rows) and summary.csv\n")
    print(er.format_summary_table(summary))


if __name__ == "__main__":
    main()
