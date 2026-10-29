"""Self-tests for vision mode: gate_map, gate_vision, course_update and the runner's
vision path. Run: python test_vision_bridge.py (from rl/isaac/ or rl/); no pytest,
no ROS, no Isaac. rclpy is replaced by a small fake to exercise RosGateVision's
threading, QoS and stale-camera_info logic.

Offline end-to-end runs fly SimVehicle with SimGateVision (a geometric camera
model, see gate_vision.py) and a PERTURBED PRIOR course (position noise per axis,
yaw noise, seeded); scoring is against the TRUE course.
"""

import math
import os
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, RL)

import numpy as np
import torch

import vision_gate_map as gm
import gate_vision as gv
import course_update as cu
import runner
from camera import quat_to_rot, make_transform, invert_transform
from gate_pose import project_gate
from camera import CameraIntrinsics
from course import Course, Gate, TRACKS_DIR
from drone_params import QuadParams
from fly_course import make_controller
from pid_controller import PidController
from vehicle import SimVehicle, VehicleState

P = QuadParams()
BEST = os.path.join(RL, "runs", "best")


def _track(name):
    return Course.load(os.path.join(TRACKS_DIR, f"{name}.json"))


def gate_det(g, d=(0, 0, 0), dyaw_deg=0.0, reliable=True, rng=5.0, t=0.0):
    """A hand-made world-frame detection of gate g, shifted by d and yawed by dyaw."""
    h = Gate(g.name, g.x + d[0], g.y + d[1], g.z + d[2], g.yaw_deg + dyaw_deg)
    return gm.Detection(h.T_world_gate(), reliable, t, rng)


def line_course(n=3, spacing=8.0):
    return Course([Gate(f"gate_{i}", spacing * (i + 1), 0.0, 2.5, 0.0) for i in range(n)], "line")


# ---------------------------------------------------------------- gate_map
def test_yaw_convention_and_prior():
    for y in (0.0, 35.0, -120.0, 179.0):
        T = Gate("g", 1, 2, 3, y).T_world_gate()
        assert abs(gm.wrap_pi(gm.yaw_of_T_world_gate(T) - math.radians(y))) < 1e-12
    c = _track("oval6")
    a, b = gm.perturbed_prior(c, 0.5, 10, 7), gm.perturbed_prior(c, 0.5, 10, 7)
    assert [g.x for g in a.gates] == [g.x for g in b.gates] and len(a) == len(c)
    assert [g.name for g in a.gates] == [g.name for g in c.gates]
    big = Course([Gate(f"g{i}", 0, 0, 0, 0) for i in range(4000)])
    pr = gm.perturbed_prior(big, 0.5, 10, 1)
    xs = np.array([[g.x, g.y, g.z] for g in pr.gates])
    assert abs(xs.std() - 0.5) < 0.03 and abs(np.std([g.yaw_deg for g in pr.gates]) - 10) < 0.5
    assert gm.perturbed_prior(c, 0.5, 10, 8).gates[0].x != a.gates[0].x
    print("  yaw convention matches Gate; perturbed_prior seeded, per-axis 0.5 m / 10 deg  OK")


def test_gate_map_fusion():
    truth = line_course()
    prior = Course([Gate(g.name, g.x + 0.5, g.y - 0.4, g.z + 0.2, 12.0) for g in truth.gates], "p")
    m = gm.GateMap(prior, truth_course=truth)
    assert m.course().gates[0].x == 8.5 and m.version == 0
    e0 = m.errors()["pos"][0]
    rng = np.random.default_rng(0)
    errs = []
    for k in range(60):
        d = rng.normal(0, 0.12, 3)
        m.update([gate_det(truth.gates[0], d, rng.normal(0, 5), True, 6.0, t=k)])
        errs.append(m.errors()["pos"][0])
    assert e0 > 0.6 and errs[0] < 0.25 and errs[-1] < 0.1, (e0, errs[0], errs[-1])
    st = m.stats()[0]
    assert st["n_det"] == 60 and st["n_yaw"] == 60 and st["last_seen"] == 59
    assert st["pos_std"] <= 0.09 and abs(m.course().gates[0].yaw_deg) < 3.0
    assert m.stats()[1]["n_det"] == 0 and m.course().gates[1].x == 16.5    # untouched
    v = m.version
    m.course()
    assert m.version == v and m.course() is m.course()                    # cached
    # range weighting: the same offset measurement moves the estimate less when far
    near, far = gm.GateMap(prior), gm.GateMap(prior)
    d = (-0.5, 0.0, 0.0)
    near.update([gate_det(truth.gates[0], d, rng=3.0)])
    far.update([gate_det(truth.gates[0], d, rng=20.0)])
    assert abs(near.course().gates[0].x - 8.5) > abs(far.course().gates[0].x - 8.5)   # moves more toward a near measurement
    print(f"  fusion: error {e0:.2f} -> {errs[0]:.2f} (1 det) -> {errs[-1]:.2f} m (60), "
          f"yaw 12 -> {m.course().gates[0].yaw_deg:.1f} deg, far detections weigh less  OK")


def test_gate_map_association():
    truth = line_course(4, 8.0)
    m = gm.GateMap(truth, assoc_radius=2.5)
    # too far from every gate -> unassociated
    assert m.update([gate_det(truth.gates[0], (0, 3.0, 0))]) == [] and m.n_unassoc == 1
    assert m.version == 0
    # nearest gate wins; two detections for the same gate -> only the nearer one is used
    out = m.update([gate_det(truth.gates[0], (0.4, 0, 0)), gate_det(truth.gates[0], (0.1, 0, 0)),
                    gate_det(truth.gates[1], (0, 0.2, 0))])
    assert out == [(1, 0), (2, 1)], out
    assert m.stats()[0]["n_det"] == 1
    # progress: passed gates are ignored (detection of gate 0 after passing it)
    m.set_progress(1)
    assert m.update([gate_det(truth.gates[0])]) == []
    # lookahead: only `lookahead` gates from the progress are candidates
    m2 = gm.GateMap(truth, lookahead=2)
    assert m2.update([gate_det(truth.gates[3])]) == [] and m2.update([gate_det(truth.gates[1])]) == [(0, 1)]
    # ambiguity between two gates both in range: the CURRENT target wins
    close = Course([Gate("a", 0, 0, 2.5, 0), Gate("b", 3.0, 0, 2.5, 0)])
    m3 = gm.GateMap(close, assoc_radius=2.5)
    mid = gm.Detection(Gate("x", 1.5, 0, 2.5, 0).T_world_gate(), True, 0.0, 5.0)
    assert m3.update([mid]) == [(0, 0)]
    m3.set_progress(1)
    assert m3.update([mid]) == [(0, 1)]
    # tuples are accepted
    m4 = gm.GateMap(truth)
    assert m4.update([(gate_det(truth.gates[0]).T_world_gate, True, 1.0)]) == [(0, 0)]
    print("  association: radius, one detection per gate (nearest), progress, lookahead, current-target tie  OK")


def test_gate_map_yaw_rules():
    truth = line_course(2)
    prior = Course([Gate(g.name, g.x, g.y, g.z, 10.0) for g in truth.gates])   # prior yaw 10 deg
    # unreliable normal: position fused, yaw untouched
    m = gm.GateMap(prior)
    m.update([gate_det(truth.gates[0], (0.3, 0, 0), dyaw_deg=-8.0, reliable=False)])
    g0 = m.course().gates[0]
    assert g0.x < 8.3 and g0.x > 8.0 and abs(g0.yaw_deg - 10.0) < 1e-9 and m.stats()[0]["n_yaw"] == 0
    # 180 deg ambiguity: a normal pointing BACK is flipped toward the prior's flight direction
    m = gm.GateMap(prior)
    for _ in range(40):
        m.update([gate_det(truth.gates[0], dyaw_deg=180.0, reliable=True)])    # truth yaw 0, flipped
    assert abs(m.course().gates[0].yaw_deg) < 1.0, m.course().gates[0].yaw_deg
    m = gm.GateMap(prior)
    for _ in range(40):
        m.update([gate_det(truth.gates[0], dyaw_deg=-175.0, reliable=True)])   # -175+180 = 5
    assert abs(m.course().gates[0].yaw_deg - 5.0) < 1.0, m.course().gates[0].yaw_deg
    # wrap-around: prior near +179, measurement near -179 must not average to ~0
    wrap = Course([Gate("w", 8, 0, 2.5, 179.0)])
    m = gm.GateMap(wrap)
    for _ in range(40):
        m.update([gate_det(Gate("w", 8, 0, 2.5, -178.0), reliable=True)])
    assert abs(abs(m.course().gates[0].yaw_deg) - 178.0) < 1.0, m.course().gates[0].yaw_deg
    # yaw outlier (80 deg from the prior after resolving) is rejected, position still used
    m = gm.GateMap(prior)
    m.update([gate_det(truth.gates[0], (0.1, 0, 0), dyaw_deg=80.0, reliable=True)])
    assert m.stats()[0]["n_yaw_rejected"] == 1 and abs(m.course().gates[0].yaw_deg - 10) < 1e-9
    print("  yaw: only reliable normals, 180 deg flip resolved toward the prior, wrap-around, outliers  OK")


# ---------------------------------------------------------------- process_frame
def _pose_looking_at_gate():
    """Drone at (0, 0, 2.5) pitched 15 deg nose down (the camera is mounted 15 deg
    up, so its axis is level), gate 8 m ahead and yawed 25 deg."""
    a = math.radians(15.0)
    quat = np.array([math.cos(a / 2), 0.0, math.sin(a / 2), 0.0])      # about +y(left): nose down
    pos = np.array([0.0, 0.0, 2.5])
    return pos, quat, Gate("gate_0", 8.0, 0.5, 2.4, 25.0)


def test_process_frame_synthetic():
    from test_color_gate_detector import render_synthetic_gate
    intr = CameraIntrinsics.from_horizontal_fov(1280, 720, 60.0)
    pos, quat, gate = _pose_looking_at_gate()
    T_body = make_transform(quat_to_rot(quat), pos)
    out = {}
    for name, mount_xyz in (("default mount", (0.30, 0, 0.05)), ("moved mount", (-0.040, 0, -0.041))):
        mount = gv.mount_matrix(mount_xyz)
        T_cam = gv.T_world_cam_cv(T_body, mount)
        T_cam_gate = invert_transform(T_cam) @ gate.T_world_gate()
        img = render_synthetic_gate(intr, T_cam_gate)
        info = {}
        dets = gv.process_frame(img, intr, T_body, mount, t=12.5, info=info)
        assert len(dets) == 1, (name, info)
        d = dets[0]
        err = np.linalg.norm(d.T_world_gate[:3, 3] - gate.center)
        assert err < 0.08, (name, err)
        assert d.t == 12.5 and abs(d.range_m - np.linalg.norm(T_cam_gate[:3, 3])) < 0.05
        assert d.normal_reliable, "a 25 deg yawed gate at 8 m should have a reliable normal"
        yaw_err = abs(math.degrees(gm.wrap_pi(gm.yaw_of_T_world_gate(d.T_world_gate) - math.radians(25))))
        assert yaw_err < 6.0, yaw_err
        out[name] = (err, yaw_err)
        if name == "moved mount":
            # the detection made with the WRONG mount is off by the mount difference
            wrong = gv.process_frame(img, intr, T_body, gv.mount_matrix((0.30, 0, 0.05)))
            assert np.linalg.norm(wrong[0].T_world_gate[:3, 3] - gate.center) > 0.2
    # a gate whose opening touches the image border is dropped, not mis-posed
    far = Gate("g", 8.0, 4.0, 2.4, 0.0)
    img = render_synthetic_gate(intr, invert_transform(gv.T_world_cam_cv(T_body, gv.mount_matrix())) @ far.T_world_gate())
    info = {}
    assert gv.process_frame(img, intr, T_body, gv.mount_matrix(), info=info) == [] and (
        info["n_corners"] == 0 or info["rej_border"] == 1)
    print("  process_frame on a synthetic render: " + "; ".join(
        f"{k}: {e:.3f} m, yaw {y:.1f} deg" for k, (e, y) in out.items()) + "  OK")


def test_image_to_rgb():
    h, w = 4, 5
    base = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
    pad = np.zeros((h, w * 3 + 7), np.uint8)
    pad[:, :w * 3] = base.reshape(h, -1)
    mk = lambda enc, arr, step: types.SimpleNamespace(encoding=enc, height=h, width=w, step=step,  # noqa: E731
                                                      data=arr.tobytes())
    assert np.array_equal(gv.image_to_rgb(mk("rgb8", pad, w * 3 + 7)), base)
    assert np.array_equal(gv.image_to_rgb(mk("bgr8", pad, w * 3 + 7)), base[..., ::-1])
    rgba = np.concatenate([base, np.full((h, w, 1), 255, np.uint8)], axis=2).reshape(h, -1)
    assert np.array_equal(gv.image_to_rgb(mk("rgba8", rgba, w * 4)), base)
    print("  image_to_rgb: rgb8 with row padding, bgr8, rgba8  OK")


# ---------------------------------------------------------------- fake rclpy: RosGateVision
class _FakeNode:
    instances = []

    def __init__(self, name):
        self.subs = []
        _FakeNode.instances.append(self)

    def create_subscription(self, typ, topic, cb, qos):
        self.subs.append((typ, topic, cb, qos))

    def destroy_node(self):
        pass


class _FakeExec:
    def add_node(self, n):
        pass

    def spin_once(self, timeout_sec=0.1):
        time.sleep(0.01)

    def shutdown(self):
        pass


def _install_fake_rclpy():
    names = ["rclpy", "rclpy.executors", "rclpy.node", "rclpy.qos", "sensor_msgs", "sensor_msgs.msg",
             "geometry_msgs", "geometry_msgs.msg"]
    saved = {n: sys.modules.get(n) for n in names}
    sentinel = object()
    mods = {n: types.ModuleType(n) for n in names}
    mods["rclpy"].ok = lambda: True
    mods["rclpy"].init = lambda: None
    mods["rclpy"].shutdown = lambda: None
    mods["rclpy.executors"].SingleThreadedExecutor = _FakeExec
    mods["rclpy.node"].Node = _FakeNode
    mods["rclpy.qos"].qos_profile_sensor_data = sentinel
    mods["sensor_msgs.msg"].Image = type("Image", (), {})
    mods["sensor_msgs.msg"].CameraInfo = type("CameraInfo", (), {})
    mods["geometry_msgs.msg"].PoseStamped = type("PoseStamped", (), {})
    sys.modules.update(mods)
    return saved, sentinel


def _restore(saved):
    for n, m in saved.items():
        if m is None:
            sys.modules.pop(n, None)
        else:
            sys.modules[n] = m


def _msgs(T_body, img, w, h, info_wh, hfov=60.0):
    from camera import CameraIntrinsics as CI
    q = None
    # rotation matrix -> xyzw via the vision helper
    from ros_camera import _rot_to_quat_xyzw
    q = _rot_to_quat_xyzw(T_body[:3, :3])
    pose = types.SimpleNamespace(pose=types.SimpleNamespace(
        position=types.SimpleNamespace(x=T_body[0, 3], y=T_body[1, 3], z=T_body[2, 3]),
        orientation=types.SimpleNamespace(x=q[0], y=q[1], z=q[2], w=q[3])))
    iw, ih = info_wh
    K = CI.from_horizontal_fov(iw, ih, hfov).K
    info = types.SimpleNamespace(width=iw, height=ih, k=K.ravel().tolist(), d=[0.0] * 5)
    image = types.SimpleNamespace(encoding="rgb8", height=h, width=w, step=w * 3, data=img.tobytes(),
                                  header=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=7, nanosec=500_000_000)))
    return pose, info, image


def _wait(cond, timeout=5.0):
    t_end = time.time() + timeout
    while time.time() < t_end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_ros_gate_vision_with_fake_rclpy():
    from test_color_gate_detector import render_synthetic_gate
    saved, sentinel = _install_fake_rclpy()
    try:
        pos, quat, gate = _pose_looking_at_gate()
        T_body = make_transform(quat_to_rot(quat), pos)
        intr = CameraIntrinsics.from_horizontal_fov(1280, 720, 60.0)
        mount = gv.mount_matrix()
        img = render_synthetic_gate(intr, invert_transform(gv.T_world_cam_cv(T_body, mount)) @ gate.T_world_gate())
        pose, good_info, image = _msgs(T_body, img, 1280, 720, (1280, 720))

        # --- good camera_info: one detection, stamped with the IMAGE time (sim clock)
        said = []
        v = gv.RosGateVision(say=said.append)
        node = _FakeNode.instances[-1]
        topics = {t: (cb, q) for _, t, cb, q in node.subs}
        assert set(topics) == {"/drone00/state/pose", "/iris_0/front_cam/camera_info",
                               "/iris_0/front_cam/rgb"}, topics.keys()
        assert all(q is sentinel for _, q in topics.values()), "all subscriptions use qos_profile_sensor_data"
        topics["/drone00/state/pose"][0](pose)
        topics["/iris_0/front_cam/camera_info"][0](good_info)
        topics["/iris_0/front_cam/rgb"][0](image)
        assert _wait(lambda: v.counters["frames_processed"] == 1)
        d = v.drain()
        assert len(d) == 1 and d[0].t == 7.5 and v.drain() == []
        assert np.linalg.norm(d[0].T_world_gate[:3, 3] - gate.center) < 0.08
        st = v.stats()
        assert st["dets_total"] == 1 and st["frames_with_dets"] == 1 and st["proc_ms_mean"] > 0
        # --- an image before any pose is skipped, not queued
        v2 = gv.RosGateVision(say=said.append)
        t2 = {t: cb for _, t, cb, _ in _FakeNode.instances[-1].subs}
        t2["/iris_0/front_cam/camera_info"](good_info)
        t2["/iris_0/front_cam/rgb"](image)
        assert _wait(lambda: v2.counters["frames_skipped_no_pose"] == 1)
        v2.close()

        # --- stale camera_info (320x240 vs a 1280x720 image): skipped with a warning ...
        said.clear()
        _, stale_info, _ = _msgs(T_body, img, 1280, 720, (320, 240), hfov=24.0)
        v3 = gv.RosGateVision(say=said.append)
        t3 = {t: cb for _, t, cb, _ in _FakeNode.instances[-1].subs}
        t3["/drone00/state/pose"](pose)
        t3["/iris_0/front_cam/camera_info"](stale_info)
        t3["/iris_0/front_cam/rgb"](image)
        assert _wait(lambda: v3.counters["frames_skipped_stale_info"] == 1)
        assert v3.drain() == [] and any("STALE" in s for s in said), said
        v3.close()
        # ... and fixed by hfov_override (intrinsics rebuilt from the IMAGE size)
        v4 = gv.RosGateVision(hfov_override=60.0, say=said.append)
        t4 = {t: cb for _, t, cb, _ in _FakeNode.instances[-1].subs}
        t4["/drone00/state/pose"](pose)
        t4["/iris_0/front_cam/camera_info"](stale_info)
        t4["/iris_0/front_cam/rgb"](image)
        assert _wait(lambda: v4.counters["frames_processed"] == 1)
        assert np.linalg.norm(v4.drain()[0].T_world_gate[:3, 3] - gate.center) < 0.08
        v4.close()

        # --- busy worker: newer images replace the pending one, nothing queues up
        orig = gv.process_frame
        gv.process_frame = lambda *a, **k: (time.sleep(0.25), [])[1]
        try:
            v5 = gv.RosGateVision(say=said.append)
            t5 = {t: cb for _, t, cb, _ in _FakeNode.instances[-1].subs}
            t5["/drone00/state/pose"](pose)
            t5["/iris_0/front_cam/camera_info"](good_info)
            for _ in range(8):
                t5["/iris_0/front_cam/rgb"](image)
                time.sleep(0.01)
            assert _wait(lambda: v5.counters["frames_processed"] >= 1)
            time.sleep(0.6)
            c = v5.counters
            assert c["frames_in"] == 8 and c["frames_processed"] <= 3 and c["frames_dropped"] >= 5, c
            v5.close()
        finally:
            gv.process_frame = orig
        v.close()
    finally:
        _restore(saved)
    print("  RosGateVision (fake rclpy): sensor QoS, pose pairing, image-time stamps, no pose, stale "
          "camera_info skipped / hfov_override, busy frames dropped  OK")


# ---------------------------------------------------------------- SimGateVision
def _state(pos, yaw_deg=0.0, pitch_deg=15.0, t=0.0):
    y, p = math.radians(yaw_deg) / 2, math.radians(pitch_deg) / 2
    qy = np.array([math.cos(p), 0, math.sin(p), 0])
    qz = np.array([math.cos(y), 0, 0, math.sin(y)])
    w1, x1, y1, z1 = qz
    w2, x2, y2, z2 = qy
    q = np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                  w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
    return VehicleState(t, np.asarray(pos, float), np.zeros(3), q, np.zeros(3))


def test_sim_gate_vision_visibility():
    c = line_course(3, 8.0)                       # gates at x = 8, 16, 24, z = 2.5
    quiet = dict(pos_noise_std=0, yaw_noise_std=0, range_bias=0, rate_hz=0, max_range=10.0)
    sv = gv.SimGateVision(c, **quiet)
    def seen(state, **kw):
        s = gv.SimGateVision(c, **{**quiet, **kw})
        s.observe(state)
        return s.drain()
    # in view: from x=2 the first gate is 6 m ahead (pitched 15 deg down: level camera)
    d = seen(_state((2, 0, 2.5)))
    assert len(d) == 1 and np.allclose(d[0].T_world_gate[:3, 3], c.gates[0].center, atol=1e-9), d
    cam = gv.T_world_cam_cv(gv.T_world_body_from_state(_state((2, 0, 2.5))), gv.mount_matrix())[:3, 3]
    assert abs(d[0].range_m - np.linalg.norm(c.gates[0].center - cam)) < 1e-9 and 5.5 < d[0].range_m < 6.0
    # level hover: the camera looks 15 deg up, the gate's lower corners are out of view
    assert seen(_state((2, 0, 2.5), pitch_deg=0.0)) == []
    # behind the camera, to the side, beyond the range, too close (corners out of the FOV)
    assert seen(_state((2, 0, 2.5), yaw_deg=180)) == []
    assert seen(_state((2, 0, 2.5), yaw_deg=55)) == []
    assert seen(_state((-20, 0, 2.5))) == []
    assert all(round(x.T_world_gate[0, 3]) != 8 for x in seen(_state((7.0, 0, 2.5))))   # 1 m from gate 0
    # two in view at once
    assert [round(x.T_world_gate[0, 3]) for x in seen(_state((9, 0, 2.5)), max_range=15.0)] == [16, 24]
    # bias: +1% of the range along the line of sight, away from the camera
    b = seen(_state((2, 0, 2.5)), range_bias=0.01)[0]
    assert abs(b.T_world_gate[0, 3] - 8.0 - 0.01 * b.range_m * 1.0) < 0.01 * b.range_m * 0.2
    # head-on is "unreliable", an oblique gate reliable
    assert seen(_state((2, 0, 2.5)))[0].normal_reliable is False
    yawed = Course([Gate("g", 8, 0, 2.5, 40.0)])
    s2 = gv.SimGateVision(yawed, **quiet)
    s2.observe(_state((2, 0, 2.5)))
    assert s2.drain()[0].normal_reliable is True
    # rate limit and noise
    s3 = gv.SimGateVision(c, rate_hz=10.0, seed=1, max_range=10.0)
    for k in range(50):
        s3.observe(_state((2, 0, 2.5), t=100 + k * 0.02))
    assert 9 <= len(s3.drain()) <= 11
    pts = []
    s4 = gv.SimGateVision(c, rate_hz=0, seed=2, pos_noise_std=0.1, max_range=10.0)
    for k in range(200):
        s4.observe(_state((2, 0, 2.5), t=k * 0.1))
    pts = np.array([x.T_world_gate[:3, 3] for x in s4.drain()])
    assert 0.05 < pts[:, 1].std() < 0.2
    print("  SimGateVision: in view -> 1, level hover / behind / sideways / far / too close -> 0, "
          "two in view, +1% LOS bias, reliability, rate, noise  OK")


# ---------------------------------------------------------------- course_update
class _Hooked:
    command = "ctbr"

    def __init__(self):
        self.calls = []

    def update_course(self, course, pos, vel):
        self.calls.append((course, pos, vel))


def test_course_update_adapters():
    c = _track("oval6")
    new = gm.perturbed_prior(c, 0.5, 10, 3)
    # PID: caches equal a fresh reset on the new course; stage, integrator, last_idx survive
    pid = PidController(P, c)
    pid.reset(1)
    pid.stage[:] = 2
    pid.last_idx[:] = 4
    pid.vel_int += 0.7
    assert cu.update_course(pid, new, np.zeros(3), np.zeros(3)) == "pid"
    fresh = PidController(P, new)
    fresh.reset(1)
    assert torch.equal(pid.centers, fresh.centers) and torch.equal(pid.normals, fresh.normals)
    assert pid.course is new and int(pid.stage[0]) == 2 and int(pid.last_idx[0]) == 4
    assert torch.allclose(pid.vel_int, torch.full((1, 3), 0.7))
    # before reset: just the course (reset builds the caches from it)
    pid2 = PidController(P, c)
    cu.update_course(pid2, new)
    pid2.reset(1)
    assert torch.equal(pid2.centers, fresh.centers)
    # a changed gate count is refused
    try:
        cu.update_course(pid, Course(new.gates[:3]))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    # the controller's own method wins
    h = _Hooked()
    assert cu.update_course(h, new, np.ones(3), np.zeros(3)) == "method" and len(h.calls) == 1
    # RL (needs the checkpoint)
    if os.path.isdir(BEST):
        rl = make_controller("rl", P, c, BEST)
        rl.reset(1)
        rl.prev_action += 0.25
        assert cu.update_course(rl, new) == "rl"
        from race_env import CoursePool
        ref = CoursePool([new])
        assert torch.equal(rl.pool.centers, ref.centers) and torch.equal(rl.pool.R, ref.R)
        assert float(rl.prev_action[0, 0]) == 0.25
        # the observation uses the new gates
        pos = torch.zeros(1, 3)
        obs = rl.observe(pos, pos, torch.tensor([[1.0, 0, 0, 0]]), pos, torch.zeros(1, dtype=torch.long))
        rl2 = make_controller("rl", P, new, BEST)
        rl2.reset(1)
        rl2.prev_action += 0.25
        assert torch.allclose(obs, rl2.observe(pos, pos, torch.tensor([[1.0, 0, 0, 0]]), pos,
                                               torch.zeros(1, dtype=torch.long)))
    else:
        print("  (RL adapter check skipped: rl/runs/best is missing)")
    print("  course_update: PID caches == fresh reset (stage/integrator kept), pre-reset, gate count "
          "guard, controller hook, RL pool + observation  OK")


# ---------------------------------------------------------------- end to end (offline)
def fly_vision(track, kind, seed, vision=True, pos_std=0.5, yaw_std=10.0, max_time=60.0, **vkw):
    c = _track(track)
    prior = gm.perturbed_prior(c, pos_std, yaw_std, seed)
    front = gv.SimGateVision(c, seed=seed, **vkw)
    if not vision:
        front.observe = lambda s: None         # detections never arrive
    gs = gv.GateSource(gm.GateMap(prior, truth_course=c), front, c)
    ctl = make_controller(kind, P, prior, BEST if kind == "rl" else None)
    log, s = runner.run_in_vehicle(SimVehicle(P), c, ctl, P, max_time=max_time, verbose=False,
                                   gate_source=gs)
    return log, s


def _row(track, kind, mode, s):
    v = s.get("vision", {})
    lat = ", ".join(f"{x:.2f}" for x in v.get("crossing_lat_err", []))
    pr = ", ".join(f"{x:.2f}" for x in v.get("prior_pos_err", []))
    lap = f"{s['lap_time']:.2f}" if s["completed"] else "-"
    return (f"| {kind} | {track} | {mode} | {'yes' if s['completed'] else 'no'} "
            f"({s['gates_passed']}/{s['n_gates']}{', ' + s['crash_kind'] if s['crash_kind'] else ''}) "
            f"| {lap} | {lat or '-'} | {pr or '-'} |")


def test_end_to_end_vision():
    """PID straight3 + oval6 and RL straight3 complete from a 0.5 m / 10 deg prior with vision;
    the same prior WITHOUT detections makes the run hit a gate (seeds 0-2 all do, on every
    controller/track here): the map correction is doing real work."""
    rows = []
    for track, kind in (("straight3", "pid"), ("oval6", "pid"), ("straight3", "rl")):
        if kind == "rl" and not os.path.isdir(BEST):
            print("  SKIPPED RL vision end-to-end: rl/runs/best is missing")
            continue
        for seed in (0, 1):
            _, s_off = fly_vision(track, kind, seed, vision=False)
            log, s_on = fly_vision(track, kind, seed, vision=True)
            rows.append(_row(track, kind, f"vision OFF, seed {seed}", s_off))
            rows.append(_row(track, kind, f"vision ON, seed {seed}", s_on))
            assert not s_off["completed"] and s_off["crash_kind"] == "gate", (track, kind, seed, s_off)
            assert s_on["completed"] and not s_on["aborted"], (track, kind, seed, s_on)
            v = s_on["vision"]
            assert v["detections_total"] > 20 and v["ticks_with_detections"] > 10
            lat = np.array(v["crossing_lat_err"])
            assert len(lat) == s_on["n_gates"] and np.all(lat[1:] < 0.2), lat    # camera-corrected gates
            assert np.all(lat < 0.4), lat                                        # all inside the opening
            assert "n_det" in log and log["n_det"].sum() > 0
            assert np.isfinite(log["map_err"]).all() and (log["n_det"] >= 0).all()
            # scoring is against the TRUE course: a straight3 prior gate is up to ~1 m off, yet
            # the summary margins/lap are from the true gates
            assert s_on["min_margin"] >= 0.3
    print("\n".join(["  | controller | track | mode | completed | lap [s] | map lat. error at crossing [m] "
                     "| prior 3-D error [m] |", "  |---|---|---|---|---|---|---|"] + ["  " + r for r in rows]))
    print("  end to end: with vision every run completes with <= 0.2 m (gates 1+) in-plane map error at crossing; "
          "without it every run hits a gate  OK")
    return rows


def test_log_and_cli():
    c = _track("straight3")
    log, s = fly_vision("straight3", "pid", 0)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "log.csv")
        runner.write_log_csv(p, log)
        head = open(p).readline().strip().split(",")
        assert head[-2:] == ["n_det", "map_err"] and len(np.loadtxt(p, delimiter=",", skiprows=1)) == len(log["t_sim"])
    txt = runner.format_summary("x", s)
    assert "vision: detections=" in txt and "map error at crossing" in txt
    # truth-mode log/summary are unchanged (no extra columns, no vision key)
    log0, s0 = runner.run_in_vehicle(SimVehicle(P), c, PidController(P, c), P, max_time=30, verbose=False)
    assert "n_det" not in log0 and "vision" not in s0 and "vision" not in runner.format_summary("x", s0)
    # CLI: --gates vision --sim --sim-vision
    with tempfile.TemporaryDirectory() as d:
        rc = runner.main(["--track", os.path.join(TRACKS_DIR, "straight3.json"), "--controller", "pid",
                          "--sim", "--gates", "vision", "--sim-vision", "--prior-seed", "1",
                          "--out-dir", d, "--max-time", "30"])
        assert rc == 0 and os.path.exists(os.path.join(d, "summary.json"))
        import json
        sj = json.load(open(os.path.join(d, "summary.json")))
        assert sj["completed"] and sj["vision"]["detections_total"] > 0
    # unknown controller kinds fail cleanly (mpc_ctbr may not exist in fly_course yet)
    try:
        make_controller("mpc_ctbr", P, c)
    except ValueError:
        print("  (mpc_ctbr is not in fly_course.make_controller yet: its CLI check is skipped)")
        rc = runner.main(["--track", os.path.join(TRACKS_DIR, "straight3.json"), "--controller", "mpc_ctbr", "--sim"])
        assert rc == 2          # reported cleanly, not a traceback
    else:
        rc = runner.main(["--track", os.path.join(TRACKS_DIR, "straight3.json"), "--controller", "mpc_ctbr", "--sim"])
        assert rc == 0
    print("  log columns n_det / map_err in the CSV, truth mode unchanged, CLI --gates vision --sim --sim-vision  OK")


if __name__ == "__main__":
    for n in ["test_yaw_convention_and_prior", "test_gate_map_fusion", "test_gate_map_association",
              "test_gate_map_yaw_rules", "test_process_frame_synthetic", "test_image_to_rgb",
              "test_ros_gate_vision_with_fake_rclpy", "test_sim_gate_vision_visibility",
              "test_course_update_adapters", "test_end_to_end_vision", "test_log_and_cli"]:
        print(n)
        globals()[n]()
    print("\nAll tests passed.")
