"""
Can the color detector + PnP see a gate along the whole lap? Checked offline.

Flies one closed lap (gate_0, gate_1, gate_2, gate_0) the way Stage 4R does:
ReplanSession (replan_node's logic, yaw_mode next_gate) -> MpcCore -> the
PX4-like attitude plant, on the true state. At the camera rate it RENDERS the
front camera (default: the Iris's level 1280x720 / 60 deg; see --resolution, --fx, --cam-pos, --cam-pitch) (every gate: 1.7 m magenta frame, 1.5 m
opening, painted far to near) and runs the real vision/color_gate_detector.py +
vision/gate_pose.py on the picture. Each quad is turned into a world gate
position with the true camera pose and scored against the manifest:

  good fix      matched to a gate, PnP within 0.3 m, quad not touching the border
  bad fix       no gate within 1.5 m, or off by >= 0.3 m (would bias the VIO)
  gap           time from engage to the last gate pass with no good fix

Good fixes also feed the session's gate map, as the camera node would.
Limits: a gate the camera is partly behind (passing through it) is not drawn,
so it cannot hide the gates behind it; flat colors, no floor, no motion blur.

    python Trajectory/check_course_camera.py                                   # the circle course
    python Trajectory/check_course_camera.py --manifest Trajectory/isaac_results/3_gates/course/3_gate_course_pod.json
    python Trajectory/check_course_camera.py --prior 0.5 --seed 0 --png vis.png   # Stage 4R: map 0.5 m off
    python Trajectory/check_course_camera.py --resolution 1024x768 --fx 600 --fy 600 --cx 512 --cy 384 \n        --cam-pos 0.07,0,0 --cam-pitch 0        # another camera (e.g. the Starling's, from its calibration)

Takes 1-3 minutes (rendering + detection at 15 Hz).
"""
import argparse
import math
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--manifest", default=os.path.join(HERE, "isaac_results", "circle_course", "circle_course.json"))
ap.add_argument("--prior", type=float, default=0.0, help="gate map error, m per axis (replan_node prior_sigma)")
ap.add_argument("--seed", type=int, default=0, help="prior seed (replan_node prior_seed)")
ap.add_argument("--cam-rate", type=float, default=15.0)
ap.add_argument("--v", type=float, default=2.0, help="v_max and a_max")
ap.add_argument("--png", default=None)
ap.add_argument("--events", action="store_true", help="print the replanner's events")
ap.add_argument("--resolution", default="1280x720", help="camera WxH")
ap.add_argument("--hfov", type=float, default=60.0, help="horizontal FOV, deg (ignored with --fx)")
ap.add_argument("--fx", type=float, default=None, help="intrinsics in pixels (with --fy --cx --cy), e.g. the drone's calibration")
ap.add_argument("--fy", type=float, default=None)
ap.add_argument("--cx", type=float, default=None)
ap.add_argument("--cy", type=float, default=None)
ap.add_argument("--cam-pos", default="0.30,0.0,0.05", help="camera position in body FLU, m (Iris: 0.30,0,0.05)")
ap.add_argument("--cam-pitch", type=float, default=0.0, help="camera up-pitch, deg (spawn CAMERA_PITCH_DEG)")
a = ap.parse_args()
sys.path[:0] = [os.path.join(REPO, s) for s in ("Trajectory", "control", "vision", "ros2_nodes", "workspace")]

from replan_session import ReplanSession, perturbed_prior              # noqa: E402
from gate_path import gates_from_manifest, gate_crossings              # noqa: E402
from mpc_core import MpcCore, TrajectoryBuffer                          # noqa: E402
from px4_attitude_plant import Px4AttitudePlant, _rot                   # noqa: E402
from drone_constraints import drone_constraints                        # noqa: E402
from vio import camera_mount_usd, rot_rpy                               # noqa: E402
from ros_camera import T_world_cam_from_transform                       # noqa: E402
from camera import CameraIntrinsics, invert_transform, project_points, transform_points   # noqa: E402
from gate_pose import gate_model_points, estimate_gate_pose            # noqa: E402
from color_gate_detector import detect_gates_color                      # noqa: E402

GATES = gates_from_manifest(a.manifest)
START, START_YAW = np.array([-0.05, -0.04, 2.55]), math.radians(-7.2)   # takeoff hover (test_replan_session)
M = drone_constraints["mass"]
W, H = (int(v) for v in a.resolution.lower().split("x"))
if a.fx is not None:
    INTR = CameraIntrinsics(fx=a.fx, fy=a.fy if a.fy is not None else a.fx, cx=a.cx if a.cx is not None else W / 2.0,
                            cy=a.cy if a.cy is not None else H / 2.0, width=W, height=H)
else:
    INTR = CameraIntrinsics.from_horizontal_fov(W, H, a.hfov)
MOUNT = camera_mount_usd(a.cam_pitch, tuple(float(v) for v in a.cam_pos.split(",")))
BAR = 0.10


def T_wc_of(x):
    T_wb = np.eye(4)
    T_wb[:3, :3], T_wb[:3, 3] = rot_rpy(*x[3:6]), x[0:3]
    return T_world_cam_from_transform(T_wb @ MOUNT, optical_frame=False)


def render(T_wc):
    """Black frame; each gate: outer square magenta, opening back to black; far to near."""
    T_cw = invert_transform(T_wc)
    img = np.zeros((INTR.height, INTR.width, 3), np.uint8)
    order = []
    for g in GATES:
        T_cg = T_cw @ g.T_world_gate
        outer = transform_points(T_cg, gate_model_points(g.side + 2 * BAR))
        if (outer[:, 2] < 0.2).any():          # straddles / behind the camera: skip (passing through it)
            continue
        order.append((T_cg[2, 3], T_cg, g))
    for _, T_cg, g in sorted(order, key=lambda o: -o[0]):
        o = project_points(transform_points(T_cg, gate_model_points(g.side + 2 * BAR)), INTR)
        i = project_points(transform_points(T_cg, gate_model_points(g.side)), INTR)
        if np.isnan(o).any() or np.isnan(i).any():
            continue
        # big coordinates overflow int32 in fillPoly: clip far outside the image
        o = np.clip(o, -20000, 20000); i = np.clip(i, -20000, 20000)
        cv2.fillPoly(img, [np.round(o).astype(np.int32)], (255, 0, 255))
        cv2.fillPoly(img, [np.round(i).astype(np.int32)], (0, 0, 0))
    return img


def whole_in_view(T_wc, g, margin_px=3):
    o = project_points(transform_points(invert_transform(T_wc) @ g.T_world_gate, gate_model_points(g.side + 2 * BAR)), INTR)
    return (not np.isnan(o).any() and (o[:, 0] > margin_px).all() and (o[:, 0] < INTR.width - margin_px).all()
            and (o[:, 1] > margin_px).all() and (o[:, 1] < INTR.height - margin_px).all())


def camera_fixes(x):
    """Run detector+PnP on the rendered frame -> list of (gate name or None, world error m, range, touches border)."""
    T_wc = T_wc_of(x)
    img = render(T_wc)
    out = []
    for q in detect_gates_color(img):
        det = estimate_gate_pose(q, INTR, bootstrap_n=0)
        if det is None:
            continue
        c_w = (T_wc @ det.T_cam_gate)[:3, 3]
        d = [np.linalg.norm(c_w - g.center) for g in GATES]
        k = int(np.argmin(d))
        q = np.asarray(q)
        border = bool((q[:, 0] < 4).any() or (q[:, 0] > INTR.width - 4).any() or (q[:, 1] < 4).any() or (q[:, 1] > INTR.height - 4).any())
        out.append((GATES[k].name if d[k] < 1.5 else None, d[k], det.range_m, border, c_w))
    return out, [g.name for g in GATES if whole_in_view(T_wc, g)]


def quat_of(g):
    from scipy.spatial.transform import Rotation
    return tuple(Rotation.from_matrix(g.T_world_gate[:3, :3]).as_quat())


def fly(max_s=60.0, dt=0.02, hover_s=3.0):
    rng = np.random.default_rng(a.seed)
    sess = ReplanSession(perturbed_prior(GATES, a.prior, a.seed), v_max=a.v, a_max=a.v, laps=1, close_loop=True,
                         yaw_mode="next_gate")
    core = MpcCore(N=10, dt=dt, m=M, attitude_lead_s=0.1)
    plant = Px4AttitudePlant(M, tau=0.08)
    x = np.zeros((12, 1)); x[0:3, 0] = START; x[5, 0] = START_YAW
    buf, t = None, 0.0
    cam_every = max(1, int(round(1.0 / (a.cam_rate * dt))))
    log = []          # (t, x, fixes, whole)
    t_engage = None
    while t < max_s:
        k = int(round(t / dt))
        if k % cam_every == 0:
            fixes, whole = camera_fixes(x[:, 0])
            log.append((t, x[:, 0].copy(), fixes, whole))
            good = [(f[4], quat_of(GATES[[g.name for g in GATES].index(f[0])])) for f in fixes if f[0] and not f[3]]
            sess.add_detections(t, good, x[0:3, 0])
        if buf is None:
            if t >= hover_s:
                p = sess.engage(t, x[0:3, 0], x[5, 0])
                buf = TrajectoryBuffer(p.t0, p.times, p.states, p.thrust)
                t_engage = t
            elif k % 25 == 0:
                sess.prepare(t, x[0:3, 0], x[5, 0])
        else:
            p = sess.tick(t, x[0:3, 0])
            if p is not None:
                buf = TrajectoryBuffer(p.t0, p.times, p.states, p.thrust)
            if sess.finished(t) and t > buf.t0 + buf.duration + 1.0:
                break
        if buf is None:
            u = np.array([[M * 9.81], [0.0], [0.0], [0.0]]); att = x[3:6, 0]
        else:
            out = core.step(x, buf.horizon(t, core.N, dt, yaw_now=float(x[5, 0])))
            u, att = out["u0"], out["x_cmd"][3:6, 0]
        if buf is not None:
            for _ in range(4):
                x = plant.step(x, u[0, 0], att, dt / 4)
        t += dt
    return sess, log, t_engage


t0 = time.time()
sess, log, t_eng = fly()
T = np.array([l[0] for l in log]); X = np.array([l[1] for l in log])
passes = [(round(tp - t_eng, 2), sess.map.gates[gid].name, how) for (tp, _, gid, how) in sess.passes()]
through = {g.name: sum(1 for c in cs if c["inside"] and c["forward"]) for g, cs in gate_crossings(T, X[:, 0:3], X[:, 6:9], GATES)}
fl = T >= t_eng
n_good = np.array([sum(1 for f in l[2] if f[0] and not f[3] and f[1] < 0.3) for l in log])
n_bad = np.array([sum(1 for f in l[2] if (f[0] is None) or f[1] >= 0.3) for l in log])
n_border = np.array([sum(1 for f in l[2] if f[3]) for l in log])
n_whole = np.array([len(l[3]) for l in log])
# gaps (no good fix) from engage to the final gate pass
t_last = max(tp for tp, _, _ in passes) + t_eng if passes else T[-1]
fl = (T >= t_eng) & (T <= t_last)
gaps, cur, start = [], 0.0, None
for i in np.where(fl)[0]:
    if n_good[i] == 0:
        start = T[i] if cur == 0 else start
        cur += 1.0 / a.cam_rate
    elif cur:
        gaps.append((round(float(start - t_eng), 1), round(cur, 2))); cur = 0.0
if cur:
    gaps.append((round(float(start - t_eng), 1), round(cur, 2)))
gap = max([g[1] for g in gaps], default=0.0)
gap_at = None
errs = [f[1] for l in log for f in l[2] if f[0] and not f[3]]
hover_whole = sorted(set(n for l in log if l[0] < t_eng for n in l[3]))
hover_fix = [sorted(set(f[0] for f in l[2] if f[0] and not f[3] and f[1] < 0.3)) for l in log if l[0] < t_eng]
print(f"camera {W}x{H} fx {INTR.fx:.0f} fy {INTR.fy:.0f}, pos {a.cam_pos}, pitch {a.cam_pitch}")
print(f"{os.path.basename(os.path.dirname(a.manifest))}/{os.path.basename(a.manifest)} prior {a.prior} seed {a.seed}: lap {T[-1]-t_eng:.1f} s, passes {passes}")
print(f"  true path through: {through}")
print(f"  hover: whole in view {hover_whole}; fixes per frame {hover_fix[-1]}; border-touching quads in hover {int(n_border[~fl].sum())}")
print(f"  flight: frames {fl.sum()}, with >=1 good fix {np.mean(n_good[fl] > 0)*100:.0f}%, "
      f"longest gap {gap:.2f} s, gaps (start, s) {gaps}, "
      f"bad fixes {int(n_bad[fl].sum())}, border quads {int(n_border[fl].sum())}")
if errs:
    print(f"  PnP world error of accepted fixes: median {np.median(errs):.3f} m, p95 {np.percentile(errs, 95):.3f}, max {max(errs):.3f}")
bad = [(round(float(l[0] - t_eng), 2), f[0], round(float(f[1]), 2), round(float(f[2]), 1), f[3], l[3])
       for l in log for f in l[2] if (f[0] is None or f[1] >= 0.3) and t_eng <= l[0] <= t_last]
for b in bad[:20]:
    print(f"  bad fix +{b[0]:.2f} s: nearest {b[1]}, off {b[2]} m, PnP range {b[3]} m, touches border {b[4]}, whole in view {b[5]}")
print(f"  [{time.time()-t0:.0f} s]")

if a.png:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axs[0]
    for g in GATES:
        c = g.corners()
        ax.plot(c[[0, 1], 1], c[[0, 1], 0], color="darkorange", linewidth=4)
        ax.annotate(g.name, (g.center[1], g.center[0]), xytext=(6, -4), textcoords="offset points")
    gi = fl & (n_good > 0)
    ax.plot(X[:, 1], X[:, 0], "k-", lw=1)
    ax.scatter(X[gi, 1], X[gi, 0], s=6, c="green", label="good fix")
    ax.scatter(X[fl & (n_good == 0), 1], X[fl & (n_good == 0), 0], s=10, c="red", label="no good fix")
    ax.invert_xaxis(); ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=.3); ax.legend()
    ax.set_xlabel("Y (m)"); ax.set_ylabel("X (m)"); ax.set_title(os.path.basename(a.manifest))
    ax = axs[1]
    ax.step(T - t_eng, n_good, where="post", label="good fixes / frame")
    ax.step(T - t_eng, n_whole, where="post", label="gates whole in view", alpha=.6)
    ax.step(T - t_eng, -n_border, where="post", label="- border-touching quads", alpha=.6)
    ax.axvline(0, color="k", lw=.8)
    for tp, name, _ in passes:
        ax.axvline(tp, color="orange", lw=.8); ax.text(tp, 1.02, name, rotation=90, fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_xlabel("s since engage"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(a.png, dpi=100)
    print("  plot", a.png)

if a.events:
    for te, txt in sess.events:
        print(f"  ev {te - t_eng:+7.2f}  {txt}")
