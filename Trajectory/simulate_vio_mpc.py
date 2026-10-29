"""
Offline rehearsal of the Isaac VIO + MPC run: the NMPC flies the 3-gate
course using ONLY the IMU and the camera (control/vio.py), no ground truth.

Everything the drone knows comes from:
  IMU     250 Hz, FRD, specific force from the velocity change (as Pegasus'
          imu.py computes it), Pegasus' noise densities, and a random gyro
          turn-on bias (Pegasus draws ~0.5 deg/s 1-sigma per axis)
  camera  1280x720 / 60 deg, --camera-rate Hz, --latency s late, true gate
          corners + --corner-noise-px through the real PnP
          (vision/gate_pose.estimate_gate_pose), --range-bias like the live sim
  map     the gate manifest (in Isaac the gates are spawned from it exactly)

Sequence (sim time):
  0 .. --hover-s   hovering at the spawn pose: the VIO calibrates the gyro
                   bias (first 2 s) and initializes from the first reliable
                   gate fix
  --hover-s        plan min-snap from the VIO's pose through the map gates
                   (Trajectory/min_snap_planner.plan_course, as ros2_nodes/min_snap_node.py does)
  then             MpcCore (control/mpc_core.py, as ros2_nodes/mpc_node.py) at 50 Hz on the VIO state;
                   the true drone at 250 Hz is, with --plant:
                     px4     (default) thrust + the MPC's attitude --attitude-lead-s ahead, the
                             attitude lagging it by --attitude-tau (control/px4_attitude_plant.py):
                             how Isaac flies it, through PX4
                     torque  the MPC's own control model (thrust + torques), as before 2026-10-29;
                             this hid that PX4 needs a lead (control/mpc_core.py "ATTITUDE LEAD")

Scored on the TRUE path: gate passes, MPC tracking error, VIO error.

    python simulate_vio_mpc.py                         # pod manifest, defaults
    python simulate_vio_mpc.py --latency 0.15 --camera-rate 10 --corner-noise-px 1.5
"""

import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for sub in ("control", "vision", "workspace", HERE):
    p = sub if os.path.isabs(sub) else os.path.join(REPO, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from simulate_min_snap import CAMERA_RESOLUTION, CAMERA_HFOV_DEG                 # noqa: E402
from camera import CameraIntrinsics, invert_transform, in_frame                   # noqa: E402
from gate_pose import estimate_gate_pose, project_gate                            # noqa: E402
from eval_isaac_gates import T_body_cam_usd_from_mount                            # noqa: E402
from gate_path import gates_from_manifest, gate_crossings                         # noqa: E402
from New_Drone_State_Equations import get_drone_dynamics                          # noqa: E402
from Runge_Kutta_4_Discretizer import get_rk4_discretizer                         # noqa: E402
from drone_constraints import drone_constraints                                   # noqa: E402
from mpc_core import MpcCore, TrajectoryBuffer, hover_reference                   # noqa: E402
from px4_attitude_plant import Px4AttitudePlant                                   # noqa: E402
from min_snap_planner import plan_course, trajectory_points                       # noqa: E402
from vio import GateVIO, rot_rpy, FRD_FROM_FLU, camera_mount_usd                  # noqa: E402
from ros_camera import T_world_cam_from_transform                                 # noqa: E402

G = 9.81
DEFAULT_MANIFEST = os.path.join(HERE, "isaac_results", "3_gates", "course", "pod_test_gates.json")
GYRO_ND, ACCEL_ND = 0.0003393695767766752, 0.004       # Pegasus imu.py defaults (per sqrt(Hz))
GYRO_TURN_ON_SIGMA = 0.008726646259971648              # rad/s (0.5 deg/s)


def spawn_pose(manifest_path):
    import json
    with open(manifest_path) as f:
        cam = json.load(f)["camera_pose_usd"]
    T_wc = np.eye(4)
    T_wc[:3, :3] = np.asarray(cam["R_world_cam_usd"], dtype=float)
    T_wc[:3, 3] = cam["position"]
    T_wb = T_wc @ invert_transform(T_body_cam_usd_from_mount())
    return T_wb[:3, 3].copy(), math.atan2(T_wb[1, 0], T_wb[0, 0])


class SimCamera:
    def __init__(self, gates, corner_noise_px, range_bias, max_range, rng, mount_usd):
        self.gates = gates
        self.mount = mount_usd
        self.intr = CameraIntrinsics.from_horizontal_fov(*CAMERA_RESOLUTION, CAMERA_HFOV_DEG)
        self.noise, self.bias, self.max_range, self.rng = corner_noise_px, range_bias, max_range, rng

    def detect(self, x_true):
        T_wb = np.eye(4)
        T_wb[:3, :3], T_wb[:3, 3] = rot_rpy(*x_true[3:6]), x_true[0:3]
        T_wc = T_world_cam_from_transform(T_wb @ self.mount, optical_frame=False)
        T_cw = invert_transform(T_wc)
        dets = []
        for g in self.gates:
            T_cg = T_cw @ g.T_world_gate
            if T_cg[2, 3] <= 0.3 or np.linalg.norm(T_cg[:3, 3]) > self.max_range:
                continue
            px = project_gate(T_cg, self.intr, g.side)
            if not np.all(in_frame(px, self.intr)):
                continue
            noisy = px + self.rng.normal(0.0, self.noise, px.shape)
            det = estimate_gate_pose(noisy, self.intr, gate_side=g.side * (1.0 + self.bias),
                                     corner_sigma_px=max(self.noise, 1e-3), bootstrap_n=8,
                                     seed=int(self.rng.integers(1 << 30)))
            if det is not None:
                dets.append(det)
        return dets


def run(args):
    rng = np.random.default_rng(args.seed)
    gates = gates_from_manifest(args.gates)
    start, start_yaw = spawn_pose(args.gates)
    m = drone_constraints["mass"]
    imu_dt = 1.0 / args.imu_rate
    sub = int(round(args.imu_rate / args.mpc_rate))                      # IMU samples per MPC step
    cam_every = int(round(args.imu_rate / args.camera_rate))

    f, xs, us = get_drone_dynamics(m)
    plant = get_rk4_discretizer(f, xs, us, imu_dt)
    core = MpcCore(N=10, dt=1.0 / args.mpc_rate, m=m, attitude_lead_s=args.attitude_lead_s)
    px4 = Px4AttitudePlant(m, tau=args.attitude_tau)
    x_cmd = None
    mount = camera_mount_usd(args.camera_pitch_deg)
    vio = GateVIO(gates, mount, imu_rate=args.imu_rate,
                  initial_guess=(start + rng.normal(0, args.guess_error, 3), start_yaw),
                  delay_compensation=not args.no_delay_compensation)
    camera = SimCamera(gates, args.corner_noise_px, args.range_bias, args.max_range, rng, mount)

    gyro_bias = rng.normal(0.0, GYRO_TURN_ON_SIGMA, 3)                  # FRD, constant
    sg, sa = GYRO_ND / math.sqrt(imu_dt), ACCEL_ND / math.sqrt(imu_dt)

    x = np.zeros((12, 1)); x[0:3, 0] = start; x[5, 0] = start_yaw
    u = np.array([[m * G], [0.0], [0.0], [0.0]])
    pending = []                                                          # (deliver_time, capture_time, dets)
    buf, t_plan, flying = None, None, False
    log = {k: [] for k in ("t", "true", "est", "ref")}
    solve_ms, vio_ms = [], []
    end_t = None
    n_steps = int((args.hover_s + 40.0) / imu_dt)
    t_wall = time.time()
    for k in range(n_steps):
        t = k * imu_dt

        # --- control at 50 Hz (on the VIO estimate only) ---
        if k % sub == 0:
            if buf is None and vio.initialized and t >= args.hover_s:
                est = vio.state[:, 0]
                _, _, ref_obj = plan_course(gates, est[0:3], float(est[5]), args.v_max, args.a_max, yaw_mode=args.yaw_mode,
                                            optimize_times=not args.no_time_opt)
                ts, st, th = trajectory_points(ref_obj)
                buf, t_plan = TrajectoryBuffer(t, ts, st, th), t
                print(f"[plan] t={t:.2f}s from VIO pose {np.round(est[0:3], 2)} (true {np.round(x[0:3, 0], 2)}): "
                      f"{buf.duration:.2f} s")
            if buf is not None:
                flying = True
                xe = vio.state
                ref = buf.horizon(t, core.N, core.dt, yaw_now=float(xe[5, 0]))
                out = core.step(xe, ref)
                u, x_cmd = out["u0"], out["x_cmd"]
                solve_ms.append(out["solve_ms"])
                log["t"].append(t); log["true"].append(x[:, 0].copy()); log["est"].append(xe[:, 0].copy())
                log["ref"].append(ref[:, 0].copy())
                if end_t is None and buf.ended(t):
                    end_t = t
                if end_t is not None and t - end_t > 1.5:
                    break

        # --- true drone (held at the spawn hover until the plan starts) ---
        if flying and args.plant == "px4":
            x_next = px4.step(x, u[0, 0], x_cmd[3:6, 0], imu_dt)
        elif flying:
            x_next = np.array(plant(x, u)).reshape(12, 1)
        else:
            x_next = x.copy()
        a_world = (x_next[6:9, 0] - x[6:9, 0]) / imu_dt
        R = rot_rpy(*x_next[3:6, 0])
        f_body = R.T @ (a_world + np.array([0.0, 0.0, G]))
        acc_frd = FRD_FROM_FLU @ f_body + rng.normal(0.0, sa, 3)
        gyro_frd = FRD_FROM_FLU @ x_next[9:12, 0] + gyro_bias + rng.normal(0.0, sg, 3)
        x = x_next

        t0 = time.perf_counter()
        vio.add_imu(t + imu_dt, acc_frd, gyro_frd)

        # --- camera: capture now, deliver after the latency ---
        if k % cam_every == 0:
            pending.append((t + args.latency, t, camera.detect(x[:, 0])))
        while pending and pending[0][0] <= t + 1e-12:
            _, t_cap, dets = pending.pop(0)
            vio.add_detections(t_cap, dets)
        vio_ms.append(1000 * (time.perf_counter() - t0))

    L = {k: np.array(v) for k, v in log.items()}
    return L, vio, gates, buf, t_plan, gyro_bias, solve_ms, vio_ms, time.time() - t_wall


def report(L, vio, gates, buf, t_plan, gyro_bias, solve_ms, vio_ms, wall):
    if buf is None:
        print("[result] never planned: the VIO did not initialize")
        return False
    ct = vio.calibration_summary
    true_bias_flu = FRD_FROM_FLU @ gyro_bias
    print(f"[vio] gyro bias true {np.round(np.degrees(true_bias_flu), 3)} deg/s, "
          f"estimated {np.round(ct['gyro_bias_deg_s'], 3)}; accel mean |a| {ct['accel_norm']:.3f} m/s^2 at hover")
    print(f"[vio] initialized at t={vio.stats['init_time']:.2f}s; fixes full {vio.stats['fixes_full']}, "
          f"position-only {vio.stats['fixes_pos']}, rejected {vio.stats['rejected']}, unmatched {vio.stats['unmatched']}")
    pe = np.linalg.norm(L["est"][:, 0:3] - L["true"][:, 0:3], axis=1)
    ye = np.degrees(np.abs((L["est"][:, 5] - L["true"][:, 5] + np.pi) % (2 * np.pi) - np.pi))
    ve = np.linalg.norm(L["est"][:, 6:9] - L["true"][:, 6:9], axis=1)
    te = np.linalg.norm(L["true"][:, 0:3] - L["ref"][:, 0:3], axis=1)
    print(f"[vio] position error mean {pe.mean():.3f} m, max {pe.max():.3f} m; velocity error mean {ve.mean():.3f} m/s; "
          f"yaw error max {ye.max():.2f} deg")
    print(f"[mpc] TRUE drone vs reference: mean {te.mean():.3f} m, max {te.max():.3f} m; "
          f"solve median {np.median(solve_ms):.1f} ms")
    print(f"[cpu] VIO per IMU sample: median {np.median(vio_ms):.2f} ms, p99 {np.percentile(vio_ms, 99):.1f} ms "
          f"(250 Hz needs < 4 ms on average); wall {wall:.0f} s")
    ok = True
    print("[result] TRUE path vs gates:")
    for gate, cs in gate_crossings(L["t"], L["true"][:, 0:3], L["true"][:, 6:9], gates):
        through = [c for c in cs if c["inside"] and c["forward"]]
        if not through:
            ok = False
            print(f"  {gate.name}: MISSED")
        for c in through:
            margin = gate.side / 2 - np.max(np.abs(c["offset"]))
            print(f"  {gate.name}: t={c['time'] - t_plan:5.2f}s  offset=({c['offset'][0]:+.2f}, {c['offset'][1]:+.2f}) m  "
                  f"frame margin {margin:.2f} m  angle {c['angle_deg']:4.1f} deg")
    print(f"[result] {'ALL GATES PASSED' if ok else 'NOT ALL GATES PASSED'}")
    return ok


def plot(L, buf, gates, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.5))
    ax = axs[0]
    for g in gates:
        c = g.corners()
        ax.plot(c[[0, 1], 1], c[[0, 1], 0], color="darkorange", linewidth=4)
        ax.annotate(g.name, (g.center[1], g.center[0]), xytext=(6, -4), textcoords="offset points", fontsize=8)
    ax.plot(buf.states[1], buf.states[0], "k--", label="plan")
    ax.plot(L["true"][:, 1], L["true"][:, 0], "b-", label="true drone")
    ax.plot(L["est"][:, 1], L["est"][:, 0], color="crimson", linewidth=1, alpha=0.8, label="VIO estimate")
    ax.invert_xaxis(); ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_xlabel("Y (m)  <- left | right ->"); ax.set_ylabel("X forward (m)"); ax.set_title("Top-down (camera orientation)")
    t = L["t"] - L["t"][0]
    axs[1].plot(t, np.linalg.norm(L["est"][:, 0:3] - L["true"][:, 0:3], axis=1), color="crimson", label="VIO position error")
    axs[1].plot(t, np.linalg.norm(L["true"][:, 0:3] - L["ref"][:, 0:3], axis=1), "b-", label="true drone vs reference")
    axs[1].set_xlabel("time since plan (s)"); axs[1].set_ylabel("m"); axs[1].grid(alpha=0.3); axs[1].legend(fontsize=8)
    axs[1].set_title("Errors")
    axs[2].plot(t, np.degrees(L["est"][:, 5] - L["true"][:, 5]), label="yaw")
    axs[2].plot(t, np.degrees(L["est"][:, 3] - L["true"][:, 3]), label="roll")
    axs[2].plot(t, np.degrees(L["est"][:, 4] - L["true"][:, 4]), label="pitch")
    axs[2].set_xlabel("time since plan (s)"); axs[2].set_ylabel("deg"); axs[2].grid(alpha=0.3); axs[2].legend(fontsize=8)
    axs[2].set_title("VIO attitude error")
    fig.suptitle("VIO + MPC rehearsal on the 3-gate course (IMU + camera only)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"[plot] {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gates", default=DEFAULT_MANIFEST)
    ap.add_argument("--imu-rate", type=float, default=250.0)
    ap.add_argument("--mpc-rate", type=float, default=50.0)
    ap.add_argument("--camera-rate", type=float, default=15.0)
    ap.add_argument("--latency", type=float, default=0.05, help="camera capture -> VIO, s (sim time)")
    ap.add_argument("--corner-noise-px", type=float, default=0.7)
    ap.add_argument("--range-bias", type=float, default=0.01, help="PnP range scale error (live sim: ~1%%)")
    ap.add_argument("--max-range", type=float, default=15.0)
    ap.add_argument("--guess-error", type=float, default=0.3, help="error of the spawn-pose hint (m)")
    ap.add_argument("--hover-s", type=float, default=3.0)
    ap.add_argument("--camera-pitch-deg", type=float, default=15.0,
                    help="front camera up-tilt (spawn_example.py today: 15)")
    ap.add_argument("--yaw-mode", choices=["velocity", "next_gate"], default="next_gate",
                    help="nose along the velocity, or at the next gate (keeps gates in view)")
    ap.add_argument("--v-max", type=float, default=None, help="default: 80%% of the drone's max (9.6 m/s)")
    ap.add_argument("--a-max", type=float, default=None, help="default: 80%% of the drone's max (12 m/s^2)")
    ap.add_argument("--no-time-opt", action="store_true", help="distance-proportional segment times (old planner)")
    ap.add_argument("--no-delay-compensation", action="store_true")
    ap.add_argument("--plant", choices=["px4", "torque"], default="px4",
                    help="px4: attitude setpoint through a lagging attitude loop (Isaac); torque: the MPC's own model")
    ap.add_argument("--attitude-tau", type=float, default=0.08, help="--plant px4: attitude lag, s")
    ap.add_argument("--attitude-lead-s", type=float, default=0.1, help="MPC attitude setpoint lead, s (mpc_node default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "sim_output", "vio_mpc_rehearsal.png"))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    res = run(args)
    ok = report(*res)
    if not args.no_plot and res[3] is not None:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        plot(res[0], res[3], res[2], args.out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
