"""
Artificial end-to-end test of the minimum-snap trajectory, from a given start
position + heading:

  1. Pose:       drone body at (x, y, z) with the given heading (deg, CCW from
                 world +X, Z-up). Camera pose = body pose @ the static mount
                 from eval_isaac_gates.T_body_cam_usd_from_mount() (mirrors
                 workspace/spawn_example.py), converted to OpenCV with
                 ros_camera.T_world_cam_from_transform(optical_frame=False).
  2. Gates:      either a manifest (--gates test_gates.json) or an artificial
                 course placed relative to the camera with the same geometry
                 spawn_test_gates.py uses in Isaac Sim (APPROACH or STAGGERED
                 layout).
  3. Perception: (--perception synthetic) each gate's corners are projected
                 into a 1280x720 / 60 deg camera, jittered by --corner-noise-px,
                 and run through the real vision pipeline: estimate_gate_pose
                 (PnP) -> gate_pose_world. Gates out of frame fall back to the
                 true pose. (--perception truth) skips this.
  4. Planning:   gate_path.plan_gate_trajectory -> minimum-snap trajectory.
  5. Control:    the trajectory is turned into the 12-state NMPC reference
                 (trajectory_reference.py) and flown in closed loop by the
                 existing control/ stack -- Drone_Race_Simulation's loop,
                 MPC_Optimizer, the RK4 plant and the EKF -- unchanged.
  6. Results:    gate passes of the FLOWN path against the TRUE gates,
                 tracking error, plots and a reference CSV.

Example:
    python simulate_min_snap.py --start 0 0 2.5 --heading 0
    python simulate_min_snap.py --start 3 -2 2.0 --heading 45 --layout staggered
    python simulate_min_snap.py --start 0 0 2.6 --heading 0 --gates /workspace/test_gates.json
    python simulate_min_snap.py --start 0 0 2.5 --heading 0 --no-mpc     # plan only, fast
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for sub in ("vision", "control", "workspace"):
    p = os.path.join(REPO, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from camera import CameraIntrinsics, make_transform, invert_transform, in_frame     # noqa: E402
from gate_pose import estimate_gate_pose, gate_pose_world, project_gate             # noqa: E402
from ros_camera import T_world_cam_from_transform                                   # noqa: E402
from eval_isaac_gates import T_body_cam_usd_from_mount, T_world_gate_from_spec      # noqa: E402
import spawn_test_gates                                                             # noqa: E402

from gate_path import Gate, gates_from_manifest, plan_gate_trajectory, gate_crossings   # noqa: E402
from drone_constraints import drone_constraints                                    # noqa: E402
from trajectory_reference import TrajectoryReference                                # noqa: E402

CAMERA_RESOLUTION = (1280, 720)    # KEEP IN SYNC with workspace/spawn_example.py
CAMERA_HFOV_DEG = 60.0


# ---------------------------------------------------------------------------
# 1. Start pose and camera
# ---------------------------------------------------------------------------

def body_pose(xyz, heading_deg) -> np.ndarray:
    """T_world_body for a level drone (Isaac body: +X forward, +Y left, +Z up)."""
    yaw = np.radians(heading_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return make_transform(R, xyz)


def camera_poses(T_world_body):
    """(T_world_cam_usd, T_world_cam in OpenCV convention) for the static front-camera mount."""
    T_world_cam_usd = T_world_body @ T_body_cam_usd_from_mount()
    return T_world_cam_usd, T_world_cam_from_transform(T_world_cam_usd, optical_frame=False)


# ---------------------------------------------------------------------------
# 2. Gates
# ---------------------------------------------------------------------------

def artificial_gates(T_world_cam_usd, layout: str):
    """Place a course relative to the camera exactly like spawn_test_gates.py does in Isaac Sim."""
    R, t = T_world_cam_usd[:3, :3], T_world_cam_usd[:3, 3]
    cam = {"position": t, "right": R[:, 0], "up": R[:, 1], "fwd": -R[:, 2]}
    specs, level = {"approach": (spawn_test_gates.APPROACH_GATE_SPECS, True),
                    "staggered": (spawn_test_gates.GATE_SPECS, False)}[layout]
    gates = []
    for i, spec in enumerate(sorted(specs, key=lambda s: s[0])):   # fly them nearest-first
        g = spawn_test_gates.gate_world_pose(spec, cam, name=f"gate_{i}", level=level)
        g, note = spawn_test_gates.clamp_to_ground(g)
        if note:
            print(f"[gates] {note}")
        gates.append(Gate(g["name"], T_world_gate_from_spec(g["x"], g["y"], g["z"], g["yaw_deg"]),
                          side=spawn_test_gates.GATE_SIDE))
    return gates


# ---------------------------------------------------------------------------
# 3. Synthetic perception through the real vision pipeline
# ---------------------------------------------------------------------------

def perceive_gates(true_gates, T_world_cam, corner_noise_px: float, seed: int = 0):
    """Project each true gate into the camera, add corner noise, solve PnP, lift to world."""
    rng = np.random.default_rng(seed)
    intr = CameraIntrinsics.from_horizontal_fov(*CAMERA_RESOLUTION, CAMERA_HFOV_DEG)
    T_cam_world = invert_transform(T_world_cam)

    est = []
    print("\n[vision] synthetic detections (1280x720, 60 deg hFOV, "
          f"corner noise {corner_noise_px} px):")
    for g in true_gates:
        px = project_gate(T_cam_world @ g.T_world_gate, intr, g.side)
        if not np.all(in_frame(px, intr)):
            print(f"  {g.name}: out of frame -> using TRUE pose")
            est.append(g)
            continue

        noisy = px + rng.normal(0.0, corner_noise_px, px.shape)
        det = estimate_gate_pose(noisy, intr, gate_side=g.side,
                                 corner_sigma_px=max(corner_noise_px, 1e-3), seed=seed)
        if det is None:
            print(f"  {g.name}: PnP failed -> using TRUE pose")
            est.append(g)
            continue

        T_est = gate_pose_world(det, T_world_cam)
        gate = Gate(g.name, T_est, side=g.side, normal_reliable=det.normal_is_reliable)
        pos_err = np.linalg.norm(gate.center - g.center)
        ang_err = np.degrees(np.arccos(np.clip(gate.normal @ g.normal, -1, 1)))
        print(f"  {g.name}: range {det.range_m:5.2f} m  pos err {pos_err:.3f} m  "
              f"normal err {ang_err:4.1f} deg  (bootstrap std {det.normal_std_deg:4.1f} deg, "
              f"{'reliable' if det.normal_is_reliable else 'UNRELIABLE -> center only'})")
        est.append(gate)
    return est


# ---------------------------------------------------------------------------
# 5. Closed loop with the existing control stack
# ---------------------------------------------------------------------------

def fly_with_mpc(ref: TrajectoryReference, dt: float, N: int, add_noise: bool, m: float = drone_constraints["mass"],
                 tail: float = 1.0):
    """
    Run control/Drone_Race_Simulation.run_race_simulation unchanged, but with
    its reference-track functions pointed at the min-snap reference. That loop
    already does NMPC solve -> RK4 plant -> noisy measurement -> EKF.
    """
    import Drone_Race_Simulation as race

    race.reference_state = ref.reference_state
    race.build_reference_horizon = ref.build_reference_horizon
    race.LAP_TIME = ref.duration + tail      # fly the trajectory, then hover at the finish
    return race.run_race_simulation(num_laps=1.0, dt=dt, N=N, add_noise=add_noise, m=m)


# ---------------------------------------------------------------------------
# 6. Reporting
# ---------------------------------------------------------------------------

def report_crossings(label, times, pos, vel, gates):
    print(f"\n{label}")
    all_ok = True
    for gate, crossings in gate_crossings(times, pos, vel, gates):
        through = [c for c in crossings if c['inside'] and c['forward']]
        others = [c for c in crossings if not (c['inside'] and c['forward'])]
        if not through:
            all_ok = False
            print(f"  {gate.name}: MISSED")
        for c in through:
            margin = gate.side / 2 - np.max(np.abs(c['offset']))
            print(f"  {gate.name}: t={c['time']:5.2f}s  offset=({c['offset'][0]:+.2f}, {c['offset'][1]:+.2f}) m  "
                  f"frame margin {margin:.2f} m  angle {c['angle_deg']:4.1f} deg  speed {c['speed']:.2f} m/s")
        for c in others:
            what = "BACKWARDS through" if c['inside'] else \
                f"past the frame ({np.max(np.abs(c['offset'])) - gate.side / 2:.2f} m outside)"
            print(f"  {gate.name}: t={c['time']:5.2f}s  {what}")
    return all_ok


def save_reference_csv(ref: TrajectoryReference, path, dt=0.02):
    """Time-parameterized reference for the controller: t + 12 NMPC states + feed-forward thrust."""
    ts = np.arange(0.0, ref.duration + 1e-9, dt)
    states = ref.build_reference_horizon(0.0, len(ts) - 1, dt)
    thrust = [ref.feedforward_thrust(t) for t in ts]
    header = "t,x,y,z,phi,theta,psi,vx,vy,vz,p,q,r,U1_ff"
    np.savetxt(path, np.column_stack((ts, states.T, thrust)), delimiter=",",
               header=header, comments="", fmt="%.6f")


def draw_gates(ax, gates, color, alpha, label, normal_len=1.0):
    for i, g in enumerate(gates):
        ax.add_collection3d(Poly3DCollection([g.corners()], facecolor=color, alpha=alpha,
                                             edgecolor=color, linewidth=2))
        tip = g.center + normal_len * g.normal
        ax.plot(*zip(g.center, tip), color=color, linewidth=1.5, label=label if i == 0 else None)
        ax.text(*(g.center + np.array([0, 0, g.side / 2 + 0.3])), g.name, fontsize=8)


def set_equal_axes(ax, pts):
    pts = np.asarray(pts)
    mid = (pts.max(axis=0) + pts.min(axis=0)) / 2
    half = (pts.max(axis=0) - pts.min(axis=0)).max() / 2 + 0.5
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(max(0.0, mid[2] - half), mid[2] + half)


def plot_results(true_gates, est_gates, wps, traj, ref, log, start, heading_deg, out_path):
    times, pos, vel, *_ = traj.sample(0.02)
    fig = plt.figure(figsize=(16, 10))

    # 3D view
    ax = fig.add_subplot(2, 3, (1, 4), projection='3d')
    draw_gates(ax, true_gates, 'darkorange', 0.25, 'True gates (normal)')
    if est_gates is not true_gates:
        draw_gates(ax, est_gates, 'royalblue', 0.08, 'Perceived gates (normal)')
    wp = np.array([w['position'] for w in wps])
    ax.scatter(*wp.T, color='white', edgecolors='black', s=30, label='Waypoints')
    ax.plot(*pos.T, 'k--', linewidth=1.5, label='Min-snap reference')
    if log is not None:
        ax.plot(*log['true'][0:3], 'b-', linewidth=1.5, label='Flown (NMPC)')
    h = np.radians(heading_deg)
    ax.quiver(*start, np.cos(h), np.sin(h), 0, length=1.5, color='green', linewidth=2, label='Start heading')
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    set_equal_axes(ax, np.vstack([pos] + [g.corners() for g in true_gates]))
    ax.set_title("Minimum-snap gate trajectory", fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)

    # Top-down
    ax2 = fig.add_subplot(2, 3, 2)
    for g in true_gates:
        c = g.corners()
        ax2.plot(c[[0, 1], 0], c[[0, 1], 1], color='darkorange', linewidth=3)
        ax2.annotate(g.name, g.center[:2], fontsize=8)
    ax2.plot(pos[:, 0], pos[:, 1], 'k--', label='Reference')
    if log is not None:
        ax2.plot(log['true'][0], log['true'][1], 'b-', label='Flown')
    ax2.plot(*start[:2], 'go')
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)"); ax2.axis('equal'); ax2.grid(alpha=0.3)
    ax2.set_title("Top-down"); ax2.legend(fontsize=8)

    # Speed
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(times, np.linalg.norm(vel, axis=1), 'k--', label='Reference')
    if log is not None:
        ax3.plot(log['t'], np.linalg.norm(log['true'][6:9], axis=0), 'b-', label='Flown')
    for w, tb in zip(wps, traj.t_breaks):
        if w['type'] == 'center':
            ax3.axvline(tb, color='darkorange', alpha=0.5)
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Speed (m/s)"); ax3.grid(alpha=0.3)
    ax3.set_title("Speed (orange = gate centers)"); ax3.legend(fontsize=8)

    # Attitude reference vs flown
    ax4 = fig.add_subplot(2, 3, 5)
    labels = ['roll', 'pitch', 'yaw']
    for i, (lbl, col) in enumerate(zip(labels, ['C0', 'C1', 'C2'])):
        ax4.plot(ref.times, np.degrees(ref.states[3 + i]), '--', color=col, label=f'{lbl} ref')
        if log is not None:
            ax4.plot(log['t'], np.degrees(log['true'][3 + i]), '-', color=col, alpha=0.7, label=f'{lbl} flown')
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("deg"); ax4.grid(alpha=0.3)
    ax4.set_title("Attitude"); ax4.legend(fontsize=7, ncol=2)

    # Tracking error
    ax5 = fig.add_subplot(2, 3, 6)
    if log is not None:
        ref_pos = ref.build_reference_horizon(0.0, len(log['t']) - 1, log['t'][1] - log['t'][0])[0:3]
        err = np.linalg.norm(log['true'][0:3] - ref_pos, axis=0)
        ax5.plot(log['t'], err, 'r-')
        ax5.set_title(f"Tracking error (mean {err.mean():.2f} m, max {err.max():.2f} m)")
    else:
        ax5.set_title("Tracking error (no MPC run)")
    ax5.set_xlabel("Time (s)"); ax5.set_ylabel("m"); ax5.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved plot to {out_path}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=float, nargs=3, default=[0.0, 0.0, 2.5], metavar=("X", "Y", "Z"),
                    help="start position of the drone body, world Z-up, meters")
    ap.add_argument("--heading", type=float, default=0.0, help="start heading, deg CCW from world +X")
    ap.add_argument("--gates", default=None, help="test_gates.json manifest (default: artificial course)")
    ap.add_argument("--layout", choices=["approach", "staggered"], default="approach",
                    help="artificial course layout (spawn_test_gates.py specs)")
    ap.add_argument("--perception", choices=["synthetic", "truth"], default="synthetic")
    ap.add_argument("--corner-noise-px", type=float, default=0.5)
    ap.add_argument("--v-max", type=float, default=4.0)
    ap.add_argument("--a-max", type=float, default=6.0)
    ap.add_argument("--mass", type=float, default=drone_constraints["mass"],
                    help="drone mass (kg); default from control/drone_constraints.py")
    ap.add_argument("--dt", type=float, default=0.02, help="control period (s)")
    ap.add_argument("--horizon", type=int, default=10, help="NMPC horizon N")
    ap.add_argument("--no-noise", action="store_true", help="no process/measurement noise in the sim")
    ap.add_argument("--no-mpc", action="store_true", help="plan only, skip the closed-loop flight")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "sim_output"))
    ap.add_argument("--show", action="store_true", help="open the plot window at the end")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    os.makedirs(args.out_dir, exist_ok=True)
    start = np.array(args.start, dtype=float)

    # 1. pose
    T_world_body = body_pose(start, args.heading)
    T_world_cam_usd, T_world_cam = camera_poses(T_world_body)
    print(f"[start] body at {np.round(start, 2)}, heading {args.heading:.1f} deg; "
          f"camera at {np.round(T_world_cam_usd[:3, 3], 2)}")

    # 2. gates
    if args.gates:
        true_gates = gates_from_manifest(args.gates)
        print(f"[gates] {len(true_gates)} gates from {args.gates}")
    else:
        true_gates = artificial_gates(T_world_cam_usd, args.layout)
        print(f"[gates] artificial '{args.layout}' course, {len(true_gates)} gates:")
    for g in true_gates:
        yaw = np.degrees(np.arctan2(g.normal[1], g.normal[0]))
        print(f"  {g.name}: center {np.round(g.center, 2)}  flight heading {yaw:+.1f} deg")

    # 3. perception
    est_gates = (perceive_gates(true_gates, T_world_cam, args.corner_noise_px, args.seed)
                 if args.perception == "synthetic" else true_gates)

    # 4. plan
    traj, wps = plan_gate_trajectory(est_gates, start, v_max=args.v_max, a_max=args.a_max)
    _, _, vel, acc, _, _ = traj.sample(0.005)
    print(f"\n[plan] {len(wps)} waypoints, {traj.n_seg} segments, duration {traj.duration:.2f} s, "
          f"peak speed {np.linalg.norm(vel, axis=1).max():.2f} m/s, "
          f"peak accel {np.linalg.norm(acc, axis=1).max():.2f} m/s^2")

    ref = TrajectoryReference(traj, start_yaw=np.radians(args.heading), m=args.mass)
    print(f"[plan] reference attitude: max |roll| {np.degrees(np.abs(ref.states[3]).max()):.1f} deg, "
          f"max |pitch| {np.degrees(np.abs(ref.states[4]).max()):.1f} deg, "
          f"thrust {ref.thrust.min():.2f}-{ref.thrust.max():.2f} N")
    csv_path = os.path.join(args.out_dir, "min_snap_reference.csv")
    save_reference_csv(ref, csv_path, dt=args.dt)
    print(f"[plan] saved NMPC reference to {csv_path}")

    t_s, p_s, v_s, *_ = traj.sample(0.005)
    report_crossings("[plan] planned trajectory vs TRUE gates:", t_s, p_s, v_s, true_gates)

    # 5. fly
    log = None
    if not args.no_mpc:
        print(f"\n[sim] flying with NMPC (N={args.horizon}, dt={args.dt}s, "
              f"mass {args.mass} kg, noise {'off' if args.no_noise else 'on'}) ...")
        t0 = time.time()
        log = fly_with_mpc(ref, args.dt, args.horizon, add_noise=not args.no_noise, m=args.mass)
        print(f"[sim] done in {time.time() - t0:.1f} s wall time")

        ref_pos = ref.build_reference_horizon(0.0, len(log['t']) - 1, args.dt)[0:3]
        err = np.linalg.norm(log['true'][0:3] - ref_pos, axis=0)
        print(f"[sim] position tracking error: mean {err.mean():.3f} m, max {err.max():.3f} m, "
              f"final {err[-1]:.3f} m")
        ok = report_crossings("[sim] FLOWN path vs TRUE gates:", log['t'], log['true'][0:3].T,
                              log['true'][6:9].T, true_gates)
        print(f"\n[result] {'ALL GATES PASSED' if ok else 'NOT ALL GATES PASSED'}")

    plot_results(true_gates, est_gates, wps, traj, ref, log, start, args.heading,
                 os.path.join(args.out_dir, "min_snap_sim.png"))
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
