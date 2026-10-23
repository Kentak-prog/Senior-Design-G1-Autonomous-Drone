"""
Plots for the 3-gate Isaac Sim course (3_gates/course/test_gates.json).

Re-runs the same pipeline as simulate_min_snap.py (synthetic perception ->
min-snap plan, seed 0) and draws two separate figures, both oriented like
the drone's camera view in Isaac: forward (world +X) points up the page and
the drone's left (world +Y) is on the left, so gate_0 shows left and gate_1
right, as they appear in the front_cam viewport.

    python_sim/min_snap_plan.png   the planned min-snap trajectory only
    python_sim/mpc_course.png      the NMPC closed-loop flight of that plan (--mpc)

The NMPC run takes a few minutes, so its log is cached in python_sim/mpc_log.npz
and reused on later runs unless --mpc is passed again.

Run from Senior-Design-G1-Autonomous-Drone/Trajectory:
    python "isaac results/3_gates/scripts/plot_3_gates.py"          # plan figure (+ MPC figure if cached)
    python "isaac results/3_gates/scripts/plot_3_gates.py" --mpc    # also fly the NMPC and cache it
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))           # 3_gates/scripts
BASE = os.path.dirname(HERE)                                 # 3_gates
COURSE_MANIFEST = os.path.join(BASE, "course", "test_gates.json")
SIM_DIR = os.path.join(BASE, "python_sim")
TRAJ_DIR = os.path.abspath(os.path.join(BASE, "..", ".."))
sys.path.insert(0, TRAJ_DIR)

import simulate_min_snap as sim                                          # noqa: E402
from gate_path import gates_from_manifest, plan_gate_trajectory, gate_crossings  # noqa: E402
from trajectory_reference import TrajectoryReference                     # noqa: E402

# Live hover pose read from /World/Iris/body when the gates were spawned.
START = np.array([-0.0469, -0.0367, 2.5484])
HEADING_DEG = -7.24
CORNER_NOISE_PX = 0.5
SEED = 0
DT = 0.02
HORIZON = 10

GATE_COLOR = 'darkorange'
REF_STYLE = dict(color='k', linestyle='--', linewidth=1.5)
FLOWN_STYLE = dict(color='royalblue', linestyle='-', linewidth=1.8)


# ---------------------------------------------------------------------------
# Pipeline (same steps and seed as simulate_min_snap.main)
# ---------------------------------------------------------------------------

def build_plan():
    T_world_cam_usd, T_world_cam = sim.camera_poses(sim.body_pose(START, HEADING_DEG))
    true_gates = gates_from_manifest(COURSE_MANIFEST)
    est_gates = sim.perceive_gates(true_gates, T_world_cam, CORNER_NOISE_PX, SEED)
    traj, wps = plan_gate_trajectory(est_gates, START, v_max=4.0, a_max=6.0)
    ref = TrajectoryReference(traj, start_yaw=np.radians(HEADING_DEG), m=sim.drone_constraints["mass"])
    return true_gates, est_gates, traj, wps, ref


def load_or_fly(ref, fly: bool):
    path = os.path.join(SIM_DIR, "mpc_log.npz")
    if fly:
        print(f"\n[sim] flying with NMPC (N={HORIZON}, dt={DT}s, noise on) ...")
        log = sim.fly_with_mpc(ref, DT, HORIZON, add_noise=True)
        np.savez(path, **{k: np.asarray(v) for k, v in log.items()})
        print(f"[sim] saved {path}")
        return log
    if os.path.exists(path):
        print(f"[sim] using cached {path}")
        return dict(np.load(path))
    return None


# ---------------------------------------------------------------------------
# Isaac-view drawing helpers: page-right = drone's right = world -Y,
# page-up = forward = world +X.
# ---------------------------------------------------------------------------

def view_3d(ax):
    """Look down the course from behind and above the start, camera-style."""
    ax.view_init(elev=22, azim=180 + HEADING_DEG)


def draw_gates_3d(ax, gates, color=GATE_COLOR, alpha=0.25, label='Gates (normal)'):
    for i, g in enumerate(gates):
        ax.add_collection3d(Poly3DCollection([g.corners()], facecolor=color, alpha=alpha,
                                             edgecolor=color, linewidth=2))
        tip = g.center + 1.0 * g.normal
        ax.plot(*zip(g.center, tip), color=color, linewidth=1.5, label=label if i == 0 else None)
        ax.text(*(g.center + np.array([0, 0, g.side / 2 + 0.3])), g.name, fontsize=9)


def setup_3d(ax, pts, title):
    # Tight per-axis limits at true scale (box aspect = data extents), so the
    # long, narrow, low course is not squeezed into a corner of a cube.
    pts = np.asarray(pts)
    lo = pts.min(axis=0) - 1.0
    hi = pts.max(axis=0) + 1.0
    lo[2] = 0.0
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo, zoom=1.15)
    ax.set_xlabel("X forward (m)"); ax.set_ylabel("Y left (m)"); ax.set_zlabel("Z up (m)")
    ax.set_title(title, fontweight='bold')
    view_3d(ax)


def draw_top_down(ax, gates, start, title):
    """World +X up the page, world +Y to the left (matches the front camera)."""
    for g in gates:
        c = g.corners()
        ax.plot(c[[0, 1], 1], c[[0, 1], 0], color=GATE_COLOR, linewidth=4, solid_capstyle='butt')
        ax.annotate(g.name, (g.center[1], g.center[0]), xytext=(8, -4),
                    textcoords='offset points', fontsize=9)
    h = np.radians(HEADING_DEG)
    ax.annotate('', xy=(start[1] + 1.2 * np.sin(h), start[0] + 1.2 * np.cos(h)),
                xytext=(start[1], start[0]), arrowprops=dict(color='green', width=2, headwidth=8))
    ax.plot(start[1], start[0], 'go', label='Start')
    ax.set_aspect('equal', adjustable='datalim')
    ax.invert_xaxis()                       # +Y (drone's left) on the page's left
    ax.set_xlabel("Y (m)   ← left | right →"); ax.set_ylabel("X forward (m)")
    ax.grid(alpha=0.3); ax.set_title(title)


def draw_side(ax, gates, title):
    """Height along the course: forward X across the page, Z up."""
    for g in gates:
        c = g.corners()
        zlo, zhi = c[:, 2].min(), c[:, 2].max()
        ax.plot([g.center[0], g.center[0]], [zlo, zhi], color=GATE_COLOR, linewidth=4)
        ax.annotate(g.name, (g.center[0], zhi), xytext=(-12, 4), textcoords='offset points', fontsize=9)
    ax.set_xlabel("X forward (m)"); ax.set_ylabel("Z up (m)")
    ax.grid(alpha=0.3); ax.set_title(title)


def mark_gate_times(ax, wps, traj):
    for w, tb in zip(wps, traj.t_breaks):
        if w['type'] == 'center':
            ax.axvline(tb, color=GATE_COLOR, alpha=0.5)


def crossing_lines(times, pos, vel, gates):
    lines, all_ok = [], True
    for gate, crossings in gate_crossings(times, pos, vel, gates):
        through = [c for c in crossings if c['inside'] and c['forward']]
        if not through:
            all_ok = False
            lines.append(f"{gate.name}: MISSED")
        for c in through:
            margin = gate.side / 2 - np.max(np.abs(c['offset']))
            lines.append(f"{gate.name}: t={c['time']:4.2f}s  off=({c['offset'][0]:+.2f},{c['offset'][1]:+.2f}) m  "
                         f"margin {margin:.2f} m  {c['angle_deg']:4.1f}°  {c['speed']:.2f} m/s")
    return lines, all_ok


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_plan(true_gates, est_gates, traj, wps, ref, out_path):
    times, pos, vel, acc, *_ = traj.sample(0.02)
    wp = np.array([w['position'] for w in wps])
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Minimum-snap plan through the 3 Isaac gates  "
                 f"(vision {CORNER_NOISE_PX} px noise → plan; {traj.duration:.2f} s, "
                 f"peak {np.linalg.norm(vel, axis=1).max():.2f} m/s)", fontweight='bold')

    ax = fig.add_subplot(2, 3, (1, 4), projection='3d')
    draw_gates_3d(ax, true_gates, label='True gates (normal)')
    draw_gates_3d(ax, est_gates, color='royalblue', alpha=0.08, label='Perceived gates (normal)')
    ax.scatter(*wp.T, color='white', edgecolors='black', s=30, label='Waypoints')
    ax.plot(*pos.T, **REF_STYLE, label='Min-snap trajectory')
    ax.scatter(*START, color='green', s=40, label='Start (hover)')
    setup_3d(ax, np.vstack([pos] + [g.corners() for g in true_gates]), "3D (view from behind the drone)")
    ax.legend(loc='upper left', fontsize=8)

    ax2 = fig.add_subplot(2, 3, 2)
    draw_top_down(ax2, true_gates, START, "Top-down (camera orientation)")
    ax2.plot(pos[:, 1], pos[:, 0], **REF_STYLE, label='Min-snap')
    ax2.scatter(wp[:, 1], wp[:, 0], color='white', edgecolors='black', s=20, zorder=3, label='Waypoints')
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(2, 3, 3)
    draw_side(ax3, true_gates, "Side view (height)")
    ax3.plot(pos[:, 0], pos[:, 2], **REF_STYLE)

    ax4 = fig.add_subplot(2, 3, 5)
    ax4.plot(times, np.linalg.norm(vel, axis=1), 'k-', label='speed (m/s)')
    ax4.plot(times, np.linalg.norm(acc, axis=1), color='C3', alpha=0.8, label='accel (m/s²)')
    mark_gate_times(ax4, wps, traj)
    ax4.set_xlabel("Time (s)"); ax4.grid(alpha=0.3); ax4.legend(fontsize=8)
    ax4.set_title("Speed and acceleration (orange = gate centers)")

    ax5 = fig.add_subplot(2, 3, 6)
    for i, (lbl, col) in enumerate(zip(['roll', 'pitch', 'yaw'], ['C0', 'C1', 'C2'])):
        ax5.plot(ref.times, np.degrees(ref.states[3 + i]), '-', color=col, label=lbl)
    mark_gate_times(ax5, wps, traj)
    ax5.set_xlabel("Time (s)"); ax5.set_ylabel("deg"); ax5.grid(alpha=0.3); ax5.legend(fontsize=8)
    ax5.set_title("Reference attitude")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_mpc(true_gates, traj, wps, ref, log, out_path):
    t = log['t']
    flown = log['true']
    ref_h = ref.build_reference_horizon(0.0, len(t) - 1, t[1] - t[0])
    err = np.linalg.norm(flown[0:3] - ref_h[0:3], axis=0)
    lines, ok = crossing_lines(t, flown[0:3].T, flown[6:9].T, true_gates)
    _, pos, vel, *_ = traj.sample(0.02)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Simulated NMPC flight of the 3-gate min-snap plan  "
                 f"({'ALL GATES PASSED' if ok else 'NOT ALL GATES PASSED'}; tracking error mean "
                 f"{err.mean():.2f} m, max {err.max():.2f} m)", fontweight='bold')

    ax = fig.add_subplot(2, 3, (1, 4), projection='3d')
    draw_gates_3d(ax, true_gates)
    ax.plot(*pos.T, **REF_STYLE, label='Min-snap reference')
    ax.plot(*flown[0:3], **FLOWN_STYLE, label='Flown (NMPC)')
    ax.scatter(*START, color='green', s=40, label='Start (hover)')
    setup_3d(ax, np.vstack([pos, flown[0:3].T] + [g.corners() for g in true_gates]),
             "3D (view from behind the drone)")
    ax.legend(loc='upper left', fontsize=8)

    ax2 = fig.add_subplot(2, 3, 2)
    draw_top_down(ax2, true_gates, START, "Top-down (camera orientation)")
    ax2.plot(pos[:, 1], pos[:, 0], **REF_STYLE, label='Reference')
    ax2.plot(flown[1], flown[0], **FLOWN_STYLE, label='Flown')
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(2, 3, 3)
    draw_side(ax3, true_gates, "Side view (height)")
    ax3.plot(pos[:, 0], pos[:, 2], **REF_STYLE, label='Reference')
    ax3.plot(flown[0], flown[2], **FLOWN_STYLE, label='Flown')
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(2, 3, 5)
    ax4.plot(t, np.linalg.norm(ref_h[6:9], axis=0), 'k--', label='Reference speed')
    ax4.plot(t, np.linalg.norm(flown[6:9], axis=0), color='royalblue', label='Flown speed')
    mark_gate_times(ax4, wps, traj)
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("m/s"); ax4.grid(alpha=0.3); ax4.legend(fontsize=8)
    ax4.set_title("Speed (orange = planned gate times)")

    ax5 = fig.add_subplot(2, 3, 6)
    ax5.plot(t, err, 'r-')
    ax5.set_ylim(0.0, err.max() * 1.6)      # headroom so the gate-pass table sits above the curve
    mark_gate_times(ax5, wps, traj)
    ax5.set_xlabel("Time (s)"); ax5.set_ylabel("m"); ax5.grid(alpha=0.3)
    ax5.set_title("Position tracking error")
    ax5.text(0.02, 0.97, "Gate passes (flown vs true gates):\n" + "\n".join(lines),
             transform=ax5.transAxes, va='top', fontsize=7.5, family='monospace',
             bbox=dict(facecolor='white', alpha=0.85, edgecolor='0.7'))

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")
    print("\n".join(["[sim] flown path vs true gates:"] + ["  " + s for s in lines]))
    print(f"[sim] tracking error mean {err.mean():.3f} m, max {err.max():.3f} m, final {err[-1]:.3f} m")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mpc", action="store_true", help="fly the NMPC (minutes) and cache mpc_log.npz")
    args = ap.parse_args()

    true_gates, est_gates, traj, wps, ref = build_plan()
    plot_plan(true_gates, est_gates, traj, wps, ref, os.path.join(SIM_DIR, "min_snap_plan.png"))
    log = load_or_fly(ref, args.mpc)
    if log is not None:
        plot_mpc(true_gates, traj, wps, ref, log, os.path.join(SIM_DIR, "mpc_course.png"))


if __name__ == "__main__":
    main()
