#!/usr/bin/env python3
"""
Animated GIF of an Isaac MPC flight from ros2_nodes/flight_monitor.py's log,
played in real (sim) time: top-down path through the gates, speed, and
tracking error, with a cursor moving along both time plots.

    python animate_flight.py ../vio_mpc/run_045716_s4
    python animate_flight.py <run_dir> --gates <test_gates.json> --fps 10 --out flight.gif

<run_dir> holds flight_log.csv (and test_gates.json unless --gates is given).
Time 0 is when the plan started (the bridge engaged OFFBOARD).
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Trajectory"))

from gate_path import gates_from_manifest, gate_crossings      # noqa: E402

# reference palette (dataviz skill): surface, ink, categorical slots 1-2 in fixed order
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
DRONE, GATE = "#2a78d6", "#eb6834"


def load(run_dir):
    with open(os.path.join(run_dir, "flight_log.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    col = lambda k: np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows])  # noqa: E731
    L = {k: col(k) for k in ("t", "traj_t", "true_x", "true_y", "true_z", "true_yaw", "true_vx", "true_vy",
                             "true_vz", "ref_x", "ref_y", "ref_z", "ref_vx", "ref_vy", "ref_vz")}
    flying = ~np.isnan(L["traj_t"]) & (L["traj_t"] >= 0)
    if not flying.any():
        raise SystemExit("no plan in this log (traj_t never >= 0)")
    t0 = L["t"][flying][0]
    L["time"] = L["t"] - t0
    done = np.array([r["phase"] == "done" for r in rows])        # the monitor marks the plan's end
    L["plan_end"] = L["time"][done][0] if done.any() else L["time"][-1]
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--gates", default=None)
    ap.add_argument("--fps", type=float, default=10.0, help="frames per second of sim time (playback is real time)")
    ap.add_argument("--before", type=float, default=1.0, help="seconds shown before the plan starts")
    ap.add_argument("--after", type=float, default=2.0, help="seconds shown after the plan ends")
    ap.add_argument("--out", default=None, help="default <run_dir>/flight_realtime.gif")
    ap.add_argument("--dpi", type=int, default=80)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    L = load(args.run_dir)
    gates = gates_from_manifest(args.gates or os.path.join(args.run_dir, "test_gates.json"))
    tt = L["time"]
    plan_end = L["plan_end"]
    win = (tt >= -args.before) & (tt <= plan_end + args.after)
    t = tt[win]
    P = np.column_stack([L["true_x"], L["true_y"], L["true_z"]])[win]
    V = np.column_stack([L["true_vx"], L["true_vy"], L["true_vz"]])[win]
    R = np.column_stack([L["ref_x"], L["ref_y"], L["ref_z"]])[win]
    RV = np.column_stack([L["ref_vx"], L["ref_vy"], L["ref_vz"]])[win]
    yaw = L["true_yaw"][win]
    speed, ref_speed = np.linalg.norm(V, axis=1), np.linalg.norm(RV, axis=1)
    err = np.linalg.norm(P - R, axis=1)
    on_plan = t >= 0
    err[~on_plan] = np.nan
    ref_speed[~on_plan] = np.nan
    passes = []
    for gate, cs in gate_crossings(t, P, V, gates):
        passes += [(c["time"], gate.name) for c in cs if c["inside"] and c["forward"]]

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 10})
    fig = plt.figure(figsize=(11, 5.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], hspace=0.42, wspace=0.18,
                          left=0.06, right=0.98, top=0.87, bottom=0.1)
    ax_map, ax_v, ax_e = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    for a in (ax_map, ax_v, ax_e):
        a.set_facecolor(SURFACE)
        a.grid(color=GRID, linewidth=0.8)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)

    # top-down, oriented like the onboard view: X forward up, Y to the left
    for g in gates:
        c = g.corners()
        ax_map.plot(c[[0, 1], 1], c[[0, 1], 0], color=GATE, linewidth=5, solid_capstyle="round", zorder=2)
        ax_map.annotate(g.name, (g.center[1], g.center[0]), xytext=(8, -12), textcoords="offset points",
                        fontsize=8, color=INK2)
    ax_map.plot(R[on_plan, 1], R[on_plan, 0], "--", color=INK2, linewidth=1.2, label="plan", zorder=3)
    path, = ax_map.plot([], [], color=DRONE, linewidth=2, label="drone (true)", zorder=4)
    dot, = ax_map.plot([], [], "o", color=DRONE, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
    nose, = ax_map.plot([], [], color=DRONE, linewidth=2, zorder=5)
    ax_map.plot([], [], color=GATE, linewidth=5, label="gate")
    ax_map.invert_xaxis()
    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.set_xlabel("Y (m)  ← left | right →")
    ax_map.set_ylabel("X forward (m)")
    ax_map.set_title("Top-down", loc="left")
    ax_map.legend(loc="lower right", fontsize=8, frameon=False)
    clock = ax_map.text(0.02, 0.97, "", transform=ax_map.transAxes, va="top", fontsize=11, color=INK)

    ax_v.plot(t, ref_speed, "--", color=INK2, linewidth=1.2, label="planned")
    ax_v.plot(t, speed, color=DRONE, linewidth=2, alpha=0.18)
    v_line, = ax_v.plot([], [], color=DRONE, linewidth=2, label="drone (true)")
    ax_v.set_ylabel("speed (m/s)")
    ax_v.set_title("Speed", loc="left")
    ax_v.set_ylim(0, max(np.nanmax(speed), np.nanmax(ref_speed)) * 1.2)
    ax_v.legend(loc="upper right", fontsize=8, frameon=False)

    ax_e.plot(t, err, color=DRONE, linewidth=2, alpha=0.18)
    e_line, = ax_e.plot([], [], color=DRONE, linewidth=2)
    ax_e.set_ylabel("error (m)")
    ax_e.set_xlabel("time since the plan started (s, sim time)")
    ax_e.set_title("Tracking error: |true position − plan|", loc="left")
    ax_e.set_ylim(0, np.nanmax(err) * 1.25)
    for a in (ax_v, ax_e):
        a.set_xlim(t[0], t[-1])
        for tp, name in passes:
            a.axvline(tp, color=GRID, linewidth=1.2, zorder=0)
            a.text(tp, 0.97, name, ha="right", va="top", fontsize=7, color=INK2, rotation=90,
                   transform=a.get_xaxis_transform())
    cursors = [a.axvline(t[0], color=INK, linewidth=1) for a in (ax_v, ax_e)]
    v_val = ax_v.text(0.02, 0.9, "", transform=ax_v.transAxes, fontsize=9, color=INK)
    e_val = ax_e.text(0.02, 0.92, "", transform=ax_e.transAxes, fontsize=9, color=INK)

    run = os.path.basename(os.path.normpath(args.run_dir))
    fig.suptitle(f"Isaac Sim: MPC flies the 3-gate course through PX4 (ground-truth state) — {run}",
                 fontsize=11, fontweight="bold", x=0.06, ha="left")
    during = on_plan & (t <= plan_end)
    fig.text(0.06, 0.915, f"Real time (sim). {len(passes)}/{len(gates)} gates passed; over the {plan_end:.1f} s plan: "
             f"tracking error mean {np.nanmean(err[during]):.2f} m, max {np.nanmax(err[during]):.2f} m; "
             f"peak speed {np.nanmax(speed[during]):.2f} m/s",
             fontsize=9, color=INK2)

    frames = np.arange(t[0], t[-1], 1.0 / args.fps)

    def draw(ft):
        i = int(np.searchsorted(t, ft, side="right")) - 1
        i = max(i, 0)
        path.set_data(P[:i + 1, 1], P[:i + 1, 0])
        dot.set_data([P[i, 1]], [P[i, 0]])
        nose.set_data([P[i, 1], P[i, 1] + 0.8 * np.sin(yaw[i])], [P[i, 0], P[i, 0] + 0.8 * np.cos(yaw[i])])
        v_line.set_data(t[:i + 1], speed[:i + 1])
        e_line.set_data(t[:i + 1], err[:i + 1])
        for c in cursors:
            c.set_xdata([t[i], t[i]])
        clock.set_text(f"t = {t[i]:+5.1f} s")
        v_val.set_text(f"{speed[i]:.2f} m/s")
        e_val.set_text("" if np.isnan(err[i]) else f"{err[i]:.2f} m")
        return path, dot, nose, v_line, e_line, clock, v_val, e_val, *cursors

    out = args.out or os.path.join(args.run_dir, "flight_realtime.gif")
    FuncAnimation(fig, draw, frames=frames, blit=False).save(out, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"{out}: {len(frames)} frames at {args.fps:g} fps = {len(frames) / args.fps:.1f} s "
          f"({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
