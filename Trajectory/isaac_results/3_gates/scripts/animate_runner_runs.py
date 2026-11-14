#!/usr/bin/env python3
"""
Animated GIF of one or more rl/isaac/runner.py runs played together in real (sim) time, in the style of
animate_flight.py: top-down path through the gates with the nose direction, speed, and (for --state vio
runs) the VIO's distance from PX4's estimate, with cursors moving along the time plots.

    python animate_runner_runs.py --track ../../../../rl/tracks/three_gate_isaac.json \\
        --runs <run_dir>:3/4 <run_dir>:5/7 <run_dir>:8/10 --out flight_realtime.gif

<run_dir> holds runner.py's log.csv and summary.json. Time 0 is each run's race start (phase 1). The
drone path is PX4's estimate (sim GPS, ~truth); the VIO error column exists only for --state vio runs.
"""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "rl"))

from course import Course       # noqa: E402

# reference palette (dataviz skill): surface, ink, categorical slots in fixed order; gates as animate_flight.py
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RUN_COLORS = ["#2a78d6", "#1baf7a", "#6250d6"]   # slots 1, 3, 7 (slot 2 orange is the gates)
GATE = "#eb6834"


def load(run_dir):
    with open(os.path.join(run_dir, "log.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    col = lambda k: np.array([float(r[k]) if r.get(k) not in ("", None) else np.nan for r in rows])  # noqa: E731
    L = {k: col(k) for k in ("phase", "t_sim", "px", "py", "pz", "vx", "vy", "vz", "qw", "qx", "qy", "qz")}
    L["vio_err"] = col("vio_err") if "vio_err" in rows[0] else np.full(len(rows), np.nan)
    race = L["phase"] == 1
    t0 = L["t_sim"][race][0]
    L["time"] = L["t_sim"] - t0
    L["yaw"] = np.arctan2(2 * (L["qw"] * L["qz"] + L["qx"] * L["qy"]), 1 - 2 * (L["qy"] ** 2 + L["qz"] ** 2))
    with open(os.path.join(run_dir, "summary.json")) as f:
        L["summary"] = json.load(f)
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--track", required=True, help="rl/tracks/<track>.json")
    ap.add_argument("--runs", nargs="+", required=True, help="run_dir:label, up to 3")
    ap.add_argument("--title", default=None)
    ap.add_argument("--fps", type=float, default=10.0, help="frames per second of sim time (playback is real time)")
    ap.add_argument("--before", type=float, default=1.0, help="seconds shown before the race starts")
    ap.add_argument("--after", type=float, default=2.0, help="seconds shown after the slowest lap ends")
    ap.add_argument("--out", default="flight_realtime.gif")
    ap.add_argument("--dpi", type=int, default=80)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    course = Course.load(args.track)
    runs = []
    for k, spec in enumerate(args.runs[:3]):
        d, _, label = spec.rpartition(":")
        L = load(d)
        L["label"], L["color"] = label, RUN_COLORS[k]
        runs.append(L)
    t_end = max((r["summary"]["lap_time"] if r["summary"]["completed"] else r["time"][r["phase"] == 1][-1])
                for r in runs) + args.after
    for r in runs:
        w = (r["time"] >= -args.before) & (r["time"] <= t_end)
        for k in ("time", "px", "py", "pz", "vx", "vy", "vz", "yaw", "vio_err", "phase"):
            r[k] = r[k][w]
        r["speed"] = np.sqrt(r["vx"] ** 2 + r["vy"] ** 2 + r["vz"] ** 2)
        r["vio_err"] = np.where(r["time"] >= 0, r["vio_err"], np.nan)

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 10})
    fig = plt.figure(figsize=(11, 5.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], hspace=0.42, wspace=0.18,
                          left=0.06, right=0.98, top=0.85, bottom=0.1)
    ax_map, ax_v, ax_e = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    for a in (ax_map, ax_v, ax_e):
        a.set_facecolor(SURFACE)
        a.grid(color=GRID, linewidth=0.8)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)

    # top-down, oriented like the onboard view: X forward up, Y to the left (as animate_flight.py)
    for i, g in enumerate(course.gates):
        T = g.T_world_gate()
        a = T[:3, 0] * g.side / 2                       # gate +X: across the opening, horizontal
        c = T[:3, 3]
        ax_map.plot([c[1] - a[1], c[1] + a[1]], [c[0] - a[0], c[0] + a[0]], color=GATE, linewidth=5,
                    solid_capstyle="round", zorder=2)
        ax_map.annotate(f"gate_{i}", (c[1], c[0]), xytext=(8, -12), textcoords="offset points", fontsize=8,
                        color=INK2)
    ax_map.plot([], [], color=GATE, linewidth=5, label="gate")
    arts = []
    for r in runs:
        full, = ax_map.plot(r["py"], r["px"], color=r["color"], linewidth=1.5, alpha=0.15, zorder=3)
        path, = ax_map.plot([], [], color=r["color"], linewidth=2, zorder=4,
                            label=f"{r['label']}: {r['summary']['lap_time']:.2f} s" if r["summary"]["completed"]
                            else f"{r['label']}: {r['summary']['gates_passed']}/{r['summary']['n_gates']}")
        dot, = ax_map.plot([], [], "o", color=r["color"], markersize=8, markeredgecolor=SURFACE,
                           markeredgewidth=2, zorder=5)
        nose, = ax_map.plot([], [], color=r["color"], linewidth=2, zorder=5)
        v_line, = ax_v.plot([], [], color=r["color"], linewidth=2, label=r["label"])
        ax_v.plot(r["time"], r["speed"], color=r["color"], linewidth=2, alpha=0.15)
        e_line, = ax_e.plot([], [], color=r["color"], linewidth=2)
        ax_e.plot(r["time"], r["vio_err"], color=r["color"], linewidth=2, alpha=0.15)
        arts.append((r, path, dot, nose, v_line, e_line))
    ax_map.invert_xaxis()
    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.set_xlabel("Y (m)  ← left | right →")
    ax_map.set_ylabel("X forward (m)")
    ax_map.set_title("Top-down (short line: where the nose and camera point)", loc="left")
    ax_map.legend(loc="lower right", fontsize=8, frameon=False)
    clock = ax_map.text(0.02, 0.97, "", transform=ax_map.transAxes, va="top", fontsize=11, color=INK)

    ax_v.set_ylabel("speed (m/s)")
    ax_v.set_title("Speed", loc="left")
    ax_v.set_ylim(0, max(np.nanmax(r["speed"]) for r in runs) * 1.2)
    ax_v.legend(loc="upper right", fontsize=8, frameon=False, ncol=len(runs))
    ax_e.set_ylabel("error (m)")
    ax_e.set_xlabel("time since the race started (s, sim time)")
    ax_e.set_title("VIO error: |VIO position − PX4's estimate|  (abort above 1.0 m)", loc="left")
    emax = np.nanmax([np.nanmax(r["vio_err"]) for r in runs])
    ax_e.set_ylim(0, max(1.1, emax * 1.2) if np.isfinite(emax) else 1.1)
    ax_e.axhline(1.0, color=INK2, linewidth=1, linestyle=":")
    t_lo, t_hi = -args.before, t_end
    for a in (ax_v, ax_e):
        a.set_xlim(t_lo, t_hi)
    cursors = [a.axvline(t_lo, color=INK, linewidth=1) for a in (ax_v, ax_e)]

    fig.suptitle(args.title or "Isaac Sim: rate-aware MPC on the VIO (no GPS), nose on the next gate",
                 fontsize=11, fontweight="bold", x=0.06, ha="left")
    fig.text(0.06, 0.905, "Real time (sim). Labels: plan limits v_max / a_max (m/s, m/s²). The controller flew on "
             "the VIO; the path drawn is PX4's GPS estimate (≈ truth).", fontsize=9, color=INK2)

    frames = np.arange(t_lo, t_hi, 1.0 / args.fps)

    def draw(ft):
        out = []
        for r, path, dot, nose, v_line, e_line in arts:
            t = r["time"]
            i = max(int(np.searchsorted(t, ft, side="right")) - 1, 0)
            path.set_data(r["py"][:i + 1], r["px"][:i + 1])
            dot.set_data([r["py"][i]], [r["px"][i]])
            y = r["yaw"][i]
            nose.set_data([r["py"][i], r["py"][i] + 0.9 * math.sin(y)], [r["px"][i], r["px"][i] + 0.9 * math.cos(y)])
            v_line.set_data(t[:i + 1], r["speed"][:i + 1])
            e_line.set_data(t[:i + 1], r["vio_err"][:i + 1])
            out += [path, dot, nose, v_line, e_line]
        for c in cursors:
            c.set_xdata([ft, ft])
        clock.set_text(f"t = {ft:+5.1f} s")
        return out + [clock, *cursors]

    FuncAnimation(fig, draw, frames=frames, blit=False).save(args.out, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"{args.out}: {len(frames)} frames at {args.fps:g} fps = {len(frames) / args.fps:.1f} s "
          f"({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
