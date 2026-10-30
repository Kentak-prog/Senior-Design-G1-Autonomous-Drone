#!/usr/bin/env python3
"""
Report graphs of the VIO during an Isaac flight, in the style of
vision/gate_error_eval/plot_gate_errors.py, from ros2_nodes/flight_monitor.py's
flight_log.csv and vio_node's vio_events.csv. Needs only numpy + matplotlib.

    python plot_vio_flight.py ../vio_mpc/run_045716_s4

Figures (one file each, in the run folder):
    fig1_vio_error_vs_range  VIO error at each fix | fix chi^2, vs the true range to that gate
    fig2_vio_error_vs_time   VIO error and reported std | speed, with fixes and gate passes marked
    fig3_vio_topdown         x-y plane: gates, true path, VIO path, where fixes happened
    fig4_fix_rate_vs_range   fixes per in-view time, by true range and gate
    fig5_gate_visibility     per gate: when it was in frame (geometry) vs fixes and rejections
plus vio_summary.txt.

WHAT THIS CAN AND CAN'T SAY. "In frame" is calculated from the TRUE pose, the
gate map and a level 1280x720, 60 deg camera; nobody looked at the images, and
occlusion by other gates is not modelled. vio_events.csv has rows only for
fixes and rejections: a frame with no detection, a failed PnP and an unmatched
detection all leave no row. Gate names are the VIO's own matches, not ground
truth. So "in frame with no fix" means the VIO got nothing from that gate, not
that the detector failed.
"""

import argparse
import csv
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Trajectory"))

from gate_path import gates_from_manifest, gate_crossings      # noqa: E402

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 10, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 100,
})
_PALETTE = plt.get_cmap("tab10").colors
_GREY = "0.55"
HALF_H = math.tan(math.radians(30.0))                # 60 deg horizontal field of view
HALF_V = HALF_H * 720.0 / 1280.0                     # 1280x720
CAM_BODY = np.array([0.30, 0.0, 0.05])               # camera mount, body FLU (control/vio.py)
CHI2_FULL = 22.5                                     # control/vio.py chi2_full
SUFFIX = "Isaac Sim, Stage 4, VIO not in the loop"


def load_flight(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    col = lambda k: np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows])  # noqa: E731
    L = {k: col(k) for k in ("t", "traj_t", "vio_std")}
    L["true"] = np.column_stack([col("true_" + k) for k in ("x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz")])
    L["vio"] = np.column_stack([col("vio_" + k) for k in ("x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz")])
    flying = ~np.isnan(L["traj_t"]) & (L["traj_t"] >= 0)
    if not flying.any():
        raise SystemExit("no plan in this log (traj_t never >= 0)")
    L["t0"] = L["t"][flying][0]
    done = np.array([r["phase"] == "done" for r in rows])
    L["plan_end"] = (L["t"][done][0] if done.any() else L["t"][-1]) - L["t0"]
    L["time"] = L["t"] - L["t0"]
    return L


def load_events(path, t0, t_from, t_to):
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            t = float(r["t"]) - t0
            if t_from <= t <= t_to:
                out.append((t, r["kind"], r["gate"], float(r["chi2"])))
    return out


def _rot(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch),
                              math.cos(yaw), math.sin(yaw))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def visibility(true, gate):
    """Per sample: (range to gate center m, center in frame, whole gate in frame, viewing angle deg)."""
    corners = np.asarray(gate.corners(), dtype=float)
    n = len(true)
    rng, center, whole, angle = np.zeros(n), np.zeros(n, bool), np.zeros(n, bool), np.zeros(n)

    def in_frame(v):
        return v[0] > 0.05 and abs(v[1] / v[0]) <= HALF_H and abs(v[2] / v[0]) <= HALF_V

    for i in range(n):
        R = _rot(*true[i, 3:6])
        cam = true[i, 0:3] + R @ CAM_BODY
        v = R.T @ (gate.center - cam)
        rng[i] = float(np.linalg.norm(gate.center - cam))
        center[i] = in_frame(v)
        whole[i] = all(in_frame(R.T @ (c - cam)) for c in corners)
        los = (gate.center - cam) / max(rng[i], 1e-6)
        angle[i] = math.degrees(math.acos(min(1.0, abs(float(los @ gate.normal)))))
    return rng, center, whole, angle


def _finish(fig, path, dpi, bottom=0.0):
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"[plot] {path}")


def _gate_handles(names, colors, marker="o"):
    return [Line2D([], [], marker=marker, linestyle="", color=colors[g], markersize=8, label=g) for g in names]


def plot_error_vs_range(ev, S, names, colors, path, dpi):
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    t, err = S["t"], S["err"]
    for g in names:
        rng_t = S["vis"][g][0]
        fx = [e[0] for e in ev if e[2] == g and e[1] in ("full", "pos") and e[0] >= 0]
        rj = [e for e in ev if e[2] == g and e[1] == "reject" and e[0] >= 0]
        if fx:
            axl.scatter(np.interp(fx, t, rng_t), np.interp(fx, t, err), s=14, alpha=0.6, color=colors[g], linewidths=0)
            chi = [e[3] for e in ev if e[2] == g and e[1] in ("full", "pos") and e[0] >= 0]
            axr.scatter(np.interp(fx, t, rng_t), np.maximum(chi, 1e-2), s=14, alpha=0.6, color=colors[g], linewidths=0)
        if rj:
            tr = [e[0] for e in rj]
            axl.scatter(np.interp(tr, t, rng_t), np.interp(tr, t, err), s=26, marker="x", color=colors[g])
            axr.scatter(np.interp(tr, t, rng_t), [e[3] for e in rj], s=26, marker="x", color=colors[g])
    axl.set_xlabel("True range to the matched gate (m)")
    axl.set_ylabel("VIO position error at that moment (m)")
    axl.set_ylim(bottom=0)
    axl.set_title("VIO error at each fix or rejection, in flight")
    axr.set_yscale("log")
    axr.axhline(CHI2_FULL, color="0.3", linewidth=1, linestyle="--")
    axr.text(5.0, CHI2_FULL * 1.15, "full-fix rejection limit (22.5)", ha="left", fontsize=9, color="0.3")
    axr.set_xlabel("True range to the matched gate (m)")
    axr.set_ylabel("Fix chi² (log)")
    axr.set_title("How well each fix agreed with the estimate")
    fig.suptitle(f"VIO fixes vs range ({SUFFIX})")
    handles = _gate_handles(names, colors) + [
        Line2D([], [], marker="o", linestyle="", color=_GREY, label="fix"),
        Line2D([], [], marker="x", linestyle="", color=_GREY, label="rejected")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    _finish(fig, path, dpi, bottom=0.08)


def plot_error_vs_time(ev, S, names, colors, passes, path, dpi):
    fig, (ax, axs) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, height_ratios=[3, 1])
    t = S["t"]
    ax.plot(t, S["err"], color="0.15", linewidth=1.6, label="VIO position error (true)")
    ax.plot(t, S["std"], color="0.15", linewidth=1.2, linestyle=":", label="VIO's reported std")
    for g in names:
        fx = [e[0] for e in ev if e[2] == g and e[1] in ("full", "pos")]
        if fx:
            ax.scatter(fx, np.interp(fx, t, S["err"]), s=12, color=colors[g], zorder=3, linewidths=0)
        rj = [e[0] for e in ev if e[2] == g and e[1] == "reject"]
        if rj:
            ax.scatter(rj, np.interp(rj, t, S["err"]), s=26, marker="x", color=colors[g], zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.01, max(np.nanmax(S["err"]) * 1.5, 1.0))
    ax.set_ylabel("m (log)")
    ax.set_title("Dots: fixes. Crosses: rejected fixes. No marker: nothing logged.", fontsize=10, color="0.35")
    axs.fill_between(t, S["speed"], color=_GREY, alpha=0.25)
    axs.plot(t, S["speed"], color=_GREY, linewidth=0.8)
    axs.set_ylabel("Speed (m/s)")
    axs.set_ylim(bottom=0)
    axs.set_xlabel("Time since the plan started (s, sim time)")
    for a in (ax, axs):
        for tp, name in passes:
            a.axvline(tp, color="0.6", linewidth=1, linestyle="--")
    for tp, name in passes:
        ax.text(tp, ax.get_ylim()[1], f"{name} passed", rotation=90, ha="right", va="top", fontsize=8, color="0.35")
    fig.suptitle(f"VIO position error vs time ({SUFFIX})")
    handles = _gate_handles(names, colors) + [
        Line2D([], [], color="0.15", linewidth=1.6, label="true error"),
        Line2D([], [], color="0.15", linewidth=1.2, linestyle=":", label="reported std")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    _finish(fig, path, dpi, bottom=0.06)


def plot_topdown(ev, S, gates, names, colors, path, dpi):
    fig, ax = plt.subplots(figsize=(8, 7))
    true, vio, t = S["true"], S["vio"], S["t"]
    keep = t <= S["plan_end"] + 2.0
    for g in gates:
        c = np.asarray(g.corners())
        ax.plot(c[[0, 1], 0], c[[0, 1], 1], color=colors[g.name], linewidth=5, solid_capstyle="round", zorder=2)
        ax.annotate(g.name, (g.center[0], g.center[1]), xytext=(8, 8), textcoords="offset points")
    ax.plot(true[keep, 0], true[keep, 1], color="0.15", linewidth=1.8, label="true path", zorder=3)
    ax.plot(vio[keep, 0], vio[keep, 1], color="0.15", linewidth=1.2, linestyle=":", label="VIO estimate", zorder=3)
    for g in names:
        fx = [e[0] for e in ev if e[2] == g and e[1] in ("full", "pos") and e[0] >= 0]
        if fx:
            ax.scatter(np.interp(fx, t, true[:, 0]), np.interp(fx, t, true[:, 1]), s=20, color=colors[g], zorder=4, linewidths=0)
        rj = [e[0] for e in ev if e[2] == g and e[1] == "reject" and e[0] >= 0]
        if rj:
            ax.scatter(np.interp(rj, t, true[:, 0]), np.interp(rj, t, true[:, 1]), s=30, marker="x", color=colors[g], zorder=4)
    ax.plot(true[0, 0], true[0, 1], marker="^", color="k", markersize=10, linestyle="", zorder=5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("World x (m)")
    ax.set_ylabel("World y (m)")
    fig.suptitle(f"Top-down: where the VIO got fixes ({SUFFIX})", fontsize=12)
    handles = _gate_handles(names, colors) + [
        Line2D([], [], color="0.15", linewidth=1.8, label="true path"),
        Line2D([], [], color="0.15", linewidth=1.2, linestyle=":", label="VIO estimate"),
        Line2D([], [], marker="o", linestyle="", color=_GREY, label="fix, at true position"),
        Line2D([], [], marker="x", linestyle="", color=_GREY, label="rejected"),
        Line2D([], [], marker="^", linestyle="", color="k", label="start")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    _finish(fig, path, dpi, bottom=0.12)


def plot_fix_rate(rate, names, colors, path, dpi, bin_m=1.0):
    """rate[g] = list of (range bin, n in-view 0.1 s bins, n of them with a fix)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    w = bin_m * 0.8 / max(len(names), 1)
    for k, g in enumerate(names):
        for r, n, m in rate[g]:
            if n == 0:
                continue
            x = r + bin_m / 2 + (k - (len(names) - 1) / 2) * w
            ax.bar(x, m / n, width=w * 0.9, color=colors[g], alpha=0.8)
            if m == 0:                                        # keep a zero rate visible, in its gate's color
                ax.plot([x - w * 0.45, x + w * 0.45], [0.006, 0.006], color=colors[g], linewidth=3, solid_capstyle="butt")
            ax.text(x, m / n + 0.02, f"{n}", ha="center", fontsize=7, color="0.3")
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("True range to the gate (m)")
    ax.set_ylabel("Fix rate (0.1 s bins with a fix / in frame)")
    ax.set_title("Only moments when the whole gate is in frame, by geometry. Number = 0.1 s bins. A rejected fix counts as none",
                 fontsize=10, color="0.35")
    fig.suptitle(f"VIO fix rate vs range, in flight ({SUFFIX})")
    fig.legend(handles=_gate_handles(names, colors, "s"), loc="lower center", ncol=len(names), frameon=False)
    _finish(fig, path, dpi, bottom=0.07)


def plot_visibility(ev, S, gates, colors, passes, path, dpi):
    fig, axes = plt.subplots(len(gates), 1, figsize=(11, 1.9 * len(gates) + 1.3), sharex=True)
    axes = np.atleast_1d(axes)
    t = S["t"]
    for ax, g in zip(axes, gates):
        rng, center, whole, angle = S["vis"][g.name]
        ax.fill_between(t, 0, 1, where=center, color=colors[g.name], alpha=0.18, linewidth=0, step="mid")
        ax.fill_between(t, 0, 1, where=whole, color=colors[g.name], alpha=0.40, linewidth=0, step="mid")
        fx = [e[0] for e in ev if e[2] == g.name and e[1] in ("full", "pos")]
        rj = [e[0] for e in ev if e[2] == g.name and e[1] == "reject"]
        if fx:
            ax.vlines(fx, 0.55, 0.95, color="0.1", linewidth=0.5)
        if rj:
            ax.scatter(rj, np.full(len(rj), 0.3), marker="x", s=22, color="0.1", linewidths=0.9)
        for tp, name in passes:
            if name == g.name:
                ax.axvline(tp, color="0.1", linewidth=1.2, linestyle="--")
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(g.name, rotation=0, ha="right", va="center")
        ax.grid(False)
    axes[-1].set_xlabel("Time since the plan started (s, sim time)")
    axes[-1].set_xlim(t[0], S["plan_end"] + 2.0)
    fig.suptitle(f"Gate in frame (geometry) vs what the VIO logged ({SUFFIX})")
    handles = [Line2D([], [], color=_GREY, linewidth=8, alpha=0.4, label="center in frame"),
               Line2D([], [], color=_GREY, linewidth=8, alpha=0.9, label="whole gate in frame"),
               Line2D([], [], color="0.1", linewidth=1, label="fix (tick)"),
               Line2D([], [], marker="x", linestyle="", color="0.1", label="rejected"),
               Line2D([], [], color="0.1", linewidth=1.2, linestyle="--", label="gate passed")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    _finish(fig, path, dpi, bottom=0.08)


def intervals(t, mask):
    """[(start, end)] of the True runs of mask."""
    out, start = [], None
    for ti, m in zip(t, mask):
        if m and start is None:
            start = ti
        if not m and start is not None:
            out.append((start, ti))
            start = None
    if start is not None:
        out.append((start, t[-1]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--gates", default=None)
    ap.add_argument("--before", type=float, default=2.0)
    ap.add_argument("--after", type=float, default=2.0)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    L = load_flight(os.path.join(args.run_dir, "flight_log.csv"))
    gates = gates_from_manifest(args.gates or os.path.join(args.run_dir, "test_gates.json"))
    win = (L["time"] >= -args.before) & (L["time"] <= L["plan_end"] + args.after)
    t = L["time"][win]
    S = {"t": t, "true": L["true"][win], "vio": L["vio"][win], "std": L["vio_std"][win], "plan_end": L["plan_end"]}
    S["err"] = np.linalg.norm(S["vio"][:, 0:3] - S["true"][:, 0:3], axis=1)
    S["speed"] = np.linalg.norm(S["true"][:, 6:9], axis=1)
    S["vis"] = {g.name: visibility(S["true"], g) for g in gates}
    ev = load_events(os.path.join(args.run_dir, "vio_events.csv"), L["t0"], t[0], t[-1])

    names = [g.name for g in gates]
    colors = {g: _PALETTE[i % len(_PALETTE)] for i, g in enumerate(names)}
    passes = []
    for gate, cs in gate_crossings(t, S["true"][:, 0:3], S["true"][:, 6:9], gates):
        passes += [(c["time"], gate.name) for c in cs if c["inside"] and c["forward"]]

    # 0.1 s bins: was the whole gate in frame, and did a fix for it land in the bin?
    edges = np.arange(t[0], t[-1], 0.1)
    rate, lines = {}, [f"run {args.run_dir}; t = 0 at sim {L['t0']:.2f}; plan {L['plan_end']:.2f} s",
                       "in frame = by geometry from the TRUE pose; events are the VIO's own gate matches", ""]
    for g in gates:
        rng, center, whole, angle = S["vis"][g.name]
        fx = np.array([e[0] for e in ev if e[2] == g.name and e[1] in ("full", "pos")])
        tbin, has_fix = [], []
        for a in edges:
            i = int(np.searchsorted(t, a))
            if i < len(t) and t[i] >= 0 and whole[i]:
                tbin.append(rng[i])
                has_fix.append(bool(fx.size) and bool(np.any((fx >= a) & (fx < a + 0.1))))
        rows = []
        for r in np.arange(0.0, 16.0, 1.0):
            sel = [h for x, h in zip(tbin, has_fix) if r <= x < r + 1.0]
            rows.append((r, len(sel), sum(sel)))
        rate[g.name] = rows
        fly = t >= 0
        kinds = {k: sum(1 for e in ev if e[2] == g.name and e[1] == k and e[0] >= 0) for k in ("full", "pos", "reject")}
        hov = t < 0
        lines += [f"{g.name}: in flight {kinds}",
                  f"  hover (before t=0): center in frame {100 * center[hov].mean():.0f} %, "
                  f"whole gate {100 * whole[hov].mean():.0f} %; range {rng[hov].mean():.1f} m",
                  "  center in frame, in flight: " + ", ".join(f"{a:+.1f}..{b:+.1f}" for a, b in intervals(t[fly], center[fly])),
                  "  whole gate in frame, in flight: " + ", ".join(f"{a:+.1f}..{b:+.1f}" for a, b in intervals(t[fly], whole[fly])),
                  f"  closest range {rng[fly].min():.2f} m"]
        for a, b in intervals(t[fly], whole[fly]):
            m = (t >= a) & (t < b)
            n_fx = int(np.sum((fx >= a) & (fx < b))) if fx.size else 0
            lines.append(f"    whole in frame {a:+.1f}..{b:+.1f} s: range {rng[m].max():.1f}->{rng[m].min():.1f} m, "
                         f"viewing angle {angle[m].min():.0f}-{angle[m].max():.0f} deg, {n_fx} fixes")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.run_dir, "vio_summary.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")

    d = args.run_dir
    plot_error_vs_range(ev, S, names, colors, os.path.join(d, "fig1_vio_error_vs_range.png"), args.dpi)
    plot_error_vs_time(ev, S, names, colors, passes, os.path.join(d, "fig2_vio_error_vs_time.png"), args.dpi)
    plot_topdown(ev, S, gates, names, colors, os.path.join(d, "fig3_vio_topdown.png"), args.dpi)
    plot_fix_rate(rate, names, colors, os.path.join(d, "fig4_fix_rate_vs_range.png"), args.dpi)
    plot_visibility(ev, S, gates, colors, passes, os.path.join(d, "fig5_gate_visibility.png"), args.dpi)


if __name__ == "__main__":
    main()
