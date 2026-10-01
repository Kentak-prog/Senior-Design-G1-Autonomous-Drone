"""
Report-quality graphs of gate-position error vs ground truth, from the
gate_errors.csv written by eval_gate_bag.py.

Step 3 of 3 of the gate-error workflow (fly -> evaluate bag -> plot). Needs
only numpy + matplotlib, no ROS, no cv2: run it on a Mac.

    python3 plot_gate_errors.py run1_out/gate_errors.csv --out-dir run1_out/figs

Figures (one file each):
    fig1_error_vs_range     3D error | range (depth) and lateral error vs true range
    fig2_error_vs_time      position error over time, drone speed shaded (hover vs moving)
    fig3_topdown            x-y plane: true gate centers, estimates, drone path
    fig4_detection_rate     detection rate vs true range (rows where the gate is in view)
plus summary_table.csv (error_records.summarize) printed as a table.

Each figure is a plot_*() function so tests can call it directly. Colors are
keyed to the sorted gate name, so a gate has the same color in every figure
(and in every CSV that contains the same gates).
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import error_records as er  # noqa: E402

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 10, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 100,
})

_PALETTE = plt.get_cmap("tab10").colors
_GREY = "0.55"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def gate_names(rows) -> list:
    return sorted({r["gate"] for r in rows})


def gate_colors(rows) -> dict:
    """Fixed color per gate, keyed by sorted name (cycles after 10 gates)."""
    return {g: _PALETTE[i % len(_PALETTE)] for i, g in enumerate(gate_names(rows))}


def _num(v):
    return float("nan") if v is None else float(v)


def _arr(rows, key) -> np.ndarray:
    return np.array([_num(r.get(key)) for r in rows], dtype=float)


def _detected(rows, gate=None) -> list:
    """Detected rows with a finite position error (optionally for one gate)."""
    out = []
    for r in rows:
        if gate is not None and r["gate"] != gate:
            continue
        if r.get("detected") and r.get("pos_err_m") is not None and np.isfinite(r["pos_err_m"]):
            out.append(r)
    return out


def _gate_handles(rows, colors, marker="o") -> list:
    """Legend entries for every gate, flagging never-detected ones."""
    handles = []
    for g in gate_names(rows):
        n = len(_detected(rows, g))
        label = g if n else f"{g} (never detected)"
        handles.append(Line2D([], [], marker=marker, linestyle="", color=colors[g],
                              markersize=7, label=label, alpha=1.0 if n else 0.5))
    return handles


def _frames(rows) -> list:
    """One (t_s, drone_x, drone_y, speed) tuple per frame, time-sorted."""
    seen = {}
    for r in rows:
        seen.setdefault(r["frame"], r)
    out = [(_num(r.get("t_s")), _num(r.get("drone_x")), _num(r.get("drone_y")),
            _num(r.get("drone_speed_mps"))) for r in seen.values()]
    return sorted(out, key=lambda x: x[0])


def _title(base, suffix) -> str:
    return f"{base} {suffix}".strip()


def _finish(fig, path, dpi, bottom=0.0):
    fig.tight_layout(rect=[0, bottom, 1, 1])
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _no_data(ax, msg="no detections"):
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
            color="0.4", fontsize=12)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def plot_error_vs_range(rows, path, title_suffix="", dpi=300):
    colors = gate_colors(rows)
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    any_det = False
    for g in gate_names(rows):
        d = _detected(rows, g)
        if not d:
            continue
        any_det = True
        rng = _arr(d, "range_true_m")
        axl.scatter(rng, _arr(d, "pos_err_m"), s=14, color=colors[g], alpha=0.6, linewidths=0)
        mr, me, sd = np.nanmean(rng), np.nanmean(_arr(d, "pos_err_m")), np.nanstd(_arr(d, "pos_err_m"))
        axl.errorbar([mr], [me], yerr=[sd], fmt="D", color=colors[g], markeredgecolor="black",
                     markersize=9, capsize=4, elinewidth=1.5, zorder=5)
        axr.scatter(rng, _arr(d, "range_err_m"), s=14, color=colors[g], alpha=0.6, linewidths=0)
        axr.scatter(rng, _arr(d, "lateral_err_m"), s=22, facecolors="none",
                    edgecolors=[colors[g]], alpha=0.8, linewidths=1.0)
    if not any_det:
        _no_data(axl)
        _no_data(axr)
    axr.axhline(0.0, color="0.3", linewidth=0.8)
    axl.set_xlabel("True range along camera axis (m)")
    axr.set_xlabel("True range along camera axis (m)")
    axl.set_ylabel("3D position error (m)")
    axr.set_ylabel("Range / lateral error (m)")
    axl.set_title("3D position error (diamond = per-gate mean $\\pm$ std)")
    axr.set_title("Range (depth) vs lateral error")
    axl.set_ylim(bottom=0)
    handles = _gate_handles(rows, colors) + [
        Line2D([], [], marker="o", linestyle="", color="0.3", markersize=7,
               label="range error (signed, filled)"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="none", color="0.3",
               markersize=7, label="lateral error (hollow)"),
        Line2D([], [], marker="D", linestyle="", color="0.3", markeredgecolor="black",
               markersize=8, label="mean $\\pm$ std (left)")]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), frameon=False)
    fig.suptitle(_title("Gate position error vs range", title_suffix), fontsize=13)
    return _finish(fig, path, dpi, bottom=0.14)


def plot_error_vs_time(rows, path, title_suffix="", dpi=300):
    colors = gate_colors(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    fr = _frames(rows)
    if fr:
        t = np.array([f[0] for f in fr])
        sp = np.array([f[3] for f in fr])
        ok = np.isfinite(sp)
        if ok.any():
            ax2.fill_between(t[ok], 0, sp[ok], color=_GREY, alpha=0.25, linewidth=0)
            ax2.plot(t[ok], sp[ok], color=_GREY, linewidth=0.8, marker="." if ok.sum() == 1 else None)
        ax2.set_ylim(0, max(1.0, float(np.nanmax(sp[ok])) * 1.25) if ok.any() else 1.0)
    ax2.set_ylabel("Drone speed (m/s)", color="0.4")
    ax2.tick_params(axis="y", colors="0.4")
    # Draw the error lines on top of the shaded speed.
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    any_det = False
    for g in gate_names(rows):
        d = sorted(_detected(rows, g), key=lambda r: r["t_s"])
        if not d:
            continue
        any_det = True
        ax.plot(_arr(d, "t_s"), _arr(d, "pos_err_m"), color=colors[g], marker="o",
                markersize=3.5, linewidth=1.2)
    if not any_det:
        _no_data(ax)
    ax.set_xlabel("Time since first frame (s)")
    ax.set_ylabel("3D position error (m)")
    ax.set_ylim(bottom=0)
    ax.set_title("Grey area = drone speed: ~0 is hover, rising is moving", fontsize=10,
                 color="0.35")
    fig.legend(handles=_gate_handles(rows, colors), loc="lower center",
               ncol=min(max(len(colors), 1), 6), frameon=False)
    fig.suptitle(_title("Gate position error vs time", title_suffix), fontsize=13)
    return _finish(fig, path, dpi, bottom=0.09)


def plot_topdown(rows, path, title_suffix="", dpi=300):
    colors = gate_colors(rows)
    fig, ax = plt.subplots(figsize=(8, 7))
    fr = _frames(rows)
    if fr:
        dx = np.array([f[1] for f in fr])
        dy = np.array([f[2] for f in fr])
        ax.plot(dx, dy, color=_GREY, linewidth=1.8, zorder=2)
        ax.plot(dx[0], dy[0], marker="^", color="black", markeredgecolor="white",
                markersize=11, linestyle="", zorder=7)
        ax.plot(dx[-1], dy[-1], marker="X", color="black", markeredgecolor="white",
                markersize=11, linestyle="", zorder=7)
    for g in gate_names(rows):
        grows = [r for r in rows if r["gate"] == g]
        d = _detected(rows, g)
        ex = _arr([r for r in d if r.get("est_x") is not None], "est_x")
        ey = _arr([r for r in d if r.get("est_x") is not None], "est_y")
        if ex.size:
            ax.scatter(ex, ey, s=16, color=colors[g], alpha=0.5, linewidths=0, zorder=6)
        tx, ty = _num(grows[0].get("true_x")), _num(grows[0].get("true_y"))
        ax.scatter([tx], [ty], s=120, color=colors[g], marker="s", edgecolors="black",
                   linewidths=1.2, zorder=5)
        ax.annotate(g, (tx, ty), textcoords="offset points", xytext=(9, 7), fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("World x (m)")
    ax.set_ylabel("World y (m)")
    handles = _gate_handles(rows, colors, marker="s") + [
        Line2D([], [], marker="o", linestyle="", color="0.4", alpha=0.5, markersize=5,
               label="estimates (dots)"),
        Line2D([], [], color=_GREY, linewidth=2, label="drone path"),
        Line2D([], [], marker="^", linestyle="", color="black", markersize=9, label="start"),
        Line2D([], [], marker="X", linestyle="", color="black", markersize=9, label="end")]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), frameon=False)
    fig.suptitle(_title("Top-down view: true vs estimated gate centers", title_suffix), fontsize=13)
    return _finish(fig, path, dpi, bottom=0.12)


def plot_detection_rate(rows, path, title_suffix="", dpi=300, bin_m=1.0):
    fig, ax = plt.subplots(figsize=(9, 5))
    iv = [r for r in rows if r.get("in_view") and r.get("range_true_m") is not None]
    if not iv:
        _no_data(ax, "no in-view samples")
    else:
        rng = np.array([r["range_true_m"] for r in iv], dtype=float)
        det = np.array([bool(r.get("detected")) for r in iv])
        lo = np.floor(rng.min() / bin_m) * bin_m
        hi = np.floor(rng.max() / bin_m) * bin_m + bin_m
        edges = np.arange(lo, hi + bin_m * 0.5, bin_m)
        for a, b in zip(edges[:-1], edges[1:]):
            sel = (rng >= a) & (rng < b)
            n = int(sel.sum())
            if n == 0:
                continue
            rate = float(det[sel].mean())
            ax.bar((a + b) / 2, rate, width=bin_m * 0.85, color="tab:blue", alpha=0.8)
            ax.text((a + b) / 2, rate + 0.02, f"n={n}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("True range along camera axis (m)")
    ax.set_ylabel("Detection rate (detected / in view)")
    ax.set_title("Only frames where the true gate center is inside the image are counted",
                 fontsize=10, color="0.35")
    fig.suptitle(_title("Detection rate vs range", title_suffix), fontsize=13)
    return _finish(fig, path, dpi)


FIGURES = [
    ("fig1_error_vs_range", plot_error_vs_range),
    ("fig2_error_vs_time", plot_error_vs_time),
    ("fig3_topdown", plot_topdown),
    ("fig4_detection_rate", plot_detection_rate),
]


def plot_all(rows, out_dir, title_suffix="", fmt="png", dpi=300) -> list:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, fn in FIGURES:
        paths.append(fn(rows, os.path.join(out_dir, f"{name}.{fmt}"), title_suffix, dpi))
    summary = er.summarize(rows)
    er.write_summary_csv(os.path.join(out_dir, "summary_table.csv"), summary)
    print(er.format_summary_table(summary))
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("csv")
    ap.add_argument("--out-dir", default=None, help="default: directory of the CSV")
    ap.add_argument("--title-suffix", default="")
    ap.add_argument("--format", choices=["png", "pdf"], default="png")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    rows = er.read_csv(args.csv)
    if not rows:
        sys.exit(f"{args.csv} has no rows")
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.csv))
    for p in plot_all(rows, out_dir, args.title_suffix, args.format, args.dpi):
        print(f"saved {p}")


if __name__ == "__main__":
    main()
