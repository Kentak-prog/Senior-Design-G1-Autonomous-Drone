"""
Plot a PX4 OFFBOARD flight of the min-snap course in Isaac Sim (a run folder
written by px4/fly_min_snap_offboard.py: plan_reference.csv, flight_log.csv,
summary.json, test_gates.json) in the same camera-view orientation as
plot_3_gates.py: forward up the page, the drone's left on the left.

Run from Senior-Design-G1-Autonomous-Drone/Trajectory:
    python "isaac_results/3_gates/scripts/plot_px4_flight.py" \
        "isaac_results/3_gates/px4_flights/20261023_152804_all_gates_passed"
(no argument = that run). Writes <run>/px4_flight.png.
"""

import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_3_gates as p3                                   # noqa: E402  (Isaac-view helpers, sys.path setup)
from gate_path import gates_from_manifest                    # noqa: E402

PHASE_STYLE = {"reposition": dict(color='0.55', linewidth=1.5, label='Reposition (flown)'),
               "settle": dict(color='0.55', linewidth=1.0),
               "trajectory": dict(color='royalblue', linewidth=2.0, label='Course (flown)')}


def load_run(run_dir):
    with open(os.path.join(run_dir, "summary.json")) as f:
        summary = json.load(f)
    with open(os.path.join(run_dir, "plan_reference.csv")) as f:
        plan = {k: np.array(v, dtype=float) for k, v in
                zip(next(csv.reader([f.readline()])), np.array([r for r in csv.reader(f)], dtype=float).T)}
    rows = list(csv.DictReader(open(os.path.join(run_dir, "flight_log.csv"))))
    log = {}
    for r in rows:
        if r["x"] == "":
            continue
        ph = log.setdefault(r["phase"], {k: [] for k in ("t", "x", "y", "z", "vx", "vy", "vz",
                                                          "sp_x", "sp_y", "sp_z", "main_mode")})
        for k in ph:
            ph[k].append(float(r[k]) if r[k] != "" else np.nan)
    log = {ph: {k: np.array(v) for k, v in d.items()} for ph, d in log.items()}
    gates = gates_from_manifest(os.path.join(run_dir, "test_gates.json"))
    return summary, plan, log, gates


def main(run_dir):
    summary, plan, log, gates = load_run(run_dir)
    ts = summary["time_scale"]
    start = np.array(summary["start_enu"])
    P = np.column_stack([plan["x"], plan["y"], plan["z"]])
    T = log["trajectory"]
    flown = np.column_stack([T["x"], T["y"], T["z"]])
    sp = np.column_stack([T["sp_x"], T["sp_y"], T["sp_z"]])
    t_traj = ts * T["t"]                                   # wall s -> trajectory s
    err = np.linalg.norm(flown - sp, axis=1)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"PX4 OFFBOARD flight of the 3-gate min-snap course in Isaac Sim  "
                 f"({'ALL GATES PASSED' if summary['gates_passed'] else 'NOT ALL GATES PASSED'}; "
                 f"tracking error mean {summary['err_mean_m']:.2f} m, max {summary['err_max_m']:.2f} m)",
                 fontweight='bold')

    ax = fig.add_subplot(2, 3, (1, 4), projection='3d')
    p3.draw_gates_3d(ax, gates)
    ax.plot(*P.T, color='k', linestyle='--', linewidth=1.5, label='Min-snap plan')
    for ph, style in PHASE_STYLE.items():
        if ph in log:
            ax.plot(log[ph]["x"], log[ph]["y"], log[ph]["z"], **style)
    ax.scatter(*start, color='green', s=40, label='Start (spawn hover pose)')
    pts = [P, flown] + [np.column_stack([log[ph][k] for k in "xyz"]) for ph in ("reposition",) if ph in log]
    p3.setup_3d(ax, np.vstack(pts + [g.corners() for g in gates]), "3D (view from behind the drone)")
    ax.legend(loc='upper left', fontsize=8)

    ax2 = fig.add_subplot(2, 3, 2)
    p3.draw_top_down(ax2, gates, start, "Top-down (camera orientation)")
    ax2.plot(P[:, 1], P[:, 0], 'k--', linewidth=1.5, label='Plan')
    for ph, style in PHASE_STYLE.items():
        if ph in log:
            ax2.plot(log[ph]["y"], log[ph]["x"], **style)
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(2, 3, 3)
    p3.draw_side(ax3, gates, "Side view (height)")
    ax3.plot(P[:, 0], P[:, 2], 'k--', linewidth=1.5, label='Plan')
    for ph, style in PHASE_STYLE.items():
        if ph in log:
            ax3.plot(log[ph]["x"], log[ph]["z"], **style)
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(2, 3, 5)
    ax4.plot(plan["t"], np.linalg.norm(np.column_stack([plan["vx"], plan["vy"], plan["vz"]]), axis=1),
             'k--', label='Plan speed')
    ax4.plot(t_traj, np.linalg.norm(np.column_stack([T["vx"], T["vy"], T["vz"]]), axis=1),
             color='royalblue', label='Flown speed (PX4 estimate)')
    ax4.set_xlabel(f"Trajectory time (s)  [= {ts:g} x wall time]"); ax4.set_ylabel("m/s (sim units)")
    ax4.grid(alpha=0.3); ax4.legend(fontsize=8); ax4.set_title("Speed")

    ax5 = fig.add_subplot(2, 3, 6)
    ax5.plot(t_traj, err, 'r-')
    ax5.set_ylim(0.0, max(err.max() * 1.9, 0.5))
    ax5.set_xlabel("Trajectory time (s)"); ax5.set_ylabel("m"); ax5.grid(alpha=0.3)
    ax5.set_title("Distance from the streamed setpoint")
    ratio = summary.get("sim_speed_ratio")
    lines = ["Gate passes (flown vs true gates; t = wall s):"] + summary["crossings"] + [
        "", f"reposition {summary.get('reposition_m') or 0:.2f} m -> start {np.round(start, 2)}",
        f"plan {summary['plan_duration_s']:.1f} s, v/a limits {summary['v_max']}/{summary['a_max']}, "
        f"time scale {ts:g}",
        f"measured sim speed {ratio:.2f}x real time" if ratio else ""]
    ax5.text(0.02, 0.97, "\n".join(lines), transform=ax5.transAxes, va='top', fontsize=7,
             family='monospace', bbox=dict(facecolor='white', alpha=0.85, edgecolor='0.7'))

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(run_dir, "px4_flight.png")
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.join(os.path.dirname(HERE), "px4_flights", "20261023_152804_all_gates_passed"))
