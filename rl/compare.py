"""
Three-way controller comparison on QuadSim: PID vs NMPC (control/) vs RL.

    python3 compare.py --model runs/best [--tracks straight3 oval6 slalom5]
                       [--mpc-vmax 4 8] [--mpc-amax 6] [--out runs/compare]

For every track and controller variant:
  * a deterministic flight from the start pose (n_envs = 1): completion, lap
    time, gates passed, crash kind, max / mean speed, min edge margin at the
    gate crossings, and the wall time of each controller.act call (mean / max;
    for the MPC also the IPOPT failure count);
  * robustness: completion rate over noisy starts (start_noise = 0.3 m,
    fixed seed): 16 envs for PID and RL, --mpc-envs (default 4) for the MPC,
    which solves envs sequentially and is slow.

Edge margin = half-side minus the larger of |x|, |y| of the plane crossing in
the gate frame, i.e. the distance from the crossing point to the opening
edge. A pass needs >= collision_radius (0.30 m); smaller margins clip the frame.

Caveats: all three controllers receive the TRUE simulator state (no EKF, no
vision noise), so this isolates the control law. The PID caps speed at 4 m/s
(v_max); the RL policy has no cap; the MPC follows a min-snap plan limited to
--mpc-vmax / --mpc-amax, so the MPC is shown at several speed limits. The
controller timings are CPU wall time on this machine (the RL/PID are batched
torch, the MPC is one IPOPT solve per env per step).

Writes results.csv, results.md and comparison.png into --out.
"""

import argparse
import csv
import os
import time

import numpy as np
import torch

from course import Course, TRACKS_DIR
from drone_params import QuadParams
from fly_course import run_controller, make_controller, draw_gates

ROBUST_NOISE = 0.3
ROBUST_SEED = 0


class Timed:
    """Wraps a controller and records the wall time of each act() call."""

    def __init__(self, ctl):
        self.ctl, self.times = ctl, []
        self.command = getattr(ctl, "command", "ctbr")

    def reset(self, *a, **k):
        self.times = []
        return self.ctl.reset(*a, **k)

    def act(self, *a, **k):
        t0 = time.perf_counter()
        out = self.ctl.act(*a, **k)
        self.times.append(time.perf_counter() - t0)
        return out


def edge_margins(log, course):
    """Per gate (in order), the edge margin [m] at the first forward crossing of
    its plane inside the outer frame, for env 0; NaN where never crossed."""
    pos = log["pos"][:, 0].astype(np.float64)
    out = []
    for g in course.gates:
        T = g.T_world_gate()
        local = (pos - T[:3, 3]) @ T[:3, :3]               # R^T (p - c)
        z = local[:, 2]
        m = np.nan
        for k in np.nonzero((z[:-1] < 0) & (z[1:] >= 0))[0]:
            a = z[k] / (z[k] - z[k + 1])
            xy = local[k, :2] + a * (local[k + 1, :2] - local[k, :2])
            if np.abs(xy).max() <= g.side / 2 + g.bar:
                m = g.side / 2 - np.abs(xy).max()
                break
        out.append(m)
    return np.array(out)


def variants(args, params, course):
    """[(label, controller factory)] for one course."""
    v = [("PID", lambda: make_controller("pid", params, course))]
    for vm in args.mpc_vmax:
        v.append((f"MPC v{vm:g}", lambda vm=vm: make_controller(
            "mpc", params, course, mpc_vmax=vm, mpc_amax=args.mpc_amax)))
    v.append(("RL", lambda: make_controller("rl", params, course, args.model)))
    return v


def evaluate(label, track, course, factory, params, args):
    ctl = Timed(factory())
    log, s = run_controller(course, ctl, params, max_time=args.max_time, n_envs=1)
    times = np.array(ctl.times)
    fails = ctl.ctl.n_failures if hasattr(ctl.ctl, "n_failures") else None
    margins = edge_margins(log, course)
    row = dict(track=track, controller=label, completed=s["completed"],
               lap_time=s["lap_time"], gates=f"{s['gates_passed']}/{s['n_gates']}",
               crash=s["crash_kind"] if s["crashed"] else "",
               max_speed=s["max_speed"], mean_speed=s["mean_speed"],
               min_edge_margin=float(np.nanmin(margins)) if np.isfinite(margins).any() else float("nan"),
               ctrl_ms_mean=1e3 * times.mean(), ctrl_ms_max=1e3 * times.max(),
               solver_failures=fails if fails is not None else "")
    n_noisy = args.mpc_envs if label.startswith("MPC") else args.envs
    nctl = Timed(factory())
    nlog, _ = run_controller(course, nctl, params, max_time=args.max_time, n_envs=n_noisy,
                             seed=ROBUST_SEED, start_noise=ROBUST_NOISE)
    row["noisy_envs"] = n_noisy
    row["noisy_completion"] = float(nlog["completed"].mean())
    return row, log


COLS = ["track", "controller", "completed", "lap_time", "gates", "crash", "max_speed",
        "mean_speed", "min_edge_margin", "ctrl_ms_mean", "ctrl_ms_max", "solver_failures",
        "noisy_envs", "noisy_completion"]


def markdown(rows):
    head = ["Track", "Controller", "Done", "Lap [s]", "Gates", "Crash", "Vmax [m/s]",
            "Vmean [m/s]", "Min margin [m]", "Ctrl ms mean/max", "Solver fails",
            "Noisy-start completion"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        lap = f"{r['lap_time']:.2f}" if np.isfinite(r["lap_time"]) else "-"
        marg = f"{r['min_edge_margin']:.2f}" if np.isfinite(r["min_edge_margin"]) else "-"
        lines.append("| " + " | ".join([
            r["track"], r["controller"], "yes" if r["completed"] else "no", lap, r["gates"],
            r["crash"] or "-", f"{r['max_speed']:.2f}", f"{r['mean_speed']:.2f}", marg,
            f"{r['ctrl_ms_mean']:.2f} / {r['ctrl_ms_max']:.1f}",
            str(r["solver_failures"]) if r["solver_failures"] != "" else "-",
            f"{100 * r['noisy_completion']:.0f}% ({r['noisy_envs']} envs)"]) + " |")
    return "\n".join(lines)


def make_plot(logs, courses, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tracks = list(courses)
    fig, ax = plt.subplots(len(tracks), 2, figsize=(14, 4.6 * len(tracks)), squeeze=False)
    for i, tr in enumerate(tracks):
        draw_gates(ax[i, 0], courses[tr])
        for label, log in logs[tr].items():
            pos, vel = log["pos"][:, 0], log["vel"][:, 0]
            ax[i, 0].plot(pos[:, 0], pos[:, 1], lw=1.2, label=label)
            ax[i, 1].plot(log["t"], np.linalg.norm(vel, axis=-1), lw=1.2, label=label)
        ax[i, 0].set_aspect("equal"); ax[i, 0].set_title(f"{tr}: top-down (ENU)")
        ax[i, 0].set_xlabel("x east [m]"); ax[i, 0].set_ylabel("y north [m]")
        ax[i, 0].legend(fontsize=8)
        ax[i, 1].set_title(f"{tr}: speed"); ax[i, 1].set_xlabel("t [s]")
        ax[i, 1].set_ylabel("m/s"); ax[i, 1].grid(alpha=0.3); ax[i, 1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="RL run directory (model.zip + vecnormalize.pkl)")
    ap.add_argument("--tracks", nargs="+", default=["straight3", "oval6", "slalom5"])
    ap.add_argument("--mpc-vmax", type=float, nargs="+", default=[4.0])
    ap.add_argument("--mpc-amax", type=float, default=6.0)
    ap.add_argument("--envs", type=int, default=16, help="noisy-start envs for PID and RL")
    ap.add_argument("--mpc-envs", type=int, default=4, help="noisy-start envs for the MPC (slow)")
    ap.add_argument("--max-time", type=float, default=60.0)
    ap.add_argument("--out", default="runs/compare")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    params = QuadParams()
    rows, logs, courses = [], {}, {}
    for tr in args.tracks:
        path = tr if tr.endswith(".json") else os.path.join(TRACKS_DIR, f"{tr}.json")
        course = Course.load(path)
        name = course.name
        courses[name], logs[name] = course, {}
        for label, factory in variants(args, params, course):
            t0 = time.time()
            row, log = evaluate(label, name, course, factory, params, args)
            rows.append(row); logs[name][label] = log
            print(f"[{name} / {label}] done in {time.time() - t0:.0f} s", flush=True)

    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader(); w.writerows(rows)
    table = markdown(rows)
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write(table + "\n")
    make_plot(logs, courses, os.path.join(args.out, "comparison.png"))
    print("\n" + table)
    print(f"\nwrote results.csv, results.md, comparison.png to {args.out}")


if __name__ == "__main__":
    main()
