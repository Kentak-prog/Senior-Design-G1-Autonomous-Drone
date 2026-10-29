#!/usr/bin/env python3
"""
Scores an Isaac VIO + MPC flight from ros2_nodes/flight_monitor.py's log, the
same way Trajectory/simulate_vio_mpc.py scores the rehearsal, so the two can
be compared line by line. No ROS.

    python3 px4/analyze_vio_mpc_flight.py /workspace/vio_runs/flight_<time>
    python3 px4/analyze_vio_mpc_flight.py <log_dir> --gates /workspace/test_gates.json

Prints: VIO error while hovering and while flying, MPC tracking (TRUE drone vs
reference), solve times, every gate pass on the TRUE path with its frame
margin, the watchdog; and, if mpc_offboard_bridge --log-dir wrote
bridge_log.csv there, how far PX4's position moved while in OFFBOARD (the
hover-hold score) and the bridge's final status. Writes <log_dir>/flight_report.png and flight_summary.txt.
Exit code 0 only if every gate was passed, the watchdog never tripped and the
bridge did not abort.
"""

import argparse
import csv
import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "Trajectory"))

from gate_path import gates_from_manifest, gate_crossings      # noqa: E402

DEFAULT_GATES = os.path.join(REPO, "Trajectory", "isaac_results", "3_gates", "course", "pod_test_gates.json")


def load_log(path):
    """flight_log.csv -> dict of float arrays (empty cells -> nan) plus 'phase' and 'watchdog' string arrays."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no rows")
    out = {}
    for k in rows[0]:
        if k in ("phase", "watchdog"):
            out[k] = np.array([r[k] for r in rows])
        else:
            out[k] = np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows])
    return out


def bridge_report(path):
    """bridge_log.csv (mpc_offboard_bridge --log-dir) -> printed OFFBOARD summary; returns the numbers or None.
    The excursion from the first OFFBOARD position is the hover-hold score (stages 3 and 6)."""
    if not os.path.exists(path):
        return None
    rows, status = [], "?"
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if row and row[0].startswith("#"):
                status = row[0].lstrip("# ")
            elif row:
                rows.append([float(v) for v in row])
    if not rows:
        print(f"[bridge] no OFFBOARD samples; {status}")
        return {"status": status}
    a = np.array(rows)
    d = a[:, 1:4] - a[0, 1:4]
    out = {"status": status, "duration": float(a[-1, 0] - a[0, 0]),
           "max_horizontal": float(np.max(np.linalg.norm(d[:, 0:2], axis=1))), "max_vertical": float(np.max(np.abs(d[:, 2]))),
           "thrust_norm_mean": float(np.mean(a[:, 11]))}
    print(f"[bridge] OFFBOARD {out['duration']:.1f} s wall, {len(a)} setpoints; PX4 position vs where OFFBOARD began: "
          f"max horizontal {out['max_horizontal']:.2f} m, max vertical {out['max_vertical']:.2f} m; "
          f"mean throttle {out['thrust_norm_mean']:.3f}; {status}")
    return out


def _cols(L, prefix, keys):
    return np.column_stack([L[f"{prefix}_{k}"] for k in keys])


def analyze(L, gates):
    """Prints the report; returns (ok, numbers)."""
    true = _cols(L, "true", ("x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz"))
    vio = _cols(L, "vio", ("x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz"))
    ref = _cols(L, "ref", ("x", "y", "z"))
    t = L["t"]
    flying = np.isin(L["phase"], ("flying", "done")) & ~np.isnan(L["traj_t"]) & (L["traj_t"] >= 0)
    before = ~flying
    has_vio = ~np.isnan(vio[:, 0])
    pe = np.linalg.norm(vio[:, 0:3] - true[:, 0:3], axis=1)
    ye = np.degrees(np.abs((vio[:, 5] - true[:, 5] + np.pi) % (2 * np.pi) - np.pi))
    ve = np.linalg.norm(vio[:, 6:9] - true[:, 6:9], axis=1)
    te = np.linalg.norm(true[:, 0:3] - ref, axis=1)
    n = {}

    def stats(name, err, mask, unit="m"):
        m = mask & ~np.isnan(err)
        if not m.any():
            print(f"[{name}] no samples")
            return None
        n[name] = (float(np.mean(err[m])), float(np.max(err[m])))
        print(f"[{name}] mean {n[name][0]:.3f} {unit}, max {n[name][1]:.3f} {unit} ({m.sum()} samples)")
        return n[name]

    print(f"[log] {len(t)} samples over {t[-1] - t[0]:.1f} s; flying {flying.sum()} samples "
          f"({(t[flying][-1] - t[flying][0]) if flying.any() else 0:.1f} s)")
    if flying.any():
        stats("vio hover position error", pe, before & has_vio)
        stats("vio flying position error", pe, flying & has_vio)
        stats("vio flying velocity error", ve, flying & has_vio, "m/s")
        stats("vio flying yaw error", ye, flying & has_vio, "deg")
    else:                       # no plan on /min_snap/trajectory (e.g. the open-loop VIO stage)
        stats("vio position error, whole log", pe, has_vio)
        stats("vio velocity error, whole log", ve, has_vio, "m/s")
        stats("vio yaw error, whole log", ye, has_vio, "deg")
    stats("mpc TRUE drone vs reference", te, flying)
    sv = ~np.isnan(L["sp_solve_ms"]) & flying
    if sv.any():
        ms = L["sp_solve_ms"][sv]
        n["solve_ms"] = (float(np.median(ms)), float(np.percentile(ms, 95)))
        invalid = int(np.sum(L["sp_valid"][sv] == 0))
        print(f"[mpc] solve median {n['solve_ms'][0]:.1f} ms, p95 {n['solve_ms'][1]:.1f} ms, "
              f"{invalid} invalid of {sv.sum()} logged samples")
    tripped = [w for w in L["watchdog"] if w]
    if tripped:
        k = int(np.argmax(L["watchdog"] != ""))
        print(f"[watchdog] TRIPPED at t={t[k] - (t[flying][0] if flying.any() else t[0]):+.2f} s "
              f"(from the start of the plan): {tripped[0]}")
    else:
        print("[watchdog] never tripped")

    ok = not tripped
    print("[result] TRUE path vs gates:")
    t0 = t[flying][0] if flying.any() else t[0]
    for gate, cs in gate_crossings(t, true[:, 0:3], true[:, 6:9], gates):
        through = [c for c in cs if c["inside"] and c["forward"]]
        if not through:
            ok = False
            print(f"  {gate.name}: MISSED")
        for c in through:
            margin = gate.side / 2 - np.max(np.abs(c["offset"]))
            print(f"  {gate.name}: t={c['time'] - t0:5.2f}s  offset=({c['offset'][0]:+.2f}, {c['offset'][1]:+.2f}) m  "
                  f"frame margin {margin:.2f} m  angle {c['angle_deg']:4.1f} deg  speed {c['speed']:.2f} m/s")
    print(f"[result] {'ALL GATES PASSED' if ok else 'NOT ALL GATES PASSED' if not tripped else 'WATCHDOG ABORT'}")
    return ok, n


def plot(L, gates, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    true = _cols(L, "true", ("x", "y", "z", "roll", "pitch", "yaw"))
    vio = _cols(L, "vio", ("x", "y", "z", "roll", "pitch", "yaw"))
    ref = _cols(L, "ref", ("x", "y", "z"))
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.5))
    ax = axs[0]
    for g in gates:
        c = g.corners()
        ax.plot(c[[0, 1], 1], c[[0, 1], 0], color="darkorange", linewidth=4)
        ax.annotate(g.name, (g.center[1], g.center[0]), xytext=(6, -4), textcoords="offset points", fontsize=8)
    ax.plot(ref[:, 1], ref[:, 0], "k--", label="plan (reference)")
    ax.plot(true[:, 1], true[:, 0], "b-", label="true drone")
    ax.plot(vio[:, 1], vio[:, 0], color="crimson", linewidth=1, alpha=0.8, label="VIO estimate")
    ax.invert_xaxis(); ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_xlabel("Y (m)  <- left | right ->"); ax.set_ylabel("X forward (m)"); ax.set_title("Top-down (camera orientation)")
    flying = ~np.isnan(L["traj_t"]) & (L["traj_t"] >= 0)
    t = L["t"] - (L["t"][flying][0] if flying.any() else L["t"][0])
    axs[1].plot(t, np.linalg.norm(vio[:, 0:3] - true[:, 0:3], axis=1), color="crimson", label="VIO position error")
    axs[1].plot(t, np.linalg.norm(true[:, 0:3] - ref, axis=1), "b-", label="true drone vs reference")
    axs[1].axvline(0, color="gray", linewidth=0.8)
    axs[1].set_xlabel("time since the plan started (s)"); axs[1].set_ylabel("m"); axs[1].grid(alpha=0.3)
    axs[1].legend(fontsize=8); axs[1].set_title("Errors")
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi  # noqa: E731
    for i, name in ((5, "yaw"), (3, "roll"), (4, "pitch")):
        axs[2].plot(t, np.degrees(wrap(vio[:, i] - true[:, i])), label=name)
    axs[2].set_xlabel("time since the plan started (s)"); axs[2].set_ylabel("deg"); axs[2].grid(alpha=0.3)
    axs[2].legend(fontsize=8); axs[2].set_title("VIO attitude error")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"[plot] {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("log_dir")
    ap.add_argument("--gates", default=None,
                    help="gate manifest (default: /workspace/test_gates.json if present, else the repo's 3-gate course)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)
    gates_path = args.gates or ("/workspace/test_gates.json" if os.path.exists("/workspace/test_gates.json")
                                else DEFAULT_GATES)
    gates = gates_from_manifest(gates_path)
    L = load_log(os.path.join(args.log_dir, "flight_log.csv"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        print(f"[gates] {gates_path}")
        ok, _ = analyze(L, gates)
        b = bridge_report(os.path.join(args.log_dir, "bridge_log.csv"))
        if b is not None and b["status"].startswith("aborted"):
            ok = False
    text = buf.getvalue()
    print(text, end="")
    with open(os.path.join(args.log_dir, "flight_summary.txt"), "w") as f:
        f.write(text)
    if not args.no_plot:
        plot(L, gates, os.path.join(args.log_dir, "flight_report.png"),
             f"Isaac VIO + MPC flight: {os.path.basename(os.path.normpath(args.log_dir))}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
