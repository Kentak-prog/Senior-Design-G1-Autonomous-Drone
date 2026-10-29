"""Self-tests for px4/analyze_vio_mpc_flight.py. Run: python3 test_analyze_vio_mpc_flight.py

No ROS. Writes synthetic flight_log.csv files with ros2_nodes/flight_monitor.py's
exact columns (a plan flown perfectly, one that skips a gate, one the watchdog
stopped) and checks the verdict, the exit code and the outputs.
"""

import csv
import io
import os
import sys
import tempfile
from contextlib import redirect_stdout

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(REPO, "ros2_nodes"), os.path.join(REPO, "Trajectory"), os.path.join(REPO, "control")]

import analyze_vio_mpc_flight as an                    # noqa: E402
from flight_monitor import COLUMNS                     # noqa: E402
from gate_path import gates_from_manifest              # noqa: E402
from min_snap_planner import plan_course, trajectory_points   # noqa: E402

GATES = gates_from_manifest(an.DEFAULT_GATES)
_PLAN = {}


def _plan():
    if not _PLAN:
        _, _, ref = plan_course(GATES, np.array([0.0, 0.0, 2.5]), 0.0, 3.0, 3.0, yaw_mode="next_gate",
                                optimize_times=False)
        _PLAN["ts"], _PLAN["x"], _ = trajectory_points(ref)
    return _PLAN["ts"], _PLAN["x"]


def _write_log(d, true_offset=None, vio_noise=0.05, trip_at=None, seed=0):
    rng = np.random.default_rng(seed)
    ts, X = _plan()
    t_engage = 100.0
    with open(os.path.join(d, "flight_log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for k in range(-100, len(ts)):                          # 2 s of hover before the plan, at 50 Hz
            tt = ts[0] + 0.02 * k if k < 0 else ts[k]
            x = X[:, max(k, 0)].copy()
            if k < 0:
                x[6:9] = 0.0
            true = x.copy()
            if true_offset is not None and k >= 0:
                true[0:3] += true_offset(tt)
            vio = true + np.r_[rng.normal(0, vio_noise, 3), np.zeros(9)]
            phase = "go" if k < 0 else "flying"
            ref = [x[i] for i in (0, 1, 2, 5, 6, 7, 8)] + [tt] if k >= 0 else [""] * 8
            reason = "VIO 1.20 m from the truth" if trip_at is not None and tt >= trip_at else ""
            w.writerow([t_engage + tt, phase] + list(true[0:9]) + list(vio[0:9]) + [0.08, 0.01] + ref
                       + [true[3], true[4], true[5], 14.7, 8.5, 1, reason])


def _run(d):
    out = io.StringIO()
    with redirect_stdout(out):
        code = an.main([d, "--gates", an.DEFAULT_GATES, "--no-plot"])
    return code, out.getvalue()


def test_perfect_flight_passes():
    with tempfile.TemporaryDirectory() as d:
        _write_log(d)
        code, text = _run(d)
        assert code == 0 and "ALL GATES PASSED" in text and text.count("frame margin") == 3, text
        assert "watchdog] never tripped" in text and "vio hover position error" in text
        assert os.path.exists(os.path.join(d, "flight_summary.txt"))
    print("  plan flown exactly, 5 cm VIO noise -> 3 gate passes, exit 0, summary written  OK")


def test_missed_gate_fails():
    with tempfile.TemporaryDirectory() as d:
        _write_log(d, true_offset=lambda t: np.array([0.0, 0.0, 3.0]))     # 3 m too high the whole way
        code, text = _run(d)
        assert code == 1 and "MISSED" in text and "NOT ALL GATES PASSED" in text, text
    print("  path 3 m above the gates -> MISSED, exit 1  OK")


def test_watchdog_abort_fails_and_plot():
    with tempfile.TemporaryDirectory() as d:
        _write_log(d, trip_at=4.0)
        code, text = _run(d)
        assert code == 1 and "TRIPPED at t=+4.00 s" in text and "WATCHDOG ABORT" in text, text
        an.plot(an.load_log(os.path.join(d, "flight_log.csv")), GATES, os.path.join(d, "r.png"), "test")
        assert os.path.getsize(os.path.join(d, "r.png")) > 10000
    print("  watchdog reason in the log -> TRIPPED at its plan time, exit 1; plot renders  OK")


def test_bridge_log_hover_score():
    with tempfile.TemporaryDirectory() as d:
        _write_log(d)
        with open(os.path.join(d, "bridge_log.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "px4_x", "px4_y", "px4_z", "qw_ned", "qx_ned", "qy_ned", "qz_ned",
                        "p_frd", "q_frd", "r_frd", "thrust_norm", "thrust_n"])
            for k in range(750):                                     # 15 s at 50 Hz, a 0.2 m drift east
                w.writerow([0.02 * k, 1.0 + 0.2 * k / 749, 2.0, 2.5 - 0.05 * (k > 400), 1, 0, 0, 0, 0, 0, 0, 0.556, 14.7])
            w.writerow(["# completed: reached --max-duration 15 s"])
        code, text = _run(d)
        assert code == 0 and "max horizontal 0.20 m, max vertical 0.05 m" in text and "completed" in text, text
        with open(os.path.join(d, "bridge_log.csv"), "a", newline="") as f:
            f.write("# aborted: watchdog: VIO 1.2 m from the truth" + chr(10))
        assert _run(d)[0] == 1                                       # a bridge abort fails the run
    print("  bridge_log.csv: OFFBOARD excursion 0.20 m / 0.05 m, status; an abort -> exit 1  OK")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        t()
    print(f"\nAll tests passed. ({len(tests)})")
