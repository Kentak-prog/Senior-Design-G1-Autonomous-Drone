"""Benchmark MpcCtbrController solver option sets / lookaheads on a track in
QuadSim. Run from anywhere:  python bench_mpc_ctbr.py [--track straight3] [--max-time 12]
Prints mean / p95 / max solve ms, lap time, completion for each configuration."""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from course import Course, TRACKS_DIR
from drone_params import QuadParams
from fly_course import run_controller
from mpc_ctbr import MpcCtbrController, SOLVER_PRESETS


def bench(name, course, P, max_time=60.0, **kw):
    c = MpcCtbrController(P, course, **kw)
    log, s = run_controller(course, c, P, max_time=max_time)
    ref = np.array([c._ref.reference_state(t)[:3] for t in log["t"]])
    err = np.linalg.norm(log["pos"][:, 0] - ref, axis=1)
    from runner import edge_margins
    mg = np.nanmin(edge_margins(log["pos"][:, 0], course))
    ms = 1e3 * np.array(c.solve_times)
    print(f"{name:34s} mean {ms.mean():6.1f}  p95 {np.percentile(ms, 95):6.1f}  max {ms.max():6.1f} ms"
          f" | lap {s['lap_time']:6.2f}s gates {s['gates_passed']}/{s['n_gates']} "
          f"done={s['completed']} fails={c.n_failures} fb={c.n_budget_fallbacks} | trk rms {np.sqrt((err**2).mean()):.3f} max {err.max():.3f} m, margin {mg:.2f}", flush=True)
    return c, s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="straight3")
    ap.add_argument("--max-time", type=float, default=60.0)
    ap.add_argument("--presets", nargs="*", default=list(SOLVER_PRESETS))
    ap.add_argument("--lookaheads", nargs="*", type=int, default=[])
    a = ap.parse_args()
    P = QuadParams()
    course = Course.load(os.path.join(TRACKS_DIR, a.track + ".json"))
    for n in a.presets:
        bench(f"preset {n}", course, P, a.max_time, solver=n)
    for k in a.lookaheads:
        bench(f"lookahead {k}", course, P, a.max_time, rate_lookahead=k)
