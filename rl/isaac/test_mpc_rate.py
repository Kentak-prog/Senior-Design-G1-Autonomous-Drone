"""Self-tests for RateAwareMpc and the delay bench. Run with DRONE_VEHICLE=starling2max:
    python test_mpc_rate.py                 (IPOPT backend, ~2 min)
    MPC_RATE_BACKEND=acados python test_mpc_rate.py   (~25 s; needs acados, README_mpc_rate.md)
Plain functions, no pytest, no Isaac. The flights are the 2026-11-12 bench cases
(README_mpc_rate.md): the slowed rate loop that crashes MpcCtbrController live."""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bench_mpc_delay as bench                     # noqa: E402
from course import Course, TRACKS_DIR              # noqa: E402
from drone_params import QuadParams                # noqa: E402

assert os.environ.get("DRONE_VEHICLE") == "starling2max", "run with DRONE_VEHICLE=starling2max"
P = QuadParams()
BACKEND = os.environ.get("MPC_RATE_BACKEND", "ipopt")
COURSES = {n: Course.load(os.path.join(TRACKS_DIR, n + ".json")) for n in ("straight3_pink36", "slalom5_pink36")}


def test_bench_rate_loop_matches_isaac():
    got = [bench.rise_time(P, bench.KP_SCALE_ISAAC, a) for a in range(3)]
    assert all(abs(g - m) <= 0.008 for g, m in zip(got, bench.ISAAC_RISE)), (got, bench.ISAAC_RISE)
    print(f"  rate loop 63 % rise {[round(1e3 * g) for g in got]} ms vs Isaac {[round(1e3 * m) for m in bench.ISAAC_RISE]}  OK")


def test_ctbr_mpc_crashes_with_the_isaac_rate_loop():
    # the reproduction: Tony's MPC, Isaac's rate loop, no delay -> tilt crash after gate 1
    c = COURSES["slalom5_pink36"]
    r = bench.fly(c, P, bench.make("ctbr", P, c), 0.0, bench.KP_SCALE_ISAAC, max_time=20.0)
    assert r["crashed"] and r["gates"] < r["n"], r
    print(f"  MpcCtbrController: {r['gates']}/{r['n']} gates, crash {r['kind']} at {r['t']:.2f} s  OK (reproduced)")


def _rate_flight(track, delay_ms, told_ms):
    c = COURSES[track]
    r = bench.fly(c, P, bench.make("rate", P, c, told_ms / 1e3, BACKEND), delay_ms / 1e3,
                  bench.KP_SCALE_ISAAC, max_time=45.0)
    assert r["done"] and not r["crashed"], r
    assert r["min_margin"] > P.collision_radius, r            # crossed with the drone clear of the frame
    assert r["max_tilt"] < 45.0 and r["osc_dps"] < 6.0, r
    if BACKEND == "acados":                                    # real time at 50 Hz with room to spare
        assert r["solve_ms_p95"] < 5.0, r
    print(f"  RateAwareMpc[{BACKEND}] {track} {delay_ms} ms (told {told_ms}): {r['gates']}/{r['n']} in {r['t']:.2f} s, "
          f"tilt {r['max_tilt']:.1f}, osc {r['osc_dps']:.2f} dps, margin {r['min_margin']:.3f} m, "
          f"solve median {r['solve_ms_median']:.1f} ms  OK")


def test_rate_mpc_flies_slalom5_with_the_isaac_rate_loop():
    _rate_flight("slalom5_pink36", 0, 0)


def test_rate_mpc_flies_with_40ms_delay_compensated_and_not():
    _rate_flight("straight3_pink36", 40, 40)
    _rate_flight("straight3_pink36", 40, 0)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    t0 = time.time()
    for t in tests:
        print(t.__name__)
        t()
    print(f"\nAll tests passed. ({len(tests)})  [{time.time() - t0:.0f} s]")
