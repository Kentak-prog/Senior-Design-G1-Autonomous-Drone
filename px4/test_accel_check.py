"""Self-tests for accel_check.py. Run: python3 test_accel_check.py

Synthetic cases, then a replay of real PX4 logs (vehicle_local_position vx/vy/vz at 50 Hz from arming)
through the check with the 3 m/s^2 default: quiet on the 2026-11-03 flights, fires on the 2026-11-24
crashes. Log replays are skipped (with a note) when a log or pyulog is missing.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import accel_check as A  # noqa: E402
from test_baro_check import LOGS, REPO  # noqa: E402


def _run(ts, vel, **kw):
    c = A.VelocityJumpCheck(**kw)
    for t, v in zip(ts, vel):
        r = c.update(t, v, t)
        if r:
            return t, r
    return None, None


def test_smooth_motion_is_quiet():
    ts = np.arange(0, 10, 0.02)
    v = np.stack([2.0 * np.sin(ts), np.zeros_like(ts), np.zeros_like(ts)], 1)   # up to 2 m/s^2
    rng = np.random.RandomState(0)
    t, r = _run(ts, v + rng.normal(0, 0.05, v.shape))
    assert t is None, r
    print("  smooth motion up to 2 m/s^2 with velocity noise: quiet  OK")


def test_single_jump_is_ignored():
    ts = np.arange(0, 5, 0.02)
    v = np.zeros((ts.size, 3))
    v[ts >= 2.0, 0] = 1.0                       # an estimator reset: one 1 m/s step, then steady
    t, r = _run(ts, v)
    assert t is None, r
    print("  one-off 1 m/s velocity step (estimator reset): ignored  OK")


def test_sustained_jerk_fires():
    ts = np.arange(0, 5, 0.02)
    v = np.zeros((ts.size, 3))
    m = (ts >= 2.0) & (ts < 3.0)
    v[m, 2] = 6.0 * (ts[m] - 2.0)                 # 6 m/s^2 for 1 s
    v[ts >= 3.0, 2] = 6.0
    t, r = _run(ts, v)
    assert t is not None and 2.3 <= t <= 2.6 and "too fast" in r, (t, r)
    print(f"  6 m/s^2 for 1 s: fires {t - 2.0:.2f} s after it starts  OK")


def test_repeated_sample_not_recounted():
    c = A.VelocityJumpCheck()
    assert c.update(0.0, [0, 0, 0], 0.0) is None
    for k in range(50):                             # the link hasn't delivered a new velocity for 1 s
        assert c.update(0.02 * k, [0, 0, 0], 0.0) is None
    assert len(c.samples) == 1
    print("  the same velocity sample fed repeatedly is counted once  OK")


def test_quintic_peak():
    import takeoff_hover as th
    T = th.climb_duration(1.0, 0.3)
    ts = np.linspace(0, T, 20001)
    v = np.array([th.quintic(1.0, T, t)[1] for t in ts])
    assert abs(np.max(np.abs(np.diff(v) / np.diff(ts))) - A.quintic_peak_accel(1.0, T)) < 1e-3
    print(f"  climb peak acceleration formula: {A.quintic_peak_accel(1.0, T):.3f} m/s^2 for 1 m at 0.3 m/s  OK")


def _replay(path):
    from pyulog import ULog
    u = ULog(path, ["vehicle_local_position", "vehicle_status"])
    d = {x.name: x.data for x in u.data_list}
    st = d["vehicle_status"]
    armed = st["arming_state"] == 2
    ta = st["timestamp"][np.argmax(armed)] / 1e6
    td = st["timestamp"][len(armed) - 1 - np.argmax(armed[::-1])] / 1e6
    lp = d["vehicle_local_position"]
    tl = lp["timestamp"] / 1e6
    c = A.VelocityJumpCheck()
    for t in np.arange(ta, min(td, tl[-1]), 0.02):
        v = [float(np.interp(t, tl, lp[k])) for k in ("vx", "vy", "vz")]
        if c.update(t, v, t):
            return t - ta
    return None


def test_real_logs():
    try:
        import pyulog  # noqa: F401
    except ImportError:
        print("  (pyulog not installed: real-log replay skipped)")
        return
    expect = {"crash_auto": (10.0, 11.3), "crash_pilot": (11.3, 12.3), "est_fell": None,
              "hover1": None, "hover2": None}
    for name, path in LOGS.items():
        if not os.path.exists(path):
            print(f"  ({name}: {os.path.relpath(path, REPO)} not here: skipped)")
            continue
        fired = _replay(path)
        want = expect[name]
        if want is None:
            assert fired is None, (name, fired)
            print(f"  {name}: quiet over the whole flight  OK")
        else:
            assert fired is not None and want[0] <= fired <= want[1], (name, fired)
            print(f"  {name}: fires {fired:.1f} s after arming  OK")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        t()
    print(f"\nAll tests passed. ({len(tests)})")
