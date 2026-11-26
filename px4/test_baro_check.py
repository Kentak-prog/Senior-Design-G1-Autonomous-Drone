"""Self-tests for baro_check.py. Run: python3 test_baro_check.py

Synthetic cases, then a replay of real PX4 logs (vehicle_local_position.z vs vehicle_air_data.baro_alt_meter
at 50 Hz) through the check with its default settings: it must fire on the 2026-11-24 crashes and stay quiet
on the 2026-11-03 hovers. Log replays are skipped (with a note) when a log or pyulog is missing.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baro_check as B  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = {
    "crash_auto": os.path.join(REPO, "flight_logs", "2026-11-24", "drone", "px4_logs", "21_49_16.ulg"),
    "crash_pilot": os.path.join(REPO, "flight_logs", "2026-11-24", "drone", "px4_logs", "21_49_43.ulg"),
    "est_fell": os.path.join(REPO, "flight_logs", "2026-11-03", "px4_logs", "22_56_38.ulg"),
    "hover1": os.path.join(REPO, "flight_logs", "2026-11-03", "px4_logs", "23_20_53.ulg"),
    "hover2": os.path.join(REPO, "flight_logs", "2026-11-03", "px4_logs", "23_34_32.ulg"),
}


def _run(ts, est_up, baro_up, **kw):
    c = B.BaroCrossCheck(**kw)
    for t, e, b in zip(ts, est_up, baro_up):
        r = c.update(t, -e, b, t)
        if r:
            return t, r
    return None, None


def test_pressure_altitude():
    assert abs(B.pressure_altitude_m(1013.25)) < 1e-9
    d = B.pressure_altitude_m(1013.25 - 0.12) - B.pressure_altitude_m(1013.25)
    assert 0.9 < d < 1.1, d                       # ~0.12 hPa per metre near sea level
    print(f"  pressure altitude: 0.12 hPa lower = {d:.2f} m higher  OK")


def test_agreement_is_quiet():
    rng = np.random.RandomState(0)
    ts = np.arange(0, 30, 0.02)
    climb = np.clip(ts / 6.0, 0, 1) * 1.0                       # 1 m climb, then hover
    t, r = _run(ts, climb, climb + rng.normal(0, 0.3, ts.size))   # barometer noise ±0.3 m
    assert t is None, r
    print("  estimate and a noisy (±0.3 m) barometer agree on a 1 m climb: quiet  OK")


def test_step_offset_is_absorbed():
    ts = np.arange(0, 30, 0.02)
    est = np.clip(ts / 6.0, 0, 1)
    baro = est - np.where(ts > 3.0, 1.0, 0.0)                  # prop-wash / ground effect: 1 m step low
    t, r = _run(ts, est, baro)
    assert t is None, r
    print("  barometer steps 1 m low at takeoff (prop wash): absorbed, quiet  OK")


def test_runaway_fires():
    ts = np.arange(0, 15, 0.02)
    est = np.where(ts < 2, 0.0, -0.15 * (ts - 2))              # estimate: "sinking"
    baro = 0.5 * np.maximum(ts - 2, 0.0) ** 1.3                 # truth: climbing away
    t, r = _run(ts, est, baro)
    assert t is not None and t < 6.5, (t, r)
    print(f"  estimate sinks while the barometer climbs: fires at {t:.1f} s  OK")


def test_lost_barometer():
    c = B.BaroCrossCheck()
    assert c.update(0.0, 0.0, 10.0, 0.0) is None
    assert c.update(0.5, 0.0, None) is None
    r = c.update(1.6, 0.0, None)
    assert r and "no barometer" in r
    c = B.BaroCrossCheck()                                       # never any barometer at all
    assert c.update(0.0, 0.0, None) is None and c.update(1.5, 0.0, None)
    c = B.BaroCrossCheck()                                       # the same old sample, repeated
    assert c.update(0.0, 0.0, 10.0, 0.0) is None and c.update(1.6, 0.0, 10.0, 0.0)
    print("  barometer stream stops, never starts, or goes stale for > 1 s: reported  OK")


def _replay(path):
    from pyulog import ULog
    u = ULog(path, ["vehicle_local_position", "vehicle_air_data", "vehicle_status"])
    d = {x.name: x.data for x in u.data_list}
    st = d["vehicle_status"]
    armed = st["arming_state"] == 2
    ta = st["timestamp"][np.argmax(armed)] / 1e6
    td = st["timestamp"][len(armed) - 1 - np.argmax(armed[::-1])] / 1e6
    tl, z = d["vehicle_local_position"]["timestamp"] / 1e6, d["vehicle_local_position"]["z"]
    tb, b = d["vehicle_air_data"]["timestamp"] / 1e6, d["vehicle_air_data"]["baro_alt_meter"]
    c = B.BaroCrossCheck()
    biggest = 0.0
    for t in np.arange(ta, min(td, tl[-1]), 0.02):
        i = np.searchsorted(tb, t) - 1
        if i < 0:
            continue
        r = c.update(t, float(np.interp(t, tl, z)), float(b[i]), float(tb[i]))
        biggest = max(biggest, abs(c.gap))
        if r:
            return t - ta, biggest
    return None, biggest


def test_real_logs():
    try:
        import pyulog  # noqa: F401
    except ImportError:
        print("  (pyulog not installed: real-log replay skipped)")
        return
    expect = {"crash_auto": (6.0, 8.0), "crash_pilot": (12.5, 14.5), "est_fell": (7.0, 9.0),
              "hover1": None, "hover2": None}
    for name, path in LOGS.items():
        if not os.path.exists(path):
            print(f"  ({name}: {os.path.relpath(path, REPO)} not here: skipped)")
            continue
        fired, biggest = _replay(path)
        want = expect[name]
        if want is None:
            assert fired is None, (name, fired)
            print(f"  {name}: quiet over the whole flight (largest gap {biggest:.2f} m)  OK")
        else:
            assert fired is not None and want[0] <= fired <= want[1], (name, fired)
            print(f"  {name}: fires {fired:.1f} s after arming  OK")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        t()
    print(f"\nAll tests passed. ({len(tests)})")
