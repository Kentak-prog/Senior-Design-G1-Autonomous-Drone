"""Self-tests for px4/flight_checks.py (behind ros2_nodes/flight_monitor.py). Run: python3 test_flight_checks.py

No ROS. Topic rate / stamp-lag statistics, the preflight GO / NO-GO table,
and the in-flight watchdog.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import flight_checks as fc                        # noqa: E402


def test_topic_stats():
    s = fc.TopicStats(window=2.0)
    for i in range(601):                           # 250 Hz for 2.4 s, stamped 20 ms before arrival
        t = i / 250.0
        s.add(t, t - 0.02)
    assert abs(s.rate() - 250.0) < 1.0 and abs(s.stamp_lag() - 0.02) < 1e-9
    assert len(s.rx) <= 501                        # only the window is kept
    assert s.rate(now=2.4 + 1.5) == 0.0            # stale -> 0 Hz
    assert fc.TopicStats().rate() == 0.0 and fc.TopicStats().stamp_lag() is None
    print("  250 Hz topic: rate, median stamp lag, window, stale -> 0  OK")


def _good():
    return dict(rates={"imu": 250, "image": 20, "vio": 50, "truth": 60, "setpoint": 50},
                lags={"imu": 0.004, "image": 0.05, "vio": 0.01}, use_sim_time=True, clock_rate=60.0,
                vio_err=[0.05, 0.12], vio_yaw_err_deg=[0.3], vio_std=0.08,
                traj={"points": 800, "duration": 16.0, "start_err": 0.1},
                setpoints={"valid_frac": 1.0, "solve_p95_ms": 11.0, "thrust": 14.8, "tilt_deg": 1.0}, alt=2.5)


def test_preflight_go_and_each_no_go():
    text, go = fc.format_checks(fc.preflight_checks(**_good()))
    assert go and "PREFLIGHT: GO" in text, text
    bad_cases = [
        (dict(use_sim_time=False), "use_sim_time"),
        (dict(clock_rate=0.0), "/clock"),
        (dict(rates={"imu": 250, "image": 0, "vio": 50, "truth": 60, "setpoint": 50}), "camera image rate"),
        (dict(lags={"imu": 3.2, "image": 0.05, "vio": 0.01}), "IMU stamp lag"),     # wall vs sim stamps
        (dict(vio_err=[0.1, 0.6]), "VIO position error"),
        (dict(vio_yaw_err_deg=[5.0]), "VIO yaw error"),
        (dict(traj=None), "trajectory received"),
        (dict(traj={"points": 800, "duration": 16.0, "start_err": 1.2}), "trajectory start"),
        (dict(setpoints={"valid_frac": 0.9, "solve_p95_ms": 11.0, "thrust": 14.8, "tilt_deg": 1.0}), "valid solves"),
        (dict(setpoints={"valid_frac": 1.0, "solve_p95_ms": 40.0, "thrust": 14.8, "tilt_deg": 1.0}), "solve time"),
        (dict(alt=0.3), "altitude"),
    ]
    for change, name in bad_cases:
        kw = _good()
        kw.update(change)
        checks = fc.preflight_checks(**kw)
        failing = [c["check"] for c in checks if not c["ok"]]
        assert len(failing) == 1 and name in failing[0], (name, failing)
        assert not fc.format_checks(checks)[1]
    kw = _good()
    checks = fc.preflight_checks(**kw, limits={"hover_thrust": 12.0})       # 14.8 N is 23 % off
    assert [c["check"] for c in checks if not c["ok"]] == ["MPC thrust vs hover thrust"]
    kw["traj"] = None                                     # plan on engage: no trajectory yet is the GO state
    assert fc.format_checks(fc.preflight_checks(**kw, limits={"plan_on_engage": True}))[1]
    kw["traj"] = _good()["traj"]                          # ... and a trajectory already running is NO-GO
    checks = fc.preflight_checks(**kw, limits={"plan_on_engage": True})
    assert [c["check"] for c in checks if not c["ok"]] == ["no trajectory before engage"]
    print(f"  all good -> GO; each of {len(bad_cases) + 2} single faults -> NO-GO naming that check  OK")


def test_watchdog():
    w = fc.Watchdog(max_vio_err_m=1.0, max_track_err_m=2.0, min_alt_m=0.5, hold_s=0.2, stale_s=0.5)
    for i in range(100):                                       # healthy for 2 s
        t = i * 0.02
        assert w.update(t, [t, 0, 2.5], [t + 0.3, 0, 2.5], t, [t, 0.5, 2.5]) is None
    assert w.update(2.0, [2, 0, 2.5], [3.5, 0, 2.5], 2.0) is None   # one bad sample: not yet
    t = 2.0
    while w.reason is None and t < 3.0:
        t += 0.02
        w.update(t, [2, 0, 2.5], [3.5, 0, 2.5], t)
    assert "VIO 1.50 m" in w.reason and 0.19 < t - 2.0 < 0.25, (w.reason, t)
    assert w.update(9.0, [0, 0, 2.5], [0, 0, 2.5], 9.0) == w.reason      # latched

    w = fc.Watchdog()
    assert w.update(0.0, [0, 0, 2.5], [0, 0, 2.5], 0.0, [0, 0, 2.5]) is None
    assert "off the reference" in next(r for r in (w.update(0.02 * i, [0, 0, 2.5], [0, 0, 2.5], 0.02 * i,
                                                            [3, 0, 2.5]) for i in range(1, 30)) if r)
    assert "altitude" in fc.Watchdog().update(0.0, [0, 0, 0.3], [0, 0, 0.3], 0.0)
    assert "no VIO" in fc.Watchdog().update(5.0, [0, 0, 2.5], [0, 0, 2.5], 4.0)
    assert "no VIO" in fc.Watchdog().update(5.0, [0, 0, 2.5])
    print("  watchdog: VIO divergence held 0.2 s, off-reference, low altitude, VIO stale; latches  OK")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        t()
    print(f"\nAll tests passed. ({len(tests)})")
