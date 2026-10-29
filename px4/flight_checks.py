"""
Go / no-go checks and the in-flight watchdog for the Isaac VIO + MPC run
(ros2_nodes/flight_monitor.py is the ROS wrapper). No ROS here, so it is
tested on its own (px4/test_flight_checks.py).

TopicStats       arrival times + header stamps of one topic -> rate, stamp lag
preflight_checks every condition that must hold before engaging OFFBOARD
Watchdog         sim-only safety net: the estimate vs the simulator's truth,
                 the truth vs the reference, altitude. Latches the first reason.

All times are seconds on ONE clock (the monitor node's ROS clock, sim time
with use_sim_time:=true), so a stamp lag is "how old the message is when it
arrives" and is only meaningful if every node shares that clock.
"""

import math
from collections import deque

import numpy as np


class TopicStats:
    """Keeps the last `window` seconds of (arrival, stamp) for one topic."""

    def __init__(self, window=5.0):
        self.window = window
        self.rx = deque()          # (arrival_time, stamp or None)
        self.count = 0

    def add(self, arrival, stamp=None):
        self.rx.append((arrival, stamp))
        self.count += 1
        while self.rx and arrival - self.rx[0][0] > self.window:
            self.rx.popleft()

    def rate(self, now=None):
        """Messages per second over the window (0 if fewer than 2, or stale at `now`)."""
        if len(self.rx) < 2:
            return 0.0
        if now is not None and now - self.rx[-1][0] > 1.0:
            return 0.0
        span = self.rx[-1][0] - self.rx[0][0]
        return (len(self.rx) - 1) / span if span > 0 else 0.0

    def stamp_lag(self):
        """Median (arrival - stamp) in seconds, None without stamps."""
        lags = [a - s for a, s in self.rx if s is not None]
        return float(np.median(lags)) if lags else None


def _check(name, value, ok, limit):
    return {"check": name, "value": value, "limit": limit, "ok": bool(ok)}


def preflight_checks(rates, lags, use_sim_time, clock_rate, vio_err, vio_yaw_err_deg, vio_std,
                     traj, setpoints, alt, limits=None):
    """
    rates/lags: {"imu": .., "image": .., "vio": .., "truth": .., "setpoint": ..} (Hz, s)
    vio_err / vio_yaw_err_deg: recent VIO vs truth (lists, m / deg); vio_std: last position std (m) or None
    traj: {"points": n, "duration": s, "start_err": m from the drone} or None. With
          limits["plan_on_engage"] (min_snap_node start_topic) there must be NO trajectory yet.
    setpoints: {"valid_frac": .., "solve_p95_ms": .., "thrust": N, "tilt_deg": ..} or None
    alt: truth altitude (m) or None
    Returns a list of {"check", "value", "limit", "ok"}.
    """
    L = {"imu_hz": 150.0, "image_hz": 8.0, "vio_hz": 40.0, "truth_hz": 20.0, "setpoint_hz": 40.0,
         "max_lag_s": 0.25, "vio_err_m": 0.30, "vio_yaw_deg": 3.0, "vio_std_m": 0.30,
         "start_err_m": 0.5, "valid_frac": 0.99, "solve_p95_ms": 15.0, "tilt_deg": 10.0,
         "hover_thrust_tol": 0.15, "min_alt": 1.5, "hover_thrust": None, "plan_on_engage": False}
    L.update(limits or {})
    out = [
        _check("use_sim_time on the monitor", use_sim_time, use_sim_time, True),
        _check("/clock publishing (Hz)", round(clock_rate, 1), clock_rate > 10.0, "> 10"),
    ]
    for key, label in (("imu", "IMU"), ("image", "camera image"), ("vio", "/vio/odometry"),
                       ("truth", "ground-truth pose"), ("setpoint", "/mpc/attitude_setpoint")):
        r = rates.get(key, 0.0)
        out.append(_check(f"{label} rate (Hz)", round(r, 1), r >= L[f"{key}_hz"], f">= {L[f'{key}_hz']:g}"))
    for key, label in (("imu", "IMU"), ("image", "camera"), ("vio", "VIO")):
        lag = lags.get(key)
        out.append(_check(f"{label} stamp lag (s)", None if lag is None else round(lag, 3),
                          lag is not None and abs(lag) <= L["max_lag_s"], f"|lag| <= {L['max_lag_s']}"))
    e = max(vio_err) if vio_err else None
    out.append(_check("VIO position error, last window (m, max)", None if e is None else round(e, 3),
                      e is not None and e <= L["vio_err_m"], f"<= {L['vio_err_m']}"))
    y = max(vio_yaw_err_deg) if vio_yaw_err_deg else None
    out.append(_check("VIO yaw error, last window (deg, max)", None if y is None else round(y, 2),
                      y is not None and y <= L["vio_yaw_deg"], f"<= {L['vio_yaw_deg']}"))
    out.append(_check("VIO reported position std (m)", None if vio_std is None else round(vio_std, 3),
                      vio_std is not None and vio_std <= L["vio_std_m"], f"<= {L['vio_std_m']}"))
    if L["plan_on_engage"]:
        out.append(_check("no trajectory before engage", "none" if traj is None else "PUBLISHED", traj is None,
                          "min_snap_node waits for the bridge"))
    elif traj is None:
        out.append(_check("min-snap trajectory received", None, False, "latched on /min_snap/trajectory"))
    else:
        out.append(_check("trajectory start vs drone (m)", round(traj["start_err"], 2),
                          traj["start_err"] <= L["start_err_m"], f"<= {L['start_err_m']}"))
    if setpoints is None:
        out.append(_check("MPC setpoints", None, False, "publishing"))
    else:
        out.append(_check("MPC valid solves (fraction)", round(setpoints["valid_frac"], 3),
                          setpoints["valid_frac"] >= L["valid_frac"], f">= {L['valid_frac']}"))
        out.append(_check("MPC solve time p95 (ms)", round(setpoints["solve_p95_ms"], 1),
                          setpoints["solve_p95_ms"] <= L["solve_p95_ms"], f"<= {L['solve_p95_ms']}"))
        out.append(_check("MPC setpoint tilt at hover (deg)", round(setpoints["tilt_deg"], 1),
                          setpoints["tilt_deg"] <= L["tilt_deg"], f"<= {L['tilt_deg']}"))
        if L["hover_thrust"]:
            rel = abs(setpoints["thrust"] / L["hover_thrust"] - 1.0)
            out.append(_check("MPC thrust vs hover thrust", round(rel, 3), rel <= L["hover_thrust_tol"],
                              f"within {L['hover_thrust_tol']:.0%}"))
    out.append(_check("altitude (m)", None if alt is None else round(alt, 2),
                      alt is not None and alt >= L["min_alt"], f">= {L['min_alt']}"))
    return out


def format_checks(checks):
    w = max(len(c["check"]) for c in checks)
    lines = [f"  {'OK ' if c['ok'] else 'NO '} {c['check']:<{w}}  {str(c['value']):>8}   ({c['limit']})"
             for c in checks]
    go = all(c["ok"] for c in checks)
    lines.append(f"PREFLIGHT: {'GO' if go else 'NO-GO'} ({sum(not c['ok'] for c in checks)} failing)")
    return "\n".join(lines), go


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Watchdog:
    """
    Sim-only safety net, fed at the monitor's rate. Trips (and stays tripped)
    when, continuously for `hold_s`:
      - the VIO position is more than max_vio_err_m from the truth, or
      - the true drone is more than max_track_err_m from the reference, or
    immediately when the true altitude drops below min_alt_m or the VIO
    stops publishing for stale_s.
    """

    def __init__(self, max_vio_err_m=1.0, max_track_err_m=2.0, min_alt_m=0.5, hold_s=0.2, stale_s=0.5):
        self.max_vio, self.max_track, self.min_alt = max_vio_err_m, max_track_err_m, min_alt_m
        self.hold_s, self.stale_s = hold_s, stale_s
        self.reason = None
        self._since = {}

    def _held(self, key, bad, t):
        if not bad:
            self._since.pop(key, None)
            return False
        self._since.setdefault(key, t)
        return t - self._since[key] >= self.hold_s

    def update(self, t, true_pos, est_pos=None, est_time=None, ref_pos=None):
        """Returns the trip reason (str) or None. Arguments are 3-vectors or None."""
        if self.reason is not None:
            return self.reason
        true_pos = np.asarray(true_pos, dtype=float)
        if true_pos[2] < self.min_alt:
            self.reason = f"true altitude {true_pos[2]:.2f} m < {self.min_alt} m"
        elif est_time is None or t - est_time > self.stale_s:
            self.reason = f"no VIO estimate for more than {self.stale_s} s"
        else:
            e = float(np.linalg.norm(np.asarray(est_pos, dtype=float) - true_pos))
            if self._held("vio", e > self.max_vio, t):
                self.reason = f"VIO {e:.2f} m from the truth (> {self.max_vio} m for {self.hold_s} s)"
            elif ref_pos is not None:
                d = float(np.linalg.norm(true_pos - np.asarray(ref_pos, dtype=float)))
                if self._held("track", d > self.max_track, t):
                    self.reason = f"drone {d:.2f} m off the reference (> {self.max_track} m for {self.hold_s} s)"
        return self.reason
