"""
Preflight checks of PX4's state estimate, run by takeoff_hover.py and fly_min_snap_offboard.py before
they ask to fly. Everything is read over the same MAVLink link (no px4-listener, so it works from the
drone's container): parameters (PARAM_REQUEST_READ) and status streams requested with
SET_MESSAGE_INTERVAL. Nothing that moves the vehicle is sent.

Why (px4/INCIDENT_2026-11-24_TAKEOFF_FLYAWAY.md): on 2026-11-24 PX4 took its height from the VIO
(EKF2_HGT_REF = 3); the VIO lost tracking 1 s after arming and PX4 flew on a stale height. These checks
refuse to fly unless the estimate's sources are the intended ones and are healthy right now.

BLOCKING checks (any failure refuses):
  vision      EKF2_HGT_REF is not 3 (vision) and EKF2_EV_CTRL is 0: the VIO is not a source. (The VIO is
              off since 2026-11-25; a leftover EV_CTRL would make it one again if it were restarted.)
  estimator   ESTIMATOR_STATUS says horizontal velocity, vertical velocity, relative horizontal position
              and vertical position are valid, and the velocity / horizontal / vertical innovation test
              ratios are below 1 (above 1, PX4 is rejecting its measurements)
  attitude    SYS_STATUS reports the attitude estimate (AHRS) healthy. (PX4 does not set ESTIMATOR_STATUS's
              attitude flag on this drone, so AHRS health is used instead.)
  barometer   SYS_STATUS reports the barometer healthy
  baro stream SCALED_PRESSURE arriving (raw pressure for baro_check.py's in-flight height cross-check)
  flow        if EKF2_OF_CTRL = 1: OPTICAL_FLOW_RAD arriving, quality >= EKF2_OF_QMIN
  rangefinder if EKF2_RNG_CTRL > 0 or EKF2_HGT_REF = 2: DISTANCE_SENSOR arriving, inside its range
              (on the ground, also <= 0.30 m: the sensor sees the floor)
  steady      on the ground: PX4's height has moved < 0.03 m/s over the window (catches barometer warm-up
              drift: ~1.3 m in 15 min after boot on 2026-11-25); in the air: |vz| < 0.3 m/s

WARNINGS (printed, not blocking):
  EKF2_HGT_REF = 2 (rangefinder as the height reference): over a gate's bottom bar the rangefinder
  reads the bar, so PX4 sees the floor jump up and the height controller reacts.
"""

import time

import numpy as np

MAV_CMD_SET_MESSAGE_INTERVAL = 511
MSG = {"SYS_STATUS": 1, "ESTIMATOR_STATUS": 230, "DISTANCE_SENSOR": 132, "OPTICAL_FLOW_RAD": 106,
       "SCALED_PRESSURE": 29}
SENSOR_AHRS = 0x200000
SENSOR_ABS_PRESSURE = 0x8
EST_VEL_HORIZ, EST_VEL_VERT, EST_POS_HORIZ_REL, EST_POS_VERT_ABS = 0x2, 0x4, 0x8, 0x20
PARAMS = ("EKF2_HGT_REF", "EKF2_EV_CTRL", "EKF2_OF_CTRL", "EKF2_OF_QMIN", "EKF2_RNG_CTRL")
HGT_REF_NAMES = {0: "barometer", 1: "GNSS", 2: "rangefinder", 3: "vision"}


def _fresh(link, kind, now, max_age):
    item = link.last.get(kind)
    return item[1] if item is not None and now - item[0] <= max_age else None


def read_params(link, names=PARAMS, timeout=3.0, clock=time.monotonic, sleep=time.sleep):
    """Request each parameter (repeating every 0.5 s) until all have arrived or timeout. -> dict."""
    t0, last_req = clock(), None
    while clock() - t0 < timeout:
        missing = [n for n in names if n not in link.params]
        if not missing:
            break
        if last_req is None or clock() - last_req >= 0.5:
            for n in missing:
                link.m.mav.param_request_read_send(link.m.target_system, 1, n.encode(), -1)
            last_req = clock()
        link.poll()
        sleep(0.02)
    return {n: link.params.get(n) for n in names}


def run_checks(link, on_ground=True, window_s=3.0, clock=time.monotonic, sleep=time.sleep):
    """Run every check. Returns (ok, results, warnings); results = [(name, passed, detail), ...]."""
    for kind, mid in MSG.items():
        hz = {"DISTANCE_SENSOR": 10, "OPTICAL_FLOW_RAD": 10, "SCALED_PRESSURE": 20}.get(kind, 5)
        link.command(MAV_CMD_SET_MESSAGE_INTERVAL, mid, 1e6 / hz)
    p = read_params(link, clock=clock, sleep=sleep)

    # watch for window_s: PX4's height for the steadiness check, and the status streams
    zs, t0 = [], clock()
    while clock() - t0 < window_s:
        link.poll()
        if link.pos_ned is not None and link.last_pos_time is not None and clock() - link.last_pos_time < 0.2:
            zs.append((clock(), float(link.pos_ned[2]), float(link.vel_ned[2]) if link.vel_ned is not None else 0.0))
        sleep(0.05)
    now = clock()
    res, warn = [], []

    def add(name, ok, detail):
        res.append((name, bool(ok), detail))

    missing = [n for n, v in p.items() if v is None]
    if missing:
        add("parameters", False, f"PX4 did not answer for {', '.join(missing)}")
    hgt, ev = p.get("EKF2_HGT_REF"), p.get("EKF2_EV_CTRL")
    add("vision", hgt is not None and ev is not None and hgt != 3 and ev == 0,
        f"EKF2_HGT_REF={hgt} ({HGT_REF_NAMES.get(hgt, '?')}), EKF2_EV_CTRL={ev}"
        + ("" if ev == 0 else "  -> set EKF2_EV_CTRL 0 while the VIO is off")
        + ("  -> height must not come from the VIO" if hgt == 3 else ""))

    e = _fresh(link, "ESTIMATOR_STATUS", now, 1.0)
    if e is None:
        add("estimator", False, "no ESTIMATOR_STATUS in the last 1 s")
    else:
        need = {"velocity xy": EST_VEL_HORIZ, "velocity z": EST_VEL_VERT, "position xy": EST_POS_HORIZ_REL,
                "position z": EST_POS_VERT_ABS}
        bad = [n for n, bit in need.items() if not e.flags & bit]
        ratios = {"vel": e.vel_ratio, "pos xy": e.pos_horiz_ratio, "pos z": e.pos_vert_ratio}
        high = [f"{k} {v:.2f}" for k, v in ratios.items() if not (v < 1.0)]
        add("estimator", not bad and not high,
            ("invalid: " + ", ".join(bad) + "; " if bad else "")
            + "innovation ratios " + ", ".join(f"{k} {v:.2f}" for k, v in ratios.items())
            + (" (>= 1: PX4 rejecting measurements)" if high else ""))

    s = _fresh(link, "SYS_STATUS", now, 2.0)
    if s is None:
        add("attitude", False, "no SYS_STATUS in the last 2 s")
        add("barometer", False, "no SYS_STATUS in the last 2 s")
    else:
        for name, bit in (("attitude", SENSOR_AHRS), ("barometer", SENSOR_ABS_PRESSURE)):
            ok = (s.onboard_control_sensors_present & bit and s.onboard_control_sensors_enabled & bit
                  and s.onboard_control_sensors_health & bit)
            add(name, ok, "healthy" if ok else "SYS_STATUS: not present/enabled/healthy")
    add("baro stream", _fresh(link, "SCALED_PRESSURE", now, 0.5) is not None,
        "SCALED_PRESSURE arriving (the in-flight barometer cross-check needs it)"
        if _fresh(link, "SCALED_PRESSURE", now, 0.5) is not None
        else "no SCALED_PRESSURE in the last 0.5 s: the in-flight barometer cross-check would abort")

    if p.get("EKF2_OF_CTRL") == 1:
        f = _fresh(link, "OPTICAL_FLOW_RAD", now, 0.5)
        qmin = p.get("EKF2_OF_QMIN") or 0
        if f is None:
            add("flow", False, "EKF2_OF_CTRL=1 but no OPTICAL_FLOW_RAD in the last 0.5 s")
        else:
            add("flow", f.quality >= qmin, f"quality {f.quality} (EKF2_OF_QMIN {qmin})")

    if (p.get("EKF2_RNG_CTRL") or 0) > 0 or hgt == 2:
        d = _fresh(link, "DISTANCE_SENSOR", now, 0.5)
        if d is None:
            add("rangefinder", False, "no DISTANCE_SENSOR in the last 0.5 s")
        else:
            dist = d.current_distance / 100.0
            in_range = d.min_distance <= d.current_distance <= d.max_distance
            ok = in_range and (not on_ground or dist <= 0.30)
            add("rangefinder", ok, f"{dist:.2f} m (range {d.min_distance / 100:.2f}-{d.max_distance / 100:.2f} m)"
                + ("" if not on_ground or dist <= 0.30 else "  -> on the ground it should see the floor (<= 0.30 m)"))

    if len(zs) < 5:
        add("steady", False, "too few position samples to judge")
    elif on_ground:
        rate = abs(zs[-1][1] - zs[0][1]) / max(zs[-1][0] - zs[0][0], 1e-3)
        add("steady", rate < 0.03, f"height moved {rate * 100:.1f} cm/s while still"
            + ("" if rate < 0.03 else "  -> let the barometer warm up until PX4's height settles"))
    else:
        vz = max(abs(z[2]) for z in zs)
        add("steady", vz < 0.3, f"|vz| up to {vz:.2f} m/s over {window_s:.0f} s")

    if hgt == 2:
        warn.append("EKF2_HGT_REF=2: height comes from the rangefinder; over a gate's bottom bar PX4 will see "
                    "the 'floor' jump up (see the MTF-02P notes: EKF2_HGT_REF 0 with EKF2_RNG_CTRL 1 avoids it)")
    return all(ok for _, ok, _ in res), res, warn


def report(ok, results, warnings, out=print):
    out("[preflight] " + ("PASS" if ok else "FAIL"))
    for name, passed, detail in results:
        out(f"  {'ok  ' if passed else 'FAIL'}  {name:12s} {detail}")
    for w in warnings:
        out(f"  WARN  {w}")
