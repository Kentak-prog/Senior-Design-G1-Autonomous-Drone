"""Self-tests for px4/sysid.py. Run: python3 test_sysid.py

A synthetic Starling 2 Max flight is generated from known "true" parameters
that differ from the profile (so a fit that just echoed the profile would
fail), written as PX4-style topics (FRD / NED, microsecond timestamps, each
topic at its own rate), and the fits must recover the true values.
"""

import dataclasses
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import sysid  # noqa: E402
from vehicles import get_vehicle  # noqa: E402  (sysid put control/ on the path)

V = get_vehicle("starling2max")
TRUE = dict(k_f=4.6e-6, k_m=4.0e-7, ixx=2.9e-3, iyy=2.2e-3, izz=4.8e-3, tau=0.035,
            drag=(0.30, 0.20), yaw_bias=0.02)          # yaw_bias: N m, e.g. prop misalignment
RPM = sysid.RPM_TO_RADS


def _doublets(t, t0, n, half, amp):
    """n +amp/-amp doublets starting at t0, each half seconds per sign, with half s rest between."""
    x = np.zeros_like(t)
    for k in range(n):
        s = t0 + k * 3 * half
        x[(t >= s) & (t < s + half)] = amp
        x[(t >= s + half) & (t < s + 2 * half)] = -amp
    return x


def synthetic_topics(seed=0, duration=70.0, maneuvers=True):
    """PX4 topics of a flight: take-off at 2 s, hover, roll / pitch / yaw doublets,
    forward passes at 1.5, 3 and 4.5 m/s, land at duration - 2 s. maneuvers=False:
    hover only, with collective RPM jitter (attitude held, as PX4 does in hover)."""
    rng = np.random.default_rng(seed)
    dt = 0.001
    t = np.arange(0.0, duration, dt)
    m = V.mass
    w_h = math.sqrt(m * 9.81 / (4 * TRUE["k_f"]))
    # differential speed commands (rad/s): roll = left up / right down, etc.
    roll = _doublets(t, 10.0, 6, 0.15, 25.0)
    pitch = _doublets(t, 16.0, 6, 0.15, 25.0)
    yaw = _doublets(t, 22.0, 6, 0.25, 40.0)
    if not maneuvers:
        roll, pitch, yaw = 0 * roll, 0 * pitch, 0 * yaw
    xy = np.array(V.rotor_xy)
    d = np.array(V.rot_dir)
    # yaw trim: CW / CCW pairs offset so the rotors cancel the bias torque at hover
    # (k_m * sum d_i (w_h + d_i e)^2 = k_m * 8 w_h e = -yaw_bias), as the CW props do in the real logs
    trim = -TRUE["yaw_bias"] / (TRUE["k_m"] * 8 * w_h)
    w_cmd = (w_h + d[None] * trim
             + np.sign(xy[:, 1])[None] * roll[:, None]
             - np.sign(xy[:, 0])[None] * pitch[:, None]
             + d[None] * yaw[:, None])
    # broadband jitter so the motor lag is visible
    jitter = np.convolve(rng.normal(0, 40.0, len(t)), np.ones(20) / 20, mode="same")
    w_cmd = w_cmd + jitter[:, None] * (np.array([1.0, 0.8, -0.9, 1.1]) if maneuvers else np.ones(4))[None]
    airborne = (t > 2.0) & (t < duration - 2.0)
    idle = V.w_min
    w_cmd[~airborne] = idle
    w_meas = np.stack([sysid.first_order(w_cmd[:, i], dt, TRUE["tau"]) for i in range(4)], 1)

    # rotational dynamics (FLU) from the measured speeds
    F = TRUE["k_f"] * w_meas ** 2
    tau_b = np.stack([F @ xy[:, 1], F @ -xy[:, 0],
                      TRUE["k_m"] * (w_meas ** 2) @ d + TRUE["yaw_bias"] * airborne], 1)
    I = np.array([TRUE["ixx"], TRUE["iyy"], TRUE["izz"]])
    alpha = tau_b / I
    alpha[~airborne] = 0.0
    rate = np.cumsum(alpha, 0) * dt

    # translation: yaw 30 deg, body-forward passes, drag along body x and y
    psi = math.radians(30.0)
    v_bx = np.interp(t, [0, 30, 33, 40, 43, 50, 53, 60, 63, duration], [0, 0, 1.5, 1.5, 3.0, 3.0, 4.5, 4.5, 0, 0])
    v_bx = v_bx if maneuvers else 0 * v_bx
    v_by = 0.5 * v_bx
    acc_flu = np.stack([-TRUE["drag"][0] * v_bx / m, -TRUE["drag"][1] * v_by / m,
                        F.sum(1) / m], 1)
    acc_flu[~airborne] = [0.0, 0.0, 9.81]
    # FLU body -> NED: velocity in NED from body velocity at heading psi (FRD body yawed psi from north)
    v_frd = np.stack([v_bx, -v_by, np.zeros_like(t)], 1)
    c, s_ = math.cos(psi), math.sin(psi)
    Rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
    v_ned = v_frd @ Rz.T
    q = np.tile([math.cos(psi / 2), 0.0, 0.0, math.sin(psi / 2)], (len(t), 1))

    acc_frd = sysid.frd_to_flu(acc_flu) + rng.normal(0, 0.15, (len(t), 3))
    gyro_frd = sysid.frd_to_flu(rate) + rng.normal(0, 0.01, (len(t), 3))
    rpm_meas = w_meas / RPM + rng.normal(0, 15.0, w_meas.shape)
    u = (w_cmd - V.px4_w_offset) / V.px4_w_scale
    u[~airborne] = np.nan                                  # PX4 logs NaN for disarmed outputs

    def topic(rate_hz, fields):
        k = np.arange(0, len(t), int(round(1.0 / (rate_hz * dt))))
        out = {"timestamp": (t[k] * 1e6 + 1e6).astype(np.int64)}
        for name, arr in fields.items():
            out[name] = arr[k]
        return out

    landed = (~airborne).astype(float)
    return {
        "sensor_combined": topic(250, {f"accelerometer_m_s2[{i}]": acc_frd[:, i] for i in range(3)}),
        "vehicle_angular_velocity": topic(500, {f"xyz[{i}]": gyro_frd[:, i] for i in range(3)}),
        "vehicle_attitude": topic(100, {f"q[{i}]": q[:, i] for i in range(4)}),
        "vehicle_local_position": topic(50, {"vx": v_ned[:, 0], "vy": v_ned[:, 1], "vz": v_ned[:, 2]}),
        "esc_status": topic(200, {f"esc[{i}].esc_rpm": rpm_meas[:, i] for i in range(4)}),
        "actuator_motors": topic(200, {f"control[{i}]": u[:, i] for i in range(4)}),
        "vehicle_land_detected": topic(10, {"landed": landed}),
    }


def test_frames():
    # heading east (yaw 90 deg), flying east: body forward
    q = np.array([[math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]])
    v = sysid.body_velocity_flu(q, np.array([[0.0, 1.0, 0.0]]))
    assert np.allclose(v, [[1.0, 0.0, 0.0]], atol=1e-12), v
    # same heading, flying north: north is to the drone's left -> +y FLU
    v = sysid.body_velocity_flu(q, np.array([[1.0, 0.0, 0.0]]))
    assert np.allclose(v, [[0.0, 1.0, 0.0]], atol=1e-12), v
    assert np.allclose(sysid.frd_to_flu([1, 2, 3]), [1, -2, -3])
    print("  NED/FRD -> body FLU velocity: heading east + flying east = forward, north = left  OK")


def test_wrench_terms_signs():
    """Left rotors faster -> +roll torque (FLU); front faster -> nose up = -pitch; CW faster -> +yaw."""
    base = np.full((1, 4), 500.0)
    left = base + np.array([[0, 10, 10, 0]])          # rotors 1, 2 are on the left (y > 0)
    front = base + np.array([[10, 0, 10, 0]])         # rotors 0, 2 are in front
    cw = base + np.array([[0, 0, 10, 10]])            # rotors 2, 3 spin CW (d = +1)
    _, r, _, _ = sysid.rotor_wrench_terms(left, V)
    _, _, p, _ = sysid.rotor_wrench_terms(front, V)
    _, _, _, y = sysid.rotor_wrench_terms(cw, V)
    assert r > 0 and p < 0 and y > 0, (r, p, y)
    print("  left rotors faster -> +roll, front faster -> -pitch (nose up), CW faster -> +yaw  OK")


def test_motor_lag_alone():
    rng = np.random.default_rng(1)
    dt = 0.005
    t = np.arange(0, 20, dt)
    cmd = 560 + np.convolve(rng.normal(0, 60, len(t)), np.ones(4) / 4, mode="same")
    meas = sysid.first_order(cmd, dt, 0.035) * 0.98 + 5 + rng.normal(0, 1.5, len(t))
    r = sysid.fit_motor_lag(t, cmd, meas, step_s=0.0025)
    assert r["ok"] and abs(r["tau_s"] - 0.035) <= 0.005, r
    flat = sysid.fit_motor_lag(t, np.full_like(t, 560.0), np.full_like(t, 560.0))
    assert not flat["ok"]
    print(f"  first-order lag 35 ms, gain 0.98, offset 5 -> {r['tau_s'] * 1000:.1f} ms; flat RPM -> not identifiable  OK")


def test_identify_synthetic_flight():
    topics = synthetic_topics()
    f = sysid.flight_from_topics(topics, V)
    assert 0.85 < f.airborne.mean() < 0.97, f.airborne.mean()
    res = sysid.identify([f], V, izz=TRUE["izz"])
    print(sysid.format_result(res, V).replace("\n", "\n    "))

    def close(got, want, tol, what):
        assert abs(got - want) <= tol * abs(want), (what, got, want)

    close(res["k_f"]["k_f"], TRUE["k_f"], 0.01, "k_f")
    close(res["tau_motor"]["value"], TRUE["tau"], 0.15, "tau_motor")
    close(res["Ixx"]["value"], TRUE["ixx"], 0.05, "Ixx")
    close(res["Iyy"]["value"], TRUE["iyy"], 0.05, "Iyy")
    close(res["km_over_Izz"]["value"], TRUE["k_m"] / TRUE["izz"], 0.08, "k_m/Izz")
    close(res["km_over_Izz"]["k_m"], TRUE["k_m"], 0.08, "k_m")
    close(res["drag"]["x"]["value"], TRUE["drag"][0], 0.05, "drag x")
    close(res["drag"]["y"]["value"], TRUE["drag"][1], 0.05, "drag y")
    print("  synthetic flight: k_f, motor lag, Ixx, Iyy, k_m, drag x/y recovered  OK")


def test_hover_only_flight_says_not_identifiable():
    """A hover with no doublets or passes (like 2026-11-03) gives k_f and the lag, nothing else."""
    f = sysid.flight_from_topics(synthetic_topics(seed=2, maneuvers=False), V)
    res = sysid.identify([f], V)
    assert res["k_f"]["ok"] and res["tau_motor"]["ok"]
    for name in ("Ixx", "Iyy", "km_over_Izz"):
        assert not res[name]["ok"] and "steps" in res[name]["why"], res[name]
    assert not res["drag"]["x"]["ok"] and not res["drag"]["y"]["ok"]
    print("  hover only: k_f and lag fitted; inertia, yaw and drag reported as not identifiable  OK")


def test_mass_override_scales_kf():
    topics = synthetic_topics()
    f = sysid.flight_from_topics(topics, V)
    heavy = dataclasses.replace(V, mass=V.mass * 1.1)
    a = sysid.fit_kf(f.t, f.acc_frd[:, 2], f.w_meas, V.mass, f.airborne, V)["k_f"]
    b = sysid.fit_kf(f.t, f.acc_frd[:, 2], f.w_meas, heavy.mass, f.airborne, heavy)["k_f"]
    assert abs(b / a - 1.1) < 1e-9
    print("  k_f scales with the mass used (thrust = -m a_z)  OK")


if __name__ == "__main__":
    for n in ("test_frames", "test_wrench_terms_signs", "test_motor_lag_alone",
              "test_identify_synthetic_flight", "test_hover_only_flight_says_not_identifiable",
              "test_mass_override_scales_kf"):
        print(n)
        globals()[n]()
    print("\nAll tests passed.")
