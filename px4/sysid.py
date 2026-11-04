"""
Identify a multirotor's rotor and rigid-body parameters from PX4 flight logs.

    python3 px4/sysid.py LOG.ulg [LOG2.ulg ...] [--vehicle starling2max] [--izz 4.2e-3]

Fits, each only when the flight excites it (otherwise reported as "not
identifiable" instead of a number):

    k_f        thrust coefficient: -m * a_z (accelerometer) = k_f * sum(w_i^2),
               0.5 s averages while airborne. Any hover gives it.
    tau_motor  motor lag: first-order time constant from commanded RPM
               (actuator_motors through the ESC mapping) to measured RPM
               (esc_status), fitted per rotor; a pure delay is fitted too and
               reported. Any flight with some RPM activity.
    Ixx, Iyy   roll / pitch inertia: body angular acceleration = tau / I + bias,
               tau from measured RPM, k_f and the rotor positions. Needs roll /
               pitch doublets.
    k_m / Izz  yaw: r_dot = (k_m / Izz) * sum(d_i w_i^2) + bias. Gives k_m only
               together with an Izz (CAD, or --izz). Needs yaw-rate steps.
    drag       body-frame accelerometer = -(d / m) * body velocity + bias, per
               axis. The accelerometer does not see gravity or thrust along
               body x / y, so this needs no thrust model. Needs runs at a
               steady speed of a few m/s.

Geometry (rotor positions, spin directions, ESC mapping, mass) comes from the
vehicle profile (control/vehicles.py). Everything here is in body FLU, like the
rest of the repo; PX4 logs FRD / NED and is converted on load.

Reading .ulg files needs pyulog (pip install pyulog). The fits are plain numpy
and are tested on synthetic flights in px4/test_sysid.py.

CAVEAT esc_status: esc[i] is in ESC channel order. On the Starling 2 Max
(2026-11-03 logs) it matched the PX4 motor order; on another airframe check
the VOXL_ESC_FUNC* parameters and pass --esc-order.
"""

import argparse
import math
import os
import sys

import numpy as np
from scipy.signal import lfilter, savgol_filter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_REPO, "control") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "control"))
from vehicles import active_vehicle, get_vehicle   # noqa: E402

G = 9.81
RPM_TO_RADS = 2.0 * math.pi / 60.0


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------
def rotation_from_quat_wxyz(q):
    """(N,4) unit quaternions w,x,y,z -> (N,3,3) rotation matrices."""
    q = np.asarray(q, dtype=float)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def frd_to_flu(v):
    """FRD vectors (..., 3) -> FLU: (x, -y, -z)."""
    v = np.asarray(v, dtype=float)
    return v * np.array([1.0, -1.0, -1.0])


def body_velocity_flu(q_ned_frd, v_ned):
    """Velocity in body FLU from PX4's attitude (FRD body -> NED) and NED velocity."""
    R = rotation_from_quat_wxyz(q_ned_frd)
    v_frd = np.einsum("nji,nj->ni", R, np.asarray(v_ned, dtype=float))     # R^T v
    return frd_to_flu(v_frd)


# --------------------------------------------------------------------------
# Rotor model
# --------------------------------------------------------------------------
def rotor_wrench_terms(w, vehicle):
    """Rotor speeds (N,4) rad/s -> per-sample regressors, all divided by k_f / k_m:
    S = sum w^2 (thrust / k_f), roll = sum y_i w_i^2, pitch = sum -x_i w_i^2
    (torques / k_f, FLU), yaw = sum d_i w_i^2 (yaw torque / k_m)."""
    w2 = np.asarray(w, dtype=float) ** 2
    xy = np.asarray(vehicle.rotor_xy, dtype=float)
    d = np.asarray(vehicle.rot_dir, dtype=float)
    return w2.sum(1), w2 @ xy[:, 1], w2 @ -xy[:, 0], w2 @ d


def command_to_rads(u, vehicle):
    """PX4 motor output u in [0, 1] -> commanded rotor speed via the ESC mapping."""
    return vehicle.px4_w_offset + vehicle.px4_w_scale * np.asarray(u, dtype=float)


# --------------------------------------------------------------------------
# Fits
# --------------------------------------------------------------------------
def _bin_means(t, cols, mask, width):
    """Means of each column over consecutive `width`-second bins where mask holds throughout."""
    t = np.asarray(t, dtype=float)
    edges = np.arange(t[0], t[-1] + width, width)
    idx = np.digitize(t, edges)
    out = []
    for b in np.unique(idx):
        sel = idx == b
        if sel.sum() >= 3 and mask[sel].all():
            out.append([np.mean(c[sel]) for c in cols])
    return np.array(out).reshape(-1, len(cols))


def fit_kf(t, acc_frd_z, w, mass, airborne, vehicle, bin_s=0.5):
    """k_f from -m a_z = k_f sum w^2 (fit through the origin on bin means)."""
    S = rotor_wrench_terms(w, vehicle)[0]
    thrust = -mass * np.asarray(acc_frd_z, dtype=float)
    B = _bin_means(t, [S, thrust], np.asarray(airborne, bool), bin_s)
    if len(B) < 10:
        return {"ok": False, "why": f"only {len(B)} airborne {bin_s} s bins (need 10)"}
    k_f = float(B[:, 0] @ B[:, 1] / (B[:, 0] @ B[:, 0]))
    resid = B[:, 1] - k_f * B[:, 0]
    return {"ok": True, "k_f": k_f, "n_bins": len(B), "rms_N": float(np.sqrt(np.mean(resid ** 2))),
            "S_range": (float(B[:, 0].min()), float(B[:, 0].max()))}


def longest_run(mask):
    """(start, stop) of the longest run of True in mask (stop exclusive); (0, 0) if none."""
    m = np.concatenate([[0], np.asarray(mask, dtype=np.int8), [0]])
    d = np.diff(m)
    starts, stops = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    if len(starts) == 0:
        return 0, 0
    i = int(np.argmax(stops - starts))
    return int(starts[i]), int(stops[i])


def first_order(u, dt, tau, delay_steps=0):
    """u delayed by delay_steps samples, then through a first-order lag (time constant tau).
    The input is averaged over each sample interval ((u[k-1] + u[k]) / 2): a smooth command
    sampled at dt, not one held constant, which would add half a sample of apparent lag."""
    u = np.asarray(u, dtype=float)
    if delay_steps:
        u = np.concatenate([np.full(delay_steps, u[0]), u[:-delay_steps]])
    if tau <= 0.0:
        return u
    a = 1.0 - math.exp(-dt / tau)
    return lfilter([a / 2, a / 2], [1.0, a - 1.0], u, zi=[(1.0 - a / 2) * u[0]])[0]


def fit_motor_lag(t, cmd, meas, max_tau_s=0.15, max_delay_s=0.06, step_s=None):
    """First-order motor lag of meas behind cmd on a uniform grid t:
    meas ~ gain * lag(cmd) + offset. Returns the best tau with no pure delay
    (what QuadSim's tau_motor models) and the best (delay, tau) pair."""
    t = np.asarray(t, dtype=float)
    dt = float(np.median(np.diff(t)))
    cmd = np.asarray(cmd, dtype=float)
    meas = np.asarray(meas, dtype=float)
    if np.std(cmd) < 1e-6 or np.std(meas) < 1e-6:
        return {"ok": False, "why": "no RPM activity"}
    step = step_s or dt
    taus = np.arange(0.0, max_tau_s + 1e-12, step)
    delays = np.arange(0, int(round(max_delay_s / dt)) + 1)
    err = np.full((len(delays), len(taus)), np.inf)
    for i, dk in enumerate(delays):
        for j, tau in enumerate(taus):
            y = first_order(cmd, dt, tau, dk)
            A = np.stack([y, np.ones_like(y)], 1)
            resid = meas - A @ np.linalg.lstsq(A, meas, rcond=None)[0]
            err[i, j] = float(np.mean(resid ** 2))
    j0 = int(np.argmin(err[0]))
    i1, j1 = np.unravel_index(int(np.argmin(err)), err.shape)
    if j0 == len(taus) - 1:
        return {"ok": False, "why": f"best tau at the {max_tau_s * 1000:.0f} ms search limit"}
    return {"ok": True, "tau_s": float(taus[j0]), "rms_rad_s": math.sqrt(err[0, j0]),
            "delay_s": float(delays[i1] * dt), "tau_with_delay_s": float(taus[j1]),
            "rms_with_delay_rad_s": math.sqrt(err[i1, j1])}


def _smooth(x, dt, window_s):
    n = max(5, int(round(window_s / dt)) | 1)
    return savgol_filter(x, n, 2, axis=0)


def _slope_with_bias(x, y):
    A = np.stack([x, np.ones_like(x)], 1)
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ np.array([a, b])
    r2 = 1.0 - resid.var() / y.var() if y.var() > 0 else 0.0
    return float(a), float(b), float(r2)


def fit_rotational(t, gyro_flu, w, k_f, vehicle, airborne, izz=None,
                   window_s=0.08, min_rate_std=(0.3, 0.3, 0.3)):
    """Ixx, Iyy from alpha = tau / I + bias, and k_m / Izz from r_dot = c sum d_i w_i^2 + bias.
    The angular acceleration is the gyro's finite difference; it and the torque
    regressors then get the same Savitzky-Golay smoother, so the smoothing does not
    bias the slope. Gyroscopic coupling (w x I w) is neglected:
    it is second order at the rates of a doublet."""
    t = np.asarray(t, dtype=float)
    dt = float(np.median(np.diff(t)))
    _, roll, pitch, yaw = rotor_wrench_terms(w, vehicle)
    gyro = np.asarray(gyro_flu, dtype=float)
    gyro_s = _smooth(gyro, dt, window_s)
    alpha = _smooth(np.gradient(gyro, dt, axis=0), dt, window_s)
    regs_s = _smooth(np.stack([roll * k_f, pitch * k_f, yaw], 1), dt, window_s)
    m = np.asarray(airborne, bool)
    out = {}
    for ax, name in enumerate(("Ixx", "Iyy", "km_over_Izz")):
        rate_std = float(np.std(gyro_s[m, ax])) if m.any() else 0.0
        if rate_std < min_rate_std[ax]:
            out[name] = {"ok": False, "why": f"rate std {rate_std:.2f} rad/s < {min_rate_std[ax]} "
                                             f"(fly {'roll' if ax == 0 else 'pitch' if ax == 1 else 'yaw'} steps)"}
            continue
        a, bias, r2 = _slope_with_bias(regs_s[m, ax], alpha[m, ax])
        if a <= 0:
            out[name] = {"ok": False, "why": f"non-positive slope {a:.3g} (check rotor order / signs)"}
            continue
        if ax < 2:
            out[name] = {"ok": True, "value": 1.0 / a, "bias_rad_s2": bias, "r2": r2, "rate_std": rate_std}
        else:
            res = {"ok": True, "value": a, "bias_rad_s2": bias, "r2": r2, "rate_std": rate_std}
            if izz is not None:
                res["k_m"] = a * izz
                res["izz_used"] = izz
            out[name] = res
    return out


def fit_drag(acc_flu_xy, v_flu_xy, mass, airborne, min_speed=1.0):
    """Linear drag per body axis: a = -(d / m) v + bias, on samples where |v_axis| > min_speed."""
    acc = np.asarray(acc_flu_xy, dtype=float)
    v = np.asarray(v_flu_xy, dtype=float)
    m = np.asarray(airborne, bool)
    out = {}
    for ax, name in enumerate(("x", "y")):
        sel = m & (np.abs(v[:, ax]) > min_speed)
        n = int(sel.sum())
        if n < 50 or np.ptp(v[sel, ax]) < 1.0:
            out[name] = {"ok": False, "why": f"{n} samples above {min_speed} m/s along body {name} "
                                             f"(fly steady passes at 2-3 speeds)"}
            continue
        a, bias, r2 = _slope_with_bias(v[sel, ax], acc[sel, ax])
        out[name] = {"ok": True, "value": -a * mass, "bias_m_s2": bias, "r2": r2, "n": n}
    return out


def hover_rpm_split(w, airborne, vehicle):
    """Mean airborne RPM per rotor minus the 4-rotor mean (yaw trim / CoM offset)."""
    m = np.asarray(airborne, bool)
    rpm = np.asarray(w, dtype=float)[m] / RPM_TO_RADS
    mean = rpm.mean(0)
    return {"mean_rpm": float(mean.mean()), "per_rotor_minus_mean": (mean - mean.mean()).tolist()}


# --------------------------------------------------------------------------
# Logs
# --------------------------------------------------------------------------
class Flight:
    """One log resampled onto a uniform grid. Arrays: t (s), acc_frd (N,3), gyro_flu (N,3),
    q (N,4, FRD->NED, wxyz), v_ned (N,3), w_meas (N,4 rad/s), w_cmd (N,4 rad/s), airborne (N,)."""

    def __init__(self, **arrays):
        self.__dict__.update(arrays)


def _field(topic, name):
    return np.asarray(topic[name], dtype=float)


def flight_from_topics(topics, vehicle, rate_hz=200.0, esc_order=(0, 1, 2, 3)):
    """topics: {topic name: {field name: array}} with 'timestamp' in microseconds, using
    pyulog's flattened field names ('q[0]', 'esc[2].esc_rpm', ...)."""
    def tt(name):
        return _field(topics[name], "timestamp") * 1e-6

    needed = ["sensor_combined", "vehicle_angular_velocity", "vehicle_attitude",
              "vehicle_local_position", "esc_status", "actuator_motors", "vehicle_land_detected"]
    missing = [n for n in needed if n not in topics]
    if missing:
        raise KeyError(f"log is missing topics {missing}")
    t0 = max(tt(n)[0] for n in needed)
    t1 = min(tt(n)[-1] for n in needed)
    t = np.arange(t0, t1, 1.0 / rate_hz)

    def interp(name, fields):
        src = tt(name)
        return np.stack([np.interp(t, src, _field(topics[name], f)) for f in fields], 1)

    acc = interp("sensor_combined", [f"accelerometer_m_s2[{i}]" for i in range(3)])
    gyro = frd_to_flu(interp("vehicle_angular_velocity", [f"xyz[{i}]" for i in range(3)]))
    q = interp("vehicle_attitude", [f"q[{i}]" for i in range(4)])
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    v_ned = interp("vehicle_local_position", ["vx", "vy", "vz"])
    w_meas = interp("esc_status", [f"esc[{i}].esc_rpm" for i in esc_order]) * RPM_TO_RADS
    u = interp("actuator_motors", [f"control[{i}]" for i in range(4)])
    w_cmd = command_to_rads(np.nan_to_num(u, nan=0.0), vehicle)
    # landed is a step signal: hold the previous value instead of interpolating
    src = tt("vehicle_land_detected")
    landed = _field(topics["vehicle_land_detected"], "landed")
    k = np.clip(np.searchsorted(src, t, side="right") - 1, 0, len(src) - 1)
    airborne = (landed[k] < 0.5) & (w_meas.min(1) > vehicle.w_min * 1.2)
    return Flight(t=t, acc_frd=acc, gyro_flu=gyro, q=q, v_ned=v_ned, w_meas=w_meas, w_cmd=w_cmd,
                  airborne=airborne)


def load_ulog(path, vehicle, **kw):
    try:
        from pyulog import ULog
    except ImportError:
        raise SystemExit("reading .ulg needs pyulog: pip install pyulog")
    ulog = ULog(path)
    topics = {}
    for d in ulog.data_list:
        if d.multi_id == 0:
            topics[d.name] = d.data
    return flight_from_topics(topics, vehicle, **kw)


def concat(flights):
    """Several flights as one, with a time gap so bins and filters never straddle two logs."""
    out, offset = {}, 0.0
    for f in flights:
        for k, v in f.__dict__.items():
            v = v - v[0] + offset if k == "t" else v
            out.setdefault(k, []).append(v)
        offset = out["t"][-1][-1] + 10.0
    return Flight(**{k: np.concatenate(v) for k, v in out.items()})


def identify(flights, vehicle, izz=None):
    """Every fit on a list of Flight. Rotational and lag fits run per flight (they
    differentiate / correlate on a uniform grid) and are averaged."""
    f_all = concat(flights)
    res = {"vehicle": vehicle.name, "mass": vehicle.mass}
    res["k_f"] = fit_kf(f_all.t, f_all.acc_frd[:, 2], f_all.w_meas, vehicle.mass, f_all.airborne, vehicle)
    k_f = res["k_f"]["k_f"] if res["k_f"]["ok"] else vehicle.k_f
    res["hover"] = hover_rpm_split(f_all.w_meas, f_all.airborne, vehicle)

    lags, delays = [], []
    for f in flights:
        a, b = longest_run(f.airborne)
        if b - a < 2.0 / np.median(np.diff(f.t)):
            continue
        sl = slice(a, min(b, a + int(60.0 / np.median(np.diff(f.t)))))   # 60 s is plenty
        for i in range(4):
            r = fit_motor_lag(f.t[sl], f.w_cmd[sl, i], f.w_meas[sl, i], step_s=0.0025)
            if r["ok"]:
                lags.append(r["tau_s"])
                delays.append((r["delay_s"], r["tau_with_delay_s"]))
    res["tau_motor"] = ({"ok": True, "value": float(np.median(lags)), "per_rotor_s": lags,
                         "delay_tau_s": delays} if lags
                        else {"ok": False, "why": "no airborne stretch of 2 s with RPM activity"})

    rot = [fit_rotational(f.t, f.gyro_flu, f.w_meas, k_f, vehicle, f.airborne, izz=izz) for f in flights]
    for name in ("Ixx", "Iyy", "km_over_Izz"):
        ok = [r[name] for r in rot if r[name]["ok"]]
        if ok:
            res[name] = dict(ok[0], value=float(np.mean([r["value"] for r in ok])), n_flights=len(ok))
            if "k_m" in ok[0]:
                res[name]["k_m"] = res[name]["value"] * izz
        else:
            res[name] = rot[0][name]

    acc_flu = frd_to_flu(f_all.acc_frd)
    v_flu = body_velocity_flu(f_all.q, f_all.v_ned)
    res["drag"] = fit_drag(acc_flu[:, :2], v_flu[:, :2], vehicle.mass, f_all.airborne)
    return res


def format_result(res, vehicle):
    def line(label, r, fmt, prior):
        if r["ok"]:
            return f"  {label:<10} {fmt(r)}   (profile {prior})"
        return f"  {label:<10} not identifiable: {r['why']}   (profile {prior})"

    out = [f"{res['vehicle']}, {res['mass']} kg"]
    out.append(line("k_f", res["k_f"], lambda r: f"{r['k_f']:.4g} N/(rad/s)^2 from {r['n_bins']} bins, "
                                                 f"rms {r['rms_N']:.3f} N", f"{vehicle.k_f:.4g}"))
    h = res["hover"]
    out.append(f"  hover      {h['mean_rpm']:.0f} RPM mean; per rotor minus mean "
               f"{', '.join(f'{d:+.0f}' for d in h['per_rotor_minus_mean'])}")
    out.append(line("tau_motor", res["tau_motor"],
                    lambda r: f"{r['value'] * 1000:.0f} ms first-order (median of {len(r['per_rotor_s'])} "
                              f"rotors); with a pure delay: {np.median([d for d, _ in r['delay_tau_s']]) * 1000:.0f} ms "
                              f"+ tau {np.median([x for _, x in r['delay_tau_s']]) * 1000:.0f} ms",
                    f"{vehicle.tau_motor * 1000:.0f} ms"))
    for name, idx in (("Ixx", 0), ("Iyy", 1)):
        out.append(line(name, res[name], lambda r: f"{r['value']:.3e} kg m^2 (R^2 {r['r2']:.2f})",
                        f"{vehicle.inertia_diag[idx]:.3e}"))
    km = res["km_over_Izz"]
    fmt = (lambda r: f"k_m/Izz {r['value']:.3e}" + (f" -> k_m {r['k_m']:.3e} with Izz {r['izz_used']:.3e}"
                                                   if "k_m" in r else " (pass --izz for k_m)"))
    out.append(line("yaw", km, fmt, f"k_m {vehicle.k_m:.3e}, Izz {vehicle.inertia_diag[2]:.3e}"))
    for ax, i in (("x", 0), ("y", 1)):
        out.append(line(f"drag {ax}", res["drag"][ax], lambda r: f"{r['value']:.3f} N/(m/s) (R^2 {r['r2']:.2f})",
                        f"{vehicle.drag_coeff[i]:.3f}"))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help=".ulg files")
    ap.add_argument("--vehicle", default=None, help="vehicle profile (default: $DRONE_VEHICLE)")
    ap.add_argument("--mass", type=float, default=None, help="override the profile mass (kg)")
    ap.add_argument("--izz", type=float, default=None, help="Izz (kg m^2) to turn k_m/Izz into k_m")
    ap.add_argument("--esc-order", default="0,1,2,3", help="esc_status index for PX4 motors 0..3")
    args = ap.parse_args(argv)
    vehicle = get_vehicle(args.vehicle) if args.vehicle else active_vehicle()
    if args.mass is not None:
        import dataclasses
        vehicle = dataclasses.replace(vehicle, mass=args.mass)
    order = tuple(int(s) for s in args.esc_order.split(","))
    flights = [load_ulog(p, vehicle, esc_order=order) for p in args.logs]
    print(format_result(identify(flights, vehicle, izz=args.izz), vehicle))


if __name__ == "__main__":
    main()
