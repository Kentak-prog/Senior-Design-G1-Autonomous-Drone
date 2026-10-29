"""
Calibration steps between our model and the live PX4 + Isaac Sim, run BEFORE
flying a track. Every subcommand works on a vehicle (`--sim` = offline
SimVehicle) and ends with AUTO.LOITER in a finally block.

    python calibrate.py params            read PX4 parameters that matter (no writes)
    python calibrate.py origin            print the MAVLink position in ENU
    python calibrate.py hover  [--alt 3]  thrust step test -> thrust scale + u_hover
    python calibrate.py rates  [--alt 3]  body-rate doublets vs QuadSim

CONVENTIONS: as vehicle.py / frames.py (ENU world, FLU body; PX4 NED / FRD on
    the wire).

params -- reads MC_{ROLL,PITCH,YAW}RATE_MAX (deg/s; for reference only: in
    PX4 v1.18 mavlink_receiver.cpp publishes SET_ATTITUDE_TARGET body rates
    straight to vehicle_rates_setpoint, and mc_rate_control only scales MANUAL
    acro input by its max rates, so our offboard rates of up to 6/6/3 rad/s are
    not clamped -- checked in the cluster's PX4 source 2026-10-20), THR_MDL_FAC (must
    be 0 for ThrustMap), MPC_THR_HOVER, COM_OF_LOSS_T and COM_OBL_RC_ACT
    (offboard-loss failsafe), MC_{ROLL,PITCH,YAW}RATE_P and MC_AIRMODE. A name
    PX4 does not answer for prints "no answer" (e.g. COM_OBL_RC_ACT on some
    versions). It never writes; vehicle.set_param exists but nothing calls it.

origin -- prints the MAVLink-derived ENU position with ZERO origin offset and its
    mean. COMPARE BY HAND with the ROS pose at the same moment:
        ros2 topic echo /drone00/state/pose --once
    then origin_offset = ros_position - printed_mean (x, y, z). Pass it as
    --origin-offset x y z to everything else. The drone must hover (not fly)
    during the measurement. The yaw printed (deg CCW from east) should equal
    the pose topic's yaw too; a mismatch means PX4's heading origin differs from
    Pegasus' world axes, which this package does NOT handle (only a translation).

hover -- holds position at --alt, then switches to CTBR with zero rates and
    steps the normalised thrust u through a list (default the model's hover u
    + [-0.04, -0.02, 0, 0.02, 0.04]), 1.5 s each (sim time), returning to the
    position hold before any step that would start with |dz| > 0.3 m or
    |vz| > 0.3 m/s (a deviation from "only if |dz| > 1 m": leftover vertical
    speed drove the next step into the floor offline). Per step the
    vertical acceleration az is the slope of vz over the step's last 1 s (the
    first 0.5 s are skipped for transients; a step that gets within 0.7 m of the
    z limits is cut short once 0.9 s are in, which keeps a large thrust error
    from aborting the whole calibration). Body-z thrust is
        T_real = m (az + g) / cos(tilt)
    and the fit is T_real = scale * T_model(u) (least squares through the
    origin). OUTPUT: `scale` (feed --thrust-scale) and u_hover for the fitted
    scale, plus the u where az crosses zero by a straight-line fit of az vs u
    (an independent check on u_hover; `--thrust-map linear --u-hover <it>`).
    If the per-step scales disagree strongly the mapping is not a pure scale
    (THR_MDL_FAC != 0, or a different motor model). Aborts to loiter if
    z < 1 m or z > 8 m.

rates -- from a hover hold, per axis a body-rate doublet at hover thrust:
    +r for 0.25 s, -r for 0.25 s, 0 for 0.5 s (13/13/25 ticks; r default 1.5 rad/s roll/pitch,
    1.0 yaw). The same command sequence is replayed in QuadSim from the same
    initial state; the table gives, per axis, the 63% rise time and the peak
    rate (in the direction of r) for the vehicle and for QuadSim. A slower
    rise than QuadSim means the PX4 rate loop is softer than our 25 ms model
    (QuadParams.rate_kp), a peak below r means rate limits or saturation. Rise
    times are resolved to one control tick (20 ms) by interpolation.
"""

import argparse
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))   # rl/

import frames  # noqa: E402
from runner import (Abort, Pacer, goto, ctbr_preroll, add_vehicle_args, make_vehicle)  # noqa: E402
from drone_params import QuadParams  # noqa: E402
from quad_sim import QuadSim  # noqa: E402

PARAM_NAMES = ["MC_ROLLRATE_MAX", "MC_PITCHRATE_MAX", "MC_YAWRATE_MAX", "THR_MDL_FAC",
               "MPC_THR_HOVER", "COM_OF_LOSS_T", "COM_OBL_RC_ACT", "MC_ROLLRATE_P",
               "MC_PITCHRATE_P", "MC_YAWRATE_P", "MC_AIRMODE"]
HOVER_U_OFFSETS = (-0.04, -0.02, 0.0, 0.02, 0.04)
STEP_S = 1.5          # seconds per hover thrust step
FIT_S = 1.0           # the vz slope is fitted over the last FIT_S seconds
Z_MIN, Z_MAX = 1.0, 8.0
SKIP_S = 0.5          # transient skipped at the start of each step
MIN_FIT_S = 0.4       # least data needed after the skip
EARLY_MARGIN = 0.7    # a step ends early within this distance of the z limits


# --------------------------------------------------------------------------
def altitude_guard(s):
    if s.pos[2] < Z_MIN or s.pos[2] > Z_MAX:
        raise Abort("altitude", f"z = {s.pos[2]:.2f} m outside [{Z_MIN}, {Z_MAX}]")


def hold_at(vehicle, params, pos, yaw, guard=altitude_guard):
    return goto(vehicle, params, pos, yaw, tol=0.2, vel_tol=0.2, settle_s=1.0, timeout=60.0,
                on_state=guard)


def cmd_params(vehicle, args, params):
    vals = vehicle.read_params(PARAM_NAMES)
    print(f"{'parameter':<18}value")
    for n in PARAM_NAMES:
        v = vals[n]
        print(f"{n:<18}{'no answer' if v is None else v}")
    return vals


def cmd_origin(vehicle, args, params):
    """Print ENU position (zero offset) every 0.5 s of vehicle time for --seconds."""
    vehicle.origin_offset_enu = np.zeros(3)
    s = vehicle.wait_state()
    t_end = s.t_sim + args.seconds
    pts = []
    t_next = s.t_sim
    while s.t_sim < t_end:
        pts.append(s.pos.copy())
        yaw = math.degrees(float(frames.yaw_of_quat_enu(s.quat)))
        print(f"t={s.t_sim:9.2f}  ENU (zero offset) = {np.round(s.pos, 3)}  yaw = {yaw:7.2f} deg")
        t_next += 0.5
        s = vehicle.step(t_next)
    mean = np.mean(pts, axis=0)
    print(f"mean ENU position (zero offset): {np.round(mean, 3)}")
    print("compare with: ros2 topic echo /drone00/state/pose --once ; "
          "origin_offset = ros_position - mean")
    return mean


# --------------------------------------------------------------------------
def fit_hover(params, us, accs, cos_tilts):
    """Pure fit. us: u per step, accs: measured az [m/s^2], cos_tilts: mean
    cos(tilt) per step. Returns dict(scale, scale_per_step, u_hover_fit,
    u_hover_zero_az). scale = argmin sum (y - scale x)^2 with
    y = m (az + g) / cos(tilt), x = model thrust at u."""
    us, accs, cos_tilts = (np.asarray(a, dtype=np.float64) for a in (us, accs, cos_tilts))
    x = np.array([params.px4_output_to_thrust(u) for u in us])
    y = params.mass * (accs + params.gravity) / cos_tilts
    scale = float((x * y).sum() / (x * x).sum())
    u_fit = params.thrust_to_px4_output(params.hover_thrust() / scale)
    slope, icpt = np.polyfit(us, accs, 1) if len(us) > 1 else (np.nan, np.nan)
    u0 = float(-icpt / slope) if np.isfinite(slope) and slope != 0 else float("nan")
    return dict(scale=scale, scale_per_step=(y / x).tolist(), u_hover_fit=float(u_fit),
                u_hover_zero_az=u0)


def cmd_hover(vehicle, args, params):
    dt = params.control_dt
    s = vehicle.wait_state()
    yaw = float(frames.yaw_of_quat_enu(s.quat))
    hold = np.array([s.pos[0], s.pos[1], args.alt])
    hold_at(vehicle, params, hold, yaw)
    us = args.u if args.u else [params.hover_motor_output() + d for d in HOVER_U_OFFSETS]
    n_ticks = int(round(STEP_S / dt))
    rows, accs, cos_tilts, us_ok = [], [], [], []
    print(f"{'u':>7} {'az [m/s2]':>10} {'cos(tilt)':>10} {'T_real [N]':>11} "
          f"{'T_model [N]':>12} {'scale_i':>8}")
    for k, u in enumerate(us):
        s = vehicle.wait_state()
        # Stricter than "only if |dz| > 1 m": the previous step leaves vertical
        # speed behind, which would carry the next (zero-offset) step downward
        # (seen offline: 3 m -> 0.99 m after three steps). Return to the hold
        # unless altitude and vertical speed are both already near nominal.
        if abs(s.pos[2] - args.alt) > 0.3 or abs(s.vel[2]) > 0.3:
            hold_at(vehicle, params, hold, yaw)
        s = vehicle.wait_state()
        pacer = Pacer(vehicle, dt, s)
        ts, vz, cs = [], [], []
        for _ in range(n_ticks):
            vehicle.send_ctbr_raw(u, np.zeros(3))
            s = pacer.advance(s)
            altitude_guard(s)
            ts.append(s.t_sim)
            vz.append(s.vel[2])
            cs.append(float(frames.quat_to_rotmat(s.quat)[2, 2]))
            rows.append((k, u, s.t_sim, s.pos[2], s.vel[2]))
            # a big thrust error sinks / climbs the drone fast: end the step early,
            # before the hard z limits, once there is enough data to fit
            if (s.pos[2] < Z_MIN + EARLY_MARGIN or s.pos[2] > Z_MAX - EARLY_MARGIN) \
                    and ts[-1] - ts[0] >= SKIP_S + MIN_FIT_S:
                break
        ts, vz, cs = np.array(ts), np.array(vz), np.array(cs)
        m = ts >= max(ts[0] + SKIP_S, ts[-1] - FIT_S)
        if m.sum() < 5:
            print(f"{u:7.4f}  too little data in this step; skipped")
            continue
        us_ok.append(u)
        az = float(np.polyfit(ts[m], vz[m], 1)[0])
        c = float(cs[m].mean())
        accs.append(az)
        cos_tilts.append(c)
        T_real = params.mass * (az + params.gravity) / c
        T_model = params.px4_output_to_thrust(u)
        print(f"{u:7.4f} {az:10.3f} {c:10.4f} {T_real:11.3f} {T_model:12.3f} "
              f"{T_real / T_model:8.4f}" + ("   (cut short)" if len(ts) < n_ticks else ""))
    fit = fit_hover(params, us_ok, accs, cos_tilts)
    print(f"\nthrust scale (real / model, least squares): {fit['scale']:.4f}   "
          f"-> --thrust-scale {fit['scale']:.4f}")
    print(f"u_hover for that scale (model map):           {fit['u_hover_fit']:.4f}   "
          f"(model prediction with scale 1: {params.hover_motor_output():.4f})")
    print(f"u where az = 0 (straight-line fit of az vs u):  {fit['u_hover_zero_az']:.4f}   "
          f"-> --thrust-map linear --u-hover {fit['u_hover_zero_az']:.4f}")
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        np.savetxt(os.path.join(args.out_dir, "hover.csv"), np.array(rows), delimiter=",",
                   header="step,u,t_sim,z,vz", comments="", fmt="%.9g")
    return fit


# --------------------------------------------------------------------------
def doublet_commands(rate, dt, plus=0.25, minus=0.25, zero=0.5):
    """Per-tick body-rate command (+r, -r, 0) for the doublet; (n,) array.
    0.25 s is 12.5 ticks at 50 Hz: rounded UP to 13 (0.26 s) so each half lasts
    at least the nominal time."""
    n1, n2, n3 = (int(math.ceil(x / dt - 1e-9)) for x in (plus, minus, zero))
    return np.concatenate([np.full(n1, rate), np.full(n2, -rate), np.zeros(n3)])


def rise_time_63(t, w, rate):
    """Time at which w (sampled at t, t[0] = command start) first reaches 63.2%
    of `rate`, linearly interpolated; NaN if it never does."""
    target = 0.632 * rate
    sign = 1.0 if rate > 0 else -1.0
    above = np.nonzero(sign * (w - target) >= 0)[0]
    if above.size == 0:
        return float("nan")
    i = above[0]
    if i == 0:
        return float(t[0])
    f = (target - w[i - 1]) / (w[i] - w[i - 1])
    return float(t[i - 1] + f * (t[i] - t[i - 1]))


def peak_along(w, rate):
    return float((np.sign(rate) * w).max())


def simulate_doublet(params, s0, T_hover, cmds_axis, axis):
    """Replay the doublet in QuadSim from state s0 (rl/ conventions). Returns
    omega samples (n+1, 3) including the initial one."""
    sim = QuadSim(1, params)
    sim.reset(pos=torch.tensor(s0.pos[None], dtype=torch.float32),
              vel=torch.tensor(s0.vel[None], dtype=torch.float32),
              quat=torch.tensor(s0.quat[None], dtype=torch.float32),
              omega=torch.tensor(s0.omega[None], dtype=torch.float32))
    out = [sim.omega[0].double().numpy().copy()]
    for r in cmds_axis:
        ref = torch.zeros(1, 3)
        ref[0, axis] = float(r)
        sim.step_ctbr(torch.tensor([T_hover]), ref)
        out.append(sim.omega[0].double().numpy().copy())
    return np.array(out)


def cmd_rates(vehicle, args, params):
    dt = params.control_dt
    s = vehicle.wait_state()
    yaw = float(frames.yaw_of_quat_enu(s.quat))
    hold = np.array([s.pos[0], s.pos[1], args.alt])
    T_hover = params.hover_thrust()
    results = []
    for axis, name, rate in ((0, "roll", args.rate_rp), (1, "pitch", args.rate_rp),
                             (2, "yaw", args.rate_yaw)):
        s0 = hold_at(vehicle, params, hold, yaw)
        # PX4 ignores CTBR for up to 0.5 s after the switch (runner.ctbr_preroll)
        s0 = ctbr_preroll(vehicle, params, s0, hold, yaw, 1.0, on_state=altitude_guard)
        cmds = doublet_commands(rate, dt)
        pacer = Pacer(vehicle, dt, s0)
        s = s0
        meas = [s0.omega.copy()]
        for r in cmds:
            ref = np.zeros(3)
            ref[axis] = r
            vehicle.send_ctbr(T_hover, ref)
            s = pacer.advance(s)
            altitude_guard(s)
            meas.append(s.omega.copy())
        meas = np.array(meas)
        ref_w = simulate_doublet(params, s0, T_hover, cmds, axis)
        t = np.arange(len(meas)) * dt
        results.append((name, rise_time_63(t, meas[:, axis], rate), peak_along(meas[:, axis], rate),
                        rise_time_63(t, ref_w[:, axis], rate), peak_along(ref_w[:, axis], rate)))
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            cmd_col = np.concatenate([[0.0], cmds])
            np.savetxt(os.path.join(args.out_dir, f"rates_{name}.csv"),
                       np.column_stack([t, cmd_col, meas, ref_w]), delimiter=",",
                       header="t,cmd,wx,wy,wz,sim_wx,sim_wy,sim_wz", comments="", fmt="%.9g")
    print(f"\n{'axis':<6}{'rise63 vehicle':>16}{'rise63 QuadSim':>16}{'peak vehicle':>14}"
          f"{'peak QuadSim':>14}   [s, s, rad/s, rad/s]")
    for name, rv, pv, rs, ps in results:
        print(f"{name:<6}{rv:16.3f}{rs:16.3f}{pv:14.3f}{ps:14.3f}")
    return results


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    add_vehicle_args(common)
    common.add_argument("--out-dir", default=None)
    sub.add_parser("params", parents=[common])
    p = sub.add_parser("origin", parents=[common])
    p.add_argument("--seconds", type=float, default=5.0)
    p = sub.add_parser("hover", parents=[common])
    p.add_argument("--alt", type=float, default=3.0)
    p.add_argument("--u", type=float, nargs="+", default=None,
                   help="thrust values to step through (default model hover u +/- 0.04)")
    p = sub.add_parser("rates", parents=[common])
    p.add_argument("--alt", type=float, default=3.0)
    p.add_argument("--rate-rp", type=float, default=1.5)
    p.add_argument("--rate-yaw", type=float, default=1.0)
    a = ap.parse_args(argv)

    params = QuadParams()
    vehicle = make_vehicle(a, params)
    fn = dict(params=cmd_params, origin=cmd_origin, hover=cmd_hover, rates=cmd_rates)[a.cmd]
    try:
        fn(vehicle, a, params)
    except Abort as e:
        print(f"[calibrate] ABORT: {e}")
        return 1
    finally:
        try:
            vehicle.set_loiter()
        except Exception as e:
            print(f"[calibrate] WARNING: set_loiter failed: {e}")
        vehicle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
