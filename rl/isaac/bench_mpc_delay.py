"""
Offline bench for the live MPC oscillation (RESULTS_2026-11-12_max_pink.md):
fly a CTBR controller in QuadSim (Starling 2 Max profile) with
  (a) the rate loop slowed to the 63 % rise times measured live in Isaac
      (77 / 67 / 156 ms; QuadSim nominal 44 / 44 / 76): rate_kp scaled by
      KP_SCALE_ISAAC (fitted by `calib`: 80 / 64 / 152 ms at 4 ms steps), and
  (b) a command delay: the command computed from the state at tick k takes
      effect `delay` s later (physics_dt resolution), as a slow solve plus
      transport does live.

Run from rl/isaac/ or rl/ with DRONE_VEHICLE=starling2max:
    python bench_mpc_delay.py calib
    python bench_mpc_delay.py sweep --controller ctbr  [--tracks ...] [--delays-ms 0 20 40]
    python bench_mpc_delay.py sweep --controller rate  [--backend ipopt|acados] [--told-ms N]

2026-11-12 (README_mpc_rate.md): MpcCtbrController crashes slalom5_pink36
with the Isaac loop at any delay and oval6_pink36 from 30 ms; RateAwareMpc
completes all 21 cases (0-60 ms, uncompensated 40 ms, 2x-fast loop).
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from course import Course, TRACKS_DIR, gate_batch, check_gate   # noqa: E402
from drone_params import QuadParams                             # noqa: E402
from quad_sim import QuadSim                                    # noqa: E402
from rotations import quat_from_euler_zyx                       # noqa: E402
from fly_course import GROUND_Z                                 # noqa: E402

ISAAC_RISE = (0.077, 0.067, 0.156)          # s, 63 % rise measured live (calibrate.py, 2026-11-12)
KP_SCALE_ISAAC = (0.3947, 0.5105, 0.3033)   # rate_kp scale giving 80 / 64 / 152 ms in QuadSim


def rise_time(P, kp_scale, axis, step=1.0, T=0.6):
    """63 % rise time of body rate `axis` for a `step` rad/s setpoint at hover thrust."""
    sim = QuadSim(1, P)
    z = torch.zeros(1)
    sim.reset(pos=torch.tensor([[0.0, 0.0, 3.0]]), quat=quat_from_euler_zyx(z, z, z))
    sim.rate_ctrl.kp = sim.rate_ctrl.kp_nominal * torch.tensor(kp_scale)
    thrust = torch.tensor([P.mass * P.gravity])
    ref = torch.zeros(1, 3); ref[0, axis] = step
    t = 0.0
    while t < T:
        torque = sim.rate_ctrl(sim.omega, ref, P.physics_dt)
        sim.step_rotor_speeds(sim.thrust_to_rotor_speed(sim.mix(thrust, torque)), P.physics_dt)
        t += P.physics_dt
        if float(sim.omega[0, axis]) >= 0.632 * step:
            return t
    return float("nan")


def calib(P):
    """rate_kp scales that reproduce ISAAC_RISE (bisection per axis)."""
    scales = []
    for a in range(3):
        lo, hi = 0.05, 1.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            s = [1.0, 1.0, 1.0]; s[a] = mid
            r = rise_time(P, s, a)
            if math.isnan(r) or r > ISAAC_RISE[a]:
                lo = mid
            else:
                hi = mid
        scales.append(round(hi, 4))
    return scales


def edge_margins(pos, course):
    """Per gate: clearance of the crossing point from the opening's edge [m] (NaN if never crossed)."""
    from runner import edge_margins as em
    return em(pos, course)


def fly(course, P, controller, delay_s=0.0, kp_scale=(1.0, 1.0, 1.0), max_time=45.0):
    """Fly one lap; returns a dict of results and a trace."""
    sim = QuadSim(1, P)
    start, yaw = course.start_pose()
    z = torch.zeros(1)
    sim.reset(pos=torch.as_tensor(start, dtype=torch.float32).reshape(1, 3), quat=quat_from_euler_zyx(z, z, z + yaw))
    sim.rate_ctrl.kp = sim.rate_ctrl.kp_nominal * torch.tensor(kp_scale)
    controller.reset(1, "cpu", torch.float32)
    dt, pdt = P.control_dt, P.physics_dt
    n_sub = int(round(dt / pdt))
    d_sub = int(round(delay_s / pdt))
    gate_idx = torch.zeros(1, dtype=torch.long)
    queue, active = [], (torch.tensor([P.mass * P.gravity]), torch.zeros(1, 3))
    tilt, rates, sp, pos, t_log = [], [], [], [sim.pos[0].numpy().copy()], []
    crashed, kind, k_phys = False, "", 0
    for k in range(int(round(max_time / dt))):
        thrust, cmd = controller.act(sim.pos, sim.vel, sim.quat, gate_idx, dt, omega=sim.omega)
        queue.append((k_phys + d_sub, thrust.reshape(-1).float(), cmd.float().reshape(1, 3)))
        p_prev = sim.pos.clone()
        for _ in range(n_sub):
            while queue and queue[0][0] <= k_phys:
                _, th, rr = queue.pop(0)
                active = (th, rr)
            torque = sim.rate_ctrl(sim.omega, active[1], pdt)
            sim.step_rotor_speeds(sim.thrust_to_rotor_speed(sim.mix(active[0], torque)), pdt)
            k_phys += 1
        q = sim.quat[0].numpy()
        tilt.append(math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2))))))
        rates.append(sim.omega[0].numpy().copy()); sp.append(active[1][0].numpy().copy())
        pos.append(sim.pos[0].numpy().copy()); t_log.append((k + 1) * dt)
        passed, hit = check_gate(p_prev, sim.pos, gate_batch(course, gate_idx, torch.float32), P.collision_radius)
        ground = sim.pos[:, 2] < GROUND_Z
        if bool(hit[0]) or bool(ground[0]) or tilt[-1] > 80.0:
            crashed, kind = True, ("gate" if bool(hit[0]) else "ground" if bool(ground[0]) else "tilt>80")
            break
        gate_idx = gate_idx + passed.long()
        if int(gate_idx[0]) >= len(course):
            break
    sp, rates = np.array(sp), np.array(rates)

    def hf_rms(x, w=10):                       # rms of x minus its 0.2 s moving mean
        if len(x) <= 2 * w:
            return float("nan")
        m = np.convolve(x, np.ones(w) / w, mode="same")
        return float(np.sqrt(np.mean((x - m)[w:-w] ** 2)))

    mg = edge_margins(np.array(pos), course)
    ms = 1e3 * np.array(controller.solve_times) if controller.solve_times else np.array([np.nan])
    return dict(gates=int(gate_idx[0]), n=len(course), done=int(gate_idx[0]) >= len(course), crashed=crashed,
                kind=kind, t=t_log[-1] if t_log else 0.0, max_tilt=max(tilt) if tilt else 0.0,
                osc_dps=math.degrees(max(hf_rms(sp[:, 0]), hf_rms(sp[:, 1]))),
                min_margin=float(np.nanmin(mg)) if np.isfinite(mg).any() else float("nan"),
                solve_ms_median=float(np.median(ms)), solve_ms_p95=float(np.percentile(ms, 95)),
                trace=dict(t=np.array(t_log), tilt=np.array(tilt), rate=rates, sp=sp))


def make(kind, P, course, told_s=0.0, backend="ipopt", v_max=4.0, a_max=6.0):
    if kind == "ctbr":
        from mpc_ctbr import MpcCtbrController
        return MpcCtbrController(P, course, v_max=v_max, a_max=a_max)
    from mpc_rate import RateAwareMpc
    return RateAwareMpc(P, course, v_max=v_max, a_max=a_max, delay_s=told_s, backend=backend)


def row(track, label, r):
    res = "DONE" if r["done"] else (f"crash {r['kind']}" if r["crashed"] else "timeout")
    return (f"{track:16s} {label:34s} | {r['gates']:>2d}/{r['n']:<2d} {res:11s} {r['t']:6.2f}s "
            f"tilt {r['max_tilt']:5.1f}  osc {r['osc_dps']:5.2f} dps  margin {r['min_margin']:+.3f} m  "
            f"solve {r['solve_ms_median']:5.1f}/{r['solve_ms_p95']:5.1f} ms")


if __name__ == "__main__":
    assert os.environ.get("DRONE_VEHICLE") == "starling2max", "run with DRONE_VEHICLE=starling2max"
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["calib", "sweep"])
    ap.add_argument("--controller", choices=["ctbr", "rate"], default="rate")
    ap.add_argument("--backend", default="ipopt")
    ap.add_argument("--tracks", nargs="*", default=["straight3_pink36", "oval6_pink36", "slalom5_pink36"])
    ap.add_argument("--delays-ms", nargs="*", type=float, default=[0, 20, 40, 60])
    ap.add_argument("--told-ms", type=float, default=None, help="delay the rate MPC is told (default: the true one)")
    ap.add_argument("--loop", choices=["isaac", "nominal"], default="isaac")
    ap.add_argument("--v-max", type=float, default=4.0, help="plan speed limit [m/s]")
    ap.add_argument("--a-max", type=float, default=6.0, help="plan acceleration limit [m/s^2]")
    a = ap.parse_args()
    P = QuadParams()
    if a.mode == "calib":
        print("nominal rise ms:", [round(1e3 * rise_time(P, (1, 1, 1), ax), 1) for ax in range(3)])
        s = calib(P)
        print("kp scales:", s, "rise ms:", [round(1e3 * rise_time(P, s, ax), 1) for ax in range(3)])
        sys.exit(0)
    ks = KP_SCALE_ISAAC if a.loop == "isaac" else (1.0, 1.0, 1.0)
    for tr in a.tracks:
        c = Course.load(os.path.join(TRACKS_DIR, tr + ".json"))
        for d in a.delays_ms:
            told = (a.told_ms if a.told_ms is not None else d) / 1e3
            r = fly(c, P, make(a.controller, P, c, told, a.backend, a.v_max, a.a_max), delay_s=d / 1e3, kp_scale=ks)
            print(row(tr, f"{a.controller} v{a.v_max:g} a{a.a_max:g} {a.loop} {d:.0f} ms (told {1e3 * told:.0f})", r), flush=True)
