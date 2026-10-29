"""
Run a controller on a course in QuadSim and summarise the flight.

A "controller" is anything with reset(n_envs, device, dtype) and
act(pos, vel, quat, gate_idx, dt, omega=None) -> (thrust_N (N,), body_rate_ref (N,3)),
i.e. the CTBR interface shared by the PID baseline and the RL policy
(class attribute command = "ctbr", the default). A controller with
command = "wrench" (the MPC) returns (thrust_N, body torque [N m]) instead and
is stepped with QuadSim.step_wrench. The
runner passes omega=sim.omega (body rates) as a keyword because the RL
observation needs it; the PID ignores it.

Each control step the runner checks the CURRENT target gate with
course.check_gate (segment p_prev -> p_now). A pass advances that env's gate
index; a collision or ground hit (z < 0.05 m) ends that env's flight (crashed).
Only the current target gate is tested for collisions, which is enough for
the tracks here (gates are >= 5 m apart). Positions are ENU, z up.

CLI:  python3 fly_course.py --track tracks/oval6.json --controller pid [--plot out.png]
      python3 fly_course.py --track tracks/oval6.json --controller rl --model runs/<name>
      python3 fly_course.py --track tracks/oval6.json --controller mpc [--mpc-vmax 4 --mpc-amax 6]
"""

import argparse
import os
import sys

import numpy as np
import torch

from course import Course, gate_batch, check_gate
from drone_params import QuadParams
from quad_sim import QuadSim
from rotations import quat_from_euler_zyx

GROUND_Z = 0.05


def run_controller(course, controller, params=None, max_time=60.0, n_envs=1,
                   seed=0, start_noise=0.0, dtype=torch.float32, device="cpu"):
    """Returns (log, summary). log arrays are numpy: t (T+1,), pos/vel (T+1,N,3),
    gates_passed (N,), crashed (N,), crash_kind list, completed (N,),
    lap_time (N,) (nan if not completed). summary is for env 0 plus aggregates."""
    params = params if params is not None else QuadParams()
    rng = np.random.default_rng(seed)
    sim = QuadSim(n_envs, params, device=device, dtype=dtype)
    start, yaw = course.start_pose()
    pos0 = torch.as_tensor(start, dtype=dtype, device=device).repeat(n_envs, 1)
    pos0 = pos0 + start_noise * torch.as_tensor(rng.standard_normal((n_envs, 3)),
                                                dtype=dtype, device=device)
    zeros = torch.zeros(n_envs, dtype=dtype, device=device)
    q0 = quat_from_euler_zyx(zeros, zeros, zeros + yaw)
    sim.reset(pos=pos0, quat=q0)
    controller.reset(n_envs, device, dtype)

    dt = params.control_dt
    n_steps = int(round(max_time / dt))
    n_gates = len(course)
    gate_idx = torch.zeros(n_envs, dtype=torch.long, device=device)
    active = torch.ones(n_envs, dtype=torch.bool, device=device)
    crashed = torch.zeros_like(active)
    crash_kind = [""] * n_envs
    finish_step = torch.full((n_envs,), -1, dtype=torch.long, device=device)

    ts, ps, vs = [0.0], [sim.pos.clone()], [sim.vel.clone()]
    for k in range(n_steps):
        thrust, cmd = controller.act(sim.pos, sim.vel, sim.quat, gate_idx, dt, omega=sim.omega)
        p_prev = sim.pos.clone()
        step = sim.step_wrench if getattr(controller, "command", "ctbr") == "wrench" else sim.step_ctbr
        step(thrust, cmd)
        p_now = sim.pos

        passed, hit = check_gate(p_prev, p_now, gate_batch(course, gate_idx, dtype),
                                 params.collision_radius)
        ground = p_now[:, 2] < GROUND_Z
        new_crash = active & (hit | ground)
        for i in torch.nonzero(new_crash).flatten().tolist():
            crash_kind[i] = "ground" if bool(ground[i]) and not bool(hit[i]) else "gate"
        crashed |= new_crash
        gate_idx = gate_idx + (active & passed & ~new_crash).long()
        done_now = active & (gate_idx >= n_gates)
        finish_step = torch.where(done_now, torch.full_like(finish_step, k + 1), finish_step)
        active = active & ~new_crash & ~done_now

        ts.append((k + 1) * dt)
        ps.append(sim.pos.clone())
        vs.append(sim.vel.clone())
        if not active.any():
            break

    t = np.array(ts)
    pos = torch.stack(ps).numpy()
    vel = torch.stack(vs).numpy()
    completed = (gate_idx >= n_gates).numpy()
    lap = np.where(completed, finish_step.numpy() * dt, np.nan)
    log = dict(t=t, pos=pos, vel=vel, gates_passed=gate_idx.numpy(),
               crashed=crashed.numpy(), crash_kind=crash_kind,
               completed=completed, lap_time=lap, course=course)

    speed = np.linalg.norm(vel[:, 0], axis=-1)
    end = int(finish_step[0]) if finish_step[0] >= 0 else len(t) - 1
    summary = dict(completed=bool(completed[0]), lap_time=float(lap[0]),
                   gates_passed=int(gate_idx[0]), n_gates=n_gates,
                   crashed=bool(crashed[0]), crash_kind=crash_kind[0],
                   max_speed=float(speed[:end + 1].max()),
                   mean_speed=float(speed[:end + 1].mean()),
                   completed_frac=float(completed.mean()))
    return log, summary


def format_summary(name, s):
    return (f"{name}: completed={s['completed']} lap_time={s['lap_time']:.2f}s "
            f"gates={s['gates_passed']}/{s['n_gates']} crashed={s['crashed']}"
            f"{' (' + s['crash_kind'] + ')' if s['crashed'] else ''} "
            f"max_speed={s['max_speed']:.2f}m/s mean_speed={s['mean_speed']:.2f}m/s")


def draw_gates(ax, course):
    """Top-down gate openings (magenta bars) with their index labels."""
    for g in course.gates:
        n, half = g.normal, g.side / 2
        right = np.array([-n[1], n[0]])            # gate +X in the ground plane
        a, b = g.center[:2] - right * half, g.center[:2] + right * half
        ax.plot([a[0], b[0]], [a[1], b[1]], "m-", lw=3)
        ax.text(g.x, g.y, g.name.split("_")[-1], fontsize=8)


def plot_log(log, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    course, t = log["course"], log["t"]
    pos, vel = log["pos"][:, 0], log["vel"][:, 0]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(pos[:, 0], pos[:, 1], "b-", lw=1.2)
    draw_gates(ax[0], course)
    ax[0].set_aspect("equal"); ax[0].set_title("top-down (ENU)")
    ax[0].set_xlabel("x east [m]"); ax[0].set_ylabel("y north [m]")
    ax[1].plot(t, pos[:, 2]); ax[1].set_title("altitude"); ax[1].set_xlabel("t [s]")
    ax[1].set_ylabel("z [m]")
    ax[2].plot(t, np.linalg.norm(vel, axis=-1)); ax[2].set_title("speed")
    ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("m/s")
    fig.tight_layout(); fig.savefig(path, dpi=120)


def make_controller(kind, params, course, model=None, mpc_vmax=4.0, mpc_amax=6.0):
    if kind == "pid":
        from pid_controller import PidController
        return PidController(params, course)
    if kind == "rl":
        if model is None:
            raise ValueError("--controller rl needs --model <run dir>")
        from rl_policy import RlPolicyController
        return RlPolicyController(params, course, model_dir=model)
    if kind == "mpc":
        from mpc_controller import MpcController
        return MpcController(params, course, v_max=mpc_vmax, a_max=mpc_amax)
    if kind == "mpc_ctbr":      # same NMPC behind thrust + body rates (PX4 offboard)
        isaac_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "isaac")
        if isaac_dir not in sys.path:
            sys.path.insert(0, isaac_dir)
        from mpc_ctbr import MpcCtbrController
        return MpcCtbrController(params, course, v_max=mpc_vmax, a_max=mpc_amax)
    raise ValueError(f"unknown controller {kind!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True)
    ap.add_argument("--controller", default="pid", choices=["pid", "rl", "mpc", "mpc_ctbr"])
    ap.add_argument("--model", default=None,
                    help="run directory with model.zip + vecnormalize.pkl (for --controller rl)")
    ap.add_argument("--mpc-vmax", type=float, default=4.0, help="MPC planner speed limit [m/s]")
    ap.add_argument("--mpc-amax", type=float, default=6.0, help="MPC planner accel limit [m/s^2]")
    ap.add_argument("--max-time", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", default=None)
    a = ap.parse_args()
    course = Course.load(a.track)
    params = QuadParams()
    log, summ = run_controller(course, make_controller(a.controller, params, course, a.model,
                                                  a.mpc_vmax, a.mpc_amax),
                               params, max_time=a.max_time, seed=a.seed)
    print(format_summary(course.name, summ))
    if a.plot:
        plot_log(log, a.plot)
        print(f"saved {a.plot}")
