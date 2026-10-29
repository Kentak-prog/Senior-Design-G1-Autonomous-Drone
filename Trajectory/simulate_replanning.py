"""
Closed-loop test of IN-FLIGHT REPLANNING: the camera keeps looking while the
drone flies, the gate map (gate_map.py) refines each gate every time it is
seen, and the replanner (replanner.py) updates the minimum-snap trajectory
whenever the map changes enough to matter.

Unlike simulate_min_snap.py there is NO fallback to true gate poses: a gate
the camera has not seen does not exist for the planner (unless you give it a
prior course map with --prior).

Each control step (dt):
  1. Camera (every 1/--camera-rate s): project every TRUE gate through the
     camera at the drone's TRUE pose (that's what the image would show), keep
     the ones fully in frame and within --max-range, add corner noise, solve
     PnP (vision/gate_pose.estimate_gate_pose), and lift to the world with the
     drone's ESTIMATED pose (that's all the drone knows) -> gate_map.update.
  2. replanner.update(t, estimated position): mark passed gates, replan if needed.
  3. NMPC (control/MPC_Optimizer) tracks replanner.build_reference_horizon,
     the RK4 plant moves the true drone, the EKF estimates it -- same loop as
     control/Drone_Race_Simulation.py.

Perception error model (defaults follow vision/HANDOFF.md's live results):
  --corner-noise-px  corner localization noise
  --range-bias       PnP range scale error (HANDOFF: estimates sit ~1% BEHIND the gate)

Examples:
    python simulate_replanning.py --start 0 0 2.5 --heading 0
    python simulate_replanning.py --start 0 0 2.5 --heading 0 --no-replan        # compare: plan once
    python simulate_replanning.py --start 0 0 2.5 --heading 35 --course-heading 0 # gates start partly out of view
    python simulate_replanning.py --start 0 0 2.5 --heading 0 --prior noisy       # known course map, 0.5 m off
    python simulate_replanning.py --laps 2                                         # full laps of the course
"""

import argparse
import os
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# importing simulate_min_snap also puts vision/, control/ and workspace/ on sys.path
from simulate_min_snap import (body_pose, camera_poses, artificial_gates, report_crossings,
                               draw_gates, set_equal_axes, CAMERA_RESOLUTION, CAMERA_HFOV_DEG)
from camera import CameraIntrinsics, make_transform, invert_transform, in_frame
from gate_pose import estimate_gate_pose, gate_pose_world, project_gate
from drone_constraints import drone_constraints
from MPC_Optimizer import create_drone_mpc_optimizer
from New_Drone_State_Equations import get_drone_dynamics
from Runge_Kutta_4_Discretizer import get_rk4_discretizer
from Kalman_Filter_MPC import DroneCasadiEKF12D

from gate_path import Gate, gates_from_manifest
from gate_map import GateMap, gate_transform
from replanner import Replanner

HERE = os.path.dirname(os.path.abspath(__file__))

# Per-state 1-sigma of the simulated state measurement for --sensor-noise realistic:
# position 5 cm, attitude 1 deg (a flight controller's AHRS), velocity 0.1 m/s,
# body rates 0.02 rad/s. --sensor-noise control keeps Kalman_Filter_MPC's
# R = 0.05*I for every state (sigma 0.22 rad = 13 deg on attitude), which is what
# Drone_Race_Simulation.py uses -- far worse attitude than a real drone, and it
# smears every camera detection lifted into the world by meters at 10-15 m range.
REALISTIC_SENSOR_STD = np.array([0.05] * 3 + [np.radians(1.0)] * 3 + [0.1] * 3 + [0.02] * 3)


# ---------------------------------------------------------------------------
# Camera at an arbitrary drone state
# ---------------------------------------------------------------------------

def T_world_body_from_state(x) -> np.ndarray:
    """12-state -> T_world_body. Same Euler convention as the control model: R = Rz(psi) Ry(theta) Rx(phi)."""
    phi, theta, psi = x[3], x[4], x[5]
    cf, sf, ct, st, cp, sp = np.cos(phi), np.sin(phi), np.cos(theta), np.sin(theta), np.cos(psi), np.sin(psi)
    Rz = np.array([[cp, -sp, 0], [sp, cp, 0], [0, 0, 1]])
    Ry = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    Rx = np.array([[1, 0, 0], [0, cf, -sf], [0, sf, cf]])
    return make_transform(Rz @ Ry @ Rx, x[0:3])


class SyntheticCamera:
    def __init__(self, true_gates, corner_noise_px, range_bias, max_range, seed=0):
        self.true_gates = true_gates
        self.intr = CameraIntrinsics.from_horizontal_fov(*CAMERA_RESOLUTION, CAMERA_HFOV_DEG)
        self.noise = corner_noise_px
        self.range_bias = range_bias
        self.max_range = max_range
        self.rng = np.random.default_rng(seed)

    def detect(self, x_true, x_est):
        """Detections (world frame, lifted with the ESTIMATED pose) + which true gate each came from."""
        _, T_wc_true = camera_poses(T_world_body_from_state(x_true))
        _, T_wc_est = camera_poses(T_world_body_from_state(x_est))
        T_cw_true = invert_transform(T_wc_true)
        out = []
        for g in self.true_gates:
            T_cg = T_cw_true @ g.T_world_gate
            rng = np.linalg.norm(T_cg[:3, 3])
            if T_cg[2, 3] <= 0 or rng > self.max_range:
                continue
            px = project_gate(T_cg, self.intr, g.side)
            if not np.all(in_frame(px, self.intr)):
                continue
            noisy = px + self.rng.normal(0.0, self.noise, px.shape)
            # A gate side assumed (1 + bias) too large scales the PnP range by (1 + bias)
            det = estimate_gate_pose(noisy, self.intr, gate_side=g.side * (1.0 + self.range_bias),
                                     corner_sigma_px=max(self.noise, 1e-3), bootstrap_n=8,
                                     seed=int(self.rng.integers(1 << 30)))
            if det is None:
                continue
            T_wg = gate_pose_world(det, T_wc_est)
            out.append({'center': T_wg[:3, 3], 'normal': T_wg[:3, 2], 'normal_reliable': det.normal_is_reliable,
                        'center_std': det.center_std_m, 'normal_std': det.normal_std_deg,
                        'range': det.range_m, 'side': g.side, 'truth': g.name})
        return out


def make_prior(true_gates, mode, sigma, seed):
    if mode == "none":
        return None
    rng = np.random.default_rng(seed + 1)
    prior = []
    for g in true_gates:
        if mode == "truth":
            prior.append(Gate(g.name, g.T_world_gate.copy(), side=g.side))
            continue
        c = g.center + rng.normal(0.0, sigma, 3)
        yaw = np.arctan2(g.normal[1], g.normal[0]) + np.radians(rng.normal(0.0, 5.0))
        prior.append(Gate(g.name, gate_transform(c, [np.cos(yaw), np.sin(yaw), 0.0]), side=g.side))
    return prior


def map_errors(gate_map, true_gates):
    """For each true gate: distance to the nearest CONFIRMED mapped gate (nan if none within 2 m)."""
    errs = []
    for g in true_gates:
        d = [np.linalg.norm(mg.center - g.center) for mg in gate_map.confirmed()]
        errs.append(min(d) if d and min(d) < 2.0 else np.nan)
    return errs


# ---------------------------------------------------------------------------
# Closed loop (same structure as control/Drone_Race_Simulation.run_race_simulation)
# ---------------------------------------------------------------------------

def run(args, true_gates, start, start_yaw):
    rng = np.random.default_rng(args.seed)
    dt, N, m = args.dt, args.horizon, args.mass

    opti, X, U, X_init, U_prev, X_ref = create_drone_mpc_optimizer(N=N, dt=dt, m=m)
    f_continuous, states, controls = get_drone_dynamics(m)
    f_plant = get_rk4_discretizer(f_continuous, states, controls, dt)
    ekf = DroneCasadiEKF12D(dt=dt, m=m)
    if args.sensor_noise == "realistic":
        ekf.R = np.diag(REALISTIC_SENSOR_STD ** 2)          # the EKF knows its sensors
    meas_std = np.sqrt(np.diag(ekf.R)).reshape(12, 1)

    x_true = np.zeros((12, 1))
    x_true[0:3, 0] = start
    x_true[5, 0] = start_yaw
    ekf.x = x_true.copy()
    u_prev = np.zeros((4, 1))
    hover_thrust = m * 9.81
    X_guess = np.tile(x_true, (1, N + 1))
    U_guess = np.tile(np.array([[hover_thrust], [0.0], [0.0], [0.0]]), (1, N))

    camera = SyntheticCamera(true_gates, args.corner_noise_px, args.range_bias, args.max_range, args.seed)
    gate_map = GateMap(start, prior=make_prior(true_gates, args.prior, args.prior_sigma, args.seed),
                       prior_sigma=args.prior_sigma)
    planner = Replanner(gate_map, start, start_yaw, v_max=args.v_max, a_max=args.a_max, m=m,
                        laps=args.laps, close_loop=not args.open_course, replan=not args.no_replan)
    cam_every = max(1, int(round(1.0 / (args.camera_rate * dt))))

    log = {k: [] for k in ("t", "true", "est", "ref", "u")}
    map_log = {"t": [], "err": [], "n_seen": []}
    end_t = None
    max_time = args.max_time if args.max_time is not None else 15.0 + 30.0 * args.laps
    max_steps = int(max_time / dt)
    for step in range(max_steps):
        t = step * dt

        # 1. camera -> map
        if step % cam_every == 0:
            x_cam_pose = x_true if args.camera_pose == "truth" else ekf.x
            dets = camera.detect(x_true.ravel(), x_cam_pose.ravel())
            gate_map.update(dets, t)
            map_log["t"].append(t)
            map_log["err"].append(map_errors(gate_map, true_gates))
            map_log["n_seen"].append(len(dets))

        # 2. replanning
        planner.update(t, ekf.x[0:3, 0])
        ref_horizon = planner.build_reference_horizon(t, N, dt)

        # 3. NMPC -> plant -> EKF
        if args.no_mpc:
            # Fast mode for testing the planning logic: the drone follows the
            # reference perfectly and the estimate is the truth plus sensor noise.
            u0 = np.array([[hover_thrust], [0.0], [0.0], [0.0]])
            x_true = ref_horizon[:, 1].reshape(12, 1).copy()
            ekf.x = x_true + (rng.normal(0.0, 1.0, size=(12, 1)) * meas_std if not args.no_noise else 0.0)
        else:
            opti.set_value(X_init, ekf.x)
            opti.set_value(U_prev, u_prev)
            opti.set_value(X_ref, ref_horizon)
            opti.set_initial(X, X_guess)
            opti.set_initial(U, U_guess)
            try:
                sol = opti.solve()
                X_sol, U_sol = sol.value(X), sol.value(U)
            except RuntimeError:
                X_sol, U_sol = opti.debug.value(X), opti.debug.value(U)
            u0 = np.array(U_sol[:, 0]).reshape(4, 1)
            X_guess = np.hstack([X_sol[:, 1:], X_sol[:, -1:]])
            U_guess = np.hstack([U_sol[:, 1:], U_sol[:, -1:]])

            x_true = np.array(f_plant(x_true, u0)).reshape(12, 1)
            if not args.no_noise:
                x_true = x_true + rng.normal(0.0, 1e-3, size=(12, 1))
            z_meas = x_true + (rng.normal(0.0, 1.0, size=(12, 1)) * meas_std if not args.no_noise else 0.0)
            ekf.predict(u0)
            ekf.update(z_meas)

        log["t"].append(t)
        log["true"].append(x_true.ravel().copy())
        log["est"].append(ekf.x.ravel().copy())
        log["ref"].append(ref_horizon[:, 0].copy())
        log["u"].append(u0.ravel())

        u_prev = u0

        if step % 100 == 0:
            print(f"t={t:6.2f}s  pos=({x_true[0,0]:6.2f}, {x_true[1,0]:6.2f}, {x_true[2,0]:5.2f})  "
                  f"gates confirmed {len(gate_map.confirmed())}, passed {planner.progress}, "
                  f"plans {len(planner.plans)}")

        if planner.finished(t):
            end_t = end_t if end_t is not None else t
            if t - end_t >= args.tail:
                break

    log = {k: np.array(v) for k, v in log.items()}
    log["true"], log["est"], log["ref"], log["u"] = (log[k].T for k in ("true", "est", "ref", "u"))
    map_log = {k: np.array(v) for k, v in map_log.items()}
    return log, map_log, gate_map, planner


# ---------------------------------------------------------------------------
# Lap report
# ---------------------------------------------------------------------------

def report_laps(planner, true_gates, log, laps, close_loop):
    """
    Count clean passes (through the opening, in the flight direction) of each TRUE
    gate on the flown path and compare with what the route needs: every gate
    once per lap, plus one more for the start/finish gate when the loop is closed.
    Lap times come from the planner's own pass log (start/finish gate to start/finish gate).
    """
    from gate_path import gate_crossings
    n = len(true_gates)
    need = [laps + (1 if (close_loop and i == 0) else 0) for i in range(n)] if close_loop else [1] * n
    print(f"\n[laps] route: {laps} lap(s){' closed through ' + true_gates[0].name if close_loop else ''}")
    ok = True
    for (gate, crossings), k in zip(gate_crossings(log["t"], log["true"][0:3].T, log["true"][6:9].T,
                                                   true_gates), need):
        clean = [c for c in crossings if c['inside'] and c['forward']]
        status = "OK" if len(clean) >= k else "SHORT"
        ok &= len(clean) >= k
        times = ", ".join(f"{c['time']:.1f}s" for c in clean)
        print(f"  {gate.name}: {len(clean)}/{k} clean pass(es)  [{times}]  {status}")
    missed = [p for p in planner.passes if p[3].startswith("MISSED")]
    if missed:
        ok = False
        print(f"  planner marked {len(missed)} pass(es) MISSED")
    for i, lt in enumerate(planner.lap_times(), 1):
        print(f"  lap {i}: {lt:.2f} s")
    return ok


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(true_gates, gate_map, planner, log, map_log, start, start_yaw, out_path):
    fig = plt.figure(figsize=(17, 10))
    replan_t = [p.t0 for p in planner.plans[1:]]
    ref_pos = log["ref"][0:3]
    idx = [int(np.argmin(np.abs(log["t"] - tr))) for tr in replan_t]

    ax = fig.add_subplot(2, 3, (1, 4), projection='3d')
    draw_gates(ax, true_gates, 'darkorange', 0.25, 'True gates')
    mapped = [mg.as_gate() for mg in gate_map.confirmed()]
    if mapped:
        draw_gates(ax, mapped, 'royalblue', 0.08, 'Mapped gates (final)')
    if planner.plans:
        _, p0, *_ = planner.plans[0].traj.sample(0.05)
        ax.plot(*p0.T, color='gray', linewidth=1, alpha=0.6, label='First plan')
    ax.plot(*ref_pos, 'k--', linewidth=1.3, label='Reference actually followed')
    ax.plot(*log["true"][0:3], 'b-', linewidth=1.5, label='Flown (NMPC)')
    if idx:
        ax.scatter(*ref_pos[:, idx], color='red', s=25, label='Replans')
    ax.quiver(*start, np.cos(start_yaw), np.sin(start_yaw), 0, length=1.5, color='green', linewidth=2)
    set_equal_axes(ax, np.vstack([log["true"][0:3].T] + [g.corners() for g in true_gates]))
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("In-flight replanning", fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)

    ax2 = fig.add_subplot(2, 3, 2)
    for g in true_gates:
        c = g.corners()
        ax2.plot(c[[0, 1], 0], c[[0, 1], 1], color='darkorange', linewidth=3)
        ax2.annotate(g.name, g.center[:2], fontsize=8)
    for g in mapped:
        c = g.corners()
        ax2.plot(c[[0, 1], 0], c[[0, 1], 1], color='royalblue', linewidth=1)
    if planner.plans:
        ax2.plot(p0[:, 0], p0[:, 1], color='gray', alpha=0.6, label='First plan')
    ax2.plot(ref_pos[0], ref_pos[1], 'k--', label='Reference followed')
    ax2.plot(log["true"][0], log["true"][1], 'b-', label='Flown')
    if idx:
        ax2.plot(ref_pos[0, idx], ref_pos[1, idx], 'rx', label='Replans')
    ax2.plot(*start[:2], 'go')
    ax2.axis('equal'); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)"); ax2.set_title("Top-down")

    ax3 = fig.add_subplot(2, 3, 3)
    err = map_log["err"]
    for j, g in enumerate(true_gates):
        ax3.plot(map_log["t"], err[:, j], label=g.name)
    for tr in replan_t:
        ax3.axvline(tr, color='red', alpha=0.25)
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Map error (m)"); ax3.grid(alpha=0.3)
    ax3.set_title("Gate estimate error vs truth (red = replans)"); ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(2, 3, 5)
    ax4.plot(log["t"], np.linalg.norm(log["ref"][6:9], axis=0), 'k--', label='Reference')
    ax4.plot(log["t"], np.linalg.norm(log["true"][6:9], axis=0), 'b-', label='Flown')
    for tr in replan_t:
        ax4.axvline(tr, color='red', alpha=0.25)
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Speed (m/s)"); ax4.grid(alpha=0.3)
    ax4.set_title("Speed"); ax4.legend(fontsize=8)

    ax5 = fig.add_subplot(2, 3, 6)
    trk = np.linalg.norm(log["true"][0:3] - ref_pos, axis=0)
    ax5.plot(log["t"], trk, 'r-')
    for tr in replan_t:
        ax5.axvline(tr, color='red', alpha=0.25)
    ax5.set_xlabel("Time (s)"); ax5.set_ylabel("m"); ax5.grid(alpha=0.3)
    ax5.set_title(f"Tracking error (mean {trk.mean():.2f} m, max {trk.max():.2f} m)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved plot to {out_path}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=float, nargs=3, default=[0.0, 0.0, 2.5], metavar=("X", "Y", "Z"))
    ap.add_argument("--heading", type=float, default=0.0, help="drone start heading, deg CCW from world +X")
    ap.add_argument("--course-heading", type=float, default=None,
                    help="heading the artificial course is laid out for (default: --heading). "
                         "Differ from --heading to start with gates out of view")
    ap.add_argument("--gates", default=None, help="test_gates.json manifest (default: artificial course)")
    ap.add_argument("--layout", choices=["approach", "staggered"], default="approach")
    ap.add_argument("--prior", choices=["none", "noisy", "truth"], default="none",
                    help="known course map given to the planner before takeoff")
    ap.add_argument("--prior-sigma", type=float, default=0.5, help="prior position error (m) for --prior noisy")
    ap.add_argument("--corner-noise-px", type=float, default=1.0)
    ap.add_argument("--range-bias", type=float, default=0.01, help="PnP range scale error (0.01 = 1%% long)")
    ap.add_argument("--max-range", type=float, default=20.0, help="max detection range (m)")
    ap.add_argument("--camera-rate", type=float, default=10.0, help="detections per second")
    ap.add_argument("--camera-pose", choices=["ekf", "truth"], default="ekf",
                    help="drone pose used to lift detections into the world")
    ap.add_argument("--sensor-noise", choices=["realistic", "control"], default="realistic",
                    help="state-measurement noise: realistic (1 deg attitude) or control/'s R=0.05*I")
    ap.add_argument("--no-replan", action="store_true", help="plan once from the first view (for comparison)")
    ap.add_argument("--v-max", type=float, default=None, help="default: 80%% of the drone's max (9.6 m/s)")
    ap.add_argument("--a-max", type=float, default=None, help="default: 80%% of the drone's max (12 m/s^2)")
    ap.add_argument("--mass", type=float, default=drone_constraints["mass"])
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--no-mpc", action="store_true",
                    help="fast mode: drone follows the reference perfectly (tests the planning logic only)")
    ap.add_argument("--laps", type=int, default=1,
                    help="laps of the course; each lap ends back through the first gate (start/finish)")
    ap.add_argument("--open-course", action="store_true",
                    help="point-to-point: fly the gates once and stop, no return to the start gate")
    ap.add_argument("--max-time", type=float, default=None, help="sim time limit (default 15 + 30 s per lap)")
    ap.add_argument("--tail", type=float, default=1.0, help="seconds to keep hovering after the finish")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "sim_output"))
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    os.makedirs(args.out_dir, exist_ok=True)
    start = np.array(args.start, dtype=float)
    start_yaw = np.radians(args.heading)

    if args.gates:
        true_gates = gates_from_manifest(args.gates)
    else:
        course_heading = args.heading if args.course_heading is None else args.course_heading
        T_wc_usd, _ = camera_poses(body_pose(start, course_heading))
        true_gates = artificial_gates(T_wc_usd, args.layout)
    print(f"[start] {np.round(start, 2)}, heading {args.heading:.1f} deg; {len(true_gates)} true gates, "
          f"prior map: {args.prior}; replanning {'OFF' if args.no_replan else 'ON'}")
    for g in true_gates:
        print(f"  {g.name}: center {np.round(g.center, 2)}")

    t0 = time.time()
    log, map_log, gate_map, planner = run(args, true_gates, start, start_yaw)
    print(f"\n[sim] {log['t'][-1]:.1f} s simulated in {time.time() - t0:.0f} s wall time; "
          f"{len(planner.plans)} plan(s), {len(gate_map.confirmed())} gate(s) confirmed, "
          f"{sum(mg.dropped for mg in gate_map.gates)} tentative detection track(s) dropped")

    print("\n[map] final gate estimates vs truth:")
    for g, e in zip(true_gates, map_log["err"][-1]):
        print(f"  {g.name}: {'never seen' if np.isnan(e) else f'{e:.3f} m'}")
    trk = np.linalg.norm(log["true"][0:3] - log["ref"][0:3], axis=0)
    print(f"[sim] tracking error: mean {trk.mean():.3f} m, max {trk.max():.3f} m")
    att_err = np.degrees(np.abs(log["est"][3:6] - log["true"][3:6]))
    pos_err = np.linalg.norm(log["est"][0:3] - log["true"][0:3], axis=0)
    print(f"[sim] EKF error (what the camera lift inherits): position mean {pos_err.mean():.3f} m, "
          f"attitude mean {att_err.mean():.2f} deg / max {att_err.max():.2f} deg")

    report_crossings("[sim] FLOWN path vs TRUE gates:", log["t"], log["true"][0:3].T,
                     log["true"][6:9].T, true_gates)
    ok = report_laps(planner, true_gates, log, args.laps, not args.open_course)
    print(f"\n[result] {'ALL GATES PASSED' if ok else 'NOT ALL GATES PASSED'}")

    name = "replanning_sim_noreplan.png" if args.no_replan else "replanning_sim.png"
    plot(true_gates, gate_map, planner, log, map_log, start, start_yaw, os.path.join(args.out_dir, name))
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
