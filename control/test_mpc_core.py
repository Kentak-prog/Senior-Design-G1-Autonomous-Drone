"""Self-tests for control/mpc_core.py (the MPC behind ros2_nodes/mpc_node.py).
Run: python3 test_mpc_core.py

No ROS. The closed-loop test flies the 3-gate Isaac course end to end through
the same objects the ROS nodes use: min_snap_planner.plan_course ->
trajectory_points -> (MinSnapTrajectory-like message) -> TrajectoryBuffer ->
MpcCore at 50 Hz -> the control model's RK4 plant.

The PX4 tests fly MpcCore the way Isaac does: thrust + an ATTITUDE setpoint
into a plant whose attitude lags the setpoint (px4_attitude_plant.py), with
the attitude taken attitude_lead_s ahead (mpc_core.py "ATTITUDE LEAD").
"""

import math
import os
import sys
import time
from types import SimpleNamespace as NS

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(REPO, "Trajectory"), os.path.join(REPO, "vision")]

from mpc_core import TrajectoryBuffer, MpcCore, hover_reference, POINT_FIELDS   # noqa: E402
from drone_constraints import drone_constraints                             # noqa: E402
from px4_attitude_plant import Px4AttitudePlant                             # noqa: E402

M = drone_constraints["mass"]

# The 3-gate course (Trajectory/isaac_results/3_gates/course/test_gates.json) and its hover pose.
GATES_3 = [(5.1644, 0.5666, 3.3667, 7.909), (8.6018, -3.2453, 3.6556, -22.091), (13.0455, -2.0747, 3.7681, -7.091)]
START = np.array([-0.0469, -0.0367, 2.5484])
START_YAW = math.radians(-7.24)


def _stamp(t):
    return NS(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)))


# --------------------------------------------------------------------------- TrajectoryBuffer

def _fake_msg(t0, ts, states, thrust):
    pts = []
    for k, t in enumerate(ts):
        pt = NS(t=float(t), thrust=float(thrust[k]))
        for i, f in enumerate(POINT_FIELDS):
            setattr(pt, f, float(states[i, k]))
        pts.append(pt)
    return NS(header=NS(stamp=_stamp(t0)), points=pts)


def test_trajectory_buffer_time_base_and_yaw():
    ts = np.linspace(0, 2, 101)
    st = np.zeros((12, len(ts)))
    st[0] = ts                    # x = t
    st[5] = np.linspace(3.0, 3.4, len(ts))
    st[5] = (st[5] + np.pi) % (2 * np.pi) - np.pi          # wraps past +pi partway
    buf = TrajectoryBuffer.from_msg(_fake_msg(100.0, ts, st, np.full(len(ts), M * 9.81)))
    h = buf.horizon(100.5, 10, 0.02)                      # 0.5 s into the plan
    assert np.allclose(h[0], 0.5 + 0.02 * np.arange(11))  # advances with time (old node: always t = 0)
    assert np.allclose(buf.horizon(99.0, 10, 0.02)[0], 0.0) and np.allclose(buf.horizon(105.0, 10, 0.02)[0], 2.0)
    assert np.all(np.diff(buf.horizon(101.0, 10, 0.02)[5]) > 0)          # continuous through +-pi
    near = buf.horizon(101.0, 10, 0.02, yaw_now=3.2 - 2 * np.pi)
    assert abs(near[5, 0] - (3.2 - 2 * np.pi)) < 0.2
    assert buf.ended(102.5) and not buf.ended(101.9)
    print("  trajectory: evaluated at now - stamp, clamped at both ends, yaw continuous and next to the drone  OK")


# --------------------------------------------------------------------------- MpcCore

def test_mpc_holds_hover():
    core = MpcCore(N=10, dt=0.02, m=M)
    x = np.zeros((12, 1)); x[0:3, 0] = [1.0, 2.0, 3.0]; x[5, 0] = 0.3
    out = core.step(x, hover_reference([1.0, 2.0, 3.0], 0.3, 10))
    assert out["ok"] and abs(out["u0"][0, 0] - M * 9.81) < 0.05 and np.allclose(out["u0"][1:, 0], 0, atol=1e-3)
    print(f"  hover: thrust {out['u0'][0, 0]:.2f} N = m g at {M} kg (old node: zero-thrust start)  OK")


def test_closed_loop_three_gate_course():
    import min_snap_planner as msn
    from gate_path import gate_crossings
    from eval_isaac_gates import T_world_gate_from_spec
    from New_Drone_State_Equations import get_drone_dynamics
    from Runge_Kutta_4_Discretizer import get_rk4_discretizer

    gates = [msn.Gate(f"gate_{i}", T_world_gate_from_spec(x, y, z, yaw)) for i, (x, y, z, yaw) in enumerate(GATES_3)]
    traj, wps, ref = msn.plan_course(gates, START, START_YAW, v_max=2.0, a_max=2.0)
    ts, states, thrust = msn.trajectory_points(ref)
    t0 = 50.0
    buf = TrajectoryBuffer.from_msg(_fake_msg(t0, ts, states, thrust))

    dt, N = 0.02, 10
    core = MpcCore(N=N, dt=dt, m=M)
    f, xs, us = get_drone_dynamics(M)
    plant = get_rk4_discretizer(f, xs, us, dt)
    x = states[:, 0:1].copy()
    log_t, log_x, err, solve = [], [], [], []
    for k in range(int((buf.duration + 1.0) / dt)):
        now = t0 + k * dt
        h = buf.horizon(now, N, dt, yaw_now=float(x[5, 0]))
        out = core.step(x, h)
        assert out["ok"], f"solve failed at t={now - t0:.2f}"
        solve.append(out["solve_ms"])
        err.append(float(np.linalg.norm(x[0:3, 0] - h[0:3, 0])))
        x = np.array(plant(x, out["u0"])).reshape(12, 1)
        log_t.append(now - t0); log_x.append(x[:, 0].copy())
    P = np.array(log_x)
    results = gate_crossings(np.array(log_t), P[:, 0:3], P[:, 6:9], gates)
    passed = [any(c["inside"] and c["forward"] for c in cs) for _, cs in results]
    assert all(passed), passed
    med = float(np.median(solve))
    assert max(err) < 0.3, max(err)
    assert med < 20.0, med
    print(f"  3-gate course, {buf.duration:.1f} s: all gates passed, tracking error max {max(err):.3f} m, "
          f"solve median {med:.1f} ms (50 Hz needs < 20)  OK")


def test_px4_plant_rates_are_body_rates():
    # the VIO integrates the gyro: integrating the plant's (p, q, r) must give back its attitude
    from px4_attitude_plant import _rot
    plant, dt = Px4AttitudePlant(M, tau=0.08), 0.004
    x, R = np.zeros((12, 1)), np.eye(3)
    for k in range(2500):                                    # 10 s of tilted yawing
        t = k * dt
        x = plant.step(x, M * 9.81, [0.25 * math.sin(1.3 * t), 0.2 * math.cos(0.9 * t), 1.5 * math.sin(0.5 * t)], dt)
        w = x[9:12, 0]
        W = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        U, _, Vt = np.linalg.svd(R @ (np.eye(3) + W * dt + 0.5 * W @ W * dt * dt))
        R = U @ Vt
    err = math.degrees(math.acos(min(1.0, max(-1.0, (np.trace(R.T @ _rot(*x[3:6, 0])) - 1) / 2))))
    assert err < 0.05, err
    print(f"  PX4-like plant: integrated gyro vs attitude after 10 s of tilted yawing {err:.3f} deg  OK")


def _hold_through_px4(lead_s, tau, seconds=10.0, dt=0.02):
    """Hover hold from 0.3 m off in x through the PX4-like plant; returns (|x error| per step, max tilt deg)."""
    core = MpcCore(N=10, dt=dt, m=M, attitude_lead_s=lead_s)
    plant = Px4AttitudePlant(M, tau=tau)
    x = np.zeros((12, 1)); x[0, 0] = 0.30; x[2, 0] = 2.5
    ref = hover_reference(np.array([0.0, 0.0, 2.5]), 0.0, 10)
    err, tilt = [], 0.0
    for _ in range(int(seconds / dt)):
        out = core.step(x, ref)
        for _ in range(4):                                   # the plant at 200 Hz
            x = plant.step(x, out["u0"][0, 0], out["x_cmd"][3:6, 0], dt / 4)
        err.append(abs(x[0, 0])); tilt = max(tilt, math.degrees(abs(x[4, 0])))
    return np.array(err), tilt


def test_hover_hold_through_px4_attitude_loop():
    # What Isaac showed (2026-10-29): sending the attitude one step (0.02 s) ahead leaves PX4 holding
    # its current attitude, and a 0.3 m offset is never corrected. 0.1 s ahead corrects it.
    for tau in (0.05, 0.10):
        e, tilt = _hold_through_px4(0.10, tau)
        assert e[149] < 0.06 and e[-1] < 0.01 and tilt < 5.0, (tau, e[149], e[-1], tilt)
        e1, _ = _hold_through_px4(0.02, tau)
        assert e1[-1] > 0.3, (tau, e1[-1])
        print(f"  PX4 lag {tau:.2f} s, 0.3 m off: lead 0.1 s -> {e[149]:.3f} m at 3 s, {e[-1]:.3f} m at 10 s "
              f"(tilt <= {tilt:.1f} deg); one-step lead -> {e1[-1]:.2f} m at 10 s  OK")


def test_closed_loop_course_through_px4_attitude_loop():
    import min_snap_planner as msn
    from gate_path import gate_crossings
    from eval_isaac_gates import T_world_gate_from_spec

    gates = [msn.Gate(f"gate_{i}", T_world_gate_from_spec(x, y, z, yaw)) for i, (x, y, z, yaw) in enumerate(GATES_3)]
    traj, wps, ref = msn.plan_course(gates, START, START_YAW, v_max=2.0, a_max=2.0, yaw_mode="next_gate")
    ts, states, thrust = msn.trajectory_points(ref)
    t0, dt, N = 50.0, 0.02, 10
    buf = TrajectoryBuffer.from_msg(_fake_msg(t0, ts, states, thrust))
    core = MpcCore(N=N, dt=dt, m=M, attitude_lead_s=0.1)
    plant = Px4AttitudePlant(M, tau=0.08)
    x = states[:, 0:1].copy()
    log_t, log_x, err = [], [], []
    for k in range(int((buf.duration + 1.0) / dt)):
        now = t0 + k * dt
        h = buf.horizon(now, N, dt, yaw_now=float(x[5, 0]))
        out = core.step(x, h)
        assert out["ok"], f"solve failed at t={now - t0:.2f}"
        err.append(float(np.linalg.norm(x[0:3, 0] - h[0:3, 0])))
        for _ in range(4):
            x = plant.step(x, out["u0"][0, 0], out["x_cmd"][3:6, 0], dt / 4)
            log_t.append(now - t0); log_x.append(x[:, 0].copy())
    P = np.array(log_x)
    results = gate_crossings(np.array(log_t), P[:, 0:3], P[:, 6:9], gates)
    passed = [any(c["inside"] and c["forward"] for c in cs) for _, cs in results]
    assert all(passed), passed
    # the lag the NMPC does not model costs tracking: about 0.3 m mean / 0.55 m max (lead swept on
    # 2026-10-29: 0.1 s is best or within 0.05 m of best for a PX4 lag of 0.05-0.12 s)
    assert np.mean(err) < 0.4 and max(err) < 0.7, (np.mean(err), max(err))
    print(f"  3-gate course at 2/2 through the PX4-like plant (lag 0.08 s, lead 0.1 s): all gates passed, "
          f"tracking error max {max(err):.3f} m, mean {np.mean(err):.3f} m  OK")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    t_all = time.time()
    for t in tests:
        print(t.__name__)
        t()
    print(f"\nAll tests passed. ({len(tests)})  [{time.time() - t_all:.0f} s]")
