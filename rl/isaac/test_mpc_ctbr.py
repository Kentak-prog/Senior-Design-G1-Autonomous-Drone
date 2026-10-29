"""Self-tests for MpcCtbrController. Run: python test_mpc_ctbr.py (from rl/isaac/ or rl/).
Plain functions, no pytest, no Isaac. ~1.5 min (flies the real NMPC three times in QuadSim,
once through SimVehicle, plus two mid-flight re-plan flights)."""

import copy
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, RL)

import numpy as np
import torch

from course import Course, TRACKS_DIR, gate_batch, check_gate
from drone_params import QuadParams
from fly_course import run_controller, make_controller
from quad_sim import QuadSim
from rotations import quat_from_euler_zyx
from mpc_ctbr import MpcCtbrController
from runner import run_in_vehicle, edge_margins
from vehicle import SimVehicle

P = QuadParams()
# Reference laps of this controller in QuadSim (v4 a6, optimize_times=False). Re-measured
# 2026-10-29 after merging origin/main (MPC model now has drag and tighter limits): was
# 9.56 / 22.68 / 28.46 s, the wrench-MPC numbers in HANDOFF and the 2026-10-24 Isaac runs.
# With the planner's optimize_times=True the laps are 5.72 / 13.88 / 19.38 s, but one plan
# takes 12 s+ (see MpcController.optimize_times).
WRENCH_LAP = {"straight3": 9.44, "oval6": 22.60, "slalom5": 28.40}
MEAN_MS_BOUND, P95_MS_BOUND = 20.0, 40.0
_ALL_MS = []


def _track(name):
    return Course.load(os.path.join(TRACKS_DIR, f"{name}.json"))


def test_quadsim_tracks_and_timing():
    for name, ref in WRENCH_LAP.items():
        c = _track(name)
        ctl = MpcCtbrController(P, c)
        _, s = run_controller(c, ctl, P)
        assert s["completed"] and s["gates_passed"] == s["n_gates"] and not s["crashed"], (name, s)
        ms = 1e3 * np.array(ctl.solve_times)
        _ALL_MS.extend(ms)
        print(f"  {name}: lap {s['lap_time']:.2f} s (wrench MPC {ref:.2f} s), solve mean "
              f"{ms.mean():.1f} / p95 {np.percentile(ms, 95):.1f} / max {ms.max():.1f} ms, "
              f"failures {ctl.n_failures}")
        # regression: laps within 1 s of the reference above
        assert abs(s["lap_time"] - ref) <= 1.0, (name, s["lap_time"], ref)
        assert ctl.n_failures == 0
        # Bounds: the control period is 20 ms, so a mean over 20 ms cannot be real time;
        # p95 < 40 ms (2 ticks) leaves room for a loaded machine (measured ~4-8 ms).
        assert ms.mean() < MEAN_MS_BOUND and np.percentile(ms, 95) < P95_MS_BOUND, name


def test_output_is_ctbr_and_clamped():
    c = _track("straight3")
    ctl = MpcCtbrController(P, c)
    assert ctl.command == "ctbr"
    ctl.reset(1)
    start, yaw = c.start_pose()
    pos = torch.tensor(np.array([start]), dtype=torch.float32)
    q = quat_from_euler_zyx(torch.zeros(1), torch.zeros(1), torch.full((1,), yaw))
    thrust, rates = ctl.act(pos, torch.zeros(1, 3), q, torch.zeros(1, dtype=torch.long), 0.02,
                            omega=torch.zeros(1, 3))
    assert thrust.shape == (1,) and rates.shape == (1, 3)
    assert 0.0 <= float(thrust) <= P.max_total_thrust()
    assert (rates.abs() <= torch.tensor(P.max_rate)).all()
    # at rest at the start the plan asks for thrust ~ weight
    assert abs(float(thrust) - P.mass * P.gravity) < 0.3 * P.mass * P.gravity, thrust
    print(f"  thrust {float(thrust):.2f} N (weight {P.mass * P.gravity:.2f}), rates {rates[0].tolist()}  OK")


def test_budget_fallback():
    """A 0.01 ms budget forces every solve after the first to fall back to the shifted
    previous solution: finite outputs, counted, no exception."""
    c = _track("straight3")
    ctl = MpcCtbrController(P, c, max_solve_ms=0.01)
    ctl.reset(1)
    start, yaw = c.start_pose()
    pos = torch.tensor(np.array([start]), dtype=torch.float32)
    q = quat_from_euler_zyx(torch.zeros(1), torch.zeros(1), torch.full((1,), yaw))
    for _ in range(6):
        T, w = ctl.act(pos, torch.zeros(1, 3), q, torch.zeros(1, dtype=torch.long), 0.02,
                       omega=torch.zeros(1, 3))
        assert torch.isfinite(T).all() and torch.isfinite(w).all()
    assert ctl.n_budget_fallbacks >= 4, ctl.n_budget_fallbacks
    print(f"  0.01 ms budget: {ctl.n_budget_fallbacks}/6 solves fell back, outputs finite  OK")
    # a generous budget never triggers
    ctl2 = MpcCtbrController(P, c, max_solve_ms=200.0)
    _, s = run_controller(c, ctl2, P, max_time=3.0)
    assert ctl2.n_budget_fallbacks == 0 and ctl2.n_failures == 0


def test_offline_through_isaac_runner():
    """run_in_vehicle on SimVehicle. NOTE: SimVehicle advances sim time per tick
    regardless of the wall time act() took, so this does NOT test real time."""
    c = _track("straight3")
    ctl = MpcCtbrController(P, c)
    log, s = run_in_vehicle(SimVehicle(P), c, ctl, P, max_time=40, verbose=False)
    assert s["completed"] and s["gates_passed"] == 3 and not s["aborted"], s
    print(f"  SimVehicle straight3: lap {s['lap_time']:.2f} s, controller ms mean "
          f"{np.nanmean(log['ctrl_ms']):.1f}  (offline: not a real-time test)")


def test_update_course_unchanged_is_cheap():
    c = _track("oval6")
    ctl = MpcCtbrController(P, c)
    ctl.reset(1)
    same = copy.deepcopy(c)
    pos = np.array(c.start_pose()[0])
    t0 = time.perf_counter()
    for _ in range(1000):
        assert ctl.update_course(same, pos, np.zeros(3)) is False
    us = 1e6 * (time.perf_counter() - t0) / 1000
    assert ctl.n_replans == 0 and us < 500, us
    # small moves below the thresholds do not re-plan either
    near = copy.deepcopy(c)
    near.gates[2].x += 0.2
    near.gates[2].yaw_deg += 3.0
    assert ctl.update_course(near, pos, np.zeros(3)) is False
    print(f"  unchanged / sub-threshold course: no re-plan, {us:.1f} us per call  OK")


def test_update_course_replans_on_move_and_yaw_and_ignores_passed_gates():
    c = _track("oval6")
    ctl = MpcCtbrController(P, c)
    ctl.reset(1)
    pos = np.array(c.start_pose()[0])
    moved = copy.deepcopy(c)
    moved.gates[3].y += 0.5
    assert ctl.update_course(moved, pos, np.zeros(3)) is True and ctl.n_replans == 1
    print(f"  gate moved 0.5 m: re-planned in {ctl.last_replan_ms:.0f} ms (oval6, 6 gates)")
    assert ctl.update_course(moved, pos, np.zeros(3)) is False      # now it IS the plan
    rot = copy.deepcopy(moved)
    rot.gates[4].yaw_deg += 8.0
    assert ctl.update_course(rot, pos, np.zeros(3)) is True and ctl.n_replans == 2
    # a gate that was already passed may move without a re-plan
    ctl._gate_idx = 2
    passed = copy.deepcopy(rot)
    passed.gates[0].x += 2.0
    passed.gates[1].y -= 1.0
    assert ctl.update_course(passed, pos, np.zeros(3)) is False
    # the remaining-gate plan starts at the current position
    assert np.linalg.norm(ctl._ref.reference_state(0.0)[:3] - pos) < 1e-6


def _fly_with_vision_update(true_course, plan_course, update_at_gate, max_time=40.0, replan=True):
    """QuadSim loop (same gate logic as run_controller). The controller is built
    on plan_course (what it believes); true_course is where the gates really
    are. When the drone has passed `update_at_gate` gates, update_course(true_course)
    is called (the vision map corrects the belief). Returns (completed, lap, margins, ctl)."""
    ctl = MpcCtbrController(P, plan_course)
    sim = QuadSim(1, P)
    start, yaw = true_course.start_pose()
    z = torch.zeros(1)
    sim.reset(pos=torch.tensor(np.array([start]), dtype=torch.float32),
              quat=quat_from_euler_zyx(z, z, z + yaw))
    ctl.reset(1)
    dt, gi, traj = P.control_dt, 0, [start.copy()]
    n = len(true_course)
    for k in range(int(max_time / dt)):
        if replan and gi >= update_at_gate:
            ctl.update_course(true_course, sim.pos[0].numpy().astype(float),
                              sim.vel[0].numpy().astype(float))
        T, w = ctl.act(sim.pos, sim.vel, sim.quat, torch.tensor([gi]), dt, omega=sim.omega)
        pp = sim.pos.clone()
        sim.step_ctbr(T, w)
        traj.append(sim.pos[0].numpy().astype(float))
        passed, hit = check_gate(pp, sim.pos, gate_batch(true_course, torch.tensor([gi])),
                                 P.collision_radius)
        if bool(hit[0]) or sim.pos[0, 2] < 0.05:
            return False, np.nan, edge_margins(np.array(traj), true_course), ctl
        if bool(passed[0]):
            gi += 1
            if gi >= n:
                return True, (k + 1) * dt, edge_margins(np.array(traj), true_course), ctl
    return False, np.nan, edge_margins(np.array(traj), true_course), ctl


def test_midflight_replan_straight3():
    base = _track("straight3")
    true = copy.deepcopy(base)
    true.gates[2].y += 0.6          # last gate really is 0.6 m to the side
    true.gates[1].z += 0.4          # and gate 1 is a bit higher
    # sanity: without the update the plan misses (opening half 0.75, drone radius needs <= 0.45)
    ok0, _, m0, _ = _fly_with_vision_update(true, base, 1, replan=False)
    ok1, lap, m1, ctl = _fly_with_vision_update(true, base, 1, replan=True)
    print(f"  no update: completed={ok0}, margins {np.round(m0, 2)}")
    print(f"  update after gate 0: completed={ok1}, lap {lap:.2f} s, margins {np.round(m1, 2)}, "
          f"re-plans {ctl.n_replans} ({ctl.last_replan_ms:.0f} ms)")
    assert ok1 and ctl.n_replans == 1
    assert not ok0 or np.nanmin(m0) < np.nanmin(m1)
    assert np.nanmin(m1) >= P.collision_radius


if __name__ == "__main__":
    for n in ["test_quadsim_tracks_and_timing", "test_output_is_ctbr_and_clamped",
              "test_budget_fallback", "test_offline_through_isaac_runner",
              "test_update_course_unchanged_is_cheap",
              "test_update_course_replans_on_move_and_yaw_and_ignores_passed_gates",
              "test_midflight_replan_straight3"]:
        print(n)
        globals()[n]()
    if _ALL_MS:
        a = np.array(_ALL_MS)
        print(f"\nsolve ms over all QuadSim flights: mean {a.mean():.1f}, p95 {np.percentile(a, 95):.1f}, "
              f"max {a.max():.1f}")
    print("\nAll tests passed.")
