"""Self-tests for Trajectory/min_snap_planner.py (the planning behind ros2_nodes/min_snap_node.py). Run: python3 test_min_snap_planner.py

No ROS. Uses the vision node's own quaternion helper, so the gate convention
is checked end to end: T_world_gate -> PoseArray orientation -> Gate.
"""

import math
import os
import sys
from types import SimpleNamespace as NS

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(REPO, "vision"), os.path.join(REPO, "control"), os.path.join(REPO, "ros2_nodes")]

import min_snap_planner as msn                                # noqa: E402
from eval_isaac_gates import T_world_gate_from_spec           # noqa: E402
from ros_camera_pose_fallback import _rot_to_quat_xyzw        # noqa: E402  (what the vision node publishes)
from gate_path import Gate, gate_crossings                    # noqa: E402
from drone_constraints import drone_constraints               # noqa: E402

GATES_3 = [(5.1644, 0.5666, 3.3667, 7.909), (8.6018, -3.2453, 3.6556, -22.091), (13.0455, -2.0747, 3.7681, -7.091)]
START = np.array([-0.0469, -0.0367, 2.5484])
START_YAW = math.radians(-7.24)


def _pose(T):
    """A PoseArray entry exactly as ros2_nodes/ros_camera_pose_fallback.py builds it."""
    q = _rot_to_quat_xyzw(T[:3, :3])
    return NS(x=T[0, 3], y=T[1, 3], z=T[2, 3]), tuple(float(v) for v in q)


def test_gate_orientation_survives_the_pose_array():
    for x, y, z, yaw in GATES_3:
        T = T_world_gate_from_spec(x, y, z, yaw)
        pos, q = _pose(T)
        g = msn.gate_from_pose(pos, q)
        true_n = Gate("t", T).normal
        assert g.normal_reliable and np.allclose(g.normal, true_n, atol=1e-9), (g.normal, true_n)
        assert abs(g.normal[2]) < 1e-9                            # horizontal, not straight up
    print("  vision gate pose -> Gate keeps the real flight direction (old node: (0, 0, 1) for every gate)  OK")


def test_identity_quaternion_means_center_only():
    g = msn.gate_from_pose(NS(x=6.0, y=1.0, z=3.0), (0.0, 0.0, 0.0, 1.0), viewer_pos=[0.0, 1.0, 3.0])
    assert not g.normal_reliable and np.allclose(g.normal, [1.0, 0.0, 0.0])
    print("  identity orientation (vision: normal unreliable) -> center only, facing away from the drone  OK")


def test_gate_latch_confirms_averages_and_orders():
    rng = np.random.default_rng(0)
    latch = msn.GateLatch(match_dist=1.0, min_hits=3)
    true = [T_world_gate_from_spec(x, y, z, yaw) for x, y, z, yaw in GATES_3]
    viewer = START
    for frame in range(10):
        dets = []
        # far gate seen first in the list, near gate last: order must come from distance, not list order
        for T in reversed(true):
            if frame == 0 and T is true[2]:
                continue                                          # missed in one frame: fine
            Tn = T.copy()
            Tn[:3, 3] += rng.normal(0, 0.05, 3)
            pos, q = _pose(Tn)
            dets.append(msn.gate_from_pose(pos, q, viewer_pos=viewer))
        latch.update(dets, now=0.1 * frame, viewer_pos=viewer)
        if frame == 1:
            assert len(latch.confirmed()) == 0                    # nothing has 3 sightings yet
    gates = latch.confirmed(viewer_pos=viewer)
    assert [g.name for g in gates] == ["gate_0", "gate_1", "gate_2"] and len(latch.gates) == 3
    for g, T in zip(gates, true):
        assert np.linalg.norm(g.center - T[:3, 3]) < 0.05, np.linalg.norm(g.center - T[:3, 3])
        assert np.degrees(np.arccos(np.clip(g.normal @ Gate("t", T).normal, -1, 1))) < 1.0
    print("  latch: 3 gates confirmed after 3 sightings, averaged to < 5 cm, in course order  OK")


def test_plan_uses_constraints_mass_and_passes_all_gates():
    gates = [Gate(f"gate_{i}", T_world_gate_from_spec(*g)) for i, g in enumerate(GATES_3)]
    traj, wps, ref = msn.plan_course(gates, START, START_YAW, v_max=2.0, a_max=2.0)
    ts, states, thrust = msn.trajectory_points(ref, dt=0.02)
    assert states.shape == (12, len(ts)) and ts[0] == 0.0
    assert np.allclose(states[0:3, 0], START, atol=1e-6) and abs(states[5, 0] - START_YAW) < 1e-6
    hover_thrust = drone_constraints["mass"] * 9.81
    assert abs(thrust[0] - hover_thrust) < 1e-6, (thrust[0], hover_thrust)   # old node: m = 1.0
    res = gate_crossings(ts, states[0:3].T, states[6:9].T, gates)
    assert all(any(c["inside"] and c["forward"] for c in cs) for _, cs in res)
    print(f"  plan from the live pose: {traj.duration:.1f} s through all 3 gates, feed-forward thrust "
          f"{thrust[0]:.2f} N at hover = {drone_constraints['mass']} kg x g  OK")


def test_default_limits_are_80_percent_of_the_drone():
    import gate_path
    assert abs(gate_path.DEFAULT_V_MAX - 0.8 * drone_constraints["linear_velocity_limits"]["max"]) < 1e-12
    assert abs(gate_path.DEFAULT_A_MAX - 0.8 * drone_constraints["linear_accel_limits"]["max"]) < 1e-12
    print(f"  planner defaults: {gate_path.DEFAULT_V_MAX:.1f} m/s, {gate_path.DEFAULT_A_MAX:.1f} m/s^2 "
          f"= 80% of control/drone_constraints.py  OK")


def test_time_optimized_split_is_faster_within_limits():
    gates = [Gate(f"gate_{i}", T_world_gate_from_spec(*g)) for i, g in enumerate(GATES_3)]
    for v, a in ((2.0, 2.0), (None, None)):
        old, _ = msn.plan_gate_trajectory(gates, START, v_max=v, a_max=a, verbose=False, optimize_times=False)
        new, _ = msn.plan_gate_trajectory(gates, START, v_max=v, a_max=a, verbose=False)
        import gate_path
        vl, al = (v or gate_path.DEFAULT_V_MAX), (a or gate_path.DEFAULT_A_MAX)
        t, p, vel, acc, _, _ = new.sample(0.002)
        assert np.linalg.norm(vel, axis=1).max() <= vl * 1.01 and np.linalg.norm(acc, axis=1).max() <= al * 1.01
        assert new.duration < 0.95 * old.duration, (new.duration, old.duration)
        res = gate_crossings(t, p, vel, gates)
        assert all(any(c["inside"] and c["forward"] for c in cs) for _, cs in res)
        print(f"  limits {vl:.1f} / {al:.1f}: {old.duration:.2f} s -> {new.duration:.2f} s "
              f"({100 * (1 - new.duration / old.duration):.0f}% faster), within limits, all gates  OK")


def test_moving_start_keeps_the_old_fit():
    gates = [Gate(f"gate_{i}", T_world_gate_from_spec(*g)) for i, g in enumerate(GATES_3)]
    st = {1: np.array([1.0, 0.0, 0.0])}
    a, _ = msn.plan_gate_trajectory(gates, START, v_max=2.0, a_max=2.0, verbose=False, start_state=st)
    b, _ = msn.plan_gate_trajectory(gates, START, v_max=2.0, a_max=2.0, verbose=False, start_state=st,
                                    optimize_times=False)
    assert abs(a.duration - b.duration) < 1e-12
    print("  in-flight replan (moving start): distance-proportional fit as before  OK")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        t()
    print(f"\nAll tests passed. ({len(tests)})")
