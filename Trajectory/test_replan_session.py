"""Self-tests for Trajectory/replan_session.py (behind ros2_nodes/replan_node.py). Run: python3 test_replan_session.py

No ROS. The closed-loop test is the Stage 4 replanning run offline: the
3-gate Isaac course, a prior map off by 0.5 m, camera gate poses only while
the whole gate is in the (level, 60 x 36 deg) frame, one closed lap
(gate_0, gate_1, gate_2, gate_0), every published plan handed to MpcCore
through TrajectoryBuffer exactly as mpc_node does, flown through the PX4-like
attitude plant.
"""

import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(REPO, "control")]

from replan_session import ReplanSession, perturbed_prior          # noqa: E402
from gate_path import gates_from_manifest, gate_crossings          # noqa: E402
from mpc_core import MpcCore, TrajectoryBuffer                      # noqa: E402
from px4_attitude_plant import Px4AttitudePlant, _rot              # noqa: E402
from drone_constraints import drone_constraints                    # noqa: E402

MANIFEST = os.path.join(HERE, "isaac_results", "3_gates", "course", "pod_test_gates.json")
GATES = gates_from_manifest(MANIFEST)
START, START_YAW = np.array([-0.05, -0.04, 2.55]), math.radians(-7.2)
M = drone_constraints["mass"]
HALF_H, HALF_V = math.tan(math.radians(30.0)), math.tan(math.radians(30.0)) * 720 / 1280
CAM = np.array([0.30, 0.0, 0.05])


def _quat_xyzw(R):
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w > 1e-6:
        return ((R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w)
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(0.0, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2.0
    q = [0.0, 0.0, 0.0]
    q[i], q[j], q[k] = s / 4.0, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s
    return (q[0], q[1], q[2], (R[k, j] - R[j, k]) / s)


def camera_poses(x, rng, max_range=12.0, reliable_range=8.0):
    """World-frame gate poses a level front camera would report from state x (whole gate in frame only)."""
    R = _rot(*x[3:6])
    cam = x[0:3] + R @ CAM
    out = []
    for g in GATES:
        corners = np.asarray(g.corners())
        vs = [R.T @ (c - cam) for c in corners]
        if not all(v[0] > 0.3 and abs(v[1] / v[0]) <= HALF_H and abs(v[2] / v[0]) <= HALF_V for v in vs):
            continue
        rng_m = float(np.linalg.norm(g.center - cam))
        if rng_m > max_range:
            continue
        c = g.center + rng.normal(0.0, 0.01 + 0.01 * rng_m, 3)
        q = _quat_xyzw(g.T_world_gate[:3, :3]) if rng_m < reliable_range else (0.0, 0.0, 0.0, 1.0)
        out.append((c, q))
    return out


def fly(seed=0, prior_sigma=0.5, hover_s=2.0, max_s=45.0, dt=0.02):
    rng = np.random.default_rng(seed)
    sess = ReplanSession(perturbed_prior(GATES, prior_sigma, seed), v_max=2.0, a_max=2.0, laps=1, close_loop=True)
    core = MpcCore(N=10, dt=dt, m=M, attitude_lead_s=0.1)
    plant = Px4AttitudePlant(M, tau=0.08)
    x = np.zeros((12, 1)); x[0:3, 0] = START; x[5, 0] = START_YAW
    buf, t, switches, log_t, log_x, err = None, 0.0, [], [], [], []
    plan_ms = []
    while t < max_s:
        if int(round(t / dt)) % 3 == 0:                                  # camera at about 17 Hz
            sess.add_detections(t, camera_poses(x[:, 0], rng), x[0:3, 0])
        if buf is None:
            if t >= hover_s:                                              # OFFBOARD confirmed
                c0 = time.perf_counter()
                p = sess.engage(t, x[0:3, 0], x[5, 0])
                plan_ms.append(1000 * (time.perf_counter() - c0))
                buf = TrajectoryBuffer(p.t0, p.times, p.states, p.thrust)
            elif int(round(t / dt)) % 25 == 0:
                sess.prepare(t, x[0:3, 0], x[5, 0])
        else:
            c0 = time.perf_counter()
            p = sess.tick(t, x[0:3, 0])
            if p is not None:
                plan_ms.append(1000 * (time.perf_counter() - c0))
                new = TrajectoryBuffer(p.t0, p.times, p.states, p.thrust)
                jump = float(np.linalg.norm(new.horizon(t, 0, dt)[0:3, 0] - buf.horizon(t, 0, dt)[0:3, 0]))
                old_rem, new_rem = buf.t0 + buf.duration - t, new.t0 + new.duration - t
                switches.append((t, p.reason, jump, new_rem / max(old_rem, 1e-6)))
                buf = new
            if sess.finished(t) and t > buf.t0 + buf.duration + 1.0:
                break
        if buf is None:
            u = np.array([[M * 9.81], [0.0], [0.0], [0.0]])
            att = x[3:6, 0]
        else:
            ref = buf.horizon(t, core.N, dt, yaw_now=float(x[5, 0]))
            out = core.step(x, ref)
            u, att = out["u0"], out["x_cmd"][3:6, 0]
            err.append(float(np.linalg.norm(x[0:3, 0] - ref[0:3, 0])))
        for _ in range(4):
            x = plant.step(x, u[0, 0], att, dt / 4) if buf is not None else x
            log_t.append(t); log_x.append(x[:, 0].copy())
        t += dt
    return sess, np.array(log_t), np.array(log_x), switches, np.array(err), plan_ms


def test_prior_and_detection_conversion():
    prior = perturbed_prior(GATES, 0.5, 0)
    off = [np.linalg.norm(p.center - g.center) for p, g in zip(prior, GATES)]
    assert all(0.0 < o < 2.0 for o in off) and [p.name for p in prior] == [g.name for g in GATES], off
    exact = perturbed_prior(GATES, 0.0, 0)
    assert all(np.allclose(p.center, g.center) for p, g in zip(exact, GATES))
    sess = ReplanSession(prior)
    g = GATES[1]
    for k in range(6):                                                   # 6 sightings of gate_1 where it truly is
        sess.add_detections(0.1 * k, [(g.center, _quat_xyzw(g.T_world_gate[:3, :3]))], START)
    snap = {name: c for name, c, n, rel, nobs in sess.map_snapshot()}
    before = np.linalg.norm(prior[1].center - g.center)
    after = np.linalg.norm(snap["gate_1"] - g.center)
    assert after < 0.5 * before, (before, after)
    print(f"  prior off by {np.round(off, 2)} m; 6 sightings pull gate_1 from {before:.2f} to {after:.2f} m  OK")


def test_prepare_holds_clock_and_engage_restamps():
    sess = ReplanSession(perturbed_prior(GATES, 0.5, 0), v_max=2.0, a_max=2.0)
    assert sess.prepare(0.0, START, START_YAW).startswith("plan ready")
    for t in (5.0, 30.0, 60.0):                                          # a long hover must not "miss" gate_0
        sess.prepare(t, START, START_YAW)
    assert sess.progress() == (0, 4) and not [e for e in sess.events if "MISSED" in e[1]], sess.events
    p = sess.engage(61.0, START + 0.1, START_YAW)
    assert p.t0 == 61.0 and p.index == 1 and p.gate_names == ["gate_0", "gate_1", "gate_2", "gate_0"], p
    assert np.linalg.norm(p.states[0:3, 0] - START) < 0.2
    sess2 = ReplanSession(perturbed_prior(GATES, 0.5, 0), v_max=2.0, a_max=2.0)
    sess2.prepare(0.0, START, START_YAW)
    p2 = sess2.engage(10.0, START + np.array([1.0, 0, 0]), START_YAW, clock=lambda: 12.5)
    assert p2.t0 == 12.5 and np.linalg.norm(p2.states[0:3, 0] - (START + [1.0, 0, 0])) < 0.05
    print("  60 s hover: no false MISSED; engage re-stamps to now; 1 m drift -> fresh plan from there  OK")


def test_yaw_mode_next_gate():
    # with the VIO flying, the nose must look at the next gate (keeps it in the camera)
    worst = {}
    for mode in ("velocity", "next_gate"):
        sess = ReplanSession(perturbed_prior(GATES, 0.0, 0), v_max=2.0, a_max=2.0, yaw_mode=mode)
        sess.prepare(0.0, START, START_YAW)
        plan = sess.replanner.plans[-1]
        t_g1 = plan.traj.t_breaks[[i for i, w in enumerate(plan.wps) if w['type'] == 'center'][1]]
        off = []
        for t in np.arange(plan.traj.t_breaks[[i for i, w in enumerate(plan.wps) if w['type'] == 'center'][0]] + 0.3,
                           t_g1 - 0.5, 0.1):                     # leaving gate_0 towards gate_1
            x = plan.ref.reference_state(t)
            d = GATES[1].center - x[0:3]
            off.append(abs(math.degrees((math.atan2(d[1], d[0]) - x[5] + math.pi) % (2 * math.pi) - math.pi)))
        worst[mode] = max(off)
    assert worst["next_gate"] < 35.0 < worst["velocity"], worst
    try:
        ReplanSession(perturbed_prior(GATES, 0.0, 0), yaw_mode="sideways").prepare(0.0, START, START_YAW)
        raise AssertionError("bad yaw_mode accepted")
    except ValueError:
        pass
    print(f"  gate_0 -> gate_1: gate_1 at most {worst['next_gate']:.0f} deg off the nose with next_gate "
          f"({worst['velocity']:.0f} deg with velocity yaw)  OK")


def test_closed_lap_with_camera_replanning():
    sess, T, X, switches, err, plan_ms = fly()
    passes = sess.passes()
    route = [name for (_, _, gid, how) in passes for name in [sess.map.gates[gid].name]]
    assert route == ["gate_0", "gate_1", "gate_2", "gate_0"], passes
    assert all(not how.startswith("MISSED") for (_, _, _, how) in passes), passes
    through = {g.name: sum(1 for c in cs if c["inside"] and c["forward"])
               for g, cs in gate_crossings(T, X[:, 0:3], X[:, 6:9], GATES)}
    assert through == {"gate_0": 2, "gate_1": 1, "gate_2": 1}, through
    assert len(switches) >= 1, "no in-flight replan: the camera never moved a planned gate"
    assert max(j for _, _, j, _ in switches) < 0.15, switches
    # an in-flight replan keeps the previous plan's timing (Replanner._timing_hint): without it the
    # rest of this lap took about 2x as long (47 s instead of about 19 s)
    assert max(r for *_, r in switches) < 1.3, switches
    assert max(err) < 0.8, max(err)
    print(f"  closed lap with a 0.5 m-off prior: passes {route} (true path: {through}); "
          f"{len(switches)} in-flight replan(s), reference jump at a switch <= {max(j for _, _, j, _ in switches):.3f} m, "
          f"remaining time x{max(r for *_, r in switches):.2f} at worst; "
          f"tracking max {max(err):.2f} m; planning {np.median(plan_ms):.0f} ms median, {max(plan_ms):.0f} max  OK")
    for t, reason, jump, r in switches:
        print(f"    t={t:5.2f}s  {reason}  (jump {jump:.3f} m, remaining time x{r:.2f})")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    t_all = time.time()
    for t in tests:
        print(t.__name__)
        t()
    print(f"\nAll tests passed. ({len(tests)})  [{time.time() - t_all:.0f} s]")
