"""Self-tests for the PID baseline. Run: python test_pid_controller.py

The closed-loop tests fly the fixed tracks in QuadSim: every gate must be
passed in order (the runner only ever tests the current target gate), with no
gate collision and no ground hit. Lap times are printed for the record.
"""

import math
import os

import torch

from course import Course, TRACKS_DIR
from drone_params import QuadParams
from fly_course import run_controller
from pid_controller import PidController
from rotations import quat_from_euler_zyx, quat_to_rotmat

P = QuadParams()


def _unit_course():
    from course import Gate
    return Course([Gate("g0", 10.0, 0.0, 2.0, 0.0)], "unit")


def test_attitude_sign():
    """Hand-derived: a drone rolled +0.2 rad (left side up) that wants to hover
    must command a NEGATIVE roll rate. A target ahead (+x) at the same height
    must command a POSITIVE pitch rate (nose down) from level attitude."""
    c = _unit_course()
    ctl = PidController(P, c)
    ctl.reset(1)
    z = torch.zeros(1)
    # hover over (10 - 1.5, 0, 2): pre point reached -> zero position error
    pos = torch.tensor([[8.5, 0.0, 2.0]])
    q = quat_from_euler_zyx(torch.full((1,), 0.2), z, z)
    thrust, rates = ctl.act(pos, torch.zeros(1, 3), q, torch.zeros(1, dtype=torch.long), 0.02)
    assert rates[0, 0].item() < -0.3, rates
    # level, far behind the pre point on the gate line: accelerate toward +x
    ctl.reset(1)
    pos = torch.tensor([[0.0, 0.0, 2.0]])
    q = quat_from_euler_zyx(z, z, z)
    thrust, rates = ctl.act(pos, torch.zeros(1, 3), q, torch.zeros(1, dtype=torch.long), 0.02)
    assert rates[0, 1].item() > 0.3, rates
    assert abs(rates[0, 0].item()) < 1e-3 and abs(rates[0, 2].item()) < 1e-3
    assert thrust[0].item() > 0.9 * P.hover_thrust()
    print("  attitude error -> rate sign (roll back, pitch forward)  OK")


def _fly(name):
    course = Course.load(os.path.join(TRACKS_DIR, f"{name}.json"))
    ctl = PidController(P, course)
    log, s = run_controller(course, ctl, P, max_time=60.0, seed=0)
    assert s["completed"], s
    assert s["gates_passed"] == len(course), s
    assert not s["crashed"], s
    assert log["pos"][:, 0, 2].min() > 0.5, "flew too low"
    assert s["lap_time"] < 45.0, s
    print(f"  {name}: lap {s['lap_time']:.2f} s, max speed {s['max_speed']:.2f} m/s, "
          f"mean {s['mean_speed']:.2f} m/s  OK")


def test_straight3():  _fly("straight3")
def test_oval6():      _fly("oval6")
def test_slalom5():    _fly("slalom5")


def test_batched_start_noise():
    """4 envs with start noise all finish straight3 (robustness + batching)."""
    course = Course.load(os.path.join(TRACKS_DIR, "straight3.json"))
    log, s = run_controller(course, PidController(P, course), P, max_time=40.0,
                            n_envs=4, seed=1, start_noise=0.15)
    assert log["completed"].all() and not log["crashed"].any(), (log["completed"], log["crash_kind"])
    print(f"  4 noisy starts all complete, laps {log['lap_time'].round(2).tolist()}  OK")


if __name__ == "__main__":
    print("test_attitude_sign");        test_attitude_sign()
    print("test_straight3");            test_straight3()
    print("test_oval6");                test_oval6()
    print("test_slalom5");              test_slalom5()
    print("test_batched_start_noise");  test_batched_start_noise()
    print("\nAll tests passed.")
