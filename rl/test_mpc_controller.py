"""Self-tests for the MPC adapter. Run: python test_mpc_controller.py

First-principles expectations (hand-derived signs and numbers), not round
trips. The last test flies straight3 with the real NMPC and takes ~1 minute.
"""

import math
import os
import time

import numpy as np
import torch

import mpc_controller as mc                       # sets up the control/ and Trajectory/ paths
from course import Course, TRACKS_DIR
from drone_params import QuadParams
from fly_course import run_controller
from quad_sim import QuadSim
from rotations import quat_to_rotmat, quat_from_euler_zyx

P = QuadParams()
DT = torch.float64
MG = P.mass * P.gravity


def test_wrench_hover():
    """Thrust m g and zero torque holds hover: drift < 1 cm after 1 s."""
    s = QuadSim(1, P, dtype=DT)
    s.reset(pos=torch.tensor([[0.0, 0.0, 2.0]], dtype=DT))
    for _ in range(int(round(1.0 / P.control_dt))):
        s.step_wrench(torch.tensor([MG], dtype=DT), torch.zeros(1, 3, dtype=DT))
    drift = (s.pos - torch.tensor([[0.0, 0.0, 2.0]], dtype=DT)).norm().item()
    assert drift < 0.01, drift
    assert s.omega.abs().max().item() < 1e-6
    print(f"  hover drift after 1 s = {drift * 1000:.4f} mm  OK")


def test_wrench_roll_direction():
    """Positive body torque about +x (forward) lifts the LEFT side (body +y gains +z)
    and gives positive roll rate."""
    s = QuadSim(1, P, dtype=DT)
    for _ in range(5):
        s.step_wrench(torch.tensor([MG], dtype=DT), torch.tensor([[0.3, 0.0, 0.0]], dtype=DT))
    R = quat_to_rotmat(s.quat)
    y_axis_z = R[0, 2, 1].item()              # world z-component of body +y
    assert y_axis_z > 0.01, y_axis_z
    assert s.omega[0, 0].item() > 0.0
    # alpha = tau / Ix, integrated over 5 control steps (0.1 s): ~ 0.3/0.029125*0.1
    assert abs(s.omega[0, 0].item() - 0.3 / 0.029125 * 0.1) < 0.05
    print(f"  +roll torque: body y z-component {y_axis_z:.3f}, p = {s.omega[0,0].item():.3f} rad/s  OK")


def test_state_conversion():
    """Yawed +90 deg, world velocity (1,0,0): psi = +pi/2, x_dot = 1 (world, not body)."""
    z = torch.zeros(1, dtype=DT)
    q = quat_from_euler_zyx(z, z, z + math.pi / 2)
    x = mc.state_vector(torch.tensor([[1.0, 2.0, 3.0]], dtype=DT), torch.tensor([[1.0, 0.0, 0.0]], dtype=DT),
                        q, torch.tensor([[0.1, 0.2, 0.3]], dtype=DT))[0]
    assert abs(x[5] - math.pi / 2) < 1e-6, x
    assert abs(x[6] - 1.0) < 1e-9 and abs(x[7]) < 1e-9
    assert abs(x[3]) < 1e-6 and abs(x[4]) < 1e-6
    assert np.allclose(x[:3], [1, 2, 3]) and np.allclose(x[9:], [0.1, 0.2, 0.3])
    # roll +0.3 rad about the forward axis at yaw 0 -> phi = +0.3
    q = quat_from_euler_zyx(z + 0.3, z, z)
    x = mc.state_vector(torch.zeros(1, 3, dtype=DT), torch.zeros(1, 3, dtype=DT), q, torch.zeros(1, 3, dtype=DT))[0]
    assert abs(x[3] - 0.3) < 1e-6
    print("  state vector (yaw 90, world velocity, roll)  OK")


def test_yaw_unwrap():
    assert abs(mc.unwrap_yaw(-3.1, 3.1) - (-3.1 + 2 * math.pi)) < 1e-12
    assert abs(mc.unwrap_yaw(-3.1, 3.1) - 3.183) < 1e-3
    assert abs(mc.unwrap_yaw(0.2, 0.1) - 0.2) < 1e-12               # unchanged near the reference
    assert abs(mc.unwrap_yaw(3.1, -3.1) - (3.1 - 2 * math.pi)) < 1e-12
    print("  yaw unwrap -3.1 -> 3.183 for reference 3.1  OK")


def test_inertia_keyword():
    """Pure roll torque on a level, still drone: dp/dt = U2 / Ix."""
    mc_ = mc  # noqa: F841  (ensures control/ is on sys.path)
    from New_Drone_State_Equations import get_drone_dynamics
    x0 = np.zeros(12)
    u = np.array([0.0, 0.1, 0.0, 0.0])
    f, _, _ = get_drone_dynamics()
    assert abs(float(f(x0, u)[9]) - 0.1 / 0.0081) < 1e-9
    f, _, _ = get_drone_dynamics(inertia=(0.029125, 0.029125, 0.055225))
    assert abs(float(f(x0, u)[9]) - 0.1 / 0.029125) < 1e-9
    u = np.array([0.0, 0.0, 0.0, 0.1])
    assert abs(float(f(x0, u)[11]) - 0.1 / 0.055225) < 1e-9
    f, _, _ = get_drone_dynamics()
    assert abs(float(f(x0, u)[11]) - 0.1 / 0.0142) < 1e-9
    print("  inertia=None -> 0.0081/0.0142; inertia=(...) -> Iris values  OK")


def test_mpc_flies_straight3():
    course = Course.load(os.path.join(TRACKS_DIR, "straight3.json"))
    ctl = mc.MpcController(P, course)
    t0 = time.time()
    log, s = run_controller(course, ctl, P, max_time=30.0)
    print(f"  straight3 wall time {time.time() - t0:.1f} s, lap {s['lap_time']:.2f} s, "
          f"solves {len(ctl.solve_times)}, failures {ctl.n_failures}")
    assert s["completed"] and s["gates_passed"] == 3, s
    assert not s["crashed"], s
    assert log["pos"][:, 0, 2].min() > 0.5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("\nAll tests passed.")
