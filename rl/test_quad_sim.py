"""Self-tests for the batched quad simulator. Run: python test_quad_sim.py

Expectations are written from first principles (hand-derived signs and
numbers from the Pegasus Iris parameters), not by round-tripping the code's
own conventions: a wrong sign in both directions would otherwise pass.
Conventions: ENU world, FLU body, quaternions WXYZ, rates about body axes.
"""

import math

import torch

from drone_params import QuadParams
from quad_sim import QuadSim
from rotations import quat_to_rotmat, quat_from_euler_zyx, rotmat_to_euler_zyx

P = QuadParams()
DT = torch.float64
MG = P.mass * P.gravity


def sim(n=1, params=P):
    return QuadSim(n, params, dtype=DT)


def hold(s, thrust, rates, seconds):
    n = s.n
    T = torch.full((n,), float(thrust), dtype=DT)
    r = torch.as_tensor(rates, dtype=DT).expand(n, 3)
    for _ in range(int(round(seconds / P.control_dt))):
        s.step_ctbr(T, r)


def body_axes(s):
    """Columns of R: body x, y, z expressed in the world frame."""
    return quat_to_rotmat(s.quat)


def test_hover():
    s = sim()
    hold(s, MG, [0, 0, 0], 2.0)
    drift = s.pos.norm().item()
    assert drift < 0.01, drift
    w_hover = P.hover_rotor_speed()
    assert abs(w_hover - 656.0) < 1.0, w_hover
    assert abs(P.hover_motor_output() - 0.556) < 0.001
    # round trip of the PX4 mapping at hover (only a sanity check on the helper)
    assert abs(P.px4_output_to_thrust(P.hover_motor_output()) - MG) < 1e-6
    assert abs(P.thrust_to_px4_output(MG) - P.hover_motor_output()) < 1e-9
    print(f"  hover drift {drift:.2e} m, w_hover {w_hover:.1f} rad/s, "
          f"u_hover {P.hover_motor_output():.3f}  OK")


def test_free_fall():
    s = sim()
    hold(s, 0.0, [0, 0, 0], 1.0)
    # constant a = -g for 1 s (drag coefficient in z is 0): v = -9.81 exactly
    assert abs(s.vel[0, 2].item() + 9.81) < 1e-6, s.vel
    assert abs(s.vel[0, 0].item()) < 1e-9 and abs(s.vel[0, 1].item()) < 1e-9
    # semi-implicit Euler lands just below the exact 0.5 g t^2 = 4.905
    assert abs(s.pos[0, 2].item() + 4.905) < 0.05, s.pos
    print(f"  after 1 s: vz = {s.vel[0, 2].item():.4f} m/s  OK")


def test_roll_direction():
    """Roll rate about body +x (forward): left side up, thrust tilts to -y."""
    s = sim()
    hold(s, MG, [1.0, 0, 0], 0.1)       # ~0.1 rad of roll
    hold(s, MG, [0, 0, 0], 0.5)         # hold attitude and let it accelerate
    R = body_axes(s)[0]
    assert R[1, 2].item() < -0.05, R[:, 2]          # body z tilted toward world -y
    assert R[1, 1].item() > 0 and R[2, 1].item() > 0.05  # body y (left) points up
    assert s.vel[0, 1].item() < -0.3, s.vel
    assert abs(s.vel[0, 0].item()) < 0.05, s.vel
    print(f"  roll pulse -> vy = {s.vel[0, 1].item():.3f} m/s (right side dips)  OK")


def test_pitch_direction():
    """Pitch rate about body +y (left): nose down, thrust tilts to +x."""
    s = sim()
    hold(s, MG, [0, 1.0, 0], 0.1)
    hold(s, MG, [0, 0, 0], 0.5)
    R = body_axes(s)[0]
    assert R[2, 0].item() < -0.05, R[:, 0]          # body x (nose) points below horizon
    assert s.vel[0, 0].item() > 0.3, s.vel
    assert abs(s.vel[0, 1].item()) < 0.05, s.vel
    print(f"  pitch pulse -> vx = {s.vel[0, 0].item():.3f} m/s (nose down)  OK")


def test_yaw_direction():
    s = sim()
    hold(s, MG, [0, 0, 1.0], 0.5)
    R = body_axes(s)[0]
    # nose direction (body x in world) rotated counter-clockwise from +x:
    # positive y component, cos > 0 for a 0.5 rad turn
    assert R[1, 0].item() > 0.4 and R[0, 0].item() > 0.8, R[:, 0]
    yaw = rotmat_to_euler_zyx(R[None])[2].item()
    # ideal would be 0.5 rad; the yaw rate loop (Kp = 20 1/s, ~50 ms lag) costs ~0.05 rad
    assert 0.40 < yaw < 0.50, yaw
    print(f"  yaw rate +1 rad/s for 0.5 s -> yaw {yaw:.3f} rad (CCW from above)  OK")


def test_euler_to_quat_hand_vectors():
    """yaw 90 deg: nose (body x) -> world +y; pitch +90deg-ish: nose points down."""
    q = quat_from_euler_zyx(torch.zeros(1, dtype=DT), torch.zeros(1, dtype=DT),
                            torch.full((1,), math.pi / 2, dtype=DT))
    R = quat_to_rotmat(q)[0]
    assert torch.allclose(R[:, 0], torch.tensor([0.0, 1.0, 0.0], dtype=DT), atol=1e-12)
    q = quat_from_euler_zyx(torch.zeros(1, dtype=DT), torch.full((1,), 0.3, dtype=DT),
                            torch.zeros(1, dtype=DT))
    R = quat_to_rotmat(q)[0]
    assert R[2, 0].item() < 0, "positive pitch must point the nose down"
    # roll +0.3: body y (left) tilts up
    q = quat_from_euler_zyx(torch.full((1,), 0.3, dtype=DT), torch.zeros(1, dtype=DT),
                            torch.zeros(1, dtype=DT))
    assert quat_to_rotmat(q)[0, 2, 1].item() > 0
    print("  euler -> quaternion -> rotation matrix hand vectors  OK")


def test_mixer_inverse():
    s = sim()
    xy = P.rotor_xy
    T = torch.tensor([MG, 18.0], dtype=DT)
    tau = torch.tensor([[0.2, -0.1, 0.02], [-0.3, 0.25, -0.03]], dtype=DT)
    f = s.mix(T, tau)
    for e in range(2):
        fe = f[e]
        assert abs(fe.sum().item() - T[e].item()) < 1e-9
        roll = sum(xy[i][1] * fe[i].item() for i in range(4))
        pitch = -sum(xy[i][0] * fe[i].item() for i in range(4))
        yaw = sum(P.rot_dir[i] * P.k_m / P.k_f * fe[i].item() for i in range(4))
        assert abs(roll - tau[e, 0].item()) < 1e-9, (roll, tau[e])
        assert abs(pitch - tau[e, 1].item()) < 1e-9
        assert abs(yaw - tau[e, 2].item()) < 1e-9
    # saturation: absurd request stays inside [0, f_max] and finite
    f = s.mix(torch.tensor([100.0], dtype=DT), torch.tensor([[50.0, -50.0, 5.0]], dtype=DT))
    assert torch.isfinite(f).all()
    assert (f >= 0).all() and (f <= P.max_rotor_thrust() + 1e-12).all()
    # roll torque request must put more thrust on the left rotors (y > 0)
    f = s.mix(torch.tensor([MG], dtype=DT), torch.tensor([[0.3, 0.0, 0.0]], dtype=DT))[0]
    assert f[2] > f[0] and f[1] > f[3], f
    print("  mixer reproduces thrust/torque, saturates inside limits  OK")


def test_yaw_torque_sign():
    s = sim()
    f = s.mix(torch.tensor([MG], dtype=DT), torch.tensor([[0.0, 0.0, 0.02]], dtype=DT))[0]
    w = s.thrust_to_rotor_speed(f)
    w_h = P.hover_rotor_speed()
    for i in range(4):
        if P.rot_dir[i] > 0:
            assert w[i] > w_h, (i, w)       # +1 rotors speed up for +yaw torque
        else:
            assert w[i] < w_h, (i, w)
    s.step_rotor_speeds(w.reshape(1, 4))
    assert s.omega[0, 2].item() > 0, s.omega        # positive yaw acceleration
    assert abs(s.omega[0, 0].item()) < 1e-9 and abs(s.omega[0, 1].item()) < 1e-9
    print("  +yaw torque: rot_dir=+1 rotors speed up, yaw accel > 0  OK")


def test_rate_step_response():
    """Roll rate step of 1 rad/s at hover thrust: 63% within ~50 ms."""
    s = sim()
    ref = torch.tensor([[1.0, 0.0, 0.0]], dtype=DT)
    T = torch.tensor([MG], dtype=DT)
    dt = P.physics_dt
    t63 = None
    for k in range(1, 251):
        torque = s.rate_ctrl(s.omega, ref, dt)
        s.step_rotor_speeds(s.thrust_to_rotor_speed(s.mix(T, torque)), dt)
        if t63 is None and s.omega[0, 0].item() >= 0.632:
            t63 = k * dt
    assert t63 is not None and 0.01 < t63 < 0.05, t63
    assert abs(s.omega[0, 0].item() - 1.0) < 0.05, s.omega
    print(f"  rate loop reaches 63% in {t63 * 1000:.0f} ms, settles at "
          f"{s.omega[0, 0].item():.3f} rad/s  OK")


def test_batching_independent():
    cmd = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.5, 0.5]], dtype=DT)
    a = sim(2)
    for k in range(50):
        r = cmd if k < 10 else torch.zeros_like(cmd)
        a.step_ctbr(torch.full((2,), MG, dtype=DT), r)
    # env 0 asked for nothing: it must not have moved
    assert a.pos[0].norm().item() < 1e-9 and a.omega[0].norm().item() < 1e-9
    assert a.vel[1].norm().item() > 0.1
    # env 1 alone must match env 1 in the batch
    b = sim(1)
    for k in range(50):
        r = cmd[1:] if k < 10 else torch.zeros(1, 3, dtype=DT)
        b.step_ctbr(torch.full((1,), MG, dtype=DT), r)
    assert torch.allclose(a.pos[1], b.pos[0], atol=1e-9)
    assert torch.allclose(a.quat[1], b.quat[0], atol=1e-9)
    print("  two envs evolve independently and match single-env runs  OK")


def test_quaternion_unit_norm():
    s = sim(4)
    gen = torch.Generator().manual_seed(0)
    for _ in range(200):
        r = (torch.rand(4, 3, generator=gen, dtype=DT) - 0.5) * 8.0
        s.step_ctbr(MG * (0.6 + torch.rand(4, generator=gen, dtype=DT)), r)
    n = s.quat.norm(dim=-1)
    assert torch.allclose(n, torch.ones_like(n), atol=1e-9), n
    print("  quaternion stays unit after aggressive commands  OK")


def test_motor_lag():
    p = QuadParams(tau_motor=0.05)
    s = QuadSim(1, p, dtype=DT)
    s.reset()
    # step the rotor-speed command up; first-order lag reaches 63% after tau
    w0, w1 = p.hover_rotor_speed(), p.hover_rotor_speed() + 100.0
    t63 = None
    for k in range(1, 400):
        s.step_rotor_speeds(torch.full((1, 4), w1, dtype=DT))
        if t63 is None and s.rotor_w[0, 0].item() >= w0 + 63.2:
            t63 = k * p.physics_dt
    assert abs(t63 - 0.05) < 0.01, t63
    print(f"  motor lag tau=50 ms -> 63% at {t63 * 1000:.0f} ms  OK")


if __name__ == "__main__":
    print("test_hover");                       test_hover()
    print("test_free_fall");                   test_free_fall()
    print("test_roll_direction");              test_roll_direction()
    print("test_pitch_direction");             test_pitch_direction()
    print("test_yaw_direction");               test_yaw_direction()
    print("test_euler_to_quat_hand_vectors");  test_euler_to_quat_hand_vectors()
    print("test_mixer_inverse");               test_mixer_inverse()
    print("test_yaw_torque_sign");             test_yaw_torque_sign()
    print("test_rate_step_response");          test_rate_step_response()
    print("test_batching_independent");        test_batching_independent()
    print("test_quaternion_unit_norm");        test_quaternion_unit_norm()
    print("test_motor_lag");                   test_motor_lag()
    print("\nAll tests passed.")
