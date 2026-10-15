"""Self-tests for the PPO racing environment. Run: python test_race_env.py

Expectations are written from first principles (hand-computed body-frame
vectors, signs of distance changes, gate geometry), not by round-tripping the
code's own output. Conventions: ENU world, FLU body, quaternions WXYZ, gate
frame +X right / +Y down / +Z flight direction (yaw 0 = flight along world +X).
Total runtime is well under a minute; the only training is a tiny smoke run.
"""

import dataclasses
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

from course import Course, Gate, gate_batch
from drone_params import QuadParams
from race_env import (RaceEnv, RaceEnvConfig, RaceVecEnv, CoursePool, build_obs,
                      make_course_pool, obs_dim)
from rl_policy import RlPolicyController
from rotations import quat_from_euler_zyx, quat_to_rotmat

P = QuadParams()
HERE = os.path.dirname(os.path.abspath(__file__))
HOVER = torch.zeros(1, 4)


def two_gate_course():
    return Course([Gate("g0", 10.0, 0.0, 2.5, 0.0), Gate("g1", 20.0, 0.0, 2.5, 0.0)], "two")


def quiet_cfg(**kw):
    """No obs noise and (by default) every reward term except the one tested off."""
    base = dict(obs_noise_std=0.0, w_progress=0.0, w_gate=0.0, w_finish=0.0, w_crash=0.0,
                w_rate=0.0, w_smooth=0.0, w_perception=0.0)
    base.update(kw)
    return RaceEnvConfig(**base)


def env1(courses=None, **cfg_kw):
    return RaceEnv(1, quiet_cfg(**cfg_kw), courses=courses or [two_gate_course()], seed=0)


def at(env, x, y, z, vel=(0.0, 0.0, 0.0), yaw=0.0, gate=0):
    zero = torch.zeros(1)
    env.set_state(torch.tensor([[x, y, z]]), vel=torch.tensor([list(vel)]),
                  quat=quat_from_euler_zyx(zero, zero, zero + yaw), gate_idx=gate)


def test_obs_dim_and_hand_values():
    env = env1()
    assert env.obs_dim == obs_dim() == 26
    at(env, 5.0, 0.0, 2.5)                  # 5 m straight behind gate 0 (x=10), yaw 0
    o = env._observe()
    assert o.shape == (1, 26) and torch.isfinite(o).all()
    # layout: vel(3) up_body(3) rates(3) prev_act(4) | gate1 rel(3) normal(3) | gate2 ...
    rel1 = o[0, 13:16]
    assert torch.allclose(rel1, torch.tensor([0.5, 0.0, 0.0]), atol=1e-6), rel1   # 5 m / 10
    assert torch.allclose(o[0, 16:19], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)  # normal = fwd
    rel2 = o[0, 19:22]                       # gate 1 is 15 m ahead
    assert torch.allclose(rel2, torch.tensor([1.5, 0.0, 0.0]), atol=1e-6), rel2
    assert abs(o[0, 25].item() - 0.25) < 1e-6                                     # z = 2.5 / 10
    assert torch.allclose(o[0, 3:6], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6)
    # last gate: the "next" gate repeats the current one
    at(env, 15.0, 0.0, 2.5, gate=1)
    o = env._observe()
    assert torch.allclose(o[0, 13:19], o[0, 19:25]), o
    assert torch.allclose(o[0, 13:16], torch.tensor([0.5, 0.0, 0.0]), atol=1e-6)
    # yawed +90 deg: nose points world +y, so the gate (world +x) is on the body
    # RIGHT = body -y side
    at(env, 5.0, 0.0, 2.5, yaw=math.pi / 2)
    rel1 = env._observe()[0, 13:16]
    assert torch.allclose(rel1, torch.tensor([0.0, -0.5, 0.0]), atol=1e-5), rel1
    # body-frame velocity: world +x velocity seen from a +90 deg yaw is body -y
    at(env, 5.0, 0.0, 2.5, vel=(3.0, 0.0, 0.0), yaw=math.pi / 2)
    assert torch.allclose(env._observe()[0, 0:3], torch.tensor([0.0, -0.3, 0.0]), atol=1e-5)
    print("  obs dim 26, gate 5 m ahead = (+0.5,0,0), yaw 90 deg -> body -y, last-gate repeat  OK")


def test_hover_action_holds_altitude():
    env = env1()
    at(env, 0.0, 0.0, 2.5)
    z0 = env.sim.pos[0, 2].item()
    for _ in range(50):                      # 1 s at 50 Hz
        env.step(HOVER)
    dz = abs(env.sim.pos[0, 2].item() - z0)
    assert dz < 0.02, dz
    print(f"  a=0 holds hover: altitude change {dz * 100:.2f} cm over 1 s  OK")


def test_progress_reward_sign():
    env = env1(w_progress=1.0)
    at(env, 5.0, 0.0, 2.5, vel=(2.0, 0.0, 0.0))        # moving toward gate at x=10
    _, r_to, term, trunc, _ = env.step(HOVER)
    at(env, 5.0, 0.0, 2.5, vel=(-2.0, 0.0, 0.0))       # moving away
    _, r_from, _, _, _ = env.step(HOVER)
    # 2 m/s * 0.02 s = 0.04 m (drag shaves a little)
    assert 0.03 < r_to.item() < 0.045, r_to
    assert -0.045 < r_from.item() < -0.03, r_from
    print(f"  progress reward toward {r_to.item():+.4f}, away {r_from.item():+.4f}  OK")


def test_gate_pass_advances_and_finish_succeeds():
    env = env1(w_gate=10.0, w_finish=10.0)
    at(env, 9.97, 0.0, 2.5, vel=(3.0, 0.0, 0.0))        # crosses x=10 within one step
    _, r, term, trunc, info = env.step(HOVER)
    assert 9.9 < r.item() < 10.1, r
    assert env.gate_idx.item() == 1 and not term.item() and not info[0]
    # now the last gate: pass gives both bonuses, terminates with success
    at(env, 19.97, 0.0, 2.5, vel=(3.0, 0.0, 0.0), gate=1)
    _, r, term, trunc, info = env.step(HOVER)
    assert 19.9 < r.item() < 20.1, r
    assert term.item() and not trunc.item()
    assert info[0]["success"] is True and info[0]["crash_reason"] == "none"
    assert info[0]["gates_passed"] == 1                 # started at gate 1, passed 1 gate
    print("  pass: +10 and target advances; last gate: +20, terminated, success  OK")


def test_bar_hit_and_ground_terminate():
    env = env1(w_crash=10.0)
    # gate +X is world -Y; crossing 0.7 m off-centre: inside the outer frame
    # (0.75 + 0.10 = 0.85) but outside the flyable opening (0.75 - 0.30 = 0.45)
    at(env, 9.97, 0.7, 2.5, vel=(3.0, 0.0, 0.0))
    _, r, term, trunc, info = env.step(HOVER)
    assert term.item() and not trunc.item()
    assert abs(r.item() + 10.0) < 1e-3, r
    assert info[0]["crash_reason"] == "gate" and info[0]["success"] is False
    # falling to the ground: 0.06 m up, 3 m/s down -> z < 0.05 after 1 step
    at(env, 0.0, 0.0, 0.06, vel=(0.0, 0.0, -3.0))
    _, r, term, trunc, info = env.step(HOVER)
    assert term.item() and abs(r.item() + 10.0) < 1e-3
    assert info[0]["crash_reason"] == "ground"
    # tilt: a drone already rolled 85 deg (past the 80 deg limit)
    zero = torch.zeros(1)
    env.set_state(torch.tensor([[0.0, 0.0, 3.0]]), quat=quat_from_euler_zyx(zero + 1.48, zero, zero))
    _, r, term, _, info = env.step(HOVER)
    assert term.item() and info[0]["crash_reason"] == "tilt"
    # out of bounds: 35 m from the target gate
    at(env, -25.0, 0.0, 2.5)
    _, r, term, _, info = env.step(HOVER)
    assert term.item() and info[0]["crash_reason"] == "oob"
    print("  bar hit / ground / tilt / out-of-bounds terminate with -10  OK")


def test_truncation_is_not_termination():
    env = env1(max_episode_time=0.1)                    # 5 control steps
    at(env, 0.0, 0.0, 2.5)
    for k in range(5):
        _, _, term, trunc, info = env.step(HOVER)
        assert not term.item()
        assert trunc.item() == (k == 4), (k, trunc)
    assert info[0]["episode"]["l"] == 5 and info[0]["crash_reason"] == "none"
    assert info[0]["success"] is False
    print("  time limit sets truncated, not terminated  OK")


def test_auto_reset_spawn():
    courses = [Course([Gate(f"g{i}", 8.0 * (i + 1), 3.0 * i, 2.5, 20.0 * i) for i in range(5)], "five")]
    env = RaceEnv(1, quiet_cfg(), courses=courses, seed=3)
    seen = set()
    for _ in range(60):
        env.set_state(torch.tensor([[0.0, 0.0, 0.06]]), vel=torch.tensor([[0.0, 0.0, -3.0]]))
        obs, _, term, _, info = env.step(HOVER)          # ground crash -> auto reset
        assert term.item() and info[0]["terminal_observation"].shape == (26,)
        assert torch.isfinite(obs).all()
        gi, cid = env.gate_idx.item(), env.course_id.item()
        seen.add(gi)
        assert 0 <= gi < 5 and cid == 0
        assert env.ep_len.item() == 0 and (env.prev_action == 0).all()
        # fresh spawn, checked in the gate frame of the chosen start gate:
        g = courses[0].gates[gi]
        T = g.T_world_gate()
        local = T[:3, :3].T @ (env.sim.pos[0].numpy() - T[:3, 3])
        assert -6.0 - 1e-4 <= local[2] <= -2.0 + 1e-4, local         # 2-6 m behind the plane
        assert abs(local[0]) < 3.0 and env.sim.pos[0, 2].item() >= 0.5 - 1e-6
        R = quat_to_rotmat(env.sim.quat)[0]
        assert abs(R[2, 2].item()) > math.cos(math.radians(15.0))    # small tilt
        heading = math.atan2(R[1, 0].item(), R[0, 0].item())
        want = math.atan2(T[1, 2], T[0, 2])
        err = abs(math.atan2(math.sin(heading - want), math.cos(heading - want)))
        assert err <= math.radians(20.0) + 1e-4, err                 # roughly facing the gate
        assert env.sim.vel[0].norm().item() < 4.0
    assert len(seen) >= 4, seen                                      # start gate is randomised
    print(f"  auto-reset: valid spawn behind a random gate (start gates seen: {sorted(seen)})  OK")


def test_vecenv_terminal_infos():
    venv = RaceVecEnv(4, quiet_cfg(), courses=[two_gate_course()], seed=0)
    venv.reset()
    zero = torch.zeros(1)
    venv.env.set_state(torch.tensor([[0.0, 0.0, 0.06]]), vel=torch.tensor([[0.0, 0.0, -3.0]]),
                       env_ids=[2])
    obs, rew, done, infos = venv.step(np.zeros((4, 4), dtype=np.float32))
    assert obs.dtype == np.float32 and rew.dtype == np.float32 and obs.shape == (4, 26)
    assert done.tolist() == [False, False, True, False]
    assert infos[2]["terminal_observation"].shape == (26,)
    assert infos[2]["TimeLimit.truncated"] is False
    assert set(infos[2]["episode"]) == {"r", "l"} and infos[2]["is_success"] is False
    assert not infos[0]
    print("  RaceVecEnv: terminal_observation, TimeLimit.truncated, episode, is_success  OK")


def test_seeding():
    def first_obs(seed):
        return RaceEnv(32, RaceEnvConfig(), seed=seed)._observe()   # noise on: it is seeded too
    assert torch.equal(first_obs(7), first_obs(7))
    assert not torch.equal(first_obs(7), first_obs(8))
    a = RaceEnv(32, RaceEnvConfig(), seed=7)
    b = RaceEnv(32, RaceEnvConfig(), seed=7)
    act = torch.rand(32, 4) * 2 - 1
    for _ in range(3):                                   # includes auto-resets' randomness
        assert torch.equal(a.step(act)[0], b.step(act)[0])
    print("  same seed -> identical observations (first and after stepping), different seed differs  OK")


def test_gate_pool_matches_course_gate_batch():
    courses = [two_gate_course(), Course([Gate("a", 1, 2, 3, 40.0), Gate("b", -4, 5, 2, -75.0),
                                          Gate("c", 0, 9, 1.5, 170.0)], "three")]
    pool = CoursePool(courses)
    cid = torch.tensor([0, 1, 1, 1, 0, 1])
    gi = torch.tensor([1, 0, 2, 5, 7, 1])          # out-of-range indices clamp to the last gate
    got = pool.gates(cid, gi)
    for k in range(len(cid)):
        want = gate_batch(courses[cid[k]], gi[k:k + 1])
        assert torch.allclose(got.center[k], want.center[0]) and torch.allclose(got.R[k], want.R[0])
        assert got.side[k] == want.side[0] and got.bar[k] == want.bar[0]
    print("  CoursePool.gates == course.gate_batch (incl. clamping)  OK")


def _untrained_run_dir(tmp):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
    venv = VecNormalize(RaceVecEnv(8, quiet_cfg(), courses=[two_gate_course()], seed=0))
    model = PPO("MlpPolicy", venv, n_steps=16, batch_size=32, device="cpu", seed=0,
                policy_kwargs=dict(net_arch=dict(pi=[16, 16], vf=[16, 16])))
    model.learn(256)                                # gives obs_rms non-trivial statistics
    model.save(os.path.join(tmp, "model.zip"))
    venv.save(os.path.join(tmp, "vecnormalize.pkl"))
    return model, venv


def test_policy_obs_parity():
    with tempfile.TemporaryDirectory() as tmp:
        model, vn = _untrained_run_dir(tmp)
        course = two_gate_course()
        env = RaceEnv(3, quiet_cfg(), courses=[course], seed=0)
        zero = torch.zeros(3)
        env.set_state(torch.tensor([[2.0, 0.3, 2.0], [9.0, -0.2, 3.0], [15.0, 1.0, 2.2]]),
                      vel=torch.tensor([[1.0, 0.0, 0.2], [2.0, 0.5, 0.0], [0.0, 1.0, -0.3]]),
                      quat=quat_from_euler_zyx(zero + 0.1, zero - 0.05, torch.tensor([0.0, 0.3, 1.0])),
                      omega=torch.tensor([[0.1, 0.2, 0.3], [0.0, -0.4, 0.1], [1.0, 0.0, -0.2]]),
                      gate_idx=torch.tensor([0, 0, 1]))
        env.prev_action = torch.tensor([[0.1, -0.2, 0.3, 0.4], [0.0, 0.0, 0.0, 0.0], [-0.5, 0.5, 0.1, 0.2]])
        ctl = RlPolicyController(P, course, model_dir=tmp)
        ctl.reset(3)
        ctl.prev_action = env.prev_action.clone()
        s = env.sim
        assert torch.equal(ctl.observe(s.pos, s.vel, s.quat, s.omega, env.gate_idx), env._observe())
        # act(): thrust/rates follow the documented mapping of the policy's action
        thrust, rates = ctl.act(s.pos, s.vel, s.quat, env.gate_idx, 0.02, omega=s.omega)
        raw = env._observe().numpy()
        a, _ = model.policy.predict(vn.normalize_obs(raw), deterministic=True)
        a = torch.as_tensor(a).clamp(-1, 1)
        assert torch.allclose(thrust, P.mass * P.gravity * (1 + a[:, 0]), atol=1e-4)
        assert torch.allclose(rates, a[:, 1:] * torch.tensor(P.max_rate), atol=1e-5)
        assert torch.equal(ctl.prev_action, a)
    print("  RlPolicyController builds the identical observation / command as the env  OK")


def test_sb3_smoke():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
    venv = VecNormalize(RaceVecEnv(16, RaceEnvConfig(n_courses=8), seed=0))
    model = PPO("MlpPolicy", venv, n_steps=64, batch_size=256, n_epochs=2, device="cpu", seed=0)
    model.learn(2048)
    assert model.num_timesteps >= 2048
    print("  PPO learns 2k steps on RaceVecEnv (16 envs)  OK")


def test_train_cli_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [sys.executable, "train.py", "--timesteps", "20000", "--n-envs", "64",
               "--name", "smoke", "--out-dir", tmp, "--eval-every", "10000",
               "--eval-max-time", "4", "--n-courses", "16"]
        r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        run = os.path.join(tmp, "smoke")
        for f in ("model.zip", "vecnormalize.pkl", "config.json"):
            assert os.path.isfile(os.path.join(run, f)), f
        assert "[eval @" in r.stdout
        # the saved policy flies through the shared controller interface
        from fly_course import make_controller, run_controller
        course = Course.load(os.path.join(HERE, "tracks", "straight3.json"))
        _, s = run_controller(course, make_controller("rl", P, course, run), P, max_time=3.0)
        assert "gates_passed" in s
    print("  train.py --timesteps 20000 --n-envs 64 runs end to end (+ eval callback, load)  OK")


if __name__ == "__main__":
    print("test_obs_dim_and_hand_values");           test_obs_dim_and_hand_values()
    print("test_hover_action_holds_altitude");       test_hover_action_holds_altitude()
    print("test_progress_reward_sign");              test_progress_reward_sign()
    print("test_gate_pass_advances_and_finish_succeeds"); test_gate_pass_advances_and_finish_succeeds()
    print("test_bar_hit_and_ground_terminate");      test_bar_hit_and_ground_terminate()
    print("test_truncation_is_not_termination");     test_truncation_is_not_termination()
    print("test_auto_reset_spawn");                  test_auto_reset_spawn()
    print("test_vecenv_terminal_infos");             test_vecenv_terminal_infos()
    print("test_seeding");                           test_seeding()
    print("test_gate_pool_matches_course_gate_batch"); test_gate_pool_matches_course_gate_batch()
    print("test_policy_obs_parity");                 test_policy_obs_parity()
    print("test_sb3_smoke");                         test_sb3_smoke()
    print("test_train_cli_end_to_end");              test_train_cli_end_to_end()
    print("\nAll tests passed.")
