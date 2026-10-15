# RL Module Handoff

For whoever (or whatever) picks up the RL stack next. Scope: `rl/` only.
Read the module docstrings in `quad_sim.py` and `course.py` before touching
a sign or a frame -- they explain *why*.

Last updated: 2026-10-15.

## Purpose

RL is a standalone controller for the racing drone, to be compared against
the PID baseline (here) and Antek's MPC (`control/`). Demo target ~2026-11-05.
This first chunk builds everything the policy needs *except* the RL itself:
a fast batched simulator of the Isaac Iris, the course/gate-crossing logic,
and a PID baseline speaking the same command interface as the future policy
(collective thrust [N] + body rates [rad/s], "CTBR"). Only numpy + torch.

## Conventions (same as `vision/`)

- World ENU (x east, y north, z up) = Pegasus "map" = `test_gates.json`.
  Gravity -z, g = 9.81.
- Body FLU (x forward, y left, z up); body rates in the body frame.
- Quaternions WXYZ, body -> world (ROS is XYZW; convert at any ROS boundary).
- `T_a_b` maps frame b -> a.
- Gate frame: +X right, +Y down, +Z flight direction; yaw 0 = flight along
  world +X, positive yaw CCW from above. `course.Gate.T_world_gate` is
  verified equal to `workspace/spawn_test_gates._T_world_gate`.
- Everything is batched over a leading env dim N (torch). Default float32.

## Files and status

| File | Status | Notes |
|---|---|---|
| `drone_params.py` | done | `QuadParams`: single source of truth for the airframe |
| `rotations.py` | done | batched quaternion / rotation-matrix / Euler helpers |
| `quad_sim.py` | done | rigid body + mixer + CTBR rate loop + optional motor lag |
| `course.py` | done | `Gate`, `Course`, `check_gate`, `random_course`, fixed tracks |
| `tracks/*.json` | done | `straight3`, `oval6`, `slalom5`; regenerate with `python3 course.py --write-tracks` |
| `pid_controller.py` | done | cascaded PID, pre/center/post waypoints per gate |
| `fly_course.py` | done | runner + CLI + optional matplotlib plot; `--controller pid\|rl`; passes `omega=` to controllers |
| `race_env.py` | done, untrained | batched torch `RaceEnv` + SB3 `RaceVecEnv`, shared `build_obs`, `CoursePool` |
| `train.py` | done, untrained | PPO + VecNormalize CLI, checkpoints, periodic eval on the fixed tracks |
| `rl_policy.py` | done | `RlPolicyController`: trained policy behind the PID's controller interface |
| `test_*.py` | done | plain-function tests, no pytest (`test_race_env.py` ~10 s incl. a smoke training run) |

## Running

```
cd rl
/usr/local/bin/python3.12 test_quad_sim.py
/usr/local/bin/python3.12 test_course.py
/usr/local/bin/python3.12 test_pid_controller.py
/usr/local/bin/python3.12 test_race_env.py
/usr/local/bin/python3.12 fly_course.py --track tracks/oval6.json --controller pid [--plot out.png]
```
Each test file ends with `All tests passed.`; the whole suite takes ~20 s.
The first `--plot` can take over a minute (matplotlib font cache).

Tests are written from first principles (hand-derived signs, hand-written
world vectors), not round trips -- see `vision/HANDOFF.md`, "Fix gate frame
convention", for why.

## RL environment, training and evaluation

```
cd rl
/usr/local/bin/python3.12 train.py --timesteps 5_000_000 --n-envs 512 --seed 0 --name run1 [--resume runs/run0]
tensorboard --logdir runs/run1/tb
/usr/local/bin/python3.12 fly_course.py --track tracks/oval6.json --controller rl --model runs/run1 [--plot out.png]
```
`runs/<name>/` (git-ignored) gets `model.zip`, `vecnormalize.pkl`, `config.json`,
`checkpoints/`, `tb/`. Every 250k steps the current policy flies the three fixed
tracks deterministically from `Course.start_pose()`; `eval/<track>/completed`,
`.../lap_time`, `eval/completed_frac` go to tensorboard. The fixed tracks are
NOT in the training pool by default (random 3-8 gate courses, seeded), so this
is a held-out test; `--train-on-fixed` adds them. PPO: MlpPolicy pi/vf [128,128]
tanh, `log_std_init` -1, n_steps 64, minibatch = n_envs*n_steps/4, 5 epochs,
lr 3e-4, gamma 0.99, GAE 0.95, clip 0.2, VecNormalize on obs and reward, CPU.

Each reset picks a random course and a random START gate and spawns 2-6 m behind
it (noise ranges are `RaceEnvConfig` fields), so every segment is trained.

| | Definition |
|---|---|
| Observation (26) | body-frame velocity /10 (3), world up in the body frame (3; gravity direction, not the full rotation, so heading is not observed), body rates / max_rate (3), previous action (4), current gate then next gate: centre relative to drone in body frame /10 m (3) + gate normal in body frame (3) each, altitude /10 (1). Last gate: next = current. Optional Gaussian noise (std 0.01 in env only). One function, `race_env.build_obs`, used by env and `RlPolicyController`. |
| Action (4) in [-1,1] | thrust = m g (1 + a0) clipped to [0, max]; body rates = a[1:4] * (6, 6, 3) rad/s. Same CTBR command and `QuadSim.step_ctbr` as the PID. |
| Reward | `+1.0 (d_prev - d_now)` to the target gate centre; `+10` gate pass (target advances); `+10` more on finishing the course (success, terminal); `-10` crash (terminal); `-0.01 ||omega||`; `-0.001 ||a - a_prev||^2`; perception `exp(-angle^4)` with weight 0 (off until the camera mount direction is known). |
| Crash | gate/bar collision (`check_gate`), z < 0.05 m, > 30 m from the target gate, tilt > 80 deg. |
| Truncation | 20 s (`max_episode_time`): truncated, not terminated (value bootstraps). |
| Info (on episode end) | `episode {r,l}`, `gates_passed`, `success`/`is_success`, `crash_reason` (none/gate/ground/oob/tilt), `terminal_observation`, `TimeLimit.truncated`. |

Throughput (Apple M2 Pro, CPU, torch default threads, float32): raw `RaceEnv.step`
with random actions ~174k env-steps/s at 512 envs and ~341k at 2048 envs. SB3 PPO
via `train.py` at 512 envs: ~64k fps (collection + updates), i.e. 5M steps in
~80 s. No Python loop over envs on the hot path; the time is spread over the many
small torch ops in `QuadSim` (5 physics substeps per control step).

Not done / open: no policy has been trained or validated yet; camera-aware reward
is off; sensor noise/latency and domain randomisation (mass, drag, thrust) are
not in the env; the gate observation is exact (no vision noise model yet).

## Parameter sources (and what is NOT verified)

| Parameter | Source | Verified? |
|---|---|---|
| mass 1.5 kg, inertia (0.029125, 0.029125, 0.055225), rotor positions | PX4 iris model | **TODO verify against the Iris USD in Isaac** |
| k_f 8.54858e-6, k_m 1e-6, rot_dir [-1,-1,1,1], w in [0,1100] | Pegasus `QuadraticThrustCurve` defaults | read from Pegasus `main` source 2026-10-05; **TODO confirm the copy installed on the cluster**, and that Pegasus's rotor index order matches `rotor_xy` |
| w = 1000 u + 100 | Pegasus `PX4MavlinkBackend` (`input_scaling`, `zero_position_armed`) | read from Pegasus `main` source 2026-10-05; **TODO confirm on the cluster** |
| drag (0.5, 0.3, 0.0) | Pegasus `MultirotorConfig` → `LinearDrag` | read from Pegasus `main` source 2026-10-05; **TODO confirm on the cluster** |
| 250 Hz physics, 50 Hz control | Pegasus PX4 rate; our choice | choice |
| rate-loop gains Kp 40/40/20, Ki 20/20/10, Kd 0 | ours (~25 ms roll/pitch lag, 50 ms yaw) | guess; not PX4's real tuning |
| rate limits 6/6/3 rad/s, collision radius 0.30 m | ours | guess |

Not modelled: rotor inertia torques, ground contact, blade flapping / drag
beyond the linear term, sensor noise and latency, wind. The drone is
`QuadParams`-driven end to end, so **when the team picks the real drone,
replace the numbers in `QuadParams`** (and re-check the rate-loop gains).

## Baseline results (PID, QuadSim, no noise)

All three tracks complete with every gate passed and no crashes; lap times
8.24 s (`straight3`), 18.70 s (`oval6`), 18.88 s (`slalom5`); peak speed
~3.95 m/s (velocity clamp 4 m/s). Lateral offset at each gate crossing stays
below 8 cm (limit 45 cm). The PID is deliberately conservative -- it is the
baseline to beat, not a tuned racer.

## Next steps

1. ~~SB3 PPO environment~~ done (`race_env.py`, `train.py`); next: run training,
   tune rewards/curriculum, compare lap times against the PID and the MPC.
2. Isaac Lab version of the same task on the cluster (Option 2).
3. PX4 offboard deployment via `SET_ATTITUDE_TARGET` in Isaac + PX4
   (`QuadParams.thrust_to_px4_output` gives the collective-thrust mapping).
4. Vision noise model from `vision/gate_error_eval/runs/*/gate_errors.csv`,
   injected into the gate observations.

## Caveat before any MPC comparison

Antek's `control/` model uses z-DOWN equations while its race sim treats z as
height, and uses m = 1 kg (not the Iris's 1.5 kg). Reconcile frames and
parameters (ideally via `QuadParams`) before comparing lap times.

## Training results so far (2026-10-15, Iris params, no noise)

Best policy: `rl/runs/best/` = `ppo_v3` checkpoint at 135M steps (runs/ is
gitignored; copy it around by hand). History:
- `ppo_v1` (0-30M) and `ppo_v1b` (30-90M): plain random courses.
- `ppo_v2` (90-150M): `--yaw-jitter 60 --n-courses 512`.
- `ppo_v3` (125-185M, from the v2 125M checkpoint): `--lr 1e-4 --w-crash 25`.

Late in v2 and v3 the policy keeps getting faster, and it starts clipping
gates on oval6 again. Pick checkpoints by completion rate over 16 noisy starts
(`run_controller(..., n_envs=16, start_noise=0.3)`), not by the last model or
one deterministic eval flight.

| Track (held out) | PID lap | RL lap (best) | RL completion over 16 noisy starts |
|---|---|---|---|
| straight3 | 8.24 s | 3.06 s | 100% |
| oval6 | 18.70 s | 7.46 s | 100% |
| slalom5 | 18.88 s | crashes at gate 3 | 0% |

slalom5 gates face ~65 deg away from the arrival direction, which is a harder
turn-in than the training courses had until `--yaw-jitter` was added. That
alone has not fixed it. Next ideas:
- reward passing near the gate centre;
- start from a slower policy;
- a curriculum on the jitter.
