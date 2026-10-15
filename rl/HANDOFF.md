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
| `mpc_controller.py` | done | `MpcController`: Antek's NMPC + Trajectory/ min-snap planner on QuadSim, wrench command (`step_wrench`) |
| `compare.py` | done | PID vs MPC vs RL table + plot |
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

## Caveat before any MPC comparison (resolved, see "Three-way comparison")

The z-down issue is already fixed in `control/` (z-up now). Mass is passed via
`m=params.mass`; inertia via the new optional `inertia=` keyword.

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

## Three-way comparison (PID vs NMPC vs RL), 2026-10-15

```
cd rl
/usr/local/bin/python3.12 compare.py --model runs/best [--tracks straight3 oval6 slalom5] \
    [--mpc-vmax 4 8] [--mpc-amax 6] [--envs 16] [--mpc-envs 4] [--out runs/compare]
/usr/local/bin/python3.12 fly_course.py --track tracks/oval6.json --controller mpc [--mpc-vmax 4 --mpc-amax 6]
/usr/local/bin/python3.12 test_mpc_controller.py     # ~1 min (flies the real NMPC on straight3)
```
Writes `results.csv`, `results.md`, `comparison.png` (paths + speed per track). The full run
with `--mpc-vmax 4 8` takes ~50 min, nearly all of it the MPC.

How the MPC is wired: `Course` gates -> `gate_path.Gate` -> `plan_gate_trajectory`
(min-snap, from `course.start_pose()`) -> `TrajectoryReference` -> `create_drone_mpc_optimizer`
(N=10, dt=0.02) solved each step like `Drone_Race_Simulation` (warm start, fall back to the last
IPOPT iterate), output = collective thrust + body torque, flown through `QuadSim.step_wrench`
(mixer + rotors, no rate loop). Only change in `control/`: optional `inertia=None` keyword on
`get_drone_dynamics` / `create_drone_mpc_optimizer` (default unchanged; we pass the Iris inertia).
The reference is time-parametrised (t = step * dt) and the measured yaw is unwrapped to the 2 pi
branch nearest the reference psi.

Caveats:
- All three controllers get the TRUE state (no EKF, no vision noise): isolates control only.
- Different command levels: PID/RL send thrust + body rates (QuadSim's rate loop is underneath),
  the MPC sends thrust + torques. The MPC's model has no drag and no mixer clipping.
- PID is capped at 4 m/s, RL is not, the MPC follows a plan limited by `--mpc-vmax/--mpc-amax`.
  The plan is ACCELERATION-limited here: with the default a_max = 6 the plan peaks at ~3.7 m/s,
  so `--mpc-vmax 8` alone changes nothing on straight3/oval6 (identical rows below). Raise
  `--mpc-amax` too (second table).
- MPC compute is ~100 ms per solve (max ~270 ms) on an M-series CPU vs the 20 ms control period:
  it is NOT real-time as built (IPOPT via CasADi Opti, max 50 iterations). Fine in sim
  (time is simulated), not for deployment. Per-env solves are sequential.
- Noisy-start robustness: start_noise 0.3, seed 0, 16 envs (PID/RL) vs 4 envs (MPC): the MPC
  percentages have a coarse 25% granularity. Min margin = distance from the crossing point to the
  opening edge (>= 0.30 m needed to pass; 0.75 m = dead centre).

Results (`--mpc-vmax 4 8`, a_max 6, 2026-10-15, RL = `runs/best`):

| Track | Controller | Done | Lap [s] | Gates | Crash | Vmax [m/s] | Vmean [m/s] | Min margin [m] | Ctrl ms mean/max | Solver fails | Noisy-start completion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| straight3 | PID | yes | 8.24 | 3/3 | - | 3.95 | 2.43 | 0.73 | 0.28 / 0.6 | - | 100% (16 envs) |
| straight3 | MPC v4 | yes | 9.56 | 3/3 | - | 3.29 | 2.10 | 0.71 | 103.21 / 263.2 | 0 | 100% (4 envs) |
| straight3 | MPC v8 | yes | 9.56 | 3/3 | - | 3.29 | 2.10 | 0.71 | 104.43 / 276.2 | 0 | 100% (4 envs) |
| straight3 | RL | yes | 3.06 | 3/3 | - | 8.89 | 6.55 | 0.55 | 0.32 / 0.6 | - | 100% (16 envs) |
| oval6 | PID | yes | 18.70 | 6/6 | - | 3.93 | 2.56 | 0.68 | 0.29 / 0.3 | - | 100% (16 envs) |
| oval6 | MPC v4 | yes | 22.68 | 6/6 | - | 3.35 | 2.19 | 0.61 | 104.56 / 258.1 | 0 | 100% (4 envs) |
| oval6 | MPC v8 | yes | 22.68 | 6/6 | - | 3.35 | 2.19 | 0.61 | 104.61 / 248.3 | 0 | 100% (4 envs) |
| oval6 | RL | yes | 7.46 | 6/6 | - | 9.86 | 6.62 | 0.31 | 0.33 / 0.8 | - | 94% (16 envs) |
| slalom5 | PID | yes | 18.88 | 5/5 | - | 3.94 | 2.80 | 0.68 | 0.29 / 0.4 | - | 100% (16 envs) |
| slalom5 | MPC v4 | yes | 28.46 | 5/5 | - | 3.73 | 1.95 | 0.62 | 107.25 / 250.2 | 0 | 100% (4 envs) |
| slalom5 | MPC v8 | yes | 24.64 | 5/5 | - | 4.30 | 2.23 | 0.55 | 104.84 / 260.8 | 0 | 100% (4 envs) |
| slalom5 | RL | no | - | 2/5 | gate | 10.95 | 7.08 | 0.28 | 0.33 / 0.8 | - | 0% (16 envs) |

Higher MPC limits (`--mpc-vmax 8 --mpc-amax 15 --mpc-envs 2`, tracks straight3 / oval6):

| Track | Controller | Done | Lap [s] | Gates | Crash | Vmax [m/s] | Vmean [m/s] | Min margin [m] | Ctrl ms mean/max | Solver fails | Noisy-start completion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| straight3 | PID | yes | 8.24 | 3/3 | - | 3.95 | 2.43 | 0.73 | 0.30 / 0.6 | - | 100% (16 envs) |
| straight3 | MPC v8 | yes | 6.08 | 3/3 | - | 5.42 | 3.30 | 0.64 | 104.51 / 257.9 | 0 | 100% (2 envs) |
| straight3 | RL | yes | 3.06 | 3/3 | - | 8.89 | 6.55 | 0.55 | 0.32 / 0.8 | - | 100% (16 envs) |
| oval6 | PID | yes | 18.70 | 6/6 | - | 3.93 | 2.56 | 0.68 | 0.29 / 0.6 | - | 100% (16 envs) |
| oval6 | MPC v8 | yes | 14.38 | 6/6 | - | 5.32 | 3.41 | 0.31 | 105.13 / 263.5 | 0 | 100% (2 envs) |
| oval6 | RL | yes | 7.46 | 6/6 | - | 9.86 | 6.62 | 0.31 | 0.34 / 0.6 | - | 94% (16 envs) |
