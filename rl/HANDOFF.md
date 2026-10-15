# RL Module Handoff

For whoever (or whatever) picks up the RL stack next. Scope: `rl/` only.
Read the module docstrings in `quad_sim.py` and `course.py` before touching
a sign or a frame -- they explain *why*.

Last updated: 2026-10-05.

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
| `fly_course.py` | done | runner + CLI + optional matplotlib plot |
| `test_*.py` | done | plain-function tests, no pytest |

## Running

```
cd rl
/usr/local/bin/python3.12 test_quad_sim.py
/usr/local/bin/python3.12 test_course.py
/usr/local/bin/python3.12 test_pid_controller.py
/usr/local/bin/python3.12 fly_course.py --track tracks/oval6.json --controller pid [--plot out.png]
```
Each test file ends with `All tests passed.`; the whole suite takes ~10 s.
The first `--plot` can take over a minute (matplotlib font cache).

Tests are written from first principles (hand-derived signs, hand-written
world vectors), not round trips -- see `vision/HANDOFF.md`, "Fix gate frame
convention", for why.

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

1. Gymnasium + SB3 PPO environment on this simulator (Option 1): observation
   = state + next-gate(s) in body frame, action = CTBR, reward = gate
   progress/pass, collision/ground penalty. `QuadSim.reset(env_ids, ...)` and
   `random_course` are built for this.
2. Isaac Lab version of the same task on the cluster (Option 2).
3. PX4 offboard deployment via `SET_ATTITUDE_TARGET` in Isaac + PX4
   (`QuadParams.thrust_to_px4_output` gives the collective-thrust mapping).
4. Vision noise model from `vision/gate_error_eval/runs/*/gate_errors.csv`,
   injected into the gate observations.

## Caveat before any MPC comparison

Antek's `control/` model uses z-DOWN equations while its race sim treats z as
height, and uses m = 1 kg (not the Iris's 1.5 kg). Reconcile frames and
parameters (ideally via `QuadParams`) before comparing lap times.
