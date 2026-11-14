# Rate-loop-aware MPC (`mpc_rate.py`) and the delay bench (`bench_mpc_delay.py`)

## The problem

Live in Isaac on the Starling 2 Max with the pink 36 in gates (`RESULTS_2026-11-12_max_pink.md`),
`MpcCtbrController` completed 2 of 12 runs: its roll/pitch rate commands grew into an oscillation and
ended in tilt crashes, worse as the solve time rose past 20 ms.

Cause, reproduced offline: that MPC plans with a torque input and sends the body rate it predicts **one
20 ms step ahead**, so it assumes the drone reaches a commanded rate within a tick. The Max's rate loop
in Isaac takes **77 / 67 / 156 ms** (roll / pitch / yaw, 63 % rise). Each correction lands 3–4 ticks
late and overshoots; the MPC, re-solving from the measured state, reverses, late again. A sudden large
demand then flips it. Command delay (solve + transport) adds to the same lag.

## The bench (`bench_mpc_delay.py`)

QuadSim with the Starling 2 Max profile, plus:
- the rate loop slowed to Isaac's measured response (`rate_kp` × `KP_SCALE_ISAAC` = 0.39 / 0.51 / 0.30
  → 80 / 64 / 152 ms; QuadSim's own loop is 44 / 44 / 76 ms), and
- a command delay applied at the 4 ms physics resolution.

Metrics: gates, crash, lap time, max tilt, rate-setpoint oscillation (rms of the roll/pitch setpoint
minus its 0.2 s moving mean), minimum gate edge margin (must exceed the 0.30 m collision radius), solve
time.

```
DRONE_VEHICLE=starling2max python bench_mpc_delay.py calib
DRONE_VEHICLE=starling2max python bench_mpc_delay.py sweep --controller ctbr --delays-ms 0 20 40
DRONE_VEHICLE=starling2max python bench_mpc_delay.py sweep --controller rate --told-ms 0
```

### `MpcCtbrController` (2026-11-12)

| Track (live result) | QuadSim's loop | Isaac's loop |
|---|---|---|
| straight3_pink36 (2/3) | done at 0–40 ms | done at 0–40 ms, tilt 44° at 40 ms |
| oval6_pink36 (0/3) | done at 0–60 ms | done at 0–20 ms; **crash** at 30, 40, 50, 60 ms |
| slalom5_pink36 (0/3) | done at 30–50 ms; crash at 60 ms | **crash at gate 1 at every delay, 0 ms included** |

## The controller (`mpc_rate.py`, `RateAwareMpc`)

Drop-in for `MpcCtbrController` (same `act()`, `command = "ctbr"`, same min-snap plan and reference).
- **Model:** position, velocity, quaternion attitude, body rates, thrust; PX4's rate loop as a first-order
  lag per axis (`tau_w`, default 80 / 64 / 152 ms) and the motor lag (35 ms); linear body drag.
- **Output:** the first input itself, `[thrust, body-rate setpoint]`. No one-step-ahead rate or
  attitude-lead trick.
- **Horizon:** 20 × 50 ms = 1 s (was 10 × 20 ms = 0.2 s).
- **Delay compensation:** `delay_s` predicts the state forward with the commands already sent.

### Results, Isaac's rate loop unless noted (2026-11-12, IPOPT backend)

| Case | straight3 | oval6 | slalom5 |
|---|---|---|---|
| 0 ms | done, osc 1.4 | done, 0.9 | done, **0.5** |
| 20 / 40 / 60 ms, compensated | done, 2.1 / 2.9 / 3.6 | done, 1.4 / 1.8 / 2.3 | done, 0.7 / 0.9 / 1.1 |
| 40 ms, MPC told 0 ms | done, 5.4 | done, 3.4 | done, 1.6 |
| QuadSim's 2× faster loop (MPC still assumes Isaac's), 0 ms | done, 1.6 | done, 1.0 | done, 0.6 |
| same, 40 ms compensated | done, 6.9 | done, 5.9 | done, 4.1 |

osc in °/s. 21 / 21 complete, max tilt 27–37°, gate edge margins 0.43 m (centred) where measured. Lap
times equal the old MPC's (9.44 / 22.60 / 28.40 s): they are set by the time-parametrised plan.

IPOPT takes 18–72 ms per solve on a laptop (20 ms tick): not real time. The bench applies delay
explicitly, so these results did not need it. Use the acados backend.

## acados backend (`mpc_rate_acados.py`, 2026-11-13)

Same model, limits and cost; SQP-RTI (one Gauss-Newton step per tick) with HPIPM on generated C. The
commands are 4 extra states and the optimizer's input is their rate of change ("delta-u"), which gives
the stage-to-stage input-change penalty. A first version with the commands as inputs chattered at
100–190 °/s on the wrong-model cases (more iterations did not help); delta-u fixed it.

| Case (Isaac's loop unless noted) | straight3 | oval6 | slalom5 |
|---|---|---|---|
| 0 ms | done, osc 1.3 | done, 0.8 | done, 0.4 |
| 20 / 40 / 60 ms, compensated | done, 1.6 / 2.2 / 3.0 | done, 1.0 / 1.4 / 1.9 | done, 0.5 / 0.7 / 1.0 |
| 40 ms, MPC told 0 ms | done, 3.3 | done, 2.1 | done, 1.0 |
| 2× faster loop, 0 ms | done, 1.2 | done, 0.7 | done, 0.4 |
| same, 40 ms compensated | done, 2.2 | done, 1.4 | done, 0.7 |

21 / 21, every case equal or better than IPOPT, margins 0.43 m, tilt ≤ 36°. **Solve 0.8 ms median, ≤ 1.6 ms
p95** (WSL, one thread) against 18–72 ms for IPOPT. The generated code is plain C, so it builds for the
VOXL 2's ARM cores the same way.

### Setup (WSL / Ubuntu; once)

```bash
sudo apt install -y build-essential cmake python3-venv
git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/acados/acados.git ~/acados
mkdir -p ~/acados/build && cd ~/acados/build && cmake -DACADOS_WITH_QPOASES=OFF .. && make -j8 install
python3 -m venv ~/venvs/acados && . ~/venvs/acados/bin/activate
pip install -e ~/acados/interfaces/acados_template casadi numpy scipy matplotlib opencv-python-headless
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
Each shell: `. ~/venvs/acados/bin/activate; export ACADOS_SOURCE_DIR=~/acados LD_LIBRARY_PATH=~/acados/lib`.
The first solver build renders and compiles C code into `~/.cache/acados_rate_mpc/<hash>/` (seconds),
reused afterwards; the first use downloads acados' `t_renderer`. Built and tested with acados `03e5311`.
Don't run git from WSL on this checkout (it reports every file as modified: file modes and line endings).

## Live in Isaac (2026-11-13): 12 / 12

`runner.py --controller mpc_rate` on the four pink 36 in tracks, 3 runs each: **12/12**, margins
0.36–0.37 m, tilt ≤ 33°, 1.7–1.8 ms per tick on the pod (old MPC: 2/12, PID: 11/12 with 0.32–0.34 m).
Details, pod setup and the run command: `RESULTS_2026-11-13_mpc_rate_pink.md`.

### Faster plans, v_max 8 / a_max 10 (2026-11-13): 18 / 18

`--mpc-vmax 8 --mpc-amax 10` on the four pink tracks plus `three_gate_isaac` and `circle3_isaac`: all
runs complete, laps 22–33 % shorter than at v4/a6 and faster than PID on every pink track, margins
unchanged, tilt ≤ 52°, rate-command oscillation up to ≈ 14 °/s on the turning tracks.
`RESULTS_2026-11-13_mpc_rate_v8a10.md`.

### On the VIO (2026-11-13): 1 / 18

`runner.py --state vio` flies the MPC on `ros2_nodes/vio_node.py`'s odometry (PX4's estimate only scores
and guards the run). One run completed (3-gate course, 3/4, VIO within 0.31 m); the rest aborted on VIO
drift or hit gates. The VIO falls behind along track because a ~3° tilt error from its first PnP fix is
never corrected, and Tony's thin pink gates give it poor fixes. Details and the proposed VIO fixes:
`RESULTS_2026-11-13_mpc_rate_vio.md`.

### Next-gate yaw on the VIO (2026-11-14): 5 / 6

`RateAwareMpc(..., yaw_mode="next_gate")` (runner `--mpc-yaw next_gate`) keeps the next gate in the front
camera. With the Swift-mode VIO it completed all three speed steps on the 3-gate course (the 8/10 lap in
5.48 s, as on GPS) and 2 of 3 on the circle. `RESULTS_2026-11-13_mpc_rate_vio.md`, follow-up 3.

## Not yet tested

Delay jitter, and the VOXL 2 itself. On the VIO: anything past one lap at 3/4 (see above).

## Tests

`DRONE_VEHICLE=starling2max python test_mpc_rate.py` (IPOPT, ~2 min) or with `MPC_RATE_BACKEND=acados`
(~25 s): the bench's rate loop matches Isaac's, `MpcCtbrController` crashes slalom5 with it (the
reproduction), and `RateAwareMpc` flies slalom5 at 0 ms and straight3 at 40 ms (compensated and not) with
margins above the collision radius; on acados also a solve-time p95 under 5 ms.
