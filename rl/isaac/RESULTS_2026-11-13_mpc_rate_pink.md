# Rate-loop-aware MPC on the pink 36 in gates, live in Isaac, 2026-11-13

Same setup as `RESULTS_2026-11-12_max_pink.md`: Isaac Sim + PX4 SITL (pod `fake-isaac`), Starling 2 Max
(`workspace/spawn_starling.py`, `px4/starling2max_sitl_params.sh`), gates from ground truth (no vision),
`runner.py --origin-offset 0 0 0.05 --thrust-scale 1.0 --max-time 90`. The controller is
`--controller mpc_rate` (`mpc_rate.py`, acados backend, `--mpc-delay-ms 0`; see `README_mpc_rate.md`).
3 runs per track, back to back, `recover.py` to each track's start before its first run.

## Result: 12 / 12

| Track | `mpc_rate` (this run) | lap | min margin | max tilt | rate-cmd osc | `mpc_ctbr` 2026-11-12 | PID 2026-11-12 |
|---|---|---|---|---|---|---|---|
| straight3_pink36 | **3/3** | 9.44 s | 0.365–0.370 m | 33° | 1.7–1.8 °/s | 2/3 (gate 2 hit) | 3/3, 8.38–8.40 s |
| oval6_pink36 | **3/3** | 22.60 s | 0.363–0.367 m | 33° | 11.3–11.9 °/s | 0/3 (tilt crash after 4/6) | 3/3, 18.96–18.98 s |
| slalom3_pink36 | **3/3** | 7.10 s | 0.370–0.373 m | 33° | 2.6–2.7 °/s | 0/3 (tilt, gate) | 2/3, 7.26–7.28 s |
| slalom5_pink36 | **3/3** | 28.40 s | 0.367–0.369 m | 27° | 0.9 °/s | 0/3 (gate, gate, tilt) | 3/3, 19.14–19.18 s |
| **Total** | **12/12** | | | | | **2/12** | **11/12** |

- **Margin** = distance from the crossing point to the opening's edge; > 0.30 m (the collision radius) is
  needed. `mpc_rate` crossed every gate with 0.36–0.37 m, more room than PID (0.32–0.34 m).
- **Controller time:** 1.7–1.8 ms mean, 4.6–5.2 ms max per tick, on the pod's 2-CPU `ros2-eval`
  (the old MPC: 10–15 ms median, 22–39 ms max live). Real-time factor 0.53–0.56, as before.
- **Repeatability:** the three runs of each track agree to the 0.01 s in lap time and to a few mm in margin.
- **Lap times** are set by the min-snap plan (`v_max` 4, `a_max` 6, distance-proportional segment times),
  not by the controller; PID flies its own faster profile on three of the tracks. Raising the plan's limits
  is the next step for speed.
- **oval6** has the largest rate-command oscillation (≈12 °/s rms of the roll/pitch rate setpoint minus its
  0.2 s moving mean; 1–3 °/s elsewhere) without hurting margins or tilt. Not yet investigated.

Videos (main viewport, one per track, 3 runs each; not in git): `isaac_mpc_rate_<track>.webm` in the
user's Downloads. Raw logs on the pod: `/workspace/runs_mpc_rate/mpc_rate_<track>_r<n>/` (`log.csv`,
`summary.json`) and `<...>.txt` (runner output).

## Pod setup used (persistent, under /workspace)

- acados `03e5311` built in `/workspace/acados` (`ros2-eval`: gcc, cmake), renderer in `/workspace/acados/bin`.
- `acados_template` + `deprecated`, `wrapt`, `scipy==1.13.1` in `/workspace/pylib_acados`, used with
  Tony's `/workspace/venvs/race` through `PYTHONPATH` (that venv is not modified).
- Generated solver code in `/workspace/acados_codegen`.
- Each run (`ros2-eval`):
  ```bash
  export OMP_NUM_THREADS=1 DRONE_VEHICLE=starling2max PYTHONPATH=/workspace/pylib_acados \
    ACADOS_SOURCE_DIR=/workspace/acados LD_LIBRARY_PATH=/workspace/acados/lib ACADOS_CODEGEN_DIR=/workspace/acados_codegen
  cd /workspace/Senior-Design-G1-Autonomous-Drone/rl/isaac
  /workspace/venvs/race/bin/python runner.py --track ../tracks/<track>.json --controller mpc_rate \
    --origin-offset 0 0 0.05 --thrust-scale 1.0 --max-time 90 --out-dir /workspace/runs_mpc_rate/<name>
  ```
- Isaac: Flat Grid, `spawn_starling.py`, Tony's poller (`kit_tools/r.py`); tracks spawned and the
  viewport camera set by queuing scripts for the poller (`spawn_track.py`, `set_camera_view`).
