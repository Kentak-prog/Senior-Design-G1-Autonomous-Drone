# Rate-loop-aware MPC with raised plan limits (v_max 8, a_max 10), live in Isaac, 2026-11-13

Same setup as `RESULTS_2026-11-13_mpc_rate_pink.md` (Isaac Sim + PX4 SITL, Starling 2 Max, gates from
ground truth, acados backend), with the min-snap plan's limits raised from `v_max` 4 / `a_max` 6 to
**8 m/s / 10 m/s²** (`runner.py --mpc-vmax 8 --mpc-amax 10`). These limits were chosen on the bench
(`bench_mpc_delay.py --v-max 8 --a-max 10`, Isaac rate loop, 20 ms delay): every track passed with tilt
≈ 49°; v10/a14 reached 64°. 3 runs per track, flown in this order:

## Result: 18 / 18

| Track | Runs | Lap (v8/a10) | Lap (v4/a6) | PID 2026-11-12 | Min margin | Max tilt | Max speed | Rate-cmd osc |
|---|---|---|---|---|---|---|---|---|
| three_gate_isaac (1.5 m gates) | **3/3** | 5.50–5.51 s | – | – | 0.69–0.70 m | 52° | 4.6 m/s | 7.2–7.7 °/s |
| straight3_pink36 | **3/3** | 7.32 s | 9.44 s | 8.38–8.40 s | 0.36–0.37 m | 52° | 4.7 m/s | 5.8 °/s |
| oval6_pink36 | **3/3** | 17.50 s | 22.60 s | 18.96–18.98 s | 0.36 m | 51° | 4.7 m/s | 13.8–14.1 °/s |
| slalom3_pink36 | **6/6** ¹ | 5.48–5.50 s | 7.10 s | 7.26–7.28 s (2/3) | 0.38 m | 52° | 4.6 m/s | 7.4–8.6 °/s |
| slalom5_pink36 | **3/3** | 19.04–19.06 s | 28.40 s | 19.14–19.18 s | 0.37 m | 51° | 5.9 m/s | 3.4–3.6 °/s |
| circle3_isaac (1.5 m gates) | **3/3** | 8.50–8.52 s | – | – | 0.68–0.69 m | 51° | 4.7 m/s | 8.1–14.0 °/s |

¹ slalom3 was flown twice (r1–r3, then r4–r6) because the noVNC page reloaded and lost the first video.

- **Faster than PID on all four pink tracks**: −13 % straight3, −8 % oval6, −24 % slalom3, −0.5 % slalom5;
  −22 % to −33 % against the same MPC at v4/a6. Laps match the bench predictions to 0.02 s.
- **Margins unchanged**: pink 36 in gates 0.36–0.38 m (collision radius 0.30 m), 1.5 m gates 0.68–0.70 m.
- **Tilt** peaked at 50–52 ° on every track (33 ° at v4/a6), well inside the runner's 80 ° abort.
- **Controller time** 1.71–1.79 ms mean, ≤ 5.1 ms max per tick: the faster plans cost nothing extra.
- **Rate-command oscillation** (rms of the roll/pitch rate setpoint minus its 0.2 s moving mean) rose from
  1–3 °/s to 3–8 °/s on most tracks; oval6 and circle3 (the tracks with sustained turning) reach ≈ 14 °/s.
  No effect on margins or repeatability. Still not investigated (see the v4/a6 results' oval6 note).
- **Max speed** stays at 4.6–4.7 m/s on the short tracks (segments too short to reach 8 m/s); slalom5's
  8 m legs reach 5.9 m/s.
- **Repeatability**: laps within 0.02 s, margins within a few mm across runs.

Videos (not in git), user's Downloads, one per track, main viewport on the left and the drone's front
camera (`/iris_0/front_cam`) on the right: `isaac_mpc_rate_v8a10_<track>.webm`. Raw logs on the pod:
`/workspace/runs_mpc_rate/mpc_rate_v8a10_<track>_r<n>/` and `<...>.txt`.

## Run command (`ros2-eval`, environment as in `RESULTS_2026-11-13_mpc_rate_pink.md`)

```bash
/workspace/venvs/race/bin/python runner.py --track ../tracks/<track>.json --controller mpc_rate \
  --mpc-vmax 8 --mpc-amax 10 --origin-offset 0 0 0.05 --thrust-scale 1.0 --max-time 90 \
  --out-dir /workspace/runs_mpc_rate/mpc_rate_v8a10_<track>_r<n>
```

Track starts used with `recover.py` before each track's first run: three_gate_isaac / slalom3 (1.2, 0.02,
3.37), straight3 (4.0, 0, 2.5), oval6 (10, −4, 2), slalom5 (4.24, 5.37, 2.5), circle3 (0.94, −0.17, 2.6).
