# slalom3 live Isaac runs, 2026-10-30

3-gate slalom (rl/tracks/slalom3.json, the team's 3_gates course), PX4 offboard CTBR, one run per controller,
gates from vision (prior noise 0.5 m / 10 deg, seed 0). Front camera recorded per run.

| run | gates | result | tracking error mean / max | peak speed |
|---|---|---|---|---|
| pid_vision | 3/3 | lap 7.18 s | 0.14 / 0.67 m | 3.86 m/s |
| mpc_ctbr_vision | 1/3 | hit gate 1 | 0.31 / 0.74 m | 4.65 m/s |
| rl_vision | 1/3 | timeout (60 s) | 0.71 / 1.77 m | 7.47 m/s |
| mpc_ctbr_truth (extra) | 2/3 | hit gate 2 | 0.29 / 0.81 m | 4.75 m/s |

Files per run: `<run>.png` (top-down, speed, tracking error), `<run>_data.csv` (t, position, velocity, speed,
plan position/speed, error), `<run>_video.mp4` (front camera on top, animated plot below).

Tracking error = distance from the true position to the nearest point of the min-snap plan through the true gates
(same plan for all controllers; optimize_times=False, v 4 / a 6). PID and RL do not follow it in time, so a
time-indexed error would be meaningless for them. Camera and flight-log clocks differ by 19.3 s; the video is aligned by correlating the magenta gate
area in the camera with the distance to the target gate in the log (19.2-19.4 s in every run; checked on the
PID run, where the gate_0 bar leaves the view within about 0.2 s of the logged crossing). Raw logs: rl/runs/isaac/rec_*.
