# Isaac Sim min-snap trajectory test — 2026-10-23

Min-snap planning + closed-loop NMPC flight through the gates spawned in the
live Isaac Sim session (pod `isaac-sim-84895b85dc-pvwjk`).

## Inputs (from Isaac)

- `test_gates.json` — transcribed from `/workspace/test_gates.json`, written
  by `spawn_test_gates.py` at 13:37:05 with the Iris hovering under PX4
  (camera z = 2.61 m, staggered layout, 5 gates kept / 0 dropped). Values
  rounded to 4 decimals (yaw 3) when printed from the Script Editor.
- Start pose = the drone's live `/World/Iris/body` pose:
  position (-0.0314, -0.021, 2.577) m, heading -8.01 deg.

## Command

From `Senior-Design-G1-Autonomous-Drone/Trajectory/`:

```
python simulate_min_snap.py --start -0.0314 -0.021 2.577 --heading -8.01 \
    --gates "isaac_results/test_gates.json" \
    --perception {truth|synthetic} --out-dir "isaac_results/<run>"
```

Defaults otherwise: v_max 4.0 m/s, a_max 6.0 m/s², dt 0.02 s, NMPC horizon 10,
noise on, seed 0, corner noise 0.5 px (synthetic run).

## Results

| | truth_perception | synthetic_perception |
|---|---|---|
| Gate poses fed to planner | true manifest poses | PnP on projected corners (0.5 px noise) |
| Perception pos err (max) | — | 0.021 m (gate_3, 12 m range) |
| Perception normal err (max) | — | 0.9 deg (gate_0) |
| Plan: waypoints / segments | 16 / 15 | 16 / 15 |
| Plan duration | 20.62 s | 20.59 s |
| Plan peak speed / accel | 3.78 m/s / 6.00 m/s² | 3.79 m/s / 6.00 m/s² |
| Ref max roll / pitch | 32.2 / 26.8 deg | 32.5 / 26.8 deg |
| Ref thrust range | 7.21–13.89 N | 7.20–13.92 N |
| Tracking error mean / max / final | 0.281 / 0.661 / 0.374 m | 0.282 / 0.661 / 0.366 m |
| **Result** | **ALL GATES PASSED** | **ALL GATES PASSED** |

Flown path vs true gates (offset in gate plane, margin to the 1.5 m frame,
angle from gate normal):

| Gate | truth: offset / margin / angle / speed | synthetic: offset / margin / angle / speed |
|---|---|---|
| gate_0 | (+0.10, +0.02) m / 0.65 m / 6.2° / 2.68 m/s | (+0.11, +0.03) m / 0.64 m / 6.5° / 2.69 m/s |
| gate_1 | (-0.02, -0.04) m / 0.71 m / 2.5° / 3.17 m/s | (-0.02, -0.04) m / 0.71 m / 2.2° / 3.18 m/s |
| gate_2 | (+0.20, -0.06) m / 0.55 m / 7.8° / 2.20 m/s | (+0.20, -0.06) m / 0.55 m / 8.6° / 2.20 m/s |
| gate_3 | (-0.13, +0.05) m / 0.62 m / 19.4° / 2.90 m/s | (-0.07, +0.04) m / 0.68 m / 20.1° / 2.82 m/s |
| gate_4 | (-0.07, -0.00) m / 0.68 m / 8.0° / 2.10 m/s | (-0.05, +0.01) m / 0.70 m / 8.7° / 2.11 m/s |

## Files

- `*/min_snap_reference.csv` — the 12-state NMPC reference sampled at 0.02 s
  (`t,x,y,z,phi,theta,psi,vx,vy,vz,p,q,r,U1_ff`).
- `*/min_snap_sim.png` — 3D path, top-down, speed, attitude, tracking error.
- `truth_perception/run_log.txt` — full console log (incl. 1 Hz flight trace).
- `synthetic_perception/run_log_summary.txt` — console log without the 1 Hz trace.

## Caveats

- This is the repo's Python sim (RK4 plant + NMPC), not a flight inside
  Isaac/PX4. Only the gate poses and start pose come from Isaac.
- The NMPC/plant use `drone_constraints["mass"]` = 1.0 kg; `rl/HANDOFF.md`
  lists the Isaac Iris at 1.5 kg (still marked TODO verify).
- The staggered test layout is not a race course: gate order forces large
  heading swings (yaw ref ±~120°) and doubling back between gate_0 → gate_1
  and gate_3 → gate_4. gate_3's ~20° crossing angle is the weakest pass.
