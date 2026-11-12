# Starling 2 Max in Isaac: calibration and pink 36 in gates, 2026-11-12

Live Isaac Sim + PX4 SITL (pod `fake-isaac`), `workspace/spawn_starling.py` (Max, C29 camera),
`px4/starling2max_sitl_params.sh`, `DRONE_VEHICLE=starling2max`. Gates from ground truth (no vision).
Branch `vehicle/starling2max-tune`. Raw logs: `rl/runs/isaac/max_calib`, `max_pink` (first matrix),
`max_pink2` (rerun), gitignored.

## Calibration (`calibrate.py`)

| | Result |
|---|---|
| origin | PX4 local vs Isaac pose: offset (0, 0, 0.05) m |
| hover | thrust scale **1.0001**; u_hover 0.1385 (model) / 0.1387 (linear fit). The Max thrust chain is right as is |
| rates, 63 % rise | roll **0.077 s**, pitch **0.067 s**, yaw **0.156 s** (QuadSim nominal 0.041 / 0.041 / 0.073) |

The measured rate response sits inside the `DynRand` ranges `best_max` was trained on (roll/pitch 26–106 ms,
yaw 48–172 ms), so no recentring of `rate_kp` / `tau_motor` is needed.

## Pink 36 in gates

The lab gates: 36 in (0.914 m) outer, 1 in pipe, pink. Tracks `rl/tracks/<name>_pink36.json` keep the layouts
of straight3 / oval6 / slalom3 / slalom5 with `side` 0.8636 m, `bar` 0.0254 m and `gate_color` hot pink
(sRGB 255,105,180). `workspace/spawn_track.py` now builds each gate at the track's size and colour.

With the Max's 0.30 m collision radius a pass needs the crossing within **±0.13 m** of the gate centre (±0.45 m
on the 1.5 m gates).

### Offline (`runner.py --sim`, Max profile)

PID and MPC complete all four tracks (worst edge margin 0.35 m; 0.30 m needed).
RL `best_max` fails all four (0–1 gates): it was trained on 1.5 m gates and needs a retrain on the smaller opening.

### Live, rerun with the braking hold (`max_pink2`, 3 runs each)

| Track | PID | PID lap | MPC (`mpc_ctbr`) | MPC lap |
|---|---|---|---|---|
| straight3_pink36 | **3/3** | 8.38–8.40 s | 2/3 (gate 2 hit) | 9.44 s |
| oval6_pink36 | **3/3** | 18.96–18.98 s | 0/3 (tilt crash after 4/6; 2 runs never started) | |
| slalom3_pink36 | **2/3** (1 start-phase fly-away, not a race failure) | 7.26–7.28 s | 0/3 (tilt, gate) | |
| slalom5_pink36 | **3/3** | 19.14–19.18 s | 0/3 (gate, gate, tilt) | |

PID edge margins 0.32–0.34 m (needs 0.30): it passes, but with little room. "Never started" rows aborted on the
geofence before the race because the previous MPC crash left the drone outside it.

## Findings

1. **Post-lap hold flipped the Max (fixed).** The runner used to send a position setpoint at the current position
   right after the last gate, at ~2.3 m/s. PX4 braked so hard that the Max pitched past −85° and flipped in 2 of 3
   runs (first matrix, `max_pink`; every later run aborted on the geofence). `runner.braking_setpoints` now
   brakes along a 1.5 m/s² stopping path: pitch peaked at −36°, no flips in the rerun.
2. **PX4's own position control on the Max is weak in SITL.** Same root as 1: `starling2max_sitl_params.sh`
   scales only the rate gains. In one start phase PX4 flew the drone 14 m the wrong way (slalom3 PID r2). Our
   CTBR controllers bypass this; anything that relies on PX4 position mode (goto, hold, loiter) needs the
   attitude/position gains tuned for the Max, or the real drone's parameter file.
3. **MPC oscillates on the Max live.** PX4 tracks MPC's rate commands; the commands themselves grow into a
   roll/pitch oscillation (±18° at straight3 gate 2, to a tilt crash on oval6). Controller compute is 10–15 ms
   median and 22–39 ms max live, against 3.6–4.5 ms offline; the oscillation grows as compute rises past
   20 ms. On the 1.5 m Iris gates (2026-10-24, MPC 16/18) this margin was never tested this tightly. Next:
   profile the Max NMPC solve on the pod (2-CPU quota), then cap or shorten it.
4. **RL needs the small gates.** Retrain `best_max` on `side` 0.8636 (or randomised side) before flying it.
