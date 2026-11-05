# Starling 2 Max in Isaac Sim: first test

**Goal.** Check that the simulated Max hovers like the real one did on
2026-11-03. Only after that do we fly courses with it.

## Bring-up

The order is the same as the Iris bring-up, but `spawn_starling2max.py`
replaces `spawn_example.py`:

1. Script Editor: ground plane, then `spawn_track.py` (any track), then
   `workspace/spawn_starling2max.py`.
   - Keep the log lines starting `[starling2max]` and `[init]`.
   - `4 rotor joint anchors updated` must appear. If it doesn't, the rotors
     kept the Iris arms.
2. Start PX4 SITL as usual (`PX4_SIM_MODEL=none_iris`). Before takeoff, run
   the parameter script in the PX4 container:
   ```bash
   sh px4/starling2max_sitl_params.sh
   ```
3. Start every controller and bridge process with
   `DRONE_VEHICLE=starling2max`.

## Checks (in order; stop at the first that fails)

| # | Do | Pass when |
|---|---|---|
| 1 | Arm on the ground, don't take off | it stays put; `px4-listener actuator_motors` shows about 0 (idle 2,000 RPM) |
| 2 | `px4-commander takeoff`, hover 30 s | climbs and holds without a growing wobble |
| 3 | `px4-listener actuator_motors` in hover | all four about **0.34** (5,407 RPM, the real hover) |
| 4 | `px4-listener hover_thrust_estimate` | `hover_thrust` about **0.14** |
| 5 | Yaw 90° in place (QGC or `px4-commander`) | turns and stops without drifting off |
| 6 | 3-gate PX4 position course (as for the Iris) | completes |

## If a check fails

- **Check 2 wobbles or oscillates.** SITL still uses the Iris rate gains,
  and the Max has much more control authority for its inertia. Halve
  `MC_ROLLRATE_P` and `MC_PITCHRATE_P`, and their `_D`.
- **Check 3 is far from 0.34.** The mass or k_f did not apply. Look for
  `body mass ... = 0.573 kg` in the spawn log.
- **Check 4 is far from 0.14.** `THR_MDL_FAC` is not 0.9. Re-run the
  parameter script.

Send me the values from checks 3 and 4 and the `[starling2max]` spawn log
lines, and I'll compare them with the model. They are the Phase 3 validation.
