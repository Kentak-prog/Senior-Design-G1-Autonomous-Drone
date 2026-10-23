# 3-gate Isaac Sim course: results

The 3-gate slalom spawned in Isaac Sim on 2026-10-23. The full story and the
recreate steps are in `../PX4_FLIGHT_UPDATE_2026-10-23.md`.

```
3_gates/
  course/        the gate course
  python_sim/    the repo's Python simulator (plan + NMPC), no Isaac in the loop
  px4_flights/   real Isaac Sim + PX4 OFFBOARD flights (logs copied from the pod)
  scripts/       plotting and recording tools
```

## course/

| File | |
|---|---|
| `test_gates.json` | 3 gates + the drone's spawn hover pose, transcribed from Isaac's Script Editor (values rounded) |
| `pod_test_gates.json` | the same manifest copied straight from `/workspace/test_gates.json` on the pod; includes the camera rotation that `--reposition` uses |

## python_sim/

Synthetic vision (0.5 px corner noise) → min-snap plan → the repo NMPC, in
`Trajectory/simulate_min_snap.py`'s Python sim.

| File | |
|---|---|
| `min_snap_plan.png` | the plan, in camera orientation (forward up, drone's left on the left) |
| `min_snap_reference.csv`, `plan_run_log.txt` | the plan as an NMPC reference, and its console log |
| `mpc_course.png` | NMPC flight of the plan: all gates passed, 0.34 m mean / 0.72 m max tracking error |
| `mpc_log.npz`, `mpc_run_log.txt` | the cached NMPC flight (lets the plots redraw without re-flying) and its log |
| `min_snap_plan_v1_unrotated.png` | the first version of the plan figure, in world axes (superseded) |

## px4_flights/

Each folder is one run of `px4/fly_min_snap_offboard.py`, copied from
`/workspace/px4_runs/min_snap_<id>/` (same `<id>`). Each one has
`flight_log.csv` (every 20 ms: setpoint and measured state, ENU),
`plan_reference.csv`, `summary.json` and the manifest that was flown.

| Run | Flags | Outcome |
|---|---|---|
| `20261023_150758_aborted` | none | aborted at t=2.24 s, 1.52 m behind: the sim ran at 0.42x real time |
| `20261023_151426_aborted` | none (forgot `--time-scale`) | aborted at t=1.82 s, 1.51 m behind; sim at 0.44x |
| `20261023_152804_all_gates_passed` | `--time-scale 0.44 --reposition` | **all 3 gates passed**, 0.27 m mean / 0.59 m max error; `px4_flight.png` |

Video of the successful run: `C:\Users\aaron\Downloads\isaac_px4_flight_152804.webm` (not in the repo).

## scripts/

Run from `Senior-Design-G1-Autonomous-Drone/Trajectory`.

| Script | |
|---|---|
| `plot_3_gates.py` | redraws `python_sim/min_snap_plan.png` and `mpc_course.png` from the cached flight. `--mpc` re-flies the NMPC (~4 min). |
| `plot_px4_flight.py [run folder]` | draws `px4_flight.png` for a PX4 run. Defaults to the successful run. |
| `record_novnc.js` | paste into the browser console on the noVNC page: `startRec()`, then `stopAndSave("name.webm")` |

```
python "isaac results/3_gates/scripts/plot_3_gates.py"
python "isaac results/3_gates/scripts/plot_px4_flight.py" "isaac results/3_gates/px4_flights/<run>"
```
