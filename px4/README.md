# px4/ — flying the min-snap course in Isaac Sim through PX4

Stage 1 of flying the gate course for real: PX4's own position controller
tracks the min-snap reference, streamed as OFFBOARD position / velocity /
acceleration / yaw setpoints over MAVLink. The repo NMPC is not in the loop
yet (that is stage 2: attitude + thrust setpoints via `SET_ATTITUDE_TARGET`).

| File | What |
|---|---|
| `fly_min_snap_offboard.py` | plan from the live PX4 pose through the manifest gates, then fly it in OFFBOARD |
| `test_fly_min_snap_offboard.py` | self-tests, including the full OFFBOARD sequence against a fake PX4 (`python3 test_fly_min_snap_offboard.py`) |

**Status: untested against the cluster.** The planning, ENU/NED conversion
and OFFBOARD sequence pass against a fake PX4; the real MAVLink link has not
been flown yet.

## Running it

Everything runs **inside the Isaac pod**. PX4 SITL only accepts MAVLink on
localhost, and the pod's containers share localhost. Use the `ros2-eval`
container: it has numpy and OpenCV, and sees the repo on `/workspace`.

1. **Isaac is set up** (see `vision/HANDOFF.md`, "Running the sim test"):
   - `spawn_example.py` has run once.
   - PX4 is connected and the drone is hovering (`commander takeoff`).
   - The gates are spawned, so `/workspace/test_gates.json` is current.
     Spawn the gates while hovering, from the same pose you'll fly from.
2. **Install pymavlink** in the container, once per container:
   ```
   pip install pymavlink
   ```
   If pip tries to install or upgrade numpy, abort: NumPy 2 breaks ROS
   Humble's `cv_bridge`.
3. **Check the plan without flying.** This needs no PX4; use the hover pose
   (ENU, read from `/World/Iris/body`):
   ```
   python3 px4/fly_min_snap_offboard.py --plan-only --start X Y Z --heading DEG
   ```
4. **Fly:**
   ```
   python3 px4/fly_min_snap_offboard.py
   ```
   It prints the live pose and the plan, including the planned gate passes,
   asks `Fly ...? [y/N]`, and then:
   1. streams hover setpoints, then switches to OFFBOARD
   2. hovers at the start for `--hover-s` seconds (default 3)
   3. flies the trajectory
   4. holds the end point
   5. switches to AUTO.LOITER

   **Ctrl-C at any time hands the drone back to AUTO.LOITER.**
5. **Results** go to `/workspace/px4_runs/min_snap_<time>/`:
   - `plan_reference.csv`: the planned path in ENU: t, position, velocity, acceleration, yaw.
   - `flight_log.csv`: every tick's setpoint and the measured state (ENU), with the PX4 mode.
   - `summary.json`: status, gate passes of the flown path, tracking error.
   - `test_gates.json`: a copy of the manifest that was flown.

## Defaults and safety

- **Speed limits:** `--v-max 2 --a-max 2` makes a slow, ~17 s run for the
  3-gate course. The repo sim used 4 m/s and 6 m/s², but PX4's default
  horizontal acceleration limit (`MPC_ACC_HOR`) is about 3 m/s². Go faster
  only after a clean slow run.
- **Refusals:** it won't fly if the drone isn't airborne (`--min-alt`, default
  1 m) or if the plan doesn't pass every gate.
- **Warning:** it warns if the drone has moved more than 1.5 m from where
  the gates were spawned.
- **In-flight aborts:** each of these holds the drone at its measured
  position for 1 s, then switches to AUTO.LOITER:
  - tracking error over `--max-error` (default 1.5 m)
  - PX4 leaving OFFBOARD mode (e.g. someone switched modes)
  - no `LOCAL_POSITION_NED` for `--stale-s` (default 0.5 s)
  - Ctrl-C
- **Gate colliders:** it's not known whether the spawned gates have
  colliders, so treat a miss as a possible crash.

## Frames

- The manifest and the planner use Pegasus world = **ENU**: x east, y north, z up.
- PX4 local = **NED**: x north, y east, z down; yaw clockwise from north.
- They share an origin (measured 2026-10-01; see
  `vision/gate_error_eval/fly_gate_approach.py`), so the conversion is:
  - position: `(e, n, u) -> (n, e, -u)`
  - yaw: `yaw_ned = 90° - yaw_enu`

Before the first flight, check this by comparing the PX4 start pose printed
by the script with `/World/Iris/body` in Isaac.
