# Update 2026-10-23: first PX4 flight of the min-snap course in Isaac Sim

## Result

The Iris flew the 3-gate min-snap course in Isaac Sim under PX4 OFFBOARD
control and passed all three gates. Recording:
`C:\Users\aaron\Downloads\isaac_px4_flight_152804.webm` (114 s, 720×500,
~10 MB; kept out of git).

| Gate | Offset from centre | Frame margin | Angle off normal | Speed |
|---|---|---|---|---|
| gate_0 | (+0.00, +0.02) m | 0.73 m | 1.4° | 1.28 m/s |
| gate_1 | (+0.08, +0.00) m | 0.67 m | 1.7° | 1.30 m/s |
| gate_2 | (−0.07, −0.01) m | 0.68 m | 1.1° | 1.49 m/s |

- **Tracking error:** 0.27 m mean, 0.59 m max, measured from the streamed setpoint.
- **Run:** a 3.9 m reposition back to the spawn pose, then a 17.4 s course
  (2 m/s, 2 m/s² limits).
- **Timing:** played at `--time-scale 0.44`, so the course took 39.6 s of wall time.
- **Controller:** PX4's own position controller tracks the min-snap reference.
  The repo NMPC is **not** in the loop yet; that's stage 2.

Plot: `3_gates/px4_flights/20261023_152804_all_gates_passed/px4_flight.png`.

## Where everything is

Everything below is on `main`. It was developed on `px4-offboard-min-snap`
(`1ac34e8` → `e02ee05`) and merged in PR #2.

| What | Where |
|---|---|
| Flight script, tests (22 pass), README | `px4/` |
| This update, all results below | `Trajectory/isaac_results/` |
| Flight video | `C:\Users\aaron\Downloads\isaac_px4_flight_152804.webm` (not in the repo) |

Inside `Trajectory/isaac_results/`:

| Path | Contents |
|---|---|
| `truth_perception/`, `synthetic_perception/`, `README.md` | Python-sim min-snap + NMPC on the earlier 5-gate staggered course |
| `3_gates/README.md` | index of the 3-gate folder |
| `3_gates/course/` | `test_gates.json`: the 3-gate course + spawn hover pose, as transcribed from Isaac. `pod_test_gates.json`: the same, copied straight from the pod, with the camera rotation |
| `3_gates/python_sim/` | min-snap plan in camera orientation (`min_snap_plan.png`), and the repo NMPC flying it in the Python sim (`mpc_course.png`, cached `mpc_log.npz`) |
| `3_gates/px4_flights/20261023_152804_all_gates_passed/` | **the successful Isaac/PX4 flight**: `flight_log.csv` (every 20 ms: setpoint + measured), `plan_reference.csv`, `summary.json`, `px4_flight.png` |
| `3_gates/px4_flights/20261023_150758_aborted/`, `20261023_151426_aborted/` | the two aborted attempts (see "What went wrong") |
| `3_gates/scripts/` | `plot_3_gates.py`, `plot_px4_flight.py` (redraws `px4_flight.png` for any run folder), `record_novnc.js` (the noVNC screen recorder used for the video) |

## How to recreate the flight and the video

You need cluster access from WSL. Inside the pod, PX4 and `ros2-eval` share
localhost.

```bash
export KUBECONFIG=/mnt/c/Users/aaron/Downloads/jup19-rke2-caai.yaml
```

The Isaac GUI is noVNC on port 6080 through a `kubectl port-forward` (see
`vision/HANDOFF.md`). Open http://localhost:6080/vnc.html and connect.

### 1. Isaac: scene, drone, PX4, hover

Follow `vision/HANDOFF.md` → "Running the sim test", on a **fresh** Isaac
process:

1. **Ground.** Add a ground. We used the Flat Grid environment.
2. **Drone.** Window → Script Editor, run **once**:
   `exec(open("/workspace/scripts/spawn_example.py").read())`.
   Expect `[init] Done`. It starts the timeline itself, and Pegasus then
   logs `Waiting for first hearbeat` until PX4 connects.
3. **PX4.** Start it in `px4-sitl` (the command is in HANDOFF). Wait for
   `Simulator connected on TCP port 4560` and `Ready for takeoff!`, then at
   the `pxh>` prompt run `commander takeoff`. The drone hovers at about 2.5 m.

### 2. Isaac: the 3 gates

With the drone hovering, run this in a **new** Script Editor tab. It uses the
pod's own `spawn_test_gates.py` with only the gate table replaced:

```python
ns={"__name__":"sg3"};exec(open("/workspace/scripts/spawn_test_gates.py").read(),ns);ns["GATE_SPECS"]=[(5.0,-14.0,-6.0,15.0),(9.0,14.0,-8.0,-15.0),(13.0,2.0,-10.0,0.0)];ns["main"]()
```

- **Gate specs:** each entry is (range m, azimuth deg, elevation deg,
  yaw-relative deg) from the live camera.
- **Expect:** `wrote /workspace/test_gates.json (3 gate(s) kept, 0 dropped)`.
  The gates land at about 5, 9 and 13 m ahead, 3.4–3.8 m high, in a
  left/right slalom.
- **Pose matters:** gate positions depend on the hover pose at that moment,
  so re-spawning later gives slightly different numbers.

### 3. Isaac: camera and window layout for the video

1. **Main viewport camera.** In a Script Editor tab, aim the viewport down
   the course from behind and above the start:
   ```python
   from isaacsim.core.utils.viewports import set_camera_view;set_camera_view(eye=[-7.0,4.0,7.5],target=[8.5,-1.5,3.4],camera_prim_path="/OmniverseKit_Persp")
   ```
2. **Drone's view.** Resize the `/iris_0/front_cam` window to about 610×380
   by dragging its bottom-right corner, then drag it by its title bar to the
   bottom-right of the Isaac window. **Don't close it**; it belongs to the
   camera that `spawn_example.py` set up.
3. **Clear the view.** Close the Script Editor so it doesn't cover the
   viewport.

The Isaac window filled the top-left 1440×1000 of the 1920×1080 VNC screen,
and the recorder crops to that area. If yours differs, change `SRC_W`/`SRC_H`
in `record_novnc.js`.

### 4. `ros2-eval`: get the flight script

```bash
kubectl exec -it deploy/isaac-sim -n fake-isaac -c ros2-eval -- bash
```

Inside the container:

```bash
cd /workspace/Senior-Design-G1-Autonomous-Drone
git fetch && git checkout main && git pull
pip install pymavlink        # once per container; ABORT if pip tries to upgrade numpy
python3 px4/test_fly_min_snap_offboard.py    # expect: All tests passed. (22)
```

### 5. Start recording

In the noVNC browser tab:

1. Press F12 → Console.
2. Paste the whole of `3_gates/scripts/record_novnc.js` and press Enter.
3. Run:
   ```js
   startRec()
   ```

### 6. Fly

In `ros2-eval`:

```bash
python3 px4/fly_min_snap_offboard.py --time-scale 0.40 --reposition
```

**Time scale:** use the sim speed the previous run printed (`measured sim
speed: 0.40x real time`). We flew 0.44 and measured 0.40 afterwards, so 0.40
should track a little better.

**Before answering `y`, check:**
- **`[start]`:** shows `armed=True`.
- **`[reposition]`:** the start is about (−0.04, 0.00, 2.56), and every gate
  clearance is above 0.6 m.
- **`[plan]`:** about 17.4 s, with all 3 gates at offset 0.00.

**The sequence after `y`** (about 70 s):
1. hold in place, then fly back to the start (about 18 s)
2. settle and hover 3 s
3. fly the course (about 40 s)
4. hold the end point 3 s
5. AUTO.LOITER

Ctrl-C at any point hands control back to AUTO.LOITER.

### 7. Stop and save the video

When the `[flight]` summary prints, run this in the browser console:

```js
stopAndSave("isaac_px4_flight.webm")
```

The browser downloads it, usually to Downloads.

### 8. Copy the logs and plot

The pod folder name, `min_snap_<id>`, is printed on the last `[flight]` line.
Copy it into `px4_flights/` as `<id>_<outcome>`, e.g.
`20261023_152804_all_gates_passed`. From WSL:

```bash
kubectl cp -c ros2-eval fake-isaac/<pod>:/workspace/px4_runs/min_snap_<id> "Trajectory/isaac_results/3_gates/px4_flights/<id>_<outcome>"
```

`<pod>` comes from `kubectl get pods -n fake-isaac`; today it was
`isaac-sim-84895b85dc-pvwjk`.

Then, from `Trajectory/`:

```bash
python "isaac_results/3_gates/scripts/plot_px4_flight.py" "isaac_results/3_gates/px4_flights/<id>_<outcome>"
```

## What went wrong on the way (and what was fixed)

All fixes are on `main` (merged from `px4-offboard-min-snap`).

1. **The dry run printed `armed=False` while the drone was flying.** The
   status was read before any heartbeat had arrived, and the first heartbeat
   came from component 0, not the autopilot. A stray heartbeat could also
   have read as "PX4 left OFFBOARD" mid-flight.
   - **Fix (`0fde721`):** trust only the PX4 autopilot's heartbeat, wait for
     it, refuse if disarmed, and address commands to component 1.
2. **Abort at t=2.2 s, 1.52 m behind** (`px4_flights/20261023_150758_aborted`). Isaac ran at
   **0.42× real time**, while PX4 (lockstep off) and the setpoints ran on the
   wall clock, so the setpoint outran the drone. The direction, frames and
   hover were all fine.
   - **Fix (`81266ba`):** `--time-scale`, plus a printed sim-speed measurement
     after every run.
3. **Abort again** (`px4_flights/20261023_151426_aborted`). That run was started without
   `--time-scale` and measured 0.44×. Each abort's overshoot also left PX4
   loitering 1.35 m in front of gate_0, and the plan from there crept through
   gate_0 at 0.12 m/s.
   - **Fix (`7146eb0`):** `--reposition`. It recovers the spawn hover pose
     from the manifest's camera pose (within 4 cm of Isaac), checks the path
     back against every gate frame bar, enters OFFBOARD holding the drone in
     place, and flies back slowly before the course.
4. **Successful run** (`px4_flights/20261023_152804_all_gates_passed`), with `--time-scale 0.44 --reposition`.

## Known issues and next steps

- **The sim runs at ~0.40× real time with PX4 lockstep off.** `--time-scale`
  works around it, but the value changes with Isaac's frame rate (~28 FPS
  today). The real fix is lockstep, or getting Isaac to real time (fewer
  cameras or lower render load).
- **Mass mismatch.** `control/drone_constraints.py` uses 1.0 kg, but
  `rl/HANDOFF.md` lists the Isaac Iris at 1.5 kg (not yet verified). It
  doesn't matter for this stage, where PX4 does the control, but it does
  for stage 2.
- **Gate colliders.** It's unknown whether the spawned gates have colliders;
  treat a miss as a possible crash.
- **noVNC.** It froze twice under heavy recording (full-size, 30 fps, VP9).
  Use the lighter settings in `record_novnc.js`. Kit menus open on
  mouse-*press* in noVNC.
- **Faster runs.** Raise `--v-max/--a-max` towards 3 once runs are repeatable.
  PX4's default horizontal accel limit `MPC_ACC_HOR` is about 3 m/s².
- **Stage 2:** NMPC in the loop, through PX4 attitude + thrust setpoints
  (`SET_ATTITUDE_TARGET`).
- **Stage 3:** plan from live gate detections (`eval_isaac_gates.py` →
  `min_snap_node.py`) instead of the manifest.
