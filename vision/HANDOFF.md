# CV Pipeline Handoff

For whoever (or whatever) picks up computer vision work on this repo next.
Scope: `vision/` only. If you're Claude Code, read this whole file before
touching anything in this directory — several of the decisions below are
non-obvious and easy to silently undo.

Last updated: 2026-10-01.

## Project context (one paragraph)

Senior design project building an autonomous racing drone that flies
through gates using vision-based pose estimation. Three control paradigms
are being compared (PID, MPC, RL); this directory only concerns the
perception stack feeding all three: camera → detected gate corners →
metric 3D gate pose → waypoints. Team: Tony owns `vision/` (this
directory) plus the RL stack; Tye owns gate detection (upstream of this
code) and path generation (downstream); Antek owns the NMPC controller.

## Repo layout

```
control/    MPC / control stack (Antek's — not covered here)
vision/     this directory
workspace/  spawn_example.py (Isaac Sim + Pegasus drone/camera spawn) and
            spawn_test_gates.py (magenta test gates + ground-truth manifest);
            both run inside Isaac Sim's Script Editor, not plain python3
```

## The one rule that matters most: frame conventions

Read `camera.py`'s module docstring and `gate_pose.py`'s module docstring
in full before writing anything that touches a pose, transform, or
quaternion. Summary, but read the originals — they explain *why*:

- **OpenCV camera frame**: +X right, +Y down, +Z forward. `cv2.solvePnP`
  always returns poses in this frame.
- **USD/Isaac camera frame**: +X right, +Y up, -Z forward. Isaac Sim's own
  APIs (`get_world_pose()`, etc.) use this. `camera.R_CV_FROM_USD` converts
  between them; convert once, at the boundary, never again.
- **ROS quaternions are XYZW.** `camera.quat_to_rot()` expects **WXYZ**.
  Every ROS boundary must route through `ros_camera.quat_xyzw_to_wxyz()`.
  This is the single easiest bug to reintroduce — there is no import error
  or crash when you get it wrong, just a silently incorrect rotation.
- **Gate model frame** (`gate_pose.gate_model_points`): +X right, +Y
  **down**, +Z along the **flight direction** — deliberately matching the
  OpenCV camera convention so a gate straight ahead, upright, is the
  identity rotation. This was WRONG until 2026-09-20 (see below) — it used
  to be +Y up, which is not right-handed from the pilot's point of view and
  silently reversed every approach waypoint. Do not "fix" it back.
- **Transform naming**: `T_a_b` maps a point from frame `b` to frame `a`:
  `p_a = T_a_b @ p_b`. Compose left to right: `T_a_c = T_a_b @ T_b_c`.

## Files and status

| File | Status | Notes |
|---|---|---|
| `camera.py` | done | pinhole model, intrinsics, frame-conversion helpers |
| `gate_pose.py` | done, bug-fixed | PnP solve, corner ordering, approach waypoints |
| `eval_pnp.py` | done (needs re-grounding) | Monte Carlo error characterization vs. synthetic corner noise — should eventually be re-run against Isaac Sim ground truth instead of synthetic noise, once the sim loop below is closed |
| `isaac_camera.py` | **superseded, not deleted** | direct-USD camera path; the project now uses the ROS 2 path instead (see decision below) — kept for reference / `capture_labeled_frame()` may still be useful for generating labeled training data offline |
| `ros_camera.py` | done; frame convention **verified** live (2026-09-28); `GatePoseNode` still TF-based and **has never run against this sim** | ROS 2 adapters: `CameraInfo` → intrinsics, TF → `T_world_cam`, `GatePoseNode` (now defaults `optical_frame=False`) |
| `verify_isaac_pipeline.py` | **superseded** | waits for a `front_cam` TF that Pegasus never publishes (see OPEN BLOCKING QUESTION below) — kept for reference, do not run it against this sim |
| `test_gate_pose.py` | passing | run with plain `python3`, no ROS needed |
| `test_ros_camera.py` | passing | run with plain `python3`, no ROS needed |
| `color_gate_detector.py` | working live (3/3 gates 2026-09-28; 5/5 at 1280x720 2026-10-01) | sim-only solid-color detector; Tye's real detector replaces this. The 3x3 morphological close is now OFF by default (`close_px=0`): it fused the nested gates into one blob |
| `eval_isaac_gates.py` | **run live** — `optical_frame=False` settled 2026-09-28; re-run 2026-10-01 at 1280x720 / 60 deg: 5/5 gates, mean 0.21 m | gate-pose accuracy + `optical_frame` decision. `--camera-pose pose` reads `/drone00/state/pose` (best-effort QoS) because this sim publishes no TF. `--hfov-deg` builds intrinsics from the **real rgb image size** (fixed 2026-10-01; it used to take camera_info's stale width/height), and a WARNING prints whenever camera_info and the image disagree |
| `test_color_gate_detector.py` | passing | run with plain `python3`, no ROS needed |
| `test_eval_isaac_gates.py` | passing | run with plain `python3`, no ROS needed |
| `test_gate_layout.py` | passing (fixed 2026-10-01) | the narrow-FOV test now uses `NESTED_GATE_SPECS`; `test_approach_layout_level_and_fits_pitched_camera` covers `APPROACH_GATE_SPECS`. Needs no ROS/omni/pxr; the one file that imports across the `vision/`/`workspace/` boundary |
| `gate_error_eval/` | **run live 2026-10-01** (hover bag, 1 m and 3 m approach flights) | error-vs-ground-truth graphs, hover or moving: `fly_gate_approach.py` (pymavlink offboard straight-line flight) → `ros2 bag record` → `eval_gate_bag.py` (bag → `gate_errors.csv`, pose interpolated to each image stamp) → `plot_gate_errors.py` (4 report PNGs + summary table, runs on a Mac). `error_records.py` holds the shared CSV schema / `PoseBuffer` / summary. See "Gate-error graphs" below. `example_output/` is **synthetic** data |
| `workspace/spawn_test_gates.py` | run live (2026-10-01: 5 gates kept, 0 dropped at 1280x720 / 60 deg). `check_layout`'s clearance is still too loose: it once passed a layout whose gate bars were 1-2 px apart | reads the camera's live pose/FOV off the USD stage and places magenta test gates. Default `GATE_SPECS` is a **staggered** layout (ranges 6-14 m, scattered az/el) sized for a wide camera; the old on-axis nested chain is kept as `NESTED_GATE_SPECS` (at a wide FOV its frames touch and the mask merges them: detector found 0/5). Writes `/workspace/test_gates.json` (shared volume). **Run it only after the drone is hovering** |
| `workspace/spawn_example.py` | run live | spawns Iris + PX4/ROS 2 backends + camera at `RESOLUTION=(1280,720)`, `HFOV_DEG=60`; after configuring the camera it calls `refresh_camera_info_helper()` (see "RESOLVED (2026-10-01)"). **Run it ONCE per Isaac Sim process** — see deployment notes |

Run the test files exactly like this before and after any change in this
directory:
```
cd vision
python3 test_gate_pose.py
python3 test_ros_camera.py
python3 test_color_gate_detector.py
python3 test_eval_isaac_gates.py
python3 test_gate_layout.py
```
All five print `All tests passed.` (plus the three in `gate_error_eval/`). None needs ROS, Isaac Sim, or
omni/pxr installed — they only need `numpy` and `cv2` (opencv-python).
`eval_isaac_gates.py` is different: it needs ROS 2 sourced and a live
simulator, and is not runnable in this offline environment.
`verify_isaac_pipeline.py` needs the same, but see above — don't bother
running it against this sim, it waits for a TF that doesn't exist.

## Decision: ROS 2 camera path, not direct USD

Two ways existed to get camera data out of Isaac Sim. `isaac_camera.py`
uses the direct USD `Camera` API (`get_rgba()`, `get_world_pose()`, etc.),
which only works inside the simulator. `spawn_example.py` (in
`workspace/`) instead uses Pegasus's `ROS2CameraGraph`, publishing
`rgb`/`depth`/`camera_info` on `/iris_0/front_cam`. The project committed
to the ROS 2 path because it's the same interface a real camera driver
exposes — the CV node doesn't change between sim and hardware.
`ros_camera.py` implements this path. Don't build new code against
`isaac_camera.py`'s API.

## RESOLVED (2026-09-28): `optical_frame=False`, and what the live sim actually publishes

`eval_isaac_gates.py --camera-pose pose --frames 5` against the live sim
(Isaac Sim 4.5.0 + Pegasus, drone hovering, 3 nested magenta gates at about
6.0 / 8.4 / 11.6 m): `optical_frame=False` won with `confident=True`.
`optical_frame=True` matched **zero** gates. Matched position error
0.07-0.21 m (mean 0.120 m, n=15), almost all of it depth; lateral error
about 3 mm. The manifest cross-check (pose topic + mount vs. the pose read
off the USD stage) agreed to 0.03 m / 0.6-1.3 deg. Camera was 320x240,
fx=763.5 (~24 deg hFOV) — a 1 px corner error is ~1% of a 100 px gate, i.e.
~0.1 m at 11 m, so this is quantization-limited, not a convention problem.

**Corrections to what this file used to claim about Pegasus:**
- Pegasus' `ROS2CameraGraph` publishes no camera TF (unchanged).
- `ROS2Backend` publishes **no TF at all** in this build (`/tf` and
  `/tf_static` never appear). The vehicle pose comes on
  **`/drone00/state/pose`** (`geometry_msgs/PoseStamped`, `frame_id: map`,
  quaternion xyzw). Namespace is `drone00`, not `drone0`.
- That topic is **BEST_EFFORT**. A default (RELIABLE) subscription is
  QoS-incompatible and silently receives nothing.

The validated camera-pose chain is therefore:

```
T_world_cam_usd = T_map_body(from /drone00/state/pose)
                @ T_body_cam_usd(static mount from spawn_example.py:
                                 translate (0.30, 0, 0.05), rotateXYZ (105, 0, -90))
T_world_cam     = T_world_cam_from_transform(T_world_cam_usd, optical_frame=False)
```

KEEP the mount constants in sync with `spawn_example.py`.

## RESOLVED (2026-10-01): stale `camera_info`

**Symptom.** After `spawn_example.py` moved the camera to 1280x720 / 60 deg
hFOV, the images were 1280x720 but `/iris_0/front_cam/camera_info` still said
320x240, `fx=763.5`, `cx=160`, `cy=120`. `eval_isaac_gates.py` built PnP
intrinsics from it and got a 1.99 m error with 1 of 5 gates matched
(the detector found all 5). Anything that reads `camera_info` — including
`GatePoseNode` — is wrong at this FOV until it is fixed.

**Cause.** `ROS2CameraGraph` publishes camera_info through
`OgnROS2CameraHelper` (`type=camera_info`, deprecated since Isaac Sim 4.1;
the replacement `OgnROS2CameraInfoHelper` behaves the same way). On its
first compute the node calls `read_camera_info()` **once** and bakes
width/height/k/p into its writer. That first compute happens before
`spawn_example.py` sets the focalLength and resolution, so it snapshots the
USD defaults: `fx=763.54` at 320 px with `horizontalAperture=20.955` is
exactly `focalLength = 50 mm`. It never re-reads. (Not understood: why the
baked size was 320x240 rather than the configured 1280x720.) It is not a
leaked second graph: one publisher per topic, and 12/12 samples identical.

**Fix.** `spawn_example.py::refresh_camera_info_helper()` toggles the helper
node's `inputs:enabled` False -> True (10 frames each), which runs
`custom_reset()` and re-initializes it against the configured camera. Node
path: `/World/Iris/body/front_cam_pub/camera_helper_camera_info`. Verified by
hand on a live sim: camera_info became `1280x720, fx=1108.5, cx=640,
cy=360` with the rgb stream unaffected (33 Hz, one publisher).
**Not yet verified from a fresh spawn** — the call was added after that
test. On the next fresh spawn, `spawn_example.py` should print
`[init] camera_info helper re-initialized`; if it prints a WARNING instead,
camera_info is stale again. `probe_camera_info_reset.py` (the one-off probe,
in `/workspace/scripts` on the cluster) does the same toggle on a running sim.

**Workaround / backstop.** `eval_isaac_gates.py --hfov-deg 60` ignores
camera_info's fx/size and uses the real image size. It should no longer be
needed after a fresh spawn with the fix, and the evaluator warns when the
two disagree, so a regression is visible.

**Re-run at 1280x720 / 60 deg (2026-10-01), drone hovering at ~2.6 m, 5
frames, `--camera-pose pose --hfov-deg 60`:** `optical_frame=False`,
`confident=True`, 5/5 gates matched, mean position error 0.210 m (std 0.150,
n=25). `optical_frame=True` matched nothing. Manifest cross-check 0.05 m /
0.14 deg. The 25 matches are 5 frames of one static scene, not independent
samples.

| Gate | Range | Pos err | Range err | Lateral err | Normal err |
|---|---|---|---|---|---|
| gate_0 | 6 m | 0.063 m | 0.058 m | 0.023 m | 0.6 deg |
| gate_1 | 8 m | 0.099 m | 0.091 m | 0.037 m | 2.1 deg |
| gate_2 | 10 m | 0.122 m | 0.120 m | 0.026 m | 3.1 deg |
| gate_3 | 12 m (24 deg off-axis) | 0.409 m | 0.369 m | 0.175 m | 0.8 deg |
| gate_4 | 14 m | 0.334 m | 0.307 m | 0.132 m | 2.5 deg |

Error grows with range and is mostly depth (about 2-3x lateral). gate_3 is
the worst despite not being the farthest; it is far off-axis, but this was
not investigated.

## Live-sim deployment notes (Kubernetes, namespace `fake-isaac`)

- Pod has three containers: `isaac-sim`, `px4-sitl`, `ros2-eval`
  (`ros:humble-ros-base`, for running the vision scripts). All share the
  `/workspace` PVC. `vision/` lives at `/workspace/vision`, sim scripts at
  `/workspace/scripts`.
- **Fast DDS shared memory:** `ros2-eval` must share `/dev/shm` with
  `isaac-sim` (the `dshm` volume, now mounted in the YAML). Without it,
  `ros2 topic list` shows topics but `hz`/`echo` hang. Workaround in a
  running shell: `FASTRTPS_DEFAULT_PROFILES_FILE` pointing at a UDP-only
  profile.
- **Do not `pip install numpy opencv-python-headless` in `ros2-eval`:** the
  wheels pull NumPy 2.x, which breaks ROS Humble's `cv_bridge`. Use the apt
  versions (NumPy 1.21.5, OpenCV 4.5.4) — the repo tests pass on them.
- The timeline must be **playing** or the camera graph publishes nothing.
- **Separate filesystems.** `/workspace` is shared; each container's `/root`
  is not. Anything written to `~/` by a script in Isaac Sim is invisible to
  your shell. Use `/workspace/...`. Run sim scripts straight from the
  shared volume in the Script Editor:
  `exec(open("/workspace/scripts/spawn_example.py").read())`.
- **ROS 2 is in `ros2-eval`**, not `px4-sitl` (`/opt/ros/humble` is only
  there). `isaac-sim` Kit log: `/workspace/logs/isaac-sim.log`
  (also `kubectl logs deploy/isaac-sim -c isaac-sim`). The GUI is noVNC on
  port 6080 via a `kubectl port-forward`; it drops whenever the pod restarts.
- **Pegasus blocks the whole sim until PX4 connects.** The PX4 backend loops
  on `Waiting for first hearbeat` in the physics step, so with no PX4 the
  sim, `/drone00/state/pose` and the camera all stay silent even though the
  timeline says "playing". Start PX4 after Play (it is the TCP client of
  Isaac's port 4560):
  `cd /workspace/PX4-Autopilot && PX4_SIM_MODEL=none_iris
  ./build/px4_sitl_default/bin/px4 ./ROMFS/px4fmu_common/ -s
  ./ROMFS/px4fmu_common/init.d-posix/rcS -i 0` (build with
  `make px4_sitl_default none`; the model string is the airframe file
  `10016_none_iris`, not a make target). Then `commander takeoff` at the
  `pxh>` prompt.
- **A PX4 left over from an earlier spawn is useless:** it stays connected
  to a dead backend and prints `poll timeout` forever; a second PX4 refuses
  to start ("server already running for instance 0"). Kill the old
  `bin/px4` process (it survives closing the terminal that launched it)
  before relaunching.
- **Never run `spawn_example.py` twice in one Isaac Sim process, and
  File -> New does not reset it.** The second run deletes `/World/Iris` and
  rebuilds it, which fails (`'NoneType' object has no attribute
  'link_names'`: the physx simulation view was invalidated), and the old
  PX4 backend keeps TCP port 4560 bound (`OSError: Address already in use`
  on the next Play). Restart the process instead:
  `kubectl exec deploy/isaac-sim -c isaac-sim -- kill <kit pid>` (find it
  with `ps -eo pid,args | grep kit/kit`), wait a few minutes for it to boot,
  add a ground plane, then spawn once.
- `[init] Done` only means the scene was built, not that the PX4 link works.
  Check the log for `Received first hearbeat` and no `Address already in use`.
- Someone scaling the deployment down wipes the pod: hover, spawned gates,
  PX4 and the GUI session are all lost. `/workspace` survives.

## Running the sim test

In order, on a freshly started Isaac Sim process (see the deployment notes
for why "fresh" matters):

1. Add a ground plane (or the Simple Room environment) to the stage.
2. Script Editor: `exec(open("/workspace/scripts/spawn_example.py").read())`,
   **once**. Expect `[init] camera 1280x720, hFOV 60.0 deg ...` and
   `[init] camera_info helper re-initialized`, then `[init] Done`.
3. Press Play. The sim may appear frozen (Pegasus is waiting for PX4).
4. Start PX4 (command in the deployment notes), confirm the Isaac log shows
   `Received first hearbeat`, then `commander takeoff` and let it settle at
   ~2.5 m. From `ros2-eval`, `ros2 topic hz /drone00/state/pose` should show
   ~130 Hz and `/iris_0/front_cam/camera_info` ~30 Hz.
5. Only now, Script Editor:
   `exec(open("/workspace/scripts/spawn_test_gates.py").read())`. Expect a
   camera height of ~2.5 m, no on-the-ground warning, and
   `wrote /workspace/test_gates.json (5 gate(s) kept, 0 dropped)`. It is safe
   to re-run (it clears its own `/World/TestGates` group).
6. In `ros2-eval`: `cd /workspace/vision && source /opt/ros/humble/setup.bash
   && python3 eval_isaac_gates.py --gates /workspace/test_gates.json
   --camera-pose pose --frames 5 --out-dir <new dir>`. Use a new `--out-dir`
   so you don't overwrite the previous run's `annotated_frame.png`/CSV. Add
   `--hfov-deg 60` if the run prints the camera_info-vs-image WARNING.

`eval_isaac_gates.py` detects the test gates with `color_gate_detector.py`,
solves a real PnP problem per gate, builds `T_world_cam_usd` from the pose
topic plus the static mount, lifts each detection into the world frame under
BOTH `optical_frame` values, and matches the results against the manifest's
known positions. The decision block at the end names the winning
`optical_frame` value, whether it's `confident`, and a one-line reason.
`--camera-pose manifest` uses the camera pose `spawn_test_gates.py` recorded
instead (no pose topic needed, but only valid if the drone hasn't moved since
the gates were spawned). Check the cross-check line: a large position or
rotation delta between the live pose and the manifest means the drone moved
or the mount constants disagree with `spawn_example.py`.

## Gate-error graphs (`gate_error_eval/`)

Graphs of PnP gate-position error against the manifest, with the drone
hovering or flying. Gates are static, so truth is always the manifest; only
the camera pose changes. Tests (no ROS, Mac: use `/usr/local/bin/python3.12`):
```
cd vision/gate_error_eval
python3 test_error_records.py && python3 test_plot_gate_errors.py && python3 test_fly_gate_approach.py
```

Runbook. Steps 1-5 of "Running the sim test" come first, so the drone is hovering and
`/workspace/test_gates.json` exists:

1. `ros2-eval` shell A: `ros2 bag record -o /workspace/bags/run1
   /iris_0/front_cam/rgb /iris_0/front_cam/camera_info /drone00/state/pose`.
   Raw 1280x720 rgb is ~90 MB/s: keep runs to ~20-30 s and check that the bag's
   image count (`ros2 bag info`) is close to 30 Hz x duration (sqlite3 may drop
   frames at that rate).
2. Shell B: `pip install pymavlink` once (abort if pip wants to touch
   numpy), then `python3 /workspace/vision/gate_error_eval/fly_gate_approach.py
   --gates /workspace/test_gates.json --distance 3 --speed 1`. It hovers 5 s,
   flies 3 m along the current heading, hovers 5 s, then goes to AUTO.LOITER.
   It refuses if the path comes within 2 m of a gate. **Untested:** the PX4
   mode numbers and `type_mask`s were checked against the MAVLink/PX4 docs, but
   the default link `udpin:0.0.0.0:14540` assumes PX4 SITL's stock onboard
   MAVLink instance. Try a 1 m flight first. Stop the bag after "done".
3. `python3 eval_gate_bag.py --bag /workspace/bags/run1 --gates
   /workspace/test_gates.json --out-dir /workspace/gate_eval/run1` (add
   `--hfov-deg 60` if it prints the stale-camera_info WARNING). Check the
   `time base used` line. Header stamps are preferred. If the image and pose
   stamps don't overlap (different clocks), it falls back to bag receive
   time, which adds transport latency to the pairing. A median image-pose
   gap of more than a few ms means the pairing is suspect for moving frames.
4. Copy `gate_errors.csv` to the Mac and run `python3 plot_gate_errors.py
   gate_errors.csv --out-dir figs`. This gives fig1 (error vs range,
   3D | depth+lateral), fig2 (error vs time with speed shaded), fig3
   (top-down), fig4 (detection rate vs range, in-view frames only) and
   `summary_table.csv`.

The bag can be re-evaluated whenever the detector changes. `evaluate_frame`
takes a `detector=` callable, so Tye's detector can be dropped into
`eval_gate_bag.evaluate_image_at` without re-flying.

**Live results (2026-10-01).** Hover, 474 frames: 0.063 / 0.093 / 0.101 /
0.443 / 0.279 m at 5.7-12.6 m, 100% detection, almost all of it a constant
positive range bias (~1% of range for gates 0-2: estimates sit just BEHIND
the gate). That points at a scale error (detected corners slightly inside
the 1.5 m opening PnP assumes, or fx ~1% off), not noise. Not investigated.
The 3 m approach at 1 m/s (overshot to 1.4 m/s): the flight script works
against the cluster PX4 (udpin:0.0.0.0:14540, OFFBOARD entry, carrot), but
with the staggered layout every gate left the TOP of the frame within ~1.5 s
of moving. The gates sit 0.7-6 m above the camera and the drone pitches
nose-down to accelerate, so almost no moving-frame data came back. In that
1.5 s, error rose (gate_0 0.06 -> 0.085 m) while range shrank, so motion
probably adds error, perhaps attitude timing: camera stamps are SIM time and
pose stamps are WALL time, so pairing falls back to bag receive time. Fix
for the layout: the approach layout (below).

**Camera mount moved (found 2026-10-01, cause unknown).** After the
flights, the live camera prim sat at (-0.040, 0, -0.041) m from
`/World/Iris/body`, not at the (0.30, 0, 0.05) `spawn_example.py` sets. Its
rotation still matched the mount to 0.1 deg. The body prim and
`/drone00/state/pose` agreed to 2 cm. The first hover run and both early
flights were consistent with the original mount, so the camera moved later;
the Kit log shows nothing that explains it. With the hardcoded mount, the
5 m approach scored ~0.5 m on every gate, a false 0.35 m offset. Anything
using the mount constants is wrong while this holds: `eval_isaac_gates.py`,
`GatePoseNode`, and the default `eval_gate_bag.py`. **Check the mount
before trusting a run.** Script Editor, while hovering:
```
from pxr import UsdGeom, Usd; import omni.usd; s = omni.usd.get_context().get_stage(); B = UsdGeom.Xformable(s.GetPrimAtPath("/World/Iris/body")).ComputeLocalToWorldTransform(Usd.TimeCode.Default()); C = UsdGeom.Xformable(s.GetPrimAtPath("/World/Iris/body/front_cam")).ComputeLocalToWorldTransform(Usd.TimeCode.Default()); print("cam in body frame:", (C * B.GetInverse()).ExtractTranslation())
```
If it is not about (0.30, 0, 0.05), pass the printed value to
`eval_gate_bag.py --mount-xyz X Y Z`. Finding what moves the prim is open.

**5 m approach at 1 m/s, measured mount (2026-10-01).** 4 approach-layout
gates, 719 frames, ~100% detection when fully in frame. Hover 0.13-0.21 m
(0.9-2.2% of range); moving (>0.5 m/s) 0.17-0.25 m (1.6-3.1%), i.e. +0.04-0.07 m.
The extra error peaks while the drone ACCELERATES (pitching) and drops while
it is still at speed. That suggests attitude/timing (image vs pose pairing
falls back to bag receive time because camera stamps are sim time and pose
stamps wall time), not motion blur. Figures: `gate_error_eval/runs/
approach_run5m_measured_mount/` (not committed).

**Approach layout.** In the Script Editor, with the drone hovering:
`import os; os.environ["SPAWN_LAYOUT"] = "approach"; exec(open("/workspace/scripts/spawn_test_gates.py").read())`. Check that it prints `layout: approach` (a plain Script Editor variable is NOT seen, so it silently spawned the staggered layout once).
This gives 4 gates placed relative to the horizon, searched so they stay in
frame along a 5 m approach (see `APPROACH_GATE_SPECS`). Then fly
`--distance 5 --speed 1`. Without the variable, the default staggered layout
is spawned as before.

## Interface contracts with teammates (unconfirmed — settle these)

- **With Tye (detection → this code):** the input contract is **four
  ordered corner pixels**, not a bounding box. A bounding box gives no
  gate orientation, so no approach waypoints can be generated. This has
  been raised with Tye but not yet confirmed in writing.
- **With Tye (this code → path generation):** unclear whether
  `gate_pose.approach_waypoints()` output (pre/center/post points on the
  gate normal) is meant to be consumed directly by Tye's path generator,
  or whether that responsibility should move downstream. Also unclear
  whether Tye's path generator expects bare XYZ or a time-parameterized
  trajectory (position/velocity/acceleration/yaw) — NMPC needs the latter.
- **With Antek (control):** not yet discussed from the vision side.

## Known gaps / next work, roughly in priority order

1. **Give `GatePoseNode` a pose-topic mode.** `optical_frame` is settled
   (False), but `GatePoseNode` still gets the camera pose from a TF lookup
   of `front_cam`, and this sim publishes no TF. It needs to subscribe to
   `/drone00/state/pose` (BEST_EFFORT QoS) and compose the static mount, as
   `eval_isaac_gates.py --camera-pose pose` already does. It also reads
   `camera_info`, so it depends on the stale-camera_info fix holding. This
   is the step between "evaluated offline against the sim" and "runs live as
   a ROS 2 node".
2. **Verify the camera_info fix from a fresh spawn.** It was proven by a
   live toggle, then added to `spawn_example.py` afterwards. One clean
   process restart + spawn should show the `[init] camera_info helper
   re-initialized` line and a correct `camera_info` with no `--hfov-deg`.
3. ~~Fix `test_gate_layout.py`~~ done 2026-10-01.
4. **Temporal fusion across frames.** Single-frame PnP is depth-dominated
   (range error ~2-3x lateral at 6-14 m, 0.06 m at 6 m up to 0.3-0.4 m at
   12-14 m in the 2026-10-01 run), and gate normals become unreliable past
   ~8 m (`GateDetection.normal_is_reliable`). Gates are static, so a
   per-gate filter (e.g. a small EKF with a constant-position model,
   association by predicted reprojection) fusing repeated sightings should
   collapse this substantially. Not yet built — probably the single
   highest-value next module in `vision/`.
5. **Tighten `check_layout`'s clearance** in `spawn_test_gates.py` so it
   rejects layouts whose gate bars come within a few pixels of each other
   (it passed a layout with 1-2 px gaps; the detector's close used to fuse
   them). The detector is fixed, but any mask-based detector will have the
   same weakness.
6. **Investigate gate_3's error** (0.41 m at 12 m, 24 deg off-axis, worse
   than the 14 m gate). Possibly off-axis geometry or corner quantization;
   not looked at.
7. **`eval_pnp.py` re-grounding.** Currently characterizes error against
   synthetic corner noise. The sim loop is closed now, so re-run it against
   Isaac Sim ground truth (`isaac_camera.capture_labeled_frame()` is a
   template for generating labeled samples, though it uses the superseded
   USD path — may need porting to the ROS 2 path, or kept as an offline
   labeling tool since it doesn't need to run live).
8. **Replace the deprecated camera_info helper.** Pegasus still uses
   `ROS2CameraHelper type=camera_info`; the toggle in `spawn_example.py` is
   a workaround, not a fix. The real fix is upstream (or in the installed
   Pegasus copy) so the node initializes after the camera is configured.
9. **`workspace/spawn_example.py` placement.** It's currently the only
   file outside the `control/`/`vision/` layout the rest of the repo has
   settled into. Not urgent, but worth a decision.

## Recent history (for context on *why* things look the way they do)

- `157af70` "starting Vision module" — initial `camera.py`, `gate_pose.py`,
  `eval_pnp.py`, `isaac_camera.py`, `test_gate_pose.py`.
- `efcac73` "control folder" — teammate's commit, moved MPC files into
  `control/`, added `Drone_Race_Simulation.py`, `MPC_Optimizer.py`. No
  overlap with `vision/`.
- `5688b90` "Fix gate frame convention so +Z is the flight direction" —
  the gate-frame bug fix described above. Caught by manually probing
  `approach_waypoints()` output, not by the original test suite — the
  original `test_world_transform_and_waypoints` only asserted waypoint
  *spacing*, never *order*, so it passed despite the bug. Added
  `test_waypoint_ordering`, deliberately built from raw pixels rather than
  a model round-trip, because a round-trip test cannot catch a convention
  bug like this one (projecting and solving with the same wrong model
  stays perfectly self-consistent). Verified the new test actually catches
  a regression by deliberately reverting the fix and confirming it fails.
- `c1d6b0d` "Add ROS 2 camera adapters for the Isaac Sim perception path" —
  `ros_camera.py` / `test_ros_camera.py`, described above.
- `8096c55` "End-to-end vision update and test" — teammate's commit:
  `color_gate_detector.py`, `eval_isaac_gates.py`, `spawn_test_gates.py`,
  `test_*` and `verify_isaac_pipeline.py` (now committed, but superseded).
- 2026-09-28 — teammate ran `eval_isaac_gates.py` live and settled
  `optical_frame=False`; found `/drone00/state/pose` is the only pose source
  and the sim publishes no TF; widened the camera to 1280x720 / 60 deg and
  replaced the nested gate layout with a staggered one.
- `e8ef802` / PR #1 (`089b979`), 2026-10-01 — synced the cluster working
  copies into the repo; fixed `--hfov-deg` to use the real image size and
  added the stale-camera_info warning; added
  `refresh_camera_info_helper()` to `spawn_example.py` after tracing the
  stale camera_info to the helper baking its values on first compute.
