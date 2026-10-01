# CV Pipeline Handoff

For whoever (or whatever) picks up computer vision work on this repo next.
Scope: `vision/` only. If you're Claude Code, read this whole file before
touching anything in this directory — several of the decisions below are
non-obvious and easy to silently undo.

Last updated: 2026-09-28.

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
workspace/  spawn_example.py — Isaac Sim + Pegasus drone spawn script
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
| `color_gate_detector.py` | working live (3/3 gates, 2026-09-28) | sim-only solid-color detector; Tye's real detector replaces this. The 3x3 morphological close is now OFF by default (`close_px=0`): it fused the nested gates into one blob |
| `eval_isaac_gates.py` | **run live, 2026-09-28** — settled `optical_frame=False` | gate-pose accuracy + `optical_frame` decision. New `--camera-pose pose` mode (reads `/drone00/state/pose`, best-effort QoS) because this sim publishes no TF |
| `test_color_gate_detector.py` | passing | run with plain `python3`, no ROS needed |
| `test_eval_isaac_gates.py` | passing | run with plain `python3`, no ROS needed |
| `test_gate_layout.py` | passing | run with plain `python3`, no ROS/omni/pxr needed; the one file that imports across the `vision/`/`workspace/` boundary (see its own docstring) |
| `workspace/spawn_test_gates.py` | run live; manifest path must be `/workspace/test_gates.json` (shared PVC), not `~/`. `check_layout`'s clearance is too loose: passed a layout whose gate bars were 1-2 px apart | reads the camera's live pose/FOV off the USD stage and lays out a nested chain of magenta test gates accordingly; writes `~/test_gates.json`; run in the Isaac Sim script editor after `spawn_example.py` |

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
All five must print `All tests passed.` None needs ROS, Isaac Sim, or
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

## Running the sim test

Four steps, in order, resolve the `optical_frame` question above (and give
a first real accuracy number) in one run:

1. In the Isaac Sim script editor: run `workspace/spawn_example.py`.
2. Press Play, and take off / hover if you want the drone airborne — under
   physics the drone otherwise just falls to the ground. (This matters:
   `spawn_test_gates.py` reads the camera's LIVE pose off the USD stage,
   so gates land in front of wherever the camera actually ends up looking.)
3. In the same script editor session: run `workspace/spawn_test_gates.py`
   (spawns a nested chain of magenta test gates along the camera's actual
   optical axis, writes `~/test_gates.json` with their ground-truth poses
   plus the camera pose used to place them).
4. In the `ros2-eval` container (ROS 2 sourced): `python3 eval_isaac_gates.py --gates
   /workspace/test_gates.json --camera-pose pose --frames 5`.

`eval_isaac_gates.py` detects the test gates with `color_gate_detector.py`,
solves a real PnP problem per gate, builds `T_world_cam_usd` from live
`map -> drone0_base_link` TF plus the static mount, lifts each detection
into the world frame under BOTH `optical_frame` values, and matches the
results against the manifest's known positions. The decision block at the
end names the winning `optical_frame` value, whether it's `confident` (low
absolute error and a wide margin over the other convention), and a
one-line reason. If TF isn't available for some reason, pass `--camera-pose
manifest` to use the camera pose `spawn_test_gates.py` recorded instead
(no TF needed, but only valid if the drone hasn't moved since gates were
spawned). `annotated_frame.png` and `gate_mask.png` are written alongside
the CSV for a closer look if the numbers look off.

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
   `eval_isaac_gates.py --camera-pose pose` already does. This is the step
   between "evaluated offline against the sim" and "runs live as a ROS 2
   node".
2. **Widen the FOV and raise the resolution.** Camera is 320x240 at
   fx=763.5 (~24 deg hFOV). That is why `spawn_test_gates.py` drops 2 of 5
   gates as out of FOV, and it limits depth accuracy (see item 4). A
   racing camera is far wider.
3. **Tighten `check_layout`'s clearance** in `spawn_test_gates.py` so it
   rejects layouts whose gate bars come within a few pixels of each other
   (it passed a layout with 1-2 px gaps; the detector's close used to fuse
   them). The detector is fixed, but any mask-based detector will have the
   same weakness.
4. **Temporal fusion across frames.** Single-frame PnP has a ~6:1
   depth-to-lateral error ratio at ~12 m (see `eval_pnp.py` /
   project memory for the full characterization), and gate normals become
   unreliable past ~8 m (`GateDetection.normal_is_reliable`). Gates are
   static, so a per-gate filter (e.g. a small EKF with a constant-position
   model, association by predicted reprojection) fusing repeated
   sightings should collapse this substantially. Not yet built — this is
   probably the single highest-value next module in `vision/`.
5. **Resolution check (see item 2).** `spawn_example.py`'s `ROS2CameraGraph` currently
   publishes at 320×240. `eval_isaac_gates.py` will print a warning if it
   detects this — at that resolution a 1.5 m gate at 12 m spans roughly
   20 px, so one pixel of corner error is already a large fraction of the
   gate, which the 6:1 depth amplification then makes worse. Worth raising
   to something like 1280×720 if compute allows.
6. **`eval_pnp.py` re-grounding.** Currently characterizes error against
   synthetic corner noise. Once the sim loop is closed, re-run it against
   Isaac Sim ground truth (`isaac_camera.capture_labeled_frame()` is a
   template for generating labeled samples, though it uses the
   superseded USD path — may need porting to the ROS 2 path, or kept as
   an offline labeling tool since it doesn't need to run live).
7. **`workspace/spawn_example.py` placement.** It's currently the only
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
- `verify_isaac_pipeline.py` — written but not yet committed as of this
  handoff; not yet run against a live sim.
