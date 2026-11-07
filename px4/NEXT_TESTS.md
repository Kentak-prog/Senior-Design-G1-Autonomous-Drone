# Next tests: what to work on (updated 2026-11-06)

**Goal:** the NMPC flies the 3-gate course in Isaac Sim using only the VIO
(IMU + camera). Ground truth is used only by the safety watchdog and for
scoring. The test plan is `px4/VIO_MPC_TEST_PLAN.md`.

## Status

| Stage | Result |
|---|---|
| 0–2. Pod prep, Isaac + PX4, dry run | done. A pod restart wipes the Stage 0 installs: redo them |
| 3. Hover hold, ground truth | **pass** (within 0.19 m, altitude ± 0.01 m) |
| 4. Course, ground truth, 2/2 | **pass**: all gates, margins 0.47–0.71 m, tracking 0.12 m mean |
| 4R. One closed lap (gate_0 → gate_1 → gate_2 → gate_0) with in-flight replanning | node ready (`ros2_nodes/replan_node.py`, defaults `laps 1`, `close_loop true`); flown only with the VIO in charge, below |
| **7. Closed lap, VIO in charge, 2/2** | **4 attempts, 4 watchdog aborts.** Best: attempt 3, through gate_1, 13.5 s |
| **Next: the Starling 2 Max (C29 camera) on the circle course: 3, 4R, 6, 7** | 2026-11-07: Stage 0 passed in the pod (Starling MPC solve median 7.7 ms). Stage 1 flipped on takeoff twice: `spawn_starling.py` wrote the rotor joint anchors in the Iris mesh frame, 90° from the body frame, so every rotor sat 90° off and PX4's roll and pitch were swapped. Fixed in `apply_starling_geometry`, checked offline (`workspace/test_starling_geometry.py`). Flown once with that fix: it spun at −200°/s on the ground as soon as it armed, because the Iris' 10-inch props overlap at the Starling spacing and their colliders hit each other. Prop colliders now disabled and rotor mass set (5 g each, totals unchanged); checked offline, not flown yet. `px4/VIO_MPC_TEST_PLAN.md`, "The Starling 2 Max (C29)": longest gap with no fix 1.2 s, VIO rehearsal max 0.21–0.40 m. With the Iris: ready, not flown. `px4/VIO_MPC_TEST_PLAN.md`, "The circle course": a gate in view except the last ~2 s before each gate, 0 biased fixes, VIO rehearsal max 0.16–0.33 m |

**The flight stack works.** The bridge, the MPC (on VIO state only), the
planner (plan starts at engage), PX4 and the watchdog all behaved. **What
stops us is the VIO.**

---

## 1. For the VIO / vision team

### The problem in one paragraph

Between gate_0 and gate_1 (about **+5.6 s to +10 s** after engage at 2 m/s)
**no gate fits wholly in the 60° × 36° camera frame**. The color detector
needs the whole opening, so the VIO gets no fixes and dead-reckons on the
IMU. It drifts **0.12–0.5 m/s** and the 1 m watchdog trips. Before and after
that gap, fixes flow at about 60 Hz and the VIO holds about 0.1–0.3 m.

Evidence: `fig5_gate_visibility.png` in every run folder below. 315 of 316
fixes in the Stage 4 run came while a whole gate was in frame, by geometry
from the true pose.

### Tasks, in priority order

**V1. Record a flight's raw data. Nothing else can be tuned without it.**
- [ ] Record everything in the next Isaac flight:
  ```
  ros2 bag record /iris_0/front_cam/rgb /iris_0/front_cam/camera_info /drone00/sensors/imu /drone00/state/pose /drone00/state/twist /clock
  ```
  None of the four flights recorded images or raw IMU.
- [ ] Log every frame in `vio_node`: detections found, PnP failures,
  unmatched, fixes, rejections. Today a frame with no row in
  `vio_events.csv` could be any of those.
- **Done when:** a bag of one full course flight exists, and `vio_node`
  logs every frame.

**V2. Get fixes through the blind turn (vision).** This is the blocker.
- [ ] Run `vision/color_gate_detector.py` + `vision/gate_pose.py` frame by
  frame on the bag from V1, against the true gate poses. During the gap,
  what does the detector see of gate_1? A clipped frame? A side-on view
  (31–46°)? Nothing?
- [ ] Get a measurement from a **partly visible** gate: 2–3 corners, or
  the frame's edges, even if only a position or bearing fix.
- [ ] Check what a wider camera would do. In Isaac, the spawn's HFOV can be
  changed (`workspace/spawn_example.py`); what does the real Starling 2 Max
  camera have?
- **Done when:** on the bag, gate fixes (any kind) arrive at **at least 10 Hz
  for the whole +5.6 to +10 s window**.

**V3. Stop biased fixes from clipped gates (vision).**
- [ ] At the start hover, gate_0 is about 10° above the camera axis at
  4.6 m, and its top edge sits about 1° out of frame. In attempt 3 its
  fixes were consistent but biased, and they pulled the VIO off course
  (drone missed gate_0 by 0.95 m). Reject quads that touch the image
  border, or flag them as unreliable.
- **Done when:** no fix is produced from a gate that isn't fully in view.

**V4. Tune the filter offline, on the bag (VIO).**
`control/vio.py` has new knobs (section 1.3). In flight they trade one
error for another:
- [ ] `accel_rw` (process noise for unmodelled acceleration, default 0.1):
  - Good: it made the reported uncertainty honest in the gap (error/std
    3.6 → 1.4).
  - Bad: it made the **velocity estimate 2–3× noisier while fixes flow**
    (0.2–0.37 m/s against 0.10 m/s with `accel_rw` = 0), and that velocity
    error is what drives the drift in the gap.
  - Sweep 0 / 0.03 / 0.05 / 0.1 by replaying the bag through `GateVIO`.
- [ ] Measurement noise of far full-pose fixes: gate_2 at 9–13 m gives most
  of the fixes, and some are off by chi² in the thousands (attempt 3,
  +2.4 to +3.8 s). `_stds()` uses 0.004·range² for position and the PnP
  normal std for angles. Check them against the bag's truth.
- **Done when:** on the bag, the velocity error is at or below 0.1 m/s when
  the gap starts, and error/std is at or below 2 in the gap.

**V5. Make the rehearsal behave like Isaac (`Trajectory/simulate_vio_mpc.py`).**
Its camera already needs the whole gate in frame (it reproduces the gap:
VIO max 0.76–0.98 m). But it lacks Isaac's other effects:
- [ ] PnP noise and bias that grow with range, as measured in V4.
- [ ] Unreliable normals (position-only fixes): it never produces them, so
  `pos_only_max_range` can't be tested offline.
- [ ] Clipped gates producing biased fixes (V3).
- **Done when:** the rehearsal's VIO error and velocity error match the bag
  within a factor of 1.5, so VIO changes can be judged offline first.

**Related, not VIO (planner / team):** the circle course (`Trajectory/make_circle_course.py`,
checked with `Trajectory/check_course_camera.py`) does this by layout; fly it next. Otherwise, a path that keeps a gate whole in
frame through the turn. For example, an aligned approach of 3 m or more to
each gate, or a planner check that flags stretches over 1 s with no usable
gate (`Trajectory/gate_path.py`). Faster speeds (3/3, 4/4) are pointless
until V2 is done: the rehearsal's VIO hits 1.40 m at 3/3.

### 1.3 What is already in the VIO (2026-11-06, `control/vio.py`, tests in `control/test_vio.py`, 11/11)

| Change | Parameter | Why | Verdict in flight |
|---|---|---|---|
| Unmodelled acceleration in the process noise | `accel_rw=0.1` (in `Kalman_Filter_MPC.py`, default 0 there) | std was 3.5× too small in the gap, so good fixes were rejected | honest std, but noisier velocity: **retune (V4)** |
| Re-lock: accept consistent rejected fixes | `relock_n=5`, `relock_window_s=0.5`, `relock_spread_m=0.3`, `relock_max_range=10` | a drifted VIO never recovered | works (recovers to about 0.1 m) |
| ...only after a blind spell | `relock_blind_s=1.0` | attempt 3 re-locked to biased gate_0 fixes over a healthy estimate | worked in attempt 4 |
| Drop far position-only fixes; weight the rest by range × attitude std | `pos_only_max_range=8.0` | attempt 2: 91 position-only fixes of gate_2 at 9–11 m moved the VIO 0.64 m sideways | worked in attempts 3–4 |

### 1.4 The four flights (Stage 7, VIO in charge, 2/2, exact gate map, look-at yaw, closed lap)

Run folders: `Trajectory/isaac_results/3_gates/vio_mpc/run_*`. Each has:
- `flight_log.csv` (truth, VIO, reference, setpoints at 50 Hz);
- `vio_events.csv` (every fix and rejection);
- `bridge_log.csv`, `monitor_events.txt`, `replan_events.csv`;
- figures from `scripts/plot_vio_flight.py`.

**No CSV under `Trajectory/isaac_results/` is in git** (`.gitignore`). Get
them from the pod: `/workspace/vio_runs/s7r*_*` and `run_045716` (these
also have `vio/vio_log.csv`, the VIO state every IMU sample), or from
whoever ran the flight.

| Attempt | VIO changes flown | gate_0 | gate_1 | Velocity error at gap start | Abort | Folder |
|---|---|---|---|---|---|---|
| 1 | original | through, 0.27 m off | missed (beside it) | 0.10 m/s | +10.5 s, VIO 1.07 m | `run_s7r_004638` |
| 2 | + `accel_rw`, re-lock | through, 0.64 m off | missed | 0.20 m/s | +9.46 s, 1.13 m | `run_s7r2_012950` |
| 3 | + far position-only cap | **missed** (re-lock to clipped gate_0) | **through**, 0.42 m margin | 0.19 m/s | **+13.47 s**, 1.02 m | `run_s7r3_014449` |
| 4 | + blind-spell rule | through, 0.47 m off | not reached | 0.28–0.37 m/s | +7.84 s, 1.14 m | `run_s7r4_015739` |

Also `run_045716_s4`: Stage 4 on ground truth, with the VIO running
alongside (same gap, 13 m drift by the end, no watchdog).

Each flight is a single sample, so some of the differences are luck. **The
gap is in every run.**

---

## 2. MPC / bridge follow-ups (not blocking)

- [ ] **Steady 0.12 m hover offset.** PX4 flies about 0.3° short of the
  commanded pitch, and the MPC has no integral term. Add a slow, bounded
  position-error integrator in `mpc_node`, frozen when not in OFFBOARD.
- [ ] **`mpc_node` latches its hold point at its first state.** Start it
  after the VIO has settled (the runbook does), or make it wait for the
  monitor's GO.
- [ ] **Hover thrust 0.58** is measured for the Iris. Make it the bridge's
  default, or read PX4's hover-thrust estimate.
- [ ] **Attitude lead 0.1 s** fixed the hold. Measure PX4's real attitude lag
  with a pitch step test and set `attitude_lead_s` and the rehearsal's
  `--attitude-tau` from it.
- [x] **`min_snap.fit_to_limits` failing to converge from a moving start**
  (a 228 s plan; on 2026-11-06 a 7×10¹² s plan and a MemoryError). It now
  stretches at most 4× per pass and gives up at 20× (`ValueError`, so the
  replanner keeps the current plan). Test:
  `Trajectory/test_min_snap_planner.py`. Why the fit diverges from some
  moving starts is still open.
- [x] **Gate map duplicated a prior gate more than 1 m off** (Stage 4R
  prior seeds). It now also matches within the gate's 99.7 % uncertainty
  (`Trajectory/gate_map.py`, `gate_chi2`). Test:
  `Trajectory/test_replan_session.py`.
- [ ] **Minor:** start the bridge's spin thread inside the `try`.

## 3. Running a flight (what worked on 2026-11-06)

1. **Pod:** after a restart, redo Stage 0 (packages, the one-thread BLAS
   line in `~/.bashrc`, the `drone_msgs` build). `ros2-eval` is limited to
   **2 CPUs**: VIO + monitor + MPC + planner fit, and adding the camera
   gate-pose node pushes MPC solves over 15 ms.
2. **Isaac:** fresh process, Flat Grid, `spawn_example.py` with
   `CAMERA_PITCH_DEG=0`, Play, PX4, `commander takeoff`, `sim_clock.py`,
   gates.
3. **Nodes:** `vio_node` + `flight_monitor` first. Once the VIO has
   settled, start `mpc_node` (odometry) and `replan_node` (odometry,
   `prior_sigma 0`, `use_camera false`, `yaw_mode next_gate`). Wait for
   `plan ready` and `PREFLIGHT: GO`, then run the bridge
   (`--hover-thrust 0.58 --max-duration 120 --max-tilt-deg 30`).
4. **Between attempts:** at `pxh>`, run `param set RTL_RETURN_ALT 5`,
   `param set RTL_DESCEND_ALT 5`, then `commander mode auto:rtl`, then
   `commander takeoff`. A straight line home clips gate_1. Restart the
   nodes for each attempt (the watchdog latches).
5. **After:** stop `vio_node` with Ctrl-C (SIGINT) so it writes
   `vio_events.csv`, then copy the run folder back.

## 4. Housekeeping

- [ ] Commit the figures and `vio_summary.txt` of runs 3 and 4. Leave the
  CSVs out (several MB each).
- [x] Reconcile the Starling 2 Max spawn and dynamics (2026-11-06).
  `workspace/spawn_starling.py` (C29 camera) is the spawn. The
  `starling2max` profile and `px4/starling2max_sitl_params.sh` match it,
  and `control/test_vehicles.py` checks all three (it passes again).
  - From Tony's `vehicle/starling2max-sim`, ported: the `THR_MDL_FAC`
    thrust chain (`control/vehicles.py`, `px4/attitude_bridge.py`), the
    SITL parameter script, the motor-lag class and tests.
  - Not ported: his `rl/` thrust maps and `spawn_starling2max.py` (a
    duplicate spawn, older camera).
  - Inertia and κ follow `STARLING2_DYNAMICS.md` (CAD; propeller theory,
    0.0129 m) instead of his scaled estimates.
  - Tony: merge or close that branch with this in mind.
