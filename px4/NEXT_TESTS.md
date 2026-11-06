# What to fix before the next Isaac tests (after 2026-10-29)

Where we stopped (`px4/VIO_MPC_TEST_PLAN.md`):

| Stage | Result |
|---|---|
| 0. Pod prep | done. Lost on a container restart: redo it |
| 1. Isaac + PX4 | GO |
| 2. Dry run | GO |
| 3. Hover hold, truth | **pass** after 3 fixes (bridge prompt, hover thrust 0.58, attitude lead 0.1 s): within 0.19 m, altitude ± 0.01 m |
| 4. Course, truth, 2/2 | **pass**: all gates, margins 0.47–0.71 m, tracking 0.12 m mean / 0.41 m max. Video: `Downloads/isaac_stage4_course_truth.webm` (not in git) |
| 5. VIO open-loop | **fails** (from Stage 4's side-by-side VIO): lost the gates after gate_0, 13 m error by the end |
| 6–8 | blocked by 5 |

Logs: `/workspace/vio_runs/run_045716/` in the pod (`s3`, `s3b`, `s3c`, `s4`,
`vio/`). Copy back before the PVC is cleaned (`VIO_MPC_TEST_PLAN.md`, "What
every run leaves").

---

## 1. VIO: must fix before Stage 6/7

What the Stage 4 log shows (`vio/vio_events.csv`, times from engage):

- **Fixes in flight:** 147 full fixes on gate_1 and 260 on gate_2; none on
  gate_0.
- **Last full fix:** +4.83 s, just after passing gate_0 (+3.73 s).
- **Rejections:** 69, all of gate_2, from +2.23 s to +9.36 s, chi² 34 to
  4032. The first ones came while the VIO was still within 0.15 m of the
  truth.
- **From +4.83 s to +8 s, and after +9.36 s:** no logged events, although
  gate_2 (passed at +12.2 s) was ahead. This does **not** prove the gates
  weren't detected: `vio_events.csv` has no row for a frame with no detection,
  a failed PnP, or a detection that didn't match a map gate. The drifted
  estimate makes that last one plausible after about +10 s (error 1.5 m and
  growing; the match tolerance is 1 m + 10 % of range). The only other hint is
  indirect: camera time fell to 5-6 ms/frame.
- **Labels:** the gate names are the VIO's own matches, not ground truth.
- **Geometry (`vio_mpc/run_045716_s4/fig5_gate_visibility.png`):** 315 of the 316
  in-flight fixes fall in the moments when the WHOLE gate is in frame, by
  calculation from the true pose. gate_0 is never whole in frame (its top edge is
  clipped even at hover), and it has no fixes. From +4.8 s to +8.4 s no gate is
  whole in frame. gate_2 is whole in frame only once after that, +8.4 to +9.4 s,
  and every fix then was rejected (VIO already about 1 m off).
- **Dead reckoning:** the error grew from 0.26 m to 1.1 m (gate_1) to 2.7 m
  (gate_2) to 13 m. The VIO's own std stayed far below its real error
  (0.17 m reported against 1.13 m actual).

### 1a. Find out what the camera saw (do first, it drives the rest)
- [ ] **Record the camera in Stage 5:** `ros2 bag record
  /iris_0/front_cam/rgb /iris_0/front_cam/camera_info /drone00/state/pose`.
  The VIO gets fed the frames, but nobody saves them.
- [ ] Log every frame in `vio_node`: detections found, PnP failures, unmatched,
  fixes and rejections. Today the unmatched count isn't logged in flight.
- [ ] Run the detector and PnP offline on the bag, frame by frame, against
  the true gate poses. That answers four questions:
  - Is gate_1 detected from +4.8 s to +8.6 s? If not, is that the side-on
    approach, then being too close? (Untested guess: a 1.5 m frame at 1.2 m
    doesn't fit in the 60° field of view.)
  - Is gate_2 detected after +9.4 s? If so, was it dropped as unmatched?
  - Why were early gate_2 fixes rejected (chi² 34–58) while the VIO was
    accurate? It could be a range bias at about 11 m, the gate seen through
    or behind another frame, or a wrong association.
  - Why are there no gate_0 fixes, even at hover? It may be clipped by the
    top of the frame (it looks that way in the onboard view). Check that and
    the association in `control/vio.py` (`assoc_tol`).

### Update 2026-11-06: first flight with the VIO in charge, and 1b / 1c done

**Flight** (Stage 4R, VIO in charge, logs and figures in
`Trajectory/isaac_results/3_gates/vio_mpc/run_s7r_004638/`):
- passed gate_0 for real;
- from +5.6 s to +10.0 s no gate was whole in frame, and the VIO drifted
  0.17 → 1.06 m, while it reported only 0.30 m of uncertainty;
- it passed beside gate_1;
- gate_2's fixes at +10 s were rejected;
- the watchdog stopped the flight at +10.5 s.

The drift, fitted over the gap: 0.17 m position error and 0.11 m/s velocity
error when the fixes stopped, plus about 0.04 m/s² from tilt error.

**1b and 1c are now in `control/vio.py`:**
- `accel_rw=0.1` (process noise for unmodelled acceleration, in
  `Kalman_Filter_MPC.py`);
- re-locking after 5 rejected full fixes of one gate that agree within 0.3 m.

Tests (`control/test_vio.py`):
- the Isaac gap gives error/std **3.4** without the term and **0.79** with it;
- a VIO 1 m off re-locks to 0.001 m;
- disagreeing fixes never re-lock.

Rehearsal at 2/2:
- VIO max 0.98 → 0.76 m;
- tracking max 0.88 → 0.75 m.

At 3/3, VIO max is 1.40 m (all gates in the rehearsal, but over the 1 m
watchdog). **Still open:** the drift itself (1a, 1d). These changes let the
VIO *recover* at gate_2; they don't keep it under 0.5 m at gate_1.

**Attempt 2, the same day, with 1b + 1c** (`vio_mpc/run_s7r2_012950/`): worse,
aborted at +9.46 s (VIO 1.13 m). Its uncertainty was honest in the gap
(error/std 1.4, against 3.6 before), but the wider uncertainty also made the
filter accept **91 position-only fixes of gate_2 at 9-11 m**. Those go through
the estimated attitude, and they moved the estimate 0.64 m sideways at +3 s,
so the drone crossed gate_0 0.64 m off-centre (margin 0.11 m). The velocity
error at the start of the gap was also about 2x larger (0.2 m/s).

**Fix:**
- position-only fixes from farther than 8 m are dropped (`pos_only_max_range`);
- the remaining ones carry range x attitude uncertainty in their noise
  (`control/test_vio.py`, `test_far_position_only_fixes_are_dropped`).

The rehearsal can't test this: its camera always reports reliable normals.
Only Isaac can.

**Attempt 3, far position-only fixes dropped** (`vio_mpc/run_s7r3_014449/`).
The furthest yet:
- first true pass of **gate_1** (margin 0.42 m);
- gap drift 0.12 m/s;
- recovery to 0.11 m once gates were back;
- watchdog at +13.47 s.

But it **missed gate_0** (0.95 m off). At +0.27 s a re-lock took gate_0's
consistent but biased fixes (half clipped at the start) over a healthy
estimate (0.09 m off). That set off a drift, and gate_2's far fixes then
looked wrong and were rejected.

**Fix:** a re-lock now needs a blind spell first: no fix accepted for 1 s
(`relock_blind_s`, test `test_no_relock_while_fixes_flow`).

### 1b. Honest uncertainty while dead-reckoning (`control/vio.py`) — done 2026-11-06
- [ ] Tune the process noise (`accel_noise`, plus a velocity random walk) so
  that the reported std covers the real error when fixes drop out. Target:
  error / std < 3 for 95 % of samples during a 6 s gap.
- [ ] Add a test in `control/test_vio.py`: fly the course, drop all fixes for
  6 s, then check the std grew to match and that the first fix after the gap
  is accepted.

### 1c. Recovery after the filter has drifted (`control/vio.py`) — done 2026-11-06
- [ ] Re-localize after N consecutive rejections of consistent fixes (say 5
  within 1 s, agreeing with each other). Accept the fix and inflate the
  covariance instead of rejecting forever.
- [ ] Test: start with a 2 m error and a falsely small std; the filter must
  converge within 1 s of seeing a gate.

### 1d. Keep a gate in view on every leg (planner)
- [ ] The path from gate_0 to gate_1 goes sideways (-y) and swings into
  gate_1 at about 1.2 m. Options, simplest first:
  - Approach waypoints further out: an aligned approach of 3 m or more
    before each gate (`gate_path.plan_gate_trajectory` pre/post distances).
  - A look-at yaw that also counts the gate after next as "in view".
  - Change the course layout so consecutive gates face each other more.
- [ ] Add a planner check: for every point of the plan, which gate is in
  the camera's field of view, and at what range and viewing angle. Flag
  stretches longer than 1 s without a gate that is usable (whole frame in
  view, viewing angle within about 45° of head-on).

### 1e. Make the rehearsal predict Isaac (`Trajectory/simulate_vio_mpc.py`)
- [ ] `SimCamera` sees ideal corners from any angle, which is why the
  rehearsal passed. Add the limits the real detector shows (from 1a):
  - a maximum viewing angle;
  - frames that must be fully in view;
  - a minimum range;
  - occlusion by other frames.
- [ ] Then re-check 2/2, 3/3 and 4/4 offline before flying them.

---

## 2. MPC / bridge follow-ups (not blocking Stage 7 at 2/2)

- [ ] **Steady 0.12 m hover offset.** PX4 flies about 0.3° short of the
  commanded pitch, and the MPC has no integral term. Add integral or
  disturbance estimation in `mpc_node` (a slow position-error integrator,
  bounded and frozen when not in OFFBOARD). Alternatively, estimate the
  bias at hover and subtract it.
- [ ] **`mpc_node` latches its hold point at its first state.** On VIO
  that's before the filter settles (0.3 m low in Stage 2). Wait for the
  state to be steady, or for the monitor's GO. Until then, restart
  `mpc_node` after the VIO settles.
- [ ] **Hover thrust 0.58** is measured. Consider making it the bridge's
  default instead of a flag, or reading PX4's hover thrust estimate over
  MAVLink.
- [ ] **Attitude lead (option A) costs tracking.** Isaac's tracking was
  better than the PX4-like model predicted, so the real attitude loop is
  probably faster than 0.08 s. Measure it with a pitch step test (a small
  bridge test mode), set `attitude_lead_s` and the rehearsal's
  `--attitude-tau` from the data, and re-check 3/3 and 4/4. If tracking
  degrades with speed, take option B (an MPC with attitude or rate inputs
  that models PX4's loop) to the team.
- [ ] **Minor:** in `mpc_offboard_bridge.py`, start the spin thread inside
  the `try`. Today, if `connect()` raises, the thread isn't joined.

---

## 3. Next session's run order

1. **Pod:** if the container restarted, redo Stage 0 (`casadi`, `scipy`,
   `matplotlib`, the one-thread BLAS line in `~/.bashrc`, the `drone_msgs`
   rebuild).
2. **Isaac:** fresh process, Flat Grid, then the spawn with
   `CAMERA_PITCH_DEG=0`, Play, PX4, `commander takeoff`, `sim_clock.py`, the
   gates. This took about 10 minutes on 2026-10-29.
3. **Stage 5 with a bag recording**, to get the data for 1a.
4. **Offline:** 1a, then 1b–1e, with the new tests passing.
5. **Stage 5 again:** the VIO must stay under 0.6 m max over the course.
6. **Stage 6:** hover hold on the VIO, restarting `mpc_node` after the VIO
   settles.
7. **Stage 7, recorded:** the goal.

## 4. Housekeeping

- [ ] Copy `run_045716` back to
  `Trajectory/isaac_results/3_gates/vio_mpc/`. Commit only the summaries and
  plots, and link the video.
- [ ] Write an update next to `PX4_FLIGHT_UPDATE_2026-10-23.md` for
  2026-10-29 (Stages 0–4 and the findings above).
