# Swift-style state estimation on the Starling 2 Max (VOXL 2): proposal

**Status:** proposal for the team, 2026-11-14. Nothing here has been built on the drone yet. VOXL SDK
names (services, pipes) are from ModalAI's documentation as we know it: check them on the drone first.

## TL;DR

- The MPC (`mpc_rate`) flies fine on a good state. On our own gate-only VIO in Isaac it failed until we
  (1) stopped gates from setting the tilt and (2) pointed the camera at the next gate. Then it flew
  5/6 runs, including the 3-gate course at 8/10 in 5.48 s, as fast as on GPS.
- What still limits us: between gates our VIO has **only the IMU**. Swift (the drone that beat the world
  champions) never has that problem because it runs a **feature-tracking VIO** underneath and uses the
  gates only to correct that VIO's drift.
- The Starling's VOXL 2 already runs a feature-tracking VIO. **Proposal:** use it as the base, add a small
  gate-based drift filter on top (Swift's design), and fly `mpc_rate` on the corrected state.

## How Swift does it (Kaufmann et al., *Champion-level drone racing using deep reinforcement learning*, Nature 2023)

Swift's state estimate has two layers.

**1. A base VIO that never stops.** An Intel RealSense T265 tracking camera (two fisheye cameras + an
IMU + its own processor) runs visual-inertial odometry on board, at 100 Hz. It tracks generic visual
features of the whole room (corners and texture of walls, floor, equipment) from frame to frame and fuses
them with its IMU. It knows nothing about gates. Its attitude is good; its position slowly drifts,
badly at racing speed (motion blur).

**2. Gates correct that VIO's position drift.**
- A small neural network (U-Net) finds the four corners of every visible gate in the T265's images
  (about 40 ms per frame on the Jetson TX2).
- IPPE, a pose solver for flat square targets, turns the corners into the camera's pose relative to the
  gate; each sighting is matched to the nearest gate in the known track map, which gives an absolute
  position.
- A Kalman filter estimates **the VIO's error**, not the drone's position: the drift offset `p_d` and
  its rate `v_d`. Each gate sighting measures the drift (VIO position minus gate-derived position).
- Output = VIO position − `p_d`, VIO velocity − `v_d`. **Attitude (roll, pitch and yaw) comes from the
  VIO only;** gates never touch it.

So between gates Swift is never blind: the base VIO keeps flying, the drift estimate is carried forward,
and every gate sighting pulls the drift back.

```
T265 (IMU + feature-tracking VIO) ──► pose, velocity @100 Hz ─────┐
                                                                   ├─► minus estimated drift ─► controller
T265 images ─► gate-corner network ─► IPPE ─► drift Kalman filter ─┘   (position, velocity only)
```

They also measured the real VIO's error against motion capture (about 50 s of flight), modelled it with
Gaussian processes, and fine-tuned the policy in simulation with that error injected, so it learned to
fly well on a drifting estimate.

## What we found in Isaac (2026-11-13 / 14)

Details: `rl/isaac/RESULTS_2026-11-13_mpc_rate_vio.md`.

| Step | 3-gate course (3/4, 5/7, 8/10) | What it showed |
|---|---|---|
| Our VIO as it was (IMU + gate fixes, gates set attitude) | 1/3 | the first gate fix left the pitch ~3° off; nothing corrected it; gravity leaked into the velocity |
| + tilt from gravity, gates correct position + yaw only (Swift's rule) | 0/3 | tilt fixed, but a gyro bias taken mid-turn now integrated unchecked; blind stretches still drifted 1 m |
| + gyro calibrated only when still, slow tilt correction | 0/3 (best 2/3 gates) | clean starts; still lost ~1 m in the 1.5 s blind stretch near gate 2 |
| + nose (camera) on the next gate | **3/3** (8.68 s, 6.54 s, 5.48 s) | longest blind stretch 1.28 s → 0.80 s; VIO stayed within 0.30–0.53 m |

The circle course went 2/3 with the same setup. Without a base VIO, error grows with the square of the
time without a gate fix, so shortening the blind stretches mattered most. A base VIO like Swift's removes
the blind stretches almost entirely.

## Plan on the VOXL 2

```
tracking camera(s) + IMU ─► ModalAI VIO (voxl-open-vins or qVIO) ──► pose/vel ──────────────┐
                                  └─► voxl-vision-hub ─► PX4 EKF2 (its own hold / failsafes) │
hires front camera (IMX412) ─► gate detector ─► IPPE ─► gate fix ──► drift filter ◄─────────┘
                                                                         │ VIO − drift
                                                   mpc_rate (acados, ARM) ─► SET_ATTITUDE_TARGET ─► PX4
```

| Piece | What it is | Status |
|---|---|---|
| Base VIO | ModalAI's feature-tracking VIO on the tracking camera(s) + IMU: `voxl-open-vins` (OpenVINS) in recent SDKs, `voxl-qvio-server` in older ones. Already forwarded to PX4 by `voxl-vision-hub`. | on the drone; verify which one, its rate and quality |
| Gate detector | hires front camera (IMX412, the one modelled in Isaac) via `voxl-camera-server`; our color detector first, the gate CNN later (`voxl-tflite-server` can run it on the GPU/NPU) | detector exists (sim); untested on real gates |
| Pose solve | `vision/gate_pose.py`, or OpenCV `SOLVEPNP_IPPE_SQUARE` (Swift's IPPE) | exists |
| **Drift filter** | **new, ~100 lines**, see below | to write |
| Controller | `rl/isaac/mpc_rate.py` (acados, plain C → builds for ARM), `--mpc-yaw next_gate` | exists; benchmark on the VOXL |
| Link to PX4 | `SET_ATTITUDE_TARGET` (thrust + body rates) through `voxl-mavlink-server`, as in Isaac | exists (runner) |
| ROS 2 bridge | `voxl-mpa-to-ros2` puts VOXL pipes (VIO, cameras) on ROS 2 topics, where our code lives | on the drone; verify |

**Safety:** PX4 keeps fusing the *raw* VIO, so if our side aborts or stops, PX4's hold mode still has a
sane estimate (as in Isaac today, with GPS replaced by VIO).

### The drift filter

- **State** (6): `p_d` (VIO position minus truth) and `v_d` (its rate), world frame.
- **Predict:** `p_d += v_d dt`, `v_d` constant; process noise sized from recorded VIO drift.
- **Update** at each reliable gate fix (image time `t_img`):
  1. look up the VIO pose at `t_img` (keep ~1 s of VIO history; all VOXL pipes share one clock);
  2. gate position in the camera (PnP) → drone position in the world, put through the **VIO's attitude**
     at `t_img` (what our Swift-mode VIO already does);
  3. measured drift `z = p_vio(t_img) − p_gate`; chi-square check; Kalman update of `p_d`, `v_d`.
- **Output:** VIO position − `p_d`, VIO velocity − `v_d`; attitude and body rates straight from the VIO /
  IMU. Publish as `nav_msgs/Odometry` so `runner.py --state vio` (`rl/isaac/vio_state.py`) can read it.
- Range cap (~9 m) and gate association as in `control/vio.py`.

### Order of work

1. **On the drone, no flying.** Confirm the VIO service, its rate and its timestamps against the hires
   camera's. Record a ROS 2 bag (VIO, IMU, hires images) while carrying or slowly flying the drone through
   real gates. That tells us whether the color detector sees real gates and how fast the real VIO drifts.
2. **In Isaac, before hardware.** Emulate the base VIO: PX4's estimate plus a slowly drifting error
   (random walk in position and velocity, sized from the bag). Write the drift filter, feed it our gate
   fixes, fly `mpc_rate` on its output on the 3-gate and circle courses at 3/4, 5/7, 8/10. Everything but
   the VOXL itself gets tested here, with the setup we already have.
3. **On the VOXL 2.** Build acados for ARM and benchmark `mpc_rate` (target: a few ms per 20 ms tick);
   measure the gate detector's frame rate on the hires feed.
4. **First flights.** Hover hold on the corrected state, then the 3-gate layout at 3/4, stepping up only
   after clean runs (the same ladder as in Isaac).

### Open questions to check on the drone

- Which VIO does our SDK version run (`voxl-inspect-vio`), at what rate, and which tracking camera(s) does
  the C29 configuration have?
- Are the VIO and hires camera timestamps on the same clock, and what is the image latency?
- Does the color detector find our real gates under the real lighting, or do we need the CNN now?
- CPU budget: VIO + detector + MPC + bridge on the VOXL 2 together.

## References

- Kaufmann, Bauersfeld, Loquercio, Müller, Koltun, Scaramuzza, *Champion-level drone racing using deep
  reinforcement learning*, Nature 620, 2023 (Swift; Methods: "Gate detection", "VIO drift estimation").
- Hanover et al., *Autonomous Drone Racing: A Survey*, IEEE T-RO 2024 (Sec. III-A: VIO + gate
  localization; perception-aware planning).
- Our results: `rl/isaac/RESULTS_2026-11-13_mpc_rate_vio.md`, `rl/isaac/README_mpc_rate.md`.
