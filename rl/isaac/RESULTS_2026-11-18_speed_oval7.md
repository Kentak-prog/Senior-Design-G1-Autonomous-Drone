# Top-speed course `speed_oval7`: 14.0 m/s in Isaac (2026-11-18)

## The course (`rl/tracks/speed_oval7.json`)

Seven 1.5 m gates, all at 2.5 m, in a stadium: a 28 m start straight (gates 0–2), a 6 m-radius right
turn (gate 3), a 22 m back straight (gates 4–5), a second turn (gate 6) back toward the start. Footprint
about 46 m × 18 m.

**Reference.** UZH's Swift race (Kaufmann et al., Nature 2023) used seven square gates in a 25 m × 25 m
hangar at Dübendorf, with a Split-S, at over 100 km/h (≈ 28 m/s; press figure, peak not found in the
sources checked). UZH's perception-aware time-optimal planning work reports real flights up to 9.8 m/s on a
Split-S track. `speed_oval7` keeps the seven gates but trades Swift's tight 3-D layout for straights,
because the Starling's limits (`control/vehicles.py`: 15 m/s, 15 m/s², 57° tilt, 17.8 N thrust cap)
need ~6 m to reach 12 m/s at 12 m/s² and as much to slow for a turn.

## What it took

1. **Segment-time optimisation** (`--mpc-optimize-times`). `mpc_rate` planned with distance-proportional
   segment times and one uniform rescale. Every gate has approach / centre / exit points 1 m apart, the
   rescale is set by the tightest of them, and the whole lap was slowed to fit: a 31.7 s plan peaking at
   8.9 m/s, ~2 m/s through every gate. `Trajectory/min_snap.optimize_segment_times` (already in the planner,
   off in `mpc_rate`) gives each segment its own share: **12.29 s, peak 11.95 m/s** at 12/12. It takes
   ~2 minutes for 7 gates, so the result is cached (`rl/mpc_controller.cached_plan`, `PLAN_CACHE_DIR`, a
   JSON recipe that moves between NumPy versions; the plans used are in
   `Trajectory/isaac_results/speed_oval7_2026-11-18/plans/`).
2. **A quaternion-sign bug in `mpc_rate` at a heading of ±180°.** On the back straight (heading −X) every
   run tipped over (tilt > 80°): when the yaw crosses ±180° the measured quaternion's sign flips (q and −q
   are the same attitude), the solver's warm start keeps the old sign, and its single SQP-RTI step
   saturated every rate command (±6 rad/s). Fixed by keeping the measured quaternion's sign continuous
   between ticks (`RateAwareMpc.act`). `test_mpc_rate.py` still passes (4/4).

## Results

Offline (QuadSim, Isaac rate loop, optimised plans, after the fix):

| Limits | Lap | Top speed | Max tilt | Smallest margin |
|---|---|---|---|---|
| 12/10 | 12.20 s | 11.91 m/s | 53° | 0.72 m |
| 12/12 | 11.18 s | 11.97 m/s | 54° | 0.70 m |
| 14/14 | 10.68 s | 13.96 m/s | 63° | 0.69 m |
| 15/15 | 10.86 s | 14.87 m/s | 64° | 0.70 m |

**Live in Isaac** (PX4 SITL, final C29 model, state = PX4's estimate, next-gate yaw):

| Limits | Result | Top speed | Max tilt | Smallest margin | Control ms mean / max |
|---|---|---|---|---|---|
| 12/12 | **7/7, 11.18 s** | **11.98 m/s** | 55° | 0.65 m | 3.5 / 11.2 |
| 14/14 | **7/7, 10.68 s** | **14.00 m/s** | 63° | 0.66 m | 3.5 / 13.4 |

- **New top speed: 14.0 m/s** (previous best in Isaac: ~4.8 m/s, set by the old plan timing, not the drone).
- Live matches offline to the hundredth of a second.
- 14/14 tilts to 63°, past the profile's 57° `tilt_limit` (chosen to keep thrust margin; altitude can be
  held to ~72° at the PX4 cap). 12/12 stays at 55°. On hardware, start at 12/12 or below.

GIF: `Trajectory/isaac_results/speed_oval7_2026-11-18/flight_realtime.gif` (speed and tilt panels).
Videos (chase camera, Downloads): `isaac_mpc_rate_v12a12_speed_oval7.webm`,
`isaac_mpc_rate_v14a14_speed_oval7.webm`.

## On the Swift VIO (2026-11-18, later)

`ros2_nodes/swift_vio_node.py` (emulated base VIO + drift filter, fixes ≤ 6 m, 17 ms image delay
compensated), drift seed 2: the base VIO was **0.89–0.95 m** off the truth for the whole flight.

| Limits | Estimate | Result | Top speed | Smallest margin | Flown-on error vs truth, mean / max |
|---|---|---|---|---|---|
| 12/12 | **Swift** | **7/7, 11.16 s** | **12.05 m/s** | 0.46 m (GPS: 0.65) | 0.16 / 0.54 m |
| 12/12 | no filter | **hit gate 1** | 11.37 m/s | — | 0.91 / 0.94 m |
| 14/14 | **Swift** | **7/7, 10.66 s** | **14.06 m/s** | 0.31 m (GPS: 0.66) | 0.19 / 0.50 m |

- **The Swift VIO flies the top-speed course at GPS lap times** (11.16 s and 10.66 s against 11.18 s and
  10.68 s), with 0.9 m of drift removed. Without the filter the same drift hits gate 1 even on the 1.5 m
  gates.
- **The cost is margin:** 0.2 m less at 12/12 and 0.35 m less at 14/14. The estimate error at speed
  (0.16–0.19 m mean) is 3–4× what it was at circle speeds (0.04–0.06 m).
- **The image delay is larger than compensated.** The estimate still sits ~0.1 m behind along the
  velocity; the residual fit is 12–13 ms on top of the 17 ms compensated, so ~30 ms in total at these
  speeds. Raising `image_delay_s` toward 0.03 is the obvious next try.
- No-filter run: the runner's VIO checks were loosened (`--vio-start-err 1.5 --vio-max-err 2.5`) so it
  would start at all.

GIF: `Trajectory/isaac_results/speed_oval7_2026-11-18/flight_realtime_vio.gif`. Logs in `.../vio/`.
Videos (Downloads): `isaac_mpc_rate_swiftvio_v{12a12,14a14}_speed_oval7_s2.webm`,
`isaac_mpc_rate_nofilter_v12a12_speed_oval7_s2.webm`.

## Drift first, or speed first?

These runs used PX4's estimate (sim GPS), not the Swift VIO. What speed does to the VIO:

- The emulated drift grows with distance flown (2 %), so per second it is ~3× faster at 14 m/s than at
  4.8 m/s, while each gate is in fix range (≤ 6 m) for only ~0.4 s and the straights leave 1–2 s between
  gates.
- The 17 ms image delay is 0.24 m at 14 m/s: compensated, but any error in it matters 3× more.
- 1.5 m gates leave ~0.5 m of room for error at the crossing; the pink gates leave 0.18 m.

Recommendation: **not to try to remove the drift first.** The drift cannot be removed, only estimated,
and the filter already does that. Two cheap next steps instead:

1. ~~Fly `speed_oval7` on the Swift VIO at 12/12~~: done (above), 7/7 at 12/12 and 14/14; the filter keeps
   up, at the cost of 0.2–0.35 m of margin.
2. Size the drift from a real VOXL 2 bag before trusting any VIO result at speed (`SWIFT_VOXL2_PLAN.md`,
   step 1). That is the real unknown: our 2 % is an estimate.

The pink gates at 12+ m/s are a later step.
