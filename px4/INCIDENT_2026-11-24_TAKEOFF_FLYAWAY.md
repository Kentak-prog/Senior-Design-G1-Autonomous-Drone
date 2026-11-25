# Incident report: autonomous takeoff fly-away and crash (2026-11-24)

**Vehicle:** Starling 2 Max (VOXL 2, PX4, VIO-based position)
**Outcome:** frame damaged on impact; onboard electronics OK; no injuries.
**Status:** root cause narrowed to PX4's height estimate; the exact setting is not yet identified.
**Source:** QGroundControl telemetry log `2026-11-24 14-55-05.tlog` (PX4 messages, modes,
position estimate, barometer altitude, RC input).

## Summary

During the first autonomous takeoff (`px4/takeoff_hover.py`, 1.0 m climb at 0.3 m/s), the drone did
not stop at 1 m: it climbed to about 5.5 m and crashed. **PX4's height estimate did not follow the
drone:** it believed the drone was sinking (to 1.2 m *below the floor*) while the barometer showed it
climbing. To "stop sinking", PX4 kept thrust above the drone's weight. A similar estimate failure
(a 6.8 m "fall" while the barometer stayed flat) caused the pilot-flown crash earlier the same day,
before that day's sensor recalibration.

![Height, climb rate, thrust and rotor speeds for both flights](../flight_logs/2026-11-24/takeoff_incident_2026-11-24.png)

*Figure: `flight_logs/2026-11-24/plot_takeoff_incident.py` on the telemetry log (copied to
`flight_logs/2026-11-24/qgc_2026-11-24_14-55-05.tlog`). Thrust uses k_f = 4.383e-6 N/(rad/s)² from
the 2026-11-03 hover identification. QGroundControl's default rates are coarse (~1 Hz for RPM and the
position estimate, ~4 Hz for the barometer).*

## Timeline (seconds after arming)

| t (s) | PX4 height estimate | Barometer | Event |
|---|---|---|---|
| 0.0 | on the floor | on the floor | Armed by the script, OFFBOARD |
| 2.7 | at the floor | at the floor | "Takeoff detected" |
| 4.2 | 0.3 m below start, sinking | rising | **"Failsafe activated"** (PX4's estimator loses its horizontal position right after) |
| 5.1 | 0.7 m below, sinking | +0.1 m | PX4 switches itself to **Altitude mode** |
| 7.2 | **1.2 m below the floor** | +1.4 m | thrust stays ~1.3× weight |
| 8.8–9.0 | 0.8 m below | +3.7 m | pilot pulls the throttle stick to minimum |
| 10.2 | 0.5 m below | **+5.3 m** (peak +5.5 m at 10.7 s) | |
| 10.9 | — | falling | PX4 throttle reaches 60% |
| 11.3 | — | — | "Attitude failure (roll)": crash |

## Findings

- **Height estimate wrong.** On the autonomous takeoff the estimate moved down while the barometer
  moved up (~5.5 m in 9 s). Climb rates from the two don't correlate in either flight (r ≈ 0, panel c).
- **Thrust stayed above weight.** From the ESC speeds, thrust was 1.25–1.35× the drone's weight from
  4 s to 9 s (panel d): a sustained climb that PX4's controller kept commanding.
- **Earlier pilot flight: same kind of fault.** In that Altitude-mode flight (before the recalibration),
  PX4 claimed the drone was falling, down to 6.8 m below the start, while the barometer barely moved.
  It also ended in "Attitude failure (roll)". The gyro, accelerometer and level recalibration did not fix it.
- **One rotor ran fast.** ESC 1 spun ~2,100 RPM above the other three from 4 s on (panel e). This
  could be the attitude controller compensating for something (a damaged prop, a motor, or the drift
  after the horizontal position was lost); it needs checking before the next flight.
- **Why the RC "didn't respond".** The RC link was solid throughout (all 16 channels, RSSI 255), and the
  pilot's throttle-down is in the log. But Altitude and Position modes both use the same wrong height,
  so "descend" became more thrust. Only Stabilized/Manual mode or the kill switch bypass the estimate,
  and neither was used during this flight (see **Kill switch and RC takeover** below).
- **The script's commands were correct.** It asked for a climb, and PX4 *reported* the opposite motion.
  The script's 0.5 m error abort could not catch it, because it compares setpoints with the same wrong estimate.
- **Estimator warnings before arming.** "Yaw estimate error" and "heading estimate not stable" appeared
  repeatedly during the session before the flight.

## Kill switch and RC takeover

**How the transmitter is set up** (read from the RC channels in the log):

| Channel | Function | Positions |
|---|---|---|
| 3 | throttle | 1000–2000 |
| 5 | **kill switch** | 1011 off, 1988 **engaged**: it flips at all 9 "Kill-switch engaged" messages in the log |
| 6 | flight-mode switch | 1011 **Altitude**, 1500 **Position**, 1988 **Offboard** |

**The kill switch works.** It was toggled on the bench several times during the session, the last time
98 s before the autonomous arm (engaged, then released 0.9 s later). In the earlier pilot flight it
ended the flight: "Kill-switch engaged" → "Disarmed by kill-switch" in 0.3 s, about 5 s after the
attitude failure.

**In the autonomous flight it was never engaged.** For the full 11.4 s the log covers:
- Channel 5 (kill) stayed at 1011 (off).
- Channel 6 (mode switch) stayed at 1500 (Position). It was never moved, so PX4's own failsafe made
  the only mode change, to Altitude at 5.1 s.
- Only the throttle stick moved. It left center at 8.8 s and was at minimum from 9.0 s, apart from
  two single 5 Hz samples near the end (1300 at 10.6 s, and full throttle, 1986, at 11.0 s, as the
  drone came down).
- The RC link itself was solid: RSSI 255 throughout.

**Telemetry ends at 11.4 s, right at the crash.** If the kill switch was used after impact, this log
can't show it; PX4's onboard log (`21_49_16.ulg`) will.

**Why the kill switch was the only way out.** With PX4's height estimate wrong, every control that
goes through it made things worse or did nothing:
- **Mode switch:** its only positions besides Offboard are **Altitude and Position**, and both hold
  the same wrong height. There is no Stabilized or Manual position on the switch, so the pilot had no
  switch position that bypassed the estimate.
- **Throttle stick:** in Altitude mode it asks for a climb rate *relative to the estimate*; at
  minimum it asks to descend, which PX4 answered with more thrust.
- **The script's abort and a geofence:** both compare against the same estimate, so neither could see
  the climb.

**The trade-off.** Killing the motors drops the drone from wherever it is. The barometer shows it
passing 1 m (its target) at ~6.9 s and 2 m at ~7.9 s, versus the ~5.5 m it reached. A kill at the
first sign it was not stopping would have meant a 1–2 m drop instead of an impact from ~5.5 m.

**What to change:**
1. **Put Stabilized (or Manual) on the mode switch,** e.g. in place of Altitude at 1011, so there is a
   takeover position that does not depend on the position or height estimate. Keep Offboard on it.
2. **Kill-switch rule for every autonomous test:** the pilot's finger rests on the kill switch for the
   whole run. If the drone passes its target height, drifts, or does anything unexpected and doesn't
   respond to the mode switch within about a second, kill it. Agree on the rule before arming.
3. **Bench-test the kill switch before each flight** (already done in this session), and include a
   low-height kill in the takeover drill.
4. **Record PX4's onboard log and raise QGroundControl's telemetry rates,** so the moments after an
   incident are captured too.

## Likely cause (to confirm)

PX4 is following a height source that does not match reality, most likely the VIO relayed by
`voxl-vision-hub`, and trusting it over the barometer. Settings to check: `EKF2_HGT_REF`, `EKF2_EV_CTRL`,
the VIO frame and orientation passed to PX4, and recent changes to `voxl-vision-hub` / `voxl-qvio-server`.

**Data still on the drone:**
- PX4 logs `/data/px4/log/2026-11-03/21_49_16.ulg` (autonomous) and `21_49_43.ulg` (pilot). The date
  folder is the drone's wrong clock.
- The VIO log from the flight.
- PX4's parameters.

## Actions before the next flight

0. **Check the motor and prop on ESC 1** (and all props after the impact): spin-up on the bench,
   matching speeds at the same throttle.
1. **Props-off lift test:** with QGroundControl connected, lift the drone ~1 m by hand. The altitude must
   read **up**. Do not fly in any mode (Altitude/Position included) until it does.
2. **Find and fix the cause** from the PX4 logs and parameters; repeat the lift test.
3. **Transmitter and pilot drill:** add a **Stabilized** position to the mode switch; for unexpected
   behavior, switch to Stabilized or use the **kill switch**, not Position/Altitude. Test the kill
   switch on the bench before each session (see **Kill switch and RC takeover**).
4. **Script safeguards** (to add to `takeoff_hover.py` / `fly_min_snap_offboard.py`):
   - Refuse to arm unless PX4 reports a healthy estimator.
   - Abort if the estimated height and the barometer move in opposite directions. In this flight
     that would have fired ~1.5 s into the climb, at about 0.5 m.
5. **Restart autonomy low:** 0.5 m takeoff with `--end land` and the pilot ready, after a clean manual
   hover in Position mode.
