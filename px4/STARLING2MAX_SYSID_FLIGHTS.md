# Starling 2 Max: system-identification flights

**Goal.** Replace the estimates in `workspace/STARLING2MAX_DYNAMICS.md` with
measurements: yaw moment $k_m$, roll / pitch inertia, thrust above hover, drag,
and the motor lag as a first-order time constant. The hover flights of
2026-11-03 already gave the mass, $k_f$ near hover, hover RPM, spin
directions and a 30–35 ms lag; they had no maneuvers, so nothing else was
identifiable.

**How.** One card per parameter. Fly the card, then run `px4/sysid.py` on
its log: it fits what the log excites and says "not identifiable" for the
rest. A card is done when its line in the output passes the GO column.

| Card | Identifies | Where | Mode | GO when `sysid.py` prints |
|---|---|---|---|---|
| 0. Prep | — | bench | — | every topic it needs is in a short test log |
| 1. Roll / pitch doublets | $I_{xx}$, $I_{yy}$ | indoors OK, 2 m clear around | Stabilized | `Ixx`, `Iyy` with R² ≥ 0.8, two flights within 10 % |
| 2. Yaw steps | $k_m/I_{zz}$ ($k_m$ with CAD $I_{zz}$) | indoors OK | Position | `yaw` with R² ≥ 0.8 |
| 3. Climbs | $k_f$ up to ~8,000 RPM | outdoors or ≥ 6 m ceiling | Position | `k_f` within 3 % of 4.383e-6 over the wider range, or a clear trend (§ Reading the results) |
| 4. Steady passes | $d_x$, $d_y$ | outdoors, wind < 2 m/s, 30 m straight | Position | `drag x` / `drag y` with R² ≥ 0.7 |

Cards 1 and 2 can share a battery; 3 and 4 need the outdoor site.

---

## Safety (every card)

- **RC link.** It dropped twice on 2026-11-03 (once for 0.5 s, once into
  auto-Land onto a table). Find out why before flying maneuvers: antenna
  placement, receiver binding, distance to the transmitter. Confirm the RC
  loss failsafe action and the kill switch on the bench.
- **Landing detection.** In flight 1, PX4 did not detect the landing and the
  kill switch ended it. Expect that again; the pilot keeps the kill switch
  ready until the motors stop.
- **Batteries** at ≥ 3.9 V per cell when the card starts: thrust follows the
  pack voltage only through the ESC's RPM control, but a sagging pack limits
  how much RPM the ESC can reach, which matters for card 3.
- A spotter for every card; nobody within 3 m during doublets and climbs.

---

## Card 0: prep (bench, props off)

1. **Save the full parameter set** (QGroundControl, or `param save` in the
   PX4 shell) and commit it next to the logs. We need `THR_MDL_FAC` (expected
   0.9), `MPC_THR_HOVER`, `MPC_THR_MAX`, `MC_BAT_SCALE_EN`, the rate-loop gains,
   the VOXL ESC RPM range and `VOXL_ESC_FUNC*` (motor order).
2. **Logging.** In `SDLOG_PROFILE` enable the system-identification and
   high-rate profiles (check the bit numbers in the parameter's description).
   `sysid.py` needs `sensor_combined`, `vehicle_angular_velocity`,
   `vehicle_attitude`, `vehicle_local_position`, `esc_status`,
   `actuator_motors` and `vehicle_land_detected`, as fast as they will log.
3. **Check the log** with a 30 s armed-on-the-bench recording:
   ```bash
   pip install pyulog
   ulog_info -v BENCH.ulg | grep -E "esc_status|actuator_motors|vehicle_angular_velocity|sensor_combined"
   ```
   Each must be there. `esc_status` should log at ≥ 100 Hz.
4. **Copy off the drone** the extrinsics file (camera and IMU poses) and the
   hires camera's stored intrinsics. Phase 3 vision needs them.

## Card 1: roll / pitch doublets (Stabilized)

Stabilized (attitude) mode, so the position controller does not damp the
excitation. Hover at 1.5–2 m first, in Position mode, then switch.

1. **Roll:** a sharp stick doublet: right about 40 % for ~0.2 s, left 40 %
   for ~0.2 s, back to center. Let it settle 2 s. Five times.
2. **Pitch:** the same on the pitch stick. Five times.
3. Back to Position mode and land.

Small and sharp beats large: the drone should tilt maybe 10–20°, not travel.
`fit_rotational` needs the roll (pitch) rate's standard deviation above
0.3 rad/s while airborne; five crisp doublets give that easily. Fly the card
twice (two logs) so the two fits can be compared.

## Card 2: yaw steps (Position)

1. Hover at 1.5 m.
2. **Yaw steps:** full yaw stick left for 1 s, center for 2 s, full right
   for 1 s, center for 2 s. Five times.
3. **Constant spin:** half yaw stick for 5 s (about one turn), then stop.
4. Land.

The output gives $k_m/I_{zz}$. With $I_{zz}$ from the CAD model, pass
`--izz` to get $k_m$. The bias it fits is the yaw trim torque of
`STARLING2MAX_DYNAMICS.md` §4.5.

## Card 3: climbs (Position, outdoors)

1. Hover at 1.5 m.
2. **Full up stick for 1.5 s**, then center; let it stop and hold. Descend
   gently to 1.5 m. Three times.
3. If the RPM printed by `sysid.py` (`S_range` of the `k_f` fit) stayed
   below ~8,000 RPM per rotor, raise `MPC_ACC_UP_MAX` one step and repeat.
   Put it back afterwards.

The climb acceleration is what pushes RPM above hover. PX4's own limits
(`MPC_Z_VEL_MAX_UP`, `MPC_ACC_UP_MAX`) cap it, so this card cannot reach the
`MPC_THR_MAX` cap by itself. That is fine: the aim is to see whether $k_f$
holds up to two-thirds of the way.

## Card 4: steady passes (Position, outdoors)

1. Pick a straight 30 m line, crosswind if there is any wind.
2. **Pass at 2 m/s:** fixed stick, at least 4 s at constant speed. Turn and
   fly back along the same line (out and back cancels a steady wind).
3. Repeat at 4 m/s and 6 m/s.
4. **Sideways:** one out-and-back at 3 m/s with the drone yawed 90° to the
   line (for $d_y$).

`fit_drag` uses samples above 1 m/s along each body axis and needs at least
1 m/s of spread, so different speeds matter more than many passes.

---

## Running the analysis

```bash
# per card
python3 px4/sysid.py CARD1_a.ulg CARD1_b.ulg --vehicle starling2max
python3 px4/sysid.py CARD2.ulg --vehicle starling2max --izz 4.23e-3     # CAD Izz when known
# everything together, with the 2026-11-03 hovers
python3 px4/sysid.py logs/*.ulg --vehicle starling2max --izz <CAD Izz>
```

Each line ends with the profile's current value for comparison.

## Reading the results

- **`k_f` over a wider range.** If the climbs pull $k_f$ up or down by more
  than 3 % from the hover value, $F=k_f\omega^2$ does not hold over the range;
  plot thrust against $\sum\omega^2$ (the fit's bins) before trusting one
  number.
- **Inertia R² below 0.8:** the doublets were too gentle, or the rotor order
  in `esc_status` is not the PX4 motor order (`--esc-order`, from
  `VOXL_ESC_FUNC*`). A negative slope is reported as such and almost always
  means the order or a sign is wrong.
- **Motor lag.** `sysid.py` reports a first-order $\tau$ and, separately, a
  pure delay plus $\tau$. The 2026-11-03 figure's 30–35 ms was a correlation
  peak, which measures delay; QuadSim models a first-order lag. Use the
  first-order value in the profile.

## After the flights

1. Put the new values in the `starling2max` entry of `control/vehicles.py`,
   with "(meas.)" and the log names.
2. Update `workspace/STARLING2MAX_DYNAMICS.md` (§2, the affected sections, and §12).
3. Commit the logs (or CSV extracts) and the parameter file under
   `px4/flight_logs/2026-11-xx/` so the numbers can be re-derived.
4. Run both test suites: `DRONE_VEHICLE=iris` and `DRONE_VEHICLE=starling2max`.
