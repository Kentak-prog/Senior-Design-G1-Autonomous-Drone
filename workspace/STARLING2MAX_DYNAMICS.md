# ModalAI Starling 2 Max — Dynamics Model

This document gives the physical parameters and equations of motion we use for
the ModalAI Starling 2 Max: the `starling2max` profile in `control/vehicles.py`,
which the MPC (`control/`), the planner (`Trajectory/`), the RL stack (`rl/`)
and the attitude bridge (`px4/`) read when `DRONE_VEHICLE=starling2max`. It
follows the structure of `STARLING2_DYNAMICS.md` (the smaller Starling 2) and
reuses its derivations where the method is the same.

Unlike the Starling 2, much of this is measured on our own drone.

**Confidence labels used below**

- **(meas.)**: identified from our flight logs or weighed (§1, F1). Valid in the range that was flown.
- **(datasheet)**: from ModalAI's official datasheet (D1).
- **(derived)**: computed exactly from the numbers above, with the equation shown.
- **(inferred)**: a configuration value deduced from logged numbers. Check it on the drone.
- **(est.)**: an engineering estimate built on stated assumptions. Each one has a way to replace it in §12.

---

## 1. Sources

| # | Source | What it gives |
|---|---|---|
| D1 | ModalAI, *Starling 2 Max Datasheet* — <https://docs.modalai.com/starling-2-max-datasheet/> (read 2026-11-03) | Take-off weight 566 g; motor-to-motor 260 mm wide × 190 mm long; prop-to-prop 512 mm diagonal; 2203.5 1500 KV motors; 180 mm tri-blade props; two 2S 18650 packs in series (4S, 16.8 V max); cruise 15 m/s, dash 20 m/s; payload 500 g, "total takeoff weight ~930g"; VOXL 2; cameras C28 / C29 |
| D2 | ModalAI, *Starling 2 Max Hardware Overview* — <https://docs.modalai.com/starling-2-max-hardware-overview/> | 322 mm diagonal frame; VOXL ESC Mini 4-in-1 with integrated power module |
| F1 | Our hover flights, 2026-11-03 (team analysis figure *rotor identification*; the logs are not in the repo yet, §12) | Mass 573 g (weighed, C29, batteries and props); two indoor Position-mode hovers on PX4/VOXL 2, 231 s of steady hover; $k_f$; hover RPM per rotor; motor lag; ESC range 2,000–12,000 RPM; PX4 `MPC_THR_MAX` = 0.6 |
| S | `workspace/STARLING2_DYNAMICS.md` | Pegasus rotor and drag models (its S6–S9), the yaw-moment scaling method (its §4.6), the 12-state model (its §8–9) |
| T | `px4/sysid.py` | The fits that re-derive the (meas.) values from `.ulg` logs, and that will fill in the (est.) ones (§12) |
| P | PX4 parameter reference, `THR_MDL_FAC`, and the multicopter tuning guide's thrust-curve section (docs.px4.io) | Normalized thrust → motor output model, used in §4.2 |

---

## 2. Parameter summary

| Symbol | Meaning | Value | Label |
|---|---|---|---|
| $m$ | mass, flight-ready (C29, batteries, props) | 0.573 kg | meas. (F1) |
| $g$ | gravity | 9.81 m/s² | — |
| $l_x$ | half fore-aft motor spacing | 0.095 m | datasheet (D1) |
| $l_y$ | half left-right motor spacing | 0.130 m | datasheet (D1) |
| $k_f$ | thrust coefficient | $4.383\times10^{-6}$ N/(rad/s)² | meas. (F1), near hover only |
| $\omega_{min},\omega_{max}$ | ESC speed range | 209.44, 1256.64 rad/s (2,000, 12,000 RPM) | meas. (F1) |
| $\omega_{hover}$ | hover rotor speed | 566.2 rad/s (5,407 RPM) | meas. (F1) |
| $\tau_m$ | motor time constant | 0.035 s | meas. (F1) |
| $\kappa=k_m/k_f$ | yaw-moment-to-thrust ratio | 0.0829 m | est. |
| $k_m$ | yaw-moment coefficient | $3.633\times10^{-7}$ N·m/(rad/s)² | est. |
| $I_{xx},I_{yy},I_{zz}$ | principal inertia | $2.47\times10^{-3}$, $2.00\times10^{-3}$, $4.23\times10^{-3}$ kg·m² | est. |
| $d_x,d_y,d_z$ | linear drag coefficients | 0.227, 0.137, 0 N/(m/s) | est. |
| `THR_MDL_FAC` | PX4 thrust model factor | 0.9 | inferred (§4.2) |
| $T_{cap}$ | total thrust at PX4's `MPC_THR_MAX` = 0.6 | 17.82 N | derived |

The measured mass is 7 g above the datasheet's 566 g. The C29 configuration
carries the ToF sensor that C28 does not, which plausibly accounts for it.

---

## 3. Frames and conventions

The same as the rest of the repo and `STARLING2_DYNAMICS.md` §3: world ENU,
body FLU, quaternions $(q_w,q_x,q_y,q_z)$, ZYX Euler angles in the MPC. PX4
logs NED / FRD; `px4/sysid.py` converts on load (FRD → FLU is
$(x,y,z)\mapsto(x,-y,-z)$).

**Rotor numbering and spin** (PX4 quad-X order = Pegasus order):

| Rotor $i$ | Position | $(x_i, y_i)$ in $\{B\}$ (m) | Prop spin, seen from above | $d_i$ |
|---|---|---|---|---|
| 0 | front-right | $(+l_x, -l_y)$ = (0.095, −0.130) | CCW | −1 |
| 1 | back-left | $(-l_x, +l_y)$ = (−0.095, 0.130) | CCW | −1 |
| 2 | front-left | $(+l_x, +l_y)$ = (0.095, 0.130) | CW | +1 |
| 3 | back-right | $(-l_x, -l_y)$ = (−0.095, −0.130) | CW | +1 |

The spins are measured (F1). $d_i=+1$ means the rotor pushes the airframe
counter-clockwise when it speeds up: a CW prop's reaction torque on the
airframe is CCW.

**Why $l_x=0.095$, $l_y=0.130$.** D1 gives motor-to-motor 190 mm long and
260 mm wide, so each motor is half of each away from the center:

$$l_x=\tfrac{0.190}{2}=0.095\ \text{m},\qquad l_y=\tfrac{0.260}{2}=0.130\ \text{m},\qquad
l=\sqrt{0.095^2+0.130^2}=0.161\ \text{m}.$$

The motor-to-motor diagonal $2l=0.322$ m matches D2's "322 mm" frame. The
long spacing is lateral, as on the Starling 2 and the Iris USD.

**Center of mass.** In hover, the front and back pairs, and the left and
right pairs, run at the same mean RPM to within 0.1 % (F1). The rotors are
symmetric about the body origin, so the center of mass is centered and no
offset is modeled.

---

## 4. Rotor model

### 4.1 PX4 motor output to rotor speed (meas.)

The VOXL ESC runs closed-loop RPM control. PX4's motor output $u_i\in[0,1]$
maps linearly onto the ESC range, 2,000–12,000 RPM (F1):

$$\text{RPM}_i=2000+10000\,u_i\qquad\Longleftrightarrow\qquad
\omega_{i,ref}=209.44+1047.20\,u_i\ \ \text{rad/s}.$$

In the profile: `px4_w_offset = 209.44`, `px4_w_scale = 1047.20`,
`w_min = 209.44`, `w_max = 1256.64`. The same mapping in Pegasus (Phase 3)
is `input_scaling = 1047.20`, `input_offset = 0`,
`zero_position_armed = 209.44` (form of `STARLING2_DYNAMICS.md` §4.1),
which makes PX4 SITL drive the sim rotors as it drives the real ESC.

### 4.2 Normalized thrust to motor output (inferred)

PX4 does not send the thrust setpoint $T_{sp}\in[0,1]$ straight to the
motors. It first inverts its thrust model (P):

$$T_{sp}=f\,u^2+(1-f)\,u,\qquad f=\texttt{THR\_MDL\_FAC}.$$

F1 gives `MPC_THR_MAX` = 0.6 as 9,628 RPM, i.e. $u=(9628-2000)/10000=0.7628$.
$f=0.9$ reproduces it:

$$0.9\times0.7628^2+0.1\times0.7628=0.5237+0.0763=0.600.$$

(With $f=0$, $T_{sp}=0.6$ would be $u=0.6$, 8,000 RPM; with $f=1$, $u=0.775$,
9,746 RPM.) **Check `THR_MDL_FAC` in the drone's parameter file.**

What follows from it, worked through at hover:

$$u_{hover}=\frac{5407-2000}{10000}=0.3407,\qquad
T_{sp,hover}=0.9\times0.3407^2+0.1\times0.3407=0.1045+0.0341=0.139.$$

So PX4's hover-thrust estimate should read about 0.14 (check
`hover_thrust_estimate` in the logs).

**Consequence for the attitude bridge.** Normalized thrust is not linear in
newtons on this drone. A linear hover scaling (the bridge's `--hover-thrust`
mode) predicts $5.62\times0.6/0.139=24.3$ N at $T_{sp}=0.6$; the true value is
17.8 N (§6). The exact chain, thrust in N → $T_{sp}$, is

$$\omega=\sqrt{\frac{T}{4k_f}},\qquad u=\frac{\omega-209.44}{1047.20},\qquad T_{sp}=f u^2+(1-f)u.$$

The profile carries $f$ as `px4_thr_mdl_fac` (0.9; 0 for the Iris), and
`px4/attitude_bridge.py`, `rl/isaac/frames.ThrustMap` and `rl/isaac/calibrate.py`
apply the whole chain. PX4 SITL is set to the same $f$ by
`px4/starling2max_sitl_params.sh`, so the sim maps thrust as the drone does.
If the drone's parameter file gives another `THR_MDL_FAC`, change the profile
and the script together (`control/test_vehicles.py` checks they agree). PX4 battery-voltage compensation
(`MC_BAT_SCALE_EN`) would add a further factor; check whether it is on.

### 4.3 Thrust per rotor (meas.)

$$F_i=k_f\,\omega_i^2,\qquad k_f=4.383\times10^{-6}\ \text{N/(rad/s)}^2.$$

F1 fits the accelerometer thrust $-m\,a_z$ against $\sum\omega_i^2$ on 0.5 s
airborne averages (the same fit as `px4/sysid.py fit_kf`). Hover checks it:

$$\omega_{hover}=5407\times\frac{2\pi}{60}=566.22\ \text{rad/s},\qquad
4k_f\omega_{hover}^2=4\times4.383\times10^{-6}\times566.22^2=5.621\ \text{N}=m g=0.573\times9.81.$$

**Validity.** The flights covered about 5,300–5,900 RPM. Above that,
$F=k_f\omega^2$ is extrapolated (§12 plans the climbs that test it).

### 4.4 Motor lag (meas.)

Measured RPM trails commanded RPM by 30–35 ms (the peak of their
cross-correlation, per rotor, F1). The profile models it as a first-order lag:

$$\dot\omega_i=\frac{\omega_{i,ref}-\omega_i}{\tau_m},\qquad\tau_m=0.035\ \text{s}.$$

QuadSim applies it (`tau_motor`). A correlation peak measures delay more than
a first-order time constant; `px4/sysid.py fit_motor_lag` fits $\tau_m$
directly (and a pure delay alongside), so re-run it on the logs (§12).

### 4.5 Yaw moment (est.)

$$M_{z,i}=d_i\,k_m\,\omega_i^2=d_i\,\kappa\,F_i.$$

No yaw maneuvers were flown, so $k_m$ is not identified. The estimate scales
the Iris value by prop diameter (`STARLING2_DYNAMICS.md` §4.6):

$$\kappa=0.11698\times\frac{0.180}{0.254}=0.0829\ \text{m},\qquad
k_m=\kappa\,k_f=0.0829\times4.383\times10^{-6}=3.633\times10^{-7}.$$

Tri-blade props probably have a higher $C_Q/C_T$ than this assumes, so $k_m$
is more likely low than high.

**Yaw trim seen in hover (F1).** The CW rotors (2, 3) hover about 38 RPM
above the mean and the CCW rotors (0, 1) about 38 below. At steady yaw the
rotors cancel some bias torque (motor or prop misalignment, CW / CCW prop
differences). With the estimated $k_m$ its size is

$$k_m\cdot2\big[(\omega_h+\delta)^2-(\omega_h-\delta)^2\big]=k_m\cdot8\,\omega_h\delta
=3.633\times10^{-7}\times8\times566.2\times3.98=0.0065\ \text{N·m},$$

with $\delta=38$ RPM $=3.98$ rad/s. That is about 1 % of the yaw authority at
the PX4 cap (§6). `fit_rotational` fits it as the bias term.

---

## 5. Control allocation (mixer)

The same form as `STARLING2_DYNAMICS.md` §5 with the Max's arms:

$$\begin{bmatrix}U_1\\U_2\\U_3\\U_4\end{bmatrix}=
\begin{bmatrix}1&1&1&1\\-l_y&l_y&l_y&-l_y\\-l_x&l_x&-l_x&l_x\\-\kappa&-\kappa&\kappa&\kappa\end{bmatrix}
\begin{bmatrix}F_0\\F_1\\F_2\\F_3\end{bmatrix}
=\begin{bmatrix}1&1&1&1\\-0.130&0.130&0.130&-0.130\\-0.095&0.095&-0.095&0.095\\-0.0829&-0.0829&0.0829&0.0829\end{bmatrix}\mathbf F$$

$U_1$ is total thrust, $U_2..U_4$ the body torques (FLU). Inverse:
$F_i=\tfrac14\big(U_1\pm U_2/l_y\pm U_3/l_x\pm U_4/\kappa\big)$ with the signs of
row $i$ of the matrix above.

---

## 6. Actuator limits and performance

Two ceilings matter. The **ESC limit** (12,000 RPM) is physical but
extrapolated (§4.3). The **PX4 cap** (`MPC_THR_MAX` = 0.6, 9,628 RPM) is what
PX4 actually allows. The profile's control limits use the cap.

| | per rotor | total | T/W |
|---|---|---|---|
| hover, 5,407 RPM (meas.) | 1.405 N | 5.62 N | 1.00 |
| top of the measured range, 5,900 RPM | 1.673 N | 6.69 N | 1.19 |
| PX4 cap, 9,628 RPM | $k_f\,(1008.24)^2=4.456$ N | 17.82 N | 3.17 |
| ESC limit, 12,000 RPM (extrapolated) | $k_f\,(1256.64)^2=6.921$ N | 27.69 N | 4.93 |

The datasheet's payload rating (500 g, ~930 g total) with ModalAI's
"T/W ≥ 2" guidance asks for at least $2\times0.930\times9.81=18.2$ N, which
sits between the two, as it should.

**At the PX4 cap** ($F_{max}=4.456$ N per rotor):

$$\tau_{x,max}=2l_yF_{max}=1.158\ \text{N·m},\quad
\tau_{y,max}=2l_xF_{max}=0.847\ \text{N·m},\quad
\tau_{z,max}=2\kappa F_{max}=0.739\ \text{N·m}.$$

The profile uses 1.15 / 0.84 / 0.73 (`body_torque_max`, a little below).

$$\dot p_{max}=\frac{1.158}{2.47\times10^{-3}}=469,\quad
\dot q_{max}=\frac{0.847}{2.00\times10^{-3}}=423,\quad
\dot r_{max}=\frac{0.739}{4.23\times10^{-3}}=175\ \ \text{rad/s}^2\ \text{(est. inertia)}.$$

$$a_{z,max}=\frac{17.82-5.62}{0.573}=21.3\ \text{m/s}^2,\qquad
\theta_{hold}=\arccos\frac{5.62}{17.82}=1.250\ \text{rad}\ (71.6^\circ),\qquad
a_{h,max}=\sqrt{(17.82/0.573)^2-9.81^2}=29.5\ \text{m/s}^2.$$

The profile limits tilt to 1.0 rad, horizontal acceleration to 15 m/s²
($g\tan1.0=15.3$), climb to 21 m/s² and speed to 15 m/s (D1 cruise).

---

## 7. Mass and inertia

### 7.1 Mass (meas.)

0.573 kg, weighed flight-ready (F1).

### 7.2 Inertia (est.)

Not yet identified. The estimate scales the Starling 2 estimate
(`STARLING2_DYNAMICS.md` §6.2) by mass and, per axis, by the square of the arm
that dominates it: $l_y$ for roll, $l_x$ for pitch, $l$ for yaw.

$$\frac{m}{m_{S2}}=\frac{0.573}{0.285}=2.011,\quad
\Big(\frac{0.130}{0.088}\Big)^2=2.182,\quad
\Big(\frac{0.095}{0.066}\Big)^2=2.072,\quad
\Big(\frac{0.161}{0.110}\Big)^2=2.143$$

$$I_{xx}=5.62\times10^{-4}\times2.011\times2.182=2.47\times10^{-3},\quad
I_{yy}=4.79\times10^{-4}\times2.011\times2.072=2.00\times10^{-3},\quad
I_{zz}=9.81\times10^{-4}\times2.011\times2.143=4.23\times10^{-3}\ \text{kg·m}^2.$$

This assumes the Max distributes its mass like the Starling 2. It does not
quite: the two 18650 packs are a large, concentrated share of the mass, which
probably makes the true values lower. Two independent replacements are
planned (§12): ModalAI's STEP model with CAD mass properties, and roll /
pitch doublets through `fit_rotational`.

---

## 8. Aerodynamic drag (est.)

Pegasus' linear model $\mathbf F_D=-\mathrm{diag}(d)\,\mathbf v_B$ (S). The
Starling 2 coefficients (0.106, 0.064, 0) scaled by planform area,
$(l/l_{S2})^2=2.143$:

$$d_x=0.106\times2.143=0.227,\qquad d_y=0.064\times2.143=0.137\ \text{N/(m/s)}.$$

Linear drag cannot hold at speed (real drag grows roughly with $v^2$), which
is one reason the profile caps velocity. `fit_drag` identifies $d$ from body
accelerometer against body velocity on steady passes (§12); passes at
several speeds also show whether a linear model fits.

---

## 9. Rigid-body equations

Unchanged from `STARLING2_DYNAMICS.md` §8–9 (Newton–Euler, the 12-state ZYX
model in `control/New_Drone_State_Equations.py`), with the numbers of §2.

---

## 10. Hover equilibrium and linearization

At hover $U_1=mg=5.621$ N, $\omega_i=566.2$ rad/s, $u_i=0.341$,
$T_{sp}\approx0.139$ (§4.2). The linearized input gains (same $\mathbf A$,
$\mathbf B$ structure as `STARLING2_DYNAMICS.md` §10.2):

$$\frac1m=1.745\ \text{kg}^{-1},\quad
\frac1{I_{xx}}=405,\quad\frac1{I_{yy}}=500,\quad\frac1{I_{zz}}=236\ \ (\text{kg·m}^2)^{-1}.$$

Against the Iris ($1/m=0.667$, $1/I_{xx}=34.3$), thrust authority per newton is
2.6× and roll authority per N·m is 11.8× larger. MPC input weights tuned on the
Iris do not carry over. This is the likely cause of the two MPC tests that
fail under `starling2max` (Phase 4).

---

## 11. Where each parameter goes in the repo

Everything is in the `starling2max` entry of `control/vehicles.py`;
`control/drone_constraints.py` and `rl/drone_params.QuadParams` read it, so
there is one place to change.

| Parameter | `control/vehicles.py` field | Pegasus spawn (`workspace/spawn_starling2max.py`) |
|---|---|---|
| $m$, $\mathbf I$ | `mass`, `inertia_diag` | body MassAPI |
| $l_x,l_y$, $d_i$ | `rotor_xy`, `rot_dir` | rotor prims, `rot_dir` |
| $k_f$, $k_m$ | `k_f`, `k_m` | `QuadraticThrustCurve` |
| ESC mapping | `px4_w_offset`, `px4_w_scale`, `w_min`, `w_max` | `PX4MavlinkBackend` `input_scaling`, `zero_position_armed`; thrust-curve `W_MIN` = 0 so the motors stop when disarmed |
| `THR_MDL_FAC` | `px4_thr_mdl_fac` | PX4 SITL parameter (`px4/starling2max_sitl_params.sh`) |
| $\tau_m$ | `tau_motor` | `LaggedQuadraticThrustCurve` in the spawn script (Pegasus rotors are otherwise instantaneous) |
| $d$ | `drag_coeff` | `LinearDrag` |
| PX4 cap | `total_thrust_max`, `body_torque_max` | PX4 params (`MPC_THR_MAX`) |

`control/test_vehicles.py test_starling2max_matches_hover_flights` checks the
F1 numbers against the profile.

---

## 12. Open items

| Item | How | Status |
|---|---|---|
| F1 logs and analysis script into the repo | copy the two `.ulg` (or CSV extracts) and the script; re-run `python3 px4/sysid.py LOG1.ulg LOG2.ulg --vehicle starling2max` and compare with §2 | to do |
| `THR_MDL_FAC`, `MPC_THR_HOVER`, `MPC_THR_MAX`, `MC_BAT_SCALE_EN`, rate gains, ESC RPM range | save the drone's full parameter set (QGroundControl, or `param save` in the PX4 shell) and commit it | to do |
| Sim camera: mount (0.11, 0, 0) m, 15° up, 60° hFOV are placeholders in `spawn_starling2max.py` | replace with the extrinsics and the IMX412 calibration below | to do |
| SITL rate gains | SITL keeps the Iris airframe's `MC_*RATE_*`; take the drone's from its parameter file | to do |
| Camera and IMU extrinsics | the extrinsics file on the drone (ModalAI: *Configure Extrinsics*, normally `/etc/modalai/extrinsics.conf`) | to do |
| Hires camera calibration (IMX412) | the drone's stored intrinsics (normally under `/data/modalai/`), or calibrate with `voxl-camera-calibration` | to do |
| $k_m$ (yaw) | yaw-rate steps (`px4/STARLING2MAX_SYSID_FLIGHTS.md` card 2) → `fit_rotational` gives $k_m/I_{zz}$; $I_{zz}$ from CAD | to fly |
| $I_{xx}$, $I_{yy}$ | roll / pitch doublets (card 1) → `fit_rotational` | to fly |
| $I_{zz}$ | CAD mass properties from ModalAI's STEP model | to do |
| $k_f$ above 5,900 RPM | short climbs (card 3) → `fit_kf` over the wider range; a curve that bends means $F=k_f\omega^2$ needs a correction | to fly |
| Drag | steady passes at 2–3 speeds (card 4) → `fit_drag` | to fly |
| Motor lag as a first-order $\tau_m$ | `fit_motor_lag` on the F1 logs | to do |
