# ModalAI Starling 2 — Dynamics Model

This document gives the physical parameters and equations of motion we use to
simulate the ModalAI Starling 2 in Isaac Sim / Pegasus
(`workspace/spawn_starling.py`), and to replace the Iris numbers in the
MPC (`control/`) and RL (`rl/`) stacks. Every number has a source or a
derivation, and every derived number is worked out step by step.

**Confidence labels used below**

- **(measured)**: stated by ModalAI staff. It still isn't from an official datasheet; see §1.
- **(derived)**: computed exactly from measured numbers with the equation shown.
- **(est.)**: an engineering estimate built on stated assumptions. Replace it with a
  measurement (bifilar pendulum for inertia, thrust stand for $k_f$/$k_m$,
  flight log for drag) or with numbers from the CAD model.

---

## 1. Sources

| # | Source | What it gives |
|---|---|---|
| S1 | ModalAI forum, *Starling 2 payload* — <https://forum.modalai.com/topic/5202/starling-2-payload.md> | Weight with battery ≈ 285 g; peak thrust ≈ 185 g per motor at 7.5 V; a thrust-to-weight of at least 2.0 is recommended |
| S2 | ModalAI forum post — <https://forum.modalai.com/post/19120> | Motor spacing center-to-center 132 mm (length) × 176 mm (width); size with props extended 267 × 311 mm; folded 154.2 × 198.2 × 86.3 mm; props about 60 mm long from the center of the screw |
| S3 | ModalAI forum post — <https://forum.modalai.com/post/20168> and user posts <https://forum.modalai.com/user/jacob/posts>, <https://forum.modalai.com/uid/2311>, <https://forum.modalai.com/post/15367>, <https://forum.modalai.com/post/15442> | Motors are 1504-size, 3000 KV (silver and black variants differ slightly) |
| S4 | PX4 docs, *VOXL 2 Starling* — <https://docs.px4.io/v1.16/en/complete_vehicles_mc/modalai_starling> | Original (gen-1) Starling, 275 g, 120 mm props. Used only as a cross-check on prop size; it's a different airframe from the Starling 2 |
| S5 | ModalAI forum, *Simulating Starling 2 Max* — <https://forum.modalai.com/topic/4993/simulating-starling-2-max-drone.md>; *Robotic simulation for Starling-2* — <https://forum.modalai.com/topic/3383/robotic-simulation-for-starling-2> | ModalAI offers a STEP CAD model, not a simulation model |
| S6 | Pegasus Simulator, `multirotor.py` — <https://github.com/PegasusSimulator/PegasusSimulator/blob/main/extensions/pegasus.simulator/pegasus/simulator/logic/vehicles/multirotor.py> | How forces are applied: thrust on `/rotorN` along rotor +z, yaw moment as a torque on `/body`, drag force on `/body`; Iris defaults |
| S7 | Pegasus, `quadratic_thrust_curve.py` — <https://github.com/PegasusSimulator/PegasusSimulator/blob/main/extensions/pegasus.simulator/pegasus/simulator/logic/thrusters/quadratic_thrust_curve.py> | $F_i=k_f\omega_i^2$, $M_z=\sum k_m\omega_i^2 d_i$, clipping, Iris defaults ($k_f=8.54858\times10^{-6}$, $k_m=10^{-6}$, $\omega_{max}=1100$) |
| S8 | Pegasus, `linear_drag.py` — <https://github.com/PegasusSimulator/PegasusSimulator/blob/main/extensions/pegasus.simulator/pegasus/simulator/logic/dynamics/linear_drag.py> | $\mathbf F_D=-\mathrm{diag}(d)\,\mathbf v_B$; Iris default $d=(0.50,0.30,0)$ |
| S9 | Pegasus, `px4_mavlink_backend.py` — <https://github.com/PegasusSimulator/PegasusSimulator/blob/main/extensions/pegasus.simulator/pegasus/simulator/logic/backends/px4_mavlink_backend.py> | PX4 output → rotor speed: $\omega=1000(u+0)+100$ rad/s; 250 Hz update |
| S10 | This repo: `rl/drone_params.py`, `control/New_Drone_State_Equations.py`, `control/drone_constraints.py`, `workspace/spawn_example.py` | Frame conventions, Iris rotor order and positions, the 12-state model used by the MPC |
| S11 | Standard propeller theory (e.g. McCormick, *Aerodynamics, Aeronautics and Flight Mechanics*; Beard & McLain, *Small Unmanned Aircraft*, ch. 4) | $T=C_T\rho n^2D^4$, $Q=C_Q\rho n^2D^5$ used to scale $k_m$ |
| S12 | Standard rigid-body mechanics (any dynamics text, e.g. Goldstein / Beard & McLain ch. 3) | Newton–Euler equations, box inertia, parallel-axis theorem, quaternion kinematics |

Caveat: S1–S3 are informal staff statements on the forum. Check them against
ModalAI's official Starling 2 datasheet before using them for hardware decisions.

---

## 2. Parameter summary

| Symbol | Meaning | Value | Label |
|---|---|---|---|
| $m$ | mass with battery | 0.285 kg | measured (S1) |
| $g$ | gravity | 9.81 m/s² | — |
| $l_x$ | half fore-aft motor spacing | 0.066 m | derived (S2) |
| $l_y$ | half left-right motor spacing | 0.088 m | derived (S2) |
| $F_{max}$ | peak thrust per rotor | 1.815 N | derived (S1) |
| $\omega_{max}$ | Pegasus max rotor speed | 1100 rad/s | Pegasus convention (S7, S9) |
| $k_f$ | thrust coefficient | $1.50\times10^{-6}$ N/(rad/s)² | derived |
| $\kappa=k_m/k_f$ | yaw-moment-to-thrust ratio | 0.0553 m | est. |
| $k_m$ | yaw-moment coefficient | $8.29\times10^{-8}$ N·m/(rad/s)² | est. |
| $I_{xx},I_{yy},I_{zz}$ | principal inertia | $5.62\times10^{-4}$, $4.79\times10^{-4}$, $9.81\times10^{-4}$ kg·m² | est. |
| $d_x,d_y,d_z$ | linear drag coefficients | 0.106, 0.064, 0 N/(m/s) | est. |
| $\omega_{hover}$ | hover rotor speed | 682.6 rad/s | derived |
| $u_{hover}$ | PX4 hover output | 0.583 | derived |
| $T_{max}/W$ | thrust-to-weight | 2.60 | derived |

---

## 3. Frames and conventions

These match the rest of the repo (S10):

- **World frame $\{W\}$: ENU.** $x$ east, $y$ north, $z$ up. Gravity is $-g\,\mathbf e_z$.
- **Body frame $\{B\}$: FLU.** $x$ forward, $y$ left, $z$ up. The origin is at the body prim.
- **Attitude.** In Pegasus/Isaac it's a unit quaternion $\mathbf q=(q_w,q_x,q_y,q_z)$. In the MPC it's ZYX Euler
  angles: roll $\phi$, pitch $\theta$, yaw $\psi$.
- **Rotor numbering and spin** (Pegasus / PX4 Iris order, kept for the Starling):

| Rotor $i$ | Position | $(x_i, y_i)$ in $\{B\}$ | spin $d_i$ |
|---|---|---|---|
| 0 | front-right | $(+l_x, -l_y)$ = (0.066, −0.088) | −1 |
| 1 | back-left | $(-l_x, +l_y)$ = (−0.066, 0.088) | −1 |
| 2 | front-left | $(+l_x, +l_y)$ = (0.066, 0.088) | +1 |
| 3 | back-right | $(-l_x, -l_y)$ = (−0.066, −0.088) | +1 |

$d_i=+1$ means that rotor pushes the airframe counter-clockwise, seen from above,
when it speeds up (S7, S10).

**Why $l_x=0.066$, $l_y=0.088$.** S2 gives a center-to-center motor spacing of
132 mm long and 176 mm wide. The motors sit at the corners of that rectangle, symmetric
about the body origin, so each one is half of each spacing away from it:

$$l_x=\tfrac{0.132}{2}=0.066\ \text{m},\qquad l_y=\tfrac{0.176}{2}=0.088\ \text{m}.$$

Assumption: "length" is fore-aft ($x$). This matches the Iris, which also has the shorter
spacing along $x$ (0.13 vs 0.22 m).

Arm length from the center to each motor:

$$l=\sqrt{l_x^2+l_y^2}=\sqrt{0.066^2+0.088^2}=\sqrt{0.004356+0.007744}=\sqrt{0.0121}=0.110\ \text{m}.$$

So the motor-to-motor diagonal is $2l=0.220$ m.

---

## 4. Rotor model

### 4.1 PX4 output to rotor speed (S9)

PX4 sends a normalized output $u_i\in[0,1]$ for each motor. Pegasus converts it to a
rotor speed command:

$$\omega_{i,ref}=s\,(u_i+o)+\omega_0,\qquad s=1000,\ o=0,\ \omega_0=100\ \text{rad/s}$$

$$\Rightarrow\ \omega_{i,ref}=1000\,u_i+100.$$

- $s$ = `input_scaling`, $o$ = `input_offset`, $\omega_0$ = `zero_position_armed`.
- When armed, $u=0$ still spins the rotor at 100 rad/s (idle). When disarmed, Pegasus sends 0.
- At $u=1$, $\omega=1100$ rad/s, which is the Pegasus $\omega_{max}$.

### 4.2 Speed clipping (S7)

$$\omega_i=\min\!\big(\max(\omega_{i,ref},\ \omega_{min}),\ \omega_{max}\big),\qquad \omega_{min}=0,\ \omega_{max}=1100\ \text{rad/s}.$$

Pegasus rotors respond instantly. There is no motor time constant ($\tau_{motor}=0$).

### 4.3 Thrust per rotor (S7)

$$F_i=k_f\,\omega_i^2.$$

Thrust grows with the square of rotor speed, which is the standard
momentum/blade-element result $T=C_T\rho n^2D^4$ (S11) with constant $C_T$, $\rho$, $D$
lumped into $k_f$. Pegasus applies $F_i$ along the rotor prim's $+z$ axis at the
rotor prim's position (S6).

### 4.4 Deriving $k_f$ for the Starling 2

S1 gives a peak thrust of 185 g-force per motor at 7.5 V. Convert it to newtons:

$$F_{max}=0.185\ \text{kg}\times 9.81\ \text{m/s}^2=1.81485\ \text{N}\approx1.815\ \text{N}.$$

We want Pegasus to produce $F_{max}$ at its maximum command $\omega_{max}=1100$ rad/s
($u=1$, §4.1). Solve §4.3 for $k_f$:

$$k_f=\frac{F_{max}}{\omega_{max}^2}=\frac{1.81485}{1100^2}=\frac{1.81485}{1.21\times10^{6}}=1.49988\times10^{-6}\approx1.50\times10^{-6}\ \tfrac{\text{N}}{(\text{rad/s})^2}.$$

**Why we don't use the real motor speed.** A 3000 KV motor at 7.5 V spins at
$3000\times7.5=22{,}500$ RPM with no load, which is
$22{,}500\times\frac{2\pi}{60}=2356$ rad/s. Pegasus' PX4 mapping tops out at 1100 rad/s.
Fitting $k_f$ to Pegasus' speed scale keeps the PX4 → thrust relationship right
(full throttle gives full thrust) without changing the backend.
$\omega$ in the simulator is therefore a *Pegasus-scaled* rotor speed, not the
physical RPM. The physical coefficient would be
$k_{f,phys}=F_{max}/\omega_{loaded}^2$, which needs the loaded max RPM (not published).

Check: $k_f\,\omega_{max}^2=1.50\times10^{-6}\times1.21\times10^{6}=1.815$ N. ✓

### 4.5 Yaw (drag) moment per rotor (S7)

$$M_{z,i}=k_m\,\omega_i^2\,d_i,\qquad M_z=\sum_{i=0}^{3}k_m\,\omega_i^2\,d_i.$$

Each spinning propeller feels an aerodynamic drag torque. The airframe feels the
reaction, about body $z$. Pegasus sums the four torques and applies the total to
`/body` (S6).

Because $F_i=k_f\omega_i^2$, the moment can also be written in terms of thrust:

$$M_{z,i}=\frac{k_m}{k_f}F_i\,d_i=\kappa\,F_i\,d_i,\qquad \kappa\equiv\frac{k_m}{k_f}\ [\text{m}].$$

### 4.6 Estimating $k_m$ (est.)

From propeller theory (S11), for rotation rate $n$, diameter $D$ and air density $\rho$:

$$T=C_T\,\rho\,n^2D^4,\qquad Q=C_Q\,\rho\,n^2D^5\ \ \Rightarrow\ \ \frac{Q}{T}=\frac{C_Q}{C_T}\,D.$$

$\kappa$ is exactly $Q/T$. If the Starling props have a $C_Q/C_T$ ratio similar to
the Iris props, then $\kappa$ scales with prop diameter:

$$\kappa_{Starling}=\kappa_{Iris}\,\frac{D_{Starling}}{D_{Iris}}.$$

- Iris (S7 defaults): $\kappa_{Iris}=\dfrac{1.0\times10^{-6}}{8.54858\times10^{-6}}=0.11698$ m.
- $D_{Iris}=10$ in $=0.254$ m (the 3DR Iris / PX4 SITL iris uses 10×4.7 props).
- $D_{Starling}\approx0.120$ m: S2 gives props about 60 mm long from the screw center
  (radius ≈ 60 mm), consistent with the 120 mm props of the gen-1 Starling (S4).

$$\kappa_{Starling}=0.11698\times\frac{0.120}{0.254}=0.11698\times0.47244=0.05527\ \text{m}.$$

$$k_m = \kappa, \quad k_f = 0.05527 \times 1.50 \times 10^{-6} = 8.29 \times 10^{-8} \ \frac{\mathrm{N}\cdot\mathrm{m}}{(\mathrm{rad/s})^2}$$

Measure this on a thrust stand that also logs torque if yaw response matters.

---

## 5. Control allocation (mixer)

### 5.1 Body forces and torques from rotor thrusts

Each rotor applies force $\mathbf F_i=(0,0,F_i)$ at $\mathbf r_i=(x_i,y_i,z_r)$ in $\{B\}$.
The torque it produces about the body origin is the cross product:

$$\boldsymbol\tau_i=\mathbf r_i\times\mathbf F_i=
\begin{vmatrix}\mathbf e_x&\mathbf e_y&\mathbf e_z\\ x_i&y_i&z_r\\ 0&0&F_i\end{vmatrix}
=\big(y_iF_i,\ -x_iF_i,\ 0\big).$$

The rotor height $z_r$ drops out because the force is parallel to $z$.

Adding up all four rotors gives the control inputs used by the MPC
($U_1$ thrust, $U_2,U_3,U_4$ torques, S10):

$$U_1=T=\sum_i F_i=F_0+F_1+F_2+F_3$$

$$U_2=\tau_x=\sum_i y_iF_i=l_y(-F_0+F_1+F_2-F_3)$$

$$U_3=\tau_y=-\sum_i x_iF_i=l_x(-F_0+F_1-F_2+F_3)$$

$$U_4=\tau_z=\sum_i\kappa F_i d_i=\kappa(-F_0-F_1+F_2+F_3)$$

- $U_2$ (roll): the left rotors (1, 2, $y>0$) push the left side up. Positive $\tau_x$ rolls right-side down.
- $U_3$ (pitch): the back rotors (1, 3, $x<0$) give positive $\tau_y$, which pitches the nose down
  (rotation about $+y$ = left).
- $U_4$ (yaw): rotors 2 and 3 ($d=+1$) yaw the airframe counter-clockwise.

In matrix form:

$$\begin{bmatrix}U_1\\U_2\\U_3\\U_4\end{bmatrix}=
\underbrace{\begin{bmatrix}1&1&1&1\\-l_y&l_y&l_y&-l_y\\-l_x&l_x&-l_x&l_x\\-\kappa&-\kappa&\kappa&\kappa\end{bmatrix}}_{\mathbf M}
\begin{bmatrix}F_0\\F_1\\F_2\\F_3\end{bmatrix}.$$

With Starling numbers:

$$\mathbf M=\begin{bmatrix}1&1&1&1\\-0.088&0.088&0.088&-0.088\\-0.066&0.066&-0.066&0.066\\-0.0553&-0.0553&0.0553&0.0553\end{bmatrix}.$$

### 5.2 Inverse allocation (thrust/torques → rotor thrusts)

The sign patterns of the four rows, $(1,1,1,1)$, $(-1,1,1,-1)$, $(-1,1,-1,1)$
and $(-1,-1,1,1)$, are mutually orthogonal, and each has squared length 4. So
$\mathbf M=\mathbf D\,\mathbf S$ with $\mathbf D=\mathrm{diag}(1,l_y,l_x,\kappa)$ and $\mathbf S^{-1}=\tfrac14\mathbf S^\top$:

$$\mathbf F=\mathbf M^{-1}\mathbf U=\tfrac14\mathbf S^\top\mathbf D^{-1}\mathbf U$$

$$F_0=\tfrac14\Big(U_1-\frac{U_2}{l_y}-\frac{U_3}{l_x}-\frac{U_4}{\kappa}\Big)$$

$$F_1=\tfrac14\Big(U_1+\frac{U_2}{l_y}+\frac{U_3}{l_x}-\frac{U_4}{\kappa}\Big)$$

$$F_2=\tfrac14\Big(U_1+\frac{U_2}{l_y}-\frac{U_3}{l_x}+\frac{U_4}{\kappa}\Big)$$

$$F_3=\tfrac14\Big(U_1-\frac{U_2}{l_y}+\frac{U_3}{l_x}+\frac{U_4}{\kappa}\Big)$$

Then the rotor speed and PX4 output for each rotor are (inverting §4.3 and §4.1):

$$\omega_i=\sqrt{F_i/k_f},\qquad u_i=\frac{\omega_i-100}{1000}\ \ (\text{clipped to }[0,1]).$$

### 5.3 Actuator limits

With $0\le F_i\le F_{max}=1.815$ N:

- Max collective thrust: $U_{1,max}=4F_{max}=4\times1.815=7.26$ N.
- Thrust at armed idle ($\omega=100$ rad/s): $U_{1,idle}=4k_f(100)^2=4\times1.5\times10^{-6}\times10^4=0.06$ N.
- Max roll torque (rotors 1, 2 at $F_{max}$, rotors 0, 3 at 0):
  $\tau_{x,max}=l_y\cdot2F_{max}=0.088\times3.63=0.319$ N·m.
- Max pitch torque: $\tau_{y,max}=l_x\cdot2F_{max}=0.066\times3.63=0.240$ N·m.
- Max yaw torque: $\tau_{z,max}=\kappa\cdot2F_{max}=0.05527\times3.63=0.201$ N·m.

These are *instantaneous* bounds with zero collective margin. Holding altitude
while producing torque leaves less (S1: the motors can't all run at full power
while the drone still has control).

---

## 6. Mass and inertia

### 6.1 Mass (measured, S1)

$$m=0.285\ \text{kg},\qquad W=mg=0.285\times9.81=2.796\ \text{N}.$$

### 6.2 Inertia estimate (est.)

Model: a central **box** (frame, VOXL 2, battery, cameras) plus four **point masses**
(motor + prop + arm tip) at the motor positions.

Assumed split:
- Motor + prop + arm-tip share ≈ 15 g each → $m_m=4\times0.015=0.060$ kg.
- Body box $m_b=m-m_m=0.285-0.060=0.225$ kg, size $a\times b\times c=0.10\times0.06\times0.04$ m
  ($x\times y\times z$), a guess at the core stack inside the 154 × 198 × 86 mm folded envelope (S2).

**Box about its own center** (solid cuboid):

$$I_{xx,b}=\tfrac{m_b}{12}(b^2+c^2),\quad I_{yy,b}=\tfrac{m_b}{12}(a^2+c^2),\quad I_{zz,b}=\tfrac{m_b}{12}(a^2+b^2)$$

$$\tfrac{m_b}{12}=\tfrac{0.225}{12}=0.01875$$

$$I_{xx,b}=0.01875\,(0.0036+0.0016)=0.01875\times0.0052=9.75\times10^{-5}$$

$$I_{yy,b}=0.01875\,(0.0100+0.0016)=0.01875\times0.0116=2.175\times10^{-4}$$

$$I_{zz,b}=0.01875\,(0.0100+0.0036)=0.01875\times0.0136=2.55\times10^{-4}$$

**Point masses** (parallel-axis theorem: a point mass $m$ at distance $r_\perp$ from an
axis adds $m r_\perp^2$). Every motor is $|y|=l_y$ from the $x$ axis, $|x|=l_x$ from the
$y$ axis, and $l$ from the $z$ axis:

$$I_{xx,m}=m_m\,l_y^2=0.060\times0.007744=4.646\times10^{-4}$$

$$I_{yy,m}=m_m\,l_x^2=0.060\times0.004356=2.614\times10^{-4}$$

$$I_{zz,m}=m_m\,(l_x^2+l_y^2)=0.060\times0.0121=7.26\times10^{-4}$$

**Total:**

$$\begin{aligned} I_{xx} &= 9.75 \times 10^{-5} + 4.646 \times 10^{-4} = 5.621 \times 10^{-4} \ \text{kg}\cdot\text{m}^2 \\

I_{yy} &= 2.175 \times 10^{-4} + 2.614 \times 10^{-4} = 4.789 \times 10^{-4} \ \text{kg}\cdot\text{m}^2 \\

I_{zz} &= 2.55 \times 10^{-4} + 7.26 \times 10^{-4} = 9.81 \times 10^{-4} \ \text{kg}\cdot\text{m}^2 \end{aligned}$$

Products of inertia are taken as zero (the airframe is symmetric about $x$–$z$ and $y$–$z$).
Sanity check: for a flat body, $I_{zz}\approx I_{xx}+I_{yy}$ (perpendicular-axis theorem);
$5.62+4.79=10.41\approx9.81$ (×10⁻⁴). ✓ The difference is the box's height $c$.

For comparison, the Iris is $(0.029125,0.029125,0.055225)$ kg·m², about **50× larger**.

### 6.3 How `spawn_starling.py` applies this to the Iris USD

The USD has a `/body` rigid body and four `/rotorN` rigid bodies. We want the whole
vehicle to come out to $m$ and $\mathbf I$, so the body gets what's left after
the rotor prims' own mass $m_{r,j}$ and inertia, using the parallel-axis theorem at the new rotor positions:

$$m_{body}=m-\sum_j m_{r,j}$$

$$I_{xx,body}=I_{xx}-\sum_j m_{r,j}y_j^2,\quad I_{yy,body}=I_{yy}-\sum_j m_{r,j}x_j^2,\quad I_{zz,body}=I_{zz}-\sum_j m_{r,j}(x_j^2+y_j^2).$$

This ignores each rotor's spin inertia about its own axis (tiny for ~2 g props).

---

## 7. Aerodynamic drag (S8)

$$\mathbf F_{D,B}=-\mathrm{diag}(d_x,d_y,d_z)\,\mathbf v_B,\qquad \mathbf v_B=\mathbf R^\top\mathbf v_W.$$

- $\mathbf v_W$ is the world-frame velocity. $\mathbf R^\top$ rotates it into the body frame.
- The force opposes motion along each body axis, and it's applied to `/body`.
- Linear, not quadratic: a simple model that is reasonable at low speed.

**Scaling the Iris coefficients (est.).** Drag scales with area. We use the area of the
rectangle formed by the motors as a stand-in:

The Iris rotors sit at $|x|=0.13$ and $|y|=0.20$ (rear) or $0.22$ (front) (S10). Using the
mean $|y|=0.21$:

$$A_{Iris}=(2\times0.13)(2\times0.21)=0.26\times0.42=0.1092\ \text{m}^2$$

And for the Starling (S2):

$$A_{Starling}=0.132\times0.176=0.02323\ \text{m}^2$$

$$\frac{A_{Starling}}{A_{Iris}}=\frac{0.02323}{0.1092}=0.2128$$

$$d_x=0.50\times0.2128=0.106,\quad d_y=0.30\times0.2128=0.064,\quad d_z=0\ \ \text{N/(m/s)}.$$

$d_z=0$ is kept from Pegasus' Iris default. This is the weakest number in this
document: identify it from a flight log (§11).

---

## 8. Rigid-body equations of motion

Isaac Sim's PhysX integrates these for the simulator. The MPC uses the same
equations in Euler-angle form (§9).

### 8.1 Translational dynamics (Newton, world frame)

$$m\,\dot{\mathbf v}_W=\mathbf R\Big(\textstyle\sum_iF_i\,\mathbf e_z+\mathbf F_{D,B}\Big)-mg\,\mathbf e_z$$

$$\Rightarrow\ \dot{\mathbf v}_W=\frac{T}{m}\,\mathbf R\,\mathbf e_z-g\,\mathbf e_z+\frac1m\mathbf R\,\mathbf F_{D,B},\qquad \dot{\mathbf p}_W=\mathbf v_W.$$

- $\mathbf R$ = body-to-world rotation. $\mathbf R\mathbf e_z$ is the thrust direction in the world frame.
- All rotor thrusts are parallel to body $z$, so they add up to a single vector $T\,\mathbf R\mathbf e_z$.

### 8.2 Rotational dynamics (Euler, body frame)

$$\mathbf I\,\dot{\boldsymbol\omega}=\boldsymbol\tau-\boldsymbol\omega\times(\mathbf I\boldsymbol\omega),\qquad \boldsymbol\omega=(p,q,r),\ \boldsymbol\tau=(U_2,U_3,U_4).$$

With diagonal $\mathbf I$, writing out the cross product
$\boldsymbol\omega\times\mathbf I\boldsymbol\omega=\big((I_{zz}-I_{yy})qr,\ (I_{xx}-I_{zz})pr,\ (I_{yy}-I_{xx})pq\big)$:

$$\dot p=\frac{U_2+(I_{yy}-I_{zz})\,q\,r}{I_{xx}}$$

$$\dot q=\frac{U_3+(I_{zz}-I_{xx})\,p\,r}{I_{yy}}$$

$$\dot r=\frac{U_4+(I_{xx}-I_{yy})\,p\,q}{I_{zz}}$$

The $\omega\times I\omega$ terms are the gyroscopic coupling of the airframe itself.
Rotor gyroscopic torques (from the spinning props) aren't modelled by Pegasus.

### 8.3 Attitude kinematics, quaternion form (what Isaac / Pegasus use)

$$\dot{\mathbf q}=\tfrac12\,\mathbf q\times\begin{bmatrix}0\\\boldsymbol\omega\end{bmatrix}
=\tfrac12\begin{bmatrix}-q_x&-q_y&-q_z\\q_w&-q_z&q_y\\q_z&q_w&-q_x\\-q_y&q_x&q_w\end{bmatrix}\begin{bmatrix}p\\q\\r\end{bmatrix}$$

This is how the orientation changes when the body rotates at body rates $\boldsymbol\omega$.
(The middle entry of $(p,q,r)$ is the pitch rate $q$, not a quaternion component.)
$\mathbf q$ stays unit length, and it has no gimbal lock.

The rotation matrix from the quaternion:

$$\mathbf R(\mathbf q)=\begin{bmatrix}
1-2(q_y^2+q_z^2)&2(q_xq_y-q_wq_z)&2(q_xq_z+q_wq_y)\\
2(q_xq_y+q_wq_z)&1-2(q_x^2+q_z^2)&2(q_yq_z-q_wq_x)\\
2(q_xq_z-q_wq_y)&2(q_yq_z+q_wq_x)&1-2(q_x^2+q_y^2)\end{bmatrix}.$$

---

## 9. The 12-state Euler model (MPC form, `control/New_Drone_State_Equations.py`)

State $\mathbf x=(x,y,z,\phi,\theta,\psi,\dot x,\dot y,\dot z,p,q,r)$, input $\mathbf U=(U_1,U_2,U_3,U_4)$.
This is §8 with ZYX Euler angles and with drag left out. The MPC treats drag as a disturbance.

### 9.1 Rotation matrix (ZYX: $\mathbf R=\mathbf R_z(\psi)\mathbf R_y(\theta)\mathbf R_x(\phi)$)

$$\mathbf R=\begin{bmatrix}
c\psi c\theta & c\psi s\theta s\phi-s\psi c\phi & c\psi s\theta c\phi+s\psi s\phi\\
s\psi c\theta & s\psi s\theta s\phi+c\psi c\phi & s\psi s\theta c\phi-c\psi s\phi\\
-s\theta & c\theta s\phi & c\theta c\phi\end{bmatrix}$$

($c\cdot=\cos$, $s\cdot=\sin$.) The third column is the thrust direction $\mathbf R\mathbf e_z$.

### 9.2 Position kinematics

$$\dot x=\dot x,\quad \dot y=\dot y,\quad \dot z=\dot z$$

The velocity states are simply the derivatives of position.

### 9.3 Euler-angle kinematics

$$\begin{bmatrix}\dot\phi\\\dot\theta\\\dot\psi\end{bmatrix}=
\begin{bmatrix}1&\sin\phi\tan\theta&\cos\phi\tan\theta\\0&\cos\phi&-\sin\phi\\0&\sin\phi/\cos\theta&\cos\phi/\cos\theta\end{bmatrix}
\begin{bmatrix}p\\q\\r\end{bmatrix}$$

Written out, as in the code:

$$\dot\phi=p+\frac{q\sin\theta\sin\phi}{\cos\theta}+\frac{r\cos\phi\sin\theta}{\cos\theta}$$

$$\dot\theta=q\cos\phi-r\sin\phi$$

$$\dot\psi=\frac{q\sin\phi}{\cos\theta}+\frac{r\cos\phi}{\cos\theta}$$

This converts body rates to Euler-angle rates. It is singular at $\theta=\pm90^\circ$
(gimbal lock). The pitch limit of ±1.2 rad in `drone_constraints.py` stays clear of it.

### 9.4 Translational accelerations (§8.1 with the third column of §9.1, no drag)

$$\ddot x=\frac{U_1}{m}(\cos\phi\sin\theta\cos\psi+\sin\phi\sin\psi)$$

$$\ddot y=\frac{U_1}{m}(\cos\phi\sin\theta\sin\psi-\sin\phi\cos\psi)$$

$$\ddot z=\frac{U_1}{m}\cos\phi\cos\theta-g$$

Tilting the thrust vector produces horizontal acceleration. The vertical component
minus gravity sets climb or descent.

### 9.5 Rotational accelerations

Identical to §8.2, with $I_x=I_{xx}=5.62\times10^{-4}$, $I_y=I_{yy}=4.79\times10^{-4}$, $I_z=I_{zz}=9.81\times10^{-4}$:

$$\dot p=\frac{U_2+(I_y-I_z)qr}{I_x},\quad \dot q=\frac{U_3+(I_z-I_x)pr}{I_y},\quad \dot r=\frac{U_4+(I_x-I_y)pq}{I_z}$$

Starling coefficients:
- $(I_y-I_z)/I_x=(4.79-9.81)/5.62=-0.893$
- $(I_z-I_x)/I_y=(9.81-5.62)/4.79=0.875$
- $(I_x-I_y)/I_z=(5.62-4.79)/9.81=0.0846$

---

## 10. Hover equilibrium and linearization

### 10.1 Hover

At hover, $\phi=\theta=0$, all velocities and rates are 0, and $\ddot z=0$. From §9.4:

$$0=\frac{U_1}{m}-g\ \Rightarrow\ U_{1,hover}=mg=2.796\ \text{N},\qquad U_2=U_3=U_4=0.$$

With four equal rotors, each one carries a quarter of the weight:

$$F_{i,hover}=\frac{mg}{4}=\frac{2.796}{4}=0.699\ \text{N}$$

$$\omega_{hover}=\sqrt{\frac{mg}{4k_f}}=\sqrt{\frac{2.796}{4\times1.5\times10^{-6}}}=\sqrt{4.660\times10^{5}}=682.6\ \text{rad/s}$$

$$u_{hover}=\frac{\omega_{hover}-100}{1000}=\frac{682.6-100}{1000}=0.583$$

This is the PX4 `MPC_THR_HOVER` starting value. (Iris: 0.556.)

### 10.2 Linearization about hover ($\psi=0$)

Write $\mathbf x=\mathbf x_h+\delta\mathbf x$, $\mathbf U=\mathbf U_h+\delta\mathbf U$ and keep first-order terms
($\sin\alpha\approx\alpha$, $\cos\alpha\approx1$, and products of small quantities dropped):

$$\delta\ddot x=g\,\delta\theta,\qquad \delta\ddot y=-g\,\delta\phi,\qquad \delta\ddot z=\frac{\delta U_1}{m}$$

$$\delta\dot\phi=\delta p,\qquad \delta\dot\theta=\delta q,\qquad \delta\dot\psi=\delta r$$

$$\delta\dot p=\frac{\delta U_2}{I_x},\qquad \delta\dot q=\frac{\delta U_3}{I_y},\qquad \delta\dot r=\frac{\delta U_4}{I_z}$$

- $\ddot x\approx\frac{U_1}{m}\theta=g\theta$: pitching nose-down ($\theta>0$) accelerates forward along $+x$.
- $\ddot y\approx-\frac{U_1}{m}\phi=-g\phi$: rolling right ($\phi>0$) accelerates toward $-y$ (right).
- The gyroscopic terms are products of two rates, so they're second order and drop out.

State-space form $\delta\dot{\mathbf x}=\mathbf A\,\delta\mathbf x+\mathbf B\,\delta\mathbf U$:

$$\mathbf A=\begin{bmatrix}
\mathbf 0_3&\mathbf 0_3&\mathbf I_3&\mathbf 0_3\\
\mathbf 0_3&\mathbf 0_3&\mathbf 0_3&\mathbf I_3\\
\mathbf 0_3&\mathbf G&\mathbf 0_3&\mathbf 0_3\\
\mathbf 0_3&\mathbf 0_3&\mathbf 0_3&\mathbf 0_3\end{bmatrix},\qquad
\mathbf G=\begin{bmatrix}0&g&0\\-g&0&0\\0&0&0\end{bmatrix}=\begin{bmatrix}0&9.81&0\\-9.81&0&0\\0&0&0\end{bmatrix}$$

$$\mathbf B=\begin{bmatrix}\mathbf 0_{6\times4}\\
\begin{matrix}0&0&0&0\\0&0&0&0\\1/m&0&0&0\end{matrix}\\
\begin{matrix}0&1/I_x&0&0\\0&0&1/I_y&0\\0&0&0&1/I_z\end{matrix}\end{bmatrix}$$

Numerically:

$$\begin{aligned} \frac{1}{m} &= \frac{1}{0.285 \text{ kg}} = 3.509 \ \text{kg}^{-1} \\ \frac{1}{I_xx} &= \frac{1}{5.621 \times 10^{-4} \ \text{kg}\cdot\text{m}^2} = 1779 \ \text{kg}^{-1}\cdot\text{m}^{-2} \\ \frac{1}{I_yy} &= \frac{1}{4.79 \times 10^{-4} \ \text{kg}\cdot\text{m}^2} = 2088 \ \text{kg}^{-1}\cdot\text{m}^{-2} \\ \frac{1}{I_zz} &= \frac{1}{9.81 \times 10^{-4} \ \text{kg}\cdot\text{m}^2} = 1019 \ \text{kg}^{-1}\cdot\text{m}^{-2} \end{aligned}$$

Compared with the Iris ($1/m=0.667$, $1/I_x=34.3$), the Starling's input gains are about 5× (thrust)
and about 50× (torque) larger. MPC input weights $R$ and rate-loop gains have to be rescaled.

---

## 11. Performance limits (derived)

**Thrust-to-weight:**

$$\frac{T_{max}}{W}=\frac{7.26}{2.796}=2.60$$

S1 recommends at least 2.0 for good control, which leaves
$\frac{7.26}{2.0\cdot9.81}-0.285=0.370-0.285=0.085$ kg of payload margin. This matches S1's "about 85 g".

**Maximum acceleration magnitude** (all thrust, no drag):

$$a_{T,max}=\frac{T_{max}}{m}=\frac{7.26}{0.285}=25.47\ \text{m/s}^2$$

**Maximum vertical acceleration** (level, full thrust):

$$a_{z,max}=a_{T,max}-g=25.47-9.81=15.66\ \text{m/s}^2\ (\approx1.6\,g)$$

**Maximum tilt that still holds altitude.** $T_{max}\cos\theta=mg$ gives

$$\cos\theta_{max}=\frac{mg}{T_{max}}=\frac{1}{2.60}=0.385\ \Rightarrow\ \theta_{max}=67.4^\circ=1.176\ \text{rad}$$

The ±1.2 rad (68.8°) attitude limit in `drone_constraints.py` is slightly beyond this. At
the limit the Starling slowly loses altitude even at full throttle. Use ≤ 1.17 rad if
the MPC must hold height.

**Maximum horizontal acceleration at constant altitude:**

$$a_{h,max}=\sqrt{a_{T,max}^2-g^2}=\sqrt{25.47^2-9.81^2}=\sqrt{648.7-96.2}=\sqrt{552.5}=23.5\ \text{m/s}^2$$

**Maximum angular accelerations** (from §5.3, at the cost of collective thrust):

$$\dot p_{max}=\frac{\tau_{x,max}}{I_x}=\frac{0.319}{5.62\times10^{-4}}=568\ \text{rad/s}^2$$

$$\dot q_{max}=\frac{\tau_{y,max}}{I_y}=\frac{0.240}{4.79\times10^{-4}}=501\ \text{rad/s}^2$$

$$\dot r_{max}=\frac{\tau_{z,max}}{I_z}=\frac{0.201}{9.81\times10^{-4}}=205\ \text{rad/s}^2$$

These are above the ±300 / ±150 rad/s² in `drone_constraints.py`, so those limits stay
conservative.

**Steady top speed with linear drag** (level flight, horizontal force balance $m a_{h,max}=d_x v$):

$$v_{max}=\frac{m\,a_{h,max}}{d_x}=\frac{0.285\times23.5}{0.106}=63\ \text{m/s}$$

This is not realistic. Real drag grows roughly with $v^2$, so linear drag
underestimates it at speed. Treat $v_{max}$ from this model as meaningless, and cap velocity in
the controller (≈ 15 m/s suggested) until drag is identified from flight data:
fly at constant speeds $v_k$ at constant altitude, log tilt $\theta_k$, and fit
$d_x=\frac{mg\tan\theta_k}{v_k}$ (from $T\sin\theta=d_xv$, $T\cos\theta=mg$).

**Battery.** S1's thrust is quoted at 7.5 V, which indicates a 2S pack:
2 × 3.7 V = 7.4 V nominal, 2 × 4.2 V = 8.4 V full. Thrust drops roughly with $V^2$ as the pack sags, because
$\omega\propto V$ at a given duty cycle and $F\propto\omega^2$.

---

## 12. Where each parameter goes in the repo

| Parameter | `workspace/spawn_starling.py` | `rl/drone_params.py` | `control/` |
|---|---|---|---|
| $m$ | `MASS` (body MassAPI) | `mass` | `drone_constraints.m` |
| $\mathbf I$ | `INERTIA_DIAG` (body MassAPI) | `inertia_diag` | `New_Drone_State_Equations` `Ix, Iy, Iz` |
| $l_x,l_y$ | `ROTOR_XY` (rotor prims + joint anchors) | `rotor_xy` | (in MPC torques via mixer) |
| $k_f$ | `K_F` → `QuadraticThrustCurve` | `k_f` | `total_thrust_limits.max = 7.26` |
| $k_m$ | `K_M` → `QuadraticThrustCurve` | `k_m` | — |
| $d$ | `DRAG_COEFF` → `LinearDrag` | `drag_coeff` | — |
| $u_{hover}$ | printed at spawn | `hover_motor_output()` | PX4 `MPC_THR_HOVER` |

## 13. Open items to verify

1. Get the official Starling 2 datasheet and check $m$, thrust and motor spacing (S1, S2).
2. Measure inertia (bifilar/trifilar pendulum), or read it from the STEP model with CAD mass properties.
3. Measure $k_f$ and $k_m$ on a thrust/torque stand (or at least prop diameter and pitch).
4. Identify drag from flight logs (§11).
5. Find the camera mount position and pitch in the CAD model (`CAMERA_POS`, `CAMERA_PITCH_DEG`).
6. Retune the PX4 rate loop (`MC_ROLLRATE_*`, `MC_PITCHRATE_*`, `MC_YAWRATE_*`). The torque-to-inertia
   ratio changed about 4× and the inertia about 50× from the Iris.
