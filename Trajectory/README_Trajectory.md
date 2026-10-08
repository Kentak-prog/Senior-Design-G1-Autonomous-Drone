# Minimum-Snap Trajectory Generation

This document explains how `min_snap.py` plans a smooth path for the drone through a list of waypoints (gates). It works through the math in the order the code uses it, with a small worked example the whole way through.

**Reference:** D. Mellinger and V. Kumar, *Minimum Snap Trajectory Generation and Control for Quadrotors*, ICRA 2011.

---

## 1. Why snap?

### Differential flatness

A quadrotor is **differentially flat**: if you know its position `[x y z]ᵀ` and its yaw angle `ψ` as smooth functions of time, you can compute everything else about how it must fly. That includes roll, pitch, body rates, total thrust, and motor torques. These four signals (`x, y, z, ψ`) are called the *flat outputs*.

Roll `φ` and pitch `θ` are **not** flat outputs. They are determined by the acceleration: the drone has to tilt its thrust vector to point where it needs to accelerate. That's why we only plan position (and optionally yaw), and never plan roll or pitch directly.

### What each derivative means physically

| Derivative | Name | What it controls on the drone |
|---|---|---|
| d/dt | velocity | speed and direction of travel |
| d²/dt² | acceleration | **thrust**: its magnitude sets total thrust `U1`, its direction sets roll and pitch |
| d³/dt³ | jerk | how fast the thrust direction turns, i.e. **body rates** `p, q` |
| d⁴/dt⁴ | snap | **angular acceleration**, i.e. the torques the motors must produce `[U2 U3]` |

So keeping snap small keeps the motor torques small and smooth. Real motors can't deliver instantaneous changes in torque, so a low-snap path is one the drone can actually follow.

---

## 2. The shape of the answer: calculus of variations

Consider a path in 3D:

```
p(t) = < x(t), y(t), z(t) >
```

We want the path between two waypoints that minimizes

```
J = ∫₀ᵀ ( p⁽⁴⁾(t) )² dt        (1)
```

The Euler–Lagrange equation for this cost is

```
p⁽⁸⁾(t) = 0
```

Anything whose 8th derivative is zero is a polynomial of degree 7. So **the optimal path on each segment is a 7th-degree polynomial**, separately for each axis.

### Setup used in the code

- The path is split into **segments**, one between each pair of consecutive waypoints.
- Each segment, on each axis, gets its own polynomial in **local time** `t ∈ [0, Tᵢ]`. Local time restarts at 0 at the start of every segment.
- Each polynomial has 8 coefficients:

```
p(t) = c₀ + c₁t + c₂t² + c₃t³ + c₄t⁴ + c₅t⁵ + c₆t⁶ + c₇t⁷ = Σᵢ₌₀⁷ cᵢ tⁱ
```

**Key fact:** the polynomial is not linear in `t`, but it is **linear in the coefficients `c`**, and so are all of its derivatives. Every constraint below ends up as "a row of numbers · c = a value," which is what makes the whole problem solvable with linear algebra.

### Derivatives of the polynomial

Differentiating `tᵏ` r times gives `k!/(k−r)! · t^(k−r)` (for `k ≥ r`; otherwise 0). So:

```
p⁽ʳ⁾(t) = Σ_{k ≥ r}  k! / (k−r)!  ·  t^(k−r)  ·  c_k
```

In particular, snap is:

```
p⁽⁴⁾(t) = Σᵢ₌₄⁷  cᵢ · i! / (i−4)! · t^(i−4)
```

Coefficients `c₀` to `c₃` disappear after four derivatives.

In the code, the table `D[r, k] = k!/(k−r)!` stores these multipliers, and `poly_deriv_row(t, r)` returns the row of numbers for "the r-th derivative at time t."

---

## 3. Turning the cost into a matrix Q

Substitute the snap polynomial into (1) and expand the square:

```
J = ∫₀ᵀ ( Σᵢ₌₄⁷ cᵢ · i!/(i−4)! · t^(i−4) ) ( Σⱼ₌₄⁷ cⱼ · j!/(j−4)! · t^(j−4) ) dt
```

The coefficients don't depend on `t`, so they come out of the integral:

```
J = Σᵢ₌₄⁷ Σⱼ₌₄⁷  cᵢ cⱼ · i! j! / ((i−4)! (j−4)!) · ∫₀ᵀ t^(i+j−8) dt
```

and

```
∫₀ᵀ t^(i+j−8) dt = T^(i+j−7) / (i+j−7)
```

So

```
J(c) = Σᵢ Σⱼ cᵢ cⱼ · Qᵢⱼ ,     Qᵢⱼ = i! j! / ((i−4)! (j−4)!) · T^(i+j−7) / (i+j−7)
```

A double sum of `cᵢ · Qᵢⱼ · cⱼ` is exactly the matrix product `cᵀ Q c`. So:

```
J = cᵀ Q c         (Q is 8×8 for one segment)
```

### What Q looks like

```
        c0  c1  c2  c3     c4         c5          c6          c7
c0   [   0   0   0   0      0          0           0           0     ]
c1   [   0   0   0   0      0          0           0           0     ]
c2   [   0   0   0   0      0          0           0           0     ]
c3   [   0   0   0   0      0          0           0           0     ]
c4   [   0   0   0   0    576 T     1440 T²     2880 T³     5040 T⁴  ]
c5   [   0   0   0   0   1440 T²    4800 T³    10800 T⁴    20160 T⁵  ]
c6   [   0   0   0   0   2880 T³   10800 T⁴    25920 T⁵    50400 T⁶  ]
c7   [   0   0   0   0   5040 T⁴   20160 T⁵    50400 T⁶   100800 T⁷  ]
```

- The first four rows and columns are zero, because `c₀…c₃` don't affect snap. The cost alone can't determine them; the constraints in Section 4 pin them down.
- Q is symmetric, because `cᵢcⱼ` and `cⱼcᵢ` are the same term.
- Sanity check: snap from `c₄t⁴` alone is `24c₄`, and `∫₀ᵀ (24c₄)² dt = 576T·c₄²`. That matches `Q₄₄`.

### Hessian view (useful for checking)

The Hessian of `J` (the matrix of second derivatives) is a constant matrix `H = 2Q`. Since `J` is purely quadratic, `J = ½ cᵀHc` exactly, so you can also get Q by differentiating `J` twice and halving it. This is a good way to verify the formula symbolically.

In the code: `snap_cost_matrix(T)` builds this 8×8 matrix.

### Multiple segments

Each segment has its own 8 coefficients. For 2 segments on one axis, stack them into one 16-long vector:

```
c = [ a₀ a₁ … a₇ | b₀ b₁ … b₇ ]ᵀ      a = segment 1, b = segment 2
```

The total cost is the sum of the segment costs. Segments share no coefficients, so there are no cross terms, and the combined matrix is block-diagonal:

```
J_tot = Σ J = cᵀ Q_tot c

Q_tot = [ Q(T₁)    0    ]      16×16, to match c (16×1)
        [   0    Q(T₂)  ]      each 0 is an 8×8 zero block
```

Each block uses its own segment duration `Tᵢ`. For all three axes, the code repeats this block-diagonal pattern for x, y and z (`np.kron(np.eye(3), Q_axis)`).

---

## 4. The rules: building A and b

Minimizing `cᵀQc` alone gives `c = 0`, a drone that never moves (zero snap). The **constraints** force the path to actually go through the waypoints, which is what makes the answer nonzero and useful.

Each constraint is one linear equation, written as one row of a matrix `A` (also called `A_eq`) and one value in a vector `b`:

```
A c = b
```

### The building block: recipe rows

Plug a specific time into the polynomial. For example, at `t = 2`:

```
p(2) = c₀ + 2c₁ + 4c₂ + 8c₃ + 16c₄ + 32c₅ + 64c₆ + 128c₇
```

That's a list of numbers times the coefficients:

```
[1 2 4 8 16 32 64 128] · c = position at t = 2
```

The same idea works for every derivative. These rows come from `poly_deriv_row(t, r)`:

| r | at t = 0 | at t = 2 |
|---|---|---|
| 0 (position) | [1 0 0 0 0 0 0 0] | [1 2 4 8 16 32 64 128] |
| 1 (velocity) | [0 1 0 0 0 0 0 0] | [0 1 4 12 32 80 192 448] |
| 2 (acceleration) | [0 0 2 0 0 0 0 0] | [0 0 2 12 48 160 480 1344] |
| 3 (jerk) | [0 0 0 6 0 0 0 0] | [0 0 0 6 48 240 960 3360] |
| 4 (snap) | [0 0 0 0 24 0 0 0] | [0 0 0 0 24 240 1440 6720] |

At `t = 0` every row has a single nonzero entry, since all terms with `t` vanish. That's why start-of-segment constraints directly fix the low coefficients that Q ignores.

### Worked example: 1D, 2 segments

Waypoints `x₀ = 0`, `x₁ = 5`, `x₂ = 8`, with both segments lasting `T = 2` s.

```
   x₀=0 ───── a ───── x₁=5 ───── b ───── x₂=8
```

Each row below has 16 entries: the left half applies to segment `a`, the right half to segment `b`. Placing a recipe in the left or right half is what tells the solver which segment a rule is about.

**Waypoint positions**

```
a starts at x₀   [1 0 0 0  0  0  0   0 | 0 0 0 0  0  0  0   0] = 0
a ends at x₁     [1 2 4 8 16 32 64 128 | 0 0 0 0  0  0  0   0] = 5
b starts at x₁   [0 0 0 0  0  0  0   0 | 1 0 0 0  0  0  0   0] = 5
b ends at x₂     [0 0 0 0  0  0  0   0 | 1 2 4 8 16 32 64 128] = 8
```

`x₁` appears twice, as the end of `a` and the start of `b`. That's how the two segments meet at the same point.

**Start at rest** (velocity, acceleration and jerk are zero at the first waypoint)

```
a vel at x₀      [0 1 0 0 0 0 0 0 | 0 … 0] = 0
a acc at x₀      [0 0 2 0 0 0 0 0 | 0 … 0] = 0
a jerk at x₀     [0 0 0 6 0 0 0 0 | 0 … 0] = 0
```

**End at rest**

```
b vel at x₂      [0 … 0 | 0 1 4 12 32  80  192  448] = 0
b acc at x₂      [0 … 0 | 0 0 2 12 48 160  480 1344] = 0
b jerk at x₂     [0 … 0 | 0 0 0  6 48 240  960 3360] = 0
```

**Smooth handoff at x₁**

```
vel matches      [0 1 4 12 32  80  192  448 | 0 -1  0  0   0 0 0 0] = 0
acc matches      [0 0 2 12 48 160  480 1344 | 0  0 -2  0   0 0 0 0] = 0
jerk matches     [0 0 0  6 48 240  960 3360 | 0  0  0 -6   0 0 0 0] = 0
snap matches     [0 0 0  0 24 240 1440 6720 | 0  0  0  0 -24 0 0 0] = 0
```

"Velocity at the end of `a` equals velocity at the start of `b`" is rewritten as

```
(a's velocity at t = T₁) − (b's velocity at t = 0) = 0
```

so the row has `a`'s end recipe on the left and the **negative** of `b`'s start recipe on the right. These are the only rows that involve both segments, so they're what ties the pieces into one path. Without them, the drone could arrive at `x₁` moving one way and leave moving another, an instant velocity change that is physically impossible.

### Summary of the rules

- **Waypoints:** each segment starts at its waypoint and ends at the next one.
- **Rest-to-rest:** velocity, acceleration and jerk are zero at the first and last waypoint. Position there is already set by the waypoint rows.
- **Continuity:** at every interior waypoint, derivatives 1 through 4 (velocity, acceleration, jerk, snap) at the end of segment `n` equal those at the start of segment `n + 1`. Position continuity comes for free, because both segments are pinned to the same waypoint.

At the optimum, derivatives 5 and 6 also come out continuous on their own (a result of the calculus of variations), so the final path is C⁶ smooth even though only up to snap is enforced.

### Counting degrees of freedom

In the example: 16 unknowns and 14 equations, leaving **2 free choices** for the optimizer. Notice what isn't fixed: the velocity and acceleration at `x₁` must match on both sides, but nothing says what they must be. The optimizer picks the values that make total snap smallest.

In general, with `M` segments on one axis: `8M` unknowns and `6M + 2` equations, leaving `2M − 2` free choices.

### In the code

```python
def add(blocks, value):
    row = np.zeros(NV)                        # start with all zeros
    for ax, seg, vec in blocks:
        start = ax * N + seg * n              # find this axis/segment's 8 slots
        row[start:start + n] += vec           # drop the recipe in
    A_rows.append(row)
    b.append(value)
```

Each `add()` call creates one row. A continuity row passes two blocks, one positive and one negative:

```python
add([(ax, s,     poly_deriv_row(self.T[s], r)),     # end of this segment
     (ax, s + 1, -poly_deriv_row(0.0, r))], 0.0)    # minus start of next
```

With three axes, each row has `3 × 8M` entries: all of x's slots, then y's, then z's.

---

## 5. Gate direction: crossing perpendicularly

We can't fly through a gate at just any angle; we want to go straight through it, perpendicular to its face.

Let `d` be the gate's **normal** (the arrow sticking straight out of the gate's face), and let the drone's velocity at the gate be

```
v = < vₓ, v_y, v_z >
```

We need `v` to point along `d`. "Points along d" isn't a linear equation, so the code says the same thing a different way: **the velocity has no sideways component.**

Pick two unit vectors `u₁` and `u₂` that lie in the gate's plane (both perpendicular to `d` and to each other). If `v` has no component along either one, the only direction left is along `d`:

```
v · u₁ = 0    and    v · u₂ = 0
```

```
              d
              ↑
              │  u₂
              │ /
       ───────┼────── u₁     (gate plane)
```

Each of these is linear in the coefficients, so each becomes one more row of `A`. Because `v · u = uₓvₓ + u_y v_y + u_z v_z`, these rows mix x, y and z velocity together. **That's why the code solves all three axes in one system** instead of three separate ones.

**Example:** a gate facing along x has `d = (1, 0, 0)`, so `u₁, u₂` are just the y and z directions. The rules become `v_y = 0` and `v_z = 0`: no sideways or vertical motion at the gate.

### What this does and doesn't guarantee

- It controls the **direction** of the velocity. The **position** constraint (the waypoint row) is what makes the drone pass through the gate's center. The two work together.
- The optimizer still chooses the **speed** along `d`.
- `v = 0` and `v` pointing **backwards** along `−d` also satisfy `v · u₁ = v · u₂ = 0`. With sensible waypoints the solver goes forward, but it's worth checking `evaluate(t_gate, 1) · d > 0` after solving.
- It only fixes the velocity at the exact instant the drone reaches the waypoint. Because the path is smooth, it stays closely aligned while inside the gate.
- Direction constraints are only applied at interior waypoints. At the first and last waypoints the drone is at rest anyway.

### In the code

`_perpendicular_basis(d)` builds `u₁` and `u₂` from cross products. It crosses `d` with straight up `(0, 0, 1)`, or with `(1, 0, 0)` if `d` itself points mostly up or down, so the cross product never collapses to zero.

**Measured effect** (waypoints `(0,0,0) → (5,2,1) → (6,6,0)`, 2 s per segment, gate at the middle facing along x):

| | Velocity at gate | Angle from gate normal | Total snap |
|---|---|---|---|
| No direction constraint | (3.28, 3.28, 0) | 45° | 1955 |
| With direction constraint | (3.28, 0, 0) | 0° | 9707 |

Without it, the drone cuts the corner diagonally. With it, the drone goes straight through, at the cost of a sharper turn and more snap.

---

## 6. Solving: Lagrange multipliers and the KKT matrix

We now have:

- a **cost** to minimize: `cᵀQc`
- **rules** that must hold exactly: `Ac = b`

### Small example to build intuition

Minimize `f(x, y) = x² + y²` subject to `g(x, y) = x + y = 2`.

The level sets of `f` are circles around the origin; the constraint is a line. The best point is where the line just touches the smallest circle, at `(1, 1)`. At that point the gradient of the cost lines up with the gradient of the constraint. If it didn't, you could slide along the line and lower the cost.

"Lines up with" means one is a multiple of the other. The multiple `λ` is the **Lagrange multiplier**:

```
∇f = −λ ∇g    →    ∇f + λ∇g = 0
```

Here `∇f = <2x, 2y>` and `∇g = <1, 1>`, which gives three linear equations:

```
2x      + λ = 0
     2y + λ = 0           →    [ 2  0  1 ] [ x ]   [ 0 ]
 x +  y     = 2                [ 0  2  1 ] [ y ] = [ 0 ]
                               [ 1  1  0 ] [ λ ]   [ 2 ]
```

Solving gives `x = 1, y = 1, λ = −2`.

### The same thing for the drone

- The cost is `cᵀQc`, and its gradient is `∇(cᵀQc) = 2Qc`.
- The constraint is `Ac = b`, whose gradients are the rows of `A`. Weighted by one multiplier per rule, their combined "push back" is `Aᵀλ`.

The optimality conditions are:

```
2Qc + Aᵀλ = 0      (no direction lowers the cost without breaking a rule)
       Ac = b      (every rule holds)
```

Stacked into one square system, this is the **KKT matrix** (Karush–Kuhn–Tucker):

```
[ 2Q   Aᵀ ] [ c ]   [ 0 ]
[ A    0  ] [ λ ] = [ b ]
```

| Block | Meaning |
|---|---|
| `2Q` | gradient of the cost |
| `Aᵀ` | constraint directions, weighted by `λ` |
| `A` | the rules themselves |
| `0` | `λ` doesn't appear in the rules |
| `[0; b]` | zeros for "gradients balance," `b` for the rule values |

For the 2-segment, 1-axis example: 16 coefficients + 14 multipliers = a 30×30 system.

### Why one solve is enough

The cost is quadratic, so its gradient is linear. The rules are linear too. So the optimality conditions are a plain linear system, solved in one step with no iteration and no starting guess. Because snap² is never negative (Q is "bowl-shaped"), the solution is a true minimum, and the constraints pin down the coefficients Q doesn't care about, so the answer is unique.

### In the code

```python
KKT[:NV, :NV] = 2.0 * Q
KKT[:NV, NV:] = A.T
KKT[NV:, :NV] = A
rhs = np.concatenate((np.zeros(NV), b))

c = np.linalg.solve(KKT, rhs)[:NV]      # keep c, discard λ
```

The multipliers `λ` are needed to make the system square and solvable, but are thrown away afterward. `c.reshape(3, M, 8)` then organizes the coefficients as `[axis][segment][coefficient]`.

**Result for the worked example** (`0 → 5 → 8`, `T = 2` s each): the drone passes `x₁ = 5` at 4.375 m/s, and the total snap cost is 645.75.

---

## 7. Using the coefficients

To get the drone's state at a global time `t`, `evaluate(t, r)`:

1. finds which segment `t` falls in (`searchsorted` on the segment start times),
2. converts to local time: `t − (start time of that segment)`,
3. dots that segment's 8 coefficients with `poly_deriv_row(local_t, r)`.

`r = 0` gives position, `1` velocity, `2` acceleration, `3` jerk, `4` snap. `sample(dt)` does the same for many times at once, which is what you'd plot or send to the controller.

---

## 8. Choosing segment times

Everything above assumes the durations `T₁, T₂, …` are already known. Q depends on them, and so do the end-of-segment rows of A.

### Initial guess

Each segment gets its own time, proportional to its length:

```
Tᵢ = distanceᵢ / cruise speed      (minimum 0.2 s)
```

This is per segment, not one constant `T` for the whole path. Cruise speed is `v_cruise_frac × v_max` (60% of max by default).

### Fitting to the drone's limits (`fit_to_limits`)

The drone has a maximum speed `v_max` and acceleration `a_max`. Rather than re-tuning every segment, the code uses a scaling trick:

> If every segment time is multiplied by the same factor `s`, the path keeps exactly the same shape. Velocity scales by `1/s` and acceleration by `1/s²`.

So the code:

1. solves with the initial times,
2. samples the result every 5 ms to find peak speed and acceleration,
3. computes
   ```
   s = max( peak_v / v_max ,  √(peak_a / a_max) )
   ```
   so whichever limit is tighter is just reached,
4. multiplies every `Tᵢ` by `s` and solves again.

In exact math one rescale is enough. The loop (up to 5 passes) only cleans up the error from estimating peaks by sampling.

### Limitations and possible improvements

- Only the **overall pace** is tuned. The **ratio** of time between segments stays fixed by distance. Full min-snap planners also optimize each segment's share of time (for example, by gradient descent on total snap with total time fixed), which can give noticeably smoother or faster paths.
- Tuning `v_cruise_frac` and the initial time split is the main lever for trading smoothness against lap time.

---

## 9. Pipeline summary

```
waypoints + gate directions
        │
        ▼
initial segment times  (distance / cruise speed)
        │
        ▼
Q blocks  (snap cost per segment)  ──┐
A, b rows (waypoints, rest,          ├──►  KKT system  ──►  one linear solve  ──►  coefficients c
           continuity, gate dir.)  ──┘
        │
        ▼
evaluate / sample  ──►  check peak v and a  ──►  rescale all times  ──►  repeat until within limits
```

## Glossary

| Term | Meaning |
|---|---|
| Snap | 4th derivative of position |
| Segment | the piece of path between two consecutive waypoints |
| Local time | time measured from the start of the current segment |
| Q | matrix that turns coefficients into total snap² cost: `J = cᵀQc` |
| A, b | constraint rows and their values: `Ac = b` |
| λ | Lagrange multipliers, one per constraint |
| KKT matrix | the combined linear system that gives the optimal `c` |
| Differential flatness | position + yaw over time determine the full drone state and inputs |
