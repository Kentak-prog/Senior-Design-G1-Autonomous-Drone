"""
Closed-loop drone racing simulation.

Wires together the existing project modules only:
    - MPC_Optimizer.create_drone_mpc_optimizer   (NMPC controller)
    - New_Drone_State_Equations.get_drone_dynamics (nonlinear plant model)
    - Runge_Kutta_4_Discretizer.get_rk4_discretizer (plant integration)
    - Kalman_Filter_MPC.DroneCasadiEKF12D          (state estimator)
    - drone_constraints / MPC_Cost_Function        (used internally by the optimizer)

A reference "race track" (a horizontal figure-8 / lemniscate with a gentle
vertical undulation) is generated in this file and fed to the MPC as the
receding-horizon reference trajectory (X_ref). Each control loop:

    1. Build the next N+1 reference states starting at the current sim time.
    2. Solve the NMPC using the EKF's current state estimate as feedback.
    3. Apply the first control input to the "true" plant (RK4 rollout).
    4. Take a noisy measurement of the true state.
    5. Run the EKF predict/update to produce the next estimate.

Results are plotted as the flown path vs. the reference track.
"""
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

from MPC_Optimizer import create_drone_mpc_optimizer
from New_Drone_State_Equations import get_drone_dynamics
from Runge_Kutta_4_Discretizer import get_rk4_discretizer
from Kalman_Filter_MPC import DroneCasadiEKF12D


# ---------------------------------------------------------------------------
# Race track definition: lemniscate (figure-8) with vertical undulation
# ---------------------------------------------------------------------------
LAP_TIME = 12.0      # seconds per lap
TRACK_SCALE = 8.0    # meters, controls the size of the figure-8
Z_CENTER = 5.0        # meters, mean altitude/height state
Z_AMPLITUDE = 1.5     # meters, altitude undulation around the track


def track_position(t):
    """Parametric position on the figure-8 race line at time t."""
    theta = 2 * np.pi * (t % LAP_TIME) / LAP_TIME
    denom = 1.0 + np.sin(theta) ** 2
    x = TRACK_SCALE * np.cos(theta) / denom
    y = TRACK_SCALE * np.sin(theta) * np.cos(theta) / denom
    z = Z_CENTER + Z_AMPLITUDE * np.sin(2 * theta)
    return np.array([x, y, z])


def reference_state(t, eps=1e-3):
    """Full 12-state reference at time t: position, level attitude with
    yaw pointed along the direction of travel, and velocity via finite
    difference of the track position."""
    pos = track_position(t)
    pos_ahead = track_position(t + eps)
    vel = (pos_ahead - pos) / eps
    yaw = np.arctan2(vel[1], vel[0])
    return np.array([
        pos[0], pos[1], pos[2],
        0.0, 0.0, yaw,
        vel[0], vel[1], vel[2],
        0.0, 0.0, 0.0
    ])


def build_reference_horizon(t0, N, dt):
    ref = np.zeros((12, N + 1))
    for k in range(N + 1):
        ref[:, k] = reference_state(t0 + k * dt)
    return ref


# ---------------------------------------------------------------------------
# Closed-loop simulation
# ---------------------------------------------------------------------------
def run_race_simulation(num_laps=1.5, dt=0.02, N=10, add_noise=True, seed=0):
    rng = np.random.default_rng(seed)

    sim_time = LAP_TIME * num_laps
    steps = int(sim_time / dt)

    # NMPC controller (Opti instance is solved repeatedly with updated parameters)
    opti, X, U, X_init, U_prev, X_ref = create_drone_mpc_optimizer(N=N, dt=dt)

    # "True" plant model, integrated with the same RK4 discretizer the MPC uses internally
    f_continuous, states, controls = get_drone_dynamics()
    f_plant = get_rk4_discretizer(f_continuous, states, controls, dt)

    # State estimator
    ekf = DroneCasadiEKF12D(dt=dt)

    x_true = reference_state(0.0).reshape(12, 1)
    ekf.x = x_true.copy()
    u_prev = np.zeros((4, 1))

    hover_thrust = 9.81  # m=1kg -> hover thrust roughly equals g
    X_guess = np.tile(x_true, (1, N + 1))
    U_guess = np.tile(np.array([[hover_thrust], [0.0], [0.0], [0.0]]), (1, N))

    log = {
        "t": np.zeros(steps),
        "true": np.zeros((12, steps)),
        "est": np.zeros((12, steps)),
        "ref": np.zeros((12, steps)),
        "u": np.zeros((4, steps)),
    }

    meas_std = np.sqrt(np.diag(ekf.R)).reshape(12, 1)

    for step in range(steps):
        t = step * dt
        ref_horizon = build_reference_horizon(t, N, dt)

        opti.set_value(X_init, ekf.x)
        opti.set_value(U_prev, u_prev)
        opti.set_value(X_ref, ref_horizon)
        opti.set_initial(X, X_guess)
        opti.set_initial(U, U_guess)

        try:
            sol = opti.solve()
            X_sol = sol.value(X)
            U_sol = sol.value(U)
        except RuntimeError:
            # IPOPT failed to converge this step; fall back to its last iterate
            X_sol = opti.debug.value(X)
            U_sol = opti.debug.value(U)

        u0 = np.array(U_sol[:, 0]).reshape(4, 1)

        # Roll the true plant forward with the applied control
        x_true = np.array(f_plant(x_true, u0)).reshape(12, 1)
        if add_noise:
            x_true = x_true + rng.normal(0.0, 1e-3, size=(12, 1))

        # Noisy sensor measurement of the true state, fed to the EKF
        z_meas = x_true + (rng.normal(0.0, 1.0, size=(12, 1)) * meas_std if add_noise else 0.0)

        ekf.predict(u0)
        ekf.update(z_meas)

        log["t"][step] = t
        log["true"][:, step] = x_true.flatten()
        log["est"][:, step] = ekf.x.flatten()
        log["ref"][:, step] = ref_horizon[:, 0]
        log["u"][:, step] = u0.flatten()

        u_prev = u0
        # Warm-start next solve by shifting this solution one step forward
        X_guess = np.hstack([X_sol[:, 1:], X_sol[:, -1:]])
        U_guess = np.hstack([U_sol[:, 1:], U_sol[:, -1:]])

        if step % 50 == 0:
            print(f"t={t:6.2f}s  pos=({x_true[0,0]:6.2f}, {x_true[1,0]:6.2f}, {x_true[2,0]:5.2f})  "
                  f"U1={u0[0,0]:5.2f}N")

    return log


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_race(log):
    fig = plt.figure(figsize=(15, 10))

    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.plot(log["ref"][0], log["ref"][1], log["ref"][2], "k--", label="Reference track", linewidth=1.5)
    ax3d.plot(log["true"][0], log["true"][1], log["true"][2], "b-", label="MPC flown path")
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.legend()
    ax3d.set_title("3D Race Trajectory")

    ax_top = fig.add_subplot(2, 2, 2)
    ax_top.plot(log["ref"][0], log["ref"][1], "k--", label="Reference track")
    ax_top.plot(log["true"][0], log["true"][1], "b-", label="MPC flown path")
    ax_top.set_xlabel("X (m)")
    ax_top.set_ylabel("Y (m)")
    ax_top.axis("equal")
    ax_top.legend()
    ax_top.set_title("Top-down view")

    ax_err = fig.add_subplot(2, 2, 3)
    pos_err = np.linalg.norm(log["true"][0:3] - log["ref"][0:3], axis=0)
    ax_err.plot(log["t"], pos_err, "r-")
    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("Position tracking error (m)")
    ax_err.set_title("Tracking Error vs Time")
    ax_err.grid(True)

    ax_vel = fig.add_subplot(2, 2, 4)
    true_speed = np.linalg.norm(log["true"][6:9], axis=0)
    ref_speed = np.linalg.norm(log["ref"][6:9], axis=0)
    ax_vel.plot(log["t"], ref_speed, "k--", label="Reference speed")
    ax_vel.plot(log["t"], true_speed, "b-", label="MPC flown speed")
    ax_vel.set_xlabel("Time (s)")
    ax_vel.set_ylabel("Speed (m/s)")
    ax_vel.set_title("Velocity vs Time")
    ax_vel.legend()
    ax_vel.grid(True)

    plt.tight_layout()
    plt.savefig("race_result.png", dpi=150)
    print("Saved plot to race_result.png")
    plt.show()


if __name__ == "__main__":
    log = run_race_simulation(num_laps=1.5)
    plot_race(log)
