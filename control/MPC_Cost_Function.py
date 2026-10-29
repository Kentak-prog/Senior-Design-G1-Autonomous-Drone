import casadi as ca
import numpy as np
from drone_constraints import drone_constraints

# --- Weight Matrices (shared by both cost functions below) ---
# Diagonal weights for the 12 drone states
Q_DIAG = [20.0, 20.0, 30.0,   # Position tracking (High priority on altitude Z)
          5.0,  5.0,  10.0,   # Attitude tracking (Roll, Pitch, Yaw)
          1.0,  1.0,  1.0,    # Linear velocities
          0.1,  0.1,  0.1]    # Body angular rates
# Diagonal weights for the 4 actuator commands [U1, U2, U3, U4]
R_DIAG = [0.01, 0.05, 0.05, 0.05]
# Stiff penalty weight on motor rate-of-change (Prevents instantaneous motor jumps)
R_RATE_DIAG = [0.8, 1.0, 1.0, 1.0]
TERMINAL_SCALE = 2.0


def _stage_and_terminal_cost(states, U, X_ref, U_previous, N, m):
    """
    Tracking + effort + rate cost for a horizon. states(k) returns the
    predicted state at step k (0..N), either a decision variable or a rollout.
    """
    Q = ca.diag(Q_DIAG)
    R = ca.diag(R_DIAG)
    R_rate = ca.diag(R_RATE_DIAG)

    #fixed bug where drone hovered offset in z because thrust effort was penalized from zero instead of hover
    # Effort is measured relative to hover, not zero. Penalizing raw U1 pulls
    # thrust below m*g, which left the drone sitting ~0.24 m off its altitude
    # reference even when hovering.
    u_hover = ca.DM([m * 9.81, 0.0, 0.0, 0.0])

    total_cost = 0.0
    for k in range(N):
        u_k = U[:, k]

        # A. State Tracking Error Cost: (x_k - x_ref)^T * Q * (x_k - x_ref)
        state_error = states(k) - X_ref[:, k]
        total_cost += ca.mtimes([state_error.T, Q, state_error])

        # B. Control Effort Cost about hover: (u_k - u_hover)^T * R * (u_k - u_hover)
        u_dev = u_k - u_hover
        total_cost += ca.mtimes([u_dev.T, R, u_dev])

        # C. Actuator Rate-of-Change Cost: (u_k - u_{k-1})^T * R_rate * (u_k - u_{k-1})
        u_error = u_k - (U_previous if k == 0 else U[:, k - 1])
        total_cost += ca.mtimes([u_error.T, R_rate, u_error])

    # --- Terminal Stage Cost ---
    # Evaluates the final predicted state (index N) against the final reference point
    terminal_error = states(N) - X_ref[:, N]
    total_cost += ca.mtimes([terminal_error.T, Q * TERMINAL_SCALE, terminal_error])
    return total_cost


def compute_mpc_cost(X, U, X_ref, U_previous, N=10, m=drone_constraints["mass"]): #kg
    """
    NMPC tracking cost on the multiple-shooting STATE VARIABLES X (12 x N+1).

    MPC_Optimizer.py already constrains X[:, 0] == X_init and
    X[:, k+1] == f_discrete(X[:, k], U[:, k]), so X[:, k] IS the predicted
    state -- the same value compute_mpc_cost_with_rk4 rebuilds by rolling RK4
    forward inside the cost. Same optimum (verified to ~1e-15 N on the
    controls), but the cost stays a simple quadratic in (X, U) instead of a
    10-deep nest of RK4 steps, so the solver's derivatives are cheap.

    Symbolic Inputs:
        X:          CasADi matrix (12 x N+1) -> predicted states (decision variables)
        U:          CasADi matrix (4 x N)    -> Control inputs over horizon (U1, U2, U3, U4)
        X_ref:      CasADi matrix (12 x N+1) -> Reference trajectory states
        U_previous: CasADi vector (4 x 1)    -> Actual input applied in the prior timestep
        m:          float                    -> Drone mass (kg)
    """
    return _stage_and_terminal_cost(lambda k: X[:, k], U, X_ref, U_previous, N, m)


def compute_mpc_cost_with_rk4(X_current, U, RL_ref_trajectory, U_previous, f_discrete, N=10, m = drone_constraints["mass"]): #kg
    """
    Same cost, but the predicted states come from rolling the RK4 discrete
    model forward from X_current inside the cost (single shooting). Kept for
    reference; MPC_Optimizer.py now uses compute_mpc_cost, which is much
    faster to solve and gives the same controls.

    Symbolic Inputs:
        X_current:          CasADi vector (12 x 1)    -> Real-time state estimate of the drone
        U:                  CasADi matrix (4 x N)    -> Control inputs over horizon (U1, U2, U3, U4)
        RL_ref_trajectory:  CasADi matrix (12 x N+1) -> Reference trajectory states from RL model
        U_previous:         CasADi vector (4 x 1)    -> Actual input applied in the prior timestep
        f_discrete:         CasADi Function          -> Your RK4 integrator: X_next = f_discrete(X, U)
        m:                  float                    -> Drone mass (kg), same value passed to get_drone_dynamics()
    """
    rollout = [X_current]
    for k in range(N):
        rollout.append(f_discrete(rollout[-1], U[:, k]))
    return _stage_and_terminal_cost(lambda k: rollout[k], U, RL_ref_trajectory, U_previous, N, m)
