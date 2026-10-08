import casadi as ca
import numpy as np
from drone_constraints import drone_constraints

def compute_mpc_cost_with_rk4(X_current, U, RL_ref_trajectory, U_previous, f_discrete, N=10, m = drone_constraints["mass"]): #kg
    """
    Computes the total NMPC tracking cost by rolling out drone dynamics using 
    the RK4 discrete function inside the horizon loop.
    
    Symbolic Inputs:
        X_current:          CasADi vector (12 x 1)    -> Real-time state estimate of the drone
        U:                  CasADi matrix (4 x N)    -> Control inputs over horizon (U1, U2, U3, U4)
        RL_ref_trajectory:  CasADi matrix (12 x N+1) -> Reference trajectory states from RL model
        U_previous:         CasADi vector (4 x 1)    -> Actual input applied in the prior timestep
        f_discrete:         CasADi Function          -> Your RK4 integrator: X_next = f_discrete(X, U)
        m:                  float                    -> Drone mass (kg), same value passed to get_drone_dynamics()
    """
    # --- 1. Define Weight Matrices ---
    # Diagonal weights for the 12 drone states
    Q_diag = [20.0, 20.0, 30.0,   # Position tracking (High priority on altitude Z)
              5.0,  5.0,  10.0,   # Attitude tracking (Roll, Pitch, Yaw)
              1.0,  1.0,  1.0,    # Linear velocities
              0.1,  0.1,  0.1]    # Body angular rates
    Q = ca.diag(Q_diag)
    
    # Diagonal weights for the 4 actuator commands [U1, U2, U3, U4]
    R = ca.diag([0.01, 0.05, 0.05, 0.05])

    #fixed bug where drone hovered offset in z because thrust effort was penalized from zero instead of hover
    # Effort is measured relative to hover, not zero. Penalizing raw U1 pulls
    # thrust below m*g, which left the drone sitting ~0.24 m off its altitude
    # reference even when hovering.
    u_hover = ca.DM([m * 9.81, 0.0, 0.0, 0.0])
    
    # Stiff penalty weight on motor rate-of-change (Prevents instantaneous motor jumps)
    R_rate = ca.diag([0.8, 1.0, 1.0, 1.0])
    
    # Initialize optimization tracking variables
    total_cost = 0.0
    x_k = X_current  # Initialize the rollout loop with the current telemetry state
    
    # --- 2. Receding Horizon Dynamic Rollout & Cost Loop ---
    for k in range(N):
        u_k = U[:, k]
        x_ref_k = RL_ref_trajectory[:, k]
        
        # A. State Tracking Error Cost: (x_k - x_ref)^T * Q * (x_k - x_ref)
        state_error = x_k - x_ref_k
        total_cost += ca.mtimes([state_error.T, Q, state_error])
        
        # B. Control Effort Cost about hover: (u_k - u_hover)^T * R * (u_k - u_hover)
        u_dev = u_k - u_hover
        total_cost += ca.mtimes([u_dev.T, R, u_dev])
        
        # C. Actuator Rate-of-Change Cost: (u_k - u_{k-1})^T * R_rate * (u_k - u_{k-1})
        if k == 0:
            u_error = u_k - U_previous
        else:
            u_error = u_k - U[:, k - 1]
            
        total_cost += ca.mtimes([u_error.T, R_rate, u_error])
        
        # D. Dynamic Physics Propagation (Predict next state using RK4 function)
        # This replaces independent equality constraints by chaining the states symbolically
        x_k = f_discrete(x_k, u_k)
        
    # --- 3. Terminal Stage Cost ---
    # Evaluates the final predicted state (index N) against the final RL reference point
    terminal_error = x_k - RL_ref_trajectory[:, N]
    Q_terminal = Q * 2.0  
    
    total_cost += ca.mtimes([terminal_error.T, Q_terminal, terminal_error])
    
    return total_cost
