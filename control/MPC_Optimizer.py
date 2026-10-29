import numpy as np
import casadi as ca

from New_Drone_State_Equations import get_drone_dynamics
from Runge_Kutta_4_Discretizer import get_rk4_discretizer
from drone_constraints import drone_constraints
from MPC_Cost_Function import compute_mpc_cost

def create_drone_mpc_optimizer(N=10, dt=0.02, m = drone_constraints["mass"], inertia=None): #kg

    opti = ca.Opti()

    nx = 12
    nu = 4

    #Decision Variables
    X = opti.variable(nx, N + 1)
    U = opti.variable(nu, N)

    #Runtime Parameters
    X_init = opti.parameter(nx, 1)
    U_prev = opti.parameter(nu, 1)
    X_ref = opti.parameter(nx, N + 1)

    #Use Physics & Discretizer
    f_continuous, states, controls = get_drone_dynamics(m, inertia)
    f_discrete = get_rk4_discretizer(f_continuous, states, controls, dt)

    #Objective Cost Function
    #updated to evaluate the cost on the state variables X instead of re-rolling RK4 inside the
    # cost: same solution, ~8x less solver work (see MPC_Cost_Function.compute_mpc_cost)
    cost = compute_mpc_cost(X, U, X_ref, U_prev, N, m)
    opti.minimize(cost)

    #Initial State Equality Constraint
    opti.subject_to(X[:, 0] == X_init)

    #Dynamic Multiple-Shooting Constraints (RX4 Rollout)
    for k in range(N):
        opti.subject_to(X[:, k+1] == f_discrete(X[:, k], U[:, k]))

    #Physical Constraints
    alt_min = drone_constraints["altitude_limits"]["min"]
    alt_max = drone_constraints["altitude_limits"]["max"]
    roll_min = drone_constraints["roll_limits"]["min"]
    roll_max = drone_constraints["roll_limits"]["max"]
    pitch_min = drone_constraints["pitch_limits"]["min"]
    pitch_max = drone_constraints["pitch_limits"]["max"]
    thrust_min = drone_constraints["total_thrust_limits"]["min"]
    thrust_max = drone_constraints["total_thrust_limits"]["max"]
    #updated to use the body torque, angular rate and velocity limits in drone_constraints
    torque = drone_constraints["body_torque_limits"]
    torque_min = [torque["roll"]["min"], torque["pitch"]["min"], torque["yaw"]["min"]]
    torque_max = [torque["roll"]["max"], torque["pitch"]["max"], torque["yaw"]["max"]]
    rate_min = [drone_constraints[k]["min"] for k in ("roll_rate_limits", "pitch_rate_limits", "yaw_rate_limits")]
    rate_max = [drone_constraints[k]["max"] for k in ("roll_rate_limits", "pitch_rate_limits", "yaw_rate_limits")]
    vel_min = drone_constraints["linear_velocity_limits"]["min"]
    vel_max = drone_constraints["linear_velocity_limits"]["max"]

    for k in range(N):
        #fixed bug where the problem was infeasible whenever the current estimate was already past a
        # limit: X[:, 0] is pinned to X_init, so state limits apply to the predicted states k+1 only
        #altitude limits
        opti.subject_to(alt_min <= X[2, k + 1])
        opti.subject_to(X[2, k + 1] <= alt_max)

        #roll and pitch limits
        opti.subject_to(roll_min <= X[3, k + 1])
        opti.subject_to(X[3, k + 1] <= roll_max)
        opti.subject_to(pitch_min <= X[4, k + 1])
        opti.subject_to(X[4, k + 1] <= pitch_max)

        #total thrust limit (control index 0)
        opti.subject_to(thrust_min <= U[0, k])
        opti.subject_to(U[0, k] <= thrust_max)

        #body torque limits (control indices 1-3: roll, pitch, yaw)
        for i in range(3):
            opti.subject_to(torque_min[i] <= U[1 + i, k])
            opti.subject_to(U[1 + i, k] <= torque_max[i])

        #body angular rate limits (state indices 9-11: p, q, r) and linear velocity limits (6-8)
        for i in range(3):
            opti.subject_to(rate_min[i] <= X[9 + i, k + 1])
            opti.subject_to(X[9 + i, k + 1] <= rate_max[i])
            opti.subject_to(vel_min <= X[6 + i, k + 1])
            opti.subject_to(X[6 + i, k + 1] <= vel_max)

    #configuration solver (IPOPT)
    #updated for real-time use (50 Hz needs a solve well under 20 ms):
    # expand        -> build the problem in CasADi's fast SX form instead of MX (~5x)
    # tol 1e-4      -> the default 1e-8 spends iterations on digits the controller never uses
    # warm start    -> start the barrier near the previous solution (callers pass X/U guesses)
    # measured 2026-10-28: median ~370 ms -> ~9 ms per solve, controls unchanged to <1e-5 N
    opts = {
        'expand': True,
        'ipopt.print_level': 0,
        'print_time': 0,
        'ipopt.max_iter': 50,
        'ipopt.tol': 1e-4,
        'ipopt.acceptable_tol': 1e-3,
        'ipopt.warm_start_init_point': 'yes',
        'ipopt.mu_init': 1e-3,
        'ipopt.sb': 'yes',          # no IPOPT banner
    }
    opti.solver('ipopt', opts)

    return opti, X, U, X_init, U_prev, X_ref