import numpy as np
import casadi as ca

from New_Drone_State_Equations import get_drone_dynamics
from Runge_Kutta_4_Discretizer import get_rk4_discretizer
from drone_constraints import drone_constraints
from MPC_Cost_Function import compute_mpc_cost_with_rk4

def create_drone_mpc_optimizer(N=10, dt=0.02):

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
    f_continuous, states, controls = get_drone_dynamics()
    f_discrete = get_rk4_discretizer(f_continuous, states, controls, dt)

    #Objective Cost Function
    cost = compute_mpc_cost_with_rk4(X_init, U, X_ref, U_prev, f_discrete, N)
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

    for k in range(N):
        #altitude limits
        opti.subject_to(alt_min <= X[2, k])
        opti.subject_to(X[2, k] <= alt_max)

        #roll and pitch limits
        opti.subject_to(roll_min <= X[3, k])
        opti.subject_to(X[3, k] <= roll_max)
        opti.subject_to(pitch_min <= X[4, k])
        opti.subject_to(X[4, k] <= pitch_max)

        #total thrust limit (control index 0)
        opti.subject_to(thrust_min <= U[0, k])
        opti.subject_to(U[0, k] <= thrust_max)

    #configuration solver (IPOPT)
    opts = {
        'ipopt.print_level': 0,
        'print_time': 0,
        'ipopt.max_iter': 50,
        'ipopt.acceptable_tol': 1e-4
    }
    opti.solver('ipopt', opts)

    return opti, X, U, X_init, U_prev, X_ref