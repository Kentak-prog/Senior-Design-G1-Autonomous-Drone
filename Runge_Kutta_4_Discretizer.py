import casadi as ca
import numpy as np

#Runge-Kutta 4 Discretizer
# Converts inputs from continuous time to discrete time for cost function.
def get_rk4_discretizer(f_continuous, states, controls, dt=0.05):
    """Discretizes the continuous drone equations using a 4th-order Runge-Kutta step."""
    X0 = ca.MX.sym('X0', states.size1())
    U = ca.MX.sym('U', controls.size1())
    
    k1 = f_continuous(X0, U)
    k2 = f_continuous(X0 + dt/2 * k1, U)
    k3 = f_continuous(X0 + dt/2 * k2, U)
    k4 = f_continuous(X0 + dt * k3, U)
    
    X_next = X0 + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    
    return ca.Function('f_discrete', [X0, U], [X_next])