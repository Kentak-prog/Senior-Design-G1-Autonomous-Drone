#Antek Singer 09/18/2026 21:29
#State equations using casADI
import casadi as ca
import numpy as np

def get_drone_dynamics():
    #1. Define Symbolic State
    # Pos: x, y, z 
    # Att: phi (roll), theta (pitch), psi (yaw)
    x = ca.MX.sym('x')
    y = ca.MX.sym('y')
    z = ca.MX.sym('z')
    phi = ca.MX.sym('phi')
    theta = ca.MX.sym('theta')
    psi = ca.MX.sym('psi')
    
    # Lin Vel: x_dot, y_dot, z_dot
    # Body Ang Vel: p, q, r
    x_dot = ca.MX.sym('x_dot')
    y_dot = ca.MX.sym('y_dot')
    z_dot = ca.MX.sym('z_dot')
    p = ca.MX.sym('p')
    q = ca.MX.sym('q')
    r = ca.MX.sym('r')
    
    states = ca.vertcat(x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p, q, r)
    
    #2. Define Symbolic Controls
    # U1: Total Thrust
    # U2, U3, U4: Roll, Pitch, Yaw Torques
    U1 = ca.MX.sym('U1')
    U2 = ca.MX.sym('U2')
    U3 = ca.MX.sym('U3')
    U4 = ca.MX.sym('U4')
    controls = ca.vertcat(U1, U2, U3, U4)
    
    #3. Constant Parameters
    g = 9.81           # m/s^2
    m = 1.0            # kg (Adjust based on your drone)
    Ix = 0.0081        # kg*m^2
    Iy = 0.0081        # kg*m^2
    Iz = 0.0142        # kg*m^2

    #4. Non-linear Differential Equations (f(x, u))
    #Attitude Kinematics (Body rates p,q,r to Euler rate changes)
    dphi_dt   = p + (r * ca.cos(phi) * ca.sin(theta)) / ca.cos(theta) + (q * ca.sin(theta) * ca.sin(phi)) / ca.cos(theta)
    dtheta_dt = q * ca.cos(phi) - r * ca.sin(phi)
    dpsi_dt   = (r * ca.cos(phi)) / ca.cos(theta) + (q * ca.sin(phi)) / ca.cos(theta)
    
    # Translational Accelerations
    dx_ddot = -(U1 / m) * (ca.cos(phi) * ca.sin(theta) * ca.cos(psi) + ca.sin(phi) * ca.sin(psi))
    dy_ddot = -(U1 / m) * (ca.cos(phi) * ca.sin(theta) * ca.sin(psi) - ca.sin(phi) * ca.cos(psi))
    dz_ddot = g - (U1 / m) * (ca.cos(phi) * ca.cos(theta)) # Included normalized mass adjustment
    
    # Rotational Accelerations
    dp_dt = (U2 + (Iy - Iz) * q * r) / Ix
    dq_dt = (U3 + (Iz - Ix) * p * r) / Iy
    dr_dt = (U4 + (Ix - Iy) * p * q) / Iz
    
    # Assemble complete x_dot derivative array
    rhs = ca.vertcat(
        x_dot,      # dx/dt
        y_dot,      # dy/dt
        z_dot,      # dz/dt
        dphi_dt,    
        dtheta_dt,  
        dpsi_dt,    
        dx_ddot,    
        dy_ddot,    
        dz_ddot,    
        dp_dt,      
        dq_dt,      
        dr_dt       
    )
    
    # Continuous time dynamics function
    f_continuous = ca.Function('f_c', [states, controls], [rhs])
    return f_continuous, states, controls


