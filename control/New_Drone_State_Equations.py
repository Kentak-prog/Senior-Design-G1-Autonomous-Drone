#Antek Singer 09/18/2026 21:29
#State equations using casADI
import casadi as ca
import numpy as np
from drone_constraints import drone_constraints, inertia as INERTIA, linear_drag as LINEAR_DRAG

def get_drone_dynamics(m = drone_constraints["mass"], inertia=None, drag=None): #kg
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
    # m (kg) defaults to the mass in drone_constraints.py
    #updated inertia and drag to match drone_constraints (Pegasus/Isaac Sim Iris model)
    # inertia=None uses drone_constraints.inertia; pass (Ix, Iy, Iz) in kg*m^2 to model another airframe
    if inertia is None:
        inertia = (INERTIA["Ixx"], INERTIA["Iyy"], INERTIA["Izz"])
    Ix, Iy, Iz = inertia   # kg*m^2
    # drag=None uses drone_constraints.linear_drag; pass (dx, dy, dz) in N per m/s (body frame)
    if drag is None:
        drag = (LINEAR_DRAG["x"], LINEAR_DRAG["y"], LINEAR_DRAG["z"])
    dx, dy, dz = drag

    #4. Non-linear Differential Equations (f(x, u))
    #Attitude Kinematics (Body rates p,q,r to Euler rate changes)
    dphi_dt   = p + (r * ca.cos(phi) * ca.sin(theta)) / ca.cos(theta) + (q * ca.sin(theta) * ca.sin(phi)) / ca.cos(theta)
    dtheta_dt = q * ca.cos(phi) - r * ca.sin(phi)
    dpsi_dt   = (r * ca.cos(phi)) / ca.cos(theta) + (q * ca.sin(phi)) / ca.cos(theta)
    
    # Translational Accelerations
    #fixed bug where z axis were not consistent between programs
    # World frame is Z-UP (matches Isaac Sim, the vision stack and the altitude
    # limits in drone_constraints.py). Thrust U1 acts along body +Z, so the
    # acceleration is (U1/m) * (third column of R = Rz(psi) Ry(theta) Rx(phi)) - g * e_z.
    #
    # Linear drag (Pegasus LinearDrag): F_body = -diag(dx, dy, dz) * v_body, with
    # v_body = R^T v_world, rotated back to the world frame: a_drag = R F_body / m.
    R = ca.vertcat(
        ca.horzcat(ca.cos(psi) * ca.cos(theta),
                   ca.cos(psi) * ca.sin(theta) * ca.sin(phi) - ca.sin(psi) * ca.cos(phi),
                   ca.cos(psi) * ca.sin(theta) * ca.cos(phi) + ca.sin(psi) * ca.sin(phi)),
        ca.horzcat(ca.sin(psi) * ca.cos(theta),
                   ca.sin(psi) * ca.sin(theta) * ca.sin(phi) + ca.cos(psi) * ca.cos(phi),
                   ca.sin(psi) * ca.sin(theta) * ca.cos(phi) - ca.cos(psi) * ca.sin(phi)),
        ca.horzcat(-ca.sin(theta), ca.cos(theta) * ca.sin(phi), ca.cos(theta) * ca.cos(phi)))
    v_body = ca.mtimes(R.T, ca.vertcat(x_dot, y_dot, z_dot))
    a_drag = ca.mtimes(R, -ca.vertcat(dx * v_body[0], dy * v_body[1], dz * v_body[2])) / m

    dx_ddot = (U1 / m) * (ca.cos(phi) * ca.sin(theta) * ca.cos(psi) + ca.sin(phi) * ca.sin(psi)) + a_drag[0]
    dy_ddot = (U1 / m) * (ca.cos(phi) * ca.sin(theta) * ca.sin(psi) - ca.sin(phi) * ca.cos(psi)) + a_drag[1]
    dz_ddot = (U1 / m) * (ca.cos(phi) * ca.cos(theta)) - g + a_drag[2]
    
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


