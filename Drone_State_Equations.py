# Antek Singer
# 09/08/26 12:18
# Autonomous Drone State Equations
"""
This code will create the matrices for 12 drone state equations. Then output a statespace matrix centered around a drone hovering (its equilibrium).
X = [x, y, z, phi(roll), theta(pitch), psi(yaw), x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot]
"""
import sympy as sp
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Define States Symbolically 
x, y, z, phi, theta, psi = sp.symbols('x y z phi theta psi')
x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot = sp.symbols('x_dot y_dot z_dot phi_dot theta_dot psi_dot')

states = [x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot]

U1, U2, U3, U4 = sp.symbols('U1 U2 U3 U4')
inputs = [U1, U2, U3, U4] # U1 is total torque, U2, U3, U4, are directional torques (x,y,z)

#Define Physical Constants
g, m, Ix, Iy, Iz = sp.symbols('g m Ix Iy Iz')
constants = {g: 9.81, m: 1.0, Ix: 0.01, Iy: 0.01, Iz: 0.02}

#Define Body Roll Rates p,q,r
p = phi_dot - (sp.sin(theta))*psi_dot
q = (sp.cos(phi))*theta_dot + (sp.cos(theta)*sp.sin(phi))*psi_dot
r = (sp.cos(theta)*sp.cos(phi))*psi_dot - sp.sin(phi)*theta_dot

#Non-Linear State Equations
eqs=[
    x_dot,
    y_dot,
    z_dot,

    #Angular Velocities Based on Body Roll Rates
    p + (r*sp.cos(phi)*sp.sin(theta))/sp.cos(theta) + (q*sp.sin(theta)*sp.sin(phi))/sp.cos(theta), # dphi/dt
    q*sp.cos(phi) - r*sp.sin(phi), # dtheta/dt
    (r*sp.cos(phi))/sp.cos(theta) + (q*sp.sin(phi))/sp.cos(theta), #dpsi/dt

    #Translational Equations of Motion
    -(U1/m)*(sp.cos(phi)*sp.sin(theta)*sp.cos(psi) + sp.sin(phi)*sp.sin(psi)), #d_dx/dt
    -(U1/m)*(sp.cos(phi)*sp.sin(theta)*sp.sin(psi) - sp.sin(phi)*sp.cos(psi)), #d_dy/dt
    g - U1*(sp.cos(phi)*sp.cos(theta)), #d_dz/dt

    #Rotational Equations of Motion
    (U2 + Iy*q*r - Iz*q*r)/Ix, #d_dphi/dt
    (U3 - Ix*p*r + Ix*p*r)/Iy, #d_dtheta/dt
    (U4 + Ix*p*q - Iy*p*q)/Iz #d_dpsi/dt
]

# Calculate Symbolic Jacobians (Linearization step)
A_sym = sp.Matrix(eqs).jacobian(states)
B_sym = sp.Matrix(eqs).jacobian(inputs)

# Equilibrium Point (Hover Position)
# At hover: all positions/angles/velocities are 0, and Thrust (U1) = m * g
hover_conditions = {
    x: 0, y: 0, z: 0, phi: 0, theta: 0, psi: 0,
    x_dot: 0, y_dot: 0, z_dot: 0, phi_dot: 0, theta_dot: 0, psi_dot: 0,
    U1: m * g,  # Thrust balances gravity
    U2: 0, U3: 0, U4: 0
}

# Substitute hover conditions and constants into jacobians
A_hover = A_sym.subs(hover_conditions).subs(constants)
B_hover = B_sym.subs(hover_conditions).subs(constants)

A_num = np.array(A_hover).astype(np.float64)
B_num = np.array(B_hover).astype(np.float64)

#Create SciPy StateSpace system
sys = signal.StateSpace(A_num, B_num, np.eye(12), np.zeros((12, 4)))
print("Linearized A matrix at hover:\n", A_num)
