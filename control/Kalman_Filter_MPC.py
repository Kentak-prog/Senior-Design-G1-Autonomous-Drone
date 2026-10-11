#Antek Singer 
#09/19/2026 15:54
'''
This script implements the Kalman filter of a Modern Predictive Controller for an autonomous drone. The filter is a 12D enhanched
Kalman filter (ekf) which will help in predicting the drones non-linear movement.
'''
import numpy as np
import casadi as ca
from New_Drone_State_Equations import get_drone_dynamics
from drone_constraints import drone_constraints
from scipy.linalg import expm

class DroneCasadiEKF12D:
    def __init__(self, dt=0.02, m = drone_constraints["mass"]): #kg
        self.dt = dt
        
        #Load CasADI Model 
        f_continuous, states, controls = get_drone_dynamics(m)
        
        #Calculate Jacobian Model
        # This takes the derivative of f_continuous with respect to the 12 states
        J_continuous = ca.jacobian(f_continuous(states, controls), states)
        
        #Compile CasADi evaluation functions
        #Turn symbolic function into normal function
        self.f_eval = f_continuous
        self.J_eval = ca.Function('J_e', [states, controls], [J_continuous])
        
        #Initialize State and Covariance arrays
        #Tune weights if needed, these are initial values. Lower = certain Higher = uncertain
        self.x = np.zeros((12, 1))  # Current state estimate
        self.P = np.eye(12) * 0.1   # State uncertainty
        # --- TUNING PROCESS NOISE (Q) ---
        # Lower values mean = higher trust in physics model
        q_diag = np.array([
            0.001, 0.001, 0.05,   # X, Y, Z positions (Increased Z to keep bounds open)
            0.01,  0.01,  0.01,   # X, Y, Z velocities
            0.001, 0.001, 0.05,   # Roll, Pitch, Yaw angles (Increased Yaw)
            0.01,  0.01,  0.01    # Roll, Pitch, Yaw angular rates
        ])
        self.Q = np.diag(q_diag)

        # --- TUNING MEASUREMENT NOISE (R) ---
        # Higher values force the EKF to smooth out jerky sensor noise
        r_diag = np.array([
            0.5,   0.5,   0.1,    # X, Y, Z positions (Increased X/Y to smooth tracking)
            0.1,   0.1,   0.1,    # X, Y, Z velocities
            0.8,   0.8,   0.1,    # Roll, Pitch, Yaw angles (Increased Roll/Pitch)
            0.2,   0.2,   0.2     # Roll, Pitch, Yaw angular rates
        ])
        self.R = np.diag(r_diag)
        
    def predict(self, u_numeric):
        """
        u_numeric: A 4x1 numpy array or list [U1, U2, U3, U4] of Thrust Inputs
        """
        dt = self.dt
        
        #Nonlinear Physics Prediction (Runge-Kutta 4th Order Discretization)
        #Elimates accumulated error or drift
        k1 = self.f_eval(self.x, u_numeric)
        k2 = self.f_eval(self.x + dt/2 * k1, u_numeric)
        k3 = self.f_eval(self.x + dt/2 * k2, u_numeric)
        k4 = self.f_eval(self.x + dt * k3, u_numeric)
        
        #State vector update (cast CasADi output back to a standard NumPy array)
        x_next = self.x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        self.x = np.array(x_next) 

        #Compute Discrete Jacobian Matrix (F)
        #Evaluate continuous Jacobian at current control/input
        Ac = np.array(self.J_eval(self.x, u_numeric))
        
        #Convert continuous-time Jacobian (Ac) to discrete-time Jacobian (F)
        F = expm(Ac * dt) #matrix exponential conversion to prevent stability blowup
        #Project Covariance Forward
        self.P = F @ self.P @ F.T + self.Q
        
        return self.x

    def update(self, z):
        """
        z: 12x1 numpy array containing current sensor measurements
        """
        H = np.eye(12) 
        y = z - (H @ self.x) #Current Innovation (actual - expected)
        S = H @ self.P @ H.T + self.R #Innovation Covariance (uncertainty + measurement)
        K = self.P @ H.T @ np.linalg.inv(S) #Given uncertainty in P or S, indicates how much to shift prediction towards measurement
        
        self.x = self.x + (K @ y) #update state estimate
        self.P = (np.eye(12) - K @ H) @ self.P #update uncertainty
        
        return self.x
