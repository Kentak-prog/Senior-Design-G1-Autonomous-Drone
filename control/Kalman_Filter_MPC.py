#Antek Singer 
#09/19/2026 15:54
'''
This script implements the state estimator for the drone's Model Predictive
Controller (MPC). It is a 12-state Extended Kalman Filter (EKF): it predicts the
drone's motion with the nonlinear model from New_Drone_State_Equations.py and
corrects that prediction with sensor measurements every step.

State order: [x, y, z, roll, pitch, yaw, vx, vy, vz, p, q, r]
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
        #P, Q and R below are covariance matrices: each diagonal entry is a variance
        #(sigma^2, in the state's units squared). Lower = more certain, higher = less certain.
        self.x = np.zeros((12, 1))  # Current state estimate
        self.P = np.eye(12) * 0.1   # Initial state uncertainty (updated every step)
        # --- TUNING PROCESS NOISE (Q) ---
        # How much the physics model is expected to be wrong over one step.
        # Lower values = higher trust in the physics model
        q_diag = np.array([
            0.001, 0.001, 0.05,   # X, Y, Z positions (Increased Z to keep bounds open)
            0.01,  0.01,  0.01,   # Roll, Pitch, Yaw angles
            0.001, 0.001, 0.05,   # X, Y, Z velocities
            0.01,  0.01,  0.01    # Roll, Pitch, Yaw angular rates
        ])
        self.Q = np.diag(q_diag)

        # --- TUNING MEASUREMENT NOISE (R) ---
        # How noisy each sensor measurement is expected to be.
        # Higher values = trust the sensor less, so the EKF smooths out jerky sensor noise
        r_diag = np.array([
            0.5,   0.5,   0.1,    # X, Y, Z positions (Increased X/Y to smooth tracking)
            0.1,   0.1,   0.1,    # Roll, Pitch, Yaw
            0.8,   0.8,   0.1,    # X, Y, Z velocities
            0.2,   0.2,   0.2     # Roll, Pitch, Yaw angular rates
        ])
        self.R = np.diag(r_diag)
        
    def predict(self, u_numeric):
        """
        u_numeric: A 4x1 numpy array or list [U1, U2, U3, U4] of control inputs:
                   U1 = total thrust (N), U2..U4 = roll, pitch, yaw body torques (N*m)
        """
        dt = self.dt
        
        #Nonlinear Physics Prediction (Runge-Kutta 4th Order Discretization)
        #RK4 integrates the model more accurately than a single Euler step. It does not
        #remove drift on its own -- the update step corrects the estimate with measurements.
        k1 = self.f_eval(self.x, u_numeric)
        k2 = self.f_eval(self.x + dt/2 * k1, u_numeric)
        k3 = self.f_eval(self.x + dt/2 * k2, u_numeric)
        k4 = self.f_eval(self.x + dt * k3, u_numeric)
        
        #State vector update (cast CasADi output back to a standard NumPy array)
        x_next = self.x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        self.x = np.array(x_next) 

        #Compute Discrete Jacobian Matrix (F)
        #Evaluate the continuous Jacobian with the current control input. Note: self.x was
        #already overwritten above, so this is evaluated at the PREDICTED state (a textbook
        #EKF evaluates it at the previous estimate; the difference is small at dt = 0.02 s).
        Ac = np.array(self.J_eval(self.x, u_numeric))
        
        #Convert continuous-time Jacobian (Ac) to discrete-time Jacobian (F)
        F = expm(Ac * dt) #matrix exponential: exact discretization of the linearized model
                          #(more accurate and better behaved than the first-order I + Ac*dt)
        #Project Covariance Forward: uncertainty grows by the model's spread plus process noise
        self.P = F @ self.P @ F.T + self.Q
        
        return self.x

    def update(self, z):
        """
        z: 12x1 numpy array containing current sensor measurements
        """
        H = np.eye(12) #Measurement model: every one of the 12 states is measured directly
        y = z - (H @ self.x) #Current Innovation (actual - expected)
        S = H @ self.P @ H.T + self.R #Innovation Covariance (prediction uncertainty + measurement noise)
        K = self.P @ H.T @ np.linalg.inv(S) #Kalman gain: how far to move the prediction toward the measurement, based on P vs R
        
        self.x = self.x + (K @ y) #update state estimate
        self.P = (np.eye(12) - K @ H) @ self.P #update uncertainty (shrinks after a measurement)
        
        return self.x
