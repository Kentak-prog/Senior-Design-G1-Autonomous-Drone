#Antek Singer 
#09/19/2026 15:54
'''
This script implements the Kalman filter of a Modern Predictive Controller for an autonomous drone. The filter is a 12D enhanched
Kalman filter (ekf) which will help in predicting the drones non-linear movement.
'''
import numpy as np
import casadi as ca
from New_Drone_State_Equations import get_drone_dynamics
import matplotlib.pyplot as plt
from scipy.linalg import expm

class DroneCasadiEKF12D:
    def __init__(self, dt=0.02):
        self.dt = dt
        
        #Load CasADI Model 
        f_continuous, states, controls = get_drone_dynamics()
        
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
        z: 12x1 numpy array containing your current sensor measurements
        """
        H = np.eye(12) 
        y = z - (H @ self.x) #Current Innovation (actual - expected)
        S = H @ self.P @ H.T + self.R #Innovation Covariance (uncertainty + measurement)
        K = self.P @ H.T @ np.linalg.inv(S) #Given uncertainty in P or S, indicates how much to shift prediction towards measurement
        
        self.x = self.x + (K @ y) #update state estimate
        self.P = (np.eye(12) - K @ H) @ self.P #update uncertainty
        
        return self.x

#Initial Object Class at dt=0.02
ekf = DroneCasadiEKF12D()

#Simulation Parameters
total_seconds = 5.0
timesteps = np.arange(0, total_seconds, ekf.dt)
num_steps=len(timesteps)

#Intialize Historical Arrays
state_history=np.zeros((num_steps,12))
sigma_history = np.zeros((num_steps, 12))
measurement_history = np.zeros((num_steps, 12))
#Initialize ground truth
x_true = np.zeros((12, 1)) 
for i, t in enumerate(timesteps):
    #Example Thrusts (U1,U2,U3,U4)
    u_numeric=np.array([0.0,0.0,0.0,0.0])
    #update the true drone physics
    k1 = ekf.f_eval(x_true, u_numeric)
    k2 = ekf.f_eval(x_true + ekf.dt/2 * k1, u_numeric)
    k3 = ekf.f_eval(x_true + ekf.dt/2 * k2, u_numeric)
    k4 = ekf.f_eval(x_true + ekf.dt * k3, u_numeric)
    x_true = x_true + (ekf.dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    x_true = np.array(x_true)
    
    ekf.predict(u_numeric)
    z_noise = x_true + np.random.normal(0, 0.1, (12, 1))
    ekf.update(z_noise)

    state_history[i,:]=ekf.x.flatten()
    sigma_history[i, :] = np.sqrt(np.diag(ekf.P))
    measurement_history[i, :] = z_noise.flatten()

#Labels
state_labels = [
    "X Position (m)", "Y Position (m)", "Z Position (m)",
    "X Velocity (m/s)", "Y Velocity (m/s)", "Z Velocity (m/s)",
    "Roll (rad)", "Pitch (rad)", "Yaw (rad)",
    "Roll Rate (rad/s)", "Pitch Rate (rad/s)", "Yaw Rate (rad/s)"
]

#Break down into seperate operations
group_ranges = [
    (0, 3, "Translational Position Tracking"),
    (3, 6, "Translational Velocity Tracking"),
    (6, 9, "Attitude / Orientation Angles"),
    (9, 12, "Body Angular Rates")
]

# Create a diagnostics figure for tracking errors
for start_idx, end_idx, title in group_ranges:
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for row, state_idx in enumerate(range(start_idx, end_idx)):
        
        # Calculate the actual error (Estimate - Ground Truth)
        error = state_history[:, state_idx] - measurement_history[:, state_idx] # or use x_true historical tracking if available
        
        # Pull the 3-sigma bounds centered at zero
        three_sigma = 3 * sigma_history[:, state_idx]
        
        # Plot Error vs Bounds
        axs[row].plot(timesteps, error, color='red', label='Estimation Error')
        axs[row].plot(timesteps, three_sigma, color='blue', linestyle='--', label=r'$+3\sigma$ Bound')
        axs[row].plot(timesteps, -three_sigma, color='blue', linestyle='--', label=r'$-3\sigma$ Bound')
        
        axs[row].set_ylabel(f"{state_labels[state_idx]} Error")
        axs[row].grid(True, linestyle=':', alpha=0.6)
        
        if row == 0:
            axs[row].legend(loc='upper right')
            
    axs[-1].set_xlabel("Time (seconds)")
    fig.suptitle(f"Error Residuals: {title}", fontsize=14, fontweight='bold')
    plt.tight_layout()

plt.show()
