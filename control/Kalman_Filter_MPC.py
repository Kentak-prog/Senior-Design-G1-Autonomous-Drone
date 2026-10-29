#Antek Singer
#09/19/2026 15:54
'''
This script implements the state estimator for the drone's Model Predictive
Controller (MPC). It is a 12-state Extended Kalman Filter (EKF).

State order: [x, y, z, roll, pitch, yaw, vx, vy, vz, p, q, r]
World frame Z-up (ENU), body frame FLU, Euler angles R = Rz(yaw) Ry(pitch) Rx(roll),
the same conventions as New_Drone_State_Equations.py.

Two ways to PREDICT (pick one per step):
  predict(u)                 the nonlinear thrust/torque model from New_Drone_State_Equations.py,
                             driven by the MPC's commanded inputs (what the Python sims use)
  predict_imu(accel, gyro)   strapdown: the IMU's gyro turns the attitude, its accelerometer
                             pushes the velocity. Needs no model of thrust, mass or drag.

and several ways to UPDATE (correct with a measurement):
  update(z)                  every one of the 12 states measured directly (the Python sims)
  update_gyro(gyro)          body rates from the IMU gyro
  update_pose(pos, rpy)      position + attitude, e.g. from vision (gate PnP against a known map)
  update_position(pos)       position only
  update_velocity(vel)       world-frame velocity only
  update_measurement(z, idx, R)   any subset of states (the general form the others use)

A vision + IMU loop: every IMU sample  predict_imu(a, w); update_gyro(w)
                     every vision fix  update_pose(p, rpy)
'''
import numpy as np
import casadi as ca
from New_Drone_State_Equations import get_drone_dynamics
from drone_constraints import drone_constraints
from scipy.linalg import expm

G = 9.81
ANGLE_IDX = (3, 4, 5)   # roll, pitch, yaw: innovations on these are wrapped to [-pi, pi)


def wrap_angle(a):
    return (np.asarray(a, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def _imu_dynamics():
    """
    Strapdown model f_imu(x, imu) for imu = [ax, ay, az, gx, gy, gz]:
    accelerometer specific force (body FLU, m/s^2; reads about +9.81 on z when
    hovering level, the REP-145 / sensor_msgs/Imu convention) and gyro body
    rates (rad/s).

        position'  = velocity
        attitude'  = Euler kinematics of the GYRO rates (same equations as the control model)
        velocity'  = R(attitude) * accel - g * e_z
        rates'     = 0          (the gyro update sets them; see update_gyro)
    """
    x = ca.SX.sym('x', 12)
    imu = ca.SX.sym('imu', 6)
    phi, theta, psi = x[3], x[4], x[5]
    a = imu[0:3]
    p, q, r = imu[3], imu[4], imu[5]

    dphi = p + (r * ca.cos(phi) * ca.sin(theta)) / ca.cos(theta) + (q * ca.sin(theta) * ca.sin(phi)) / ca.cos(theta)
    dtheta = q * ca.cos(phi) - r * ca.sin(phi)
    dpsi = (r * ca.cos(phi)) / ca.cos(theta) + (q * ca.sin(phi)) / ca.cos(theta)

    cf, sf, ct, st, cp, sp = ca.cos(phi), ca.sin(phi), ca.cos(theta), ca.sin(theta), ca.cos(psi), ca.sin(psi)
    R = ca.vertcat(
        ca.horzcat(cp * ct, cp * st * sf - sp * cf, cp * st * cf + sp * sf),
        ca.horzcat(sp * ct, sp * st * sf + cp * cf, sp * st * cf - cp * sf),
        ca.horzcat(-st, ct * sf, ct * cf))
    acc_world = ca.mtimes(R, a) - ca.vertcat(0, 0, G)

    rhs = ca.vertcat(x[6:9], dphi, dtheta, dpsi, acc_world, ca.SX.zeros(3))
    f = ca.Function('f_imu', [x, imu], [rhs])
    J = ca.Function('J_imu', [x, imu], [ca.jacobian(rhs, x)])
    return f, J


class DroneCasadiEKF12D:
    def __init__(self, dt=0.02, m = drone_constraints["mass"], #kg
                 accel_noise=0.1, gyro_noise=0.01):
        """
        dt:           step of predict() (model mode); predict_imu() takes its own dt
        accel_noise:  accelerometer white noise, 1-sigma per sample (m/s^2)
        gyro_noise:   gyro white noise, 1-sigma per sample (rad/s)
        """
        self.dt = dt
        self.accel_noise = accel_noise
        self.gyro_noise = gyro_noise

        #Load CasADI Model
        f_continuous, states, controls = get_drone_dynamics(m)

        #Calculate Jacobian Model
        # This takes the derivative of f_continuous with respect to the 12 states
        J_continuous = ca.jacobian(f_continuous(states, controls), states)

        #Compile CasADi evaluation functions
        #Turn symbolic function into normal function
        self.f_eval = f_continuous
        self.J_eval = ca.Function('J_e', [states, controls], [J_continuous])
        self.f_imu, self.J_imu = _imu_dynamics()

        #Initialize State and Covariance arrays
        #P, Q and R below are covariance matrices: each diagonal entry is a variance
        #(sigma^2, in the state's units squared). Lower = more certain, higher = less certain.
        self.x = np.zeros((12, 1))  # Current state estimate
        self.P = np.eye(12) * 0.1   # Initial state uncertainty (updated every step)
        # --- TUNING PROCESS NOISE (Q) ---
        # How much the physics model is expected to be wrong over one step.
        # Lower values = higher trust in the physics model
        #fixed bug where the values sat on the wrong states: the comments were re-labelled to the
        # state order [pos, angles, vel, rates] but the numbers were not moved with them, so the
        # velocities got the angle values and the angles the velocity values
        q_diag = np.array([
            0.001, 0.001, 0.05,   # X, Y, Z positions (Increased Z to keep bounds open)
            0.001, 0.001, 0.05,   # Roll, Pitch, Yaw angles (Increased Yaw)
            0.01,  0.01,  0.01,   # X, Y, Z velocities
            0.01,  0.01,  0.01    # Roll, Pitch, Yaw angular rates
        ])
        self.Q = np.diag(q_diag)

        # --- TUNING MEASUREMENT NOISE (R) ---
        # How noisy each sensor measurement is expected to be (used by update(z)).
        # Higher values = trust the sensor less, so the EKF smooths out jerky sensor noise
        r_diag = np.array([
            0.5,   0.5,   0.1,    # X, Y, Z positions (Increased X/Y to smooth tracking)
            0.8,   0.8,   0.1,    # Roll, Pitch, Yaw angles (Increased Roll/Pitch)
            0.1,   0.1,   0.1,    # X, Y, Z velocities
            0.2,   0.2,   0.2     # Roll, Pitch, Yaw angular rates
        ])
        self.R = np.diag(r_diag)

    # ------------------------------------------------------------------ predict
    def predict(self, u_numeric):
        """
        Model-based prediction over self.dt.
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

        x_next = self.x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
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

    def predict_imu(self, accel_body, gyro_body, dt=None):
        """
        Strapdown prediction from one IMU sample, held constant over dt.
          accel_body: specific force (m/s^2), body FLU -- about (0, 0, 9.81) hovering level
          gyro_body:  body rates (rad/s), body FLU
          dt:         time since the previous IMU sample (s); defaults to self.dt
        Process noise comes from accel_noise / gyro_noise, so no thrust, mass or drag
        model is involved.
        """
        dt = self.dt if dt is None else float(dt)
        imu = np.concatenate([np.asarray(accel_body, dtype=float).ravel(),
                              np.asarray(gyro_body, dtype=float).ravel()])
        x0 = self.x
        k1 = np.array(self.f_imu(x0, imu))
        k2 = np.array(self.f_imu(x0 + dt / 2 * k1, imu))
        k3 = np.array(self.f_imu(x0 + dt / 2 * k2, imu))
        k4 = np.array(self.f_imu(x0 + dt * k3, imu))
        A = np.array(self.J_imu(x0, imu))          # at the previous estimate (textbook EKF)
        self.x = x0 + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        F = expm(A * dt)

        # Noise entering through the IMU over this step: gyro noise turns the attitude,
        # accelerometer noise pushes velocity (and, integrated, position).
        sa2, sg2 = self.accel_noise ** 2, self.gyro_noise ** 2
        q = np.concatenate([np.full(3, sa2 * dt ** 4 / 4),   # position
                            np.full(3, sg2 * dt ** 2),       # angles
                            np.full(3, sa2 * dt ** 2),       # velocity
                            np.full(3, sg2)])                # rates (replaced by update_gyro)
        self.P = F @ self.P @ F.T + np.diag(q) + 1e-9 * np.eye(12)
        return self.x

    # ------------------------------------------------------------------ update
    def update(self, z):
        """
        z: 12x1 numpy array containing current sensor measurements of every state,
        weighted by self.R (what the Python sims provide)
        """
        return self.update_measurement(z, range(12), self.R)

    def update_measurement(self, z, idx, R_meas):
        """
        Correct the estimate with a measurement of the states listed in idx.
          z:      measured values, same order as idx
          idx:    state indices measured, e.g. (0, 1, 2) for position
          R_meas: measurement covariance (len(idx) x len(idx)), or a vector of variances
        Angle innovations are wrapped, so a yaw of +179 deg measured as -179 deg is a
        2 deg correction, not 358.
        """
        idx = list(idx)
        z = np.asarray(z, dtype=float).reshape(len(idx), 1)
        R_meas = np.asarray(R_meas, dtype=float)
        if R_meas.ndim < 2:
            R_meas = np.diag(np.broadcast_to(R_meas, (len(idx),)))

        H = np.zeros((len(idx), 12)) #Measurement model: pick the measured states
        H[np.arange(len(idx)), idx] = 1.0
        y = z - H @ self.x #Current Innovation (actual - expected)
        for row, i in enumerate(idx):
            if i in ANGLE_IDX:
                y[row, 0] = wrap_angle(y[row, 0])
        S = H @ self.P @ H.T + R_meas #Innovation Covariance (prediction uncertainty + measurement noise)
        K = self.P @ H.T @ np.linalg.inv(S) #Kalman gain: how far to move the prediction toward the measurement, based on P vs R

        self.x = self.x + (K @ y) #update state estimate
        #update uncertainty (shrinks after a measurement). Joseph form: stays symmetric and
        #positive definite even when only a few states are measured
        I_KH = np.eye(12) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_meas @ K.T

        return self.x

    def update_gyro(self, gyro_body, std=None):
        """Body rates p, q, r from the IMU gyro (rad/s)."""
        s = self.gyro_noise if std is None else std
        return self.update_measurement(gyro_body, (9, 10, 11), np.full(3, s ** 2))

    def update_pose(self, pos, rpy, pos_std=0.05, ang_std=np.radians(1.0)):
        """Position (m, world ENU) + roll/pitch/yaw (rad), e.g. a vision fix from a known gate."""
        z = np.concatenate([np.asarray(pos, dtype=float).ravel(), np.asarray(rpy, dtype=float).ravel()])
        std = np.concatenate([np.broadcast_to(pos_std, (3,)), np.broadcast_to(ang_std, (3,))])
        return self.update_measurement(z, range(6), std ** 2)

    def update_position(self, pos, std=0.05):
        """Position only (m, world ENU)."""
        return self.update_measurement(pos, (0, 1, 2), np.broadcast_to(np.asarray(std, dtype=float) ** 2, (3,)))

    def update_velocity(self, vel, std=0.1):
        """World-frame velocity only (m/s)."""
        return self.update_measurement(vel, (6, 7, 8), np.broadcast_to(np.asarray(std, dtype=float) ** 2, (3,)))
