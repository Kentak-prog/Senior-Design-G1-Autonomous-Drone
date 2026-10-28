#Aaron Arnold
#physical constraints of a drone
#
# Revised to match the Pegasus/Isaac Sim Iris model:
#   mass 1.5 kg, inertia (0.029125, 0.029125, 0.055225) kg*m^2   (read from iris.usd)
#   thrust T = 8.54858e-6 * omega^2, omega_max = 1100 rad/s       (quadratic_thrust_curve.py defaults)
#   yaw reaction torque = 1e-6 * omega^2 per motor
#   linear drag [0.50, 0.30, 0.0] N per m/s                        (iris.py)
#   rotor offsets (x, y) from USD: r1 (-0.207, -0.138), r2 (0.219, 0.125),
#                                  r3 (0.203, -0.138), r4 (-0.222, 0.125)

# Drone mass -- the single place to change it. Used by the state equations,
# the MPC cost/optimizer, the EKF, the race sim and the trajectory reference.
m = 1.5 #kg

# Principal moments of inertia (kg*m^2) -- use these in get_drone_dynamics and the sympy script
inertia = {
    "Ixx": 0.029125,
    "Iyy": 0.029125,
    "Izz": 0.055225,
}

# Linear drag applied by Pegasus (N per m/s, body x, y, z). The MPC dynamics should include it.
linear_drag = {
    "x": 0.50,
    "y": 0.30,
    "z": 0.0,
}

drone_constraints = {
    # 0. Mass
    "mass": m, #kg

    # 1. Attitude Angles (in radians)
    # Max tilt that can still hold altitude is arccos(1/2.81) = 1.21 rad (T/W = 2.81).
    # 1.0 rad leaves thrust margin: hold thrust at 1.0 rad is 14.7/cos(1.0) = 27 N of 41 N.
    "roll_limits": {
        "min": -1.0,
        "max": 1.0
    },
    "pitch_limits": {
        "min": -1.0,
        "max": 1.0
    },
    "yaw_limits": {
        "min": -3.14, # Full operational heading range (-180 deg)
        "max": 3.14  # Full operational heading range (+180 deg)
    },

    # 2. Actuator Constraints
    # Per-motor yaw REACTION torque: 1e-6 * 1100^2 = 1.21 N*m at full speed.
    # Note: this is not the body torque the MPC commands. See body_torque_limits below.
    "motor_torque_limits": {
        "min": 0.0,
        "max": 1.21
    },

    # Body torques (N*m) commanded by the MPC (U2..U4). Full-saturation values from rotor
    # geometry. Conservative values that hold near hover (a motor cannot go below zero thrust)
    # are roll +/-1.8, pitch +/-3.1, yaw +/-0.85.
    "body_torque_limits": {
        "roll":  {"min": -2.6, "max": 2.6},
        "pitch": {"min": -4.4, "max": 4.4},
        "yaw":   {"min": -2.4, "max": 2.4}
    },

    # 3. Actuator Rate Limits (Slew Rates / Delta U)
    # Pegasus' quadratic thrust curve has no motor lag, so this is a smoothing choice, not physics.
    # Unchanged from before; tune against the MPC's dt (0.02 s).
    "motor_torque_rate_limits": {
        "min": -50.0, # N*m/s
        "max": 50.0
    },

    # 4. Motor Speed (RPM) Limits
    # 1100 rad/s = 10,504 RPM. Hover is about 6,270 RPM.
    "motor_rpm_limits": {
        "min": 1500.0,
        "max": 10500.0
    },

    # 5. Thrust Limits (Newtons)
    # 8.54858e-6 * 1100^2 = 10.34 N per motor, 41.4 N total. Limit set slightly below saturation.
    "motor_thrust_limits": {
        "min": 0.0,
        "max": 10.3
    },
    "total_thrust_limits": {
        "min": 0.0,
        "max": 41.0
    },

    # 6. Angular Rate Limits (Gyro Limits, in rad/s)
    # Not capped by the sim; these are practical limits for gate flying.
    "roll_rate_limits": {
        "min": -10.0,
        "max": 10.0
    },
    "pitch_rate_limits": {
        "min": -10.0,
        "max": 10.0
    },
    "yaw_rate_limits": {
        "min": -4.0,
        "max": 4.0
    },

    # 7. Angular Acceleration Limits (rad/s^2) at full torque saturation:
    # roll 2.6/0.029125, pitch 4.4/0.029125, yaw 2.4/0.055225
    "roll_accel_limits": {
        "min": -89.0,
        "max": 89.0
    },
    "pitch_accel_limits": {
        "min": -150.0,
        "max": 150.0
    },
    "yaw_accel_limits": {
        "min": -44.0,
        "max": 44.0
    },

    # 8. Linear Velocity Limits (m/s)
    # Not capped by the sim. PX4's default is 12 m/s and the planner uses v_max = 4.
    "linear_velocity_limits": {
        "min": -12.0,
        "max": 12.0
    },

    # 9. Linear Acceleration Limits (m/s^2)
    # Horizontal per axis: max about 25.7 at the tilt limit of 1.21 rad, 15 is conservative.
    # Vertical: up (41.4 - 14.7)/1.5 = 17.8 m/s^2, down limited to about -9.8 (free fall).
    "linear_accel_limits": {
        "min": -15.0,
        "max": 15.0
    },
    "vertical_accel_limits": {
        "min": -9.0,
        "max": 17.0
    },

    # 10. Power System Limits
    # Pegasus has no battery model, so these are unused in sim. Real Iris: 3S 11.1 V 3.5 Ah.
    # The current limit is an estimate (average draw is about 15-20 A).
    "battery_voltage_limits": {
        "min": 9.9,   # 3S cutoff under load
        "max": 12.6   # 3S fully charged
    },
    "battery_current_limits": {
        "min": 0.0,
        "max": 45.0
    },

    # 11. Structural / Thermal Limits
    "max_load_factor_g": 2.8,  # T/W = 2.81, so this is the most the thrust can produce
    "motor_temperature_limits": {
        "min": -20.0,
        "max": 110.0
    },

    # 12. Altitude / Height Limits
    "altitude_limits": {
        "min": 0.0,   # Ground level
        "max": 120.0  # Regulatory/track ceiling (m)
    }
}
