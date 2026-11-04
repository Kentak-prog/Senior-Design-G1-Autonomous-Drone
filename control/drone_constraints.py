#Aaron Arnold
#physical constraints of a drone
#
# The numbers come from the active vehicle profile in vehicles.py, chosen with
# the DRONE_VEHICLE environment variable (default "iris", the Pegasus/Isaac Sim
# Iris). Change a vehicle's numbers there, not here. For the Iris:
#   mass 1.5 kg, inertia (0.029125, 0.029125, 0.055225) kg*m^2   (read from iris.usd)
#   thrust T = 8.54858e-6 * omega^2, omega_max = 1100 rad/s       (quadratic_thrust_curve.py defaults)
#   yaw reaction torque = 1e-6 * omega^2 per motor
#   linear drag [0.50, 0.30, 0.0] N per m/s                        (iris.py)

import math

from vehicles import active_vehicle

vehicle = active_vehicle()
_v = vehicle
_g = 9.81

# Drone mass. Used by the state equations, the MPC cost/optimizer, the EKF,
# the race sim and the trajectory reference.
m = _v.mass #kg

# Principal moments of inertia (kg*m^2) -- use these in get_drone_dynamics and the sympy script
inertia = {
    "Ixx": _v.inertia_diag[0],
    "Iyy": _v.inertia_diag[1],
    "Izz": _v.inertia_diag[2],
}

# Linear drag applied by Pegasus (N per m/s, body x, y, z). The MPC dynamics should include it.
linear_drag = {
    "x": _v.drag_coeff[0],
    "y": _v.drag_coeff[1],
    "z": _v.drag_coeff[2],
}

_tau = _v.body_torque_max
_rates = _v.rate_limits

drone_constraints = {
    # 0. Mass
    "mass": m, #kg

    # 1. Attitude Angles (in radians)
    # Below the max tilt that still holds altitude, arccos(m g / T_max), to leave thrust margin.
    "roll_limits": {
        "min": -_v.tilt_limit,
        "max": _v.tilt_limit
    },
    "pitch_limits": {
        "min": -_v.tilt_limit,
        "max": _v.tilt_limit
    },
    "yaw_limits": {
        "min": -3.14, # Full operational heading range (-180 deg)
        "max": 3.14  # Full operational heading range (+180 deg)
    },

    # 2. Actuator Constraints
    # Per-motor yaw REACTION torque k_m * w_max^2 (Iris: 1.21 N*m at full speed).
    # Note: this is not the body torque the MPC commands. See body_torque_limits below.
    "motor_torque_limits": {
        "min": 0.0,
        "max": _v.k_m * _v.w_max ** 2
    },

    # Body torques (N*m) commanded by the MPC (U2..U4). Full-saturation values from rotor
    # geometry (see the vehicle profile).
    "body_torque_limits": {
        "roll":  {"min": -_tau[0], "max": _tau[0]},
        "pitch": {"min": -_tau[1], "max": _tau[1]},
        "yaw":   {"min": -_tau[2], "max": _tau[2]}
    },

    # 3. Actuator Rate Limits (Slew Rates / Delta U)
    # Pegasus' quadratic thrust curve has no motor lag, so this is a smoothing choice, not physics.
    # Unchanged from before; tune against the MPC's dt (0.02 s).
    "motor_torque_rate_limits": {
        "min": -50.0, # N*m/s
        "max": 50.0
    },

    # 4. Motor Speed (RPM) Limits
    # w_max in RPM (Iris: 1100 rad/s = 10,504 RPM; hover is about 6,270 RPM).
    "motor_rpm_limits": {
        "min": 1500.0,
        "max": _v.w_max * 60.0 / (2.0 * math.pi)
    },

    # 5. Thrust Limits (Newtons)
    # k_f * w_max^2 per motor. The total is set slightly below 4x that in the profile.
    "motor_thrust_limits": {
        "min": 0.0,
        "max": _v.max_rotor_thrust()
    },
    "total_thrust_limits": {
        "min": 0.0,
        "max": _v.total_thrust_max
    },

    # 6. Angular Rate Limits (Gyro Limits, in rad/s)
    # Not capped by the sim; these are practical limits for gate flying.
    "roll_rate_limits": {
        "min": -_rates[0],
        "max": _rates[0]
    },
    "pitch_rate_limits": {
        "min": -_rates[1],
        "max": _rates[1]
    },
    "yaw_rate_limits": {
        "min": -_rates[2],
        "max": _rates[2]
    },

    # 7. Angular Acceleration Limits (rad/s^2) at full torque saturation: torque / inertia
    "roll_accel_limits": {
        "min": -_tau[0] / inertia["Ixx"],
        "max": _tau[0] / inertia["Ixx"]
    },
    "pitch_accel_limits": {
        "min": -_tau[1] / inertia["Iyy"],
        "max": _tau[1] / inertia["Iyy"]
    },
    "yaw_accel_limits": {
        "min": -_tau[2] / inertia["Izz"],
        "max": _tau[2] / inertia["Izz"]
    },

    # 8. Linear Velocity Limits (m/s)
    # Not capped by the sim. PX4's default is 12 m/s and the planner uses 80 % of it.
    "linear_velocity_limits": {
        "min": -_v.max_velocity,
        "max": _v.max_velocity
    },

    # 9. Linear Acceleration Limits (m/s^2)
    # Horizontal per axis: conservative, below g * tan(tilt limit).
    # Vertical: up (T_max - m g) / m, down limited to about -9.8 (free fall).
    "linear_accel_limits": {
        "min": -_v.max_accel,
        "max": _v.max_accel
    },
    "vertical_accel_limits": {
        "min": _v.vertical_accel[0],
        "max": _v.vertical_accel[1]
    },

    # 10. Power System Limits
    # Pegasus has no battery model, so these are unused in sim.
    "battery_voltage_limits": {
        "min": _v.battery_voltage[0],   # cutoff under load
        "max": _v.battery_voltage[1]    # fully charged
    },
    "battery_current_limits": {
        "min": 0.0,
        "max": _v.battery_current_max
    },

    # 11. Structural / Thermal Limits
    "max_load_factor_g": _v.total_thrust_max / (m * _g),  # the most the thrust can produce
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
