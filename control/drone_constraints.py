#9/20/2026 - updated physical constraints to include:
# rpm limits, total thrust, gyro limits, angular accel limits,
# linear velocity limit, linear accel limit, battery limit (based on 4S LiPro),
# and a height limit

#Aaron Arnold
#physical constraints of a drone

drone_constraints = {
    # 1. Attitude Angles (in radians)
    "roll_limits": {
        "min": -1.2,  # approx. -68 degrees
        "max": 1.2    # approx.  68 degrees
    },
    "pitch_limits": {
        "min": -1.2,  # approx. -68 degrees
        "max": 1.2    # approx.  68 degrees
    },
    "yaw_limits": {
        "min": -3.14, # Full operational heading range (-180 deg)
        "max": 3.14  # Full operational heading range (+180 deg)
    },

    # 2. Actuator Constraints: Motor Torques (in N·m)
    "motor_torque_limits": {
        "min": 0.0,   # Motors cannot produce negative torque/thrust
        "max": 15.0   # Maximum torque output per motor
    },

    # 3. Actuator Rate Limits (Slew Rates / Delta U)
    # Enforces that motors take physical time to spool up and down (no instantaneous RPM changes)
    "motor_torque_rate_limits": {
        "min": -50.0, # Maximum downward rate of change per time step (N·m/s)
        "max": 50.0   # Maximum upward rate of change per time step (N·m/s)
    },

    # 4. Motor Speed (RPM) Limits
    # Torque limits alone don't capture motor/ESC speed saturation
    "motor_rpm_limits": {
        "min": 1500.0,  # Idle spin — motors kept above zero to avoid desync
        "max": 28000.0  # Max commandable RPM
    },

    # 5. Total Collective Thrust Limits (sum across all motors, in Newtons)
    "total_thrust_limits": {
        "min": 0.0,
        "max": 40.0  # Determined by max thrust per motor x motor count
    },

    # 6. Angular Rate Limits (Gyro Limits, in rad/s)
    # Bounds how fast attitude angles can change — dominant limiter for aggressive racing maneuvers
    "roll_rate_limits": {
        "min": -25.0, # approx. -1430 deg/s
        "max": 25.0   # approx.  1430 deg/s
    },
    "pitch_rate_limits": {
        "min": -25.0,
        "max": 25.0
    },
    "yaw_rate_limits": {
        "min": -15.0, # approx. -860 deg/s
        "max": 15.0   # approx.  860 deg/s
    },

    # 7. Angular Acceleration Limits (rad/s^2)
    # Bounded by torque-to-inertia ratio of the airframe
    "roll_accel_limits": {
        "min": -300.0,
        "max": 300.0
    },
    "pitch_accel_limits": {
        "min": -300.0,
        "max": 300.0
    },
    "yaw_accel_limits": {
        "min": -150.0,
        "max": 150.0
    },

    # 8. Linear Velocity Limits (m/s, body/world frame)
    "linear_velocity_limits": {
        "min": -40.0,
        "max": 40.0   # approx. 144 km/h top speed
    },

    # 9. Linear Acceleration Limits (m/s^2)
    # Governed by thrust-to-weight ratio; racing frames typically 2-6g of usable acceleration
    "linear_accel_limits": {
        "min": -40.0,
        "max": 40.0  # approx. 4g
    },

    # 10. Power System Limits
    "battery_voltage_limits": {
        "min": 13.2,  # Cutoff voltage (4S LiPo under load) — below this, thrust capability degrades sharply
        "max": 16.8   # Fully charged voltage (4S LiPo)
    },
    "battery_current_limits": {
        "min": 0.0,
        "max": 120.0  # Max continuous current draw (A), bounded by battery C-rating / ESC rating
    },

    # 11. Structural / Thermal Limits
    "max_load_factor_g": 8.0,  # Max G-force airframe/props are rated to survive
    "motor_temperature_limits": {
        "min": -20.0,
        "max": 110.0  # Deg C — sustained operation above this requires throttle derating
    },

    # 12. Altitude / Height Limits
    "altitude_limits": {
        "min": 0.0,   # Ground level
        "max": 120.0  # Regulatory/track ceiling (m)
    }
}