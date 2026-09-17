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
    }
}