# MPC output (control/mpc_node.py) for a flight controller's attitude loop
# (px4/mpc_offboard_bridge.py turns it into MAVLink SET_ATTITUDE_TARGET).
# Frames: world ENU, body FLU (header.frame_id "map"). The bridge converts to
# PX4's NED / FRD.

std_msgs/Header header
geometry_msgs/Quaternion orientation   # desired attitude, body (FLU) -> world (ENU), x y z w
geometry_msgs/Vector3 body_rate        # desired body rates (rad/s, FLU), feed-forward
float64 thrust                         # desired total thrust (N)
float64 solve_time_ms                  # how long the MPC solve took (diagnostics)
bool valid                             # false: the solver failed and this is a hover fallback
