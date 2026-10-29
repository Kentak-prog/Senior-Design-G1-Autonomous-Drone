# One sample of a minimum-snap reference (Trajectory/min_snap_node.py).
# World frame ENU (x east, y north, z up), body frame FLU, Euler angles
# R = Rz(yaw) Ry(pitch) Rx(roll) -- the conventions of control/New_Drone_State_Equations.py.

float64 t        # seconds after MinSnapTrajectory.header.stamp

float64 x        # position (m)
float64 y
float64 z

float64 roll     # attitude the drone needs to fly this point (rad, differential flatness)
float64 pitch
float64 yaw

float64 vx       # velocity (m/s, world frame)
float64 vy
float64 vz

float64 p        # body rates (rad/s)
float64 q
float64 r

float64 thrust   # feed-forward total thrust (N), for warm-starting the MPC
