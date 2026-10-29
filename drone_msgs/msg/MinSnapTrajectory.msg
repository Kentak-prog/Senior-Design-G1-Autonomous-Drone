# A whole minimum-snap reference. header.stamp is the ROS time of t = 0, so a
# follower evaluates the reference at (now - header.stamp); with use_sim_time
# that is sim time. A new plan gets a new stamp; republishing the same plan
# keeps the old one.
std_msgs/Header header
MinSnapPoint[] points
