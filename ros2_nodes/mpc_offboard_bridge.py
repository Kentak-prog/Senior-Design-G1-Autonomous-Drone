#!/usr/bin/env python3
"""
Bridge node: MPC attitude + thrust setpoints (ros2_nodes/mpc_node.py) ->
PX4 OFFBOARD over MAVLink (SET_ATTITUDE_TARGET).

*** UNTESTED AGAINST THE CLUSTER. *** The conversions and the OFFBOARD
sequence (px4/attitude_bridge.py) are tested against a fake PX4.

WHAT IT DOES:
    connect -> live PX4 status (must be armed and hovering, e.g. after
    `commander takeoff`) -> wait for fresh, valid setpoints on
    /mpc/attitude_setpoint -> confirm -> stream them, switch to OFFBOARD ->
    forward every new setpoint at --rate Hz -> on Ctrl-C or --max-duration,
    AUTO.LOITER. Once PX4 confirms OFFBOARD it publishes True (latched) on
    --engaged-topic, which starts min_snap_node's plan (start_topic). Its own
    log (what it sent, PX4's position) goes to --log-dir/bridge_log.csv.

ABORTS to AUTO.LOITER: no new setpoint for --stale-s, the MPC reporting
invalid (solver failed) for --invalid-s, a setpoint tilted more than
--max-tilt-deg, any reason on --abort-topic (ros2_nodes/flight_monitor.py's
watchdog), or Ctrl-C. If someone switches PX4 out of OFFBOARD the bridge
stops sending and exits without fighting them.

Run (ros2-eval, ROS sourced, drone_msgs built, mpc_node running):
    python3 ros2_nodes/mpc_offboard_bridge.py --ros-args -p use_sim_time:=true
    (plain flags go before --ros-args, e.g. --hover-thrust 0.55 --yes)
"""

import argparse
import csv
import math
import os
import sys
import threading
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_REPO, "px4"), os.path.join(_REPO, "control")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from attitude_bridge import AttitudeBridge, thrust_normalized, tilt_deg, G   # noqa: E402
from fly_min_snap_offboard import connect, wait_for_pose, yaw_ned_to_enu     # noqa: E402
from fly_gate_approach import ned_to_enu                                     # noqa: E402
from drone_constraints import drone_constraints                             # noqa: E402


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--connection", default="udpin:0.0.0.0:14540")
    ap.add_argument("--setpoint-topic", default="/mpc/attitude_setpoint")
    ap.add_argument("--rate", type=float, default=50.0, help="MAVLink send rate, Hz (default 50)")
    ap.add_argument("--stale-s", type=float, default=0.3, help="abort if no new setpoint for this long (s)")
    ap.add_argument("--invalid-s", type=float, default=0.5, help="abort if the MPC reports invalid this long (s)")
    ap.add_argument("--max-tilt-deg", type=float, default=45.0)
    ap.add_argument("--hover-thrust", type=float, default=None,
                    help="normalized hover throttle (e.g. PX4 MPC_THR_HOVER); default: Pegasus Iris thrust curve")
    ap.add_argument("--max-duration", type=float, default=120.0, help="hand back to AUTO.LOITER after this (s)")
    ap.add_argument("--min-alt", type=float, default=1.0, help="refuse below this altitude (m)")
    ap.add_argument("--abort-topic", default="/flight_monitor/abort",
                    help="std_msgs/String; any non-empty message aborts (empty string: don't listen)")
    ap.add_argument("--engaged-topic", default="/mpc_offboard_bridge/engaged",
                    help="std_msgs/Bool, latched True once in OFFBOARD (empty string: don't publish)")
    ap.add_argument("--log-dir", default="", help="write bridge_log.csv here (default: no file)")
    ap.add_argument("--yes", action="store_true", help="do not ask before engaging")
    return ap


def main():
    argv = sys.argv[1:]
    ros_args = []
    if "--ros-args" in argv:
        i = argv.index("--ros-args")
        argv, ros_args = argv[:i], argv[i:]
    args = build_parser().parse_args(argv)

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from std_msgs.msg import Bool, String
    from drone_msgs.msg import AttitudeThrustSetpoint

    rclpy.init(args=["bridge"] + ros_args)
    node = Node("mpc_offboard_bridge")
    latest = {"sp": None, "t": None}

    def on_sp(msg):
        latest["sp"], latest["t"] = msg, time.monotonic()
    node.create_subscription(AttitudeThrustSetpoint, args.setpoint_topic, on_sp, 10)
    abort = {"reason": None}

    def on_abort(msg):
        if msg.data and abort["reason"] is None:
            abort["reason"] = msg.data
    if args.abort_topic:
        node.create_subscription(String, args.abort_topic, on_abort, 10)
    engaged_pub = None
    if args.engaged_topic:
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        engaged_pub = node.create_publisher(Bool, args.engaged_topic, latched)
        engaged_pub.publish(Bool(data=False))

    def on_offboard():
        if engaged_pub is not None:
            engaged_pub.publish(Bool(data=True))
            print(f"[bridge] OFFBOARD confirmed -> True on {args.engaged_topic} (min_snap_node plans now)")

    # Callbacks run on their own thread, so setpoints keep arriving (and being
    # time-stamped) while the main thread blocks, e.g. on the confirmation prompt.
    # Spinning once per step instead left the newest setpoint seconds old after
    # the prompt, and run() aborted as "stale" before OFFBOARD (Isaac, 2026-10-29).
    from rclpy.executors import SingleThreadedExecutor
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    def get_setpoint():
        return latest["sp"], latest["t"]

    link = connect(args.connection)
    try:
        if not wait_for_pose(link):
            print("no LOCAL_POSITION_NED / ATTITUDE / autopilot HEARTBEAT within 10 s; aborting")
            return 1
        pos = ned_to_enu(link.pos_ned)
        print(f"[px4] ENU {np.round(pos, 2)}, heading {math.degrees(yaw_ned_to_enu(link.yaw)):+.1f} deg, "
              f"armed={link.armed}, main mode {link.main_mode}")
        if not link.armed or link.pos_ned[2] > -args.min_alt:
            print("REFUSING: the drone must be armed and hovering (e.g. `commander takeoff`).")
            return 1

        time.sleep(0.5)                      # let a latched abort reason arrive
        if abort["reason"]:
            print(f"REFUSING: the flight monitor has already tripped: {abort['reason']}")
            return 1
        bridge = AttitudeBridge(link, get_setpoint, rate_hz=args.rate, stale_s=args.stale_s,
                                invalid_s=args.invalid_s, max_tilt_deg=args.max_tilt_deg,
                                hover_thrust=args.hover_thrust, external_abort=lambda: abort["reason"],
                                on_offboard=on_offboard)
        print(f"waiting for MPC setpoints on {args.setpoint_topic} ...")
        sp = bridge.wait_for_setpoints()
        if sp is None:
            print("REFUSING: no fresh, valid MPC setpoints within 10 s (is mpc_node running?)")
            return 1
        u = thrust_normalized(sp.thrust, args.hover_thrust)
        tilt = tilt_deg((sp.orientation.x, sp.orientation.y, sp.orientation.z, sp.orientation.w))
        print(f"[mpc] thrust {sp.thrust:.2f} N -> normalized {u:.3f} (hover at {drone_constraints['mass']} kg "
              f"= {thrust_normalized(drone_constraints['mass'] * G, args.hover_thrust):.3f}), tilt {tilt:.1f} deg, "
              f"solve {sp.solve_time_ms:.1f} ms")
        if not args.yes and input("Hand control to the MPC in OFFBOARD? [y/N] ").strip().lower() not in ("y", "yes"):
            print("not engaging.")
            return 0
        status = bridge.run(max_duration=args.max_duration)
        print(f"[bridge] {status}; sent {bridge.sent} setpoints")
        if args.log_dir:
            os.makedirs(args.log_dir, exist_ok=True)
            path = os.path.join(args.log_dir, "bridge_log.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "px4_x", "px4_y", "px4_z", "qw_ned", "qx_ned", "qy_ned", "qz_ned",
                            "p_frd", "q_frd", "r_frd", "thrust_norm", "thrust_n"])
                w.writerows(bridge.rows)
                w.writerow([f"# {status}"])
            print(f"[bridge] log {path}")
        return 0 if status.startswith("completed") or status == "stopped: Ctrl-C" else 1
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)     # else rclpy aborts the process at exit
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main())
