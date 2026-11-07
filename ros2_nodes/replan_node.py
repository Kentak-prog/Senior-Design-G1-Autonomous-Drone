#!/usr/bin/env python3
"""
ROS 2 node: min-snap planning that REPLANS IN FLIGHT as the camera corrects
the gate map, for one or more laps (Trajectory/replanner.py). A drop-in for
min_snap_node.py: same output topic and message, so mpc_node and
flight_monitor need no change.

Inputs:
  state                          pegasus (ground truth) or odometry, as mpc_node
  gates_file                     the course manifest (test_gates.json): the PRIOR map, perturbed by
                                 prior_sigma (0 = exact) with prior_seed
  /iris_0/front_cam/gate_poses   geometry_msgs/PoseArray, world frame, from
                                 ros2_nodes/ros_camera_pose_fallback.py; corrects the map
  start_topic                    std_msgs/Bool from mpc_offboard_bridge (OFFBOARD confirmed)
Output:
  /min_snap/trajectory           drone_msgs/MinSnapTrajectory, latched. A NEW message on every
                                 replan; header.stamp = that plan's t = 0. Each plan starts from
                                 the current reference state, so mpc_node's reference stays
                                 continuous when it switches.

Sequence: the state arrives for settle_s -> plans from the hover pose with the
current map (re-planned while hovering if the camera moves a gate) -> on the
start signal re-stamps that plan to now and publishes it -> every tick: marks
gates passed, replans when a planned gate moves (more than max(pos_tol 0.10 m,
2 % of its range)), turns, gets a reliable normal, or the window runs out;
never inside the last commit_dist (1.5 m) before the next gate.

Logs (log_dir): replan_events.csv (every plan, pass and miss) and
gate_map.csv (the map at each plan). The ROS-free logic is
Trajectory/replan_session.py (Trajectory/test_replan_session.py).

Run (ros2-eval, ROS sourced, drone_msgs built), instead of min_snap_node:
  python3 ros2_nodes/replan_node.py --ros-args -p use_sim_time:=true -p state_source:=pegasus \\
      -p gates_file:=/workspace/test_gates.json -p prior_sigma:=0.5 -p laps:=1 -p close_loop:=true \\
      -p v_max:=2.0 -p a_max:=2.0 -p start_topic:=/mpc_offboard_bridge/engaged -p log_dir:=<run dir>
"""

import csv
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_REPO, "Trajectory"), os.path.join(_REPO, "control")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from replan_session import ReplanSession, perturbed_prior               # noqa: E402
from min_snap_planner import POINT_FIELDS, DEFAULT_GATE_SIDE             # noqa: E402
from ros_state import StateTracker, subscribe_state                     # noqa: E402


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
    from geometry_msgs.msg import PoseArray
    from std_msgs.msg import Bool
    from drone_msgs.msg import MinSnapPoint, MinSnapTrajectory
    from gate_path import gates_from_manifest

    class ReplanNode(Node):
        def __init__(self):
            super().__init__("replan_node")
            dp = self.declare_parameter
            dp("state_source", "pegasus")
            dp("pose_topic", "/drone00/state/pose")
            dp("twist_topic", "/drone00/state/twist")
            dp("odom_topic", "/vio/odometry")
            dp("gate_topic", "/iris_0/front_cam/gate_poses")
            dp("trajectory_topic", "/min_snap/trajectory")
            dp("start_topic", "/mpc_offboard_bridge/engaged")
            dp("gates_file", "/workspace/test_gates.json")
            dp("prior_sigma", 0.5)          # m; 0 = the manifest exactly
            dp("prior_seed", 0)
            dp("laps", 1)
            dp("close_loop", True)          # end back through the first gate
            dp("v_max", 2.0)
            dp("a_max", 2.0)
            dp("settle_s", 3.0)
            dp("start_tol_m", 0.5)
            dp("rate_hz", 20.0)
            dp("prepare_every_s", 1.0)      # before engage: how often to re-check the plan against the map
            dp("use_camera", True)
            dp("yaw_mode", "velocity")      # next_gate: look at the next gate (needed when the VIO flies)
            dp("gate_side", DEFAULT_GATE_SIDE)
            dp("max_gate_range_m", 0.0)     # ignore camera gate poses farther than this (0 = no limit);
                                            # 9.0 for the Starling's wide C29 camera (replan_session.py)
            dp("map_new_gates", True)       # false: the camera only corrects the prior's gates, never
                                            # adds one (a known course; gate_map.GateMap allow_new_gates)
            dp("log_dir", "")
            gp = lambda n: self.get_parameter(n).value   # noqa: E731
            self.gp = gp

            true_gates = gates_from_manifest(gp("gates_file"))
            prior = perturbed_prior(true_gates, float(gp("prior_sigma")), int(gp("prior_seed")))
            self.session = ReplanSession(prior, v_max=float(gp("v_max")), a_max=float(gp("a_max")),
                                         laps=int(gp("laps")), close_loop=bool(gp("close_loop")),
                                         start_tol_m=float(gp("start_tol_m")), gate_side=float(gp("gate_side")),
                                         prior_sigma=max(float(gp("prior_sigma")), 0.05),
                                         max_detection_range_m=float(gp("max_gate_range_m")),
                                         allow_new_gates=bool(gp("map_new_gates")),
                                         yaw_mode=str(gp("yaw_mode")))
            for p, g in zip(prior, true_gates):
                self.get_logger().info(f"prior {p.name}: {np.round(p.center, 2)} "
                                       f"({np.linalg.norm(p.center - g.center):.2f} m from the manifest)")

            self.tracker = StateTracker()
            subscribe_state(self, self.tracker, gp("state_source"), gp("pose_topic"), gp("twist_topic"),
                            gp("odom_topic"))
            if gp("use_camera"):
                self.create_subscription(PoseArray, gp("gate_topic"), self.on_gates, qos_profile_sensor_data)
            latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.pub = self.create_publisher(MinSnapTrajectory, gp("trajectory_topic"), latched)
            self.create_subscription(Bool, gp("start_topic"), self.on_start, latched)
            self.started = False
            self.first_state_t = None
            self.last_prepare_t = None
            self.n_gate_msgs = 0
            self.done_logged = False

            log_dir = gp("log_dir") or os.path.join("/workspace/vio_runs", time.strftime("replan_%Y%m%d_%H%M%S"))
            os.makedirs(log_dir, exist_ok=True)
            self.log_dir = log_dir
            self.ev_f = open(os.path.join(log_dir, "replan_events.csv"), "w", newline="")
            self.ev = csv.writer(self.ev_f)
            self.ev.writerow(["t", "event"])
            self.map_f = open(os.path.join(log_dir, "gate_map.csv"), "w", newline="")
            self.mapw = csv.writer(self.map_f)
            self.mapw.writerow(["t", "plan", "gate", "x", "y", "z", "nx", "ny", "nz", "normal_reliable", "n_obs"])
            self._n_events = 0
            self.create_timer(1.0 / float(gp("rate_hz")), self.tick)
            self.get_logger().info(f"replanning {gp('laps')} lap(s), close_loop={gp('close_loop')}, "
                                   f"yaw {gp('yaw_mode')}, state {gp('state_source')}, "
                                   f"v/a {gp('v_max')}/{gp('a_max')}, camera gates within "
                                   f"{gp('max_gate_range_m') if float(gp('max_gate_range_m')) > 0 else 'any'} m, "
                                   f"{'new gates allowed' if gp('map_new_gates') else 'prior gates only'}; "
                                   f"waiting for True on {gp('start_topic')}; "
                                   f"log {log_dir}; use_sim_time={self.get_parameter('use_sim_time').value}")

        def _now(self):
            return self.get_clock().now().nanoseconds * 1e-9

        def _flush_events(self):
            for t, text in self.session.events[self._n_events:]:
                self.ev.writerow([f"{t:.3f}", text])
                self.get_logger().info(f"t={t:.2f} {text}")
            self._n_events = len(self.session.events)
            self.ev_f.flush()

        def on_gates(self, msg):
            x = self.tracker.state
            if x is None:
                return
            poses = [((p.position.x, p.position.y, p.position.z),
                      (p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)) for p in msg.poses]
            if poses:
                self.session.add_detections(self._now(), poses, x[0:3, 0])
                self.n_gate_msgs += 1

        def on_start(self, msg):
            if msg.data and not self.started:
                self.started = True
                x = self.tracker.state
                if x is None:
                    self.get_logger().error("start received but no state: cannot plan")
                    return
                self.get_logger().info("start received: publishing the plan")
                p = self.session.engage(self._now(), x[0:3, 0], float(x[5, 0]), clock=self._now)
                self.publish(p)

        def tick(self):
            x = self.tracker.state
            if x is None:
                return
            now = self._now()
            if not self.started:
                if self.first_state_t is None:
                    self.first_state_t = now
                if now - self.first_state_t < float(self.gp("settle_s")):
                    return
                if self.last_prepare_t is None or now - self.last_prepare_t >= float(self.gp("prepare_every_s")):
                    self.last_prepare_t = now
                    text = self.session.prepare(now, x[0:3, 0], float(x[5, 0]))
                    if text:
                        self._log_map(now, 0)
                    self._flush_events()
                return
            p = self.session.tick(now, x[0:3, 0])
            self._flush_events()
            if p is not None:
                self.publish(p)
            if self.session.finished(now) and not self.done_logged:
                self.done_logged = True
                passed, total = self.session.progress()
                self.get_logger().info(f"route finished: {passed}/{total} route entries passed; "
                                       f"{self.session.n_published} plan(s) published, "
                                       f"{self.session.n_detections} gate detections fused")

        def publish(self, p):
            msg = MinSnapTrajectory()
            sec = int(p.t0)
            msg.header.stamp.sec, msg.header.stamp.nanosec = sec, int(round((p.t0 - sec) * 1e9))
            msg.header.frame_id = "map"
            for k, t in enumerate(p.times):
                pt = MinSnapPoint()
                pt.t = float(t)
                for i, f in enumerate(POINT_FIELDS):
                    setattr(pt, f, float(p.states[i, k]))
                pt.thrust = float(p.thrust[k])
                msg.points.append(pt)
            self.pub.publish(msg)
            self._log_map(p.t0, p.index)
            self.get_logger().info(f"published plan #{p.index} ({p.reason}): {p.times[-1]:.2f} s through "
                                   f"{', '.join(p.gate_names)}; t0={p.t0:.2f} (now {self._now():.2f})")

        def _log_map(self, t, plan_index):
            for name, c, n, rel, nobs in self.session.map_snapshot():
                self.mapw.writerow([f"{t:.3f}", plan_index, name, *[f"{v:.4f}" for v in c],
                                    *[f"{v:.4f}" for v in n], int(rel), nobs])
            self.map_f.flush()

        def close(self):
            self._flush_events()
            self.ev_f.close()
            self.map_f.close()
            print(f"logs in {self.log_dir}")

    rclpy.init()
    node = ReplanNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
