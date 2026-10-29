#!/usr/bin/env python3
"""
Flight monitor for the Isaac VIO + MPC run: preflight GO / NO-GO, one CSV of
the whole flight, and a sim-only watchdog the bridge listens to.
The checks are ROS-free in px4/flight_checks.py (px4/test_flight_checks.py).

What it does
  every window_s   prints the preflight table (px4/flight_checks.preflight_checks):
                   /clock, topic rates, stamp lags (all on sim time?), VIO vs
                   truth, no trajectory yet (it starts on engage), MPC hover
                   solves, altitude -> GO / NO-GO
                   on /flight_monitor/preflight (std_msgs/String)
  at rate_hz       logs truth, VIO, reference and setpoint to <log_dir>/flight_log.csv
  after first GO   arms the watchdog (VIO vs truth, truth vs reference, altitude,
                   VIO stale). When it trips it publishes the reason on
                   /flight_monitor/abort (std_msgs/String, 10 Hz) and
                   mpc_offboard_bridge.py hands PX4 to AUTO.LOITER.

The ground truth is used ONLY here (scoring and the safety net); nothing
in the flight stack reads it.

Run (ros2-eval, ROS sourced, drone_msgs built), before the bridge:
  python3 ros2_nodes/flight_monitor.py --ros-args -p use_sim_time:=true
Afterwards: python3 px4/analyze_vio_mpc_flight.py <log_dir>
"""

import csv
import math
import os
import sys
import time
from collections import deque

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_REPO, "control"), os.path.join(_REPO, "px4")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from flight_checks import TopicStats, preflight_checks, format_checks, Watchdog, wrap   # noqa: E402
from mpc_core import TrajectoryBuffer                                                    # noqa: E402
from ros_state import StateTracker, rpy_from_quat_xyzw, stamp_to_sec                     # noqa: E402

COLUMNS = (["t", "phase"]
           + [f"true_{k}" for k in ("x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz")]
           + [f"vio_{k}" for k in ("x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz")]
           + ["vio_std", "vio_age"]
           + [f"ref_{k}" for k in ("x", "y", "z", "yaw", "vx", "vy", "vz")] + ["traj_t"]
           + ["sp_roll", "sp_pitch", "sp_yaw", "sp_thrust", "sp_solve_ms", "sp_valid", "watchdog"])


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
    from geometry_msgs.msg import PoseStamped, TwistStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu, Image
    from std_msgs.msg import String
    from rosgraph_msgs.msg import Clock
    from drone_msgs.msg import MinSnapTrajectory, AttitudeThrustSetpoint

    class FlightMonitor(Node):
        def __init__(self):
            super().__init__("flight_monitor")
            dp = self.declare_parameter
            dp("imu_topic", "/drone00/sensors/imu")
            dp("image_topic", "/iris_0/front_cam/rgb")
            dp("odom_topic", "/vio/odometry")
            dp("pose_topic", "/drone00/state/pose")
            dp("twist_topic", "/drone00/state/twist")
            dp("trajectory_topic", "/min_snap/trajectory")
            dp("setpoint_topic", "/mpc/attitude_setpoint")
            dp("rate_hz", 50.0)
            dp("window_s", 5.0)
            dp("hover_thrust_n", 0.0)          # 0 = skip the thrust check (mass * g is printed anyway)
            dp("max_vio_err_m", 1.0)
            dp("max_track_err_m", 2.0)
            dp("min_alt_m", 0.5)
            dp("watchdog", True)
            dp("plan_on_engage", True)         # min_snap_node runs with start_topic (the runbook)
            dp("log_dir", "")
            gp = lambda n: self.get_parameter(n).value  # noqa: E731

            self.stats = {k: TopicStats(gp("window_s")) for k in ("imu", "image", "vio", "truth", "setpoint", "clock")}
            self.truth = StateTracker()
            self.vio = None                     # (12-state, stamp, arrival, std)
            self.traj = None
            self.sp = None
            self.sp_hist = deque(maxlen=int(gp("window_s") * 60))
            self.err_hist = deque(maxlen=int(gp("window_s") * gp("rate_hz")))
            self.yaw_hist = deque(maxlen=int(gp("window_s") * gp("rate_hz")))
            self.window_s = float(gp("window_s"))
            self.limits = {"hover_thrust": float(gp("hover_thrust_n")) or None,
                           "plan_on_engage": bool(gp("plan_on_engage"))}
            self.watchdog = Watchdog(gp("max_vio_err_m"), gp("max_track_err_m"), gp("min_alt_m")) \
                if gp("watchdog") else None
            self.armed = False
            self.phase = "preflight"
            self.last_report = None

            sd = qos_profile_sensor_data
            self.create_subscription(Imu, gp("imu_topic"), lambda m: self._rx("imu", m), sd)
            self.create_subscription(Image, gp("image_topic"), lambda m: self._rx("image", m), sd)
            self.create_subscription(Clock, "/clock", lambda m: self.stats["clock"].add(self._now()), 10)
            self.create_subscription(Odometry, gp("odom_topic"), self.on_vio, sd)
            self.create_subscription(PoseStamped, gp("pose_topic"), self.on_pose, sd)
            self.create_subscription(TwistStamped, gp("twist_topic"), self.truth.on_twist, sd)
            latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(MinSnapTrajectory, gp("trajectory_topic"), self.on_traj, latched)
            self.create_subscription(AttitudeThrustSetpoint, gp("setpoint_topic"), self.on_sp, 10)
            self.pub_pre = self.create_publisher(String, "/flight_monitor/preflight", 10)
            self.pub_abort = self.create_publisher(String, "/flight_monitor/abort", 10)

            log_dir = gp("log_dir") or os.path.join("/workspace/vio_runs", time.strftime("flight_%Y%m%d_%H%M%S"))
            os.makedirs(log_dir, exist_ok=True)
            self.log_dir = log_dir
            self.log_f = open(os.path.join(log_dir, "flight_log.csv"), "w", newline="")
            self.log = csv.writer(self.log_f)
            self.log.writerow(COLUMNS)
            self.events_f = open(os.path.join(log_dir, "monitor_events.txt"), "w")
            self.create_timer(1.0 / gp("rate_hz"), self.tick)
            self.create_timer(0.1, self.publish_abort)
            self._event(f"monitor up; log {log_dir}; use_sim_time={self.get_parameter('use_sim_time').value}")

        def _now(self):
            return self.get_clock().now().nanoseconds * 1e-9

        def _event(self, text):
            self.get_logger().info(text)
            self.events_f.write(f"{self._now():.3f} {text}\n")
            self.events_f.flush()

        def _rx(self, key, msg):
            self.stats[key].add(self._now(), stamp_to_sec(msg.header.stamp))

        def on_pose(self, msg):
            self._rx("truth", msg)
            self.truth.on_pose(msg)

        def on_vio(self, msg):
            self._rx("vio", msg)
            tr = StateTracker()
            tr.on_odometry(msg)
            c = msg.pose.covariance
            std = math.sqrt(max(0.0, c[0] + c[7] + c[14]))
            self.vio = (tr.state, tr.stamp, self._now(), std)

        def on_traj(self, msg):
            try:
                buf = TrajectoryBuffer.from_msg(msg)
            except ValueError as e:
                self.get_logger().warn(f"ignoring trajectory: {e}")
                return
            if self.traj is None or abs(buf.t0 - self.traj.t0) > 1e-6:
                self._event(f"trajectory: {len(buf.times)} points, {buf.duration:.2f} s, t0={buf.t0:.2f}")
            self.traj = buf

        def on_sp(self, msg):
            self._rx("setpoint", msg)
            self.sp = msg
            q = msg.orientation
            r, p, _ = rpy_from_quat_xyzw(q.x, q.y, q.z, q.w)
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, math.cos(r) * math.cos(p)))))
            self.sp_hist.append((bool(msg.valid), float(msg.solve_time_ms), float(msg.thrust), tilt))

        # ------------------------------------------------------------------
        def tick(self):
            now = self._now()
            xt = self.truth.state
            if xt is None:
                return
            vio_x, vio_age, vio_std = None, None, None
            if self.vio is not None:
                vio_x, _, arrived, vio_std = self.vio
                vio_age = now - arrived
                self.err_hist.append(float(np.linalg.norm(vio_x[0:3, 0] - xt[0:3, 0])))
                self.yaw_hist.append(abs(math.degrees(wrap(float(vio_x[5, 0] - xt[5, 0])))))

            ref, traj_t = None, None
            if self.traj is not None:
                traj_t = now - self.traj.t0
                ref = self.traj.horizon(now, 0, 0.02)[:, 0]
                if self.phase in ("preflight", "go") and traj_t >= 0:
                    self.phase = "flying"
                    self._event("trajectory time 0 reached: flying")
                if self.phase == "flying" and self.traj.ended(now):
                    self.phase = "done"
                    self._event("trajectory ended")

            reason = None
            if self.watchdog is not None and self.armed:
                ref_pos = ref[0:3] if ref is not None and traj_t is not None and traj_t >= 0 else None
                was = self.watchdog.reason
                reason = self.watchdog.update(now, xt[0:3, 0], None if vio_x is None else vio_x[0:3, 0],
                                              None if vio_x is None else now - vio_age, ref_pos)
                if reason and not was:
                    self._event(f"WATCHDOG TRIPPED: {reason} -> /flight_monitor/abort")

            sp = self.sp
            row = [f"{now:.4f}", self.phase] + [f"{v:.5f}" for v in xt[0:9, 0]]
            row += [f"{v:.5f}" for v in vio_x[0:9, 0]] if vio_x is not None else [""] * 9
            row += [f"{vio_std:.4f}" if vio_std is not None else "", f"{vio_age:.4f}" if vio_age is not None else ""]
            row += ([f"{ref[i]:.5f}" for i in (0, 1, 2, 5, 6, 7, 8)] + [f"{traj_t:.4f}"]) if ref is not None \
                else [""] * 8
            if sp is not None:
                q = sp.orientation
                row += [f"{v:.5f}" for v in rpy_from_quat_xyzw(q.x, q.y, q.z, q.w)]
                row += [f"{sp.thrust:.4f}", f"{sp.solve_time_ms:.2f}", int(bool(sp.valid))]
            else:
                row += [""] * 6
            row.append(reason or "")
            self.log.writerow(row)

            if self.last_report is None or now - self.last_report >= self.window_s:
                self.last_report = now
                if self.phase in ("preflight", "go"):
                    self.report(now, xt)

        def report(self, now, xt):
            rates = {k: s.rate(now) for k, s in self.stats.items()}
            lags = {k: self.stats[k].stamp_lag() for k in ("imu", "image", "vio")}
            traj = None
            if self.traj is not None:
                traj = {"points": len(self.traj.times), "duration": self.traj.duration,
                        "start_err": float(np.linalg.norm(self.traj.states[0:3, 0] - xt[0:3, 0]))}
            sps = None
            if self.sp_hist:
                v, ms, th, tl = (np.array(c, dtype=float) for c in zip(*self.sp_hist))
                sps = {"valid_frac": float(v.mean()), "solve_p95_ms": float(np.percentile(ms, 95)),
                       "thrust": float(th[-1]), "tilt_deg": float(tl[-1])}
            checks = preflight_checks(rates, lags, bool(self.get_parameter("use_sim_time").value),
                                      rates["clock"], list(self.err_hist), list(self.yaw_hist),
                                      None if self.vio is None else self.vio[3], traj, sps, float(xt[2, 0]),
                                      self.limits)
            text, go = format_checks(checks)
            self._event("preflight\n" + text)
            self.pub_pre.publish(String(data="GO" if go else "NO-GO"))
            if go and not self.armed:
                self.armed = True
                self.phase = "go"
                self._event("watchdog ARMED" if self.watchdog is not None else "watchdog disabled")

        def publish_abort(self):
            if self.watchdog is not None and self.watchdog.reason:
                self.pub_abort.publish(String(data=self.watchdog.reason))

        def close(self):
            self.log_f.close()
            self._event("monitor stopped")
            self.events_f.close()
            print(f"logs in {self.log_dir}")

    rclpy.init()
    node = FlightMonitor()
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
