"""
The VIO's state for runner.py --state vio (2026-11-13).

RosVioState subscribes to ros2_nodes/vio_node.py's /vio/odometry (nav_msgs/Odometry:
frame "map" = Isaac world ENU = the track files' frame, child "base_link" = body FLU,
twist in the BODY frame per REP-105) and keeps the newest sample. `latest()` returns
it in rl/ conventions: pos ENU, vel ENU (world), quat WXYZ body->world, the arrival
time on time.monotonic() (the age check uses wall time: the odometry is stamped in
Isaac sim time, PX4's t_sim is PX4's boot clock, and the two are not aligned).

rclpy comes from the ROS environment of the shell (source /opt/ros/humble/setup.bash and
FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/fastdds_udp.xml, as for every ros2-eval node).
"""

import threading
import time

import numpy as np


def quat_rotate_wxyz(q, v):
    """Rotate vector v by the unit quaternion q = (w, x, y, z)."""
    w, x, y, z = q
    u = np.array([x, y, z])
    v = np.asarray(v, dtype=np.float64)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


class VioSample:
    __slots__ = ("pos", "vel", "quat", "stamp", "t_arrival", "pos_std")

    def __init__(self, pos, vel, quat, stamp, t_arrival, pos_std):
        self.pos, self.vel, self.quat = pos, vel, quat
        self.stamp, self.t_arrival, self.pos_std = stamp, t_arrival, pos_std

    def age(self):
        return time.monotonic() - self.t_arrival


class RosVioState:
    def __init__(self, topic="/vio/odometry", node_name="rl_vio_state"):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from nav_msgs.msg import Odometry
        self._rclpy = rclpy
        self._own_init = not rclpy.ok()
        if self._own_init:
            rclpy.init()
        self.node = Node(node_name)
        self.node.create_subscription(Odometry, topic, self._on_odom, 50)
        self._lock = threading.Lock()
        self._latest = None
        self.n = 0
        self._run = True
        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self.node)
        self._thread = threading.Thread(target=self._spin, name="vio_state_spin", daemon=True)
        self._thread.start()

    def _spin(self):
        while self._run:
            try:
                self._exec.spin_once(timeout_sec=0.1)
            except Exception:       # shutdown races
                if self._run:
                    time.sleep(0.1)

    def _on_odom(self, m):
        p, o, tl = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist.linear
        quat = np.array([o.w, o.x, o.y, o.z], dtype=np.float64)
        quat /= np.linalg.norm(quat)
        vel = quat_rotate_wxyz(quat, [tl.x, tl.y, tl.z])          # body -> world
        cov = m.pose.covariance
        std = np.sqrt(np.maximum([cov[0], cov[7], cov[14]], 0.0))
        s = VioSample(np.array([p.x, p.y, p.z], dtype=np.float64), vel, quat,
                      m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec, time.monotonic(), std)
        with self._lock:
            self._latest = s
            self.n += 1

    def latest(self):
        with self._lock:
            return self._latest

    def wait(self, timeout=10.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self._latest is not None:
                return self._latest
            time.sleep(0.05)
        return None

    def close(self):
        self._run = False
        self._thread.join(timeout=1.0)
        try:
            self.node.destroy_node()
            if self._own_init:
                self._rclpy.shutdown()
        except Exception:
            pass
