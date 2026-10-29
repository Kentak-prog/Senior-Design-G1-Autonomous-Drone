"""Self-tests for ros2_nodes/ros_state.py (ROS message -> 12-state). Run: python3 test_ros_state.py

No ROS: messages are plain stand-in objects with the same attributes.
"""

import os
import sys
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ros_state   # noqa: E402


def _v(x, y, z):
    return NS(x=x, y=y, z=z)


def _q(x, y, z, w):
    return NS(x=x, y=y, z=z, w=w)


def _stamp(t):
    return NS(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)))


def test_rpy_quaternion_round_trip_and_rotation():
    rng = np.random.default_rng(0)
    for _ in range(50):
        rpy = rng.uniform([-1.2, -1.2, -3.1], [1.2, 1.2, 3.1])
        q = ros_state.quat_xyzw_from_rpy(*rpy)
        assert np.allclose(ros_state.rpy_from_quat_xyzw(*q), rpy, atol=1e-9)
        # the quaternion's rotation equals the control model's Rz Ry Rx
        R = ros_state.rotation_from_rpy(*rpy)
        x, y, z, w = q
        Rq = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                       [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                       [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        assert np.allclose(R, Rq, atol=1e-9)
    print("  rpy <-> quaternion round trip, same rotation as the control model (R = Rz Ry Rx)  OK")


def test_state_tracker_pegasus_and_odometry():
    rpy = (0.1, -0.2, 0.7)
    q = _q(*ros_state.quat_xyzw_from_rpy(*rpy))
    v_body = np.array([1.0, 0.5, -0.2])
    v_world = ros_state.rotation_from_rpy(*rpy) @ v_body

    tr = ros_state.StateTracker()
    tr.on_pose(NS(header=NS(stamp=_stamp(12.5), frame_id="map"), pose=NS(position=_v(1, 2, 3), orientation=q)))
    tr.on_twist(NS(header=NS(stamp=_stamp(12.5), frame_id="base_link"),
                   twist=NS(linear=_v(*v_body), angular=_v(0.01, 0.02, 0.03))))
    x = tr.state[:, 0]
    assert np.allclose(x[0:3], [1, 2, 3]) and np.allclose(x[3:6], rpy, atol=1e-9)
    assert np.allclose(x[6:9], v_world) and np.allclose(x[9:12], [0.01, 0.02, 0.03]) and tr.stamp == 12.5

    tr2 = ros_state.StateTracker()                          # world-frame twist is used as is
    tr2.on_pose(NS(header=NS(stamp=_stamp(1.0), frame_id="map"), pose=NS(position=_v(0, 0, 1), orientation=q)))
    tr2.on_twist(NS(header=NS(stamp=_stamp(1.0), frame_id="map"), twist=NS(linear=_v(*v_world), angular=_v(0, 0, 0))))
    assert np.allclose(tr2.state[6:9, 0], v_world)

    tr3 = ros_state.StateTracker()                          # nav_msgs/Odometry, twist in the body
    tr3.on_odometry(NS(header=NS(stamp=_stamp(2.0), frame_id="map"), child_frame_id="base_link",
                       pose=NS(pose=NS(position=_v(4, 5, 6), orientation=q)),
                       twist=NS(twist=NS(linear=_v(*v_body), angular=_v(0, 0, 0.3)))))
    assert np.allclose(tr3.state[6:9, 0], v_world) and np.allclose(tr3.state[3:6, 0], rpy, atol=1e-9)
    print("  state: roll/pitch kept (old node zeroed them), body twist rotated into the world  OK")

if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        t()
    print(f"\nAll tests passed. ({len(tests)})")
