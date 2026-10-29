"""Self-tests for Kalman_Filter_MPC.DroneCasadiEKF12D. Run: python3 test_kalman_filter_mpc.py

No ROS, no simulator. Covers the Q/R tables, IMU (strapdown) prediction,
partial-state updates (gyro, pose, position, velocity), angle wrapping, and a
vision + IMU loop that recovers from a wrong initial position.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Kalman_Filter_MPC import DroneCasadiEKF12D, wrap_angle   # noqa: E402

G = 9.81


def _ekf(**kw):
    ekf = DroneCasadiEKF12D(dt=0.01, **kw)
    ekf.x = np.zeros((12, 1))
    ekf.x[2, 0] = 2.5
    return ekf


def test_q_r_tables_match_their_labels():
    ekf = _ekf()
    q, r = np.diag(ekf.Q), np.diag(ekf.R)
    # state order [pos, angles, vel, rates]
    assert np.allclose(q[0:3], [0.001, 0.001, 0.05])   # positions (increased Z)
    assert np.allclose(q[3:6], [0.001, 0.001, 0.05])   # angles (increased yaw)
    assert np.allclose(q[6:9], [0.01, 0.01, 0.01])     # velocities
    assert np.allclose(q[9:12], [0.01, 0.01, 0.01])    # rates
    assert np.allclose(r[0:3], [0.5, 0.5, 0.1])        # positions (increased X/Y)
    assert np.allclose(r[3:6], [0.8, 0.8, 0.1])        # angles (increased roll/pitch)
    assert np.allclose(r[6:9], [0.1, 0.1, 0.1])        # velocities
    assert np.allclose(r[9:12], [0.2, 0.2, 0.2])       # rates
    print("  Q and R values sit on the states their labels name  OK")


def test_imu_hover_stays_put():
    ekf = _ekf()
    for _ in range(200):                                  # 2 s of level hover
        ekf.predict_imu([0.0, 0.0, G], [0.0, 0.0, 0.0])
    assert np.allclose(ekf.x[0:3, 0], [0, 0, 2.5], atol=1e-9) and np.allclose(ekf.x[6:9, 0], 0, atol=1e-9)
    print("  IMU hover (specific force +g on body z): no drift  OK")


def test_imu_forward_acceleration_and_yaw_rate():
    ekf = _ekf()
    for _ in range(100):                                  # 1 s at 1 m/s^2 forward, level
        ekf.predict_imu([1.0, 0.0, G], [0.0, 0.0, 0.0])
    assert abs(ekf.x[6, 0] - 1.0) < 1e-6 and abs(ekf.x[0, 0] - 0.5) < 1e-6, ekf.x[[0, 6], 0]
    ekf = _ekf()
    for _ in range(100):                                  # 1 s at 0.5 rad/s yaw rate
        ekf.predict_imu([0.0, 0.0, G], [0.0, 0.0, 0.5])
    assert abs(ekf.x[5, 0] - 0.5) < 1e-6, ekf.x[5, 0]
    print("  IMU: 1 m/s^2 for 1 s -> v 1.0, x 0.5; 0.5 rad/s for 1 s -> yaw 0.5  OK")


def test_imu_tilted_hover_matches_world_frame():
    # Pitched 10 deg nose-down (positive pitch tilts thrust toward +x in this model's
    # R = Rz Ry Rx). Thrust along body z of g/cos: the world-vertical part cancels
    # gravity and the drone accelerates along +x at g*tan(10 deg).
    th = np.radians(10.0)
    ekf = _ekf()
    ekf.x[4, 0] = th
    for _ in range(100):
        ekf.predict_imu([0.0, 0.0, G / np.cos(th)], [0.0, 0.0, 0.0])
    assert abs(ekf.x[6, 0] - G * np.tan(th)) < 1e-6 and abs(ekf.x[8, 0]) < 1e-6, ekf.x[6:9, 0]
    print(f"  IMU tilted 10 deg: vx {ekf.x[6, 0]:.3f} m/s = g*tan(10 deg), vz 0  OK")


def test_gyro_update_sets_rates():
    ekf = _ekf()
    ekf.predict_imu([0.0, 0.0, G], [0.1, -0.2, 0.3])
    ekf.update_gyro([0.1, -0.2, 0.3])
    assert np.allclose(ekf.x[9:12, 0], [0.1, -0.2, 0.3], atol=1e-3), ekf.x[9:12, 0]
    print("  gyro update: rate states follow the gyro  OK")


def test_pose_update_only_moves_what_it_should():
    ekf = _ekf()
    ekf.P = np.eye(12) * 0.1
    P0 = ekf.P.copy()
    ekf.update_pose([1.0, 0.0, 2.5], [0.0, 0.0, 0.0], pos_std=0.05)
    assert ekf.x[0, 0] > 0.9                               # moved most of the way to the fix
    assert np.allclose(ekf.x[6:12, 0], 0.0)                # uncorrelated states untouched
    assert np.all(np.diag(ekf.P)[0:6] < np.diag(P0)[0:6]) and np.allclose(np.diag(ekf.P)[6:12], np.diag(P0)[6:12])
    assert np.allclose(ekf.P, ekf.P.T)
    print(f"  pose update: x 0 -> {ekf.x[0, 0]:.2f} toward the 1.0 fix, other states and their variance untouched  OK")


def test_yaw_innovation_wraps():
    ekf = _ekf()
    ekf.x[5, 0] = np.radians(179.0)
    ekf.update_measurement([np.radians(-179.0)], [5], [1e-6])
    assert abs(wrap_angle(ekf.x[5, 0] - np.radians(-179.0))) < 1e-3, np.degrees(ekf.x[5, 0])
    print("  yaw +179 measured as -179: a 2 deg correction, not 358  OK")


def test_full_update_still_works():
    ekf = _ekf()
    ekf.update(np.full((12, 1), 0.1))
    assert np.all(ekf.x[:, 0] != 0.0)
    print("  update(z) with all 12 states still works (used by the Python sims)  OK")


def test_vision_imu_loop_recovers_position_and_velocity():
    """
    A drone flying at 1 m/s along +x, level. The EKF starts 2 m off in x and
    knows nothing of the velocity; it only gets the IMU at 100 Hz and a vision
    position+attitude fix at 10 Hz. Velocity is never measured -- it has to be
    inferred from the fixes through the IMU model.
    """
    rng = np.random.default_rng(0)
    ekf = _ekf(accel_noise=0.05, gyro_noise=0.005)
    ekf.x[0, 0] = -2.0                     # wrong start
    ekf.P = np.diag([4.0] * 3 + [0.01] * 3 + [1.0] * 3 + [0.01] * 3)
    true_x, v = 0.0, 1.0
    for k in range(500):                   # 5 s
        true_x += v * 0.01
        ekf.predict_imu(np.array([0.0, 0.0, G]) + rng.normal(0, 0.05, 3), rng.normal(0, 0.005, 3), dt=0.01)
        ekf.update_gyro(rng.normal(0, 0.005, 3))
        if k % 10 == 0:
            ekf.update_pose([true_x, 0.0, 2.5] + rng.normal(0, 0.05, 3), rng.normal(0, np.radians(1.0), 3))
    pos_err, vel_err = abs(ekf.x[0, 0] - true_x), abs(ekf.x[6, 0] - v)
    assert pos_err < 0.1 and vel_err < 0.1, (pos_err, vel_err)
    print(f"  vision 10 Hz + IMU 100 Hz from 2 m off: pos err {pos_err:.3f} m, vel err {vel_err:.3f} m/s "
          f"(velocity never measured)  OK")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        t()
    print(f"\nAll tests passed. ({len(tests)})")
