"""Self-tests for px4/attitude_bridge.py (behind ros2_nodes/mpc_offboard_bridge.py). Run: python3 test_attitude_bridge.py

No pymavlink, no ROS. Frame and thrust conversions, and the OFFBOARD sequence
against FakePX4 (from test_fly_min_snap_offboard.py) taught to accept
SET_ATTITUDE_TARGET. Does NOT prove the real PX4 link (untested on the cluster).
"""

import math
import os
import sys
from types import SimpleNamespace as NS

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "ros2_nodes")]

import attitude_bridge as br                                # noqa: E402
import fly_min_snap_offboard as fms                        # noqa: E402
from test_fly_min_snap_offboard import FakePX4, _quiet, OFFBOARD, AUTO, POSCTL   # noqa: E402
from ros_state import quat_xyzw_from_rpy, rotation_from_rpy                      # noqa: E402

M = br.drone_constraints["mass"]


def _ned_rpy(q_wxyz):
    """roll, pitch, yaw (deg) of a NED/FRD quaternion, aerospace ZYX."""
    w, x, y, z = q_wxyz
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1, min(1, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return [math.degrees(a) for a in (roll, pitch, yaw)]


def test_attitude_and_rates_to_ned_frd():
    # level, facing east (ENU yaw 0) -> PX4 yaw +90 (east)
    assert np.allclose(_ned_rpy(br.attitude_ned_frd(quat_xyzw_from_rpy(0, 0, 0))), [0, 0, 90], atol=1e-9)
    # level, facing north (ENU yaw +90) -> PX4 yaw 0
    assert np.allclose(_ned_rpy(br.attitude_ned_frd(quat_xyzw_from_rpy(0, 0, math.pi / 2))), [0, 0, 0], atol=1e-9)
    # matches the yaw conversion the PX4 position script already flew with
    for yaw_enu in np.radians([-7.24, 30, 150, -120]):
        got = _ned_rpy(br.attitude_ned_frd(quat_xyzw_from_rpy(0, 0, yaw_enu)))[2]
        want = math.degrees(fms.yaw_enu_to_ned(yaw_enu))
        assert abs((got - want + 180) % 360 - 180) < 1e-9
    # positive pitch in the control model tilts thrust toward body +x (nose DOWN):
    # PX4 FRD calls nose-down a NEGATIVE pitch
    R = rotation_from_rpy(0, math.radians(10), 0)
    assert R[0, 2] > 0                                          # thrust tilted forward
    assert np.allclose(_ned_rpy(br.attitude_ned_frd(quat_xyzw_from_rpy(0, math.radians(10), 0))), [0, -10, 90],
                       atol=1e-9)
    # roll right: FLU +roll lifts the left side (right side down); PX4 FRD also calls that +roll
    R = rotation_from_rpy(math.radians(15), 0, 0)
    assert R[2, 1] > 0                                          # body left axis points up
    assert np.allclose(_ned_rpy(br.attitude_ned_frd(quat_xyzw_from_rpy(math.radians(15), 0, 0)))[:2], [15, 0],
                       atol=1e-9)
    assert np.allclose(br.rates_frd([0.1, 0.2, 0.3]), [0.1, -0.2, -0.3])
    print("  ENU/FLU -> NED/FRD: yaw east = 90, nose-down = -pitch, roll right = +roll in both, rates (p,-q,-r)  OK")


def test_thrust_normalization():
    hover = br.thrust_normalized(M * 9.81)
    omega = 1000 * hover + 100
    assert abs(4 * br.KF * omega ** 2 - M * 9.81) < 1e-6 and 0.5 < hover < 0.6, hover
    assert br.thrust_normalized(0.0) == 0.0 and br.thrust_normalized(1e6) == 1.0
    assert abs(br.thrust_normalized(M * 9.81, hover_thrust=0.47) - 0.47) < 1e-12
    print(f"  thrust: hover {M * 9.81:.2f} N -> {hover:.3f} (Pegasus Iris curve), or the given hover throttle  OK")


class FakePX4Attitude(FakePX4):
    """FakePX4 that also accepts SET_ATTITUDE_TARGET (counts as streaming setpoints for OFFBOARD)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.att = []

    def set_attitude_target_send(self, t_boot, ts, tc, mask, q, p, q_rate, r, thrust):
        assert mask == 0 and len(q) == 4
        self.att.append((self.now, tuple(q), (p, q_rate, r), thrust))
        self.sp_times.append(self.now)


def _setpoint_source(fake, *, stale_after=None, invalid_after=None, tilt_deg=0.0):
    """Fresh hover setpoints every call, optionally going stale / invalid after some time."""
    t_start = fake.now

    def get():
        t = fake.now - t_start
        if stale_after is not None and t > stale_after:
            return sp_last[0], arrived_last[0]
        sp = NS(orientation=NS(**dict(zip("xyzw", quat_xyzw_from_rpy(0.0, math.radians(tilt_deg), 0.3)))),
                body_rate=NS(x=0.0, y=0.0, z=0.0), thrust=M * 9.81, solve_time_ms=8.0,
                valid=not (invalid_after is not None and t > invalid_after))
        sp_last[0], arrived_last[0] = sp, fake.now
        return sp, fake.now
    sp_last, arrived_last = [None], [None]
    return get


def _bridge(fake, get, **kw):
    link = fms.PX4Link(fake, clock=fake.clock)
    fms.wait_for_pose(link, clock=fake.clock, sleep=fake.sleep)
    return br.AttitudeBridge(link, get, clock=fake.clock, sleep=fake.sleep, out=lambda *a: None, **kw)


def test_runs_in_offboard_then_hands_back():
    fake = FakePX4Attitude()
    b = _bridge(fake, _setpoint_source(fake))
    status = b.run(max_duration=3.0)
    assert status.startswith("completed") and fake.mode == AUTO and fake.loiter_commands >= 1, status
    assert fake.offboard_since is not None and b.sent > 150
    q = fake.att[-1][1]
    assert np.allclose(_ned_rpy(q), [0, 0, math.degrees(fms.yaw_enu_to_ned(0.3))], atol=1e-6)
    assert abs(fake.att[-1][3] - br.thrust_normalized(M * 9.81)) < 1e-12
    rate = (len(fake.att) - 1) / (fake.att[-1][0] - fake.att[0][0])
    assert 45 < rate < 55, rate
    print(f"  streams, enters OFFBOARD, forwards {b.sent} setpoints at {rate:.0f} Hz, then AUTO.LOITER  OK")


def test_aborts_on_stale_invalid_or_tilted():
    for kw, expect in ((dict(stale_after=2.5), "no new MPC setpoint"),
                       (dict(invalid_after=2.5), "MPC reported invalid"),
                       (dict(tilt_deg=50.0), "setpoint tilt")):
        fake = FakePX4Attitude()
        b = _bridge(fake, _setpoint_source(fake, **kw))
        status = b.run(max_duration=10.0)
        assert status.startswith("aborted") and expect in status and fake.loiter_commands >= 1, status
    print("  aborts to AUTO.LOITER on a stale setpoint, a failing solver, or a 50 deg tilt  OK")


def test_does_not_fight_a_mode_switch():
    fake = FakePX4Attitude(mode_switch_after=1.0)       # "pilot" switches to POSCTL 1 s into OFFBOARD
    b = _bridge(fake, _setpoint_source(fake))
    status = b.run(max_duration=10.0)
    assert status.startswith("stopped: PX4 left OFFBOARD") and fake.mode == POSCTL and fake.loiter_commands == 0
    print("  someone switches PX4 out of OFFBOARD -> bridge stops, no LOITER fight  OK")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        t()
    print(f"\nAll tests passed. ({len(tests)})")
