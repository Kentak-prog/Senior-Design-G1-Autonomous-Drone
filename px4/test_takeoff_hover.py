"""Self-tests for takeoff_hover.py. Run: python3 test_takeoff_hover.py

No pymavlink, no vehicle: the sequence (arm, OFFBOARD, climb, hover, Hold / Land) and every abort
run against test_fly_min_snap_offboard.FakePX4, extended to start on the ground and handle arming.
Like that fake, it does NOT prove the real PX4 / MAVLink link behaves the same way.
"""

import io
import contextlib
import json
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import takeoff_hover as th  # noqa: E402
import fly_min_snap_offboard as fms  # noqa: E402
from test_fly_min_snap_offboard import FakePX4, OFFBOARD, AUTO, POSCTL  # noqa: E402

GROUND = np.array([2.37, -0.48, 0.76])    # the bench pose of 2026-11-24 (local origin not the floor)
HEADING = -178.6


class GroundPX4(FakePX4):
    """Starts disarmed on the ground in Position mode; handles ARM / DISARM. Options: refuse to arm,
    refuse OFFBOARD, stop sending position after N s of OFFBOARD, report motion while on the ground."""

    def __init__(self, refuse_arm=False, refuse_offboard=False, drop_pos_after=None, moving=False, **kw):
        super().__init__(start_enu=kw.pop("start_enu", GROUND), heading_enu_deg=HEADING, **kw)
        self.mode, self.armed = POSCTL, False
        self.refuse_arm, self.refuse_offboard = refuse_arm, refuse_offboard
        self.drop_pos_after = drop_pos_after
        self.moving = moving
        self.disarm_commands = 0
        self.mode_params = []
        self.sp_after_takeover = 0
        self.taken_over_at = None

    def _step(self, h):
        was = self.mode
        super()._step(h)
        if was == OFFBOARD and self.mode == POSCTL and self.taken_over_at is None:
            self.taken_over_at = self.now

    def recv_match(self, blocking=False):
        msg = super().recv_match(blocking)
        if msg is not None and msg.get_type() == "LOCAL_POSITION_NED":
            if self.moving:
                msg.vx = 0.5
            t_off = None if self.offboard_since is None else self.now - self.offboard_since
            if self.drop_pos_after is not None and t_off is not None and t_off > self.drop_pos_after:
                return None
        return msg

    def set_position_target_local_ned_send(self, *a):
        super().set_position_target_local_ned_send(*a)
        if self.taken_over_at is not None:
            self.sp_after_takeover += 1

    def command_long_send(self, ts, tc, cmd, conf, p1, p2, p3, *rest):
        if cmd == th.MAV_CMD_COMPONENT_ARM_DISARM:
            if int(p1) == 1 and not self.refuse_arm:
                self.armed = True
            elif int(p1) == 0:
                self.disarm_commands += 1
                self.armed = False
            return
        if cmd == 176:
            self.mode_params.append((int(p2), int(p3)))
            if int(p2) == OFFBOARD and self.refuse_offboard:
                return
        super().command_long_send(ts, tc, cmd, conf, p1, p2, p3, *rest)


def _args(extra=()):
    tmp = tempfile.mkdtemp()
    return th.build_parser().parse_args(["--out-dir", os.path.join(tmp, "run"), "--hold-s", "2", *extra])


def _run(fake, extra=("--yes",), ask=lambda _: "y"):
    args = _args(extra)
    link = fms.PX4Link(fake, clock=fake.clock)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = th.run(args, link=link, clock=fake.clock, sleep=fake.sleep, ask=ask)
    summ = os.path.join(args.out_dir, "summary.json")
    return rc, buf.getvalue(), (json.load(open(summ)) if os.path.exists(summ) else None)


def _alt(fake):
    return float(fms.ned_to_enu(fake.pos)[2])


def test_quintic():
    h, T = 1.0, th.climb_duration(1.0, 0.3)
    v = [th.quintic(h, T, t)[1] for t in np.linspace(0, T, 2001)]
    assert abs(max(v) - 0.3) < 1e-3 and th.quintic(h, T, 0)[0] == 0 and th.quintic(h, T, T)[0] == h
    assert th.quintic(h, T, T / 2)[0] == 0.5 * h and th.quintic(h, T, T + 5) == (h, 0.0)
    print(f"  quintic climb: 0 -> 1 m in {T:.2f} s, peak {max(v):.3f} m/s, smooth ends  OK")


def test_takeoff_climbs_hovers_and_holds():
    fake = GroundPX4()
    rc, log, s = _run(fake, extra=("--yes", "--height", "1.0"))
    assert rc == 0 and s["status"].startswith("completed"), log
    assert abs(_alt(fake) - (GROUND[2] + 1.0)) < 0.05, _alt(fake)
    assert fake.armed and fake.mode == AUTO and (AUTO, th.PX4_SUB_AUTO_LOITER) in fake.mode_params
    xy = fms.ned_to_enu(fake.pos)[:2]
    assert np.linalg.norm(xy - GROUND[:2]) < 0.02 and s["hover_err_max_m"] < 0.05
    print(f"  arm -> OFFBOARD -> climb 1.0 m above the start (to {_alt(fake):.2f} m local) -> hover "
          f"(max error {s['hover_err_max_m'] * 100:.1f} cm) -> Hold  OK")


def test_end_land():
    fake = GroundPX4()
    rc, log, s = _run(fake, extra=("--yes", "--end", "land"))
    assert rc == 0 and (AUTO, th.PX4_SUB_AUTO_LAND) in fake.mode_params and "AUTO.LAND" in s["status"]
    print("  --end land hands PX4 to AUTO.LAND  OK")


def test_refuses_when_already_armed():
    fake = GroundPX4()
    fake.armed = True
    rc, log, _ = _run(fake)
    assert rc == 1 and "REFUSING" in log and not fake.sp_times and fake.offboard_commands == 0
    print("  refuses if already armed (sends nothing)  OK")


def test_refuses_when_moving():
    fake = GroundPX4(moving=True)
    rc, log, _ = _run(fake)
    assert rc == 1 and "moving" in log and not fake.sp_times
    print("  refuses if PX4 reports motion on the ground  OK")


def test_refuses_bad_height_and_rate():
    for extra in (("--yes", "--height", "3.5"), ("--yes", "--height", "0.1"), ("--yes", "--climb-rate", "2")):
        fake = GroundPX4()
        rc, log, _ = _run(fake, extra=extra)
        assert rc == 2 and "REFUSING" in log and not fake.sp_times, extra
    print("  refuses --height outside 0.3-2.5 m and --climb-rate outside 0.05-1 m/s  OK")


def test_declining_sends_nothing():
    fake = GroundPX4()
    rc, log, _ = _run(fake, extra=(), ask=lambda _: "n")
    assert rc == 0 and "not flying" in log and not fake.sp_times and not fake.armed
    print("  answering 'n' sends nothing  OK")


def test_arm_refused():
    fake = GroundPX4(refuse_arm=True)
    rc, log, s = _run(fake)
    assert rc == 1 and "did not arm" in s["status"] and fake.offboard_commands == 0 and _alt(fake) == GROUND[2]
    print("  PX4 refuses to arm: stops, never asks for OFFBOARD  OK")


def test_offboard_refused_disarms():
    fake = GroundPX4(refuse_offboard=True)
    rc, log, s = _run(fake)
    assert rc == 1 and "did not enter OFFBOARD" in s["status"]
    assert fake.disarm_commands == 1 and not fake.armed and _alt(fake) == GROUND[2]
    print("  armed but OFFBOARD refused: disarms on the ground  OK")


def test_pilot_takeover_stops_sending():
    fake = GroundPX4(mode_switch_after=2.0)          # pilot flips to Position 2 s into OFFBOARD
    rc, log, s = _run(fake)
    assert rc == 1 and "pilot took over" in s["status"]
    # The script only learns the mode from PX4's heartbeat (the fake's: 2 Hz; the drone's: 10 Hz), so
    # a few setpoints still go out after the switch; PX4 ignores them outside OFFBOARD. It must stop
    # within one heartbeat period and send NO mode change of its own (that would fight the pilot).
    assert fake.sp_after_takeover <= 0.5 * 50 and (AUTO, th.PX4_SUB_AUTO_LOITER) not in fake.mode_params
    assert len(fake.mode_params) == 1 and fake.mode == POSCTL
    print(f"  pilot switches mode mid-climb: script stops within one heartbeat "
          f"({fake.sp_after_takeover} setpoints after), sends no mode change  OK")


def test_tracking_error_aborts_to_hold():
    fake = GroundPX4(freeze_after=1.5)               # vehicle stops following 1.5 s into OFFBOARD
    rc, log, s = _run(fake)
    assert rc == 1 and "from the setpoint" in s["status"] and fake.mode == AUTO
    print("  drone stops following the climb: aborts to Hold  OK")


def test_lost_position_aborts_to_hold():
    fake = GroundPX4(drop_pos_after=1.0)
    rc, log, s = _run(fake)
    assert rc == 1 and "no position" in s["status"] and fake.mode == AUTO
    print("  position stream stops: aborts to Hold  OK")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        t()
    print(f"\nAll tests passed. ({len(tests)})")
