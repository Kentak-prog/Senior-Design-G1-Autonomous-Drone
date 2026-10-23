"""Self-tests. Run: python3 test_fly_min_snap_offboard.py

No pymavlink, no ROS, no vehicle. Covers the frame conversions, planning and
setpoints, and the whole OFFBOARD sequence (enter OFFBOARD, hover, stream the
trajectory, hold, AUTO.LOITER, aborts) against FakePX4 below: a stand-in for
a pymavlink master whose vehicle tracks the setpoints with a simple
first-order model in simulated time. It does NOT prove the real PX4 / MAVLink
link behaves the same way (untested against the cluster).
"""

import json
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fly_min_snap_offboard as fms

# The 3-gate course spawned in Isaac on 2026-10-23 and the drone's hover pose then.
GATES_3 = {
    "gates": [
        {"name": "gate_0", "x": 5.1644, "y": 0.5666, "z": 3.3667, "yaw_deg": 7.909, "side": 1.5},
        {"name": "gate_1", "x": 8.6018, "y": -3.2453, "z": 3.6556, "yaw_deg": -22.091, "side": 1.5},
        {"name": "gate_2", "x": 13.0455, "y": -2.0747, "z": 3.7681, "yaw_deg": -7.091, "side": 1.5},
    ],
    "camera_pose_usd": {"position": [0.2585, -0.0359, 2.6123]},
}
START_ENU = np.array([-0.0469, -0.0367, 2.5484])
HEADING_ENU_DEG = -7.24

OFFBOARD = fms.PX4_MAIN_OFFBOARD
POSCTL = 3
AUTO = 4


# --------------------------------------------------------------------------
# Fake PX4 (pymavlink master stand-in) with its own simulated clock
# --------------------------------------------------------------------------

class _Msg:
    def __init__(self, kind, **kw):
        self._kind = kind
        self.__dict__.update(kw)

    def get_type(self):
        return self._kind

    def get_srcSystem(self):
        return 1


class FakePX4:
    """Vehicle tracks OFFBOARD setpoints: v = v_ff + gain * (p_sp - p). Emits
    LOCAL_POSITION_NED + ATTITUDE at 50 Hz and HEARTBEAT at 2 Hz."""

    target_system = 1
    target_component = 1

    def __init__(self, start_enu=START_ENU, heading_enu_deg=HEADING_ENU_DEG, gain=4.0,
                 freeze_after=None, mode_switch_after=None):
        self.now = 1000.0
        self.pos = fms.enu_to_ned(start_enu)
        self.vel = np.zeros(3)
        self.yaw = fms.yaw_enu_to_ned(math.radians(heading_enu_deg))
        self.mode = POSCTL if start_enu[2] > 0.5 else 0
        self.armed = start_enu[2] > 0.5
        self.gain = gain
        self.sp = None
        self.sp_times = []
        self.offboard_since = None
        self.freeze_after = freeze_after          # vehicle stops responding N s after OFFBOARD
        self.mode_switch_after = mode_switch_after  # "pilot" switches to POSCTL N s after OFFBOARD
        self.loiter_commands = 0
        self.offboard_commands = 0
        self.queue = []
        self.next_pos_msg = self.now
        self.next_hb = self.now
        self.mav = self

    # clock --------------------------------------------------------------
    def clock(self):
        return self.now

    def sleep(self, dt):
        dt = max(float(dt), 0.0)
        steps = max(1, int(round(dt / 0.005))) if dt > 0 else 0
        for _ in range(steps):
            self._step(dt / steps)

    def _step(self, h):
        self.now += h
        in_off = self.mode == OFFBOARD
        t_off = None if self.offboard_since is None else self.now - self.offboard_since
        if in_off and self.mode_switch_after is not None and t_off > self.mode_switch_after:
            self.mode = POSCTL
            in_off = False
        frozen = self.freeze_after is not None and t_off is not None and t_off > self.freeze_after
        if in_off and self.sp is not None and not frozen:
            p_sp, v_ff, yaw = self.sp
            self.vel = v_ff + self.gain * (p_sp - self.pos)
            self.yaw = yaw
        else:
            self.vel = np.zeros(3)
        self.pos = self.pos + self.vel * h

    # pymavlink master API ----------------------------------------------
    def recv_match(self, blocking=False):
        if not self.queue:
            if self.now >= self.next_pos_msg:
                self.queue.append(_Msg("LOCAL_POSITION_NED", x=self.pos[0], y=self.pos[1], z=self.pos[2],
                                       vx=self.vel[0], vy=self.vel[1], vz=self.vel[2]))
                self.queue.append(_Msg("ATTITUDE", yaw=self.yaw))
                self.next_pos_msg = self.now + 0.02
            if self.now >= self.next_hb:
                self.queue.append(_Msg("HEARTBEAT", base_mode=128 if self.armed else 0,
                                       custom_mode=self.mode << 16))
                self.next_hb = self.now + 0.5
        return self.queue.pop(0) if self.queue else None

    # mav.* senders
    def set_position_target_local_ned_send(self, t_boot, ts, tc, frame, mask, x, y, z,
                                           vx, vy, vz, ax, ay, az, yaw, yaw_rate):
        assert frame == fms.MAV_FRAME_LOCAL_NED
        v_ff = np.zeros(3) if mask & fms.IGN_VX else np.array([vx, vy, vz])
        self.sp = (np.array([x, y, z]), v_ff, yaw)
        self.sp_times.append(self.now)

    def command_long_send(self, ts, tc, cmd, conf, p1, p2, p3, *rest):
        if cmd == 176 and int(p2) == OFFBOARD:
            self.offboard_commands += 1
            recent = [t for t in self.sp_times if self.now - t <= 1.0]
            if len(recent) >= 10 and self.mode != OFFBOARD:      # PX4 wants setpoints streaming first
                self.mode = OFFBOARD
                self.offboard_since = self.now
        elif cmd == 176 and int(p2) == AUTO:
            self.loiter_commands += 1
            self.mode = AUTO


def _args(tmp, extra=()):
    path = os.path.join(tmp, "test_gates.json")
    with open(path, "w") as f:
        json.dump(GATES_3, f)
    return fms.build_parser().parse_args(["--gates", path, "--out-dir", os.path.join(tmp, "run"),
                                          "--hover-s", "1.0", *extra])


def _fly(fake, extra=("--yes",), ask=lambda _: "y"):
    tmp = tempfile.mkdtemp()
    args = _args(tmp, extra)
    link = fms.PX4Link(fake, clock=fake.clock)
    rc = fms.run(args, link=link, clock=fake.clock, sleep=fake.sleep, ask=ask)
    return rc, args.out_dir


def _quiet(fn, *a, **kw):
    """Run fn with stdout silenced (the planner and the flight are chatty)."""
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **kw)
    return out, buf.getvalue()


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_import_without_pymavlink():
    assert "pymavlink" not in sys.modules
    assert fms.TYPE_MASK_POS_VEL_ACC_YAW == 2048 and fms.TYPE_MASK_HOLD == 2552
    print("  imports without pymavlink; type masks 2048 / 2552  OK")


def test_yaw_conversions():
    cases = [(0.0, 90.0), (90.0, 0.0), (180.0, -90.0), (-90.0, 180.0), (-7.24, 97.24)]
    for enu, ned in cases:
        got = math.degrees(fms.yaw_enu_to_ned(math.radians(enu)))
        assert abs(((got - ned) + 180) % 360 - 180) < 1e-9, (enu, ned, got)
    for a in np.linspace(-3.1, 3.1, 25):
        assert abs(fms.wrap_pi(fms.yaw_ned_to_enu(fms.yaw_enu_to_ned(a)) - a)) < 1e-12
    print("  ENU<->NED yaw (east=90 deg NED, north=0, round trip)  OK")


def test_position_conversion():
    p = np.array([1.0, 2.0, 3.0])
    assert np.allclose(fms.enu_to_ned(p), [2.0, 1.0, -3.0])
    assert np.allclose(fms.ned_to_enu(fms.enu_to_ned(p)), p)
    print("  ENU (x east, y north, z up) -> NED (north, east, down)  OK")


def _plan(v_max=2.0, a_max=2.0):
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "g.json")
    with open(path, "w") as f:
        json.dump(GATES_3, f)
    gates = fms.gates_from_manifest(path)
    (traj, wps, ref), _ = _quiet(fms.plan_course, gates, START_ENU, math.radians(HEADING_ENU_DEG), v_max, a_max)
    return gates, traj, ref


def test_plan_limits_and_gates():
    gates, traj, _ = _plan()
    v, a = fms.plan_peaks(traj)
    assert v <= 2.0 * 1.03 and a <= 2.0 * 1.03, (v, a)
    ok, lines = fms.planned_report(traj, gates)
    assert ok, lines
    print(f"  plan from the hover pose: {traj.duration:.1f} s, peak {v:.2f} m/s / {a:.2f} m/s^2, "
          f"all 3 gates  OK")


def test_setpoints():
    _, traj, ref = _plan()
    sp = fms.Setpoints(traj, ref)
    s0 = sp.at(0.0)
    assert np.allclose(s0["pos_enu"], START_ENU, atol=1e-6)
    assert np.allclose(s0["vel_enu"], 0.0, atol=1e-6)
    assert abs(math.degrees(s0["yaw_enu"]) - HEADING_ENU_DEG) < 0.5
    assert np.allclose(s0["pos_ned"], fms.enu_to_ned(START_ENU), atol=1e-6)
    end = sp.at(sp.duration + 5.0)
    assert np.allclose(end["vel_enu"], 0.0) and np.allclose(end["acc_enu"], 0.0)
    assert np.allclose(end["pos_enu"], traj.evaluate(traj.duration, 0))
    ts = np.arange(0.0, sp.duration, 0.02)
    p = np.array([sp.at(t)["pos_enu"] for t in ts])
    assert np.max(np.linalg.norm(np.diff(p, axis=0), axis=1)) < 2.1 * 0.02 * 1.05   # no jumps at 2 m/s
    print("  setpoints start at the hover pose, hold the end, no jumps  OK")


def test_full_flight_against_fake_px4():
    fake = FakePX4()
    (rc, out_dir), log = _quiet(_fly, fake)
    assert rc == 0, log
    with open(os.path.join(out_dir, "summary.json")) as f:
        s = json.load(f)
    assert s["status"] == "completed" and s["gates_passed"], s
    assert s["err_max_m"] < 0.5, s
    assert fake.loiter_commands >= 1 and fake.mode == AUTO
    for name in ("plan_reference.csv", "flight_log.csv", "summary.json", "test_gates.json"):
        assert os.path.exists(os.path.join(out_dir, name)), name
    rate = len(fake.sp_times) / (fake.sp_times[-1] - fake.sp_times[0])
    assert 45.0 < rate < 55.0, rate
    print(f"  fake PX4: OFFBOARD -> trajectory -> LOITER, all gates passed, "
          f"err max {s['err_max_m']:.2f} m, setpoints {rate:.0f} Hz  OK")


def test_abort_on_tracking_error():
    fake = FakePX4(freeze_after=4.0)          # 1 s hover, then ~3 s into the trajectory it stops
    (rc, out_dir), log = _quiet(_fly, fake)
    with open(os.path.join(out_dir, "summary.json")) as f:
        s = json.load(f)
    assert rc == 1 and s["status"].startswith("aborted: tracking error"), s["status"]
    assert fake.loiter_commands >= 1
    print(f"  vehicle stops responding -> '{s['status'][:40]}...' + LOITER  OK")


def test_abort_when_px4_leaves_offboard():
    fake = FakePX4(mode_switch_after=3.0)
    (rc, out_dir), _ = _quiet(_fly, fake)
    with open(os.path.join(out_dir, "summary.json")) as f:
        s = json.load(f)
    assert rc == 1 and s["status"].startswith("aborted: PX4 left OFFBOARD"), s["status"]
    print("  mode switched away mid-flight -> abort + LOITER  OK")


def test_refuses_on_ground():
    fake = FakePX4(start_enu=np.array([0.0, 0.0, 0.1]))
    (rc, _), log = _quiet(_fly, fake)
    assert rc == 1 and "not airborne" in log and fake.offboard_commands == 0
    print("  refuses when not airborne (no OFFBOARD request)  OK")


def test_declining_does_not_fly():
    fake = FakePX4()
    (rc, _), log = _quiet(_fly, fake, extra=(), ask=lambda _: "n")
    assert rc == 0 and "not flying" in log and fake.offboard_commands == 0 and not fake.sp_times
    print("  answering 'n' sends nothing  OK")


def test_plan_only_needs_no_px4():
    tmp = tempfile.mkdtemp()
    args = _args(tmp, ["--plan-only", "--start", *map(str, START_ENU), "--heading", str(HEADING_ENU_DEG)])
    rc, _ = _quiet(fms.run, args)
    assert rc == 0 and os.path.exists(os.path.join(args.out_dir, "plan_reference.csv"))
    assert "pymavlink" not in sys.modules
    print("  --plan-only writes the plan without pymavlink  OK")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        t()
    print(f"\nAll tests passed. ({len(tests)})")
