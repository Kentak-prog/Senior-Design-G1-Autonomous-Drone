"""Self-tests for the Isaac/PX4 bridge. Run: python test_isaac_bridge.py (from rl/isaac/
or rl/); no pytest, no pymavlink, no Isaac.

Expectations are hand-derived (NED/FRD <-> ENU/FLU cases worked out on paper in
the comments, MAVLink numbers written out as literals), not round trips only.
The end-to-end tests fly the real PID / RL policy through run_in_vehicle on
SimVehicle, which keeps its state in PX4's NED/FRD conventions, so every sign
in frames.py is exercised; a deliberately wrong conversion must make them fail.
"""

import json
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.dirname(HERE)
ROOT = os.path.dirname(RL)
sys.path.insert(0, HERE)
sys.path.insert(0, RL)
sys.path.insert(0, os.path.join(ROOT, "workspace"))
sys.path.insert(0, os.path.join(ROOT, "vision", "gate_error_eval"))

import numpy as np
import torch

import frames
import vehicle as vmod
from vehicle import MavlinkVehicle, SimVehicle, LinkTimeout, VehicleState
from frames import ThrustMap
import runner
from runner import run_in_vehicle, Abort, edge_margins
import calibrate
from course import Course, Gate, TRACKS_DIR, YAW_CONVENTION
from drone_params import QuadParams
from fly_course import run_controller, make_controller
from pid_controller import PidController

P = QuadParams()
S = math.sqrt(0.5)
BEST = os.path.join(RL, "runs", "best")


def close(a, b, tol=1e-9):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol)


def body_axis_world(q_enu_flu, axis):
    return frames.quat_to_rotmat(np.asarray(q_enu_flu))[:, axis]


# ---------------------------------------------------------------- frames
def test_vectors():
    assert close(frames.ned_to_enu([1, 2, 3]), [2, 1, -3])
    assert close(frames.enu_to_ned([2, 1, -3]), [1, 2, 3])
    v = np.array([[1, 2, 3], [-4, 5, 0.5]])
    assert close(frames.ned_to_enu(v), [[2, 1, -3], [5, -4, -0.5]])
    assert close(frames.enu_to_ned(frames.ned_to_enu(v)), v)
    print("  NED (1,2,3) -> ENU (2,1,-3), batched, inverse  OK")


def test_quat_level_north_and_east():
    # Level, nose north: NED/FRD identity. In ENU the nose points world +y
    # (north), i.e. yaw = +90 deg CCW from east; body z (up) stays up.
    q = frames.quat_ned_frd_to_enu_flu([1, 0, 0, 0])
    assert q[0] >= 0 and close(q, [S, 0, 0, S]), q
    assert close(body_axis_world(q, 0), [0, 1, 0])
    assert close(body_axis_world(q, 1), [-1, 0, 0])     # left of a north-facing drone is west
    assert close(body_axis_world(q, 2), [0, 0, 1])
    # Facing east: NED yaw +90 deg (clockwise from north) -> ENU yaw 0, identity.
    q_ned = [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]
    q = frames.quat_ned_frd_to_enu_flu(q_ned)
    assert close(q, [1, 0, 0, 0]), q
    assert close(body_axis_world(q, 0), [1, 0, 0])
    assert abs(frames.yaw_of_quat_enu(q)) < 1e-9
    print("  north -> ENU yaw +90 deg, east -> ENU yaw 0  OK")


def test_quat_pitch_roll():
    # Nose-up pitch of +10 deg about FRD y (right-hand rule: x -> -z_frd = up).
    a = math.radians(10)
    q = frames.quat_ned_frd_to_enu_flu([math.cos(a / 2), 0, math.sin(a / 2), 0])
    nose = body_axis_world(q, 0)
    assert nose[2] > 0 and close(nose, [0, math.cos(a), math.sin(a)]), nose   # still facing north
    # Roll right (+15 deg about FRD x: y_frd -> z_frd, right side down). The
    # FLU y axis points LEFT, so it goes UP: z component +sin(15), still west-ish.
    r = math.radians(15)
    q = frames.quat_ned_frd_to_enu_flu([math.cos(r / 2), math.sin(r / 2), 0, 0])
    left = body_axis_world(q, 1)
    assert left[2] > 0 and close(left, [-math.cos(r), 0, math.sin(r)]), left
    print("  nose-up pitch -> body x gets +z; roll right -> body y (left) gets +z  OK")


def test_rates_and_roundtrip():
    assert close(frames.rates_frd_to_flu([1, 2, 3]), [1, -2, -3])
    assert close(frames.rates_flu_to_frd([1, -2, -3]), [1, 2, 3])
    # nose-up (positive FRD pitch rate) is a negative rotation about the LEFT axis
    assert frames.rates_frd_to_flu([0, 0.5, 0])[1] < 0
    rng = np.random.default_rng(0)
    q = rng.standard_normal((200, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    back = frames.quat_ned_frd_to_enu_flu(frames.quat_enu_flu_to_ned_frd(q))
    canon = np.where(q[:, :1] < 0, -q, q)
    assert close(back, canon, 1e-12) and (back[:, 0] >= 0).all()
    # independent check against the matrix definition R_enu_flu = M R_ned_frd F
    for qi in q[:20]:
        R_ned = frames.quat_to_rotmat(qi)
        R_enu = frames.quat_to_rotmat(frames.quat_ned_frd_to_enu_flu(qi))
        assert close(R_enu, frames.M_NED_TO_ENU @ R_ned @ np.diag([1, -1, -1]), 1e-12)
    print("  rates (1,2,3) -> (1,-2,-3); 200 random quats round trip + match M R F  OK")


def test_yaw():
    for enu, ned in [(0.0, math.pi / 2), (math.pi / 2, 0.0), (math.pi, -math.pi / 2)]:
        d = frames.wrap_pi(frames.yaw_enu_to_ned(enu) - ned)
        assert abs(d) < 1e-12, (enu, ned)
    south = frames.yaw_enu_to_ned(-math.pi / 2)          # facing south = NED yaw 180 deg
    assert abs(abs(south) - math.pi) < 1e-12
    for y in np.linspace(-3, 3, 13):
        assert abs(frames.wrap_pi(frames.yaw_ned_to_enu(frames.yaw_enu_to_ned(y)) - y)) < 1e-12
    print("  yaw_ned = pi/2 - yaw_enu (east 0 <-> 90, north 90 <-> 0)  OK")


# ---------------------------------------------------------------- thrust map
def test_thrust_map():
    tm = ThrustMap("model")
    # hand: T = 1.5*9.81 = 14.715 N; w = sqrt(14.715 / (4*8.54858e-6)) = 656.0 rad/s;
    # u = (656.0 - 100)/1000 = 0.556
    assert abs(tm.to_px4(14.715) - 0.556) < 5e-4, tm.to_px4(14.715)
    assert abs(tm.from_px4(0.556) - 14.715) < 0.05
    assert tm.to_px4(0.0) == 0.0 and tm.to_px4(-3.0) == 0.0 and tm.to_px4(1e4) == 1.0
    assert abs(tm.from_px4(0.0) - 4 * 8.54858e-6 * 100.0 ** 2) < 1e-9     # 0.3419 N at u = 0
    assert tm.from_px4(1.7) == tm.from_px4(1.0) and tm.from_px4(-1) == tm.from_px4(0)
    lin = ThrustMap("linear", u_hover=0.5)
    mg = 1.5 * 9.81
    assert abs(lin.to_px4(mg) - 0.5) < 1e-12 and abs(lin.to_px4(mg / 2) - 0.25) < 1e-12
    assert lin.to_px4(2.5 * mg) == 1.0 and abs(lin.from_px4(0.5) - mg) < 1e-9
    # scale: the real thrust is 1.1x the model, so asking for T needs the u of T/1.1
    s = ThrustMap("model", scale=1.1)
    assert abs(s.to_px4(1.1 * 14.715) - tm.to_px4(14.715)) < 1e-12
    assert abs(s.from_px4(0.556) - 1.1 * tm.from_px4(0.556)) < 1e-9
    ls = ThrustMap("linear", u_hover=0.5, scale=0.9)       # weaker than assumed: needs more u
    assert abs(ls.to_px4(mg) - 0.5 / 0.9) < 1e-12
    print("  u(m g) = 0.556, clipping, linear mode, scale  OK")


# ---------------------------------------------------------------- MAVLink
class FakeMsg:
    def __init__(self, t, sysid=1, **kw):
        self._t, self._sys = t, sysid
        self.__dict__.update(kw)

    def get_type(self):
        return self._t

    def get_srcSystem(self):
        return self._sys


class FakeMav:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def rec(*a, **k):
            self.calls.append((name, a))
        return rec


class FakeMaster:
    target_system, target_component = 1, 1

    def __init__(self):
        self.mav = FakeMav()
        self.queue = []

    def wait_heartbeat(self, timeout=None):
        pass

    def recv_match(self, blocking=False, timeout=None):
        return self.queue.pop(0) if self.queue else None

    def close(self):
        pass


def _pos_msg(ms, x=1.0, y=2.0, z=-3.0, vx=0.1, vy=0.2, vz=-0.3):
    return FakeMsg("LOCAL_POSITION_NED", time_boot_ms=ms, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)


def _att_msg(ms, q=(1, 0, 0, 0), r=(1.0, 2.0, 3.0)):
    return FakeMsg("ATTITUDE_QUATERNION", time_boot_ms=ms, q1=q[0], q2=q[1], q3=q[2], q4=q[3],
                   rollspeed=r[0], pitchspeed=r[1], yawspeed=r[2])


def test_mavlink_constants():
    assert vmod.TYPE_MASK_CTBR == 128                       # ATTITUDE_TARGET_TYPEMASK attitude-ignore
    # position + yaw only: ignore vx,vy,vz (8+16+32), ax,ay,az (64+128+256), yaw rate (2048)
    assert vmod.TYPE_MASK_POS_YAW == 2552 == 56 + 448 + 2048
    assert 2552 & (1 | 2 | 4 | 1024) == 0                   # x, y, z, yaw are NOT ignored
    assert (vmod.PX4_MAIN_OFFBOARD, vmod.PX4_MAIN_AUTO, vmod.PX4_SUB_AUTO_LOITER) == (6, 4, 3)
    assert (vmod.MSG_ID_HEARTBEAT, vmod.MSG_ID_ATTITUDE_QUATERNION,
            vmod.MSG_ID_LOCAL_POSITION_NED) == (0, 31, 32)
    assert (vmod.MAV_CMD_DO_SET_MODE, vmod.MAV_CMD_COMPONENT_ARM_DISARM,
            vmod.MAV_CMD_SET_MESSAGE_INTERVAL) == (176, 400, 511)
    # agree with the constants verified against the cluster in fly_gate_approach.py
    import fly_gate_approach as fga
    assert vmod.TYPE_MASK_POS_YAW == fga.TYPE_MASK_POS_YAW
    assert (vmod.PX4_MAIN_OFFBOARD, vmod.PX4_MAIN_AUTO, vmod.PX4_SUB_AUTO_LOITER) == \
        (fga.PX4_MAIN_OFFBOARD, fga.PX4_MAIN_AUTO, fga.PX4_SUB_AUTO_LOITER)
    assert vmod.MAV_CMD_DO_SET_MODE == fga.MAV_CMD_DO_SET_MODE
    assert vmod.MAV_CMD_SET_MESSAGE_INTERVAL == fga.MAV_CMD_SET_MESSAGE_INTERVAL
    assert vmod.MSG_ID_LOCAL_POSITION_NED == fga.MSG_ID_LOCAL_POSITION_NED
    assert vmod.MAV_FRAME_LOCAL_NED == fga.MAV_FRAME_LOCAL_NED
    print("  type masks 128 / 2552, custom modes 6 / (4,3), ids 0/31/32  OK")


def test_mavlink_vehicle_with_fake_master():
    m = FakeMaster()
    v = MavlinkVehicle(master=m, origin_offset_enu=(10, 20, 30), thrust_map=ThrustMap("model"))
    cmds = [a for n, a in m.mav.calls if n == "command_long_send"]
    # SET_MESSAGE_INTERVAL: (sys, comp, 511, confirm, msg id, interval us, ...)
    assert [(c[2], c[4], c[5]) for c in cmds] == [(511, 32, 10000.0), (511, 31, 10000.0),
                                                  (511, 0, 100000.0)]
    assert v.state() is None
    m.queue += [_pos_msg(5000)]
    v.poll()
    assert v.state() is None                                # attitude still missing
    m.queue += [_att_msg(5010)]
    v.poll()
    s = v.state()
    # NED (1,2,-3) -> ENU (2,1,3) + offset; vel NED (.1,.2,-.3) -> ENU (.2,.1,.3);
    # identity NED/FRD attitude -> yaw +90 deg; FRD rates (1,2,3) -> FLU (1,-2,-3)
    assert abs(s.t_sim - 5.01) < 1e-12
    assert close(s.pos, [12, 21, 33]) and close(s.vel, [0.2, 0.1, 0.3])
    assert close(s.quat, [S, 0, 0, S]) and close(s.omega, [1, -2, -3])
    assert not v.is_offboard
    m.queue += [FakeMsg("HEARTBEAT", base_mode=128 | 1, custom_mode=6 << 16),
                FakeMsg("HEARTBEAT", sysid=255, base_mode=0, custom_mode=0)]   # other system ignored
    v.poll()
    assert v.is_offboard and v.armed

    m.mav.calls.clear()
    v.set_offboard()
    assert m.mav.calls == [("command_long_send", (1, 1, 176, 0, 1, 6, 0, 0.0, 0.0, 0.0, 0.0))]
    m.mav.calls.clear()
    v.set_loiter()
    assert m.mav.calls == [("command_long_send", (1, 1, 176, 0, 1, 4, 3, 0.0, 0.0, 0.0, 0.0))] * 3
    m.mav.calls.clear()
    v.arm()
    assert m.mav.calls[0][1][2:5] == (400, 0, 1)

    # CTBR: thrust m g -> u = 0.556, FLU rates (0.1, 0.2, 0.3) -> FRD (0.1, -0.2, -0.3)
    m.mav.calls.clear()
    u = v.send_ctbr(14.715, [0.1, 0.2, 0.3])
    (name, a), = m.mav.calls
    assert name == "set_attitude_target_send" and abs(u - 0.556) < 5e-4
    assert a[3] == 128 and list(a[4]) == [1.0, 0.0, 0.0, 0.0]
    assert close(a[5:8], [0.1, -0.2, -0.3]) and abs(a[8] - u) < 1e-12

    # position: ENU (3,1,2.5) with offset (10,20,30) -> local ENU (-7,-19,-27.5)
    # -> NED (-19,-7,27.5); yaw ENU 0 (east) -> NED yaw pi/2
    m.mav.calls.clear()
    v.send_position([3, 1, 2.5], 0.0)
    (name, a), = m.mav.calls
    assert name == "set_position_target_local_ned_send"
    assert a[3:5] == (1, 2552)                                       # frame LOCAL_NED, mask
    assert close(a[5:8], [-19, -7, 27.5]) and abs(a[14] - math.pi / 2) < 1e-12
    print("  fake pymavlink master: intervals, conversions, mode / setpoint messages  OK")


def test_mavlink_step_timeout_and_params():
    m = FakeMaster()
    v = MavlinkVehicle(master=m, stale_timeout=0.2)
    m.queue += [_pos_msg(1000), _att_msg(1000)]
    assert v.step(0.5).t_sim == 1.0                         # already ahead: returns at once
    try:
        v.step(2.0)
        raise AssertionError("expected LinkTimeout")
    except LinkTimeout:
        pass
    m.queue += [FakeMsg("PARAM_VALUE", param_id="THR_MDL_FAC", param_value=0.0, param_type=9),
                FakeMsg("PARAM_VALUE", param_id=b"MC_AIRMODE\x00\x00", param_value=1.0,
                        param_type=6)]
    out = v.read_params(["THR_MDL_FAC", "MC_AIRMODE", "COM_OBL_RC_ACT"], timeout=0.1)
    assert out["THR_MDL_FAC"] == 0.0 and out["MC_AIRMODE"] == 1 and out["COM_OBL_RC_ACT"] is None
    # INT32 params are sent bytewise (documented guess): 1 -> the float with bits 0x00000001
    import struct
    m.mav.calls.clear()
    v.set_param("MC_AIRMODE", 1, timeout=0.01)
    name, a = m.mav.calls[0]
    assert name == "param_set_send" and a[2] == b"MC_AIRMODE" and a[4] == 6
    assert struct.unpack("<i", struct.pack("<f", a[3]))[0] == 1
    m.mav.calls.clear()
    v.set_param("MC_ROLLRATE_P", 0.2, timeout=0.01)
    assert m.mav.calls[0][1][3] == 0.2 and m.mav.calls[0][1][4] == 9
    print("  step() wall timeout, read_params / set_param encodings  OK")


# ---------------------------------------------------------------- SimVehicle
def test_sim_vehicle_basics():
    v = SimVehicle(P, start_pos=(0, 0, 3), origin_offset_enu=(5, -3, 0))
    # raw NED by hand: ENU (0,0,3) - offset (5,-3,0) = (-5,3,3) -> NED (3,-5,-3)
    assert close(v._pos_ned, [3, -5, -3]), v._pos_ned
    assert close(v._q_ned_frd, [S, 0, 0, S])                   # start yaw 0 = east = NED yaw 90 deg
    s = v.state()
    assert close(s.pos, [0, 0, 3], 1e-5) and close(s.quat, [1, 0, 0, 0], 1e-6) and s.t_sim == 100.0
    # offboard only after setpoints streamed; a dead stream drops back to loiter
    v.set_offboard()
    assert not v.is_offboard
    v.send_position([0, 0, 3], 0.0)
    v.set_offboard()
    assert v.is_offboard
    v.step(v.t + 0.2)
    assert v.is_offboard
    v.step(v.t + 1.0)                                          # no setpoints for 1 s
    assert not v.is_offboard
    # CTBR thrust goes through the plant map: u of m g hovers
    v2 = SimVehicle(P)
    for _ in range(25):
        v2.send_ctbr(P.hover_thrust(), [0, 0, 0])
        v2.set_offboard()
        v2.step(v2.t + P.control_dt)
    assert v2.is_offboard and abs(v2.state().pos[2] - 3.0) < 0.02
    print("  SimVehicle: NED/FRD raw state with offset, offboard gating, hover u  OK")


# ---------------------------------------------------------------- end to end
def _track(name):
    return Course.load(os.path.join(TRACKS_DIR, f"{name}.json"))


def _reference_lap(course, kind, model=None):
    _, s = run_controller(course, make_controller(kind, P, course, model), P)
    return s


def test_offline_pid_straight3():
    c = _track("straight3")
    ref = _reference_lap(c, "pid")
    log, s = run_in_vehicle(SimVehicle(P), c, PidController(P, c), P, max_time=30, verbose=False)
    assert s["completed"] and s["gates_passed"] == 3 and not s["aborted"], s
    # Same controller, same plant; the only difference is the start state (phase A settles
    # within 0.25 m / 0.25 m/s of the start pose instead of starting exactly on it, and the
    # PID integrators begin from a slightly different state). At ~2.4 m/s that is ~0.1 s per
    # 0.25 m, so 0.5 s is generous.
    assert abs(s["lap_time"] - ref["lap_time"]) <= 0.5, (s["lap_time"], ref["lap_time"])
    # edge margin helper agrees with compare.edge_margins on the same positions
    import compare
    race = log["phase"] == 1
    mine = edge_margins(log["pos"][race], c)
    theirs = compare.edge_margins({"pos": log["pos"][race][:, None, :]}, c)
    assert close(mine, theirs, 1e-9) and abs(s["min_margin"] - np.nanmin(mine)) < 1e-12
    assert s["min_margin"] > 0.3 and s["ctrl_ms_mean"] > 0 and s["real_time_factor"] > 0
    print(f"  PID straight3: lap {s['lap_time']:.2f} s vs fly_course {ref['lap_time']:.2f} s  OK")
    return s["lap_time"], ref["lap_time"]


def test_offline_pid_other_tracks_and_origin_offset():
    # a non-zero origin offset and another track exercise the offset sign end to end
    c = _track("oval6")
    ref = _reference_lap(c, "pid")
    v = SimVehicle(P, start_pos=(2, 1, 3), origin_offset_enu=(7.0, -4.0, 0.5))
    _, s = run_in_vehicle(v, c, PidController(P, c), P, max_time=60, verbose=False)
    assert s["completed"] and abs(s["lap_time"] - ref["lap_time"]) <= 0.5, (s, ref)
    print(f"  PID oval6 with origin offset: lap {s['lap_time']:.2f} s vs {ref['lap_time']:.2f} s  OK")


def test_offline_rl():
    if not os.path.isdir(BEST):
        print("  SKIPPED RL end-to-end: rl/runs/best is missing")
        return {}
    out = {}
    for name in ("straight3", "oval6"):
        c = _track(name)
        ref = _reference_lap(c, "rl", BEST)
        _, s = run_in_vehicle(SimVehicle(P), c, make_controller("rl", P, c, BEST), P,
                              max_time=40, verbose=False)
        assert s["completed"] and not s["aborted"], (name, s)
        # RL is more start-state sensitive than the PID (peak 9-10 m/s), so also allow 0.5 s
        assert abs(s["lap_time"] - ref["lap_time"]) <= 0.5, (name, s["lap_time"], ref["lap_time"])
        out[name] = (s["lap_time"], ref["lap_time"])
        print(f"  RL {name}: lap {s['lap_time']:.2f} s vs fly_course {ref['lap_time']:.2f} s  OK")
    return out


def _fails(sim_kwargs=None, controller="pid", track="straight3"):
    c = _track(track)
    ctl = make_controller(controller, P, c, BEST if controller == "rl" else None)
    _, s = run_in_vehicle(SimVehicle(P, **(sim_kwargs or {})), c, ctl, P, max_time=25,
                          start_timeout=25, verbose=False)
    return s


def test_wrong_conversions_are_caught():
    """The end-to-end test must NOT be insensitive to sign errors."""
    good = frames.rates_flu_to_frd
    try:
        # outbound pitch rate not negated
        frames.rates_flu_to_frd = lambda w: np.asarray(w, float) * np.array([1.0, 1.0, -1.0])
        s = _fails()
        assert not s["completed"], "PID survived a wrong pitch-rate sign"
    finally:
        frames.rates_flu_to_frd = good
    good = frames.quat_ned_frd_to_enu_flu
    try:
        # forgetting the FRD -> FLU flip F (q_M * q only)
        frames.quat_ned_frd_to_enu_flu = lambda q: frames._canon(frames._qmul(frames.Q_M, np.asarray(q, float)))
        s = _fails()
        assert not s["completed"], "PID survived a wrong attitude conversion"
    finally:
        frames.quat_ned_frd_to_enu_flu = good
    good = frames.yaw_enu_to_ned
    try:
        frames.yaw_enu_to_ned = lambda y: frames.wrap_pi(-np.asarray(y, float))   # sign slip
        # oval6 starts facing north: a wrong yaw setpoint is flagged by phase A's yaw check
        s = _fails(track="oval6")
        assert not s["completed"], "wrong yaw conversion went unnoticed"
    finally:
        frames.yaw_enu_to_ned = good
    # and with everything restored it flies again
    assert _fails()["completed"]
    if os.path.isdir(BEST):
        # (RL's observation of omega is NOT a good detector: flipping the inbound rate
        # signs leaves the oval6 lap time unchanged, the policy barely relies on it. The
        # outbound command path is what breaks it.)
        good = frames.rates_flu_to_frd
        try:
            frames.rates_flu_to_frd = lambda w: np.asarray(w, float) * np.array([1.0, 1.0, -1.0])
            s = _fails(controller="rl")
            assert not s["completed"], "RL survived a wrong outbound pitch-rate sign"
        finally:
            frames.rates_flu_to_frd = good
    print("  wrong rate / attitude / yaw conversions make the offline flights fail  OK")


def test_thrust_map_mismatch_visible():
    # controller believes scale 1.0, plant thrust is 25% weaker: 'hover' at u(m g) sinks
    v = SimVehicle(P, plant_thrust_map=ThrustMap("model", P, scale=0.75))
    for _ in range(50):
        v.send_ctbr(P.hover_thrust(), [0, 0, 0])
        v.set_offboard()
        v.step(v.t + P.control_dt)
    assert v.is_offboard and v.state().vel[2] < -1.0, v.state().vel
    print("  a thrust-map mismatch shows up as a sinking hover  OK")


# ---------------------------------------------------------------- runner aborts
class FakeVehicle:
    """Scripted vehicle: state k (tick index, 20 ms) comes from script(k)."""

    def __init__(self, script):
        self.script, self.calls, self.k = script, [], 0
        self.is_offboard = True

    def wait_state(self, timeout=0):
        return self._state()

    def _state(self):
        pos, quat = self.script(self.k)
        return VehicleState(self.k * 0.02, np.asarray(pos, float), np.zeros(3),
                            np.asarray(quat, float), np.zeros(3))

    def step(self, t_target):
        self.k = int(round(t_target / 0.02))
        return self._state()

    def send_position(self, p, yaw):
        self.calls.append("pos")

    def send_ctbr(self, T, r):
        self.calls.append("ctbr")
        return 0.5

    def set_offboard(self):
        self.calls.append("offboard")

    def set_loiter(self):
        self.calls.append("loiter")


class HoverCtl:
    command = "ctbr"

    def __init__(self, boom_at=None, exc=KeyboardInterrupt):
        self.n, self.boom_at, self.exc = 0, boom_at, exc

    def reset(self, n, *a, **k):
        pass

    def act(self, pos, vel, quat, gate_idx, dt, omega=None):
        self.n += 1
        if self.boom_at is not None and self.n == self.boom_at:
            raise self.exc("boom")
        return torch.tensor([14.715]), torch.zeros(1, 3)


STRAIGHT = _track("straight3")
START = [4.0, 0.0, 2.5]
LEVEL = [1.0, 0.0, 0.0, 0.0]


def _script(bad_pos=None, bad_quat=None, from_k=120):
    def f(k):
        if k >= from_k:
            return (bad_pos if bad_pos is not None else START, bad_quat or LEVEL)
        return START, LEVEL
    return f


def _abort(script, ctl=None, **kw):
    v = FakeVehicle(script)
    _, s = run_in_vehicle(v, STRAIGHT, ctl or HoverCtl(), P, verbose=False, **kw)
    assert v.calls[-1] == "loiter", v.calls
    return s, v


def test_runner_aborts():
    a85 = math.radians(85) / 2
    s, v = _abort(_script(bad_quat=[math.cos(a85), math.sin(a85), 0, 0]))
    assert s["crash_kind"] == "tilt" and s["crashed"] and not s["completed"]
    assert v.calls.count("loiter") == 1 and "pos" in v.calls and "ctbr" in v.calls
    s, _ = _abort(_script(bad_pos=[4.0, 0.0, 0.05]))
    assert s["crash_kind"] == "ground" and s["crashed"]
    s, _ = _abort(_script(bad_pos=[7.0, 0.0, 2.5]), geofence=([-5, -5, 0], [6, 5, 10]))
    assert s["crash_kind"] == "geofence" and s["aborted"] and not s["crashed"]
    s, _ = _abort(_script(bad_pos=[100.0, 0.0, 2.5]))
    assert s["crash_kind"] == "oob" and s["crashed"]
    s, _ = _abort(_script(), max_time=1.0)
    assert s["crash_kind"] == "timeout" and not s["crashed"] and s["aborted"]
    # gate collision: crossing gate 0's plane (x = 8) at y = 0.8 (opening shrunk by the
    # 0.3 m radius is |y| <= 0.45; frame outer edge 0.85) clips the frame
    def hit(k):
        return ([7.9, 0.8, 2.5] if k == 120 else [8.1, 0.8, 2.5] if k > 120 else START), LEVEL
    s, _ = _abort(hit)
    assert s["crash_kind"] == "gate" and s["crashed"]
    # Ctrl-C inside the controller, any other exception, link loss: always loiter
    s, _ = _abort(_script(), ctl=HoverCtl(boom_at=3))
    assert s["crash_kind"] == "interrupt" and s["aborted"]
    v = FakeVehicle(_script())
    try:
        run_in_vehicle(v, STRAIGHT, HoverCtl(boom_at=3, exc=RuntimeError), P, verbose=False)
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        assert v.calls[-1] == "loiter"
    v = FakeVehicle(_script())
    v.step = lambda t: (_ for _ in ()).throw(LinkTimeout("dead"))
    _, s = run_in_vehicle(v, STRAIGHT, HoverCtl(), P, verbose=False)
    assert s["crash_kind"] == "link" and v.calls[-1] == "loiter"
    # MPC (wrench) controllers are refused before the vehicle is touched
    class Wrench(HoverCtl):
        command = "wrench"
    v = FakeVehicle(_script())
    try:
        run_in_vehicle(v, STRAIGHT, Wrench(), P, verbose=False)
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError as e:
        assert "Phase 4" in str(e) and v.calls == []
    print("  aborts: tilt, ground, geofence, oob, timeout, gate, Ctrl-C, exception, link, wrench  OK")


def test_auto_geofence():
    # oval6 gates span x -10..10, y -5.2..5.2, z 2..3.5 (hand-read from tracks/oval6.json)
    lo, hi = runner.auto_geofence(Course.load(os.path.join(TRACKS_DIR, "oval6.json")))
    assert np.allclose(lo, (-18.0, -13.2, 0.5), atol=0.05), lo
    assert np.allclose(hi, (18.0, 13.2, 7.5), atol=0.05), hi
    print("  auto_geofence(oval6) = track box + 8 m, z 0.5 .. top + 4  OK")


def test_runner_finish_holds_then_loiters():
    def fly(k):          # gate-sized steps along +x: crosses all three gates head on
        x = 4.0 + 0.2 * max(0, k - 60)
        return [min(x, 40.0), 0.0, 2.5], LEVEL
    v = FakeVehicle(fly)
    # preroll off: this fake scripts positions by tick count, so extra ticks shift it
    log, s = run_in_vehicle(v, STRAIGHT, HoverCtl(), P, verbose=False, hold_s=1.0, preroll_s=0)
    assert s["completed"] and s["gates_passed"] == 3 and s["crash_kind"] == ""
    assert v.calls[-1] == "loiter" and v.calls[-2] == "pos"
    assert v.calls[-1 - 50:-1] == ["pos"] * 50                 # 1 s of position hold at 50 Hz
    assert (log["phase"] == 2).sum() == 50
    print("  completion: 1 s position hold, then loiter  OK")


def test_log_csv_roundtrip():
    c = _track("straight3")
    log, s = run_in_vehicle(SimVehicle(P), c, PidController(P, c), P, max_time=1.0,
                            verbose=False)
    assert s["crash_kind"] == "timeout"
    with tempfile.TemporaryDirectory() as d:
        runner.write_log_csv(os.path.join(d, "log.csv"), log)
        back = np.genfromtxt(os.path.join(d, "log.csv"), delimiter=",", names=True)
        assert len(back) == len(log["t_sim"]) and close(back["px"], log["pos"][:, 0], 1e-6)
        assert np.allclose(back["u"], log["u"], atol=1e-6, equal_nan=True)   # NaN where no command
    print("  log.csv columns  OK")


# ---------------------------------------------------------------- calibration
def test_calibration_math():
    # hand: with scale 0.9 the real thrust is 0.9 * model, so az = 0.9 T_model / m - g
    us = [0.516, 0.536, 0.556, 0.576, 0.596]
    acc = [0.9 * P.px4_output_to_thrust(u) / P.mass - P.gravity for u in us]
    fit = calibrate.fit_hover(P, us, acc, [1.0] * 5)
    assert abs(fit["scale"] - 0.9) < 1e-9
    assert abs(P.px4_output_to_thrust(fit["u_hover_fit"]) * 0.9 - P.hover_thrust()) < 1e-6
    assert abs(fit["u_hover_zero_az"] - fit["u_hover_fit"]) < 2e-3
    # tilt correction: T cos(tilt) carries the weight
    fit2 = calibrate.fit_hover(P, [0.556], [0.0], [0.5])
    assert abs(fit2["scale"] - 2.0 * P.hover_thrust() / P.px4_output_to_thrust(0.556)) < 1e-9
    # rise time: first-order response 1 - exp(-t/0.05) crosses 63.2% at ~50 ms
    t = np.arange(0, 0.5, 0.02)
    w = 1.5 * (1 - np.exp(-t / 0.05))
    assert abs(calibrate.rise_time_63(t, w, 1.5) - 0.05) < 4e-3
    assert abs(calibrate.rise_time_63(t, -w, -1.5) - 0.05) < 4e-3
    assert math.isnan(calibrate.rise_time_63(t, 0.1 * w, 1.5))
    d = calibrate.doublet_commands(1.5, 0.02)
    assert len(d) == 51 and d[0] == 1.5 and d[12] == 1.5 and d[13] == -1.5 and d[25] == -1.5 \
        and d[26] == 0.0
    print("  fit_hover / rise_time_63 / doublet  OK")


def test_calibration_offline():
    import argparse
    out = {}
    for plant_scale in (1.0, 0.9):
        v = SimVehicle(P, plant_thrust_map=ThrustMap("model", P, scale=plant_scale))
        args = argparse.Namespace(alt=3.0, u=None, out_dir=None)
        fit = calibrate.cmd_hover(v, args, P)
        assert abs(fit["scale"] - plant_scale) < 0.01, (plant_scale, fit)
        out[plant_scale] = fit["scale"]
    v = SimVehicle(P)
    args = argparse.Namespace(alt=3.0, rate_rp=1.5, rate_yaw=1.0, out_dir=None)
    res = calibrate.cmd_rates(v, args, P)
    for name, rv, pv, rs, ps in res:
        assert abs(rv - rs) < 0.02 and abs(pv - ps) < 0.05 and pv > 0.9 * (1.5 if name != "yaw" else 1.0)
    # parameters: offline defaults
    assert calibrate.cmd_params(SimVehicle(P), None, P)["THR_MDL_FAC"] == 0.0
    print(f"  offline hover fit recovers plant scale {out}; rates match QuadSim  OK")


# ---------------------------------------------------------------- spawn_track
def test_spawn_track():
    import spawn_track as st
    import spawn_test_gates as sgt
    with open(os.path.join(TRACKS_DIR, "straight3.json")) as f:
        d = json.load(f)
    specs = st.track_gate_specs(d)
    assert [s["x"] for s in specs] == [8.0, 16.0, 24.0]
    assert all(s["z"] == 2.5 and s["yaw_deg"] == 0.0 and s["y"] == 0.0 for s in specs)
    assert [s["name"] for s in specs] == ["gate_0", "gate_1", "gate_2"]
    for name in ("straight3", "oval6", "slalom5"):
        with open(os.path.join(TRACKS_DIR, f"{name}.json")) as f:
            d = json.load(f)
        course = Course.from_dict(d)
        for g, s in zip(course.gates, st.track_gate_specs(d)):
            T = sgt._T_world_gate(s["x"], s["y"], s["z"], s["yaw_deg"])
            assert np.allclose(g.T_world_gate(), T, atol=1e-12), (name, g.name)
            assert s["side"] == sgt.GATE_SIDE and s["bar"] == sgt.BAR_THICKNESS
    man = st.build_manifest(d, st.track_gate_specs(d), d["yaw_convention"])
    assert set(man) == {"yaw_convention", "course_name", "gates"} and man["course_name"] == "slalom5"
    assert set(man["gates"][0]) == {"name", "x", "y", "z", "yaw_deg", "side"}
    assert man["yaw_convention"] == YAW_CONVENTION
    assert Course.from_dict(man).gates[1].x == course.gates[1].x     # a Course reads it back
    # shared builder is imported, not run; no omni needed for the pure part
    assert "omni" not in sys.modules and st.load_spawn_test_gates().ROOT_PATH == "/World/TestGates"
    assert st.TRACK_ROOT != sgt.ROOT_PATH
    assert st.resolve_track("oval6").endswith(os.path.join("rl", "tracks", "oval6.json"))
    os.environ.pop("SPAWN_TRACK", None)
    try:
        st.main()
        raise AssertionError("expected failure without SPAWN_TRACK")
    except RuntimeError as e:
        assert "SPAWN_TRACK" in str(e)
    print("  spawn_track: specs, T_world_gate == spawn_test_gates, manifest schema, fatal when unset  OK")


if __name__ == "__main__":
    for n in ["test_vectors", "test_quat_level_north_and_east", "test_quat_pitch_roll",
              "test_rates_and_roundtrip", "test_yaw", "test_thrust_map", "test_mavlink_constants",
              "test_mavlink_vehicle_with_fake_master", "test_mavlink_step_timeout_and_params",
              "test_sim_vehicle_basics", "test_offline_pid_straight3",
              "test_offline_pid_other_tracks_and_origin_offset", "test_offline_rl",
              "test_wrong_conversions_are_caught", "test_thrust_map_mismatch_visible",
              "test_runner_aborts", "test_auto_geofence", "test_runner_finish_holds_then_loiters",
              "test_log_csv_roundtrip", "test_calibration_math", "test_calibration_offline",
              "test_spawn_track"]:
        print(n)
        globals()[n]()
    print("\nAll tests passed.")
