"""
The vehicle abstraction the runner and the calibration tools talk to.

    Vehicle (base)     conversion logic shared by both implementations: raw PX4
                       data (NED / FRD) <-> rl/ conventions (ENU / FLU), thrust
                       mapping, world origin offset. Nothing in the subclasses
                       converts frames by itself.
    MavlinkVehicle     the real thing: pymavlink to PX4 SITL (lockstep with
                       Isaac Sim). pymavlink is imported lazily, only when a
                       connection string is given.
    SimVehicle         an offline stand-in on QuadSim that behaves like PX4 as
                       far as the runner can tell, INCLUDING the frame
                       conventions: it keeps its state as PX4 would report it
                       (NED position/velocity, FRD body quaternion and rates) and
                       the base class converts it back exactly as for the real
                       link. Its purpose is an end-to-end offline test of every
                       sign in frames.py: a wrong conversion makes the PID crash.

CONVENTIONS: rl/ side ENU / FLU / WXYZ body->world; PX4 side NED / FRD, see
    frames.py. world_enu = ned_to_enu(local_ned) + origin_offset_enu, where the
    offset is where PX4's EKF origin (roughly the spawn point on the ground)
    sits in Pegasus world coordinates; `calibrate.py origin` measures it.

TIME. t_sim is PX4's clock (time_boot_ms of ATTITUDE_QUATERNION). PX4 runs in
    lockstep with Isaac, so t_sim is SIM time and advances more slowly than the
    wall clock when Isaac is slow. Everything in the runner is paced on it:
    `step(t_target)` blocks until the vehicle's state has t_sim >= t_target.

MAVLINK NUMBERS (common.xml / PX4 docs; the ones fly_gate_approach.py already
    uses are verified against the cluster PX4, the rest are NOT):
    MAV_CMD_DO_SET_MODE 176, MAV_CMD_COMPONENT_ARM_DISARM 400,
    MAV_CMD_SET_MESSAGE_INTERVAL 511            (verified, fly_gate_approach)
    message ids HEARTBEAT 0, ATTITUDE_QUATERNION 31, LOCAL_POSITION_NED 32
                                                 (common.xml; 32 verified)
    SET_ATTITUDE_TARGET type_mask (ATTITUDE_TARGET_TYPEMASK, a SET bit means
        IGNORE): 1 roll rate, 2 pitch rate, 4 yaw rate, 64 throttle, 128
        attitude. CTBR = ignore attitude only = 128.  (verified 2026-10-20 in the
        cluster's PX4 v1.18 mavlink_receiver.cpp: attitude ignored + roll and
        pitch rates set + throttle set -> vehicle_rates_setpoint, OFFBOARD only)
    SET_POSITION_TARGET_LOCAL_NED type_mask 2552 = position + yaw only
                                                 (verified, fly_gate_approach)
    MAV_PARAM_TYPE INT32 = 6, REAL32 = 9                        (common.xml)
"""

import os
import struct
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))   # rl/

import frames  # noqa: E402
from frames import ThrustMap  # noqa: E402
from drone_params import QuadParams  # noqa: E402

# --- MAVLink / PX4 constants (restated, like fly_gate_approach.py) ---------
MAV_FRAME_LOCAL_NED = 1
MAV_CMD_DO_SET_MODE = 176
MAV_CMD_COMPONENT_ARM_DISARM = 400
MAV_CMD_SET_MESSAGE_INTERVAL = 511
MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1
MAV_MODE_FLAG_SAFETY_ARMED = 128
MSG_ID_HEARTBEAT = 0
MSG_ID_ATTITUDE_QUATERNION = 31
MSG_ID_LOCAL_POSITION_NED = 32
PX4_MAIN_AUTO = 4
PX4_MAIN_OFFBOARD = 6
PX4_SUB_AUTO_LOITER = 3
# SET_POSITION_TARGET_LOCAL_NED type_mask bits (set = ignore)
IGN_X, IGN_Y, IGN_Z = 1, 2, 4
IGN_VX, IGN_VY, IGN_VZ = 8, 16, 32
IGN_AX, IGN_AY, IGN_AZ = 64, 128, 256
IGN_FORCE, IGN_YAW, IGN_YAW_RATE = 512, 1024, 2048
TYPE_MASK_POS_YAW = (IGN_VX | IGN_VY | IGN_VZ | IGN_AX | IGN_AY | IGN_AZ
                     | IGN_YAW_RATE)                                  # = 2552
# SET_ATTITUDE_TARGET type_mask bits (set = ignore)
ATT_IGN_ROLL_RATE, ATT_IGN_PITCH_RATE, ATT_IGN_YAW_RATE = 1, 2, 4
ATT_IGN_THROTTLE, ATT_IGN_ATTITUDE = 64, 128
TYPE_MASK_CTBR = ATT_IGN_ATTITUDE                                     # = 128
MAV_PARAM_TYPE_INT32, MAV_PARAM_TYPE_REAL32 = 6, 9

# PX4 parameters stored as INT32 (everything else is sent as REAL32).
PARAM_INT_NAMES = {"MC_AIRMODE", "COM_OBL_RC_ACT", "COM_OBL_ACT", "COM_RC_IN_MODE", "COM_LOW_BAT_ACT",
                   "SYS_AUTOSTART", "MAV_TYPE"}


class LinkTimeout(TimeoutError):
    """No fresh state arrived from the vehicle within the wall-clock limit."""


@dataclass
class VehicleState:
    """All in rl/ conventions, float64. t_sim [s] is PX4's (sim) clock."""
    t_sim: float
    pos: np.ndarray      # ENU world, m
    vel: np.ndarray      # ENU world, m/s
    quat: np.ndarray     # WXYZ, body FLU -> world ENU
    omega: np.ndarray    # body FLU, rad/s


# --------------------------------------------------------------------------
class Vehicle:
    """Conversion layer. Subclasses fill the raw PX4-side fields and implement
    the `_emit_*` hooks; the public API below is identical for both."""

    def __init__(self, origin_offset_enu=(0.0, 0.0, 0.0), thrust_map=None):
        self.origin_offset_enu = np.asarray(origin_offset_enu, dtype=np.float64).reshape(3)
        self.thrust_map = thrust_map if thrust_map is not None else ThrustMap("model")
        # raw PX4 data, NED / FRD, as received
        self._t_att = None
        self._t_pos = None
        self._pos_ned = None
        self._vel_ned = None
        self._q_ned_frd = None
        self._rates_frd = None
        self.main_mode = None
        self.armed = False

    # --- reading -----------------------------------------------------------
    def poll(self):
        """Drain pending messages without blocking (no-op offline)."""

    def state(self):
        """Latest VehicleState, or None until both position and attitude have
        arrived. t_sim is the time of the latest ATTITUDE_QUATERNION; the
        position can be up to one message period older."""
        if self._t_att is None or self._t_pos is None:
            return None
        return VehicleState(
            t_sim=float(self._t_att),
            pos=frames.ned_to_enu(self._pos_ned) + self.origin_offset_enu,
            vel=frames.ned_to_enu(self._vel_ned),
            quat=frames.quat_ned_frd_to_enu_flu(self._q_ned_frd),
            omega=frames.rates_frd_to_flu(self._rates_frd),
        )

    def wait_state(self, timeout=10.0):
        """Block (wall clock) until the first full state exists."""
        t0 = time.time()
        while True:
            self.poll()
            s = self.state()
            if s is not None:
                return s
            if time.time() - t0 > timeout:
                raise LinkTimeout("no LOCAL_POSITION_NED + ATTITUDE_QUATERNION received")
            time.sleep(0.01)

    def step(self, t_target):
        """Return the state once the vehicle's clock has reached t_target."""
        raise NotImplementedError

    @property
    def is_offboard(self):
        return self.main_mode == PX4_MAIN_OFFBOARD

    # --- commanding (rl/ conventions in, PX4 conventions out) -----------------
    def send_position(self, pos_enu, yaw_enu):
        """Position + yaw setpoint, world ENU / rad CCW from east."""
        pos_ned = frames.enu_to_ned(np.asarray(pos_enu, dtype=np.float64) - self.origin_offset_enu)
        self._emit_position(pos_ned, float(frames.yaw_enu_to_ned(yaw_enu)))

    def send_ctbr(self, thrust_N, rates_flu):
        """Collective thrust [N] + body rates FLU [rad/s]. Returns the normalised
        thrust u that was sent."""
        u = self.thrust_map.to_px4(thrust_N)
        self.send_ctbr_raw(u, rates_flu)
        return u

    def send_ctbr_raw(self, u, rates_flu):
        """Like send_ctbr but with the normalised thrust u given directly,
        bypassing the thrust map (the hover calibration steps u itself)."""
        rates_frd = frames.rates_flu_to_frd(np.asarray(rates_flu, dtype=np.float64))
        self._emit_attitude_target(min(max(float(u), 0.0), 1.0), rates_frd)

    def set_offboard(self):
        raise NotImplementedError

    def set_loiter(self):
        raise NotImplementedError

    def arm(self):
        raise NotImplementedError

    def read_params(self, names, timeout=2.0):
        raise NotImplementedError

    def set_param(self, name, value, timeout=2.0):
        raise NotImplementedError

    def close(self):
        pass

    def _emit_position(self, pos_ned, yaw_ned):
        raise NotImplementedError

    def _emit_attitude_target(self, u, rates_frd):
        raise NotImplementedError


# --------------------------------------------------------------------------
# real PX4 over MAVLink
# --------------------------------------------------------------------------
def _decode_param(value, ptype):
    """PARAM_VALUE float -> python number. PX4 sends INT32 parameters bytewise
    (the int's bits reinterpreted as the float; MAV_PROTOCOL_CAPABILITY_
    PARAM_ENCODE_BYTEWISE) -- from memory, NOT verified. Guard: if the float
    already looks like a small whole number it is treated as a cast value."""
    if ptype != MAV_PARAM_TYPE_INT32:
        return float(value)
    v = float(value)
    if np.isfinite(v) and v == round(v) and abs(v) < 1e6 and v != 0.0:
        return int(v)
    return struct.unpack("<i", struct.pack("<f", v))[0]


def _encode_param(value, ptype):
    if ptype == MAV_PARAM_TYPE_INT32:
        return struct.unpack("<f", struct.pack("<i", int(value)))[0]
    return float(value)


class MavlinkVehicle(Vehicle):
    """pymavlink link to PX4. `master` can be injected (tests pass a fake);
    otherwise `connection` is opened with pymavlink (imported lazily)."""

    def __init__(self, connection="udpin:0.0.0.0:14540", origin_offset_enu=(0.0, 0.0, 0.0),
                 thrust_map=None, *, master=None, heartbeat_timeout=30.0, stream_hz=100.0,
                 stale_timeout=3.0, pos_slack=0.015, request_streams=True):
        super().__init__(origin_offset_enu, thrust_map)
        if master is None:
            from pymavlink import mavutil  # lazy: pure code must import without it
            master = mavutil.mavlink_connection(connection)
        self.m = master
        self.stale_timeout = stale_timeout
        # position may lag the attitude by up to one message period; waiting for
        # it to catch up to within pos_slack keeps pos and quat nearly simultaneous
        self.pos_slack = pos_slack
        self.m.wait_heartbeat(timeout=heartbeat_timeout)
        if request_streams:
            self.request_interval(MSG_ID_LOCAL_POSITION_NED, stream_hz)
            self.request_interval(MSG_ID_ATTITUDE_QUATERNION, stream_hz)
            self.request_interval(MSG_ID_HEARTBEAT, 10.0)   # fast mode feedback; PX4 may ignore

    # --- low level ----------------------------------------------------------
    def command(self, cmd, *params):
        p = list(params) + [0.0] * (7 - len(params))
        self.m.mav.command_long_send(self.m.target_system, self.m.target_component,
                                     cmd, 0, *p)

    def request_interval(self, msg_id, hz):
        self.command(MAV_CMD_SET_MESSAGE_INTERVAL, msg_id, 1e6 / hz)

    def _handle(self, msg):
        t = msg.get_type()
        if t == "LOCAL_POSITION_NED":
            self._pos_ned = np.array([msg.x, msg.y, msg.z], dtype=np.float64)
            self._vel_ned = np.array([msg.vx, msg.vy, msg.vz], dtype=np.float64)
            self._t_pos = msg.time_boot_ms * 1e-3
        elif t == "ATTITUDE_QUATERNION":
            self._q_ned_frd = np.array([msg.q1, msg.q2, msg.q3, msg.q4], dtype=np.float64)
            self._rates_frd = np.array([msg.rollspeed, msg.pitchspeed, msg.yawspeed],
                                       dtype=np.float64)
            self._t_att = msg.time_boot_ms * 1e-3
        elif t == "HEARTBEAT" and msg.get_srcSystem() == self.m.target_system:
            self.armed = bool(msg.base_mode & MAV_MODE_FLAG_SAFETY_ARMED)
            self.main_mode = (msg.custom_mode >> 16) & 0xFF
        return t

    def poll(self):
        while True:
            msg = self.m.recv_match(blocking=False)
            if msg is None:
                return
            self._handle(msg)

    def step(self, t_target):
        deadline = time.time() + self.stale_timeout
        while True:
            if (self._t_att is not None and self._t_pos is not None
                    and self._t_att >= t_target - 1e-9
                    and self._t_pos >= t_target - self.pos_slack):
                # drain whatever else is already buffered so a controller that
                # runs slower than PX4's clock acts on the freshest state, not a
                # growing backlog
                self.poll()
                return self.state()
            msg = self.m.recv_match(blocking=True, timeout=0.05)
            if msg is not None:
                if self._handle(msg) in ("ATTITUDE_QUATERNION", "LOCAL_POSITION_NED"):
                    deadline = time.time() + self.stale_timeout
            elif time.time() > deadline:
                raise LinkTimeout(f"no fresh state for {self.stale_timeout:.1f} s wall "
                                  f"(sim paused? t_target={t_target:.3f}, t_att={self._t_att})")

    # --- modes --------------------------------------------------------------
    def set_offboard(self):
        self.command(MAV_CMD_DO_SET_MODE, MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                     PX4_MAIN_OFFBOARD, 0)

    def set_loiter(self, repeat=3):
        """Sent several times back to back: UDP, and this is the safety exit."""
        for _ in range(repeat):
            self.command(MAV_CMD_DO_SET_MODE, MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                         PX4_MAIN_AUTO, PX4_SUB_AUTO_LOITER)

    def arm(self):
        self.command(MAV_CMD_COMPONENT_ARM_DISARM, 1)

    # --- setpoints ----------------------------------------------------------
    def _emit_position(self, pos_ned, yaw_ned):
        self.m.mav.set_position_target_local_ned_send(
            0, self.m.target_system, self.m.target_component, MAV_FRAME_LOCAL_NED,
            TYPE_MASK_POS_YAW, float(pos_ned[0]), float(pos_ned[1]), float(pos_ned[2]),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(yaw_ned), 0.0)

    def _emit_attitude_target(self, u, rates_frd):
        # q is ignored (ATTITUDE_IGNORE); body rates [rad/s] FRD and thrust in
        # [0, 1] are used. The cluster's pymavlink signature is (time_boot_ms,
        # target_system, target_component, type_mask, q, roll, pitch, yaw rate,
        # thrust) -- no thrust_body argument (checked 2026-10-20).
        self.m.mav.set_attitude_target_send(
            0, self.m.target_system, self.m.target_component, TYPE_MASK_CTBR,
            [1.0, 0.0, 0.0, 0.0], float(rates_frd[0]), float(rates_frd[1]),
            float(rates_frd[2]), float(u))

    # --- parameters (never called automatically) -------------------------------
    @staticmethod
    def param_type(name):
        return MAV_PARAM_TYPE_INT32 if name in PARAM_INT_NAMES else MAV_PARAM_TYPE_REAL32

    def _wait_param(self, name, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.m.recv_match(blocking=True, timeout=0.05)
            if msg is None:
                continue
            if self._handle(msg) == "PARAM_VALUE":
                pid = msg.param_id
                pid = pid.decode() if isinstance(pid, (bytes, bytearray)) else pid
                if pid.rstrip("\x00") == name:
                    return _decode_param(msg.param_value, msg.param_type)
        return None

    def read_params(self, names, timeout=2.0):
        """{name: value or None}. None = no PARAM_VALUE within `timeout` wall
        seconds (PX4 does not answer for parameters that do not exist)."""
        out = {}
        for name in names:
            self.m.mav.param_request_read_send(self.m.target_system, self.m.target_component,
                                               name.encode(), -1)
            out[name] = self._wait_param(name, timeout)
        return out

    def set_param(self, name, value, timeout=2.0):
        """PARAM_SET with REAL32 (default) or INT32 (PARAM_INT_NAMES). Returns the
        value PX4 echoes back, or None. Changing PX4 parameters is the user's
        call: nothing in this package invokes it."""
        ptype = self.param_type(name)
        self.m.mav.param_set_send(self.m.target_system, self.m.target_component,
                                  name.encode(), _encode_param(value, ptype), ptype)
        return self._wait_param(name, timeout)

    def close(self):
        try:
            self.m.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# offline stand-in
# --------------------------------------------------------------------------
# PX4 defaults, from memory, for read_params() when offline
SIM_PARAM_DEFAULTS = {
    "MC_ROLLRATE_MAX": 220.0, "MC_PITCHRATE_MAX": 220.0, "MC_YAWRATE_MAX": 200.0,
    "THR_MDL_FAC": 0.0, "MPC_THR_HOVER": 0.5, "COM_OF_LOSS_T": 1.0,
    "COM_OBL_RC_ACT": 0, "MC_ROLLRATE_P": 0.15, "MC_PITCHRATE_P": 0.15,
    "MC_YAWRATE_P": 0.2, "MC_AIRMODE": 0,
}


class _PositionHold:
    """Cascaded position -> velocity -> attitude -> body-rate controller on
    QuadSim state, the same structure as pid_controller.PidController but
    aimed at a fixed point. Stands in for PX4's own position controller."""

    def __init__(self, params, kp_pos=1.2, v_max=2.5, kp_vel=4.0, ki_vel=0.8,
                 kp_att=(10.0, 10.0, 3.0), max_tilt_deg=35.0, a_max=8.0):
        self.p = params
        self.kp_pos, self.v_max, self.kp_vel, self.ki_vel = kp_pos, v_max, kp_vel, ki_vel
        self.kp_att = torch.tensor(kp_att)
        self.tan_max = float(np.tan(np.radians(max_tilt_deg)))
        self.a_max = a_max
        self.vel_int = torch.zeros(1, 3)

    def reset(self):
        self.vel_int.zero_()

    def act(self, sim, target, yaw, dt):
        from rotations import quat_to_rotmat
        p = self.p
        target = torch.as_tensor(target, dtype=sim.dtype).reshape(1, 3)
        v_des = self.kp_pos * (target - sim.pos)
        sp = v_des.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        v_des = v_des * torch.clamp(self.v_max / sp, max=1.0)
        v_err = v_des - sim.vel
        self.vel_int = (self.vel_int + v_err * dt).clamp(-2.0, 2.0)
        a = self.kp_vel * v_err + self.ki_vel * self.vel_int
        an = a.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        a = a * torch.clamp(self.a_max / an, max=1.0)
        F = p.mass * a
        F[:, 2] = (F[:, 2] + p.mass * p.gravity).clamp_min(0.3 * p.mass * p.gravity)
        Fh = F[:, :2]
        Fh_n = Fh.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        Fh = Fh * torch.clamp(self.tan_max * F[:, 2:3] / Fh_n, max=1.0)
        F = torch.cat([Fh, F[:, 2:3]], dim=-1)
        R = quat_to_rotmat(sim.quat)
        thrust = (F * R[:, :, 2]).sum(-1).clamp(0.0, p.max_total_thrust())
        b3 = F / F.norm(dim=-1, keepdim=True)
        heading = torch.tensor([[np.cos(yaw), np.sin(yaw), 0.0]], dtype=sim.dtype)
        b2 = torch.cross(b3, heading, dim=-1)
        b2 = b2 / b2.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        b1 = torch.cross(b2, b3, dim=-1)
        Rd = torch.stack([b1, b2, b3], dim=-1)
        M = Rd.transpose(-1, -2) @ R
        S = 0.5 * (M - M.transpose(-1, -2))
        e_R = torch.stack([S[:, 2, 1], S[:, 0, 2], S[:, 1, 0]], dim=-1)
        rates = -self.kp_att * e_R
        mr = torch.tensor(p.max_rate, dtype=sim.dtype)
        return thrust, torch.maximum(torch.minimum(rates, mr), -mr)


class SimVehicle(Vehicle):
    """Offline PX4 stand-in on QuadSim (n_envs = 1). See the module docstring.

    Starts in "loiter" hovering at start_pos (like the real drone after
    `commander takeoff`). set_offboard() is only accepted while setpoints are
    streaming (last one younger than cmd_timeout, as in PX4); if the stream
    stops for longer, the vehicle drops back to loiter (offboard-loss failsafe).
    Position setpoints and loiter are flown by an internal position hold;
    CTBR setpoints go to QuadSim.step_ctbr.

    thrust_map is what the runner side believes (used in send_ctbr);
    plant_thrust_map is the "true" PX4+Pegasus mapping used to turn the
    received u into Newtons. They default to the same ThrustMap("model"); give
    them different scales to see a mapping error show up in the flight.
    `time` starts at t0 (not 0) so code that wrongly assumes t_sim starts at
    zero fails offline too.
    """

    def __init__(self, params=None, *, start_pos=(0.0, 0.0, 3.0), start_yaw=0.0,
                 origin_offset_enu=(0.0, 0.0, 0.0), thrust_map=None, plant_thrust_map=None,
                 t0=100.0, cmd_timeout=0.5):
        from quad_sim import QuadSim
        from rotations import quat_from_euler_zyx
        self.p = params if params is not None else QuadParams()
        super().__init__(origin_offset_enu,
                         thrust_map if thrust_map is not None else ThrustMap("model", self.p))
        self.plant_map = (plant_thrust_map if plant_thrust_map is not None
                          else ThrustMap("model", self.p))
        self.sim = QuadSim(1, self.p)
        z = torch.zeros(1)
        self.sim.reset(pos=torch.tensor([start_pos], dtype=torch.float32),
                       quat=quat_from_euler_zyx(z, z, z + float(start_yaw)))
        self.dt = self.p.control_dt
        self.t = float(t0)
        self.cmd_timeout = cmd_timeout
        self.main_mode = PX4_MAIN_AUTO      # loiter
        self.armed = True
        self._cmd = None                    # ("pos", pos_enu, yaw_enu) | ("ctbr", u, rates_frd)
        self._cmd_t = -1e9
        self._hold = _PositionHold(self.p)
        self._hold_target = (np.asarray(start_pos, dtype=np.float64), float(start_yaw))
        self.params = dict(SIM_PARAM_DEFAULTS)
        self.n_steps = 0
        self._update_raw()

    # --- PX4-side raw state, produced from QuadSim like PX4 would report it ----
    def _update_raw(self):
        pos_enu = self.sim.pos[0].double().numpy()
        vel_enu = self.sim.vel[0].double().numpy()
        self._pos_ned = frames.enu_to_ned(pos_enu - self.origin_offset_enu)
        self._vel_ned = frames.enu_to_ned(vel_enu)
        self._q_ned_frd = frames.quat_enu_flu_to_ned_frd(self.sim.quat[0].double().numpy())
        self._rates_frd = frames.rates_flu_to_frd(self.sim.omega[0].double().numpy())
        self._t_att = self._t_pos = self.t

    # --- hooks ---------------------------------------------------------------
    def _emit_position(self, pos_ned, yaw_ned):
        pos_enu = frames.ned_to_enu(pos_ned) + self.origin_offset_enu
        self._cmd = ("pos", pos_enu, float(frames.yaw_ned_to_enu(yaw_ned)))
        self._cmd_t = self.t

    def _emit_attitude_target(self, u, rates_frd):
        self._cmd = ("ctbr", float(u), np.array(rates_frd, dtype=np.float64))
        self._cmd_t = self.t

    def set_offboard(self):
        if self._cmd is not None and self.t - self._cmd_t <= self.cmd_timeout + 1e-9:
            self.main_mode = PX4_MAIN_OFFBOARD

    def set_loiter(self):
        if self.main_mode != PX4_MAIN_AUTO:
            self.main_mode = PX4_MAIN_AUTO
            self._start_hold_here()

    def arm(self):
        self.armed = True

    def _start_hold_here(self):
        from rotations import quat_to_rotmat
        R = quat_to_rotmat(self.sim.quat)[0]
        yaw = float(np.arctan2(R[1, 0].item(), R[0, 0].item()))
        self._hold_target = (self.sim.pos[0].double().numpy().copy(), yaw)
        self._hold.reset()

    # --- time ------------------------------------------------------------------
    def _advance(self):
        if self.main_mode == PX4_MAIN_OFFBOARD and self.t - self._cmd_t > self.cmd_timeout + 1e-9:
            self.main_mode = PX4_MAIN_AUTO           # offboard-loss failsafe -> loiter
            self._start_hold_here()
        kw = dict(dtype=self.sim.dtype)
        if self.main_mode == PX4_MAIN_OFFBOARD and self._cmd[0] == "ctbr":
            _, u, rates_frd = self._cmd
            thrust = self.plant_map.from_px4(u)
            rates = frames.rates_frd_to_flu(rates_frd)
            self.sim.step_ctbr(torch.tensor([thrust], **kw), torch.tensor(rates[None], **kw))
        else:
            if self.main_mode == PX4_MAIN_OFFBOARD:
                target, yaw = self._cmd[1], self._cmd[2]
            else:
                target, yaw = self._hold_target
            thrust, rates = self._hold.act(self.sim, target, yaw, self.dt)
            self.sim.step_ctbr(thrust, rates)
        self.n_steps += 1
        self.t += self.dt
        self._update_raw()

    def step(self, t_target):
        while self.t < t_target - 1e-9:
            self._advance()
        return self.state()

    # --- parameters ---------------------------------------------------------------
    def read_params(self, names, timeout=2.0):
        return {n: self.params.get(n) for n in names}

    def set_param(self, name, value, timeout=2.0):
        self.params[name] = value
        return value
