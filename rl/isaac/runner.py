"""
Fly an rl/ controller (PID or the RL policy) on a vehicle: PX4 in Isaac Sim, or
the offline SimVehicle. The controller is the SAME object fly_course.py flies in
QuadSim; only the plant behind it changes.

CONVENTIONS: rl/ conventions throughout (world ENU, body FLU, WXYZ body->world,
    batched torch tensors with N = 1 at the controller boundary). Conversions to
    and from PX4 live in vehicle.py / frames.py, never here.

PHASES
    A  position: stream position setpoints (>1 s) and switch to OFFBOARD if not
       already, then fly a rate-limited carrot to course.start_pose() (position
       + yaw). Done when |pos error| < start_tol, |vel| < start_vel_tol and the
       yaw error is small, all for settle_s of SIM time.
    B  race: controller.reset(1); one control tick every params.control_dt of SIM
       time (never wall time: PX4 is in lockstep with Isaac and Isaac may run
       slower than real time). Each tick: check the gate crossing between the
       previous and the current state with course.check_gate (exactly as
       fly_course.run_controller), safety aborts, then controller.act ->
       vehicle.send_ctbr.
    After the last gate: brake along a constant-deceleration path (hold_decel m/s^2)
       to a stop, hold there for hold_s, then AUTO.LOITER.
    On ANY abort or exception: AUTO.LOITER immediately (try/finally).

ABORT KINDS (summary["crash_kind"]; "" = none)
    gate (frame/bar hit), ground (z < ground_z), tilt (> max_tilt_deg), oob
    (> max_gate_dist from the target gate), geofence (outside the ENU box),
    timeout (race longer than max_time or start phase longer than start_timeout),
    interrupt (Ctrl-C), link (no fresh state from PX4), offboard (PX4 refused
    OFFBOARD). summary["crashed"] is True only for the physical ones (gate,
    ground, tilt, oob); summary["aborted"] is True for any non-empty kind.

LOG: one row per observed state (phase 0 = going to the start, 1 = race,
    2 = final hold): t_sim, t_wall, pos, vel, quat, omega, cmd thrust [N], cmd
    rates (FLU), u sent, gate_idx, controller compute ms. Commands are NaN in rows
    where none was computed.

Edge margin = half-side minus the larger of |x|, |y| of the plane crossing in the
    gate frame (distance from the crossing point to the opening edge), same
    definition as compare.edge_margins; >= collision_radius is needed to pass.

VISION MODE (gate_source=..., CLI --gates vision): the controller is built from
    a ROUGH PRIOR course and is told where the gates are by a GateMap that camera
    detections correct (vision_gate_map.py, gate_vision.py). The `course` argument stays
    the TRUE course: scoring (gate pass / collision), the safety "oob" check and
    the summary all use it; the controller only ever sees the map. Phase A flies
    to the PRIOR start pose. Each state (phase A and B): gate_source.step(state,
    gate_idx) (sim vision: observe, then drain detections, set progress, update
    the map); in phase B, when the map changed, course_update.update_course pushes
    it into the controller (the controller's own update_course if it has one, else
    adapters for PID / RL). The map's course is also pushed once just after
    controller.reset (phase A's detections). Extra log columns n_det and map_err (3-D error of the
    current target gate vs truth, logging only) and summary["vision"].

VIO MODE (vio=..., CLI --state vio, 2026-11-13): the CONTROLLER flies on the VIO
    (ros2_nodes/vio_node.py's /vio/odometry via vio_state.RosVioState): position, velocity
    and attitude from the VIO, body rates from PX4 (the IMU gyro). Its gate index advances on
    the VIO path. Everything else stays on PX4's estimate (sim GPS, ~truth): phase A, the
    pre-roll hold, scoring, margins and the safety checks. Before the race the runner holds
    the start until the VIO is fresh and within vio_start_err of PX4 (up to vio_wait_s);
    during it, abort kind "vio" when the odometry is older than vio_max_age (wall s) or the
    VIO is more than vio_max_err from PX4's estimate for vio_err_ticks ticks in a row. Extra
    log columns vio_px/py/pz, vio_err, vio_age_ms; summary["vio"].

CLI: python runner.py --track ../tracks/oval6.json --controller pid [--sim] ...
     python runner.py --track ../tracks/oval6.json --controller pid --gates vision [--sim --sim-vision]
"""

import argparse
import json
import math
import os
import sys
import time

# One thread per numeric library, set before numpy/torch load. The ros2-eval
# container has a 2-CPU cgroup quota but reports 20 cores, so torch, OpenCV and
# IPOPT's BLAS each started ~20 threads; CFS throttling then stalled the 50 Hz
# loop for seconds (MPC ticks of 1-3 s, 2026-10-24).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

torch.set_num_threads(1)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))   # rl/

import frames  # noqa: E402
from vehicle import (MavlinkVehicle, SimVehicle, LinkTimeout, VehicleState)  # noqa: E402
from frames import ThrustMap  # noqa: E402
from course import Course, gate_batch, check_gate  # noqa: E402
from drone_params import QuadParams  # noqa: E402
from course_update import update_course  # noqa: E402

LOG_KEYS = ("phase", "t_sim", "t_wall", "pos", "vel", "quat", "omega", "thrust_N",
            "rate_cmd", "u", "gate_idx", "ctrl_ms")
PHYSICAL_CRASHES = ("gate", "ground", "tilt", "oob")


class Abort(Exception):
    """Raised by the safety checks; `kind` becomes summary['crash_kind']."""

    def __init__(self, kind, msg=""):
        super().__init__(f"{kind}: {msg}" if msg else kind)
        self.kind, self.msg = kind, msg


# --------------------------------------------------------------------------
# safety
# --------------------------------------------------------------------------
class Safety:
    """Per-state abort conditions. geofence = (min_xyz, max_xyz) in ENU or None."""

    def __init__(self, ground_z=0.15, max_tilt_deg=80.0, max_gate_dist=30.0, geofence=None):
        self.ground_z, self.max_tilt = ground_z, math.radians(max_tilt_deg)
        self.max_gate_dist = max_gate_dist
        self.geofence = None if geofence is None else (np.asarray(geofence[0], float),
                                                       np.asarray(geofence[1], float))

    def check(self, s, gate_center=None):
        """Raise Abort if state `s` violates a limit."""
        if s.pos[2] < self.ground_z:
            raise Abort("ground", f"z = {s.pos[2]:.2f} m < {self.ground_z} m")
        tilt = float(frames.tilt_of_quat(s.quat))
        if tilt > self.max_tilt:
            raise Abort("tilt", f"tilt {math.degrees(tilt):.0f} deg")
        if self.geofence is not None:
            lo, hi = self.geofence
            if np.any(s.pos < lo) or np.any(s.pos > hi):
                raise Abort("geofence", f"pos {np.round(s.pos, 2)} outside [{lo}, {hi}]")
        if gate_center is not None:
            d = float(np.linalg.norm(s.pos - gate_center))
            if d > self.max_gate_dist:
                raise Abort("oob", f"{d:.1f} m from the target gate")


# --------------------------------------------------------------------------
# pacing and logging
# --------------------------------------------------------------------------
class Pacer:
    """Advance the vehicle's clock by one control period per call. The target
    accumulates (so the average rate is exactly 1/dt even though PX4's
    message timestamps are quantised) but never lags the latest state by more
    than half a period (a slow controller skips ahead instead of bursting)."""

    def __init__(self, vehicle, dt, s):
        self.vehicle, self.dt, self.t_next = vehicle, dt, s.t_sim

    def advance(self, s):
        self.t_next = max(self.t_next + self.dt, s.t_sim + 0.5 * self.dt)
        return self.vehicle.step(self.t_next)


class _Log:
    """extras=True (vision mode) adds the n_det and map_err columns; vio=True the
    vio_px, vio_py, vio_pz, vio_err, vio_age_ms columns."""

    VIO_NAMES = ("vio_px", "vio_py", "vio_pz", "vio_err", "vio_age_ms")

    def __init__(self, extras=False, vio=False):
        self.rows = []
        self.extra_rows = [] if extras else None
        self.extra = (0, np.nan)      # (n_det, map_err) stamped on the next rows
        self.vio_rows = [] if vio else None
        self.vio_extra = (np.nan,) * 5
        self.t_wall0 = time.perf_counter()

    def add(self, phase, s, gate_idx, thrust=np.nan, rates=(np.nan,) * 3, u=np.nan, ctrl_ms=np.nan):
        self.rows.append((phase, s.t_sim, time.perf_counter() - self.t_wall0, s.pos.copy(),
                          s.vel.copy(), s.quat.copy(), s.omega.copy(), thrust,
                          np.asarray(rates, dtype=np.float64), u, gate_idx, ctrl_ms))
        if self.extra_rows is not None:
            self.extra_rows.append(self.extra)
        if self.vio_rows is not None:
            self.vio_rows.append(self.vio_extra)

    def arrays(self):
        if not self.rows:
            out = {k: np.zeros(0) for k in LOG_KEYS}
        else:
            cols = list(zip(*self.rows))
            out = {k: np.array(c, dtype=np.float64) for k, c in zip(LOG_KEYS, cols)}
        if self.extra_rows is not None:
            ex = np.array(self.extra_rows, dtype=np.float64).reshape(-1, 2)
            out["n_det"], out["map_err"] = ex[:, 0], ex[:, 1]
        if self.vio_rows is not None:
            vx = np.array(self.vio_rows, dtype=np.float64).reshape(-1, 5)
            for i, k in enumerate(self.VIO_NAMES):
                out[k] = vx[:, i]
        return out


def write_log_csv(path, log):
    names = ["phase", "t_sim", "t_wall", "px", "py", "pz", "vx", "vy", "vz", "qw", "qx", "qy",
             "qz", "wx", "wy", "wz", "thrust_N", "rate_cmd_x", "rate_cmd_y", "rate_cmd_z", "u",
             "gate_idx", "ctrl_ms"]
    cols = [log["phase"], log["t_sim"], log["t_wall"], *log["pos"].T, *log["vel"].T,
            *log["quat"].T, *log["omega"].T, log["thrust_N"], *log["rate_cmd"].T, log["u"],
            log["gate_idx"], log["ctrl_ms"]]
    if "n_det" in log:
        names += ["n_det", "map_err"]
        cols += [log["n_det"], log["map_err"]]
    if "vio_err" in log:
        names += list(_Log.VIO_NAMES)
        cols += [log[k] for k in _Log.VIO_NAMES]
    data = np.column_stack(cols) if len(log["t_sim"]) else np.zeros((0, len(names)))
    np.savetxt(path, data, delimiter=",", header=",".join(names), comments="", fmt="%.9g")


def edge_margins(pos, course):
    """Per gate (in order): edge margin [m] at the first forward crossing of its
    plane inside the outer frame, NaN if never crossed. pos is (T,3) ENU.
    Same math as compare.edge_margins (which takes a (T,N,3) log)."""
    pos = np.asarray(pos, dtype=np.float64)
    out = []
    for g in course.gates:
        T = g.T_world_gate()
        local = (pos - T[:3, 3]) @ T[:3, :3]
        z = local[:, 2]
        m = np.nan
        for k in np.nonzero((z[:-1] < 0) & (z[1:] >= 0))[0]:
            a = z[k] / (z[k] - z[k + 1])
            xy = local[k, :2] + a * (local[k + 1, :2] - local[k, :2])
            if np.abs(xy).max() <= g.side / 2 + g.bar:
                m = g.side / 2 - np.abs(xy).max()
                break
        out.append(m)
    return np.array(out)


# --------------------------------------------------------------------------
# phase A (also used by calibrate.py)
# --------------------------------------------------------------------------
def braking_setpoints(pos, vel, dt, *, decel=1.5, hold_s=1.0):
    """Position setpoints (ENU) that bring a drone at `pos` moving with `vel` to
    rest at constant deceleration `decel` [m/s^2], then hold the stop point for
    `hold_s`. One setpoint per control tick `dt`."""
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    speed = float(np.linalg.norm(vel))
    t_stop = speed / decel if speed > 1e-6 else 0.0
    out = []
    for k in range(1, int(math.ceil(t_stop / dt)) + 1):
        t = min(k * dt, t_stop)
        out.append(pos + vel * t - 0.5 * decel * t * t * vel / speed)
    stop = pos + 0.5 * vel * t_stop
    out += [stop] * int(round(hold_s / dt))
    return out


def goto(vehicle, params, pos, yaw, *, speed=2.0, tol=0.25, vel_tol=0.25, yaw_tol=0.15,
         settle_s=1.0, timeout=60.0, stream_s=1.2, offboard_tries=5, on_state=None):
    """Fly to (pos ENU, yaw ENU) with position setpoints and return the settled
    state. Enters OFFBOARD first if needed (setpoints streamed for stream_s
    >1 s before the switch, as PX4 requires). The setpoint is a carrot moving
    toward the target at <= `speed` m/s so PX4 never sees a large step.
    on_state(s) is called with every new state and may raise Abort.
    Raises Abort("timeout") / Abort("offboard") on failure."""
    dt = params.control_dt
    pos = np.asarray(pos, dtype=np.float64)
    s = vehicle.wait_state()
    pacer = Pacer(vehicle, dt, s)
    sp = s.pos.copy()
    t_start = s.t_sim

    def tick():
        nonlocal s
        vehicle.send_position(sp, yaw)
        s = pacer.advance(s)
        if on_state is not None:
            on_state(s)

    if not vehicle.is_offboard:
        for _ in range(int(math.ceil(stream_s / dt))):
            tick()
        for _ in range(offboard_tries):
            vehicle.set_offboard()
            for _ in range(int(math.ceil(0.5 / dt))):
                tick()
                if vehicle.is_offboard:
                    break
            if vehicle.is_offboard:
                break
        else:
            raise Abort("offboard", "PX4 did not enter OFFBOARD")

    settled_since = None
    while True:
        step = pos - sp
        n = np.linalg.norm(step)
        sp = pos.copy() if n <= speed * dt else sp + step * (speed * dt / n)
        tick()
        err = np.linalg.norm(s.pos - pos)
        yerr = abs(float(frames.wrap_pi(frames.yaw_of_quat_enu(s.quat) - yaw)))
        ok = err < tol and np.linalg.norm(s.vel) < vel_tol and yerr < yaw_tol
        if ok:
            settled_since = s.t_sim if settled_since is None else settled_since
            if s.t_sim - settled_since >= settle_s - 1e-9:
                return s
        else:
            settled_since = None
        if s.t_sim - t_start > timeout:
            raise Abort("timeout", f"start pose not reached in {timeout:.0f} s sim time "
                                   f"(pos err {err:.2f} m)")


def auto_geofence(course, margin=8.0, z_min=0.5, z_extra=4.0):
    """ENU box around every gate centre and the start pose, `margin` m wider
    horizontally (RL overshoots turns at ~9 m/s), z in [z_min, top gate +
    z_extra]. Used when no --geofence is given: a hand-typed box was too tight
    for oval6 on the first dry run (2026-10-20)."""
    pts = np.array([g.center for g in course.gates] + [course.start_pose()[0]])
    lo, hi = pts.min(0), pts.max(0)
    return ((lo[0] - margin, lo[1] - margin, z_min), (hi[0] + margin, hi[1] + margin, hi[2] + z_extra))


def ctbr_preroll(vehicle, params, s, pos, yaw, seconds=1.0, on_state=None):
    """Hold (pos, yaw) with CTBR commands for `seconds` and return the last state.

    WHY: PX4 v1.18's commander refreshes its control-mode flags only every
    0.5 s. Measured live 2026-10-24: CTBR streaming started at t_sim 339.10, PX4
    dropped position control at 339.54. Until then it ignores our rates and
    thrust (RL's opening thrust-cut + pitch dive did nothing for 0.45 s, then the
    drone over-rotated into gate 0's bottom bar). Holding the start pose through
    the switch with a simple CTBR position hold (vehicle._PositionHold, the
    SimVehicle stand-in's controller) means PX4 is already in rate mode when the
    race controller's first command arrives."""
    from types import SimpleNamespace
    from vehicle import _PositionHold
    hold = _PositionHold(params)
    dt = params.control_dt
    pacer = Pacer(vehicle, dt, s)
    for _ in range(int(round(seconds / dt))):
        pos_t, vel_t, quat_t, _ = _to_tensors(s)
        st = SimpleNamespace(pos=pos_t, vel=vel_t, quat=quat_t, dtype=torch.float32)
        thrust, rates = hold.act(st, pos, yaw, dt)
        vehicle.send_ctbr(float(thrust[0]), rates[0].double().numpy())
        s = pacer.advance(s)
        if on_state is not None:
            on_state(s)
    return s


def _to_tensors(s):
    t = lambda x, n: torch.as_tensor(x, dtype=torch.float32).reshape(1, n)  # noqa: E731
    return t(s.pos, 3), t(s.vel, 3), t(s.quat, 4), t(s.omega, 3)


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------
def run_in_vehicle(vehicle, course, controller, params=None, *, max_time=60.0, start_tol=0.25,
                   start_vel_tol=0.25, start_yaw_tol=0.15, settle_s=1.0, start_timeout=60.0,
                   approach_speed=2.0, ground_z=0.15, max_tilt_deg=80.0, max_gate_dist=30.0,
                   geofence=None, hold_s=1.0, hold_decel=1.5, verbose=True, gate_source=None,
                   preroll_s=1.0, vio=None, vio_max_err=1.0, vio_max_age=0.5, vio_start_err=0.3,
                   vio_wait_s=20.0, vio_err_ticks=10):
    """Returns (log, summary); see the module docstring. `geofence` is
    (min_xyz, max_xyz) in ENU or None. `course` is always the TRUE course
    (scoring); with gate_source (a gate_vision.GateSource) the controller must
    have been built from gate_source.map.prior and is fed the map instead."""
    params = params if params is not None else QuadParams()
    if getattr(controller, "command", "ctbr") == "wrench":
        raise NotImplementedError("MPC wrench->CTBR comes in Phase 4")
    dt = params.control_dt
    n_gates = len(course)
    safety = Safety(ground_z, max_tilt_deg, max_gate_dist, geofence)
    gs = gate_source
    log = _Log(extras=gs is not None, vio=vio is not None)
    say = print if verbose else (lambda *a, **k: None)

    def vio_stamp(s):
        """VIO mode: the newest VIO sample and its error vs PX4's estimate; stamps the log."""
        v = vio.latest()
        if v is None:
            log.vio_extra = (np.nan,) * 5
            return None, float("nan")
        err = float(np.linalg.norm(v.pos - s.pos))
        log.vio_extra = (*v.pos, err, 1e3 * v.age())
        return v, err

    def vio_controller_state(s):
        """The state the controller sees in VIO mode; Abort("vio") if the VIO is stale."""
        v, err = vio_stamp(s)
        if v is None or v.age() > vio_max_age:
            raise Abort("vio", "no VIO odometry" if v is None else f"VIO odometry {v.age():.2f} s old")
        return VehicleState(s.t_sim, v.pos.copy(), v.vel.copy(), v.quat.copy(), s.omega.copy()), err

    def vision_tick(s, gate_idx):
        """Vision mode: update the map from the newest detections, stamp the log."""
        n = gs.step(s, gate_idx)
        log.extra = (n, gs.target_error(gate_idx))
        return n

    kind = ""
    gate_idx = 0
    finished = False
    lap_time = float("nan")
    t_race0 = None
    try:
        # ---- phase A: fly to the start pose ----
        # vision mode: the controller (and phase A) know only the PRIOR
        start_pos, start_yaw = (gs.map.prior if gs is not None else course).start_pose()

        def on_a(s):
            if gs is not None:
                vision_tick(s, 0)
            if vio is not None:
                vio_stamp(s)
            log.add(0, s, 0)
            safety.check(s)

        # warm-up: the MPC builds and compiles its IPOPT problem in its first
        # reset (seconds on the pod). Done at the race start, no setpoints reached
        # PX4 meanwhile and its offboard-loss failsafe fired (2026-10-24). Later
        # resets reuse the problem; PID / RL resets are cheap.
        controller.reset(1)
        if gs is not None and hasattr(controller, "async_replan"):
            controller.async_replan = True   # never block the 50 Hz loop on a re-plan (mpc_ctbr.py)
        s0 = vehicle.wait_state()
        say(f"[runner] start phase: from {np.round(s0.pos, 2)} to {np.round(start_pos, 2)} "
            f"yaw {math.degrees(start_yaw):.0f} deg (offboard={vehicle.is_offboard})")
        if s0.pos[2] < ground_z + 0.3:
            say("[runner] vehicle looks to be on the ground; take off first (commander takeoff)")
        safety.check(s0)
        s = goto(vehicle, params, start_pos, start_yaw, speed=approach_speed, tol=start_tol,
                 vel_tol=start_vel_tol, yaw_tol=start_yaw_tol, settle_s=settle_s,
                 timeout=start_timeout, on_state=on_a)
        if preroll_s > 0:       # hand PX4 over to rate control before the race (see ctbr_preroll)
            s = ctbr_preroll(vehicle, params, s, start_pos, start_yaw, preroll_s, on_state=on_a)
        if vio is not None:      # hold the start until the VIO agrees with PX4 for 1 s
            t_w0, good_since = s.t_sim, None
            while True:
                v, err = vio_stamp(s)
                ok = v is not None and v.age() <= vio_max_age and err < vio_start_err
                good_since = (good_since if good_since is not None else s.t_sim) if ok else None
                if good_since is not None and s.t_sim - good_since >= 1.0:
                    say(f"[runner] VIO ready: {err:.2f} m from PX4's estimate")
                    break
                if s.t_sim - t_w0 > vio_wait_s:
                    why = "no odometry" if v is None else f"age {v.age():.2f} s"
                    raise Abort("vio", f"VIO not within {vio_start_err} m of PX4 after {vio_wait_s:.0f} s "
                                       f"(last error {err:.2f} m, {why})")
                s = ctbr_preroll(vehicle, params, s, start_pos, start_yaw, 0.2, on_state=on_a)

        # ---- phase B: race ----
        say(f"[runner] at start pose (t_sim {s.t_sim:.2f}); racing {course.name} "
            f"({n_gates} gates)")
        controller.reset(1)
        if gs is not None:        # after reset: a controller's own update_course may need its reset state
            update_course(controller, gs.current_course(), s.pos, s.vel, gate_idx=gate_idx)
        pacer = Pacer(vehicle, dt, s)
        t_race0 = s.t_sim
        p_prev = None
        ctl_gate_idx, vio_prev, vio_bad = 0, None, 0
        while True:
            if p_prev is not None:
                pp = torch.as_tensor(p_prev, dtype=torch.float32).reshape(1, 3)
                pn = torch.as_tensor(s.pos, dtype=torch.float32).reshape(1, 3)
                gb = gate_batch(course, torch.tensor([gate_idx]))
                passed, hit = check_gate(pp, pn, gb, params.collision_radius)
                if bool(hit[0]):
                    log.add(1, s, gate_idx)
                    raise Abort("gate", f"collision with gate {gate_idx}")
                if bool(passed[0]):
                    if gs is not None:
                        gs.record_crossing(gate_idx)
                    gate_idx += 1
                    say(f"[runner] passed gate {gate_idx}/{n_gates} at t = "
                        f"{s.t_sim - t_race0:.2f} s")
                    if gate_idx >= n_gates:
                        finished = True
                        lap_time = s.t_sim - t_race0
                        log.add(1, s, gate_idx)
                        break
            target_gate = course.gates[min(gate_idx, n_gates - 1)].center
            try:
                safety.check(s, target_gate)
            except Abort:
                log.add(1, s, gate_idx)
                raise
            if s.t_sim - t_race0 >= max_time:
                log.add(1, s, gate_idx)
                raise Abort("timeout", f"race longer than {max_time:.0f} s")

            if gs is not None:
                vision_tick(s, gate_idx)
                if gs.changed():
                    update_course(controller, gs.current_course(), s.pos, s.vel, gate_idx=gate_idx)
            sc, cg = s, gate_idx
            if vio is not None:
                sc, err = vio_controller_state(s)
                vio_bad = vio_bad + 1 if err > vio_max_err else 0
                if vio_bad >= vio_err_ticks:
                    log.add(1, s, gate_idx)
                    raise Abort("vio", f"VIO {err:.2f} m from PX4's estimate for {vio_err_ticks} ticks")
                if vio_prev is not None and ctl_gate_idx < n_gates:   # the controller's own gate count
                    gb = gate_batch(course, torch.tensor([ctl_gate_idx]))
                    pv, _ = check_gate(torch.as_tensor(vio_prev, dtype=torch.float32).reshape(1, 3),
                                       torch.as_tensor(sc.pos, dtype=torch.float32).reshape(1, 3),
                                       gb, params.collision_radius)
                    ctl_gate_idx += int(bool(pv[0]))
                vio_prev = sc.pos
                cg = min(ctl_gate_idx, n_gates - 1)
            pos_t, vel_t, quat_t, om_t = _to_tensors(sc)
            t0 = time.perf_counter()
            thrust, rates = controller.act(pos_t, vel_t, quat_t,
                                           torch.tensor([cg], dtype=torch.long), dt,
                                           omega=om_t)
            ctrl_ms = 1e3 * (time.perf_counter() - t0)
            T = float(thrust.reshape(-1)[0])
            r = rates.reshape(3).double().numpy()
            u = vehicle.send_ctbr(T, r)
            log.add(1, s, gate_idx, T, r, u, ctrl_ms)
            p_prev = s.pos
            s = pacer.advance(s)

        # ---- finished: brake to a stop, then hold ----
        # The setpoint follows a constant-deceleration stopping path instead of
        # jumping to the current position: a position step at race speed made PX4
        # brake so hard that the Isaac Starling 2 Max flipped (2026-11-12, 2 of 3 runs).
        say(f"[runner] lap complete in {lap_time:.2f} s; braking to a hold")
        hold_yaw = float(frames.yaw_of_quat_enu(s.quat))
        for sp in braking_setpoints(s.pos, s.vel, dt, decel=hold_decel, hold_s=hold_s):
            vehicle.send_position(sp, hold_yaw)
            s = pacer.advance(s)
            log.add(2, s, gate_idx)
    except Abort as a:
        kind = a.kind
        say(f"[runner] ABORT ({a})")
    except KeyboardInterrupt:
        kind = "interrupt"
        say("[runner] Ctrl-C")
    except LinkTimeout as e:
        kind = "link"
        say(f"[runner] link lost: {e}")
    finally:
        try:
            vehicle.set_loiter()
        except Exception as e:   # the link may be what failed; nothing more to do
            print(f"[runner] WARNING: set_loiter failed: {e}")

    arrays = log.arrays()
    summary = _summarise(arrays, course, params, kind, finished, gate_idx, lap_time)
    if gs is not None:
        summary["vision"] = gs.summary()
    if vio is not None:
        race = arrays["phase"] == 1
        e = arrays["vio_err"][race]
        e = e[np.isfinite(e)]
        summary["vio"] = dict(err_mean=float(e.mean()) if e.size else float("nan"),
                              err_max=float(e.max()) if e.size else float("nan"),
                              err_p95=float(np.percentile(e, 95)) if e.size else float("nan"),
                              samples=int(vio.n))
    return arrays, summary


def _summarise(log, course, params, kind, finished, gate_idx, lap_time):
    race = log["phase"] == 1
    speed = np.linalg.norm(log["vel"][race], axis=-1)
    margins = edge_margins(log["pos"][race], course) if race.any() else np.array([])
    ms = log["ctrl_ms"][race]
    ms = ms[np.isfinite(ms)]
    rtf = float("nan")
    if race.sum() > 1:
        ts, tw = log["t_sim"][race], log["t_wall"][race]
        if tw[-1] > tw[0]:
            rtf = float((ts[-1] - ts[0]) / (tw[-1] - tw[0]))
    return dict(
        completed=bool(finished), lap_time=float(lap_time), gates_passed=int(gate_idx),
        n_gates=len(course), crashed=kind in PHYSICAL_CRASHES, crash_kind=kind,
        aborted=bool(kind),
        max_speed=float(speed.max()) if speed.size else float("nan"),
        mean_speed=float(speed.mean()) if speed.size else float("nan"),
        completed_frac=1.0 if finished else 0.0,
        min_margin=float(np.nanmin(margins)) if np.isfinite(margins).any() else float("nan"),
        margins=[None if not np.isfinite(m) else float(m) for m in margins],
        ctrl_ms_mean=float(ms.mean()) if ms.size else float("nan"),
        ctrl_ms_max=float(ms.max()) if ms.size else float("nan"),
        real_time_factor=rtf)


def format_vision_summary(s):
    """Extra lines for a vision-mode summary ('' otherwise)."""
    v = s.get("vision")
    if not v:
        return ""
    vs = v.get("vision", {})
    ce = ", ".join(f"{x:.2f}" for x in v["crossing_pos_err"]) or "-"
    cl = ", ".join(f"{x:.2f}" for x in v["crossing_lat_err"]) or "-"
    prior = ", ".join(f"{x:.2f}" for x in v.get("prior_pos_err", []))
    return (f"\n  vision: detections={v['detections_total']} (associated {v['detections_associated']}) "
            f"ticks with detections={v['ticks_with_detections']}/{v['ticks']} "
            f"frames={vs.get('frames_processed', '-')} proc_ms={vs.get('proc_ms_mean', float('nan')):.1f}"
            f"\n  map error at crossing, 3-D per gate [m]: {ce}  (mean {v['crossing_pos_err_mean']:.2f})"
            f"\n  map error at crossing, in the gate plane [m]: {cl}  (mean {v['crossing_lat_err_mean']:.2f})"
            f"\n  prior 3-D error per gate [m]: {prior}")


def format_summary(name, s):
    v = s.get("vio")
    vio = "" if not v else (f"\n  vio: error vs PX4 estimate mean {v['err_mean']:.2f} m, p95 {v['err_p95']:.2f} m, "
                            f"max {v['err_max']:.2f} m ({v['samples']} odometry messages)")
    return format_summary_line(name, s) + format_vision_summary(s) + vio


def format_summary_line(name, s):
    return (f"{name}: completed={s['completed']} lap_time={s['lap_time']:.2f}s "
            f"gates={s['gates_passed']}/{s['n_gates']} crash_kind={s['crash_kind'] or '-'} "
            f"max_speed={s['max_speed']:.2f}m/s min_margin={s['min_margin']:.2f}m "
            f"ctrl={s['ctrl_ms_mean']:.2f}/{s['ctrl_ms_max']:.1f}ms rtf={s['real_time_factor']:.2f}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def add_vehicle_args(ap):
    """Arguments shared with calibrate.py."""
    ap.add_argument("--connection", default="udpin:0.0.0.0:14540")
    ap.add_argument("--origin-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                    metavar=("X", "Y", "Z"),
                    help="world ENU position of PX4's local-NED origin (calibrate.py origin)")
    ap.add_argument("--thrust-map", choices=["model", "linear"], default="model")
    ap.add_argument("--u-hover", type=float, default=None,
                    help="hover thrust u for --thrust-map linear (default: model prediction)")
    ap.add_argument("--thrust-scale", type=float, default=1.0,
                    help="real thrust / model thrust (calibrate.py hover)")
    ap.add_argument("--sim", action="store_true", help="offline SimVehicle instead of MAVLink")
    ap.add_argument("--sim-start", type=float, nargs=3, default=(0.0, 0.0, 3.0),
                    metavar=("X", "Y", "Z"), help="--sim only: initial hover position")


def make_vehicle(args, params):
    tm = ThrustMap(args.thrust_map, params, u_hover=args.u_hover, scale=args.thrust_scale)
    if args.sim:
        return SimVehicle(params, start_pos=tuple(args.sim_start),
                          origin_offset_enu=tuple(args.origin_offset), thrust_map=tm)
    return MavlinkVehicle(args.connection, tuple(args.origin_offset), tm)


def main(argv=None):
    from fly_course import make_controller
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--track", required=True)
    ap.add_argument("--controller", default="pid", choices=["pid", "rl", "mpc_ctbr", "mpc_rate"])
    ap.add_argument("--mpc-delay-ms", type=float, default=0.0,
                    help="mpc_rate: command delay to compensate [ms of SIM time]")
    ap.add_argument("--mpc-backend", default="acados", choices=["acados", "ipopt"],
                    help="mpc_rate solver (ipopt is not real time; README_mpc_rate.md)")
    ap.add_argument("--mpc-vmax", type=float, default=4.0, help="MPC plan speed limit [m/s] (mpc_ctbr, mpc_rate)")
    ap.add_argument("--mpc-amax", type=float, default=6.0, help="MPC plan acceleration limit [m/s^2]")
    ap.add_argument("--mpc-yaw", default="velocity", choices=["velocity", "next_gate"],
                    help="mpc_rate yaw: along the velocity, or the nose (camera) on the next gate (VIO)")
    ap.add_argument("--model", default=None, help="run dir with model.zip (for --controller rl)")
    add_vehicle_args(ap)
    ap.add_argument("--geofence", type=float, nargs=6, default=None,
                    metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
                    help="ENU box [m]; leaving it aborts to loiter (default: auto_geofence "
                         "around the track)")
    ap.add_argument("--max-time", type=float, default=60.0)
    ap.add_argument("--preroll-s", type=float, default=1.0,
                    help="CTBR position hold at the start pose before the race, so PX4 has "
                         "switched to rate control (its mode flags update every 0.5 s)")
    ap.add_argument("--ground-z", type=float, default=0.15)
    ap.add_argument("--out-dir", default=None)
    vg = ap.add_argument_group("vision mode (see README 'Vision mode')")
    vg.add_argument("--gates", choices=["truth", "vision"], default="truth",
                    help="truth: the controller gets the track file's gates (default). vision: it "
                         "gets a perturbed prior of them, corrected by camera detections")
    vg.add_argument("--prior-pos-std", type=float, default=0.5, help="prior position noise [m] per axis")
    vg.add_argument("--prior-yaw-std", type=float, default=10.0, help="prior yaw noise [deg]")
    vg.add_argument("--prior-seed", type=int, default=0)
    vg.add_argument("--mount-xyz", type=float, nargs=3, default=(0.30, 0.0, 0.05),
                    metavar=("X", "Y", "Z"), help="camera position in the body frame [m]; it was "
                    "once found at -0.040 0 -0.041: measure it before trusting a run")
    vg.add_argument("--mount-rot-xyz-deg", type=float, nargs=3, default=(105.0, 0.0, -90.0),
                    metavar=("RX", "RY", "RZ"), help="USD rotateXYZ of the camera mount [deg]")
    vg.add_argument("--hfov-override", type=float, default=None,
                    help="build intrinsics from the image size and this hFOV [deg] (60 for "
                         "spawn_example.py) instead of camera_info, which can be stale")
    vg.add_argument("--sim-vision", action="store_true",
                    help="with --sim: SimGateVision (geometric camera model) instead of ROS")
    vg.add_argument("--cam-namespace", default="/iris_0")
    vg.add_argument("--pose-topic", default="/drone00/state/pose")
    sg = ap.add_argument_group("state source (see the module docstring, VIO MODE)")
    sg.add_argument("--state", choices=["px4", "vio"], default="px4",
                    help="px4: the controller flies on PX4's estimate (sim GPS). vio: on /vio/odometry")
    sg.add_argument("--vio-topic", default="/vio/odometry")
    sg.add_argument("--vio-max-err", type=float, default=1.0, help="abort when the VIO is this far from PX4 [m]")
    sg.add_argument("--vio-start-err", type=float, default=0.3, help="start only when within this [m]")
    sg.add_argument("--vio-wait-s", type=float, default=20.0, help="how long to wait for that [s sim]")
    a = ap.parse_args(argv)

    params = QuadParams()
    course = Course.load(a.track)           # the TRUE course: scoring only, in vision mode
    gate_source = None
    ctl_course = course
    if a.gates == "vision":
        import vision_gate_map as gm
        import gate_vision as gv
        if a.sim_vision and not a.sim:
            ap.error("--sim-vision needs --sim")
        ctl_course = gm.perturbed_prior(course, a.prior_pos_std, a.prior_yaw_std, a.prior_seed)
        gmap = gm.GateMap(ctl_course, truth_course=course)
        if a.sim_vision:
            front = gv.SimGateVision(course, a.mount_xyz, a.mount_rot_xyz_deg, seed=a.prior_seed)
        else:
            front = gv.RosGateVision(a.cam_namespace, a.pose_topic, mount_xyz=a.mount_xyz,
                                     mount_rot_xyz_deg=a.mount_rot_xyz_deg,
                                     hfov_override=a.hfov_override)
            if not front.wait_ready(15.0):
                front.close()
                print("[runner] vision is not delivering frames; not flying")
                return 2
        gate_source = gv.GateSource(gmap, front, course)
        print(f"[runner] VISION mode: prior pos std {a.prior_pos_std} m/axis, yaw std "
              f"{a.prior_yaw_std} deg, seed {a.prior_seed}; prior 3-D error per gate "
              f"{np.round(gv._prior_errors(gmap, course), 2)}")
    try:
        controller = make_controller(a.controller, params, ctl_course, a.model,
                                     mpc_vmax=a.mpc_vmax, mpc_amax=a.mpc_amax, mpc_yaw=a.mpc_yaw,
                                     mpc_delay_s=a.mpc_delay_ms / 1e3, mpc_backend=a.mpc_backend)
    except ValueError as e:
        if gate_source is not None:
            gate_source.vision.close()
        print(f"[runner] cannot build controller {a.controller!r}: {e}")
        return 2
    geofence = auto_geofence(ctl_course) if a.geofence is None else (a.geofence[:3], a.geofence[3:])
    print(f"[runner] geofence ENU {np.round(geofence[0], 1)} .. {np.round(geofence[1], 1)}")
    vio = None
    if a.state == "vio":
        from vio_state import RosVioState
        vio = RosVioState(a.vio_topic)
        if vio.wait(10.0) is None:
            vio.close()
            print(f"[runner] no VIO odometry on {a.vio_topic}; not flying")
            return 2
        print(f"[runner] VIO mode: controller state from {a.vio_topic}")
    vehicle = make_vehicle(a, params)
    try:
        log, summ = run_in_vehicle(vehicle, course, controller, params, max_time=a.max_time, preroll_s=a.preroll_s,
                                   ground_z=a.ground_z, geofence=geofence,
                                   gate_source=gate_source, vio=vio, vio_max_err=a.vio_max_err,
                                   vio_start_err=a.vio_start_err, vio_wait_s=a.vio_wait_s)
    finally:
        vehicle.close()
        if vio is not None:
            vio.close()
        if gate_source is not None:
            gate_source.vision.close()
    print(format_summary(course.name, summ))
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        write_log_csv(os.path.join(a.out_dir, "log.csv"), log)
        with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
            json.dump(dict(summ, track=a.track, controller=a.controller, args=vars(a)), f,
                      indent=2, default=str)
        print(f"wrote {a.out_dir}/log.csv and summary.json")
    return 0 if summ["completed"] else 1


if __name__ == "__main__":
    sys.exit(main())
