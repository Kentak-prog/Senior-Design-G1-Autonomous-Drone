"""
Fly the minimum-snap gate trajectory in Isaac Sim through PX4 OFFBOARD
(MAVLink position / velocity / acceleration / yaw setpoints). Stage 1 of
flying the course: PX4's own position controller tracks the min-snap
reference; the repo NMPC is not in the loop yet.

*** UNTESTED AGAINST THE CLUSTER. *** The planning, frame conversion,
setpoint generation and the whole OFFBOARD sequence are tested against a
fake PX4 (test_fly_min_snap_offboard.py); the real MAVLink link has not been
flown. Start slow (the defaults: 2 m/s, 2 m/s^2) and keep a hand on Ctrl-C,
which hands the drone back to AUTO.LOITER.

WHAT IT DOES:
    heartbeat -> live pose from LOCAL_POSITION_NED / ATTITUDE (the drone must
    already be hovering, e.g. after `commander takeoff`) -> plan min-snap
    from that pose through the TRUE gates in the manifest
    (Trajectory/gate_path.plan_gate_trajectory, unchanged) -> refuse if the
    plan misses a gate -> confirm -> stream hover setpoints, switch to
    OFFBOARD, hover --hover-s -> stream the trajectory at --rate Hz -> hold
    the end point --hover-s -> AUTO.LOITER. Writes the plan, a per-tick
    setpoint/measurement log and a summary to --out-dir, and reports how the
    FLOWN path crossed each true gate.

ABORTS (hold the measured position briefly, then AUTO.LOITER): tracking
error above --max-error, PX4 leaving OFFBOARD (e.g. someone switched mode),
no LOCAL_POSITION_NED for --stale-s, or Ctrl-C.

WHERE TO RUN: inside the Isaac pod (the ros2-eval or px4-sitl container).
PX4 SITL only talks MAVLink on localhost ("MAVLink only on localhost" in the
PX4 console), and containers in one pod share localhost. Default connection
udpin:0.0.0.0:14540 is PX4 instance 0's offboard/API link.

DEPENDENCIES: numpy + OpenCV (already in ros2-eval via apt) and pymavlink
(`pip install pymavlink`; if pip ever tries to upgrade numpy, ABORT -- NumPy
2 breaks ROS Humble's cv_bridge, see vision/HANDOFF.md). pymavlink is
imported lazily so --plan-only and the tests run without it.

FRAMES: the manifest and the planner use Pegasus world = ENU (x east,
y north, z up). PX4 local is NED (x north, y east, z down; yaw from north,
clockwise). The two share an origin (measured 2026-10-01, see
vision/gate_error_eval/fly_gate_approach.py), so positions convert with
enu_to_ned / ned_to_enu and yaw with yaw_ned = pi/2 - yaw_enu.

Examples (from the repo root, inside the pod):
    python3 px4/fly_min_snap_offboard.py --plan-only --start -0.05 -0.04 2.55 --heading -7.2
    python3 px4/fly_min_snap_offboard.py                      # plan from the live pose, confirm, fly
    python3 px4/fly_min_snap_offboard.py --v-max 3 --a-max 3  # faster, once the slow run is clean
"""

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("vision", os.path.join("vision", "gate_error_eval"), "Trajectory", "control"):
    _p = os.path.join(REPO, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fly_gate_approach import (Vehicle, enu_to_ned, ned_to_enu,       # noqa: E402
                               MAV_FRAME_LOCAL_NED, MSG_ID_ATTITUDE, MSG_ID_LOCAL_POSITION_NED,
                               PX4_MAIN_OFFBOARD, MAV_MODE_FLAG_SAFETY_ARMED,
                               IGN_VX, IGN_VY, IGN_VZ, IGN_AX, IGN_AY, IGN_AZ, IGN_YAW_RATE)
from gate_path import gates_from_manifest, plan_gate_trajectory, gate_crossings   # noqa: E402
from trajectory_reference import TrajectoryReference                               # noqa: E402

# SET_POSITION_TARGET_LOCAL_NED type_mask (a SET bit means IGNORE that field).
# Position + velocity + acceleration feed-forward + yaw; bit 512 (force) is
# left clear so afx/afy/afz are accelerations.
TYPE_MASK_POS_VEL_ACC_YAW = IGN_YAW_RATE                                         # = 2048
TYPE_MASK_HOLD = IGN_VX | IGN_VY | IGN_VZ | IGN_AX | IGN_AY | IGN_AZ | IGN_YAW_RATE   # = 2552
MAV_AUTOPILOT_PX4 = 12
MAV_COMP_ID_AUTOPILOT1 = 1

DEFAULT_MANIFEST = "/workspace/test_gates.json"
LOG_FIELDS = ["phase", "t",
              "sp_x", "sp_y", "sp_z", "sp_vx", "sp_vy", "sp_vz", "sp_yaw_deg",
              "x", "y", "z", "vx", "vy", "vz", "yaw_deg", "main_mode"]


# --------------------------------------------------------------------------
# Pure geometry / planning (no pymavlink)
# --------------------------------------------------------------------------

def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_enu_to_ned(yaw_enu: float) -> float:
    """ENU yaw (rad CCW from east) -> PX4 yaw (rad CW from north)."""
    return wrap_pi(math.pi / 2.0 - yaw_enu)


def yaw_ned_to_enu(yaw_ned: float) -> float:
    """PX4 yaw (rad CW from north) -> ENU yaw (rad CCW from east). Same formula: it is an involution."""
    return wrap_pi(math.pi / 2.0 - yaw_ned)


def plan_course(gates, start_enu, heading_enu: float, v_max: float, a_max: float, verbose: bool = False):
    """Min-snap plan from start_enu through gates (ENU). Returns (traj, waypoints, ref)."""
    traj, wps = plan_gate_trajectory(gates, np.asarray(start_enu, dtype=float),
                                     v_max=v_max, a_max=a_max, verbose=verbose)
    return traj, wps, TrajectoryReference(traj, start_yaw=heading_enu)


def plan_peaks(traj):
    _, _, vel, acc, _, _ = traj.sample(0.005)
    return float(np.linalg.norm(vel, axis=1).max()), float(np.linalg.norm(acc, axis=1).max())


class Setpoints:
    """Time-indexed setpoints from a planned trajectory. Past the end it holds
    the final position with zero velocity / acceleration."""

    def __init__(self, traj, ref):
        self.traj = traj
        self.ref = ref
        self.duration = float(traj.duration)

    def at(self, t: float) -> dict:
        tc = float(np.clip(t, 0.0, self.duration))
        pos = self.traj.evaluate(tc, 0)
        if t >= self.duration:
            vel = np.zeros(3)
            acc = np.zeros(3)
        else:
            vel = self.traj.evaluate(tc, 1)
            acc = self.traj.evaluate(tc, 2)
        yaw_enu = float(np.interp(tc, self.ref.times, self.ref.states[5]))
        return {"t": float(t), "pos_enu": pos, "vel_enu": vel, "acc_enu": acc, "yaw_enu": yaw_enu,
                "pos_ned": enu_to_ned(pos), "vel_ned": enu_to_ned(vel), "acc_ned": enu_to_ned(acc),
                "yaw_ned": yaw_enu_to_ned(yaw_enu)}


def crossing_report(times, pos_enu, vel_enu, gates):
    """(all_passed, lines) for a path against gates, like simulate_min_snap.report_crossings."""
    lines, ok = [], True
    for gate, crossings in gate_crossings(np.asarray(times), np.asarray(pos_enu), np.asarray(vel_enu), gates):
        through = [c for c in crossings if c['inside'] and c['forward']]
        if not through:
            ok = False
            lines.append(f"{gate.name}: MISSED")
        for c in through:
            margin = gate.side / 2 - np.max(np.abs(c['offset']))
            lines.append(f"{gate.name}: t={c['time']:5.2f}s  offset=({c['offset'][0]:+.2f}, "
                         f"{c['offset'][1]:+.2f}) m  frame margin {margin:.2f} m  "
                         f"angle {c['angle_deg']:4.1f} deg  speed {c['speed']:.2f} m/s")
        for c in crossings:
            if not (c['inside'] and c['forward']):
                what = "BACKWARDS through" if c['inside'] else \
                    f"past the frame ({np.max(np.abs(c['offset'])) - gate.side / 2:.2f} m outside)"
                lines.append(f"{gate.name}: t={c['time']:5.2f}s  {what}")
    return ok, lines


def planned_report(traj, gates):
    t, p, v, *_ = traj.sample(0.005)
    return crossing_report(t, p, v, gates)


def tracking_error(meas_enu, sp_enu) -> float:
    return float(np.linalg.norm(np.asarray(meas_enu, dtype=float) - np.asarray(sp_enu, dtype=float)))


def flight_summary(rows, gates):
    """Gate passes and tracking error of the logged 'trajectory' phase."""
    traj_rows = [r for r in rows if r["phase"] == "trajectory" and r["x"] is not None]
    if len(traj_rows) < 2:
        return {"gates_passed": False, "crossings": ["no trajectory samples logged"],
                "err_mean_m": None, "err_max_m": None, "samples": len(traj_rows)}
    t = np.array([r["t"] for r in traj_rows])
    pos = np.array([[r["x"], r["y"], r["z"]] for r in traj_rows])
    vel = np.array([[r["vx"], r["vy"], r["vz"]] for r in traj_rows])
    sp = np.array([[r["sp_x"], r["sp_y"], r["sp_z"]] for r in traj_rows])
    err = np.linalg.norm(pos - sp, axis=1)
    ok, lines = crossing_report(t, pos, vel, gates)
    return {"gates_passed": bool(ok), "crossings": lines, "err_mean_m": float(err.mean()),
            "err_max_m": float(err.max()), "samples": len(traj_rows)}


def write_plan_csv(path, setpoints: Setpoints, dt: float = 0.02):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az", "yaw_enu_deg"])
        for t in np.arange(0.0, setpoints.duration + 1e-9, dt):
            s = setpoints.at(t)
            w.writerow([f"{t:.3f}"] + [f"{v:.5f}" for v in (*s["pos_enu"], *s["vel_enu"], *s["acc_enu"])]
                       + [f"{math.degrees(s['yaw_enu']):.3f}"])


def write_log_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else
                            (f"{r[k]:.5f}" if isinstance(r[k], float) else r[k])) for k in LOG_FIELDS})


# --------------------------------------------------------------------------
# MAVLink (pymavlink master passed in; UNTESTED against the cluster)
# --------------------------------------------------------------------------

class PX4Link(Vehicle):
    """fly_gate_approach.Vehicle plus measured velocity, a receive timestamp
    and full pos/vel/acc/yaw setpoints."""

    def __init__(self, master, clock=time.monotonic):
        super().__init__(master)
        self.vel_ned = None
        self.last_pos_time = None
        self.clock = clock

    def poll(self):
        while True:
            msg = self.m.recv_match(blocking=False)
            if msg is None:
                return
            t = msg.get_type()
            if t == "LOCAL_POSITION_NED":
                self.pos_ned = np.array([msg.x, msg.y, msg.z], dtype=float)
                self.vel_ned = np.array([msg.vx, msg.vy, msg.vz], dtype=float)
                self.last_pos_time = self.clock()
            elif t == "ATTITUDE":
                self.yaw = float(msg.yaw)
            elif (t == "HEARTBEAT" and msg.get_srcSystem() == self.m.target_system
                  and msg.autopilot == MAV_AUTOPILOT_PX4):
                # Only the flight controller's heartbeat carries the real
                # arm state and mode; other components of system 1 on this
                # link (2026-10-23 dry run: the first heartbeat came from
                # component 0) report their own, which would read as
                # "PX4 left OFFBOARD".
                self.armed = bool(msg.base_mode & MAV_MODE_FLAG_SAFETY_ARMED)
                self.main_mode = (msg.custom_mode >> 16) & 0xFF

    def send_full(self, pos_ned, vel_ned, acc_ned, yaw_ned, type_mask):
        self.m.mav.set_position_target_local_ned_send(
            0, self.m.target_system, self.m.target_component, MAV_FRAME_LOCAL_NED, type_mask,
            float(pos_ned[0]), float(pos_ned[1]), float(pos_ned[2]),
            float(vel_ned[0]), float(vel_ned[1]), float(vel_ned[2]),
            float(acc_ned[0]), float(acc_ned[1]), float(acc_ned[2]),
            float(yaw_ned), 0.0)


class OffboardRun:
    """
    The OFFBOARD sequence. clock/sleep are injectable so the tests can fly it
    against a fake PX4 in simulated time.
    """

    def __init__(self, link: PX4Link, setpoints: Setpoints, rate_hz=50.0, hover_s=3.0,
                 max_error=1.5, stale_s=0.5, clock=time.monotonic, sleep=time.sleep, out=print):
        self.v = link
        self.sp = setpoints
        self.dt = 1.0 / rate_hz
        self.hover_s = hover_s
        self.max_error = max_error
        self.stale_s = stale_s
        self.clock = clock
        self.sleep = sleep
        self.out = out
        self.rows = []
        self.status = "not started"

    # -- helpers ---------------------------------------------------------
    def _measured(self):
        if self.v.pos_ned is None:
            return None, None, None
        vel = self.v.vel_ned if self.v.vel_ned is not None else np.zeros(3)
        yaw = yaw_ned_to_enu(self.v.yaw) if self.v.yaw is not None else None
        return ned_to_enu(self.v.pos_ned), ned_to_enu(vel), yaw

    def _log(self, phase, t, sp_pos_enu, sp_vel_enu, sp_yaw_enu):
        pos, vel, yaw = self._measured()
        self.rows.append({
            "phase": phase, "t": float(t),
            "sp_x": float(sp_pos_enu[0]), "sp_y": float(sp_pos_enu[1]), "sp_z": float(sp_pos_enu[2]),
            "sp_vx": float(sp_vel_enu[0]), "sp_vy": float(sp_vel_enu[1]), "sp_vz": float(sp_vel_enu[2]),
            "sp_yaw_deg": math.degrees(sp_yaw_enu),
            "x": None if pos is None else float(pos[0]), "y": None if pos is None else float(pos[1]),
            "z": None if pos is None else float(pos[2]),
            "vx": None if vel is None else float(vel[0]), "vy": None if vel is None else float(vel[1]),
            "vz": None if vel is None else float(vel[2]),
            "yaw_deg": None if yaw is None else math.degrees(yaw),
            "main_mode": self.v.main_mode})

    def _stream(self, phase, seconds, fn, until=None, check=None):
        """Send fn(t) -> (pos_enu, vel_enu, acc_enu, yaw_enu, mask) every tick on
        a fixed schedule for `seconds`, or until until() is true. check(t) may
        return an abort reason. Returns (elapsed, abort_reason or None)."""
        t0 = self.clock()
        k = 0
        while True:
            t = self.clock() - t0
            if t >= seconds or (until is not None and until()):
                return t, None
            pos, vel, acc, yaw, mask = fn(t)
            self.v.send_full(enu_to_ned(pos), enu_to_ned(vel), enu_to_ned(acc), yaw_enu_to_ned(yaw), mask)
            self.v.poll()
            self._log(phase, t, pos, vel, yaw)
            if check is not None:
                reason = check(t, pos)
                if reason:
                    return t, reason
            k += 1
            self.sleep(max(0.0, t0 + k * self.dt - self.clock()))

    def _hold(self, pos_enu, yaw_enu):
        zero = np.zeros(3)
        return lambda t: (pos_enu, zero, zero, yaw_enu, TYPE_MASK_HOLD)

    # -- sequence --------------------------------------------------------
    def enter_offboard(self, attempts=5) -> bool:
        start = self.sp.at(0.0)
        hold = self._hold(start["pos_enu"], start["yaw_enu"])
        self.out("streaming hover setpoints at the trajectory start ...")
        self._stream("pre_offboard", 1.5, hold)
        for _ in range(attempts):
            self.v.set_offboard()
            self._stream("pre_offboard", 0.5, hold, until=lambda: self.v.main_mode == PX4_MAIN_OFFBOARD)
            if self.v.main_mode == PX4_MAIN_OFFBOARD:
                return True
        return False

    def _trajectory_check(self, t, sp_pos_enu):
        if self.v.main_mode != PX4_MAIN_OFFBOARD:
            return f"PX4 left OFFBOARD (main mode {self.v.main_mode})"
        if self.v.last_pos_time is None or self.clock() - self.v.last_pos_time > self.stale_s:
            return f"no LOCAL_POSITION_NED for more than {self.stale_s} s"
        pos, _, _ = self._measured()
        err = tracking_error(pos, sp_pos_enu)
        if err > self.max_error:
            return f"tracking error {err:.2f} m > --max-error {self.max_error} m at t={t:.2f}s"
        return None

    def execute(self) -> str:
        try:
            if not self.enter_offboard():
                self.status = "aborted: PX4 did not enter OFFBOARD"
                return self.status
            start = self.sp.at(0.0)
            self.out(f"OFFBOARD. hovering at the start {self.hover_s:.1f} s ...")
            _, reason = self._stream("hover", self.hover_s, self._hold(start["pos_enu"], start["yaw_enu"]),
                                     check=lambda t, p: (None if self.v.main_mode == PX4_MAIN_OFFBOARD
                                                         else "PX4 left OFFBOARD during the hover"))
            if reason:
                return self._abort(reason)

            self.out(f"flying the min-snap trajectory ({self.sp.duration:.2f} s) ...")

            def follow(t):
                s = self.sp.at(t)
                return s["pos_enu"], s["vel_enu"], s["acc_enu"], s["yaw_enu"], TYPE_MASK_POS_VEL_ACC_YAW

            _, reason = self._stream("trajectory", self.sp.duration, follow, check=self._trajectory_check)
            if reason:
                return self._abort(reason)

            end = self.sp.at(self.sp.duration)
            self.out(f"holding the end point {self.hover_s:.1f} s ...")
            self._stream("end_hold", self.hover_s, self._hold(end["pos_enu"], end["yaw_enu"]))
            self.status = "completed"
            return self.status
        except KeyboardInterrupt:
            self.status = "aborted: Ctrl-C"
            self.out("\nCtrl-C")
            return self.status
        finally:
            self.out("commanding AUTO.LOITER")
            for _ in range(5):
                self.v.set_loiter()
                self.sleep(0.1)

    def _abort(self, reason) -> str:
        self.status = f"aborted: {reason}"
        self.out(f"ABORT: {reason}. Holding the measured position, then AUTO.LOITER.")
        pos, _, yaw = self._measured()
        if pos is not None and self.v.main_mode == PX4_MAIN_OFFBOARD:
            self._stream("abort_hold", 1.0, self._hold(pos, yaw if yaw is not None else 0.0))
        return self.status


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def default_out_dir():
    base = "/workspace/px4_runs" if os.path.isdir("/workspace") else os.path.join(REPO, "px4", "runs")
    return os.path.join(base, time.strftime("min_snap_%Y%m%d_%H%M%S"))


def connect(connection, clock=time.monotonic):
    from pymavlink import mavutil   # lazy: not needed for planning or the tests
    master = mavutil.mavlink_connection(connection)
    print(f"waiting for heartbeat on {connection} ...")
    master.wait_heartbeat(timeout=30)
    print(f"heartbeat from system {master.target_system} component {master.target_component}")
    if master.target_component != MAV_COMP_ID_AUTOPILOT1:
        print(f"  (addressing commands to the autopilot, component {MAV_COMP_ID_AUTOPILOT1})")
        master.target_component = MAV_COMP_ID_AUTOPILOT1
    link = PX4Link(master, clock=clock)
    link.request_interval(MSG_ID_LOCAL_POSITION_NED, 50)
    link.request_interval(MSG_ID_ATTITUDE, 50)
    return link


def wait_for_pose(link, timeout=10.0, clock=time.monotonic, sleep=time.sleep):
    """Wait for position, attitude AND an autopilot heartbeat (1 Hz, so the
    arm state / mode are real before anything is reported or decided)."""
    deadline = clock() + timeout
    while (link.pos_ned is None or link.yaw is None or link.main_mode is None) and clock() < deadline:
        link.poll()
        sleep(0.05)
    return link.pos_ned is not None and link.yaw is not None and link.main_mode is not None


def run(args, link=None, clock=time.monotonic, sleep=time.sleep, ask=input) -> int:
    with open(args.gates) as f:
        manifest_raw = json.load(f)
    gates = gates_from_manifest(args.gates)
    print(f"[gates] {len(gates)} true gates from {args.gates}")

    # ---- start pose ----
    if args.plan_only:
        if args.start is None or args.heading is None:
            print("--plan-only needs --start X Y Z and --heading DEG (ENU)")
            return 2
        start_enu = np.array(args.start, dtype=float)
        heading_enu = math.radians(args.heading)
    else:
        link = link or connect(args.connection, clock=clock)
        if not wait_for_pose(link, clock=clock, sleep=sleep):
            print("no LOCAL_POSITION_NED / ATTITUDE / autopilot HEARTBEAT within 10 s; aborting")
            return 1
        start_enu = ned_to_enu(link.pos_ned)
        heading_enu = yaw_ned_to_enu(link.yaw)
        print(f"[start] live pose ENU {np.round(start_enu, 3)}, heading {math.degrees(heading_enu):+.1f} deg "
              f"(PX4 NED {np.round(link.pos_ned, 3)}, yaw {math.degrees(link.yaw):+.1f} deg), "
              f"armed={link.armed}, main mode {link.main_mode}")
        if not link.armed:
            print("REFUSING: PX4 reports the vehicle is not armed. Hover first, e.g. `commander takeoff`.")
            return 1
        if link.pos_ned[2] > -args.min_alt:
            print(f"REFUSING: not airborne (altitude {-link.pos_ned[2]:.2f} m < {args.min_alt} m). "
                  f"Hover first, e.g. `commander takeoff` at the pxh> prompt.")
            return 1
    cam = manifest_raw.get("camera_pose_usd", {}) if isinstance(manifest_raw, dict) else {}
    if cam.get("position") is not None:
        drift = float(np.linalg.norm(np.asarray(cam["position"])[:2] - start_enu[:2]))
        if drift > args.max_start_drift:
            print(f"WARNING: the drone is {drift:.2f} m (horizontally) from where the camera was when "
                  f"the gates were spawned; check the plan below before flying.")

    # ---- plan ----
    traj, wps, ref = plan_course(gates, start_enu, heading_enu, args.v_max, args.a_max)
    sp = Setpoints(traj, ref)
    v_pk, a_pk = plan_peaks(traj)
    print(f"[plan] {len(wps)} waypoints, {traj.n_seg} segments, {traj.duration:.2f} s, "
          f"peak speed {v_pk:.2f} m/s, peak accel {a_pk:.2f} m/s^2 (limits {args.v_max} / {args.a_max})")
    ok, lines = planned_report(traj, gates)
    print("[plan] planned trajectory vs TRUE gates:")
    for line in lines:
        print("  " + line)
    if not ok:
        print("REFUSING: the plan does not pass every gate.")
        return 2

    os.makedirs(args.out_dir, exist_ok=True)
    write_plan_csv(os.path.join(args.out_dir, "plan_reference.csv"), sp)
    shutil.copyfile(args.gates, os.path.join(args.out_dir, "test_gates.json"))
    print(f"[plan] wrote {args.out_dir}/plan_reference.csv")
    if args.plan_only:
        return 0

    # ---- fly ----
    if not args.yes:
        reply = ask(f"Fly this {traj.duration:.1f} s trajectory in OFFBOARD? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("not flying.")
            return 0
    runner = OffboardRun(link, sp, rate_hz=args.rate, hover_s=args.hover_s, max_error=args.max_error,
                         stale_s=args.stale_s, clock=clock, sleep=sleep)
    status = runner.execute()
    print(f"\n[flight] {status}")

    summary = flight_summary(runner.rows, gates)
    write_log_csv(os.path.join(args.out_dir, "flight_log.csv"), runner.rows)
    summary.update({"status": status, "start_enu": start_enu.tolist(),
                    "heading_enu_deg": math.degrees(heading_enu), "plan_duration_s": traj.duration,
                    "plan_peak_speed": v_pk, "plan_peak_accel": a_pk,
                    "v_max": args.v_max, "a_max": args.a_max, "rate_hz": args.rate})
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[flight] FLOWN path vs TRUE gates:")
    for line in summary["crossings"]:
        print("  " + line)
    if summary["err_mean_m"] is not None:
        print(f"[flight] tracking error mean {summary['err_mean_m']:.3f} m, max {summary['err_max_m']:.3f} m")
    print(f"[flight] {'ALL GATES PASSED' if summary['gates_passed'] else 'NOT ALL GATES PASSED'}; "
          f"log in {args.out_dir}")
    return 0 if status == "completed" else 1


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--gates", default=DEFAULT_MANIFEST, help=f"test_gates.json (default {DEFAULT_MANIFEST})")
    ap.add_argument("--connection", default="udpin:0.0.0.0:14540")
    ap.add_argument("--v-max", type=float, default=2.0, help="planner speed limit, m/s (default 2)")
    ap.add_argument("--a-max", type=float, default=2.0, help="planner accel limit, m/s^2 (default 2)")
    ap.add_argument("--rate", type=float, default=50.0, help="setpoint rate, Hz (default 50)")
    ap.add_argument("--hover-s", type=float, default=3.0, help="hover before and after, s (default 3)")
    ap.add_argument("--max-error", type=float, default=1.5,
                    help="abort if the drone is this far from the setpoint, m (default 1.5)")
    ap.add_argument("--stale-s", type=float, default=0.5,
                    help="abort if no position for this long, s (default 0.5)")
    ap.add_argument("--min-alt", type=float, default=1.0, help="refuse below this altitude, m (default 1)")
    ap.add_argument("--max-start-drift", type=float, default=1.5,
                    help="warn if the drone is this far from the spawn-time camera, m (default 1.5)")
    ap.add_argument("--out-dir", default=None, help="default /workspace/px4_runs/min_snap_<time>")
    ap.add_argument("--yes", action="store_true", help="do not ask before flying")
    ap.add_argument("--plan-only", action="store_true", help="plan and write the CSV, no PX4 connection")
    ap.add_argument("--start", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="--plan-only start position, ENU m")
    ap.add_argument("--heading", type=float, default=None, help="--plan-only start heading, deg CCW from east")
    return ap


def main():
    args = build_parser().parse_args()
    if args.v_max <= 0 or args.a_max <= 0 or args.rate <= 0:
        sys.exit("--v-max, --a-max and --rate must be positive")
    args.out_dir = args.out_dir or default_out_dir()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
