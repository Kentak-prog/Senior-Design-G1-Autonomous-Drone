"""
Autonomous takeoff to a hover, through PX4 OFFBOARD, for the real Starling 2 Max (VOXL 2).

*** Written 2026-11-24; tested only against a fake PX4 (px4/test_takeoff_hover.py). ***
A pilot stands by on the RC the whole time: the Position-mode switch (or the sticks, with
COM_RC_OVERRIDE) takes over at any moment, and the kill switch stops the motors.

WHAT IT DOES:
    heartbeat -> live pose (LOCAL_POSITION_NED / ATTITUDE, i.e. PX4's VIO-based estimate) ->
    refuse unless disarmed, still and the pose is fresh -> confirm -> stream a setpoint at the
    current spot -> ARM -> OFFBOARD -> climb straight up --height m at up to --climb-rate m/s
    (smooth start and stop) -> hover there --hold-s s -> hand PX4 to Hold (AUTO.LOITER), or
    AUTO.LAND with --end land. Writes a per-tick log and a summary to --out-dir.

    The height is relative to where the drone sits when the script starts, not to PX4's
    local origin (which need not be the floor: the bench read 0.76 m on 2026-11-24).

    In Hold the drone keeps hovering on PX4's own position control, with nothing streaming,
    so fly_min_snap_offboard.py (e.g. --gate-ahead) can start from there, or the pilot can
    take over and fly it.

PREFLIGHT (px4/preflight.py, before the confirm prompt): refuses unless the VIO is not an estimate source,
PX4's estimator reports valid velocity and position with sane innovations, attitude and barometer are
healthy, optical flow and the rangefinder are streaming sensibly, and the height reading is steady.

IN FLIGHT (px4/baro_check.py): PX4's height change is compared with the barometer's; if they keep
diverging (0.7 m for 1 s after step offsets fade), the script aborts, prints a pilot alarm, and hands off.
On the 2026-11-24 crash log it fires at 7.1 s; on normal hovers it stays quiet.
IN FLIGHT (px4/accel_check.py): if PX4's velocity changes faster than 3 m/s^2 beyond the climb's own
acceleration (--max-accel) for 0.3 s, the flight is cancelled and handed to LAND, with a pilot alarm.

ABORTS:
    - PX4 leaves OFFBOARD (the pilot switched mode): stop sending at once, the pilot has it.
    - Arming or OFFBOARD refused / not confirmed in time: disarm (still on the ground), exit.
    - Tracking error above --max-error, no position for --stale-s, barometer disagreement, or Ctrl-C: Hold.
    - Velocity changing too fast (accel_check.py): LAND.

HAND-OFF (every way out of OFFBOARD except a pilot takeover): the script keeps streaming a hold setpoint
at the measured position and repeats the Hold/Land request until PX4's heartbeat confirms it
(fly_min_snap_offboard.hand_off). If PX4 doesn't confirm within 3 s, it stops and says so, leaving
PX4's offboard-loss failsafe (COM_OBL_RC_ACT) to act. On 2026-11-24 a single unconfirmed Hold request
was ignored and that failsafe chose Altitude mode.

ON THE DRONE (docker/voxl2-tflite-gpu container, pymavlink in /repo/.pylibs; voxl-vision-hub
relays PX4's MAVLink to localhost:14551, and its own offboard_mode must be "off"):
    cd /data/repo/docker/voxl2-tflite-gpu
    ./run.sh env PYTHONPATH=/repo/.pylibs python3 px4/takeoff_hover.py --height 1.0 \\
        --connection udpin:127.0.0.1:14551 --out-dir /repo/px4_runs/takeoff_01
"""

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fly_min_snap_offboard as fms  # noqa: E402  (PX4Link, connect, wait_for_pose, constants)
import preflight  # noqa: E402
import baro_check  # noqa: E402
import accel_check  # noqa: E402

MAV_CMD_COMPONENT_ARM_DISARM = 400
PX4_MAIN_AUTO = 4
PX4_SUB_AUTO_LOITER = 3
PX4_SUB_AUTO_LAND = 6
MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1
MAV_CMD_DO_SET_MODE = 176

MAX_HEIGHT = 2.5          # m above the start: refuse more (indoor lab)
MIN_HEIGHT = 0.3


def quintic(h: float, duration: float, t: float):
    """Smooth 0 -> h in `duration` s (zero velocity and acceleration at both ends).
    Returns (position, velocity). Peak velocity is 1.875 h / duration."""
    if t <= 0:
        return 0.0, 0.0
    if t >= duration:
        return h, 0.0
    s = t / duration
    pos = h * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)
    vel = h * (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / duration
    return pos, vel


def climb_duration(height: float, climb_rate: float) -> float:
    return 1.875 * height / climb_rate


class Takeoff:
    """The ARM -> OFFBOARD -> climb -> hover -> Hold/Land sequence on a PX4Link."""

    def __init__(self, link, height, climb_rate=0.3, hold_s=5.0, end="hold", max_error=0.5,
                 stale_s=0.5, rate_hz=50.0, arm_timeout=3.0, mode_timeout=2.0, baro=True,
                 max_accel=None, clock=time.monotonic, sleep=time.sleep):
        if max_accel is None:
            max_accel = accel_check.DEFAULT_MARGIN + accel_check.quintic_peak_accel(
                height, climb_duration(height, climb_rate))
        self.accel = accel_check.VelocityJumpCheck(max_accel)
        self.baro = baro_check.BaroCrossCheck() if baro else None
        self.link, self.height, self.climb_rate = link, height, climb_rate
        self.hold_s, self.end, self.max_error, self.stale_s = hold_s, end, max_error, stale_s
        self.dt = 1.0 / rate_hz
        self.arm_timeout, self.mode_timeout = arm_timeout, mode_timeout
        self.clock, self.sleep = clock, sleep
        self.rows = []
        self.handoff = None
        self.start_ned = np.array(link.pos_ned, dtype=float)
        self.yaw = float(link.yaw)
        self.target_ned = self.start_ned + np.array([0.0, 0.0, -height])

    # -- helpers ---------------------------------------------------------------
    def _send(self, pos, vel=(0.0, 0.0, 0.0)):
        vel = np.asarray(vel, dtype=float)
        mask = fms.TYPE_MASK_HOLD if not np.any(vel) else fms.TYPE_MASK_POS_VEL_ACC_YAW
        self.link.send_full(pos, vel, np.zeros(3), self.yaw, mask)

    def _tick(self, phase, t, sp):
        self.link.poll()
        p = self.link.pos_ned
        self.rows.append({"phase": phase, "t": round(t, 3),
                          "sp_n": sp[0], "sp_e": sp[1], "sp_d": sp[2],
                          "n": None if p is None else p[0], "e": None if p is None else p[1],
                          "d": None if p is None else p[2],
                          "armed": self.link.armed, "main_mode": self.link.main_mode})

    def _wait(self, phase, cond, timeout, sp):
        t0 = self.clock()
        while self.clock() - t0 < timeout:
            self._send(sp)
            self._tick(phase, self.clock() - t0, sp)
            if cond():
                return True
            self.sleep(self.dt)
        return False

    def _set_mode(self, main, sub=0):
        self.link.command(MAV_CMD_DO_SET_MODE, MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, main, sub)

    def _disarm(self):
        self.link.command(MAV_CMD_COMPONENT_ARM_DISARM, 0)

    def _finish(self, land=False):
        """Hand PX4 to Hold (or Land) and CONFIRM it, holding the measured position meanwhile
        (fms.hand_off). 2026-11-24 the single unconfirmed Hold request was never taken."""
        self.link.poll()
        hold = np.array(self.link.pos_ned if self.link.pos_ned is not None else self.target_ned, dtype=float)
        sub = PX4_SUB_AUTO_LAND if (land or self.end == "land") else PX4_SUB_AUTO_LOITER
        try:
            ok, msg = fms.hand_off(self.link, lambda: self._send(hold), main=PX4_MAIN_AUTO, sub=sub,
                                   rate_hz=1.0 / self.dt, clock=self.clock, sleep=self.sleep)
        except KeyboardInterrupt:
            ok, msg = False, "Ctrl-C during the hand-off; left to PX4's offboard-loss failsafe"
        self.handoff = {"confirmed": ok, "message": msg}
        return msg

    # -- the sequence ------------------------------------------------------------
    def execute(self) -> str:
        start = self.start_ned
        try:
            if self.baro is not None:
                baro_check.request_baro(self.link)
            # PX4 wants setpoints streaming before it accepts OFFBOARD.
            self._wait("prestream", lambda: False, 1.0, start)
            self.link.arm()
            if not self._wait("arm", lambda: self.link.armed, self.arm_timeout, start):
                return "aborted: PX4 did not arm (check the pre-arm messages in QGroundControl)"
            self.link.set_offboard()
            if not self._wait("offboard", lambda: self.link.main_mode == fms.PX4_MAIN_OFFBOARD,
                              self.mode_timeout, start):
                self._disarm()
                return "aborted: PX4 did not enter OFFBOARD; disarmed (still on the ground)"

            T = climb_duration(self.height, self.climb_rate)
            self.accel.reset()
            t0 = self.clock()
            while True:
                t = self.clock() - t0
                phase = "climb" if t < T else "hover"
                if t >= T + self.hold_s:
                    break
                dz, vz = quintic(self.height, T, t)
                sp = start + np.array([0.0, 0.0, -dz])
                self._send(sp, (0.0, 0.0, -vz))
                self._tick(phase, t, sp)
                if self.link.main_mode != fms.PX4_MAIN_OFFBOARD:
                    return "pilot took over (PX4 left OFFBOARD); stopped sending"
                if self.link.last_pos_time is None or self.clock() - self.link.last_pos_time > self.stale_s:
                    return f"aborted: no position for {self.stale_s} s -> {self._finish()}"
                reason = self.accel.update(self.clock(), self.link.vel_ned, self.link.last_pos_time)
                if reason:              # before the tracking check: violent motion always goes to LAND
                    print(accel_check.PILOT_ALARM, flush=True)
                    return f"aborted: {reason} -> {self._finish(land=True)}"
                err = float(np.linalg.norm(self.link.pos_ned - sp))
                if err > self.max_error:
                    if self.accel.violent:
                        print(accel_check.PILOT_ALARM, flush=True)
                        return (f"aborted: {err:.2f} m from the setpoint while the velocity changes at "
                                f"{self.accel.accel:.1f} m/s^2 -> {self._finish(land=True)}")
                    return f"aborted: {err:.2f} m from the setpoint -> {self._finish()}"
                if self.baro is not None:
                    alt, t_alt = baro_check.baro_from_link(self.link)
                    reason = self.baro.update(self.clock(), self.link.pos_ned[2], alt, t_alt)
                    if reason:
                        print(baro_check.PILOT_ALARM, flush=True)
                        return f"aborted: {reason} -> {self._finish()}"
                self.sleep(self.dt)
            return f"completed -> {self._finish()}"
        except KeyboardInterrupt:
            if self.link.armed:
                return f"Ctrl-C -> {self._finish()}"
            return "Ctrl-C before arming; nothing flown"


# ----------------------------------------------------------------------------------------- run

def run(args, link=None, clock=time.monotonic, sleep=time.sleep, ask=input) -> int:
    if not (MIN_HEIGHT <= args.height <= MAX_HEIGHT):
        print(f"REFUSING: --height {args.height} m is outside {MIN_HEIGHT}-{MAX_HEIGHT} m.")
        return 2
    if not (0.05 <= args.climb_rate <= 1.0):
        print(f"REFUSING: --climb-rate {args.climb_rate} m/s is outside 0.05-1.0 m/s.")
        return 2
    link = link or fms.connect(args.connection, clock=clock)
    if not fms.wait_for_pose(link, clock=clock, sleep=sleep):
        print("no LOCAL_POSITION_NED / ATTITUDE / autopilot HEARTBEAT within 10 s; aborting")
        return 1
    p, yaw = link.pos_ned, link.yaw
    print(f"[start] PX4 local NED {np.round(p, 3)}, yaw {math.degrees(yaw):+.1f} deg, "
          f"armed={link.armed}, main mode {link.main_mode}")
    if link.armed:
        print("REFUSING: already armed. This script takes off from the ground; if the drone is "
              "already flying, use fly_min_snap_offboard.py or the pilot.")
        return 1
    if link.vel_ned is not None and float(np.linalg.norm(link.vel_ned)) > 0.2:
        print(f"REFUSING: PX4 says the drone is moving ({np.linalg.norm(link.vel_ned):.2f} m/s). "
              f"It must sit still on the ground (and the VIO must have settled).")
        return 1
    if args.skip_preflight:
        print("WARNING: --skip-preflight: PX4's estimate sources and health were NOT checked (simulator only).")
    else:
        ok, results, warnings = preflight.run_checks(link, on_ground=True, clock=clock, sleep=sleep)
        preflight.report(ok, results, warnings)
        if not ok:
            print("REFUSING: preflight checks failed (see px4/preflight.py for what each one means).")
            return 1
    T = climb_duration(args.height, args.climb_rate)
    print(f"[plan] arm, OFFBOARD, climb {args.height:.2f} m straight up in {T:.1f} s "
          f"(peak {args.climb_rate:.2f} m/s) to PX4 local z {p[2] - args.height:+.2f} (NED), "
          f"hover {args.hold_s:.0f} s, then {'AUTO.LAND' if args.end == 'land' else 'Hold'}")
    if not args.yes:
        reply = ask("Take off? Pilot ready on the RC, props clear? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("not flying.")
            return 0
    if args.no_baro_check:
        print("WARNING: --no-baro-check: PX4's height is NOT cross-checked against the barometer in flight.")
    run_ = Takeoff(link, args.height, args.climb_rate, args.hold_s, args.end, args.max_error,
                   args.stale_s, args.rate, baro=not args.no_baro_check, max_accel=args.max_accel,
                   clock=clock, sleep=sleep)
    print(f"[failsafe] cancel and LAND if the velocity changes faster than {run_.accel.max_accel:.1f} m/s^2 "
          f"for {run_.accel.sustain:.1f} s")
    status = run_.execute()
    print(f"[takeoff] {status}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "takeoff_log.csv"), "w", newline="") as f:
        if run_.rows:
            w = csv.DictWriter(f, fieldnames=list(run_.rows[0]))
            w.writeheader()
            w.writerows(run_.rows)
    hover = [r for r in run_.rows if r["phase"] == "hover" and r["d"] is not None]
    err = [math.dist((r["n"], r["e"], r["d"]), (r["sp_n"], r["sp_e"], r["sp_d"])) for r in hover]
    summary = {"status": status, "handoff": run_.handoff, "height_m": args.height, "climb_rate": args.climb_rate,
               "start_ned": run_.start_ned.tolist(), "target_ned": run_.target_ned.tolist(),
               "hover_err_mean_m": float(np.mean(err)) if err else None,
               "hover_err_max_m": float(np.max(err)) if err else None}
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if err:
        print(f"[takeoff] hover error mean {summary['hover_err_mean_m']:.3f} m, "
              f"max {summary['hover_err_max_m']:.3f} m")
    print(f"[takeoff] log in {args.out_dir}")
    return 0 if status.startswith("completed") else 1


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--height", type=float, default=1.0,
                    help=f"climb this far above where the drone sits, m ({MIN_HEIGHT}-{MAX_HEIGHT}, default 1)")
    ap.add_argument("--climb-rate", type=float, default=0.3, help="peak climb speed, m/s (default 0.3)")
    ap.add_argument("--hold-s", type=float, default=5.0, help="hover in OFFBOARD before handing over, s")
    ap.add_argument("--end", choices=("hold", "land"), default="hold",
                    help="after the hover: Hold (AUTO.LOITER, default) or AUTO.LAND")
    ap.add_argument("--max-error", type=float, default=0.5, help="abort to Hold beyond this, m (default 0.5)")
    ap.add_argument("--stale-s", type=float, default=0.5, help="abort if no position for this long, s")
    ap.add_argument("--rate", type=float, default=50.0, help="setpoint rate, Hz (default 50)")
    ap.add_argument("--connection", default="udpin:127.0.0.1:14551",
                    help="default: voxl-vision-hub's localhost MAVLink relay on the VOXL 2")
    ap.add_argument("--out-dir", default=None, help="default px4_runs/takeoff_<time>")
    ap.add_argument("--yes", action="store_true", help="do not ask before flying")
    ap.add_argument("--no-baro-check", action="store_true",
                    help="turn off the in-flight height-vs-barometer cross-check (px4/baro_check.py)")
    ap.add_argument("--max-accel", type=float, default=None,
                    help="cancel and LAND if PX4's velocity changes faster than this, m/s^2, for 0.3 s "
                         "(default 3 + the climb's own peak acceleration)")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="skip px4/preflight.py's estimator checks (simulators without optical flow only)")
    return ap


def main():
    args = build_parser().parse_args()
    args.out_dir = args.out_dir or os.path.join("px4_runs", time.strftime("takeoff_%Y%m%d_%H%M%S"))
    sys.exit(run(args))


if __name__ == "__main__":
    main()
