"""
Scripted straight-line approach flight (PX4 offboard over MAVLink) for
recording gate-error data. Step 1 of 3 of the gate-error workflow.

*** UNTESTED AGAINST THE CLUSTER. *** The geometry below is unit-tested
(test_fly_gate_approach.py); the MAVLink flow and the PX4 mode numbers have
only been written from the PX4 / MAVLink documentation. Try it first with a
short --distance and watch it in the sim; Ctrl-C commands AUTO.LOITER.

BEFORE FLYING, in another shell (ros2-eval container, ROS sourced):
    ros2 bag record -o gate_run1 /iris_0/front_cam/rgb \
        /iris_0/front_cam/camera_info /drone00/state/pose
(1280x720 rgb at ~30 Hz is ~90 MB/s uncompressed: keep runs short.) Start the
recording first, then this script; stop it after "done" is printed.

WHAT IT DOES:
    wait for heartbeat -> (optional --takeoff-alt arm + climb, else the drone
    must already be hovering, e.g. after `commander takeoff`) -> stream
    hover setpoints >1 s, switch to OFFBOARD -> hover --hover-s -> fly
    --distance m along the CURRENT heading at --speed (a moving "carrot"
    setpoint, position + velocity feed-forward, yaw and altitude fixed) ->
    hold --hover-s -> switch back to AUTO.LOITER.

CONNECTION: default udpin:0.0.0.0:14540, PX4 SITL's offboard/API link for
instance 0 (PX4 sends to 14540, listens on 14580). Override with
--connection (e.g. a mavlink-router endpoint).

DEPENDENCY: `pip install pymavlink` does not pull numpy. If pip ever tries to
install/upgrade numpy for it, ABORT: NumPy 2 breaks ROS Humble's cv_bridge
(see vision/HANDOFF.md). pymavlink is imported lazily in main() so the
geometry here imports without it.

FRAMES. PX4's local frame is NED (x north, y east, z down; yaw from north,
clockwise). The manifest (test_gates.json) is Pegasus world = ENU (x east,
y north, z up, "map"). Pegasus converts between them, so the two worlds are
assumed to share an origin and heading. The optional gate-clearance check
(--gates) therefore works ENTIRELY in ENU, using the manifest's
camera_pose_usd position as the start point and the camera's horizontal
forward direction (-Z of the USD camera, flattened) as the flight direction.
ASSUMPTIONS: the drone has not moved or yawed since spawn_test_gates.py ran
(HANDOFF says run it while hovering) and the flight heading equals the camera
heading (the camera is only pitched, so this holds). The PX4 yaw is compared
with the manifest heading and a mismatch >15 deg is printed as a warning.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

# vision/ is the parent directory; keep this importable like its siblings
# (the geometry has no vision imports today, but this keeps layout uniform).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- PX4 / MAVLink constants (written out so they are unit-testable and so
# the file does not need pymavlink to be importable) ----------------------
MAV_FRAME_LOCAL_NED = 1
MAV_CMD_DO_SET_MODE = 176
MAV_CMD_COMPONENT_ARM_DISARM = 400
MAV_CMD_SET_MESSAGE_INTERVAL = 511
MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1
MAV_MODE_FLAG_SAFETY_ARMED = 128
MSG_ID_ATTITUDE = 30
MSG_ID_LOCAL_POSITION_NED = 32
# PX4 custom modes: main mode in the custom_mode byte 2, sub mode in byte 3.
PX4_MAIN_AUTO = 4
PX4_MAIN_OFFBOARD = 6
PX4_SUB_AUTO_LOITER = 3
# SET_POSITION_TARGET_LOCAL_NED type_mask: a SET bit means IGNORE that field.
IGN_X, IGN_Y, IGN_Z = 1, 2, 4
IGN_VX, IGN_VY, IGN_VZ = 8, 16, 32
IGN_AX, IGN_AY, IGN_AZ = 64, 128, 256
IGN_FORCE, IGN_YAW, IGN_YAW_RATE = 512, 1024, 2048
# Use position + velocity feed-forward + yaw; ignore accel and yaw rate.
TYPE_MASK_POS_VEL_YAW = IGN_AX | IGN_AY | IGN_AZ | IGN_YAW_RATE   # = 2496
# Pure position + yaw hold (used while climbing / for a stationary hover).
TYPE_MASK_POS_YAW = (IGN_VX | IGN_VY | IGN_VZ | IGN_AX | IGN_AY | IGN_AZ
                     | IGN_YAW_RATE)                              # = 2552

SETPOINT_HZ = 20.0


# --------------------------------------------------------------------------
# Pure geometry (no pymavlink)
# --------------------------------------------------------------------------

def heading_dir_ned(yaw_rad: float) -> np.ndarray:
    """Unit forward direction in NED for a PX4 yaw (rad from north, CW)."""
    return np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0])


def ned_to_enu(v) -> np.ndarray:
    n, e, d = np.asarray(v, dtype=float).ravel()
    return np.array([e, n, -d])


def enu_to_ned(v) -> np.ndarray:
    e, n, u = np.asarray(v, dtype=float).ravel()
    return np.array([n, e, -u])


def carrot_state(start_ned, yaw_rad: float, distance: float, speed: float, t: float):
    """
    Moving setpoint at time t (s since the motion began): position, velocity
    and a `done` flag. Travels along the fixed heading at `speed`, clamps at
    `distance` (velocity feed-forward drops to zero once there). Altitude
    (NED z) and yaw stay at the start values.
    """
    start = np.asarray(start_ned, dtype=float).ravel()
    d = heading_dir_ned(yaw_rad)
    if speed <= 0.0 or distance <= 0.0:
        return {"pos": start.copy(), "vel": np.zeros(3), "done": True, "travelled": 0.0}
    travelled = min(max(speed * t, 0.0), distance)
    done = speed * t >= distance
    return {"pos": start + travelled * d,
            "vel": np.zeros(3) if done else speed * d,
            "done": bool(done), "travelled": float(travelled)}


def flight_duration_s(distance: float, speed: float) -> float:
    return distance / speed if speed > 0 else 0.0


def camera_forward_horizontal_enu(R_world_cam_usd) -> np.ndarray:
    """Horizontal unit forward direction (ENU) of a USD-convention camera:
    forward is the camera's local -Z. Raises if the camera looks straight up
    or down."""
    R = np.asarray(R_world_cam_usd, dtype=float).reshape(3, 3)
    f = -R[:, 2]
    h = np.array([f[0], f[1], 0.0])
    n = np.linalg.norm(h)
    if n < 1e-6:
        raise ValueError("camera looks straight up/down; no horizontal heading")
    return h / n


def yaw_ned_from_heading_enu(dir_enu) -> float:
    """PX4 yaw (rad from north, CW) of a horizontal ENU direction."""
    e, n = dir_enu[0], dir_enu[1]
    return math.atan2(e, n)


def _point_segment(p, a, b):
    """(min distance, fraction along a->b) from point p to segment a-b."""
    ab = b - a
    L2 = float(np.dot(ab, ab))
    u = 0.0 if L2 <= 0.0 else float(np.clip(np.dot(p - a, ab) / L2, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + u * ab))), u


def clearance_report(start_enu, dir_enu, distance: float, gates: list) -> list:
    """Per gate: {name, min_dist_m (closest approach of the path segment to
    the gate center, 3D), s_closest_m (distance along the path at that
    point), end_dist_m}. gates: [{"name", "x", "y", "z"}, ...] in ENU."""
    a = np.asarray(start_enu, dtype=float).ravel()
    d = np.asarray(dir_enu, dtype=float).ravel()
    d = d / np.linalg.norm(d)
    b = a + distance * d
    out = []
    for g in gates:
        p = np.array([g["x"], g["y"], g["z"]], dtype=float)
        md, u = _point_segment(p, a, b)
        out.append({"name": g["name"], "min_dist_m": md, "s_closest_m": u * distance,
                    "end_dist_m": float(np.linalg.norm(p - b))})
    return out


def check_clearance(start_enu, dir_enu, distance: float, gates: list,
                    min_clearance_m: float):
    """(ok, report). ok is False if the path would bring the camera within
    min_clearance_m of any gate center."""
    report = clearance_report(start_enu, dir_enu, distance, gates)
    ok = all(r["min_dist_m"] >= min_clearance_m for r in report)
    return ok, report


def max_safe_distance(start_enu, dir_enu, gates: list, min_clearance_m: float,
                      limit: float = 1000.0) -> float:
    """Largest path length that keeps every gate >= min_clearance_m away
    (bisection on the monotone closest-approach distance)."""
    if check_clearance(start_enu, dir_enu, limit, gates, min_clearance_m)[0]:
        return limit
    lo, hi = 0.0, limit
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if check_clearance(start_enu, dir_enu, mid, gates, min_clearance_m)[0]:
            lo = mid
        else:
            hi = mid
    return lo


def manifest_start_and_heading(manifest_raw: dict):
    """(start_enu, dir_enu, gates) from a test_gates.json dict. Raises
    ValueError if the manifest lacks camera_pose_usd."""
    cam = manifest_raw.get("camera_pose_usd") if isinstance(manifest_raw, dict) else None
    if not cam:
        raise ValueError("manifest has no 'camera_pose_usd' (re-run spawn_test_gates.py)")
    start = np.asarray(cam["position"], dtype=float)
    direction = camera_forward_horizontal_enu(cam["R_world_cam_usd"])
    return start, direction, manifest_raw["gates"]


# --------------------------------------------------------------------------
# MAVLink flow (needs pymavlink; UNTESTED against the cluster)
# --------------------------------------------------------------------------

class Vehicle:
    """Thin wrapper over a pymavlink connection that keeps the latest
    LOCAL_POSITION_NED / ATTITUDE / HEARTBEAT and streams setpoints."""

    def __init__(self, master):
        self.m = master
        self.pos_ned = None
        self.yaw = None
        self.armed = False
        self.main_mode = None

    def poll(self):
        """Drain pending messages without blocking."""
        while True:
            msg = self.m.recv_match(blocking=False)
            if msg is None:
                return
            t = msg.get_type()
            if t == "LOCAL_POSITION_NED":
                self.pos_ned = np.array([msg.x, msg.y, msg.z])
            elif t == "ATTITUDE":
                self.yaw = float(msg.yaw)
            elif t == "HEARTBEAT" and msg.get_srcSystem() == self.m.target_system:
                self.armed = bool(msg.base_mode & MAV_MODE_FLAG_SAFETY_ARMED)
                self.main_mode = (msg.custom_mode >> 16) & 0xFF

    def command(self, cmd, *params):
        p = list(params) + [0.0] * (7 - len(params))
        self.m.mav.command_long_send(self.m.target_system, self.m.target_component,
                                     cmd, 0, *p)

    def request_interval(self, msg_id, hz):
        self.command(MAV_CMD_SET_MESSAGE_INTERVAL, msg_id, 1e6 / hz)

    def set_offboard(self):
        self.command(MAV_CMD_DO_SET_MODE, MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                     PX4_MAIN_OFFBOARD, 0)

    def set_loiter(self):
        self.command(MAV_CMD_DO_SET_MODE, MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                     PX4_MAIN_AUTO, PX4_SUB_AUTO_LOITER)

    def arm(self):
        self.command(MAV_CMD_COMPONENT_ARM_DISARM, 1)

    def send_setpoint(self, pos_ned, vel_ned, yaw, type_mask=TYPE_MASK_POS_VEL_YAW):
        self.m.mav.set_position_target_local_ned_send(
            0, self.m.target_system, self.m.target_component, MAV_FRAME_LOCAL_NED,
            type_mask, float(pos_ned[0]), float(pos_ned[1]), float(pos_ned[2]),
            float(vel_ned[0]), float(vel_ned[1]), float(vel_ned[2]), 0.0, 0.0, 0.0,
            float(yaw), 0.0)

    def stream(self, seconds, fn, until=None):
        """Call fn(t) -> (pos, vel, yaw, mask) and send it at SETPOINT_HZ for
        `seconds` (or until until() is true)."""
        t0 = time.time()
        while True:
            t = time.time() - t0
            if t >= seconds or (until is not None and until()):
                return t
            pos, vel, yaw, mask = fn(t)
            self.send_setpoint(pos, vel, yaw, mask)
            self.poll()
            time.sleep(1.0 / SETPOINT_HZ)


def _fly(args) -> int:
    # ---- optional clearance pre-check (no vehicle needed) ----
    manifest_heading = None
    if args.gates:
        with open(args.gates) as f:
            raw = json.load(f)
        raw = raw if isinstance(raw, dict) else {"gates": raw}
        try:
            start_enu, dir_enu, gates = manifest_start_and_heading(raw)
        except ValueError as e:
            print(f"REFUSING: cannot check gate clearance: {e}")
            return 2
        manifest_heading = yaw_ned_from_heading_enu(dir_enu)
        ok, report = check_clearance(start_enu, dir_enu, args.distance, gates,
                                     args.min_gate_clearance)
        print(f"gate clearance check (ENU, start {np.round(start_enu, 2)}, "
              f"min {args.min_gate_clearance} m):")
        for r in report:
            print(f"  {r['name']}: closest approach {r['min_dist_m']:.2f} m at "
                  f"{r['s_closest_m']:.2f} m along the path, {r['end_dist_m']:.2f} m "
                  f"from the end point")
        if not ok:
            safe = max_safe_distance(start_enu, dir_enu, gates, args.min_gate_clearance)
            print(f"REFUSING: --distance {args.distance} m brings the camera within "
                  f"{args.min_gate_clearance} m of a gate. Max safe distance: {safe:.2f} m.")
            return 2

    print("\nREMINDER: in another shell, start recording BEFORE this flight:\n"
          "  ros2 bag record -o gate_run1 /iris_0/front_cam/rgb "
          "/iris_0/front_cam/camera_info /drone00/state/pose\n")

    from pymavlink import mavutil  # lazy: not needed for the geometry
    master = mavutil.mavlink_connection(args.connection)
    print(f"waiting for heartbeat on {args.connection} ...")
    master.wait_heartbeat(timeout=30)
    print(f"heartbeat from system {master.target_system} component {master.target_component}")
    v = Vehicle(master)
    v.request_interval(MSG_ID_LOCAL_POSITION_NED, 20)
    v.request_interval(MSG_ID_ATTITUDE, 20)

    deadline = time.time() + 10
    while (v.pos_ned is None or v.yaw is None) and time.time() < deadline:
        v.poll()
        time.sleep(0.05)
    if v.pos_ned is None or v.yaw is None:
        print("no LOCAL_POSITION_NED / ATTITUDE within 10 s; aborting")
        return 1
    yaw = v.yaw
    print(f"start NED {np.round(v.pos_ned, 2)}, yaw {math.degrees(yaw):.1f} deg, "
          f"armed={v.armed}")
    if manifest_heading is not None:
        diff = abs((math.degrees(yaw - manifest_heading) + 180) % 360 - 180)
        if diff > 15:
            print(f"WARNING: PX4 yaw differs from the manifest camera heading by "
                  f"{diff:.0f} deg -- the drone yawed since the gates were spawned, so the "
                  f"clearance check above does not describe this flight.")
        # Re-check from where the drone actually is: the manifest start goes
        # stale after any earlier flight (2026-10-01: a 3 m run began 1 m
        # forward of it). Pegasus world and PX4 local share an origin (pose
        # topic x matched PX4 NED y to 2 cm), so NED -> ENU is enough.
        live_start = ned_to_enu(v.pos_ned)
        live_dir = ned_to_enu(heading_dir_ned(yaw))
        ok, report = check_clearance(live_start, live_dir, args.distance, gates,
                                     args.min_gate_clearance)
        worst = min(report, key=lambda r: r["min_dist_m"])
        print(f"live clearance check from ENU {np.round(live_start, 2)}: closest is "
              f"{worst['name']} at {worst['min_dist_m']:.2f} m")
        if not ok:
            safe = max_safe_distance(live_start, live_dir, gates, args.min_gate_clearance)
            print(f"REFUSING: from the current position, --distance {args.distance} m comes "
                  f"within {args.min_gate_clearance} m of {worst['name']}. Max safe "
                  f"distance: {safe:.2f} m.")
            return 2

    airborne = v.pos_ned[2] < -0.5
    hold = {"pos": v.pos_ned.copy()}
    zero = np.zeros(3)

    try:
        if not airborne:
            if args.takeoff_alt is None:
                print("not airborne and no --takeoff-alt given; hover first "
                      "(e.g. `commander takeoff`) or pass --takeoff-alt. Aborting.")
                return 1
            # Take off via offboard: stream a climb setpoint, switch to
            # OFFBOARD, arm, wait for altitude (avoids AMSL altitude handling
            # in MAV_CMD_NAV_TAKEOFF).
            target = hold["pos"] + np.array([0.0, 0.0, -args.takeoff_alt])
            v.stream(1.5, lambda t: (hold["pos"], zero, yaw, TYPE_MASK_POS_YAW))
            v.set_offboard()
            time.sleep(0.2)
            v.arm()
            print(f"climbing to {args.takeoff_alt} m ...")
            v.stream(30.0, lambda t: (target, zero, yaw, TYPE_MASK_POS_YAW),
                     until=lambda: v.pos_ned[2] <= target[2] + 0.3)
            hold["pos"] = v.pos_ned.copy()

        # Stream hover setpoints >1 s (>=10 Hz) before requesting OFFBOARD.
        print("streaming hover setpoints ...")
        v.stream(1.5, lambda t: (hold["pos"], zero, yaw, TYPE_MASK_POS_YAW))
        for attempt in range(5):
            v.set_offboard()
            v.stream(0.5, lambda t: (hold["pos"], zero, yaw, TYPE_MASK_POS_YAW),
                     until=lambda: v.main_mode == PX4_MAIN_OFFBOARD)
            if v.main_mode == PX4_MAIN_OFFBOARD:
                break
        else:
            print("PX4 did not enter OFFBOARD; aborting")
            return 1
        print(f"OFFBOARD. hovering {args.hover_s} s ...")
        v.stream(args.hover_s, lambda t: (hold["pos"], zero, yaw, TYPE_MASK_POS_YAW))

        start = v.pos_ned.copy() if args.start_from_measured else hold["pos"].copy()
        print(f"flying {args.distance} m at {args.speed} m/s along heading "
              f"{math.degrees(yaw):.1f} deg ...")

        def carrot(t):
            s = carrot_state(start, yaw, args.distance, args.speed, t)
            hold["pos"] = s["pos"]
            return s["pos"], s["vel"], yaw, TYPE_MASK_POS_VEL_YAW

        v.stream(flight_duration_s(args.distance, args.speed), carrot)
        end = carrot_state(start, yaw, args.distance, args.speed, 1e9)["pos"]
        hold["pos"] = end
        print(f"holding end point {args.hover_s} s ...")
        v.stream(args.hover_s, lambda t: (end, zero, yaw, TYPE_MASK_POS_VEL_YAW))
        print("done: switching to AUTO.LOITER. You can stop `ros2 bag record` now.")
    except KeyboardInterrupt:
        print("\nCtrl-C: commanding AUTO.LOITER")
    finally:
        for _ in range(5):
            v.set_loiter()
            time.sleep(0.1)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--connection", default="udpin:0.0.0.0:14540")
    ap.add_argument("--distance", type=float, default=3.0, help="meters (default 3)")
    ap.add_argument("--speed", type=float, default=1.0, help="m/s (default 1)")
    ap.add_argument("--hover-s", type=float, default=5.0,
                    help="hover before and after the flight, seconds (default 5)")
    ap.add_argument("--takeoff-alt", type=float, default=None,
                    help="if the drone is on the ground: arm and climb to this height (m)")
    ap.add_argument("--gates", default=None,
                    help="test_gates.json: check gate clearance along the path first")
    ap.add_argument("--min-gate-clearance", type=float, default=2.0)
    ap.add_argument("--start-from-measured", action="store_true",
                    help="start the carrot at the measured position after the hover instead "
                         "of the held setpoint (may cause a small jump if the drone drifted)")
    args = ap.parse_args()
    if args.distance <= 0 or args.speed <= 0:
        sys.exit("--distance and --speed must be positive")
    sys.exit(_fly(args))


if __name__ == "__main__":
    main()
