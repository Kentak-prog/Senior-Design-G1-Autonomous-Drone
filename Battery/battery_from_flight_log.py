#!/usr/bin/env python3
"""Run the team's Starling 2 Max Thevenin battery model (Battery/Battery_Discharge_Node.py)
on a flight_log.csv from px4/fly_mpc_offboard.py. No ROS, no px4_msgs: numpy-free, stdlib only.

    python3 battery_from_flight_log.py flight_log.csv --time-scale 0.44
    python3 battery_from_flight_log.py flight_log.csv --time-scale 0.44 --out run_power_dynamics.csv

The constants and equations are copied from Battery_Discharge_Node.py (same cell curve, same
loads, same Coulomb counting). Differences, all deliberate:
  * the throttle comes from the log's cmd_thrust_norm (the thrust the NMPC commanded, 0..1),
    not from PX4's actuator outputs. Only the MPC ("trajectory") rows have it; the reposition,
    hover and settle phases are PX4's own controller and are NOT included.
  * time: the log's `t` is wall-clock seconds, the sim runs at --time-scale x real time, so one
    log row is 0.02 * time_scale seconds of simulated flight. Energy is per simulated second.
The output CSV has the same columns as the node's run_power_dynamics.csv (plus SoC), so
Battery/Battery_Discharge_Plotter.py reads it unchanged.
"""
import argparse
import csv
import sys

S_COUNT = 4
CAPACITY_AH = 3.0
R_TH = 0.070
R_BOARD = 14.6
R_CAMERAS = 73.0
R_ALFA = 109.5
P_IDLE_SYSTEM = 4.0 * (10.0 * 0.3)
P_MAX_MOTOR_AERO = 288.0


def cell_voc(soc):
    soc = max(0.0, min(soc, 1.0))
    v = (4.15 - 0.4 * (1 - soc) - 0.3 * (1 - soc) ** 2
         + 0.5 * (1 - soc) ** 3 - 0.8 * (1 - soc) ** 4 + 0.15 * soc)
    return max(3.0, min(v, 4.2))


def step(remaining_as, total_as, throttle, dt):
    """One update of the node's actuator_callback. Returns the new remaining charge and a row."""
    soc = remaining_as / total_as
    v_th = cell_voc(soc) * S_COUNT
    throttle = max(0.0, min(throttle, 1.0))
    p_motors = (throttle ** 3) * P_MAX_MOTOR_AERO + P_IDLE_SYSTEM
    r_motors = (v_th ** 2) / p_motors if p_motors > 0 else 1e6
    inv = 1.0 / r_motors + 1.0 / R_BOARD + 1.0 / R_CAMERAS + 1.0 / R_ALFA
    i_load = v_th / (R_TH + 1.0 / inv)
    v_esc = v_th - i_load * R_TH
    p_payload = (v_esc ** 2) * (1.0 / R_BOARD + 1.0 / R_CAMERAS + 1.0 / R_ALFA)
    remaining_as = max(0.0, remaining_as - i_load * dt)
    return remaining_as, (soc, v_th, v_esc, i_load, p_motors, p_payload)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="flight_log.csv")
    ap.add_argument("--time-scale", type=float, default=0.44,
                    help="sim speed vs real time during the flight (the run prints it; 0.40-0.45 on our cluster)")
    ap.add_argument("--soc0", type=float, default=1.0, help="starting state of charge, 0..1 (default 1.0)")
    ap.add_argument("--phase", default="trajectory", help="log phase to use (default trajectory)")
    ap.add_argument("--out", default="run_power_dynamics.csv")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.log)) if r["phase"] == args.phase and r["cmd_thrust_norm"] != ""]
    if len(rows) < 2:
        sys.exit(f"no rows with phase={args.phase!r} and cmd_thrust_norm in {args.log}")

    total_as = CAPACITY_AH * 3600.0
    remaining = total_as * args.soc0
    out, t_sim, prev_t = [], 0.0, None
    for r in rows:
        t_wall = float(r["t"])
        dt = 0.0 if prev_t is None else (t_wall - prev_t) * args.time_scale
        prev_t = t_wall
        t_sim += dt
        remaining, (soc, v_th, v_esc, i, pm, pp) = step(remaining, total_as, float(r["cmd_thrust_norm"]), dt)
        out.append((t_sim, v_th, v_esc, i, pm, pp, soc))

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Time", "V_th", "V_esc", "I_load", "P_motors", "P_payload", "SoC"])
        w.writerows(out)

    dur = out[-1][0] - out[0][0]
    cur = [o[3] for o in out]
    p_tot = [o[4] + o[5] for o in out]
    wh = sum(p_tot[k] * (out[k][0] - out[k - 1][0]) for k in range(1, len(out))) / 3600.0
    ah = (out[0][6] - out[-1][6]) * CAPACITY_AH
    print(f"rows used            : {len(out)} ({args.phase} phase), time scale {args.time_scale}")
    print(f"simulated duration   : {dur:.1f} s")
    print(f"charge used          : {ah * 1000:.1f} mAh ({(out[0][6] - out[-1][6]) * 100:.2f} % of {CAPACITY_AH * 1000:.0f} mAh)")
    print(f"energy used          : {wh:.2f} Wh")
    print(f"current mean / peak  : {sum(cur) / len(cur):.2f} / {max(cur):.2f} A")
    print(f"power mean / peak    : {sum(p_tot) / len(p_tot):.1f} / {max(p_tot):.1f} W")
    print(f"V_th start / end     : {out[0][1]:.2f} / {out[-1][1]:.2f} V")
    print(f"V_esc minimum (sag)  : {min(o[2] for o in out):.2f} V")
    print(f"written              : {args.out}")


if __name__ == "__main__":
    main()
