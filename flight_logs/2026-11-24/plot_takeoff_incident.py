"""
Figure of the 2026-11-24 Starling 2 Max crashes, from QGroundControl's telemetry log (.tlog), in the
style of flight_logs/2026-11-03/plot_hover_identification.py. See px4/INCIDENT_2026-11-24_TAKEOFF_FLYAWAY.md.

Panels
  a. Autonomous takeoff (px4/takeoff_hover.py): height from PX4's estimate vs the barometer, the
     planned climb, and the events PX4 reported
  b. The earlier pilot-flown flight (Altitude mode): the same comparison
  c. Climb rate: PX4's estimate vs the barometer's, both flights (should lie on y = x)
  d. Thrust from the measured rotor speeds (T = k_f * sum(w^2), k_f from 2026-11-03) over weight,
     with PX4's throttle command and the pilot's throttle stick, autonomous flight
  e. Per-rotor speed vs the 2026-11-03 hover speed, autonomous flight

QGroundControl streams at its defaults: ESC RPM and the position estimate ~1 Hz, VFR_HUD ~4 Hz,
RC ~5 Hz. VFR_HUD's altitude is barometric here (it matches ALTITUDE.altitude_monotonic) while its
climb rate is the estimate's.

Usage:
    python plot_takeoff_incident.py qgc_2026-11-24_14-55-05.tlog [-o out.png]
Needs numpy and matplotlib only (a small .tlog reader is included; no pymavlink).
"""
import argparse
import struct

import matplotlib.pyplot as plt
import numpy as np

M, G = 0.573, 9.81               # kg (measured 2026-11-03)
KF = 4.383e-6                    # N/(rad/s)^2, identified 2026-11-03
HOVER_RPM = 5407                 # 2026-11-03 mean hover speed
RPM2RAD = 2 * np.pi / 60

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
EST, BARO = SERIES[1], SERIES[0]          # estimate orange, barometer blue in every panel
ROTOR_NAMES = ["ESC 0", "ESC 1", "ESC 2", "ESC 3"]

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "legend.fontsize": 8,
})

PX4_MAIN = {1: "Manual", 2: "Altitude", 3: "Position", 4: "Auto", 5: "Acro", 6: "Offboard", 7: "Stabilized"}


# ------------------------------------------------------------------------------------- reading

def packets(path):
    """(t_s, sysid, compid, msgid, payload) for every MAVLink v1/v2 packet of a QGC .tlog
    (8-byte big-endian microsecond timestamp before each packet)."""
    data = open(path, "rb").read()
    i, n = 0, len(data)
    while i + 8 < n:
        ts = struct.unpack(">Q", data[i:i + 8])[0] / 1e6
        j = i + 8
        if data[j] == 0xFD and j + 10 <= n:
            ln, flags = data[j + 1], data[j + 2]
            s, c = data[j + 5], data[j + 6]
            mid = data[j + 7] | (data[j + 8] << 8) | (data[j + 9] << 16)
            p = data[j + 10:j + 10 + ln]
            end = j + 12 + ln + (13 if flags & 1 else 0)
        elif data[j] == 0xFE and j + 6 <= n:
            ln = data[j + 1]
            s, c, mid = data[j + 3], data[j + 4], data[j + 5]
            p = data[j + 6:j + 6 + ln]
            end = j + 8 + ln
        else:
            i += 1
            continue
        yield ts, s, c, mid, p
        i = end


def _pad(p, n):
    return p + b"\0" * max(0, n - len(p))


def load(path):
    """Arrays of the messages used, time in seconds since the log start."""
    out = {k: [] for k in ("pos", "hud", "rc", "esc", "att", "hb", "msg")}
    t0 = None
    for ts, s, c, mid, p in packets(path):
        t0 = ts if t0 is None else t0
        t = ts - t0
        if s != 1:
            continue
        if mid == 32:                                                   # LOCAL_POSITION_NED
            _, x, y, z, vx, vy, vz = struct.unpack("<Iffffff", _pad(p, 28))
            out["pos"].append((t, z, vz))
        elif mid == 74:                                                 # VFR_HUD
            _, _, alt, climb, _, thr = struct.unpack("<ffffhH", _pad(p, 20))
            out["hud"].append((t, alt, climb, thr))
        elif mid == 65:                                                 # RC_CHANNELS
            v = struct.unpack("<I18HBB", _pad(p, 42))
            # channel 3 = throttle, 5 = kill switch (1011 off / 1988 engaged: it flips at every
            # "Kill-switch engaged" message), 6 = mode switch (1011 Altitude, 1500 Position, 1988 Offboard)
            out["rc"].append((t, v[3], v[5], v[6]))
        elif mid == 291:                                                # ESC_STATUS
            f = struct.unpack("<Q4i4f4fB", _pad(p, 57))
            out["esc"].append((t, *f[1:5]))
        elif mid == 30:                                                 # ATTITUDE
            _, roll, pitch, *_ = struct.unpack("<I6f", _pad(p, 28))
            out["att"].append((t, np.degrees(roll), np.degrees(pitch)))
        elif mid == 0 and c == 1:                                       # HEARTBEAT (PX4)
            custom, _, ap, base, *_ = struct.unpack("<IBBBBB", _pad(p, 9))
            if ap == 12:
                out["hb"].append((t, (custom >> 16) & 0xFF, bool(base & 128)))
        elif mid == 253:                                                # STATUSTEXT
            out["msg"].append((t, _pad(p, 51)[1:51].split(b"\0")[0].decode(errors="replace").strip()))
    return {k: (np.array(v, dtype=float) if k not in ("msg",) else v) for k, v in out.items()}


def first_msg(d, text, after):
    return next(t for t, m in d["msg"] if t >= after and text in m)


def flight(d, arm_text, t_after, before=4.0, after=14.0):
    """Window around one flight: arm time and the per-message arrays cut and re-zeroed at arming."""
    ta = first_msg(d, arm_text, t_after)
    lo, hi = ta - before, ta + after

    def cut(a):
        a = a[(a[:, 0] >= lo) & (a[:, 0] <= hi)].copy()
        a[:, 0] -= ta
        return a

    f = {k: cut(d[k]) for k in ("pos", "hud", "rc", "esc", "att", "hb")}
    pre = f["hud"][f["hud"][:, 0] < 0]
    f["baro0"] = float(np.median(pre[:, 1]))                            # barometer on the ground
    pre_pos = f["pos"][f["pos"][:, 0] < 0]
    f["z0"] = float(np.median(pre_pos[:, 1]))
    f["events"] = [(t - ta, m) for t, m in d["msg"] if lo <= t <= hi]
    f["t_arm"] = ta
    return f


def mode_change(f, main_mode):
    """First time after arming PX4's heartbeat reports this main mode (2 = Altitude), or None."""
    hb = f["hb"]
    k = np.flatnonzero((hb[:, 0] > 0) & (hb[:, 1] == main_mode))
    return float(hb[k[0], 0]) if k.size else None


# -------------------------------------------------------------------------------------- figure

def events_lines(ax, f, wanted, y_top, ha_map=None):
    seen = set()
    for t, m in f["events"]:
        for key, label in wanted:
            if key in m and key not in seen and t >= -0.5:
                seen.add(key)
                ax.axvline(t, color=MUTED, lw=0.8, ls=(0, (3, 3)))
                ha = (ha_map or {}).get(key, "left")
                ax.text(t + (0.08 if ha == "left" else -0.08), y_top, label, fontsize=7.5, color=INK2,
                        va="top", ha=ha)
                y_top -= 0.55


def height_panel(ax, f, title, plan=None, ylim=(-1.6, 6.0), wanted=(), ha_map=None, end_t=None):
    hud, pos = f["hud"], f["pos"]
    ax.plot(hud[:, 0], hud[:, 1] - f["baro0"], color=BARO, lw=2, label="barometer (independent of the VIO)")
    ax.plot(pos[:, 0], -(pos[:, 1] - f["z0"]), color=EST, lw=2, marker="o", ms=3.5,
            label="PX4's height estimate (what it controls)")
    if plan is not None:
        ax.plot(*plan, color=INK, lw=1.2, ls=(0, (4, 3)), label="commanded climb (takeoff_hover.py)")
    ax.axhline(0, color=AXIS, lw=1)
    ax.axhspan(ylim[0], 0, color=EST, alpha=0.06, lw=0)
    ax.text(f["pos"][0, 0] + 0.1, ylim[0] + 0.15, "below the floor", fontsize=7.5, color=INK2)
    events_lines(ax, f, wanted, ylim[1] - 0.1, ha_map)
    ax.set_ylim(*ylim)
    ax.set_xlim(-2, end_t if end_t else f["hud"][-1, 0])
    ax.set_xlabel("Time since arming (s)")
    ax.set_ylabel("Height above take-off (m)")
    ax.set_title(title)
    ax.legend(loc="upper left", ncol=1, bbox_to_anchor=(0.0, 0.62))


def climb_rates(f):
    """(estimate climb, barometer climb) pairs at the HUD rate, airborne part, 1 s windows."""
    h = f["hud"]
    t, alt, climb = h[:, 0], h[:, 1], h[:, 2]
    k = (t > 1.5) & (t < t.max() - 0.5)
    tb = np.arange(t[k].min(), t[k].max() - 1.0, 0.5)
    est, bar = [], []
    for a in tb:
        m = (t >= a) & (t < a + 1.0)
        if m.sum() >= 3:
            bar.append(np.polyfit(t[m], alt[m], 1)[0])
            est.append(climb[m].mean())
    return np.array(est), np.array(bar)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tlog")
    ap.add_argument("-o", "--out", default="takeoff_incident_2026-11-24.png")
    a = ap.parse_args()
    d = load(a.tlog)

    auto = flight(d, "Armed by external command", 0, before=3, after=12.5)
    pilot = flight(d, "Armed by RC", 0, before=3, after=18.5)

    # the commanded climb: 1.0 m in 1.875 * 1.0 / 0.3 s, from PX4 entering OFFBOARD (= arming here)
    T = 1.875 * 1.0 / 0.3
    tt = np.linspace(0, T + 5, 200)
    s = np.clip(tt / T, 0, 1)
    plan = (tt, 1.0 * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5))

    fig = plt.figure(figsize=(12, 12.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1, 1], hspace=0.42, wspace=0.32,
                          left=0.07, right=0.98, top=0.915, bottom=0.115)
    fig.text(0.07, 0.965, "Starling 2 Max, 2026-11-24: PX4's height estimate did not follow the drone, two crashes",
             fontsize=15, fontweight="bold", ha="left")
    fig.text(0.07, 0.945, "From QGroundControl's telemetry log. 573 g, C29, PX4 on VOXL2, indoors. "
             "PX4's height estimate (EKF) vs the barometer, an independent height reference.",
             ha="left", color=INK2, fontsize=9.5)

    # a: autonomous takeoff
    ax = fig.add_subplot(gs[0, :])
    height_panel(ax, auto, "a  Autonomous takeoff (takeoff_hover.py, 1.0 m at 0.3 m/s): PX4 believed it was sinking",
                 plan=plan, wanted=[("Armed by", "armed, OFFBOARD"), ("Takeoff detected", "takeoff detected"),
                                    ("Failsafe activated", "failsafe")], end_t=12.0)
    # PX4's own mode change (heartbeat), the pilot's throttle stick leaving center and the crash: event
    # lines with labels placed clear of the curves (ha "right" = label left of its line)
    t_alt = mode_change(auto, 2)
    thr = auto["rc"]
    k = np.flatnonzero((thr[:, 0] > 0) & (thr[:, 1] < 1400))
    t_crash = next((t for t, m in auto["events"] if "Attitude failure" in m and t > 0), None)
    for t, label, y, ha in ((t_alt, "Altitude mode", 4.25, "left"),
                            (thr[k[0], 0] if k.size else None, "pilot: throttle down", 4.25, "right"),
                            (t_crash, "attitude failure\n(crash)", 2.5, "right")):
        if t is not None:
            ax.axvline(t, color=MUTED, lw=0.8, ls=(0, (3, 3)))
            ax.text(t + (0.08 if ha == "left" else -0.08), y, label, fontsize=7.5, color=INK2, va="top", ha=ha)
    hud = auto["hud"]
    i = np.argmax(hud[:, 1])
    ax.annotate(f"barometer: +{hud[i, 1] - auto['baro0']:.1f} m", (hud[i, 0], hud[i, 1] - auto["baro0"]),
                (hud[i, 0] - 3.2, 5.2), fontsize=8, color=BARO,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    p = auto["pos"]
    est = -(p[:, 1] - auto["z0"])
    before = p[:, 0] < (t_alt if t_alt is not None else 0) + 5.5          # lowest point before the crash
    j = np.flatnonzero(before)[np.argmin(est[before])]
    ax.annotate(f"estimate: {est[j]:+.1f} m", (p[j, 0], est[j]), (p[j, 0] - 1.6, -1.45), fontsize=8, color=EST,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))

    # b: pilot flight
    ax = fig.add_subplot(gs[1, 0])
    height_panel(ax, pilot, "b  Earlier pilot flight (Altitude mode): estimate fell 6.8 m, barometer flat",
                 ylim=(-7.5, 3.0), wanted=[("Takeoff detected", "takeoff"),
                                           ("Attitude failure", "attitude failure"),
                                           ("Kill-switch engaged", "kill switch")],
                 ha_map={"Kill-switch engaged": "left"}, end_t=18.0)
    ax.legend().remove()

    # c: climb rates
    ax = fig.add_subplot(gs[1, 1])
    lim = 4.5
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(2.3, 2.75, "y = x\n(a healthy estimate)", color=INK2, fontsize=8, ha="left")
    for f, name, col in ((auto, "autonomous takeoff", SERIES[1]), (pilot, "pilot flight", SERIES[2])):
        est, bar = climb_rates(f)
        ax.scatter(bar, est, s=22, color=col, edgecolor=SURFACE, linewidth=0.6, label=name, zorder=3)
        if len(est) > 3:
            r = np.corrcoef(bar, est)[0, 1]
            ax.text(-lim + 0.2, (-3.4 if col == SERIES[1] else -4.0), f"{name}: r = {r:+.2f}", color=col, fontsize=8)
    ax.axhline(0, color=AXIS, lw=1)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Climb rate from the barometer (m/s, 1 s fits)")
    ax.set_ylabel("Climb rate PX4 estimated (m/s)")
    ax.set_title("c  Estimated vs barometric climb rate: no agreement")
    ax.text(lim - 0.2, -1.9, "climbing,\nestimated sinking", color=INK2, fontsize=7.5, ha="right")
    ax.text(-lim + 0.2, 1.6, "sinking,\nestimated climbing", color=INK2, fontsize=7.5, ha="left")
    ax.legend(loc="upper left")

    # d: thrust from rotor speed / weight, with throttle
    ax = fig.add_subplot(gs[2, 0])
    e = auto["esc"]
    w = e[:, 1:5] * RPM2RAD
    tw = KF * (w ** 2).sum(1) / (M * G)
    air = e[:, 0] > -0.5
    ax.plot(e[air, 0], tw[air], color=INK, lw=2, marker="o", ms=4.5, label="thrust / weight (k_f Σω², ESC RPM)")
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.text(-1.9, 1.04, "thrust = weight (hover)", fontsize=7.5, color=INK2)
    h = auto["hud"]
    ax.plot(h[:, 0], h[:, 3] / 100 * 3.0, color=SERIES[3], lw=1.6, label="PX4 throttle command (right axis)")
    r = auto["rc"]
    ax.plot(r[:, 0], (r[:, 1] - 1000) / 1000 * 3.0, color=SERIES[2], lw=1.6, ls=(0, (4, 2)),
            label="pilot's throttle stick (right axis)")
    ax.plot(r[:, 0], (r[:, 2] - 1000) / 1000 * 3.0, color=SERIES[1], lw=2.2, ls=(0, (1, 1.5)),
            label="kill switch, ch 5 (right axis; 100% = engaged)")
    ax.text(-1.8, 1.7, "kill switch: 0% (off) for the whole flight, until the log ends at the crash",
            fontsize=7.5, color=SERIES[1], ha="left")
    if t_alt is not None:
        ax.axvspan(t_alt, 12, color=EST, alpha=0.06, lw=0)
        ax.text(t_alt + 0.1, 2.85, "Altitude mode (after the failsafe)", fontsize=7.5, color=INK2, va="top")
    ax.set_ylim(0, 3.0)
    ax.set_xlim(-2, 12)
    sec = ax.secondary_yaxis("right", functions=(lambda y: y / 3.0 * 100, lambda p: p / 100 * 3.0))
    sec.set_ylabel("Throttle (%)")
    sec.tick_params(colors=MUTED)
    ax.set_xlabel("Time since arming (s)")
    ax.set_ylabel("Thrust / weight")
    ax.set_title("d  Thrust from the measured rotor speeds (autonomous takeoff)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.88))

    # e: per-rotor RPM
    ax = fig.add_subplot(gs[2, 1])
    for k in range(4):
        ax.plot(e[:, 0], e[:, 1 + k], color=SERIES[k], lw=1.6, marker="o", ms=3.5, label=ROTOR_NAMES[k])
    ax.axhline(HOVER_RPM, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.text(-1.9, HOVER_RPM + 150, f"hover speed on 2026-11-03: {HOVER_RPM:,} RPM", fontsize=7.5, color=INK2)
    ax.set_xlim(-2, 12)
    ax.set_ylim(0, max(12500, np.nanmax(e[:, 1:5]) * 1.1))
    ax.set_xlabel("Time since arming (s)")
    ax.set_ylabel("Rotor speed (RPM)")
    ax.set_title("e  Rotor speeds (autonomous takeoff, ~1 Hz from QGroundControl)")
    ax.legend(loc="upper left", ncol=2)
    k1 = (e[:, 0] > 3.5) & (e[:, 0] < 9.5)
    if k1.any():
        gap = float(np.mean(e[k1, 2] - e[k1][:, [1, 3, 4]].mean(1)))
        ax.annotate(f"ESC 1 ~{round(gap, -2):,.0f} RPM above the others", (6.0, float(e[k1, 2].mean())), (3.2, 12500),
                    fontsize=7.5, color=SERIES[1], arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))

    fig.text(0.07, 0.018,
             "PX4's height estimate did not follow the barometer in either flight (panels a-c). On the autonomous takeoff it read the "
             "drone sinking while it climbed ~5.5 m,\nso its height controller held thrust above weight (panel d). The commanded climb "
             "(panel a, dashed) was correct. ESC 1 ran ~2,100 RPM above the others from 4 s (panel e).\nThe kill switch (ch 5) stayed off "
             "for the whole flight; the mode switch only offers Altitude/Position, which use the same estimate. Telemetry ends at the "
             "crash (11.4 s).\nCause of the bad estimate: not yet identified (PX4 logs and parameters on the drone).",
             ha="left", color=INK2, fontsize=8.5)
    fig.savefig(a.out, dpi=160, facecolor=SURFACE)
    print(f"saved {a.out}")
    print(f"autonomous: arm at log {auto['t_arm']:.2f} s; pilot: arm at log {pilot['t_arm']:.2f} s")


if __name__ == "__main__":
    main()
