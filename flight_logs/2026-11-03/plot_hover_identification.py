"""
Figure of the 2026-11-03 Starling 2 Max hover flights: what the logs show and the
rotor parameters identified from them (workspace/STARLING2_DYNAMICS.md, section 4).

Panels
  a. Height above take-off vs time, both flights, with steady-hover windows and events
  b. Thrust (accelerometer) vs sum of squared rotor speeds, with the hover k_f line
  c. Per-rotor hover RPM relative to the four-motor mean (yaw trim / balance)
  d. Thrust per rotor vs RPM: measured band, hover, PX4 60% cap, ESC limit
  e. Motor lag: commanded-to-measured RPM cross-correlation per rotor (flight 2)
  f. Identified values

Usage:
    python plot_hover_identification.py <23_20_53.ulg> <23_34_32.ulg> [-o out.png]
Needs pyulog, numpy, matplotlib.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from pyulog import ULog

M = 0.573                       # kg, measured
G = 9.81
RPM2RAD = 2 * np.pi / 60

# Reference data-viz palette (light): categorical slots 1-4, ink, chrome.
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
ROTOR_NAMES = ["0 front-right (CCW)", "1 back-left (CCW)", "2 front-left (CW)", "3 back-right (CW)"]

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


def load(path):
    u = ULog(path)
    t0 = np.int64(u.start_timestamp)
    d = {x.name: x.data for x in u.data_list if x.multi_id == 0}

    def ts(name):
        return (d[name]["timestamp"].astype(np.int64) - t0) / 1e6

    p = u.initial_parameters
    esc = d["esc_status"]
    te = ts("esc_status")
    rpm_esc = np.stack([esc[f"esc[{k}].esc_rpm"].astype(float) for k in range(4)], 1)
    # ESC index -> PX4 rotor (VOXL_ESC_FUNCn: 101 = motor 1 = CA_ROTOR0)
    rotor_of_esc = [int(p[f"VOXL_ESC_FUNC{k + 1}"]) - 101 for k in range(4)]
    rpm = np.zeros_like(rpm_esc)
    for k, r in enumerate(rotor_of_esc):
        rpm[:, r] = rpm_esc[:, k]

    lp = d["vehicle_local_position"]
    tl = ts("vehicle_local_position")
    z_up = -lp["z"]
    alt0 = np.median(z_up[tl < 3])
    vs = d["vehicle_status"]
    tv = ts("vehicle_status")
    on = lambda tt, src_t, v: np.interp(tt, src_t, v)
    nav = on(te, tv, vs["nav_state"].astype(float))
    fs = on(te, tv, vs["failsafe"].astype(float))
    hag = on(te, tl, z_up) - alt0
    vz = on(te, tl, -lp["vz"])
    vxy = on(te, tl, np.hypot(lp["vx"], lp["vy"]))
    acc = d["vehicle_acceleration"]
    az = on(te, ts("vehicle_acceleration"), acc["xyz[2]"])

    steady = (np.round(nav) == 2) & (fs < 0.5) & (np.abs(vz) < 0.15) & (vxy < 0.15) & (hag > 0.5) \
        & (rpm.min(1) > 3000)
    air = (hag > 0.3) & (rpm.min(1) > 3000) & (fs < 0.5)

    # motor lag: commanded RPM (actuator_outputs_debug, ESC order) vs measured
    ao = d["actuator_outputs_debug"]
    tao = ts("actuator_outputs_debug")
    fs_hz = 200.0
    tg = np.arange(max(tao[0], te[0]) + 1, min(tao[-1], te[-1]) - 1, 1 / fs_hz)
    airg = np.interp(tg, te, air.astype(float)) > 0.99
    lags = np.arange(0, int(0.15 * fs_hz))
    xcorr = np.zeros((4, len(lags)))
    for k, r in enumerate(rotor_of_esc):
        cmd = np.interp(tg, tao, ao[f"output[{k}]"])[airg]
        meas = np.interp(tg, te, rpm_esc[:, k])[airg]
        cmd, meas = cmd - cmd.mean(), meas - meas.mean()
        xcorr[r] = [np.dot(cmd[: len(cmd) - L], meas[L:])
                    / (np.linalg.norm(cmd[: len(cmd) - L]) * np.linalg.norm(meas[L:])) for L in lags]

    # events
    rc = d["input_rc"]
    trc = ts("input_rc")
    lost = rc["rc_lost"].astype(int)
    rc_lost_t = trc[1:][(lost[1:] == 1) & (lost[:-1] == 0)]
    navs = vs["nav_state"].astype(int)
    land_t = tv[1:][(navs[1:] == 18) & (navs[:-1] != 18)]
    arm = vs["arming_state"].astype(int)
    disarm_t = tv[1:][(arm[1:] != 2) & (arm[:-1] == 2)]

    return dict(te=te, rpm=rpm, hag=hag, az=az, steady=steady, air=air, lags=lags / fs_hz,
                xcorr=xcorr, rc_lost=rc_lost_t, land=land_t, disarm=disarm_t, tl=tl,
                hag_lp=z_up - alt0, params=p)


def windows(mask, t):
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if start is not None and (not m or i == len(mask) - 1):
            if t[i - 1] - t[start] >= 5.0:
                out.append((t[start], t[i - 1]))
            start = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flight1")
    ap.add_argument("flight2")
    ap.add_argument("-o", "--out", default="hover_identification.png")
    a = ap.parse_args()
    F = [load(a.flight1), load(a.flight2)]
    names = ["Flight 1 (23:20)", "Flight 2 (23:34)"]

    # identification
    w2 = np.concatenate([(f["rpm"][f["steady"]] * RPM2RAD) ** 2 for f in F])
    kf = M * G / (4 * w2.mean())
    rpm_h = np.sqrt(w2.mean()) / RPM2RAD
    p = F[0]["params"]
    rmin, rmax = p["VOXL_ESC_RPM_MIN"], p["VOXL_ESC_RPM_MAX"]
    fmdl = p["THR_MDL_FAC"]
    x_cap = (-(1 - fmdl) + np.sqrt((1 - fmdl) ** 2 + 4 * fmdl * p["MPC_THR_MAX"])) / (2 * fmdl)
    rpm_cap = rmin + (rmax - rmin) * x_cap
    thrust = lambda r: kf * (r * RPM2RAD) ** 2

    fig = plt.figure(figsize=(12, 12.6))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.22, left=0.07, right=0.97, top=0.895, bottom=0.075)
    fig.text(0.07, 0.965, "Starling 2 Max hover flights, 2026-11-03: rotor identification", ha="left",
             fontsize=14, fontweight="bold")
    fig.text(0.07, 0.942, f"573 g, C29, PX4 on VOXL2, Position mode indoors. Two logs, "
             f"{sum(f['steady'].sum() for f in F) / 199:.0f} s of steady hover in total.",
             ha="left", color=INK2, fontsize=9.5)

    # a. timeline; steady-hover windows as strips under the baseline
    ax = fig.add_subplot(gs[0, :])
    for i, f in enumerate(F):
        y = -0.18 - 0.14 * i
        for s, e in windows(f["steady"], f["te"]):
            ax.plot([s, e], [y, y], color=SERIES[i], lw=5, solid_capstyle="round")
        ax.plot(f["tl"], f["hag_lp"], color=SERIES[i], lw=2, label=names[i])
    ax.text(1, -0.25, "steady-hover\nwindows", fontsize=7.5, color=INK2, va="center")
    ax.axhline(0, color=AXIS, lw=1)
    f1, f2 = F
    notes = [  # (time, text, label height)
        (f2["rc_lost"][0], "RC lost 0.5 s → Land,\npilot resumed", 2.95),
        (f1["disarm"][0] if len(f1["disarm"]) else 107.2, "flight 1: kill switch\n(PX4 hadn't detected landing)", 2.95),
        (f2["rc_lost"][1], "flight 2: RC link lost\n→ auto Land", 2.95),
        (192.0, "touches down on\na table (~0.8 m)", 2.35),
        (203.0, "caught and lifted\nby hand", 1.75),
    ]
    for t, txt, yt in notes:
        ax.axvline(t, color=MUTED, lw=0.8, ls=(0, (3, 3)))
        ha = "right" if t > 180 else "left"
        ax.text(t + (-1.5 if ha == "right" else 1.5), yt, txt, fontsize=7.5, color=INK2, va="top", ha=ha)
    ax.set_xlim(0, 216)
    ax.set_ylim(-0.5, 3.2)
    ax.set_xlabel("Time since log start (s)")
    ax.set_ylabel("Height above take-off (m)")
    ax.set_title("a  Flight timeline")
    ax.legend(loc="upper left", ncols=2)
    ax.text(140, 1.42, "flight 2 hover, 1.30 m", color=INK2, fontsize=8)
    ax.text(62, 0.98, "flight 1 hover, 1.12 m", color=INK2, fontsize=8)

    # b. thrust vs sum w^2 (0.5 s bins)
    ax = fig.add_subplot(gs[1, 0])
    for i, f in enumerate(F):
        m = f["air"]
        t = f["te"][m]
        sw2 = ((f["rpm"][m] * RPM2RAD) ** 2).sum(1)
        T = -M * f["az"][m]
        b = np.floor(t / 0.5)
        _, inv = np.unique(b, return_inverse=True)
        n = np.bincount(inv)
        ax.scatter(np.bincount(inv, sw2) / n / 1e6, np.bincount(inv, T) / n, s=14, color=SERIES[i],
                   edgecolor=SURFACE, linewidth=0.6, label=names[i], zorder=3)
    xs = np.linspace(1.215, 1.375, 10)
    ax.plot(xs, kf * xs * 1e6, color=INK, lw=1.5, zorder=4)
    ax.axhline(M * G, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.text(1.212, M * G + 0.015, f"weight mg = {M * G:.2f} N", color=INK2, fontsize=8)
    ax.text(1.343, 5.72, f"T = k_f Σω²\nk_f = {kf:.3e} N/(rad/s)²", color=INK, fontsize=8.5, ha="left")
    ax.set_xlim(1.21, 1.42)
    ax.set_ylim(5.2, 6.15)
    ax.set_xlabel("Σ ω² over 4 rotors (×10⁶ rad²/s²)")
    ax.set_ylabel("Thrust from accelerometer, −m·a_z (N)")
    ax.set_title("b  Thrust vs rotor speed² (0.5 s averages, airborne)")
    ax.legend(loc="upper left")

    # c. per-rotor hover RPM deviation
    ax = fig.add_subplot(gs[1, 1])
    width = 0.36
    x = np.arange(4)
    for i, f in enumerate(F):
        r = f["rpm"][f["steady"]].mean(0)
        dev = r - r.mean()
        bars = ax.bar(x + (i - 0.5) * (width + 0.02), dev, width, color=SERIES[i], label=names[i],
                      edgecolor=SURFACE, linewidth=1)
        for xb, v in zip(x + (i - 0.5) * (width + 0.02), dev):
            ax.text(xb, v + (4 if v >= 0 else -4), f"{v:+.0f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=7.5, color=INK2)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks(x, ROTOR_NAMES, fontsize=8)
    ax.tick_params(axis="x", colors=INK2)
    ax.set_ylim(-75, 75)
    ax.set_ylabel("Hover RPM minus 4-rotor mean")
    ax.set_title("c  Per-rotor hover speed: CW props run 60–90 RPM faster (yaw trim)")
    ax.legend(loc="upper left")
    ax.text(1.5, -68, "front/back and left/right pairs balance within 0.1% → center of mass is centered",
            ha="center", fontsize=7.5, color=INK2)

    # d. thrust per rotor vs RPM
    ax = fig.add_subplot(gs[2, 0])
    lo = min(np.percentile(f["rpm"][f["air"]].mean(1), 0.5) for f in F)
    hi = max(np.percentile(f["rpm"][f["air"]].mean(1), 99.5) for f in F)
    r_all = np.linspace(rmin, rmax, 300)
    r_meas = np.linspace(lo, hi, 50)
    ax.axvspan(lo, hi, color=SERIES[0], alpha=0.10, lw=0)
    ax.plot(r_all, thrust(r_all), color=SERIES[0], lw=2, ls=(0, (4, 3)))
    ax.plot(r_meas, thrust(r_meas), color=SERIES[0], lw=2.5, solid_capstyle="round")
    pts = [(rpm_h, f"hover\n{rpm_h:,.0f} RPM, {thrust(rpm_h):.2f} N"),
           (rpm_cap, f"PX4 cap (MPC_THR_MAX 0.6)\n{rpm_cap:,.0f} RPM, {thrust(rpm_cap):.2f} N → T/W {4 * thrust(rpm_cap) / (M * G):.1f}"),
           (rmax, f"ESC limit\n{rmax:,.0f} RPM, {thrust(rmax):.2f} N → T/W {4 * thrust(rmax) / (M * G):.1f}")]
    for r, txt in pts:
        ax.plot([r], [thrust(r)], "o", ms=7, color=SERIES[0], mec=SURFACE, mew=1.5, zorder=4)
    ax.annotate(pts[0][1], (rpm_h, thrust(rpm_h)), (rpm_h - 2900, thrust(rpm_h) + 1.6), fontsize=8,
                color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate(pts[1][1], (rpm_cap, thrust(rpm_cap)), (rpm_cap - 6400, thrust(rpm_cap) + 1.4), fontsize=8,
                color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate(pts[2][1], (rmax, thrust(rmax)), (rmax - 5600, thrust(rmax) + 0.15), fontsize=8,
                color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.text(lo + 60, 0.35, "measured\nrange", fontsize=8, color=INK2)
    ax.text(10300, 2.3, "dashed: F = k_f ω²\nextrapolated (not yet verified)", fontsize=7.5, color=MUTED)
    ax.set_xlim(rmin, rmax + 300)
    ax.set_ylim(0, 8.3)
    ax.set_xlabel("Rotor speed (RPM)")
    ax.set_ylabel("Thrust per rotor (N)")
    ax.set_title("d  Thrust curve per rotor from the measured k_f")

    # e. motor lag
    ax = fig.add_subplot(gs[2, 1])
    f = F[1]
    lag_ms = f["lags"] * 1000
    peaks = []
    for r in range(4):
        ax.plot(lag_ms, f["xcorr"][r], color=SERIES[r], lw=2, label=ROTOR_NAMES[r])
        j = int(np.argmax(f["xcorr"][r]))
        peaks.append(lag_ms[j])
        ax.plot([lag_ms[j]], [f["xcorr"][r][j]], "o", ms=7, color=SERIES[r], mec=SURFACE, mew=1.5, zorder=4)
    ax.axvline(35, color=INK, lw=1, ls=(0, (3, 3)))
    ax.text(37, 0.73, "τ_m = 35 ms\n(used in spawn_starling.py)", fontsize=8, color=INK)
    order = np.argsort([f["xcorr"][r][-1] for r in range(4)])
    for rank, r in enumerate(order):
        ax.text(lag_ms[-1] + 2, f["xcorr"][r][-1], f"rotor {r}: peak {peaks[r]:.0f} ms",
                fontsize=7.5, color=INK2, va="center")
    ax.set_xlim(0, 185)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("Lag of measured RPM behind commanded RPM (ms)")
    ax.set_ylabel("Correlation")
    ax.set_title("e  Motor lag: commanded vs measured RPM (flight 2)")
    ax.legend(loc="lower right", fontsize=7.5)

    fig.text(0.07, 0.022,
             f"Identified: k_f = {kf:.3e} N/(rad/s)², hover {rpm_h:,.0f} RPM, motor lag 30–40 ms (τ_m ≈ 35 ms), "
             f"ESC range {rmin:,.0f}–{rmax:,.0f} RPM.\n"
             f"Not identifiable from these flights: yaw torque k_m (no yaw maneuvers) and thrust above ~5,900 RPM.",
             ha="left", va="bottom", color=INK2, fontsize=8.5)
    fig.savefig(a.out, dpi=160)
    print("saved", a.out, f"k_f {kf:.4e}, hover RPM {rpm_h:.0f}, cap RPM {rpm_cap:.0f}")


if __name__ == "__main__":
    main()
