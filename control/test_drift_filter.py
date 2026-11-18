"""
control/drift_filter.py offline: the 3-gate course's min-snap path at 8/10 as the truth, the emulated base
VIO drifting off it, synthetic gate sightings from the C29 front camera (90.7 x 74.4 deg, <= 9 m, PnP noise
growing with range), the drift filter correcting the base VIO.

    python control/test_drift_filter.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for sub in ("control", "Trajectory", "vision"):
    p = os.path.join(REPO, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from drift_filter import DriftEmulator, DriftFilter, R_CV_FROM_USD, _inv   # noqa: E402
from vio import camera_mount_usd                                           # noqa: E402
from gate_path import gates_from_manifest, plan_gate_trajectory            # noqa: E402
from gate_pose import GateDetection                                        # noqa: E402

TRACK = os.path.join(REPO, "rl", "tracks", "three_gate_isaac.json")
MOUNT = camera_mount_usd(0.0, (0.0711, -0.0003, -0.0070))
HFOV, VFOV = math.radians(90.7 / 2), math.radians(74.4 / 2)


def truth(laps_dt=0.01):
    gates = gates_from_manifest(TRACK)
    g0 = gates[0]
    start = g0.center - 4.0 * g0.normal
    traj, _ = plan_gate_trajectory(gates, start, v_max=8.0, a_max=10.0, verbose=False, optimize_times=False)
    t, pos, vel, *_ = traj.sample(laps_dt)
    return gates, t, pos, vel


def R_level(vel, yaw_prev):
    yaw = math.atan2(vel[1], vel[0]) if np.linalg.norm(vel[:2]) > 0.3 else yaw_prev
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), yaw


def sightings(gates, T_wb, rng, max_range=9.0):
    """Synthetic PnP detections of the gates the camera sees from the true pose."""
    T_bc = MOUNT.copy()
    T_bc[:3, :3] = T_bc[:3, :3] @ R_CV_FROM_USD.T
    T_cw = _inv(T_wb @ T_bc)
    out = []
    for g in gates:
        T_cg = T_cw @ g.T_world_gate
        c = T_cg[:3, 3]
        r = float(np.linalg.norm(c))
        if c[2] <= 0.5 or r > max_range or abs(math.atan2(c[0], c[2])) > HFOV or abs(math.atan2(c[1], c[2])) > VFOV:
            continue
        std = max(0.02, 0.004 * r ** 2)
        T = T_cg.copy()
        T[:3, 3] = c + rng.normal(0.0, std, 3)
        out.append(GateDetection(T_cam_gate=T, reprojection_error_px=1.0, range_m=r, corners_px=np.zeros((4, 2)),
                                 center_std_m=std, normal_std_deg=5.0))
    return out


def fly(seed, correct=True, cam_hz=30, latency=0.06, outlier_every=0):
    gates, t, pos, vel = truth()
    rng = np.random.default_rng(100 + seed)
    emu = DriftEmulator(seed=seed)
    flt = DriftFilter(gates, MOUNT)
    yaw, err_base, err_corr, n_cam, pending = 0.0, [], [], 0, []
    next_cam = 0.0
    for k in range(len(t)):
        R, yaw = R_level(vel[k], yaw)
        p_vio, v_vio = emu.step(t[k], pos[k], vel[k])
        flt.add_vio(t[k], p_vio, v_vio, R)
        if t[k] >= next_cam:                       # an image taken now, processed `latency` later
            next_cam += 1.0 / cam_hz
            T_wb = np.eye(4)
            T_wb[:3, :3], T_wb[:3, 3] = R, pos[k]
            dets = sightings(gates, T_wb, rng)
            n_cam += 1
            if outlier_every and dets and n_cam % outlier_every == 0:   # a wrong solve: 1.5 m off
                dets[0].T_cam_gate[:3, 3] += np.array([1.5, 0.0, 0.0])
            pending.append((t[k] + latency, t[k], dets))
        while pending and pending[0][0] <= t[k]:
            _, t_img, dets = pending.pop(0)
            if correct and dets:
                flt.add_detections(t_img, dets)
        p_c, _ = flt.corrected(p_vio, v_vio)
        err_base.append(np.linalg.norm(p_vio - pos[k]))
        err_corr.append(np.linalg.norm(p_c - pos[k]))
    return np.array(err_base), np.array(err_corr), flt, t


def test_drift_is_removed():
    for seed in range(4):
        eb, ec, flt, t = fly(seed)
        settled = t > 1.0
        print(f"  seed {seed}: lap {t[-1]:.1f} s; base VIO error mean {eb.mean():.2f} m, max {eb.max():.2f}; "
              f"corrected mean {ec[settled].mean():.2f} m, max {ec[settled].max():.2f} m; fixes "
              f"{flt.stats['fixes']}, rejected {flt.stats['rejected']}")
        assert ec[settled].mean() < 0.5 * eb[settled].mean() and ec[settled].max() < 0.35


def test_bad_solves_are_rejected():
    eb, ec, flt, t = fly(1, outlier_every=7)
    settled = t > 1.0
    st = flt.stats
    print(f"  every 7th frame's nearest gate 1.5 m off: {st['unmatched']} matched no gate, {st['rejected']} failed "
          f"the chi-square test, {st['fixes']} accepted; corrected max {ec[settled].max():.2f} m")
    assert st["rejected"] + st["unmatched"] >= 5 and ec[settled].max() < 0.4


def test_without_fixes_it_is_the_base_vio():
    eb, ec, flt, t = fly(2, correct=False)
    print(f"  no fixes: corrected = base, error at the end {ec[-1]:.2f} m")
    assert np.allclose(eb, ec)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in tests:
        print(f.__name__)
        f()
    print(f"\nAll tests passed. ({len(tests)})")
