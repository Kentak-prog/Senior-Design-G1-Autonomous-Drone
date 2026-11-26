"""
In-flight velocity-jump failsafe, used by takeoff_hover.py and fly_min_snap_offboard.py while they fly in
OFFBOARD: if PX4's velocity changes faster than `max_accel` (m/s^2) for `sustain_s`, the flight is
cancelled and PX4 is handed to LAND (not Hold).

How: every new LOCAL_POSITION_NED velocity is compared with the velocity `window_s` earlier; the change
divided by the time between them is the acceleration. It has to stay above the limit for `sustain_s`, so
a single jump (an estimator reset steps the velocity once) does not cancel a flight, but a real jerk,
oscillation, a hit or a runaway does.

Limit: 3 m/s^2 above what the plan itself asks for (takeoff_hover: the climb's peak acceleration, ~0.15
m/s^2 for 1 m at 0.3 m/s; fly_min_snap_offboard: --a-max). Tuned by replaying PX4 logs
(vehicle_local_position vx/vy/vz, 50 Hz, 0.2 s window, 0.3 s sustain; test_accel_check.py):
  2026-11-03 hovers (23_20_53, 23_34_32.ulg)     quiet: at most 0.3 and 2.3 m/s^2 held for 0.3 s
  22_56_38.ulg (estimate "fell" 2 m)              quiet: at most 2.1 m/s^2
  2026-11-24 autonomous crash (21_49_16.ulg)     fires 10.4 s after arming (impact 11.3 s)
  2026-11-24 pilot crash (21_49_43.ulg)          fires 11.8 s (it flipped at 11.9 s)

LIMITS, honestly: it watches PX4's ESTIMATED velocity. On 2026-11-24 that estimate was wrong (it said the
drone was sinking slowly while it climbed), so this check saw nothing until the last second. The
barometer cross-check (baro_check.py, fires 7.1 s on that log) is the one for a wrong estimate; this one
is for real violent motion. And PX4's Land mode descends on the same estimate: if the drone keeps
climbing or flips after the hand-off, the pilot takes over in Stabilized or KILLS.
"""

from collections import deque

import numpy as np

DEFAULT_MARGIN = 3.0       # m/s^2 above the planned peak acceleration
PILOT_ALARM = ("PILOT: velocity changing too fast: flight CANCELLED, PX4 asked to LAND. If it does not "
               "settle and descend, take over in STABILIZED, or KILL.")


def quintic_peak_accel(height_m, duration_s):
    """Peak acceleration of takeoff_hover.quintic (10/sqrt(3) * h / T^2)."""
    return 5.7735 * height_m / duration_s ** 2


class VelocityJumpCheck:
    def __init__(self, max_accel=DEFAULT_MARGIN, window_s=0.2, sustain_s=0.3):
        self.max_accel, self.window, self.sustain = float(max_accel), window_s, sustain_s
        self.reset()

    def reset(self):
        self.samples = deque()          # (t, velocity ndarray)
        self.over_since = None
        self.last_sample_t = None
        self.accel = 0.0
        self.peak = 0.0

    def update(self, t, vel_ned, sample_t=None):
        """Feed one tick: now, PX4's velocity (NED, m/s) and when that velocity arrived (so a repeated old
        sample isn't counted twice). Returns None, or a reason string."""
        if vel_ned is None:
            return None
        st = t if sample_t is None else sample_t
        if self.last_sample_t is not None and st <= self.last_sample_t:
            return self._judge(t)                       # nothing new: keep the running verdict
        self.last_sample_t = st
        v = np.asarray(vel_ned, dtype=float)
        self.samples.append((st, v))
        while len(self.samples) > 1 and st - self.samples[1][0] >= self.window:
            self.samples.popleft()
        t0, v0 = self.samples[0]
        if st - t0 < 0.5 * self.window:
            return None                                  # not enough history yet
        self.accel = float(np.linalg.norm(v - v0)) / (st - t0)
        if self.accel <= self.max_accel:
            self.over_since = None
        elif self.over_since is None:
            self.over_since = st
        return self._judge(t)

    @property
    def violent(self):
        """True while the velocity is changing faster than the limit (not yet for sustain_s). Any OTHER
        abort that happens meanwhile (e.g. the tracking error, which a hard push can trip first) should
        also go to LAND."""
        return self.over_since is not None

    def _judge(self, t):
        if self.over_since is None:
            return None
        held = (self.last_sample_t or t) - self.over_since
        self.peak = max(self.peak, self.accel)
        if held >= self.sustain:
            return (f"velocity changing at {self.accel:.1f} m/s^2 (limit {self.max_accel:.1f}) for "
                    f"{held:.2f} s: too fast, landing")
        return None
