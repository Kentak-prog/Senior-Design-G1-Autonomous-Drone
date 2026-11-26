"""
In-flight cross-check of PX4's height estimate against the barometer, used by takeoff_hover.py and
fly_min_snap_offboard.py while they fly in OFFBOARD.

Why (px4/INCIDENT_2026-11-24_TAKEOFF_FLYAWAY.md): on 2026-11-24 PX4's height came from a VIO that had
stopped tracking; the estimate stayed near 0, then said "sinking", while the drone climbed ~5.5 m. The
script's tracking-error abort compares setpoints with that same estimate, so it can't see such a fault.
The barometer is an independent height reference: raw pressure (MAVLink SCALED_PRESSURE), turned into a
height change here, outside PX4's estimator.

How: from the moment OFFBOARD starts, compare the CHANGE in PX4's height (-z, NED) with the change in
barometric height, each averaged over `window_s` (the barometer is noisy, ~±0.3 m under prop wash). A bias
term follows their difference with time constant `offset_tau_s`, so STEP offsets fade within a few
seconds (the barometer jumps by up to ~1 m at takeoff and in hover from prop wash and ground effect:
2026-11-03 flight 2 read ~1 m low the whole hover). What remains is a SUSTAINED divergence: if the
estimate and the barometer keep moving apart, the gap builds to (rate x tau). Above `max_gap_m` for
`sustain_s`, it reports. It therefore catches divergence faster than ~0.25 m/s (0.7 m / 3 s); slow drift
is left to preflight.py's steadiness check.

Tuned by replaying real PX4 logs (vehicle_air_data.baro_alt_meter vs vehicle_local_position.z, 50 Hz):
  2026-11-24 autonomous crash (21_49_16.ulg)   fires 7.1 s after arming (drone ~1.2 m up by the barometer;
                                               it crashed at 11.3 s from ~5.5 m)
  2026-11-24 pilot crash (21_49_43.ulg)        fires 13.7 s (that flight had already flipped at 11.9 s)
  22_56_38.ulg (PX4's estimate "fell" 2 m)      fires 7.9 s
  2026-11-03 hovers (23_20_53, 23_34_32.ulg;   quiet (largest gaps 0.36 m and 1.21 m before the bias
    107 s and 215 s, Position mode)              absorbs them)
Faster settings (0.5 m) false-alarm on 23_34_32; slower ones (1.0 m) fire ~0.6 s later on the crash.
"""

from collections import deque


def pressure_altitude_m(press_hpa, ref_hpa=1013.25):
    """Barometric altitude (m) from absolute pressure (hPa), standard atmosphere. Only changes matter."""
    return 44330.77 * (1.0 - (press_hpa / ref_hpa) ** 0.190263)


class BaroCrossCheck:
    def __init__(self, max_gap_m=0.7, sustain_s=1.0, window_s=1.0, offset_tau_s=3.0, stale_s=1.0):
        self.max_gap, self.sustain, self.window = max_gap_m, sustain_s, window_s
        self.tau, self.stale = offset_tau_s, stale_s
        self.reset()

    def reset(self):
        self.base = None                 # (est_up0, baro0) at the start
        self.samples = deque()           # (t, est_up, baro_up) within the window
        self.bias = 0.0
        self.over_since = None
        self.last_baro_t = None
        self.last_t = None
        self.t_start = None
        self.gap = 0.0

    def update(self, t, z_ned, baro_alt_m, baro_t=None):
        """Feed one tick: PX4's z (NED, m), the newest barometric altitude (m) and when it was received.
        Returns None, or a reason string once the estimate and the barometer have disagreed for long enough."""
        if self.t_start is None:
            self.t_start = t
        if baro_alt_m is None or (baro_t is not None and t - baro_t > self.stale):
            since = self.last_baro_t if self.last_baro_t is not None else self.t_start
            if t - since > self.stale:
                return (f"no barometer data for {t - since:.1f} s" if self.last_baro_t is not None
                        else f"no barometer data at all in {t - since:.1f} s (SCALED_PRESSURE not streaming?)")
            return None
        if baro_t is not None:
            self.last_baro_t = baro_t
        else:
            self.last_baro_t = t
        est_up = -float(z_ned)
        if self.base is None:
            self.base = (est_up, float(baro_alt_m))
            self.last_t = t
        self.samples.append((t, est_up - self.base[0], float(baro_alt_m) - self.base[1]))
        while self.samples and t - self.samples[0][0] > self.window:
            self.samples.popleft()
        if t - self.samples[0][0] < 0.5 * self.window:
            return None                                    # not enough history yet
        n = len(self.samples)
        est = sum(s[1] for s in self.samples) / n
        baro = sum(s[2] for s in self.samples) / n
        dt = max(0.0, t - (self.last_t if self.last_t is not None else t))
        self.last_t = t
        raw = est - baro
        self.gap = raw - self.bias
        self.bias += (raw - self.bias) * min(1.0, dt / self.tau)         # step offsets fade with tau
        if abs(self.gap) <= self.max_gap:
            self.over_since = None
            return None
        if self.over_since is None:
            self.over_since = t
        if t - self.over_since >= self.sustain:
            word = "below" if self.gap < 0 else "above"
            return (f"PX4's height disagrees with the barometer by {abs(self.gap):.2f} m for {t - self.over_since:.1f} s "
                    f"(estimate {est:+.2f} m since start, barometer {baro:+.2f} m: PX4 thinks it is {word} where the "
                    f"barometer says)")
        return None


MSG_ID_SCALED_PRESSURE = 29
PILOT_ALARM = ("PILOT: PX4's height may be WRONG. Automatic modes (Hold, Position, Altitude) use it too: "
               "take over in STABILIZED, or KILL if it keeps climbing.")


def request_baro(link, hz=20.0):
    """Ask PX4 for SCALED_PRESSURE at hz (MAV_CMD_SET_MESSAGE_INTERVAL)."""
    link.command(511, MSG_ID_SCALED_PRESSURE, 1e6 / hz)


def baro_from_link(link):
    """(altitude m, receive time) from the newest SCALED_PRESSURE on a fly_min_snap_offboard.PX4Link."""
    item = link.last.get("SCALED_PRESSURE")
    if item is None:
        return None, None
    t, msg = item
    return pressure_altitude_m(msg.press_abs), t
