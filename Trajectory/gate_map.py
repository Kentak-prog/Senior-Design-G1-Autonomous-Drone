"""
Gate map: remembers every gate the camera has seen and refines each estimate
every time the gate is seen again.

Why: a single PnP solve is depth-dominated and gets worse with range
(vision/HANDOFF.md: ~0.06 m at 6 m, 0.3-0.4 m at 12-14 m, and the gate normal
is unreliable past ~8 m). Gates are static, so every new sighting is another
measurement of the same thing. Each gate keeps:

  - center: 3D position + 3x3 covariance, updated with a constant-position
    Kalman filter (measurement noise grows with range)
  - normal: confidence-weighted average of the measured flight directions
    (sign-aligned, since a square gate looks the same from behind)
  - normal_reliable: True once ANY sighting had a reliable normal
    (GateDetection.normal_is_reliable); until then the planner only gets the
    center, per the HANDOFF rule

A gate that drops out of view keeps its last estimate. When it comes back,
the new detection is matched to it by distance (nearest mapped gate within
the gating distance, which grows with range) and fused in.

Track management: a detection that matches nothing starts a TENTATIVE gate.
It becomes CONFIRMED after `min_hits` sightings; a tentative gate not seen
again within `tentative_timeout` seconds is dropped. Only confirmed gates go
to the planner, so one bad detection (wrong pose at that instant, a
mismatched corner) cannot add a phantom gate to the course.

Course order: with a prior course map (known layout), gates keep the prior's
order and the camera only corrects them. Without one, gates are ordered by
distance from where the drone started when each gate was FIRST seen (matches
how the artificial and spawn_test_gates.py courses are laid out).

NOTE: this overlaps with the "temporal fusion across frames" item in
vision/HANDOFF.md. It lives here for now; it could move to vision/ later.
"""

from dataclasses import dataclass, field

import numpy as np

from gate_path import Gate, DEFAULT_GATE_SIDE


def gate_transform(center, normal) -> np.ndarray:
    """T_world_gate from a center + flight direction, in the vision gate-model
    convention (+X right, +Y down, +Z = flight direction), assuming the gate is
    upright (its +Y is as close to world -Z as possible)."""
    z = np.asarray(normal, dtype=float)
    z = z / np.linalg.norm(z)
    down = np.array([0.0, 0.0, -1.0])
    x = np.cross(down, z)
    if np.linalg.norm(x) < 1e-6:                       # flying straight up/down through it
        x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, center
    return T


@dataclass
class MappedGate:
    gate_id: int
    center: np.ndarray
    P: np.ndarray                       # 3x3 center covariance
    normal_sum: np.ndarray              # confidence-weighted sum of unit normals
    side: float = DEFAULT_GATE_SIDE
    n_reliable: int = 0                 # sightings with a reliable normal
    n_obs: int = 0                      # camera sightings (0 = prior only)
    last_seen: float = -np.inf
    from_prior: bool = False
    name: str = field(default="")
    order_key: float = 0.0              # fixed at creation so the course order never flips
    dropped: bool = False               # tentative gate that was never confirmed

    @property
    def normal(self) -> np.ndarray:
        return self.normal_sum / np.linalg.norm(self.normal_sum)

    @property
    def normal_reliable(self) -> bool:
        return self.n_reliable > 0 or self.from_prior

    def confirmed(self, min_hits) -> bool:
        return not self.dropped and (self.from_prior or self.n_obs >= min_hits)

    @property
    def center_std(self) -> float:
        return float(np.sqrt(np.trace(self.P) / 3.0))

    def as_gate(self) -> Gate:
        return Gate(self.name, gate_transform(self.center, self.normal), side=self.side,
                    normal_reliable=self.normal_reliable)


class GateMap:
    """
    gating_dist:     min distance (m) between a detection and a mapped gate
                     for them to be treated as the same gate; the actual gate
                     is max(gating_dist, gating_frac * range)
    min_hits:        sightings before a gate is confirmed and used for planning
    tentative_timeout: seconds a tentative gate survives without a new sighting
    range_noise_frac: measurement 1-sigma added per meter of range, on top of
                     the PnP bootstrap std (covers the depth bias and drone
                     pose error that the bootstrap does not see)
    """

    def __init__(self, origin, gating_dist: float = 1.0, gating_frac: float = 0.1,
                 range_noise_frac: float = 0.02, noise_floor: float = 0.03,
                 min_hits: int = 3, tentative_timeout: float = 0.5,
                 prior=None, prior_sigma: float = 0.5):
        self.origin = np.asarray(origin, dtype=float)
        self.gating_dist = gating_dist
        self.gating_frac = gating_frac
        self.min_hits = min_hits
        self.tentative_timeout = tentative_timeout
        self.range_noise_frac = range_noise_frac
        self.noise_floor = noise_floor
        self.gates = []
        self.version = 0               # bumps whenever any estimate changes
        for g in (prior or []):
            self._add(g.center, g.normal, side=g.side, P=np.eye(3) * prior_sigma ** 2,
                      weight=1.0, from_prior=True, name=g.name)

    def _add(self, center, normal, side, P, weight, from_prior=False, name=None, reliable=False, t=-np.inf):
        gid = len(self.gates)
        mg = MappedGate(gid, np.array(center, dtype=float), np.array(P, dtype=float),
                        weight * np.asarray(normal, dtype=float), side=side,
                        n_reliable=int(reliable), n_obs=0 if from_prior else 1, last_seen=t,
                        from_prior=from_prior, name=name or f"map_{gid}",
                        order_key=float(gid) if from_prior else
                        float(np.linalg.norm(np.asarray(center) - self.origin)))
        self.gates.append(mg)
        return mg

    def update(self, detections, t: float):
        """
        detections: list of dicts with 'center' (3,), 'normal' (3,), 'normal_reliable',
        'center_std' (m, may be nan), 'normal_std' (deg, may be nan), 'range' (m), 'side'.
        Returns the list of MappedGate each detection was fused into (in order).
        """
        used, out = set(), []
        for det in sorted(detections, key=lambda d: d['range']):     # nearest = most trustworthy first
            c = np.asarray(det['center'], dtype=float)
            std = det.get('center_std', np.nan)
            sigma = np.hypot(std if np.isfinite(std) else 0.0,
                             self.noise_floor + self.range_noise_frac * det['range'])
            R = np.eye(3) * sigma ** 2

            n = np.asarray(det['normal'], dtype=float)
            n_std = det.get('normal_std', np.nan)
            w = 1.0 / max(n_std if np.isfinite(n_std) else 30.0, 1.0) ** 2
            if not det['normal_reliable']:
                w *= 0.1                                    # still a hint, but a weak one

            # Associate with the nearest mapped gate not already used this frame
            gate_r = max(self.gating_dist, self.gating_frac * det['range'])
            cands = [(np.linalg.norm(mg.center - c), mg) for mg in self.gates
                     if mg.gate_id not in used and not mg.dropped]
            cands = [cm for cm in cands if cm[0] < gate_r]
            if not cands:
                mg = self._add(c, n, det.get('side', DEFAULT_GATE_SIDE), R, w,
                               reliable=det['normal_reliable'], t=t)
            else:
                mg = min(cands, key=lambda cm: cm[0])[1]
                # Kalman update, constant-position model (gates don't move)
                K = mg.P @ np.linalg.inv(mg.P + R)
                mg.center = mg.center + K @ (c - mg.center)
                mg.P = (np.eye(3) - K) @ mg.P
                # A square gate looks the same from both sides: align the sign first
                if n @ mg.normal_sum < 0:
                    n = -n
                mg.normal_sum = mg.normal_sum + w * n
                mg.n_reliable += int(det['normal_reliable'])
                mg.n_obs += 1
                mg.last_seen = t
            used.add(mg.gate_id)
            out.append(mg)

        # Drop tentative gates that were never seen again
        for mg in self.gates:
            if not mg.dropped and not mg.confirmed(self.min_hits) and \
                    t - mg.last_seen > self.tentative_timeout:
                mg.dropped = True
        if out:
            self.version += 1
        return out

    def confirmed(self):
        return [mg for mg in self.gates if mg.confirmed(self.min_hits)]

    def ordered(self):
        """CONFIRMED gates in course order (prior order, else distance from the start)."""
        # order_key is frozen when a gate is first seen: two gates at similar range
        # must not swap places as their estimates wobble.
        gates = self.confirmed()
        prior = [mg for mg in gates if mg.from_prior]
        extra = sorted((mg for mg in gates if not mg.from_prior), key=lambda mg: mg.order_key)
        return prior + extra
