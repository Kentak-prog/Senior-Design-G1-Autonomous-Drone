"""
GateMap: a world-frame map of the course's gates that starts from a ROUGH PRIOR
and is corrected by camera detections. Pure numpy; no ROS, no vision/ imports.

CONVENTIONS (rl/ and vision/ agree): world ENU, gate frame +X right, +Y down,
    +Z flight direction (course.Gate.T_world_gate). T_a_b maps frame b -> a.
    A detection is a gate pose in the WORLD frame, as vision's gate_pose_world
    returns it.

WHAT IS ASSUMED
    * The gate count and order are known (they come from the prior course).
    * Gates are upright: only the centre (x, y, z) and the yaw about world Z
      are estimated; roll/pitch of a detection are ignored.
    * The gates do not move.

ASSOCIATION (update)
    For one batch of detections, every (detection, candidate gate) pair whose
    centre distance to the gate's CURRENT estimate is <= assoc_radius is a
    candidate. Candidates are gates with index >= the progress set by
    set_progress(k) (gates already passed are ignored) and < progress + lookahead.
    The cost is  distance + index_penalty * (gate_idx - progress)  so the
    current target and the next ones win ties. Pairs are taken greedily in
    cost order, ONE detection per gate per batch (the first = cheapest); the
    rest are dropped (counted in n_unassoc).

FUSION (per gate, independent)
    Position: scalar-variance recursive weighted mean (a Kalman filter for a
    static point with isotropic variance). State: estimate x, variance P
    (m^2, initial pos_sigma0^2 = how much the prior is trusted). For a
    measurement z with variance r:  K = P / (P + r);  x += K (z - x);
    P = max((1 - K) P, pos_floor^2). r = (meas_sigma + range_gain * range)^2
    when the detection carries its camera range (error grows with range, as the
    live gate-error runs show: ~1% bias plus noise), else (meas_sigma +
    range_gain * 8 m)^2. The floor keeps the filter responsive (the live errors
    are correlated frame to frame, so P must not shrink to zero).
    Yaw: only from detections with normal_reliable=True (near head-on the
    normal is noise). The detection's yaw is atan2 of its gate +Z projected on
    the ground. 180-DEG AMBIGUITY: a normal pointing against the PRIOR's
    flight direction (negative dot product with the prior gate's horizontal
    normal) is flipped by 180 deg, i.e. the gate +Z must point along the
    course direction. A yaw further than yaw_outlier_deg from the prior after
    that is rejected (counted in n_yaw_rejected). Yaw is then fused like the
    position, on the wrapped angle difference, with its own variance
    (yaw_sigma0 initial, floor yaw_floor).

INTERFACE
    update(detections) -> list of (detection index, gate index) associations
    set_progress(k), course() -> rl.course.Course of the current estimates
    (cached; `version` increases on every change, so a consumer can push only
    when it changed), stats(), errors(truth) (logging only), perturbed_prior().
"""

import math
from typing import NamedTuple, Optional

import numpy as np

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # rl/
from course import Course, Gate  # noqa: E402


class Detection(NamedTuple):
    """One gate observation in the world frame."""
    T_world_gate: np.ndarray        # 4x4
    normal_reliable: bool
    t: float                        # image time [s] (sim clock)
    range_m: Optional[float] = None  # camera -> gate centre, if known


def as_detection(d):
    """Accept a Detection or a plain (T, reliable, t[, range]) tuple."""
    return d if isinstance(d, Detection) else Detection(*d)


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_of_T_world_gate(T):
    """Gate yaw [rad] (course.Gate.yaw_deg convention: flight direction = world
    (cos yaw, sin yaw)) from a gate pose; None if the normal is near vertical."""
    n = np.asarray(T)[:3, 2]
    h = float(np.hypot(n[0], n[1]))
    if h < 0.5 * float(np.linalg.norm(n)):
        return None
    return math.atan2(n[1], n[0])


def perturbed_prior(course, pos_std=0.5, yaw_std_deg=10.0, seed=0):
    """A rough prior: every gate shifted by N(0, pos_std) PER AXIS (x, y, z)
    and yawed by N(0, yaw_std_deg), seeded. Same names/order/sizes."""
    rng = np.random.default_rng(seed)
    gates = []
    for g in course.gates:
        dx, dy, dz = rng.normal(0.0, pos_std, 3)
        dyaw = rng.normal(0.0, yaw_std_deg)
        gates.append(Gate(name=g.name, x=g.x + dx, y=g.y + dy, z=g.z + dz,
                          yaw_deg=g.yaw_deg + dyaw, side=g.side, bar=g.bar))
    return Course(gates=gates, name=course.name + "_prior")


class _GateState:
    def __init__(self, g, pos_sigma0, yaw_sigma0):
        self.pos = g.center.copy()
        self.pvar = pos_sigma0 ** 2
        self.yaw = math.radians(g.yaw_deg)
        self.yvar = yaw_sigma0 ** 2
        self.n_det = 0
        self.n_yaw = 0
        self.n_yaw_rejected = 0
        self.last_seen = float("nan")


class GateMap:
    def __init__(self, prior_course, assoc_radius=2.5, lookahead=3, index_penalty=0.25,
                 pos_sigma0=0.5, yaw_sigma0_deg=15.0, meas_sigma=0.10, range_gain=0.02,
                 pos_floor=0.08, yaw_meas_sigma_deg=6.0, yaw_range_gain_deg=0.5,
                 yaw_floor_deg=2.0, yaw_outlier_deg=60.0, truth_course=None):
        self.prior = prior_course
        self.assoc_radius, self.lookahead, self.index_penalty = assoc_radius, lookahead, index_penalty
        self.meas_sigma, self.range_gain, self.pos_floor = meas_sigma, range_gain, pos_floor
        self.yaw_meas_sigma = math.radians(yaw_meas_sigma_deg)
        self.yaw_range_gain = math.radians(yaw_range_gain_deg)
        self.yaw_floor = math.radians(yaw_floor_deg)
        self.yaw_outlier = math.radians(yaw_outlier_deg)
        self.truth = truth_course
        self.states = [_GateState(g, pos_sigma0, math.radians(yaw_sigma0_deg))
                       for g in prior_course.gates]
        self.prior_yaw = [math.radians(g.yaw_deg) for g in prior_course.gates]
        self.progress = 0
        self.version = 0
        self.n_unassoc = 0
        self.n_batches = 0
        self._course = None
        self._course_version = -1

    def __len__(self):
        return len(self.states)

    def set_progress(self, gate_idx):
        """Gates with index < gate_idx are passed and no longer updated."""
        self.progress = int(gate_idx)

    # ------------------------------------------------------------------
    def update(self, detections):
        """Fuse a batch of detections (Detection or (T, reliable, t[, range])).
        Returns [(detection index, gate index), ...] for the associated ones."""
        dets = [as_detection(d) for d in detections]
        if not dets:
            return []
        self.n_batches += 1
        lo, hi = self.progress, min(len(self.states), self.progress + self.lookahead)
        pairs = []
        for di, d in enumerate(dets):
            c = np.asarray(d.T_world_gate)[:3, 3]
            for gi in range(lo, hi):
                dist = float(np.linalg.norm(c - self.states[gi].pos))
                if dist <= self.assoc_radius:
                    pairs.append((dist + self.index_penalty * (gi - lo), di, gi))
        pairs.sort()
        used_d, used_g, out = set(), set(), []
        for _cost, di, gi in pairs:
            if di in used_d or gi in used_g:
                continue
            used_d.add(di)
            used_g.add(gi)
            self._fuse(gi, dets[di])
            out.append((di, gi))
        self.n_unassoc += len(dets) - len(out)
        if out:
            self.version += 1
        return sorted(out)

    def _fuse(self, gi, d):
        st = self.states[gi]
        T = np.asarray(d.T_world_gate, dtype=np.float64)
        rng = d.range_m if d.range_m is not None and np.isfinite(d.range_m) else 8.0
        r = (self.meas_sigma + self.range_gain * rng) ** 2
        K = st.pvar / (st.pvar + r)
        st.pos = st.pos + K * (T[:3, 3] - st.pos)
        st.pvar = max((1.0 - K) * st.pvar, self.pos_floor ** 2)
        st.n_det += 1
        if np.isfinite(d.t):
            st.last_seen = max(st.last_seen, float(d.t)) if np.isfinite(st.last_seen) else float(d.t)
        if not d.normal_reliable:
            return
        yaw = yaw_of_T_world_gate(T)
        if yaw is None:
            return
        # 180-deg ambiguity: gate +Z must point along the prior's flight direction
        py = self.prior_yaw[gi]
        if math.cos(yaw - py) < 0.0:
            yaw = wrap_pi(yaw + math.pi)
        if abs(wrap_pi(yaw - py)) > self.yaw_outlier:
            st.n_yaw_rejected += 1
            return
        ry = (self.yaw_meas_sigma + self.yaw_range_gain * rng) ** 2
        Ky = st.yvar / (st.yvar + ry)
        st.yaw = wrap_pi(st.yaw + Ky * wrap_pi(yaw - st.yaw))
        st.yvar = max((1.0 - Ky) * st.yvar, self.yaw_floor ** 2)
        st.n_yaw += 1

    # ------------------------------------------------------------------
    def course(self):
        """Course of the current estimates (cached per version)."""
        if self._course is None or self._course_version != self.version:
            gates = []
            for g, st in zip(self.prior.gates, self.states):
                gates.append(Gate(name=g.name, x=float(st.pos[0]), y=float(st.pos[1]),
                                  z=float(st.pos[2]), yaw_deg=math.degrees(st.yaw),
                                  side=g.side, bar=g.bar))
            self._course = Course(gates=gates, name=self.prior.name)
            self._course_version = self.version
        return self._course

    def stats(self):
        """Per gate: detections, yaw updates/rejections, last seen time,
        position std [m] and yaw std [deg] of the filter."""
        return [dict(gate=i, n_det=s.n_det, n_yaw=s.n_yaw, n_yaw_rejected=s.n_yaw_rejected,
                     last_seen=s.last_seen, pos_std=math.sqrt(s.pvar),
                     yaw_std_deg=math.degrees(math.sqrt(s.yvar)))
                for i, s in enumerate(self.states)]

    def _err_one(self, g, st):
        e = st.pos - g.center
        local = g.T_world_gate()[:3, :3].T @ e
        return (float(np.linalg.norm(e)), float(np.hypot(local[0], local[1])),
                abs(math.degrees(wrap_pi(st.yaw - math.radians(g.yaw_deg)))))

    def errors(self, truth=None):
        """Current error vs a truth course (LOGGING ONLY, never fed back).
        Returns dict of arrays (per gate): pos (3-D distance), lat (distance
        in the TRUE gate plane, the part that decides passing), yaw_deg (abs)."""
        truth = truth if truth is not None else self.truth
        if truth is None:
            raise ValueError("no truth course")
        e = np.array([self._err_one(g, st) for g, st in zip(truth.gates, self.states)])
        return dict(pos=e[:, 0], lat=e[:, 1], yaw_deg=e[:, 2])

    def error_of(self, gate_idx, truth=None):
        """(pos_err, lat_err, yaw_err_deg) of one gate (NaNs without truth)."""
        truth = truth if truth is not None else self.truth
        if truth is None or not (0 <= gate_idx < len(self)):
            return (float("nan"),) * 3
        return self._err_one(truth.gates[gate_idx], self.states[gate_idx])
