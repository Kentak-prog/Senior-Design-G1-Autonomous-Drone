"""
Race course: gate geometry, JSON I/O, gate-crossing detection, track generation.

CONVENTIONS
    World frame ENU (x east, y north, z up) = Pegasus "map" = test_gates.json.
    Gate frame (identical to vision/gate_pose.py and
    workspace/spawn_test_gates._T_world_gate): +X right, +Y DOWN, +Z flight
    direction, origin at the opening's center. At yaw_deg = 0 the flight
    direction is world +X, gate +X = world -Y, gate +Y = world -Z; positive yaw
    rotates the gate counter-clockwise about world +Z. The gate lies in its
    local z = 0 plane; "crossing" means the drone path pierces that plane.
    T_a_b maps frame b -> a, as in vision/.

WHY T_world_gate is duplicated instead of imported: workspace/spawn_test_gates.py
    needs Isaac's interpreter, and rl/ must run in plain python. The two are
    kept identical by rl/test_course.py, which compares them directly.

GATE CROSSING (check_gate)
    Work in the gate frame. A step p_prev -> p_now that goes from z < 0 to
    z >= 0 pierces the plane at a point (x, y). With half = side/2:
      pass       the point is inside the opening shrunk by the drone radius
                 (|x|, |y| <= half - radius) and the step is FORWARD.
      collision  the crossing (either direction) lies inside the outer frame
                 (|x|, |y| <= half + bar) but not inside the shrunk opening, OR
                 at p_now the drone center is within `radius` of any of the four
                 bars (bars modelled as boxes, distance to a box).
      neither    crossing wide of the frame, or no crossing.
    Backwards crossings are never a pass.
"""

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch

GATE_SIDE = 1.5     # opening, inner edge to inner edge [m]  (spawn_test_gates.GATE_SIDE)
BAR_THICKNESS = 0.10  # [m]                                  (spawn_test_gates.BAR_THICKNESS)

YAW_CONVENTION = (
    "yaw_deg=0 means gate +Z (flight direction) is world +X; positive "
    "yaw rotates the gate counter-clockwise about world +Z (right-hand "
    "rule); at yaw=0, gate +X = world -Y and gate +Y = world -Z."
)

TRACKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracks")


# ----------------------------------------------------------------------
# Gate / Course
# ----------------------------------------------------------------------
@dataclass
class Gate:
    name: str
    x: float
    y: float
    z: float
    yaw_deg: float
    side: float = GATE_SIDE
    bar: float = BAR_THICKNESS

    def T_world_gate(self) -> np.ndarray:
        """4x4 gate -> world; same construction as spawn_test_gates._T_world_gate."""
        R_yaw0 = np.array([[0.0, 0.0, 1.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0]])
        yaw = math.radians(self.yaw_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        T = np.eye(4)
        T[:3, :3] = Rz @ R_yaw0
        T[:3, 3] = [self.x, self.y, self.z]
        return T

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @property
    def normal(self) -> np.ndarray:
        """World-frame flight direction (gate +Z)."""
        return self.T_world_gate()[:3, 2]


@dataclass
class Course:
    gates: List[Gate] = field(default_factory=list)
    name: str = "course"

    def __len__(self):
        return len(self.gates)

    def start_pose(self, back: float = 4.0):
        """Default start: `back` m behind gate 0 along its flight direction, at
        gate height, facing the gate. Returns (pos (3,), yaw_rad)."""
        g = self.gates[0]
        pos = g.center - back * g.normal
        return pos, math.radians(g.yaw_deg)

    # --- JSON (schema of spawn_test_gates.main()'s manifest) ---------------
    def to_dict(self) -> dict:
        gates = []
        for g in self.gates:
            d = {"name": g.name, "x": g.x, "y": g.y, "z": g.z,
                 "yaw_deg": g.yaw_deg, "side": g.side}
            if g.bar != BAR_THICKNESS:
                d["bar"] = g.bar
            gates.append(d)
        return {"yaw_convention": YAW_CONVENTION, "course_name": self.name,
                "gates": gates}

    @staticmethod
    def from_dict(d: dict, name: str = "course") -> "Course":
        """Extra keys (camera_pose_usd, dropped, ...) are ignored."""
        gates = [Gate(name=g["name"], x=float(g["x"]), y=float(g["y"]),
                      z=float(g["z"]), yaw_deg=float(g["yaw_deg"]),
                      side=float(g.get("side", GATE_SIDE)),
                      bar=float(g.get("bar", BAR_THICKNESS)))
                 for g in d["gates"]]
        return Course(gates=gates, name=d.get("course_name", name))

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path: str) -> "Course":
        with open(path) as f:
            d = json.load(f)
        return Course.from_dict(d, name=os.path.splitext(os.path.basename(path))[0])


# ----------------------------------------------------------------------
# Batched gate tensors and crossing check
# ----------------------------------------------------------------------
@dataclass
class GateBatch:
    center: torch.Tensor   # (N,3) world
    R: torch.Tensor        # (N,3,3) gate -> world rotation
    side: torch.Tensor     # (N,)
    bar: torch.Tensor      # (N,)


def gate_batch(course: Course, idx: torch.Tensor, dtype=torch.float32) -> GateBatch:
    """Gather the gates `idx` (N,) (long, clamped into range) into batched tensors."""
    dev = idx.device
    Ts = np.stack([g.T_world_gate() for g in course.gates])
    T = torch.as_tensor(Ts, dtype=dtype, device=dev)
    side = torch.tensor([g.side for g in course.gates], dtype=dtype, device=dev)
    bar = torch.tensor([g.bar for g in course.gates], dtype=dtype, device=dev)
    i = idx.clamp(0, len(course) - 1)
    return GateBatch(center=T[i, :3, 3], R=T[i, :3, :3], side=side[i], bar=bar[i])


def _box_distance(p: torch.Tensor, center: torch.Tensor, half: torch.Tensor) -> torch.Tensor:
    """Distance from points p (N,3) to axis-aligned boxes (center, half-extents)."""
    d = (p - center).abs() - half
    return d.clamp_min(0.0).norm(dim=-1)


def check_gate(p_prev: torch.Tensor, p_now: torch.Tensor, gates: GateBatch,
               radius: float):
    """Batched gate test for the segment p_prev -> p_now (world, (N,3)).
    Returns (passed, collided) bool tensors (N,). See module docstring."""
    def to_gate(p):
        return torch.einsum("nji,nj->ni", gates.R, p - gates.center)   # R^T (p - c)

    a, b = to_gate(p_prev), to_gate(p_now)
    half = gates.side / 2.0
    outer = half + gates.bar

    za, zb = a[:, 2], b[:, 2]
    forward = (za < 0) & (zb >= 0)
    backward = (za >= 0) & (zb < 0)
    crossed = forward | backward
    t = za / (za - zb).where(crossed, torch.ones_like(za))   # safe where not crossed
    cx = a[:, 0] + t * (b[:, 0] - a[:, 0])
    cy = a[:, 1] + t * (b[:, 1] - a[:, 1])

    in_open = (cx.abs() <= half - radius) & (cy.abs() <= half - radius)
    in_outer = (cx.abs() <= outer) & (cy.abs() <= outer)
    passed = forward & in_open
    hit_cross = crossed & in_outer & ~in_open

    # proximity to the four bars at p_now. Boxes in the gate frame: thickness
    # `bar` along z (+-bar/2 about the plane).
    ones = torch.ones_like(half)
    z0 = torch.zeros_like(half)
    bz = gates.bar / 2.0
    mid = half + gates.bar / 2.0
    boxes = [  # (center xy, half extents xyz) for top/bottom/left/right
        ((z0, -mid), (outer, gates.bar / 2.0, bz)),
        ((z0, mid), (outer, gates.bar / 2.0, bz)),
        ((-mid, z0), (gates.bar / 2.0, half, bz)),
        ((mid, z0), (gates.bar / 2.0, half, bz)),
    ]
    near = torch.zeros_like(passed)
    for (cxy, hext) in boxes:
        c = torch.stack([cxy[0], cxy[1], z0], dim=-1)
        h = torch.stack(list(hext), dim=-1)
        near = near | (_box_distance(b, c, h) < radius)
    return passed, hit_cross | near


# ----------------------------------------------------------------------
# Course generation
# ----------------------------------------------------------------------
def random_course(rng: np.random.Generator, n_gates: int, name: str = "random",
                  spacing=(5.0, 12.0), max_turn_deg: float = 60.0,
                  height=(1.5, 4.0), min_gate_dist: float = 3.0,
                  yaw_jitter_deg: float = 0.0) -> Course:
    """Random course. Each leg is `spacing` metres long (HORIZONTAL distance),
    its heading differs from the previous leg by at most max_turn_deg, heights
    are uniform in `height` (so vertical steps are up to 2.5 m), and gate yaw is
    the circular mean of the incoming and outgoing leg headings (the last gate
    keeps its incoming heading), i.e. approximately the direction of travel.
    Candidate gates closer than min_gate_dist (3D) to an earlier gate are
    rejected, so gates never overlap.

    yaw_jitter_deg adds a uniform +-jitter to every gate's yaw, so the gate no
    longer faces the direction of travel and the drone has to turn INTO it
    (e.g. rl/tracks/slalom5.json arrives ~65 deg off a gate's normal). Without
    it a trained policy only ever sees gates within ~30 deg of its path. The
    jitter is drawn after the layout, so jitter = 0 reproduces the old courses
    for the same seed exactly."""
    for _attempt in range(200):
        heading = rng.uniform(-math.pi, math.pi)
        pts = [np.array([rng.uniform(5, 12), 0.0, rng.uniform(*height)])]
        pts[0][:2] = pts[0][0] * np.array([math.cos(heading), math.sin(heading)])
        headings = [heading]
        ok = True
        while len(pts) < n_gates and ok:
            for _try in range(50):
                h = headings[-1] + math.radians(rng.uniform(-max_turn_deg, max_turn_deg))
                d = rng.uniform(*spacing)
                p = pts[-1] + np.array([d * math.cos(h), d * math.sin(h), 0.0])
                p[2] = rng.uniform(*height)
                if all(np.linalg.norm(p - q) >= min_gate_dist for q in pts):
                    pts.append(p)
                    headings.append(h)
                    break
            else:
                ok = False
        if ok:
            break
    else:
        raise RuntimeError("random_course: could not place gates")

    jitter = (rng.uniform(-yaw_jitter_deg, yaw_jitter_deg, len(pts)) if yaw_jitter_deg > 0
              else np.zeros(len(pts)))
    gates = []
    for i, p in enumerate(pts):
        h_in = headings[i]
        h_out = headings[i + 1] if i + 1 < len(pts) else h_in
        # circular mean of the two headings (they differ by < pi)
        yaw = h_in + 0.5 * math.atan2(math.sin(h_out - h_in), math.cos(h_out - h_in))
        gates.append(Gate(f"gate_{i:02d}", float(p[0]), float(p[1]), float(p[2]),
                          math.degrees(yaw) + float(jitter[i])))
    return Course(gates, name)


def _straight3() -> Course:
    return Course([Gate(f"gate_{i}", 8.0 * (i + 1), 0.0, 2.5, 0.0) for i in range(3)],
                  "straight3")


def _oval6() -> Course:
    """6 gates on an ellipse (semi-axes 10 x 6 m, ~20x12 m), counter-clockwise,
    each facing the local tangent, heights alternating 2.0 / 3.5 m."""
    a, b = 10.0, 6.0
    gates = []
    for k in range(6):
        th = 2 * math.pi * k / 6
        tangent = math.atan2(b * math.cos(th), -a * math.sin(th))
        gates.append(Gate(f"gate_{k}", a * math.cos(th), b * math.sin(th),
                          2.0 if k % 2 == 0 else 3.5, math.degrees(tangent)))
    return Course(gates, "oval6")


def _slalom5() -> Course:
    """5 gates 8 m apart along +x, lateral offsets alternating +4 / -4 m, each
    yawed 20 deg toward the next one's side so it is approached head-on-ish."""
    gates = []
    for i in range(5):
        y = 4.0 if i % 2 == 0 else -4.0
        yaw = -20.0 if i % 2 == 0 else 20.0
        gates.append(Gate(f"gate_{i}", 8.0 * (i + 1), y, 2.5, yaw))
    return Course(gates, "slalom5")


FIXED_TRACKS = {"straight3": _straight3, "oval6": _oval6, "slalom5": _slalom5}


def write_tracks(out_dir: str = TRACKS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    for name, make in FIXED_TRACKS.items():
        path = os.path.join(out_dir, f"{name}.json")
        make().save(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-tracks", action="store_true",
                    help="regenerate rl/tracks/*.json")
    args = ap.parse_args()
    if args.write_tracks:
        write_tracks()
    else:
        ap.print_help()
