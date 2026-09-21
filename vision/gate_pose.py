"""
Gate pose estimation: four detected corner pixels -> 6-DOF gate pose in world.

This is the missing link between gate detection (Tye's OpenCV module) and the
waypoint generator. A bounding box is not a position; this module turns pixel
corners into metric coordinates using the known physical gate size.

Pipeline:
    corners (4x2 px)  ->  solvePnP  ->  T_cam_gate  ->  T_world_gate
                                                    ->  approach waypoints

GATE MODEL FRAME:
    Origin at the center of the gate opening, square lying in the z = 0
    plane. +X right, +Y DOWN, +Z along the FLIGHT DIRECTION through the
    gate. This is the same handedness convention as the OpenCV camera
    frame, which is what makes T_cam_gate compose without surprises.

    +Y is down rather than up on purpose. "+X right, +Y up, +Z flight
    direction" is NOT right-handed from the pilot's point of view: with a
    camera approaching along the gate's -Z, requiring the gate to appear
    upright forces X_cam = -X_gate, so the gate's "+X right" is actually
    the pilot's left. Defining +Y down removes the contradiction and makes
    solvePnP return a normal that points the way the drone flies, rather
    than back at the camera.

CORNER ORDERING:
    Everything below assumes (top-left, top-right, bottom-right, bottom-left)
    as seen in the image. Use order_corners() on raw detector output first.

SOLVER NOTE:
    Uses SQPNP, then Levenberg-Marquardt refinement. Do NOT switch this to
    SOLVEPNP_IPPE_SQUARE without re-running test_gate_pose.py: on OpenCV
    4.13 the IPPE solvers returned rotations that reprojected with >100 px
    RMS on inputs where SQPNP was exact to 1e-14.
"""

from dataclasses import dataclass
import numpy as np
import cv2

from camera import (CameraIntrinsics, make_transform, transform_points,
                    project_points)

# Gate opening, inner edge to inner edge, in meters. Measure the actual gate
# prim in Isaac Sim and set this to match: a 5% error here becomes a 5% range
# error, which test_gate_pose.py asserts explicitly.
DEFAULT_GATE_SIDE = 1.5


def gate_model_points(side: float = DEFAULT_GATE_SIDE) -> np.ndarray:
    """
    4x3 gate corners in the gate frame, ordered TL, TR, BR, BL to match the
    image-space ordering produced by order_corners().

    Y is negated relative to the naive "+Y is up" layout so the frame is
    right-handed with +Z along the flight direction (see module docstring).
    Flipping these signs flips the recovered gate normal, which silently
    reverses every approach waypoint. test_waypoint_ordering guards it.
    """
    h = side / 2.0
    return np.array([
        [-h, -h, 0.0],  # top-left      (+Y is DOWN in the gate frame)
        [h, -h, 0.0],   # top-right
        [h, h, 0.0],    # bottom-right
        [-h, h, 0.0],   # bottom-left
    ], dtype=np.float64)


def order_corners(pixels: np.ndarray) -> np.ndarray:
    """
    Sort four unordered corner pixels into TL, TR, BR, BL.

    Sorts by angle about the centroid, which survives the gate being rotated
    or seen obliquely, then rotates the cycle to start at the corner nearest
    the top-left of the bounding box. Breaks down past roughly 45 degrees of
    roll; de-rotate the pixels with the IMU roll estimate first if you get
    there.
    """
    p = np.asarray(pixels, dtype=np.float64).reshape(4, 2)
    c = p.mean(axis=0)
    # Image y grows downward, so negate it for a conventional CCW angle, then
    # sort descending to walk clockwise: TL -> TR -> BR -> BL.
    ang = np.arctan2(-(p[:, 1] - c[1]), p[:, 0] - c[0])
    p = p[np.argsort(-ang)]
    start = int(np.argmin(np.linalg.norm(p - p.min(axis=0), axis=1)))
    return np.roll(p, -start, axis=0)


def _reprojection_rms(model, corners, rvec, tvec, intr) -> float:
    """RMS pixel distance between observed corners and a candidate pose."""
    proj, _ = cv2.projectPoints(model, rvec, tvec, intr.K, intr.dist)
    d = proj.reshape(-1, 2) - corners
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def _solve(model, corners, intr):
    """SQPNP + LM refinement. Returns (rvec, tvec) or None."""
    ok, rvec, tvec = cv2.solvePnP(model, corners, intr.K, intr.dist,
                                  flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(model, corners, intr.K, intr.dist,
                                      rvec, tvec)
    return rvec, tvec


@dataclass
class GateDetection:
    """One gate observation with an honest uncertainty attached."""

    T_cam_gate: np.ndarray          # 4x4, gate pose in the OpenCV camera frame
    reprojection_error_px: float    # RMS over the four corners
    range_m: float                  # distance camera -> gate center
    corners_px: np.ndarray          # 4x2, the ordered input corners
    center_std_m: float = np.nan    # bootstrap 1-sigma on the gate center
    normal_std_deg: float = np.nan  # bootstrap 1-sigma on the gate normal

    @property
    def normal_is_reliable(self) -> bool:
        """
        Near head-on, a gate tilted left and one tilted right project to
        almost the same quadrilateral, so the NORMAL becomes noise long
        before the CENTER does. When this is False, still use the center as
        a waypoint, but do not trust the approach direction.
        """
        return bool(self.normal_std_deg < 15.0)

    def center_cam(self) -> np.ndarray:
        return self.T_cam_gate[:3, 3]

    def normal_cam(self) -> np.ndarray:
        """Unit gate normal, pointing along the flight direction (away from
        the camera), expressed in the OpenCV camera frame."""
        return self.T_cam_gate[:3, 2]


def estimate_gate_pose(corners_px: np.ndarray,
                       intr: CameraIntrinsics,
                       gate_side: float = DEFAULT_GATE_SIDE,
                       assume_ordered: bool = False,
                       corner_sigma_px: float = 0.0,
                       bootstrap_n: int = 12,
                       seed: int = 0) -> "GateDetection | None":
    """
    Solve for the gate pose in the camera frame from four corner pixels.

    corner_sigma_px is the detector's corner-localization error. Pass the
    value measured from Tye's detector and the returned pose carries a
    bootstrap uncertainty: the corners are re-jittered by that sigma and
    re-solved, and the spread of the results is reported. Pass 0 to skip it
    (about 12x cheaper, still well under a millisecond either way).

    Returns None if the solver fails.
    """
    corners = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    if not assume_ordered:
        corners = order_corners(corners)

    model = gate_model_points(gate_side)
    sol = _solve(model, corners, intr)
    if sol is None:
        return None
    rvec, tvec = sol

    R, _ = cv2.Rodrigues(rvec)
    t = np.asarray(tvec, dtype=float).ravel()
    T_cam_gate = make_transform(R, t)

    center_std, normal_std = np.nan, np.nan
    if corner_sigma_px > 0.0 and bootstrap_n > 1:
        rng = np.random.default_rng(seed)
        centers, normals = [], []
        for _ in range(bootstrap_n):
            jittered = corners + rng.normal(0.0, corner_sigma_px, corners.shape)
            s = _solve(model, jittered, intr)
            if s is None:
                continue
            Rb, _ = cv2.Rodrigues(s[0])
            centers.append(np.asarray(s[1]).ravel())
            normals.append(Rb[:, 2])
        if len(centers) > 2:
            centers = np.array(centers)
            normals = np.array(normals)
            center_std = float(np.sqrt(np.mean(
                np.sum((centers - centers.mean(axis=0)) ** 2, axis=1))))
            mean_n = normals.mean(axis=0)
            mean_n /= max(np.linalg.norm(mean_n), 1e-12)
            cosang = np.clip(normals @ mean_n, -1.0, 1.0)
            normal_std = float(np.degrees(np.sqrt(np.mean(np.arccos(cosang) ** 2))))

    return GateDetection(
        T_cam_gate=T_cam_gate,
        reprojection_error_px=_reprojection_rms(model, corners, rvec, tvec, intr),
        range_m=float(np.linalg.norm(t)),
        corners_px=corners,
        center_std_m=center_std,
        normal_std_deg=normal_std,
    )


def gate_pose_world(det: GateDetection, T_world_cam: np.ndarray) -> np.ndarray:
    """
    Lift a camera-frame detection into the world frame.

    T_world_cam comes from the drone's state estimate composed with the
    body->camera extrinsic. Error in the drone's own pose adds directly to the
    gate position, which is exactly what Task 2.2 quantifies.
    """
    return T_world_cam @ det.T_cam_gate


def approach_waypoints(T_world_gate: np.ndarray,
                       standoff: float = 1.0) -> np.ndarray:
    """
    Turn a gate pose into the three points the path generator actually wants:
    a pre-gate point, the gate center, and a post-gate point, all on the gate
    normal. Feeding these into a spline instead of bare gate centers is what
    stops the drone clipping a gate frame while cutting a corner.

    Returns 3x3 in world coordinates, in FLIGHT ORDER: row 0 is reached
    first, row 2 last. That ordering depends entirely on +Z of the gate
    frame being the flight direction -- see gate_model_points().

    Check det.normal_is_reliable first; if it is False, hand the planner the
    center alone.
    """
    center = T_world_gate[:3, 3]
    normal = T_world_gate[:3, 2]
    normal = normal / max(np.linalg.norm(normal), 1e-9)
    return np.array([center - standoff * normal,
                     center,
                     center + standoff * normal])


def project_gate(T_cam_gate: np.ndarray,
                 intr: CameraIntrinsics,
                 gate_side: float = DEFAULT_GATE_SIDE) -> np.ndarray:
    """Forward model: gate pose -> 4x2 corner pixels. Used by the evaluator."""
    pts_cam = transform_points(T_cam_gate, gate_model_points(gate_side))
    return project_points(pts_cam, intr)
