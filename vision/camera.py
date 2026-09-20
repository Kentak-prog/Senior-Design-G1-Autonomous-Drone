"""
Pinhole camera model and frame-convention helpers.

FRAME CONVENTIONS (get these wrong and everything downstream is silently wrong):

  OpenCV camera frame:  +X right, +Y DOWN, +Z FORWARD (into the scene)
  USD / Isaac camera:   +X right, +Y UP,   -Z FORWARD

  A point expressed in the USD camera frame maps to the OpenCV camera frame by
  R_CV_FROM_USD = diag(1, -1, -1).  cv2.solvePnP always returns poses in the
  OpenCV frame, so convert once, at the boundary, and never again.

  World frame is assumed Z-up (Isaac Sim default stage up-axis).
"""

from dataclasses import dataclass, field
import numpy as np

# Rotation mapping USD/Isaac camera-local coords -> OpenCV camera coords.
R_CV_FROM_USD = np.diag([1.0, -1.0, -1.0])


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics in pixels, plus optional distortion."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    # OpenCV distortion vector (k1, k2, p1, p2, k3). Isaac's ideal pinhole
    # camera has none, so this defaults to zeros.
    distortion: np.ndarray = field(default_factory=lambda: np.zeros(5))

    @classmethod
    def from_horizontal_fov(cls, width: int, height: int, hfov_deg: float):
        """Square pixels assumed. Handy for quick sanity checks."""
        fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        return cls(fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0,
                   width=width, height=height)

    @classmethod
    def from_usd_camera(cls, width: int, height: int,
                        focal_length: float,
                        horizontal_aperture: float,
                        vertical_aperture: float | None = None):
        """
        Build intrinsics from the attributes on a USD camera prim, which is
        what Isaac Sim actually exposes. focal_length and *_aperture must be
        in the same units (USD uses tenths of a world unit by default, but the
        ratio is all that matters).
        """
        fx = width * focal_length / horizontal_aperture
        if vertical_aperture is None:
            # Square pixels: vertical aperture is implied by the image aspect.
            fy = fx
        else:
            fy = height * focal_length / vertical_aperture
        return cls(fx=fx, fy=fy, cx=width / 2.0, cy=height / 2.0,
                   width=width, height=height)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]])

    @property
    def dist(self) -> np.ndarray:
        return np.asarray(self.distortion, dtype=float).reshape(-1, 1)

    def to_dict(self) -> dict:
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
                "width": self.width, "height": self.height,
                "distortion": np.asarray(self.distortion).tolist()}


def project_points(points_cam: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
    """
    Project Nx3 points given in the OpenCV camera frame to Nx2 pixels.
    Points at or behind the image plane (z <= 0) come back as NaN.
    """
    p = np.atleast_2d(np.asarray(points_cam, dtype=float))
    z = p[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = intr.fx * p[:, 0] / z + intr.cx
        v = intr.fy * p[:, 1] / z + intr.cy
    uv = np.stack([u, v], axis=1)
    uv[z <= 1e-9] = np.nan
    return uv


def in_frame(pixels: np.ndarray, intr: CameraIntrinsics, margin: int = 0) -> np.ndarray:
    """Boolean mask of which Nx2 pixels land inside the image."""
    uv = np.atleast_2d(pixels)
    ok = (~np.isnan(uv).any(axis=1)
          & (uv[:, 0] >= margin) & (uv[:, 0] < intr.width - margin)
          & (uv[:, 1] >= margin) & (uv[:, 1] < intr.height - margin))
    return ok


# --------------------------------------------------------------------------
# Rigid transform helpers. A transform T_a_b maps points from frame b to
# frame a:   p_a = T_a_b @ p_b.  Compose left to right: T_a_c = T_a_b @ T_b_c.
# --------------------------------------------------------------------------

def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(t, dtype=float).ravel()
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    p = np.atleast_2d(np.asarray(points, dtype=float))
    return (T[:3, :3] @ p.T).T + T[:3, 3]


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """
    Quaternion (w, x, y, z) -> 3x3 rotation matrix.
    Isaac Sim's get_world_pose() returns wxyz ordering. ROS uses xyzw. Check
    which one you are holding before calling this.
    """
    w, x, y, z = np.asarray(q, dtype=float).ravel()
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def usd_camera_pose_to_cv(position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """
    Take the world pose of a USD/Isaac camera prim and return T_world_cam in
    the OpenCV camera convention, ready to compose with a solvePnP result.
    """
    R_world_usd = quat_to_rot(quat_wxyz)
    R_world_cv = R_world_usd @ R_CV_FROM_USD.T
    return make_transform(R_world_cv, position)
