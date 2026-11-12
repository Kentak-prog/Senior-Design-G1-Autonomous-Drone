"""Shared settings for the gate CNN.

Change these to match your Isaac Sim scene. The two that matter most:
  GATE_SIZE_M  - outer width/height of a gate (used by PnP to get distance)
  CAM_FOV_DEG  - horizontal and vertical field of view of the front camera
"""
import numpy as np

# ---- Gate geometry (metres) -------------------------------------------------
GATE_SIZE_M = 1.9        # outer edge: 1.5 m opening + 0.2 m frame on each side (from Tony)
GATE_BORDER_M = 0.2      # width of the purple frame bars (from Tony)
GATE_DEPTH_M = 0.12      # how deep the gate box is (Isaac gates are 3D boxes)

# Gate colour in BGR (OpenCV order), measured from Aaron's Isaac recording.
GATE_BGR = (180, 5, 140)

# ---- Camera ------------------------------------------------------------------
# Native render / camera resolution used for evaluation (keeps the 60:36 aspect).
IMG_W, IMG_H = 256, 160
CAM_FOV_DEG = (60.0, 36.0)   # same FOV Tony uses in race_env

# The CNN runs on a half-size copy of the frame.
NET_W, NET_H = 128, 80
NET_STRIDE = 4               # output heatmap is (NET_H/4, NET_W/4) = 20 x 32


def intrinsics(w=IMG_W, h=IMG_H, fov=CAM_FOV_DEG):
    """Pinhole camera matrix K for an image of size w x h."""
    fx = (w / 2) / np.tan(np.radians(fov[0] / 2))
    fy = (h / 2) / np.tan(np.radians(fov[1] / 2))
    return np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], dtype=np.float64)


def gate_corners_local(size=GATE_SIZE_M):
    """Outer corners in the gate's own frame: x right, y up, z = 0 plane.
    Order is TL, TR, BR, BL as seen from the front (the order cv2.SOLVEPNP_IPPE_SQUARE wants)."""
    s = size / 2
    return np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
