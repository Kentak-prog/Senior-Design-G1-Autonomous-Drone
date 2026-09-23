"""
Simulation-only gate detector: finds the solid-magenta test gates spawned by
`workspace/spawn_test_gates.py` by color-segmenting the image and extracting
the four corners of the gate OPENING (the hole in the magenta frame, not the
frame's outer silhouette).

This is a stand-in, not the real thing -- Tye's actual gate detector
(upstream of this directory) will replace it. The only reason it exists is
to exercise `eval_isaac_gates.py` and `ros_camera.GatePoseNode`'s detector
contract end-to-end against a live sim before Tye's detector is ready. Its
return type deliberately matches that contract:

    Callable[[np.ndarray], Sequence[np.ndarray]]

mapping one HxWx3 RGB image to a sequence of per-gate corner arrays, each a
4x2 float64 array of the gate OPENING corners in pixels, in ANY order --
`gate_pose.order_corners()` sorts them downstream, so this module does not
need to.
"""

from typing import Sequence, Tuple

import numpy as np
import cv2

# OpenCV HSV: H in [0, 180), S and V in [0, 255]. Tuned for
# spawn_test_gates.GATE_COLOR = (1.0, 0.0, 1.0) (magenta) under a flat
# UsdPreviewSurface + DistantLight -- widen these if lighting in the actual
# scene shifts the rendered hue away from pure magenta.
MAGENTA_HSV_LO = (135, 80, 60)
MAGENTA_HSV_HI = (170, 255, 255)


def gate_mask(rgb: np.ndarray,
             hsv_lo: Tuple[int, int, int] = MAGENTA_HSV_LO,
             hsv_hi: Tuple[int, int, int] = MAGENTA_HSV_HI) -> np.ndarray:
    """
    RGB (NOT BGR) image -> uint8 0/255 mask of magenta pixels.

    cv2.cvtColor(..., COLOR_RGB2HSV) expects RGB channel order, matching
    what cv_bridge hands back for encoding="rgb8" throughout this pipeline
    (see ros_camera.py's imgmsg_to_cv2 calls) -- do not swap this to
    COLOR_BGR2HSV just because that is the more common OpenCV tutorial
    example; doing so silently rotates the hue and this stops finding
    anything.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lo = np.array(hsv_lo, dtype=np.uint8)
    hi = np.array(hsv_hi, dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _refine_corners(mask: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Sub-pixel corner refinement via cv2.cornerSubPix, guarded against points
    too close to the image border (cornerSubPix's search window would run
    off the edge of the mask). Falls back to the unrefined points on any
    failure -- refinement is a bonus, not something a hole near the frame
    edge should be dropped over.

    cornerSubPix is gradient-based and expects a locally smooth intensity
    surface; `mask` is hard 0/255 with a one-pixel-wide step edge, which
    makes the iteration noisy right at the boundary. A light Gaussian blur
    gives it a real gradient to climb without moving the edge location
    enough to matter at the scales this module cares about.
    """
    h, w = mask.shape[:2]
    margin = 4
    if not np.all((pts[:, 0] >= margin) & (pts[:, 0] <= w - 1 - margin)
                  & (pts[:, 1] >= margin) & (pts[:, 1] <= h - 1 - margin)):
        return pts
    smoothed = cv2.GaussianBlur(mask, (5, 5), 0)
    corners = pts.astype(np.float32).reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    try:
        refined = cv2.cornerSubPix(smoothed, corners, (3, 3), (-1, -1), criteria)
    except cv2.error:
        return pts
    return refined.reshape(4, 2).astype(np.float64)


def _find_gate_corners(mask: np.ndarray, min_area_px: float,
                       refine: bool) -> list:
    """
    RETR_CCOMP gives a 2-level hierarchy: outer (frame) contours have
    parent == -1, and each frame's hole (the opening) is one of its
    children. For every big-enough outer contour, take its largest hole,
    approximate it to a quadrilateral, and accept only clean, convex
    quads -- a hole that is occluded, too small, or not resolvable at the
    current range/resolution just does not produce a 4-gon and is skipped,
    which is the correct behavior (contribute nothing rather than a bogus
    pose).
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or len(contours) == 0:
        return []
    hierarchy = hierarchy[0]  # (N, 4): [next, prev, first_child, parent]

    found = []  # (hole_area, corners)
    for i, h in enumerate(hierarchy):
        _next, _prev, first_child, parent = h
        if parent != -1:
            continue  # only outer (frame) contours
        if cv2.contourArea(contours[i]) < min_area_px:
            continue

        best_hole, best_area = None, 0.0
        child = first_child
        while child != -1:
            area = cv2.contourArea(contours[child])
            if area > best_area:
                best_area, best_hole = area, contours[child]
            child = hierarchy[child][0]
        if best_hole is None or best_area < min_area_px:
            continue

        peri = cv2.arcLength(best_hole, True)
        approx = cv2.approxPolyDP(best_hole, 0.04 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        pts = approx.reshape(4, 2).astype(np.float64)
        if refine:
            pts = _refine_corners(mask, pts)
        found.append((best_area, pts))

    found.sort(key=lambda item: -item[0])
    return [pts for _area, pts in found]


def detect_gates_color_debug(rgb: np.ndarray,
                             hsv_lo: Tuple[int, int, int] = MAGENTA_HSV_LO,
                             hsv_hi: Tuple[int, int, int] = MAGENTA_HSV_HI,
                             min_area_px: float = 64,
                             refine: bool = True):
    """
    HxWx3 RGB image -> (corners, mask): corners is a list of 4x2 float64
    arrays, one per detected gate opening, sorted by hole area descending
    (biggest/closest gate first); mask is the intermediate uint8
    segmentation the evaluator saves for debugging. This is where the real
    work happens; detect_gates_color() is a thin wrapper that drops mask to
    match the plain detector contract.
    """
    mask = gate_mask(rgb, hsv_lo, hsv_hi)
    corners = _find_gate_corners(mask, min_area_px, refine)
    return corners, mask


def detect_gates_color(rgb: np.ndarray,
                       hsv_lo: Tuple[int, int, int] = MAGENTA_HSV_LO,
                       hsv_hi: Tuple[int, int, int] = MAGENTA_HSV_HI,
                       min_area_px: float = 64,
                       refine: bool = True):
    """
    HxWx3 RGB image -> list of 4x2 float64 arrays, one per detected gate
    opening, sorted by hole area descending (biggest/closest gate first).

    This is the pure Callable[[np.ndarray], Sequence[np.ndarray]] contract
    ros_camera.GatePoseNode expects as `detector` -- no extra parameters
    beyond the image. Use detect_gates_color_debug() to also get the mask.
    """
    corners, _mask = detect_gates_color_debug(rgb, hsv_lo, hsv_hi, min_area_px, refine)
    return corners
