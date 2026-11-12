"""Colour-threshold gate detector: the classic approach, used here as the comparison.

HSV threshold for purple -> largest blob -> simplify its outline to 4 corners.
This is the same idea as the team's current colour gate detector; thresholds are
generic, so tune HSV_LO / HSV_HI to your scene before comparing on real frames.
"""
import cv2
import numpy as np

HSV_LO = np.array([125, 80, 35])
HSV_HI = np.array([170, 255, 255])


def order_corners(pts):
    """Return 4 points as TL, TR, BR, BL (image y points down)."""
    c = pts.mean(0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]                       # clockwise in image coords, starting near -180 deg
    start = np.argmin(np.abs(((np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]) + np.pi * 0.75 + np.pi)
                              % (2 * np.pi)) - np.pi))   # closest to the -135 deg (top-left) direction
    return np.roll(pts, -start, axis=0)


def detect(frame_bgr):
    """Returns (corners (4,2) or None, mask)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LO, HSV_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, mask
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 12:
        return None, mask
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in np.linspace(0.01, 0.12, 23):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float64)
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float64)
    return order_corners(quad), mask
