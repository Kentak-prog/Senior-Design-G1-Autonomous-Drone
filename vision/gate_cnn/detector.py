"""Gate detection API: drop this into the Isaac vision loop.

    from detector import GateCNN, estimate_pose
    cnn = GateCNN("gate_cnn.npz")
    corners, conf = cnn.detect(frame_bgr)          # 4 x 2 pixels (TL, TR, BR, BL), 4 confidences
    if conf.min() > 0.3:
        ok, t_cam, R_cam = estimate_pose(corners, K)  # gate centre in camera frame (metres)

`corners` come back in the pixel coordinates of whatever frame you pass in, at any
resolution, as long as its aspect ratio is about 60:36 (5:3). Pass K for that resolution.
The output plugs into the same PnP step the colour detector feeds today.
"""
import cv2
import numpy as np

import config as C
from nn_numpy import GateNet, spatial_softmax
from train import cell_to_img


class GateCNN:
    def __init__(self, weights="gate_cnn.npz"):
        self.net = GateNet.load(weights)

    def detect_batch(self, frames_bgr):
        """frames: list of BGR images. Returns corners (N,4,2) and confidence (N,4)."""
        h0, w0 = frames_bgr[0].shape[:2]
        x = np.stack([cv2.resize(f, (C.NET_W, C.NET_H), interpolation=cv2.INTER_AREA) for f in frames_bgr])
        p = spatial_softmax(self.net.forward(self.net.preprocess(x)))     # (N, 20, 32, 4)
        N, hh, hw, K = p.shape
        corners = np.zeros((N, K, 2))
        conf = np.zeros((N, K))
        for n in range(N):
            for k in range(K):
                pk = p[n, :, :, k]
                cy, cx = np.unravel_index(np.argmax(pk), pk.shape)
                y0, y1, x0, x1 = max(cy - 1, 0), min(cy + 2, hh), max(cx - 1, 0), min(cx + 2, hw)
                win = pk[y0:y1, x0:x1]
                ys, xs = np.mgrid[y0:y1, x0:x1]
                s = win.sum()
                corners[n, k] = cell_to_img(np.array([(win * xs).sum() / s, (win * ys).sum() / s]))
                conf[n, k] = s      # probability mass near the peak: ~1 = sure, ~0 = no corner seen
        corners *= [w0 / C.IMG_W, h0 / C.IMG_H]
        return corners, conf

    def detect(self, frame_bgr):
        c, p = self.detect_batch([frame_bgr])
        return c[0], p[0]


def fully_in_view(corners, frame_shape, margin_frac=0.02):
    """False when a corner sits near the frame edge. A gate cut off by the edge gives
    corners on the border, which would make PnP wrong, so skip those detections."""
    h, w = frame_shape[:2]
    m = margin_frac * w
    return bool(((corners[:, 0] > m) & (corners[:, 0] < w - m) &
                 (corners[:, 1] > m) & (corners[:, 1] < h - m)).all())


def edges_on_gate(frame_bgr, corners, min_frac=0.75):
    """Sanity check: the 4 sides of the detected quad should run along gate-coloured pixels.
    Catches the case where no single gate is fully visible and the CNN stitches corners from
    two different gates (its sides then cross empty background)."""
    import baseline
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, baseline.HSV_LO, baseline.HSV_HI)
    r = max(2, frame_bgr.shape[1] // 120)
    mask = cv2.dilate(mask, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
    h, w = mask.shape
    for k in range(4):
        a, b = corners[k], corners[(k + 1) % 4]
        s = np.linspace(0.1, 0.9, 25)[:, None]
        pts = np.round(a + (b - a) * s).astype(int)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        if (mask[pts[:, 1], pts[:, 0]] > 0).mean() < min_frac:
            return False
    return True


def good_detection(frame_bgr, corners, conf, conf_min=0.45):
    """All checks in one place: confident, not edge-on, whole gate in view, sides lie on the gate."""
    sides = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
    not_edge_on = sides.min() > 0.04 * frame_bgr.shape[1]   # a gate seen edge-on is useless for PnP
    squarish = sides.min() / sides.max() > 0.3                # a single bar of a gate is long and thin
    return bool(conf.min() > conf_min and not_edge_on and squarish and fully_in_view(corners, frame_bgr.shape)
                and edges_on_gate(frame_bgr, corners) and is_hollow(frame_bgr, corners))


def is_hollow(frame_bgr, corners, max_fill=0.4):
    """A gate is a frame with an opening: the middle of the detected quad must be mostly NOT gate-coloured.
    Rejects a solid bar of a gate (seen up close) being mistaken for a whole gate."""
    import baseline
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, baseline.HSV_LO, baseline.HSV_HI)
    c = corners.mean(0)
    inner = c + (corners - c) * 0.5
    region = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(region, [np.round(inner).astype(np.int32)], 255)
    n = (region > 0).sum()
    return n > 0 and ((mask > 0) & (region > 0)).sum() / n < max_fill


def refine_with_colour(frame_bgr, corners, max_shift_frac=0.03):
    """Optional sharpening step. The CNN decides WHICH gate and roughly where its corners are;
    the colour fit, restricted to that gate's region, then snaps each corner to sub-pixel edges.
    A corner only moves if the colour fit agrees with the CNN (within max_shift_frac of the
    image width), so blur, noise or a gate behind can't drag it away."""
    import baseline
    h, w = frame_bgr.shape[:2]
    region = np.zeros((h, w), np.uint8)
    c = corners.mean(0)
    grown = c + (corners - c) * 1.08 + np.sign(corners - c) * 2
    cv2.fillPoly(region, [np.round(grown).astype(np.int32)], 255)
    masked = cv2.bitwise_and(frame_bgr, frame_bgr, mask=region)
    quad, _ = baseline.detect(masked)
    if quad is None:
        return corners
    # Motion blur smears purple outward, which makes the colour outline too big. If the colour
    # fit is more than 3% larger than the CNN's gate, trust the CNN.
    if cv2.contourArea(quad.astype(np.float32)) > 1.03 * cv2.contourArea(corners.astype(np.float32)):
        return corners
    d = np.linalg.norm(quad - corners, axis=1)
    keep = d < max_shift_frac * w
    out = corners.copy()
    out[keep] = quad[keep]
    return out


def estimate_pose(corners, K, size=C.GATE_SIZE_M):
    """PnP on the 4 outer corners. Returns (ok, t, R): gate centre and orientation in the
    camera frame (OpenCV axes: x right, y down, z forward)."""
    obj = C.gate_corners_local(size)
    ok, rvec, tvec = cv2.solvePnP(obj, np.asarray(corners, np.float64).reshape(4, 1, 2), K, None,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return False, None, None
    return True, tvec.ravel(), cv2.Rodrigues(rvec)[0]
