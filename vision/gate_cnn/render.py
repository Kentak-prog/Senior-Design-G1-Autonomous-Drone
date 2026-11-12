"""Synthetic camera frames of purple gates on a black background.

This stands in for Isaac Sim until real captures are collected. Every frame comes
with exact ground truth: the pixel position of the nearest gate's 4 outer corners
and that gate's 3D position relative to the camera.

Camera frame is OpenCV's: x right, y down, z forward.
"""
import numpy as np
import cv2

import config as C

SS = 2  # supersampling factor for anti-aliased edges


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# Maps gate-local (x right, y up, z toward viewer) into camera axes for a gate seen head-on.
FACING = np.diag([1.0, -1.0, -1.0])


def gate_pose(t, yaw=0.0, pitch=0.0, roll=0.0):
    """Rotation (gate->camera) for a gate at camera-frame position t."""
    R = rot_z(roll) @ rot_x(pitch) @ rot_y(yaw) @ FACING
    return R, np.asarray(t, dtype=np.float64)


def project(P_cam, K):
    z = P_cam[:, 2:3]
    return (P_cam[:, :2] / z) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]


def draw_floor(img, K, fl):
    """Isaac-style ground: light blue-white plane with a 1 m white grid, finite size, black beyond.
    fl: dict(up=unit 'up' vector in camera frame, h=camera height (m), amax, bmax=extent (m),
             base=BGR, line=BGR, grid=m)."""
    H, W = img.shape[:2]
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    up = np.asarray(fl["up"], np.float32)
    denom = d @ up
    s = np.where(denom < -1e-4, -fl["h"] / np.minimum(denom, -1e-4), 1e6).astype(np.float32)
    P = d * s[..., None]
    fwd = np.array([0, 0, 1.0]) - up * up[2]
    fwd /= np.linalg.norm(fwd)
    right = np.cross(up, fwd)  # any in-plane axis works for a grid
    a = P @ fwd.astype(np.float32) + fl.get("a0", 0)
    b = P @ right.astype(np.float32) + fl.get("b0", 0)
    on = (a < fl["amax"]) & (np.abs(b) < fl["bmax"]) & (s < 200)
    g = fl.get("grid", 1.0)
    # line width grows with distance a little so far lines don't vanish completely
    lw = 0.03 + 0.004 * np.minimum(s, 60)
    fa = np.abs(((a / g) + 0.5) % 1 - 0.5) * g
    fb = np.abs(((b / g) + 0.5) % 1 - 0.5) * g
    line = ((fa < lw) | (fb < lw)).astype(np.float32)
    shade = np.clip(1.05 - 0.004 * np.minimum(s, 60), 0.8, 1.05)[..., None]
    col = (np.array(fl["base"], np.float32) * (1 - line[..., None]) + np.array(fl["line"], np.float32) * line[..., None]) * shade
    img[on] = col[on]
    return img


def render(gates, w=C.IMG_W, h=C.IMG_H, color=C.GATE_BGR, floor=None, styles=None):
    """Draw gates (list of (R, t)) far-to-near. Returns uint8 BGR image and, per gate,
    its 4 front outer corner pixels (TL, TR, BR, BL) in the w x h image.
    styles: optional per-gate dicts with 'border' (m), 'depth' (m) and 'color' (BGR).
    floor: optional dict for draw_floor()."""
    K = C.intrinsics(w, h)
    Kss = C.intrinsics(w * SS, h * SS)
    img = np.zeros((h * SS, w * SS, 3), np.float32)
    if floor is not None:
        img = draw_floor(img, Kss, floor)
    corners_px = []
    order = np.argsort([-g[1][2] for g in gates])  # far first
    for gi in order:
        R, t = gates[gi]
        st = (styles[gi] if styles else None) or {}
        border = st.get("border", C.GATE_BORDER_M)
        depth = st.get("depth", C.GATE_DEPTH_M)
        base = np.array(st.get("color", color), np.float32)
        outer = C.gate_corners_local()
        inner = C.gate_corners_local(C.GATE_SIZE_M - 2 * border)
        back = np.array([0, 0, -depth])
        Pts = {k: v @ R.T + t for k, v in
               dict(fo=outer, fi=inner, bo=outer + back, bi=inner + back).items()}
        if any((P[:, 2] < 0.15).any() for P in Pts.values()):
            continue
        px = {k: np.round(project(P, Kss) * 16).astype(np.int32) for k, P in Pts.items()}
        dist_fade = np.clip(1.15 - 0.02 * t[2], 0.75, 1.0)

        def paint(polys_add, polys_cut, shade):
            nonlocal img
            mask = np.zeros(img.shape[:2], np.uint8)
            for p in polys_add:
                cv2.fillPoly(mask, [p], 255, cv2.LINE_AA, shift=4)
            for p in polys_cut:
                cv2.fillPoly(mask, [p], 0, cv2.LINE_AA, shift=4)
            a = (mask.astype(np.float32) / 255)[..., None]
            img = img * (1 - a) + base * shade * dist_fade * a

        if depth > 0:
            paint([px["bo"]], [px["bi"]], 0.45)
            # side walls: local face normals (outer: top +y, right +x, bottom -y, left -x)
            ups = [0.9, 0.62, 0.35, 0.62]
            for e in range(4):
                n = (e + 1) % 4
                paint([np.array([px["fo"][e], px["fo"][n], px["bo"][n], px["bo"][e]])], [], ups[e])
                paint([np.array([px["fi"][e], px["fi"][n], px["bi"][n], px["bi"][e]])], [], ups[(e + 2) % 4])
        normal_cam = R @ np.array([0, 0, 1.0])
        lit = 0.7 + 0.3 * abs(normal_cam[2])
        paint([px["fo"]], [px["fi"]], lit)
    for R, t in gates:
        corners_px.append(project(C.gate_corners_local() @ R.T + t, K))
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return np.clip(img, 0, 255).astype(np.uint8), corners_px


# ---------------------------------------------------------------- augmentation

def motion_blur(img, k, angle):
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1)
    ker = cv2.warpAffine(ker, M, (k, k))
    ker /= ker.sum()
    return cv2.filter2D(img, -1, ker)


def clutter(img, rng, strength=1.0):
    """Faint scenery: grey floor lines plus a few small coloured lights, some purple-ish."""
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    layer = np.zeros_like(out)
    for _ in range(rng.integers(3, 9)):
        p1 = (int(rng.integers(0, w)), int(rng.integers(h // 2, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(h // 2, h)))
        g = float(rng.uniform(15, 45))
        cv2.line(layer, p1, p2, (g, g, g), 1, cv2.LINE_AA)
    for _ in range(rng.integers(1, 5)):
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        r = int(rng.integers(1, 4))
        col = [(180, 60, 200), (220, 90, 120), (60, 60, 220), (200, 160, 60)][rng.integers(0, 4)]
        cv2.circle(layer, c, r, col, -1, cv2.LINE_AA)
    out = np.maximum(out, layer * strength)
    return np.clip(out, 0, 255).astype(np.uint8)


def augment(img, rng, p):
    """p: dict of augmentation settings. Missing keys mean 'off'."""
    out = img
    if p.get("clutter", 0) and rng.random() < p["clutter"]:
        out = clutter(out, rng)
    if p.get("mblur", 0) and rng.random() < p["mblur"][0]:
        out = motion_blur(out, int(rng.integers(p["mblur"][1], p["mblur"][2] + 1)) | 1,
                          rng.uniform(0, 180))
    if p.get("gblur", 0) and rng.random() < p["gblur"]:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.5, 1.2))
    f = out.astype(np.float32)
    if "gain" in p:
        f *= rng.uniform(*p["gain"])
    if "chan" in p:
        f *= 1 + rng.uniform(-p["chan"], p["chan"], 3)
    if "lift" in p:
        f += rng.uniform(0, p["lift"])
    if "noise" in p:
        f += rng.normal(0, rng.uniform(*p["noise"]), f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


EXACT_GATE = False  # True = every gate drawn with the real frame width from config (used for testing)

TRAIN_AUG = dict(clutter=0.35, mblur=(0.3, 3, 9), gblur=0.15, gain=(0.35, 1.3), chan=0.15,
                 lift=12, noise=(0, 14))

# Evaluation conditions (each is a fixed recipe so methods are compared on identical frames)
EVAL_CONDITIONS = {
    "clean":        dict(scene=dict()),
    "sensor noise": dict(aug=dict(noise=(10, 18))),
    "motion blur":  dict(aug=dict(mblur=(1.0, 7, 11))),
    "low light":    dict(aug=dict(gain=(0.25, 0.4), noise=(3, 6))),
    "clutter":      dict(aug=dict(clutter=1.0, lift=10)),
    "gates behind": dict(scene=dict(distractors=(2, 3), overlap=True)),
    "Isaac floor":  dict(scene=dict(distractors=(0, 2), floor_p=1.0)),
}


# ---------------------------------------------------------------- scene sampling

def sample_scene(rng, distractors=None, overlap=False, all_corners_in=False, near=False, floor_p=0.65):
    """Random camera view. The first gate is the nearest one and is the label.
    Returns (gates, styles, floor) ready for render()."""
    K = C.intrinsics()
    with_floor = rng.random() < floor_p
    cam_roll, cam_pitch = rng.uniform(-0.45, 0.45), rng.uniform(-0.2, 0.2)

    def pose(t, yaw):
        if with_floor:   # upright gates: same tilt as the camera sees the ground
            return gate_pose(t, yaw, cam_pitch + rng.normal(0, 0.03), cam_roll + rng.normal(0, 0.03))
        return gate_pose(t, yaw, rng.uniform(-0.2, 0.2), rng.uniform(-0.5, 0.5))

    for _ in range(100):
        dist = float(np.exp(rng.uniform(np.log(1.4 if near else 1.6), np.log(3.0 if near else 10))))
        u = rng.uniform(-0.1, 1.1) * C.IMG_W if not all_corners_in else rng.uniform(0.25, 0.75) * C.IMG_W
        v = rng.uniform(-0.1, 1.1) * C.IMG_H if not all_corners_in else rng.uniform(0.25, 0.75) * C.IMG_H
        ray = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
        t = ray / np.linalg.norm(ray) * dist
        g = pose(t, rng.uniform(-0.95, 0.95))
        px = project(C.gate_corners_local() @ g[0].T + g[1], K)
        if all_corners_in:
            if not ((px[:, 0] > 2) & (px[:, 0] < C.IMG_W - 2) & (px[:, 1] > 2) & (px[:, 1] < C.IMG_H - 2)).all():
                continue
        else:
            inside = (px[:, 0] > 0) & (px[:, 0] < C.IMG_W) & (px[:, 1] > 0) & (px[:, 1] < C.IMG_H)
            if inside.sum() < 2:
                continue
        break
    gates = [g]
    n_d = rng.integers(distractors[0], distractors[1] + 1) if distractors else \
        (rng.integers(1, 3) if rng.random() < 0.5 else 0)
    for _ in range(n_d):
        d2 = dist + rng.uniform(1.5, 8)
        if overlap:  # place behind and roughly in line, so it shows through the near gate
            ray2 = t / np.linalg.norm(t) + rng.normal(0, 0.12, 3)
        else:
            ray2 = np.array([rng.uniform(-0.7, 0.7), rng.uniform(-0.4, 0.4), 1.0])
        gates.append(pose(ray2 / np.linalg.norm(ray2) * d2, rng.uniform(-0.9, 0.9)))

    styles = []
    for _ in gates:
        base = np.array(C.GATE_BGR if rng.random() < 0.75 else (200, 40, 170), float)
        border = C.GATE_BORDER_M if EXACT_GATE else rng.uniform(0.08, 0.2) * C.GATE_SIZE_M / 1.5
        styles.append(dict(border=border, depth=rng.uniform(0.0, 0.2),
                           color=tuple(np.clip(base + rng.uniform(-25, 25, 3), 0, 255))))
    floor = None
    if with_floor:
        up = rot_z(cam_roll) @ rot_x(cam_pitch) @ np.array([0, -1.0, 0])
        R0, t0 = gates[0]
        low = max(-(up @ (R0 @ c + t0)) for c in C.gate_corners_local())
        base = np.array([215, 205, 190]) + rng.uniform(-30, 20, 3)
        floor = dict(up=up, h=low + rng.uniform(0.3, 1.6), amax=rng.uniform(6, 40), bmax=rng.uniform(4, 25),
                     a0=rng.uniform(0, 1), b0=rng.uniform(0, 1), base=base,
                     line=np.minimum(base + rng.uniform(15, 40), 255), grid=rng.choice([0.5, 1.0, 1.0, 2.0]))
    return gates, styles, floor


def make_example(rng, aug=None, **scene_kw):
    gates, styles, floor = sample_scene(rng, **scene_kw)
    img, corners = render(gates, styles=styles, floor=floor)
    if aug:
        img = augment(img, rng, aug)
    return img, corners[0], gates[0]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    tiles = []
    for i in range(12):
        img, cps, _ = make_example(rng, aug=TRAIN_AUG)
        vis = img.copy()
        for k, (x, y) in enumerate(cps):
            cv2.circle(vis, (int(x), int(y)), 3, [(0, 255, 0), (0, 255, 255), (255, 255, 0), (0, 128, 255)][k], 1)
        tiles.append(vis)
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, 12, 4)]
    cv2.imwrite("samples.png", np.vstack(rows))
    print("wrote samples.png")
