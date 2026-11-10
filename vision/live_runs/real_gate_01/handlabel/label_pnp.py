"""Hand-labelled PnP for real_gate_01: snaps eyeballed corners to the PVC pipe
centreline (top-hat ridge + robust line fit per side), solves PnP with the
approximate intrinsics, writes label_<frame>.jpg + labels.json to OUT.

    python3 label_pnp.py <out_dir>      # e.g. . to regenerate in place
"""
import sys, json, os, cv2, numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # vision/
from camera import CameraIntrinsics
from gate_pose import estimate_gate_pose, project_gate
RUN = os.path.dirname(_HERE)
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
ROUGH = {  # TL, TR, BR, BL (pixels), eyeballed from gridded frames
 360: [(485,168),(755,132),(770,443),(476,428)],
 420: [(440,145),(722,115),(748,416),(437,416)],
 540: [(460,130),(705,78),(798,307),(488,365)],
 600: [(494,86),(751,98),(781,346),(494,350)],
 720: [(458,93),(718,93),(744,349),(454,349)],
 900: [(463,98),(722,104),(741,353),(456,349)],
 1200:[(463,98),(722,104),(741,353),(456,349)],
 1500:[(463,98),(722,104),(741,353),(456,349)],
 1800:[(458,104),(715,88),(748,347),(458,355)],
 2100:[(458,104),(715,88),(748,347),(458,355)],
}
SIDE = 0.9144
intr = CameraIntrinsics(fx=504.9, fy=504.9, cx=512, cy=384, width=1024, height=768)

def ridge(gray):
    g = gray.astype(np.float32)
    return cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))

def fit_side(R, a, b, rad=22):
    a, b = np.float64(a), np.float64(b); d = b - a; L = np.linalg.norm(d); u = d / L; n = np.array([-u[1], u[0]])
    pts = []
    for s in np.linspace(0.15, 0.85, 40):
        p0 = a + d * s
        offs = np.arange(-rad, rad + 1)
        q = p0 + offs[:, None] * n
        xi, yi = np.clip(q[:, 0].round().astype(int), 0, R.shape[1]-1), np.clip(q[:, 1].round().astype(int), 0, R.shape[0]-1)
        v = R[yi, xi]
        k = int(np.argmax(v))
        if v[k] < 25: continue
        # sub-pixel centroid of the ridge (pipe centreline) around the max
        w = np.clip(v[max(k-4,0):k+5] - 0.5 * v[k], 0, None)
        o = offs[max(k-4,0):k+5]
        pts.append(p0 + n * ((w * o).sum() / max(w.sum(), 1e-6)))
    pts = np.array(pts)
    if len(pts) < 10: return None, pts
    # robust line fit
    line = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return line, pts

def intersect(l1, l2):
    (vx1, vy1, x1, y1), (vx2, vy2, x2, y2) = l1, l2
    A = np.array([[vx1, -vx2], [vy1, -vy2]]); t = np.linalg.solve(A, [x2 - x1, y2 - y1])
    return np.array([x1 + vx1 * t[0], y1 + vy1 * t[0]])

rows, labels = [], {}
for f, rough in ROUGH.items():
    img = cv2.imread(f"{RUN}/raw_jpg/{f:05d}.jpg"); R = ridge(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    lines, allpts = [], []
    for k in range(4):
        l, p = fit_side(R, rough[k], rough[(k+1) % 4]); lines.append(l); allpts.append(p)
    if any(l is None for l in lines):
        print(f, "side fit failed"); continue
    c = np.array([intersect(lines[(k-1) % 4], lines[k]) for k in range(4)])  # corner k = side(k-1) x side(k)
    shift = np.linalg.norm(c - np.array(rough, float), axis=1)
    g = estimate_gate_pose(c, intr, gate_side=SIDE)
    labels[f] = c.round(2).tolist()
    vis = img.copy()
    for p in allpts:
        for q in p: cv2.circle(vis, tuple(q.round().astype(int)), 2, (255, 0, 0), -1)
    cv2.polylines(vis, [c.round().astype(int).reshape(-1, 1, 2)], True, (0, 255, 255), 2)
    rep = project_gate(g.T_cam_gate, intr, SIDE).round().astype(int)
    cv2.polylines(vis, [rep.reshape(-1, 1, 2)], True, (0, 255, 0), 1)
    x, y, z = g.T_cam_gate[:3, 3]; nrm = g.T_cam_gate[:3, 2]
    yaw = np.degrees(np.arctan2(nrm[0], nrm[2]))
    for i, t in enumerate([f"frame {f}  range {g.range_m:.2f} m  reproj {g.reprojection_error_px:.1f}px",
                           f"xyz {x:+.2f} {y:+.2f} {z:+.2f}  normal yaw {yaw:+.0f} deg",
                           "approx intrinsics (no distortion), side 0.914 m"]):
        cv2.putText(vis, t, (10, 25 + 24 * i), 0, 0.65, (0, 0, 0), 3); cv2.putText(vis, t, (10, 25 + 24 * i), 0, 0.65, (0, 255, 0), 1)
    cv2.imwrite(f"{OUT}/label_{f:05d}.jpg", vis)
    rows.append((f, g.range_m, x, y, z, yaw, g.reprojection_error_px, shift.max()))
    print(f"{f:5d} range {g.range_m:5.2f} m  xyz {x:+.2f} {y:+.2f} {z:+.2f}  yaw {yaw:+5.1f}  reproj {g.reprojection_error_px:4.1f}px  max snap {shift.max():4.1f}px")
json.dump({"intrinsics": "approx fx=504.9 cx=512 cy=384 no distortion", "gate_side_m": SIDE,
           "corner_order": "TL,TR,BR,BL pipe centreline", "corners": labels}, open(f"{OUT}/labels.json", "w"), indent=1)
