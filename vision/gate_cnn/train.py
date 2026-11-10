"""Train the gate corner CNN.

    python train.py                       # synthetic data only (what was used for the results)
    python train.py --real isaac_frames/  # also mix in labelled Isaac captures (see label_isaac.py)

Writes gate_cnn.npz (the weights) and train_log.csv.
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np

import config as C
import render as R
from nn_numpy import GateNet, Adam, heatmap_loss

HH, HW = C.NET_H // C.NET_STRIDE, C.NET_W // C.NET_STRIDE   # 20 x 32
SIGMA = 0.8                                                  # target blob size, in heatmap cells
FLIP = [1, 0, 3, 2]  # horizontal flip swaps TL<->TR and BR<->BL


def img_to_cell(xy_img):
    """Full-image pixel -> heatmap cell coordinate (continuous)."""
    x_net = (xy_img - 0.5) / (C.IMG_W / C.NET_W)
    return (x_net - (C.NET_STRIDE - 1) / 2) / C.NET_STRIDE


def cell_to_img(xy_cell):
    x_net = xy_cell * C.NET_STRIDE + (C.NET_STRIDE - 1) / 2
    return x_net * (C.IMG_W / C.NET_W) + 0.5


def make_target(corners_img):
    """Gaussian target heatmaps (HH, HW, 4) and visibility mask (4,)."""
    ys, xs = np.mgrid[0:HH, 0:HW].astype(np.float32)
    T = np.zeros((HH, HW, 4), np.float32)
    m = np.zeros(4, np.float32)
    for k, (x, y) in enumerate(corners_img):
        if not (0 <= x < C.IMG_W and 0 <= y < C.IMG_H):
            continue
        cx, cy = img_to_cell(np.array([x, y]))
        g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * SIGMA ** 2))
        T[..., k] = g / g.sum()
        m[k] = 1
    return T, m


def to_net(img):
    return cv2.resize(img, (C.NET_W, C.NET_H), interpolation=cv2.INTER_AREA)


def build_synthetic(n, seed):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, C.NET_H, C.NET_W, 3), np.uint8)
    Y = np.zeros((n, 4, 2), np.float32)
    for i in range(n):
        img, cps, _ = R.make_example(rng, aug=R.TRAIN_AUG, near=rng.random() < 0.15)
        X[i], Y[i] = to_net(img), cps
    return X, Y


def load_real(folder):
    """Frames saved by label_isaac.py: <name>.png + <name>.txt with 8 numbers (TL TR BR BL x,y)."""
    X, Y = [], []
    for txt in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        img = cv2.imread(txt[:-4] + ".png")
        if img is None:
            continue
        sx, sy = C.IMG_W / img.shape[1], C.IMG_H / img.shape[0]
        cps = np.loadtxt(txt).reshape(4, 2) * [sx, sy]
        X.append(to_net(cv2.resize(img, (C.IMG_W, C.IMG_H), interpolation=cv2.INTER_AREA)))
        Y.append(cps)
    return np.array(X, np.uint8), np.array(Y, np.float32)


def batch(X, Y, idx, rng, real_aug=False):
    xb = X[idx].copy()
    yb = Y[idx].copy()
    T = np.zeros((len(idx), HH, HW, 4), np.float32)
    M = np.zeros((len(idx), 4), np.float32)
    for j in range(len(idx)):
        if rng.random() < 0.5:  # mirror image: corner identities swap
            xb[j] = xb[j][:, ::-1]
            yb[j] = yb[j][FLIP]
            yb[j][:, 0] = C.IMG_W - 1 - yb[j][:, 0]
        T[j], M[j] = make_target(yb[j])
    return xb, T, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--real", default=None, help="folder of labelled Isaac frames")
    ap.add_argument("--init", default=None, help="start from existing weights (fine-tuning)")
    ap.add_argument("--out", default="gate_cnn.npz")
    a = ap.parse_args()

    t0 = time.time()
    X, Y = build_synthetic(a.n, seed=1)
    if a.real:
        Xr, Yr = load_real(a.real)
        print(f"real frames: {len(Xr)}")
        if len(Xr):  # oversample real frames so they make up ~1/3 of each batch
            reps = max(1, len(X) // (2 * len(Xr)))
            X = np.concatenate([X] + [Xr] * reps)
            Y = np.concatenate([Y] + [Yr] * reps)
    print(f"dataset {len(X)} frames built in {time.time() - t0:.0f}s")

    net = GateNet.load(a.init) if a.init else GateNet(seed=0)
    opt = Adam(net, lr=a.lr)
    rng = np.random.default_rng(2)
    log = open("train_log.csv", "w")
    log.write("step,loss,lr,seconds\n")
    run = None
    for step in range(1, a.steps + 1):
        opt.lr = a.lr * 0.5 * (1 + np.cos(np.pi * step / a.steps)) + 1e-5   # cosine decay
        idx = rng.integers(0, len(X), a.bs)
        xb, T, M = batch(X, Y, idx, rng)
        logits = net.forward(net.preprocess(xb), train=True)
        loss, g, _ = heatmap_loss(logits, T, M)
        net.backward(g)
        opt.step()
        run = loss if run is None else 0.98 * run + 0.02 * loss
        if step % 50 == 0:
            log.write(f"{step},{run:.4f},{opt.lr:.5f},{time.time() - t0:.0f}\n")
            log.flush()
        if step % 250 == 0:
            print(f"step {step:5d}  loss {run:.3f}  ({time.time() - t0:.0f}s)", flush=True)
            net.save(a.out)
    net.save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
