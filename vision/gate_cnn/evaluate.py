"""Compare the CNN against the colour-threshold detector on identical test frames.

    python evaluate.py         -> results.json, results.png, examples.png

For each condition, 300 fresh frames (never seen in training, different random seed)
with the whole nearest gate in view. Both detectors see the same frames.
Metrics:
  corner error   mean pixel distance between detected and true corners (256x160 image)
  position error PnP gate-centre error as a % of the distance to the gate
  success        position error under 10% of distance (e.g. within 40 cm at 4 m)
"""
import json
import time

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import render as R
import baseline
from detector import GateCNN, estimate_pose, refine_with_colour

N_PER = 300
K = C.intrinsics()


def test_set(cond, seed):
    rng = np.random.default_rng(seed)
    spec = R.EVAL_CONDITIONS[cond]
    out = []
    while len(out) < N_PER:
        sc = dict(distractors=(0, 0), all_corners_in=True, floor_p=0.0)
        sc.update(spec.get("scene", {}))
        img, cps, gate = R.make_example(rng, aug=spec.get("aug"), **sc)
        if not ((cps[:, 0] > 1) & (cps[:, 0] < C.IMG_W - 1) & (cps[:, 1] > 1) & (cps[:, 1] < C.IMG_H - 1)).all():
            continue
        out.append((img, cps, gate[1]))
    return out


def score(corners, cps, t_true):
    if corners is None:
        return np.nan, np.nan
    cerr = np.linalg.norm(corners - cps, axis=1).mean()
    ok, t, _ = estimate_pose(corners, K)
    perr = np.linalg.norm(t - t_true) / np.linalg.norm(t_true) * 100 if ok else np.nan
    return cerr, perr


def main():
    R.EXACT_GATE = True   # test on gates with the real dimensions (1.9 m outside, 0.2 m frame)
    cnn = GateCNN("gate_cnn.npz")
    results, examples = {}, {}
    times = {"cnn": [], "colour": []}
    for ci, cond in enumerate(R.EVAL_CONDITIONS):
        data = test_set(cond, seed=1000 + ci)
        imgs = [d[0] for d in data]
        t0 = time.time()
        cnn_c, cnn_p = cnn.detect_batch(imgs)
        times["cnn"].append((time.time() - t0) / len(imgs))
        rows = {"cnn": [], "cnn+refine": [], "colour": []}
        for i, (img, cps, t_true) in enumerate(data):
            rows["cnn"].append(score(cnn_c[i], cps, t_true))
            rows["cnn+refine"].append(score(refine_with_colour(img, cnn_c[i]), cps, t_true))
            t0 = time.time()
            bc, _ = baseline.detect(img)
            times["colour"].append(time.time() - t0)
            rows["colour"].append(score(bc, cps, t_true))
            if i == 3:
                examples[cond] = (img, cps, cnn_c[i], bc)
        results[cond] = {}
        for m, r in rows.items():
            r = np.array(r, float)
            perr = np.where(np.isnan(r[:, 1]), 999, r[:, 1])
            results[cond][m] = dict(
                corner_px=float(np.nanmean(np.minimum(r[:, 0], 50))),   # cap so one wild miss doesn't dominate
                pos_err_median_pct=float(np.median(perr)),
                success_pct=float((perr < 10).mean() * 100),
                no_detection_pct=float(np.isnan(r[:, 0]).mean() * 100))
        print(f"{cond:13s} " + " | ".join(f"{m}: {results[cond][m]['success_pct']:5.1f}% ok "
              f"{results[cond][m]['corner_px']:.2f}px" for m in rows))
    results["_timing_ms_per_frame"] = {k: float(np.mean(v) * 1000) for k, v in times.items()}
    json.dump(results, open("results.json", "w"), indent=2)
    print("timing ms/frame", results["_timing_ms_per_frame"])
    plot(results)
    plot_examples(examples)


INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
COL = {"cnn": "#c4b5fd", "cnn+refine": "#7c3aed", "colour": "#9ca3af"}


def plot(res):
    conds = [c for c in res if not c.startswith("_")]
    x = np.arange(len(conds))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), dpi=150)
    for ax, key, title, unit in [
        (axes[0], "success_pct", "Good pose estimates (error < 10% of distance)", "% of frames"),
        (axes[1], "corner_px", "Average corner error", "pixels (lower is better)")]:
        for j, (m, lab) in enumerate([("colour", "Color threshold (current)"), ("cnn", "CNN alone"),
                                      ("cnn+refine", "CNN + color refine (recommended)")]):
            vals = [res[c][m][key] for c in conds]
            bars = ax.bar(x + (j - 1) * 0.27, vals, 0.25, color=COL[m], label=lab, zorder=3)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}" if key == "success_pct" else f"{v:.1f}",
                        ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.set_xticks(x, conds, fontsize=8.5, color=INK)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.set_ylabel(unit, fontsize=8.5, color=MUTED)
        ax.grid(axis="y", color=GRID, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].set_ylim(0, 110)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8.5, ncol=3, loc="upper right", bbox_to_anchor=(0.995, 0.995))
    fig.suptitle("Gate detection: CNN vs color threshold, 300 test frames per condition",
                 fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig("results.png", facecolor="white")


def plot_examples(ex):
    tiles = []
    for cond, (img, cps, cc, bc) in ex.items():
        v = cv2.resize(img, (C.IMG_W * 2, C.IMG_H * 2), interpolation=cv2.INTER_NEAREST)
        if bc is not None:
            cv2.polylines(v, [np.round(bc * 2).astype(np.int32)], True, (60, 60, 255), 1, cv2.LINE_AA)
        cv2.polylines(v, [np.round(cc * 2).astype(np.int32)], True, (80, 255, 80), 1, cv2.LINE_AA)
        for p in cps * 2:
            cv2.circle(v, tuple(np.round(p).astype(int)), 4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(v, cond, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(v)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    sheet = np.vstack(rows)
    leg = np.zeros((28, sheet.shape[1], 3), np.uint8)
    cv2.putText(leg, "white circles = true corners   green = CNN   red = color threshold",
                (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.imwrite("examples.png", np.vstack([sheet, leg]))


if __name__ == "__main__":
    main()
