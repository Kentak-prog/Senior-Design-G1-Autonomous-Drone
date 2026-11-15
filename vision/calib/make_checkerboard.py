"""Printable checkerboard for calibrating hires_front (see vision/calib/README or chat notes).

US Letter landscape, 10 x 7 squares of 25 mm = 9 x 6 INNER corners (what OpenCV's
findChessboardCorners / voxl-camera-calibration ask for). Odd x even so the board's
orientation is unambiguous. A 100 mm scale bar lets you check the print came out at 100%.

    python3 make_checkerboard.py            # writes checkerboard_9x6_25mm_letter.pdf
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SQUARE_MM = 25.0
COLS, ROWS = 10, 7                 # squares; inner corners = 9 x 6
PAGE_W, PAGE_H = 279.4, 215.9      # US Letter landscape, mm

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   f"checkerboard_{COLS - 1}x{ROWS - 1}_{SQUARE_MM:g}mm_letter.pdf")

fig = plt.figure(figsize=(PAGE_W / 25.4, PAGE_H / 25.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, PAGE_W)
ax.set_ylim(0, PAGE_H)
ax.set_aspect("equal")
ax.axis("off")

bw, bh = COLS * SQUARE_MM, ROWS * SQUARE_MM
x0 = (PAGE_W - bw) / 2
y0 = (PAGE_H - bh) / 2 + 6          # nudge up to leave room for the label below
for r in range(ROWS):
    for c in range(COLS):
        if (r + c) % 2 == 0:
            ax.add_patch(Rectangle((x0 + c * SQUARE_MM, y0 + r * SQUARE_MM),
                                   SQUARE_MM, SQUARE_MM, color="black", lw=0))

# 100 mm scale bar + label in the bottom margin, clear of the board's white border
ty = y0 - 9
ax.add_patch(Rectangle((x0, ty), 100, 1.2, color="black", lw=0))
for t in (0, 50, 100):
    ax.add_patch(Rectangle((x0 + t - 0.3, ty - 1.5), 0.6, 4.2, color="black", lw=0))
ax.text(x0 + 104, ty + 0.6, "100 mm: measure this; squares must be exactly 25.0 mm",
        va="center", fontsize=7)
ax.text(x0, ty - 6, f"{COLS}x{ROWS} squares, {SQUARE_MM:g} mm, {COLS - 1}x{ROWS - 1} inner corners. "
        "Print at 100% / Actual size (no 'fit to page'). Mount flat on rigid board.",
        fontsize=7)

fig.savefig(out)
print(out)
