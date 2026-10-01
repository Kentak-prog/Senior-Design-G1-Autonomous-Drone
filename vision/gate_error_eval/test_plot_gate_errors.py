"""Self-tests. Run: python3 test_plot_gate_errors.py

matplotlib + numpy only. Builds a synthetic rows list (5 gates, 40 frames, a
hover then a moving segment, one gate never detected), runs every plot
function into a temp dir, and checks the edge cases (single frame, no
detections at all). Also exposes make_synthetic_rows() / write_example_output()
so the example figures in example_output/ can be regenerated:

    python3 test_plot_gate_errors.py --write-example
"""

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import error_records as er
import plot_gate_errors as pg

GATES = [("gate_0", 6.0, 0.0), ("gate_1", 8.0, 1.5), ("gate_2", 10.0, -1.0),
         ("gate_3", 12.0, 2.5), ("gate_4", 14.0, -2.0)]  # (name, x, y); gate_4 never detected


def make_synthetic_rows(n_frames=40, hover_frames=15, seed=0):
    """Drone hovers at the origin for `hover_frames`, then flies +x at 1 m/s
    (frame rate 10 Hz). Error grows with range and is mostly depth, as in the
    real runs; gate_4 is never detected."""
    rng = np.random.default_rng(seed)
    rows = []
    x = 0.0
    for k in range(n_frames):
        t = k * 0.1
        speed = 0.0 if k < hover_frames else 1.0
        if k >= hover_frames:
            x += 0.1
        for name, gx, gy in GATES:
            range_true = gx - x
            in_view = range_true > 1.0
            detected = in_view and name != "gate_4" and rng.random() > 0.05 * range_true / 6.0
            row = {"t_s": t, "frame": k, "gate": name, "detected": bool(detected),
                   "est_x": None, "est_y": None, "est_z": None,
                   "true_x": gx, "true_y": gy, "true_z": 2.5,
                   "drone_x": x, "drone_y": 0.0, "drone_z": 2.5,
                   "drone_speed_mps": speed, "range_true_m": range_true,
                   "off_axis_deg": float(np.degrees(np.arctan2(abs(gy), max(range_true, 1e-3)))),
                   "in_view": bool(in_view),
                   "pos_err_m": None, "range_err_m": None, "lateral_err_m": None,
                   "normal_err_deg": None, "reproj_rms_px": None, "normal_is_reliable": None}
            if detected:
                re_ = rng.normal(0.0, 0.012 * range_true) + 0.004 * range_true
                le = abs(rng.normal(0.0, 0.004 * range_true))
                row.update({"est_x": gx + re_, "est_y": gy + le * rng.choice([-1, 1]),
                            "est_z": 2.5 + rng.normal(0, 0.01),
                            "pos_err_m": float(np.hypot(re_, le)), "range_err_m": float(re_),
                            "lateral_err_m": float(le),
                            "normal_err_deg": float(abs(rng.normal(0, 3))),
                            "reproj_rms_px": float(abs(rng.normal(0, 0.3))),
                            "normal_is_reliable": bool(range_true < 9.0)})
            rows.append(row)
    return rows


def _nonempty(path):
    assert os.path.isfile(path) and os.path.getsize(path) > 1000, path


def test_all_figures():
    rows = make_synthetic_rows()
    with tempfile.TemporaryDirectory() as d:
        paths = pg.plot_all(rows, d, "(synthetic data)", "png", dpi=60)
        assert [os.path.basename(p) for p in paths] == [
            "fig1_error_vs_range.png", "fig2_error_vs_time.png",
            "fig3_topdown.png", "fig4_detection_rate.png"]
        for p in paths:
            _nonempty(p)
        _nonempty(os.path.join(d, "summary_table.csv"))
        pdf = pg.plot_error_vs_range(rows, os.path.join(d, "f.pdf"), dpi=60)
        _nonempty(pdf)
    print("  four figures + summary_table.csv written, non-empty (png and pdf)  OK")


def test_gate_colors_stable():
    rows = make_synthetic_rows()
    c1 = pg.gate_colors(rows)
    c2 = pg.gate_colors(list(reversed(rows)))
    assert c1 == c2 and len(set(c1.values())) == 5
    print("  gate colors keyed by sorted name, stable across row order  OK")


def test_single_frame_and_empty():
    one = [r for r in make_synthetic_rows() if r["frame"] == 0]
    with tempfile.TemporaryDirectory() as d:
        for p in pg.plot_all(one, d, "single", "png", dpi=50):
            _nonempty(p)
        # CSV round trip of a single-frame run, as the CLI would see it.
        csv_path = os.path.join(d, "one.csv")
        er.write_csv(csv_path, one)
        for p in pg.plot_all(er.read_csv(csv_path), os.path.join(d, "o2"), "", "png", dpi=50):
            _nonempty(p)
        # Nothing detected, speed unknown (None), NaN-ish values.
        none = []
        for r in make_synthetic_rows(n_frames=5):
            r = dict(r, detected=False, est_x=None, pos_err_m=None, range_err_m=None,
                     lateral_err_m=None, drone_speed_mps=None)
            none.append(r)
        for p in pg.plot_all(none, os.path.join(d, "o3"), "", "png", dpi=50):
            _nonempty(p)
        nan_rows = make_synthetic_rows(n_frames=5)
        nan_rows[0]["pos_err_m"] = float("nan")
        nan_rows[0]["detected"] = True
        for p in pg.plot_all(nan_rows, os.path.join(d, "o4"), "", "png", dpi=50):
            _nonempty(p)
    print("  single frame, zero detections, None speed, NaN error: no crash  OK")


def write_example_output():
    out = os.path.join(HERE, "example_output")
    rows = make_synthetic_rows()
    csv_path_dir = out
    os.makedirs(csv_path_dir, exist_ok=True)
    er.write_csv(os.path.join(out, "synthetic_gate_errors.csv"), rows)
    # Go through the CLI path (CSV -> figures) like a real run would.
    import subprocess
    subprocess.check_call([sys.executable, os.path.join(HERE, "plot_gate_errors.py"),
                           os.path.join(out, "synthetic_gate_errors.csv"), "--out-dir", out,
                           "--title-suffix", "(synthetic data)", "--dpi", "150"])


if __name__ == "__main__":
    if "--write-example" in sys.argv:
        write_example_output()
        sys.exit(0)
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("\nAll tests passed.")
