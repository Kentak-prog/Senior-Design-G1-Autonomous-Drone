# Live gate-error runs (Isaac Sim, 2026-10-01)

Each folder has `gate_errors.csv` (from `eval_gate_bag.py`), `summary_table.csv`
and fig1-fig4 (from `plot_gate_errors.py`). The bags are on the cluster at
`/workspace/bags/<name>` (~1.2-1.9 GB each, not in git). Camera 1280x720 / 60 deg
hFOV. All runs use bag receive time for image-pose pairing: camera stamps are
sim time and pose stamps are wall time.

| Folder | What | Gates | Mount used | Trust |
|---|---|---|---|---|
| `hover_run1` | 15 s hover, 474 frames | staggered (5) | default (0.30, 0, 0.05) | yes: mount was intact |
| `move_test1m` | 1 m at 0.5 m/s | staggered (5) | default | yes; gates above the drone leave the frame top when it moves |
| `move_run3m` | 3 m at 1 m/s | staggered (5) | default | yes, but almost no moving frames: every gate left the frame within ~1.5 s |
| `approach_run5m` | 5 m at 1 m/s | approach (4) | default | **NO**: the camera prim had moved to (-0.040, 0, -0.041) m from the body, so every gate reads a false ~0.35 m extra range error. Kept as the evidence for that |
| `approach_run5m_measured_mount` | same bag as above | approach (4) | `--mount-xyz -0.0398 0 -0.0411` | yes: **the moving-drone result** |

Headline (`approach_run5m_measured_mount`): hover 0.13-0.21 m (0.9-2.2% of
range), moving above 0.5 m/s 0.17-0.25 m (1.6-3.1%). The extra error peaks
during acceleration. See vision/HANDOFF.md, "Gate-error graphs".

To regenerate figures from a CSV: `python3 ../plot_gate_errors.py <run>/gate_errors.csv --title-suffix "(...)"`.
