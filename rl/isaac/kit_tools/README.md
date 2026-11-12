# kit_tools: driving Isaac Sim on the cluster without a person at the GUI

Copies of the pod helpers, kept in git because pod resets keep wiping `/workspace/race`.

| file | where it runs | what it does |
|---|---|---|
| `xtype.py` | `isaac-sim` container, kit python, `PYTHONPATH=/workspace/pylib` (`pip install --target /workspace/pylib python-xlib`) | X input on display `:1`: `click X Y`, `type TEXT`, `key ctrl+Return`, `sleep S`, `shot OUT.ppm [x0 y0 x1 y1]`, chained with `;;` |
| `r.py` | Isaac Script Editor, once per Kit process: `exec(open('/workspace/race/r.py').read())` | update-loop poller: execs `/workspace/race/isaac_next.py` when it appears; log in `isaac_next.log` |
| `q.sh` | Mac | queue a local script for the poller (atomic write + `mv`) |
| `xs.sh` | Mac | run xtype ops, then screenshot the Kit window to `$SCRATCH/s.png` |
| `run_pink.sh` | `ros2-eval` | the 2026-11-12 pink 36 in gate matrix (PID + MPC, 3 repeats, 4 tracks) |

Opening the Script Editor by X input: Window menu (129,11), Script Editor (170,363), click the editor (700,500),
type the line, click Run (382,735). Key and button presses must be held ~80 ms or Kit drops them.
