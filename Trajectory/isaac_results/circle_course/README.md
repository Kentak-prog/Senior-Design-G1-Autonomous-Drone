# Circle course

Three gates 120° apart on a 5 m circle, each facing along the flight path,
centers at camera height (z 2.6 m). gate_0 is 5 m straight ahead of the usual
takeoff hover, so the spawn pose and `vio_node`'s spawn hint match the
3-gate course (`../3_gates/course/`). Built so the front camera keeps a whole
gate in view for the color detector + PnP.

| File | |
|---|---|
| `circle_course.json` | the manifest (same format as `test_gates.json`), written by `Trajectory/make_circle_course.py` |
| `camera_check_circle.png` | `Trajectory/check_course_camera.py`, Iris camera: 3.5 s longest gap, 0 bad fixes |
| `camera_check_circle_c29.png` | the same with the Starling C29 front camera: 1.2 s longest gap, fixes on 89% of frames |

| Gate | x | y | z | yaw |
|---|---|---|---|---|
| gate_0 | 4.911 | −0.667 | 2.6 | −7.2° |
| gate_1 | 8.267 | −8.650 | 2.6 | −127.2° |
| gate_2 | −0.325 | −7.565 | 2.6 | +112.8° |

**Spawn it** in the Isaac Script Editor (it is copied to
`/workspace/test_gates.json`, which every node reads):
```python
import os; os.environ["GATES_MANIFEST"] = "/workspace/Senior-Design-G1-Autonomous-Drone/Trajectory/isaac_results/circle_course/circle_course.json"; exec(open("/workspace/Senior-Design-G1-Autonomous-Drone/workspace/spawn_gates_from_manifest.py").read())
```
Without `GATES_MANIFEST`, the spawner loads the 3-gate course.
