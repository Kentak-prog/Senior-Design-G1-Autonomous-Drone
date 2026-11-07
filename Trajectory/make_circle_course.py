"""
Write a 3-gate CIRCLE course manifest (test_gates.json format) for Isaac.

Why a circle: the front camera is 60 x 36 deg, and the color detector + PnP
needs a whole gate in frame. The 2026-10-23 course crosses itself and needs a
U-turn, so after gate_0 and on the way back to gate_0 no gate fits in the frame
for 4-5 s at a time (px4/NEXT_TESTS.md, V2). On a circle, with each gate facing
along the flight direction and the nose on the next gate (yaw_mode next_gate),
the next gate is in view, at most 60 deg side-on, from the moment the drone
passes a gate until it is about 3 m from the next one. A 1.7 m gate does not fit
in the 36 deg vertical view any closer than that, on any layout, so the last
~2 s before each gate stays blind. Check a layout with check_course_camera.py.

    python Trajectory/make_circle_course.py                     # the course in isaac_results/circle_course
    python Trajectory/make_circle_course.py --radius 6 --out my_course.json

Gate centers are at the camera's hover height, so the camera looks straight at
them. camera_pose_usd is copied from the 2026-10-23 manifest: same takeoff
spot, so vio_node's spawn hint and the rehearsals' start pose stay valid.
"""

import argparse
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "isaac_results", "3_gates", "course", "3_gate_course_pod.json")
OUT = os.path.join(HERE, "isaac_results", "circle_course", "circle_course.json")
START_XY = (-0.05, -0.04)          # takeoff hover (Trajectory/test_replan_session.py START)
START_YAW_DEG = -7.2


def circle_gates(radius, approach, z, turn="right"):
    psi0 = math.radians(START_YAW_DEG)
    s = -1.0 if turn == "right" else 1.0                    # right = clockwise seen from above
    g0 = (START_XY[0] + approach * math.cos(psi0), START_XY[1] + approach * math.sin(psi0))
    c = (g0[0] - s * radius * math.sin(psi0), g0[1] + s * radius * math.cos(psi0))
    gates = []
    for k in range(3):
        a = s * math.radians(120.0 * k)
        v = (g0[0] - c[0], g0[1] - c[1])
        p = (c[0] + v[0] * math.cos(a) - v[1] * math.sin(a), c[1] + v[0] * math.sin(a) + v[1] * math.cos(a))
        yaw = (math.degrees(psi0 + a) + 180.0) % 360.0 - 180.0
        gates.append(dict(name=f"gate_{k}", x=round(p[0], 3), y=round(p[1], 3), z=z, yaw_deg=round(yaw, 2), side=1.5))
    return gates


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radius", type=float, default=5.0)
    ap.add_argument("--approach", type=float, default=5.0, help="gate_0 distance straight ahead of the takeoff hover, m")
    ap.add_argument("--z", type=float, default=2.6, help="gate center height (camera height at the 2.55 m hover)")
    ap.add_argument("--turn", choices=["right", "left"], default="right")
    ap.add_argument("--base", default=BASE, help="manifest to copy camera_pose_usd from")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    base = json.load(open(a.base))
    doc = {"yaw_convention": base["yaw_convention"],
           "course": f"3-gate circle (Trajectory/make_circle_course.py): radius {a.radius} m, gate_0 {a.approach} m "
                     f"straight ahead of the takeoff hover, turning {a.turn}, gate centers at z {a.z} m",
           "gates": circle_gates(a.radius, a.approach, a.z, a.turn),
           "camera_pose_usd": base["camera_pose_usd"], "dropped": []}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=2)
    for g in doc["gates"]:
        print(f"{g['name']}: ({g['x']:+.3f}, {g['y']:+.3f}, {g['z']:.2f})  yaw {g['yaw_deg']:+.1f} deg")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
