"""Fly the drone back to a point with the carrot goto (no geofence), then AUTO.LOITER.
python recover.py X Y Z  (ENU, origin offset 0 0 0.05 as in the 2026-11-12 runs)."""
import sys
import runner  # sets sys.path for rl/
from runner import MavlinkVehicle, ThrustMap, QuadParams
x, y, z = map(float, sys.argv[1:4])
p = QuadParams()
v = MavlinkVehicle("udpin:0.0.0.0:14540", (0.0, 0.0, 0.05), ThrustMap("model", p, scale=1.0))
try:
    s = runner.goto(v, p, [x, y, z], 0.0, speed=1.5, timeout=120.0)
    print("recovered at", s.pos.round(2))
finally:
    v.set_loiter()
    v.close()
