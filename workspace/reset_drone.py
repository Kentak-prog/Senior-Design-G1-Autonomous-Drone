"""
Put the spawned Pegasus drone back upright at the takeoff spot, at rest,
without restarting Isaac (e.g. after a crash). Run in Isaac's Script Editor
with PX4 disarmed:

    exec(open("/workspace/Senior-Design-G1-Autonomous-Drone/workspace/reset_drone.py").read())

Then restart PX4 (`shutdown` at pxh>, remove /tmp/px4-sock-0, start it again):
its estimator does not follow the jump and keeps the old attitude.
Used on 2026-11-07 after the Starling flipped (see
Trajectory/isaac_results/circle_course/starling_c29_2026-11-07/README.md).
"""
from pegasus.simulator.logic.vehicle_manager import VehicleManager

RESET_POS = (0.0, 0.0, 0.25)          # m, world: just above the Flat Grid at the takeoff spot
RESET_QUAT = (0.0, 0.0, 0.0, 1.0)     # (x, y, z, w): level, facing world +x, as spawned

_veh = list(VehicleManager.get_vehicle_manager().vehicles.values())[0]
_dc = _veh.get_dc_interface()
_root = _dc.get_articulation_root_body(_dc.get_articulation(_veh._stage_prefix))
_tf = _dc.get_rigid_body_pose(_root)
print(f"[reset_drone] was at p={tuple(round(v, 3) for v in _tf.p)} r={tuple(round(v, 3) for v in _tf.r)}")
_tf.p = RESET_POS
_tf.r = RESET_QUAT
_dc.set_rigid_body_pose(_root, _tf)
_dc.set_rigid_body_linear_velocity(_root, (0.0, 0.0, 0.0))
_dc.set_rigid_body_angular_velocity(_root, (0.0, 0.0, 0.0))
_tf = _dc.get_rigid_body_pose(_root)
print(f"[reset_drone] now at p={tuple(round(v, 3) for v in _tf.p)} r={tuple(round(v, 3) for v in _tf.r)}; restart PX4 next")
