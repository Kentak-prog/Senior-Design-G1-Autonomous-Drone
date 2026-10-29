"""
Put every ROS 2 topic from Isaac Sim on ONE clock: simulation time.

Run ONCE per Isaac Sim process, in the Script Editor, AFTER spawn_example.py:
    exec(open("/workspace/scripts/sim_clock.py").read())

Why: Isaac's camera helper stamps images with SIMULATION time
(ROS2CameraHelper inputs:useSystemTime defaults to False), but Pegasus'
ROS2Backend stamps /drone00/sensors/imu and /drone00/state/* with
node.get_clock().now() on a node created without use_sim_time -- WALL time.
With the sim at ~0.4x real time the two drift apart by more than a second per
second, so a VIO cannot tell which IMU samples belong to which image. This:

  1. adds an OmniGraph that publishes /clock from the simulation time
     (OnPlaybackTick -> ReadSimulationTime -> ROS2PublishClock), and
  2. switches Pegasus' ROS 2 node(s) to use_sim_time, so their stamps come
     from /clock (the node is spun every physics step, so it receives it).

Then run every ROS node with use_sim_time:=true. Check with (ros2-eval):
    ros2 topic echo --once /clock
    ros2 topic echo --once /drone00/sensors/imu --field header.stamp
    ros2 topic echo --once /iris_0/front_cam/camera_info --field header.stamp
The three stamps should be within a few tenths of a second of each other.
"""

import omni.graph.core as og

GRAPH_PATH = "/World/SimClockGraph"

# Node type names moved from omni.isaac.* to isaacsim.* in Isaac Sim 4.5; try both.
_TYPES = {
    "tick": ["omni.graph.action.OnPlaybackTick"],
    "sim_time": ["isaacsim.core.nodes.IsaacReadSimulationTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"],
    "pub_clock": ["isaacsim.ros2.bridge.ROS2PublishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"],
}


def _build_clock_graph():
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(GRAPH_PATH).IsValid():
        print(f"[sim_clock] {GRAPH_PATH} already exists; leaving it")
        return True
    keys = og.Controller.Keys
    last_err = None
    for i in range(2):
        try:
            names = {k: v[min(i, len(v) - 1)] for k, v in _TYPES.items()}
            og.Controller.edit(
                {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: [(k, t) for k, t in names.items()],
                    keys.CONNECT: [
                        ("tick.outputs:tick", "pub_clock.inputs:execIn"),
                        ("sim_time.outputs:simulationTime", "pub_clock.inputs:timeStamp"),
                    ],
                    keys.SET_VALUES: [("pub_clock.inputs:topicName", "/clock")],
                },
            )
            print(f"[sim_clock] /clock publisher created ({names['pub_clock']})")
            return True
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if stage.GetPrimAtPath(GRAPH_PATH).IsValid():
                stage.RemovePrim(GRAPH_PATH)
    print(f"[sim_clock] FAILED to create the /clock graph: {last_err}")
    return False


def _pegasus_nodes_to_sim_time():
    from rclpy.parameter import Parameter
    from pegasus.simulator.logic.vehicle_manager import VehicleManager
    switched = 0
    for name, vehicle in VehicleManager.get_vehicle_manager().vehicles.items():
        for backend in getattr(vehicle, "_backends", []):
            node = getattr(backend, "node", None)
            if node is None:
                continue
            node.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
            switched += 1
            print(f"[sim_clock] {name}: ROS 2 node '{node.get_name()}' now uses sim time")
    if switched == 0:
        print("[sim_clock] WARNING: no Pegasus ROS 2 backend node found -- run spawn_example.py first")
    return switched


if _build_clock_graph():
    _pegasus_nodes_to_sim_time()
    print("[sim_clock] done. Press Play if it is not playing; /clock only ticks while the timeline plays.")
