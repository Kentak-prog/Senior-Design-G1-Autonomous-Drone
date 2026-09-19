import asyncio
import omni.usd
from pxr import UsdGeom, Gf
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend, PX4MavlinkBackendConfig,
)
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.graphs.ros2_camera_graph import ROS2CameraGraph
from pegasus.simulator.params import ROBOTS

DRONE_PATH  = "/World/Iris"
CAMERA_NAME = "front_cam"

def ensure_scene(world):
    if hasattr(world, '_scene') and world._scene is not None:
        return True
    try:
        from isaacsim.core.api.scenes.scene import Scene
        world._scene = Scene()
        return True
    except Exception:
        pass
    try:
        from omni.isaac.core.scenes.scene import Scene
        world._scene = Scene()
        return True
    except Exception:
        pass
    return False

async def main():
    pi = PegasusInterface()
    if pi.world is None:
        pi.initialize_world()
    await pi.world.initialize_simulation_context_async()

    # Clear registries (safe for re-runs)
    if pi.vehicle_manager.vehicles:
        pi.vehicle_manager.remove_all_vehicles()
    if hasattr(pi.world, '_scene') and pi.world._scene is not None:
        try:
            pi.world.scene.remove_object(DRONE_PATH, registry_only=True)
        except Exception:
            pass
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(DRONE_PATH).IsValid():
        stage.RemovePrim(DRONE_PATH)

    # Load your USD world as sublayer (uncomment and set path):
    # root_layer = stage.GetRootLayer()
    # WORLD_USD = "/workspace/assets/your_world.usd"
    # if WORLD_USD not in root_layer.subLayerPaths:
    #     root_layer.subLayerPaths.append(WORLD_USD)

    if not ensure_scene(pi.world):
        print("FATAL: could not initialize scene")
        return

    # Camera
    cam_graph = ROS2CameraGraph(
        camera_prim_path=f"body/{CAMERA_NAME}",
        config={
            "resolution":  [320, 240],
            "types":       ["rgb", "depth", "camera_info"],
            "namespace":   "/iris_0",
            "topic":       f"/{CAMERA_NAME}",
            "tf_frame_id": CAMERA_NAME,
        },
    )

    # PX4 backend — Pegasus listens on TCP 4560, PX4 connects to it
    px4_config = PX4MavlinkBackendConfig(config={
        "vehicle_id":          0,
        "px4_autolaunch":      False,
        "connection_type":     "tcpin",       # Pegasus = TCP server
        "connection_ip":       "localhost",
        "connection_baseport": 4560,
        "enable_lockstep":     False,         # Must be False for manual PX4 launch
    })

    config = MultirotorConfig()
    config.backends = [
        PX4MavlinkBackend(px4_config),
        ROS2Backend(vehicle_id=0, num_rotors=4),
    ]
    config.graphical_sensors = []
    config.graphs            = [cam_graph]

    # Spawn drone
    Multirotor(
        stage_prefix=DRONE_PATH,
        usd_file=ROBOTS["Iris"],
        init_pos=[0.0, 0.0, 2.5],
        init_orientation=[0.0, 0.0, 0.0, 1.0],   # quaternion [x,y,z,w], NOT euler
        config=config,
    )

    await pi.world.reset_async()

    # Fix camera transform (known bug — ROS2CameraGraph bakes spawn height)
    cam_prim = stage.GetPrimAtPath(f"{DRONE_PATH}/body/{CAMERA_NAME}")
    if cam_prim.IsValid():
        pitch_deg = 15.0
        xf = UsdGeom.Xformable(cam_prim)
        xf.ClearXformOpOrder()
        cam_prim.RemoveProperty("xformOp:translate")
        cam_prim.RemoveProperty("xformOp:orient")
        cam_prim.RemoveProperty("xformOp:rotateXYZ")
        cam_prim.RemoveProperty("xformOp:scale")
        xf.AddTranslateOp().Set(Gf.Vec3d(0.30, 0.0, 0.05))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(90.0 + pitch_deg, 0.0, -90.0))
        xf.AddScaleOp().Set(Gf.Vec3d(1.0, 1.0, 1.0))

    print("[init] Done — start PX4 SITL now")

asyncio.ensure_future(main())