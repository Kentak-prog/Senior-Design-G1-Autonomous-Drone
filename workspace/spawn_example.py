import asyncio
import math
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

# Camera image size and horizontal field of view. KEEP RESOLUTION IN SYNC with
# spawn_test_gates.RESOLUTION (it derives the vertical FOV from this aspect
# ratio). Was 320x240 at ~24 deg hFOV (fx=763): too narrow to hold a racing
# course, and 320 px wide limits depth accuracy. fx = (width/2)/tan(hfov/2).
RESOLUTION = (1280, 720)
HFOV_DEG   = 60.0

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

async def refresh_camera_info_helper():
    """
    Make the published camera_info match the camera as configured below.

    ROS2CameraGraph publishes camera_info through an OmniGraph helper node
    that calls read_camera_info() ONCE, on its first compute, and bakes
    width/height/k/p into its writer (see OgnROS2CameraHelper.py in
    isaacsim.ros2.bridge). That first compute happens before the focalLength
    and resolution set in main() reach the camera, so without this the topic
    keeps reporting the USD default (320x240, fx=763.5 at focalLength=50 mm)
    while the images are RESOLUTION at HFOV_DEG. Toggling the node's `enabled`
    input False -> True runs custom_reset() and re-initializes it, which
    re-reads the now-correct camera. Verified live on 2026-10-01: camera_info
    went from 320x240/fx=763.5 to 1280x720/fx=1108.5 with the rgb stream
    unaffected. Failure here must not abort the spawn, so it only warns.
    """
    try:
        import omni.graph.core as og
        import omni.kit.app
        node_path = f"{DRONE_PATH}/body/{CAMERA_NAME}_pub/camera_helper_camera_info"
        node = og.Controller.node(node_path)
        if not node.is_valid():
            print(f"[init] WARNING: {node_path} not found; camera_info may be stale "
                  f"(pass --hfov-deg to eval_isaac_gates.py as a workaround)")
            return
        attr = og.Controller.attribute("inputs:enabled", node)
        og.Controller.set(attr, False)
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()
        og.Controller.set(attr, True)
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()
        print("[init] camera_info helper re-initialized (reads the configured camera)")
    except Exception as exc:
        print(f"[init] WARNING: could not refresh camera_info helper ({exc}); "
              f"camera_info may be stale")


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
            "resolution":  list(RESOLUTION),
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

        # Field of view: set focalLength for the requested hFOV, keeping the
        # prim's own horizontalAperture. (Isaac derives the vertical FOV from
        # the render aspect ratio, not from verticalAperture.)
        cam_schema = UsdGeom.Camera(cam_prim)
        h_ap = cam_schema.GetHorizontalApertureAttr().Get() or 20.955
        focal = h_ap / (2.0 * math.tan(math.radians(HFOV_DEG) / 2.0))
        cam_schema.GetFocalLengthAttr().Set(focal)
        print(f"[init] camera {RESOLUTION[0]}x{RESOLUTION[1]}, hFOV {HFOV_DEG} deg "
              f"(focalLength={focal:.3f}, horizontalAperture={h_ap:.3f}, "
              f"expected fx={RESOLUTION[0] / (2.0 * math.tan(math.radians(HFOV_DEG) / 2.0)):.1f})")
        await refresh_camera_info_helper()

    print("[init] Done — start PX4 SITL now")

asyncio.ensure_future(main())