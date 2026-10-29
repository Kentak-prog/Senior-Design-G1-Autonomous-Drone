"""
Spawn a ModalAI Starling 2 in Isaac Sim (Pegasus) with the same PX4 / ROS 2 /
camera setup as spawn_example.py.

Pegasus ships no Starling model, so this loads the Iris USD and re-dimensions
it before the first physics step:
  - rotor0..3 moved to the Starling 2 motor spacing (132 mm x 176 mm),
  - body mass / diagonal inertia set so the vehicle is 0.285 kg with the
    estimated Starling 2 inertia,
  - Starling thrust curve, yaw-moment coefficient and linear drag passed to
    Pegasus through MultirotorConfig,
  - front camera moved to the Starling's nose.
Every number, with its derivation and source, is in STARLING2_DYNAMICS.md.
Values marked (est.) there are estimates; verify before trusting absolutes.

If a converted Starling USD exists (e.g. from ModalAI's STEP file), set
STARLING_USD to it. The rotor / mass overrides still run, so the USD only has
to provide geometry with Pegasus' body + rotor0..3 structure.

DRONE_PATH and the ROS namespace keep the Iris names on purpose: spawn_test_gates
(CAMERA_PRIM_PATH), min_snap_node, the vision eval scripts and the docs all
hard-code /World/Iris and /iris_0. Rename them here only together with those.

Run ONCE per Isaac Sim process, like spawn_example.py.
"""
import asyncio
import math
import omni.usd
from pxr import UsdGeom, UsdPhysics, Gf
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend, PX4MavlinkBackendConfig,
)
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.graphs.ros2_camera_graph import ROS2CameraGraph
from pegasus.simulator.logic.thrusters.quadratic_thrust_curve import QuadraticThrustCurve
from pegasus.simulator.logic.dynamics.linear_drag import LinearDrag
from pegasus.simulator.params import ROBOTS

DRONE_PATH  = "/World/Iris"     # see module docstring before renaming
NAMESPACE   = "/iris_0"
CAMERA_NAME = "front_cam"

# None -> re-dimensioned Iris USD; otherwise a path to a Starling USD.
STARLING_USD = None

# Camera: identical to spawn_example.py. KEEP RESOLUTION IN SYNC with
# spawn_test_gates.RESOLUTION.
RESOLUTION = (1280, 720)
HFOV_DEG   = 60.0
# Camera mount in body FLU (m) and up-pitch (deg). (est.) -- the Iris used
# (0.30, 0, 0.05), which is outside the Starling's 154 mm long body.
CAMERA_POS       = (0.07, 0.0, 0.0)
CAMERA_PITCH_DEG = 15.0

# ---- Starling 2 parameters (STARLING2_DYNAMICS.md, section 2) ----
MASS         = 0.285                        # kg, with battery
INERTIA_DIAG = (5.62e-4, 4.79e-4, 9.81e-4)  # kg m^2 (Ixx, Iyy, Izz), (est.)
ARM_X        = 0.066                        # m, half of 132 mm (fore-aft)
ARM_Y        = 0.088                        # m, half of 176 mm (left-right)
# Rotor order / spin directions follow the Iris (Pegasus' PX4 motor order):
# rotor0 front-right, rotor1 back-left, rotor2 front-left, rotor3 back-right.
ROTOR_XY     = ((ARM_X, -ARM_Y), (-ARM_X, ARM_Y), (ARM_X, ARM_Y), (-ARM_X, -ARM_Y))
ROT_DIR      = [-1, -1, 1, 1]
K_F          = 1.50e-6                      # N/(rad/s)^2: 1.815 N per rotor at 1100 rad/s
K_M          = 8.29e-8                      # N m/(rad/s)^2, (est.)
W_MIN        = 0.0                          # rad/s
W_MAX        = 1100.0                       # rad/s, Pegasus PX4 mapping at u = 1
DRAG_COEFF   = [0.106, 0.064, 0.0]          # N/(m/s), body FLU, (est.)


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
    """Re-initialize the camera_info helper node so it reads the configured
    camera. Same fix as spawn_example.refresh_camera_info_helper (see its
    docstring for why). Failure only warns."""
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


def _set_translate(prim, xyz):
    """Set a prim's local translation, reusing its translate op if it has one."""
    xf = UsdGeom.Xformable(prim)
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(*xyz))
            return
    UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(*xyz))


def apply_starling_geometry(stage):
    """
    Move rotor0..3 to the Starling 2 motor spacing and set the body mass /
    inertia. Must run after Multirotor() creates the prims and before
    world.reset_async() starts physics.

    Rotors: each rotor prim's x/y are replaced (its z is kept) and so is the
    body-side anchor (localPos0) of every physics joint whose body1 is that
    rotor, so the joint does not snap the rotor back to the Iris position.
    Pegasus applies each rotor's thrust at the rotor prim, so these positions
    are the moment arms of the roll / pitch torques.

    Mass: the body gets MASS minus the rotors' own masses, and the inertia
    INERTIA_DIAG minus the rotors' point-mass (parallel-axis) contribution, so
    the whole vehicle matches MASS / INERTIA_DIAG.
    """
    rotor_paths = [f"{DRONE_PATH}/rotor{i}" for i in range(4)]
    rotor_mass = 0.0
    rotor_inertia = [0.0, 0.0, 0.0]

    for i, path in enumerate(rotor_paths):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            print(f"[starling] WARNING: {path} not found; rotor {i} keeps the Iris arm")
            continue
        x, y = ROTOR_XY[i]
        z = UsdGeom.Xformable(prim).GetLocalTransformation().ExtractTranslation()[2]
        _set_translate(prim, (x, y, z))

        m_r = 0.0
        if prim.HasAPI(UsdPhysics.MassAPI):
            m_r = UsdPhysics.MassAPI(prim).GetMassAttr().Get() or 0.0
        rotor_mass += m_r
        rotor_inertia[0] += m_r * y * y
        rotor_inertia[1] += m_r * x * x
        rotor_inertia[2] += m_r * (x * x + y * y)

    # Joint anchors: find joints (anywhere under the drone) that attach a rotor.
    n_joints = 0
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(DRONE_PATH) or not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        targets = [str(t) for t in joint.GetBody1Rel().GetTargets()]
        for i, path in enumerate(rotor_paths):
            if path in targets:
                old = joint.GetLocalPos0Attr().Get() or Gf.Vec3f(0.0, 0.0, 0.0)
                x, y = ROTOR_XY[i]
                joint.GetLocalPos0Attr().Set(Gf.Vec3f(x, y, old[2]))
                n_joints += 1
    print(f"[starling] rotors moved to x=+-{ARM_X} m, y=+-{ARM_Y} m "
          f"({n_joints} rotor joint anchors updated)")
    if n_joints != 4:
        print("[starling] WARNING: expected 4 rotor joints; check the rotor prims in the "
              "Physics inspector (if the body is scaled, localPos0 is in scaled units)")

    body = stage.GetPrimAtPath(f"{DRONE_PATH}/body")
    if not body.IsValid():
        print(f"[starling] WARNING: {DRONE_PATH}/body not found; mass/inertia unchanged")
        return
    body_mass = MASS - rotor_mass
    body_inertia = [max(I - Ir, 1e-7) for I, Ir in zip(INERTIA_DIAG, rotor_inertia)]
    if body_mass <= 0.0:
        print(f"[starling] WARNING: rotor prims weigh {rotor_mass:.3f} kg >= {MASS} kg; "
              f"body mass left unchanged")
        return
    mass_api = UsdPhysics.MassAPI.Apply(body)
    mass_api.CreateMassAttr().Set(body_mass)
    mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*body_inertia))
    print(f"[starling] body mass {body_mass:.4f} kg + rotors {rotor_mass:.4f} kg = {MASS} kg, "
          f"body inertia {tuple(round(v, 7) for v in body_inertia)} kg m^2")


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
            "namespace":   NAMESPACE,
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
    # Starling thrust curve: F_i = K_F w_i^2, yaw moment = sum K_M w_i^2 rot_dir_i
    config.thrust_curve = QuadraticThrustCurve(config={
        "num_rotors":                 4,
        "rotor_constant":             [K_F] * 4,
        "rolling_moment_coefficient": [K_M] * 4,
        "rot_dir":                    ROT_DIR,
        "min_rotor_velocity":         [W_MIN] * 4,
        "max_rotor_velocity":         [W_MAX] * 4,
    })
    config.drag = LinearDrag(DRAG_COEFF)
    config.backends = [
        PX4MavlinkBackend(px4_config),
        ROS2Backend(vehicle_id=0, num_rotors=4),
    ]
    config.graphical_sensors = []
    config.graphs            = [cam_graph]

    # Spawn drone
    Multirotor(
        stage_prefix=DRONE_PATH,
        usd_file=STARLING_USD or ROBOTS["Iris"],
        init_pos=[0.0, 0.0, 2.5],
        init_orientation=[0.0, 0.0, 0.0, 1.0],   # quaternion [x,y,z,w], NOT euler
        config=config,
    )

    apply_starling_geometry(stage)

    await pi.world.reset_async()

    # Fix camera transform (known bug — ROS2CameraGraph bakes spawn height)
    cam_prim = stage.GetPrimAtPath(f"{DRONE_PATH}/body/{CAMERA_NAME}")
    if cam_prim.IsValid():
        xf = UsdGeom.Xformable(cam_prim)
        xf.ClearXformOpOrder()
        cam_prim.RemoveProperty("xformOp:translate")
        cam_prim.RemoveProperty("xformOp:orient")
        cam_prim.RemoveProperty("xformOp:rotateXYZ")
        cam_prim.RemoveProperty("xformOp:scale")
        xf.AddTranslateOp().Set(Gf.Vec3d(*CAMERA_POS))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(90.0 + CAMERA_PITCH_DEG, 0.0, -90.0))
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

    hover_w = math.sqrt(MASS * 9.81 / (4.0 * K_F))
    print(f"[init] Starling 2: {MASS} kg, max thrust {4 * K_F * W_MAX ** 2:.2f} N, "
          f"hover {hover_w:.0f} rad/s = PX4 output {(hover_w - 100.0) / 1000.0:.3f} "
          f"(set MPC_THR_HOVER to about this)")
    print("[init] Done — start PX4 SITL now")

asyncio.ensure_future(main())
