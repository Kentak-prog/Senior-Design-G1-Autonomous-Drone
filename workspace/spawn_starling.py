"""
Spawn a ModalAI Starling 2 Max (C29) in Isaac Sim (Pegasus) with the same
PX4 / ROS 2 setup as spawn_example.py, and the C29 front camera (see Camera below).

Pegasus ships no Starling model, so this loads the Iris USD and re-dimensions
it before the first physics step:
  - rotor0..3 moved to the Starling 2 Max motor spacing (190 mm x 260 mm),
  - body mass / diagonal inertia set from the ModalAI CAD model
    (D0012-4-V3-C28-M36, SolidWorks mass properties),
  - thrust curve (with a first-order motor lag), yaw-moment coefficient and
    linear drag passed to Pegasus through MultirotorConfig,
  - PX4 output -> rotor speed mapped to the drone's real ESC range
    (VOXL_ESC_RPM_MIN/MAX = 2000/12000 RPM),
  - front camera: the C29 front IMX412 (FOV from its datasheet) at its
    mount from the drone's extrinsics.conf (level, on the nose).
Every number, with its derivation and source, is in STARLING2_DYNAMICS.md.
Mass is the measured 573 g; k_f and the motor lag are measured from hover logs.
PROVISIONAL: inertia is the CAD value (whole-assembly override, uniform density)
until part masses are set in CAD; k_m is estimated (no yaw test yet).

The values must equal the starling2max entry of control/vehicles.py, which the
controllers and bridges use with DRONE_VEHICLE=starling2max
(control/test_vehicles.py checks).

PX4 SITL must get the drone's thrust model and rotor geometry (PX4_SITL_PARAMS
below, printed at the end of the spawn as `param set` lines for pxh>; the same
set is px4/starling2max_sitl_params.sh). Without them SITL keeps the Iris
airframe's rotor geometry and a linear thrust model.

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
from pxr import Usd, UsdGeom, UsdPhysics, Gf
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

# Camera: the C29 front hires camera, ModalAI M0161 (IMX412, 4056x3040 active,
# 1.55 um square pixels = 6.29 x 4.71 mm, 2.7 mm lens; https://docs.modalai.com/M0161/,
# Sony IMX412-AACK flyer p. 2).
# Its FOV row reads "DxHxV 120.4 x 93.5 x 146", which is impossible as labeled;
# as H x V x D it sits between a pinhole (98.7 x 82.2 x 111) and an equidistant
# fisheye (133 x 100 x 167) of that lens, i.e. barrel distortion. Isaac renders
# an ideal pinhole, so this is the UNDISTORTED image with the lens's central
# focal length: fx = 2.7 mm / 1.55 um = 1742 px at full width = 439.8 px at
# 1024 wide, HFOV 2*atan(512/439.8) = 98.7 deg (VFOV 82.3 deg at 4:3).
# 1024x768 (the sensor's 4:3 full-resolution mode) matches the drone's
# voxl-camera-server.conf (flight_logs/2026-11-09_drone_measurements/): hires_front
# streams small_video 1024x768 and large_video 4056x3040, both 4:3, but has
# en_preview false, so onboard vision needs its preview stream enabled (or the
# h264 decoded). The same file's exif_focal_length is 3.1 mm (17 mm in 35 mm
# terms), which would give HFOV 90.8 deg; it is a config value, not a
# measurement, and differs from M0161's 2.7 mm. A calibration of this camera
# decides between them (none is on the drone yet). The sensor's 16:9 modes
# CROP rows (4K2K 4056x2288, 1080p binned from 4056x2224), so they keep the
# 98.7 deg HFOV and see only ~66 deg vertically: for those set
# CAMERA_RESOLUTION="1024x578" (Isaac derives the vertical FOV from the aspect).
# An extra horizontal resolution only adds pixels; the FOV does not depend on it.
# The Iris (spawn_example.py) is 1280x720 / 60 deg. spawn_test_gates' camera-
# relative layout assumes that; spawn courses from a manifest
# (spawn_gates_from_manifest.py) with this camera.
RESOLUTION = (1024, 768)
HFOV_DEG   = 98.7
# Camera mount in body FLU (m) and up-pitch (deg): the drone's own
# /etc/modalai/extrinsics.conf (copy: starling2max_c29_extrinsics.conf),
# hires_front (the front IMX412) chained through imu_apps:
#   body->imu_apps   T (0.0303, -0.0057, -0.0116), RPY (0, 0, 0)
#   imu_apps->hires_front T (0.0408, 0.006, 0.0186), RPY (0, 90, 90)
# ModalAI frames are FRD, so the camera is at (0.0711, 0.0003, 0.0070) FRD =
# (0.0711, -0.0003, -0.0070) FLU, and RPY (0, 90, 90) points the lens straight
# along body +X with image-right = body right: a LEVEL camera (pitch 0).
CAMERA_POS       = (0.0711, -0.0003, -0.0070)
CAMERA_PITCH_DEG = 0.0

# Overrides, set BEFORE exec'ing this file (os.environ, not Script Editor
# variables -- see spawn_test_gates.main's note), e.g. the C29 front camera:
#   import os; os.environ.update(CAMERA_RESOLUTION="1280x720", CAMERA_HFOV_DEG="60")   # the Iris camera
# Isaac renders an ideal pinhole with square pixels and a centered principal
# point, so only the horizontal FOV is set; take it from the calibration as
# 2*atan(width / (2*fx)).
import os as _os
if "CAMERA_RESOLUTION" in _os.environ:
    RESOLUTION = tuple(int(v) for v in _os.environ.pop("CAMERA_RESOLUTION").lower().split("x"))
if "CAMERA_HFOV_DEG" in _os.environ:
    HFOV_DEG = float(_os.environ.pop("CAMERA_HFOV_DEG"))
if "CAMERA_POS" in _os.environ:
    CAMERA_POS = tuple(float(v) for v in _os.environ.pop("CAMERA_POS").split(","))
if "CAMERA_PITCH_DEG" in _os.environ:
    CAMERA_PITCH_DEG = float(_os.environ.pop("CAMERA_PITCH_DEG"))
print(f"[starling] camera {RESOLUTION[0]}x{RESOLUTION[1]}, hFOV {HFOV_DEG} deg, "
      f"at {CAMERA_POS} m (body FLU), pitch {CAMERA_PITCH_DEG} deg up")

# ---- Starling 2 Max parameters (STARLING2_DYNAMICS.md, section 2) ----
MASS         = 0.573                        # kg, measured with both batteries (C29)
INERTIA_DIAG = (3.456e-3, 2.465e-3, 5.469e-3)  # kg m^2 (Ixx, Iyy, Izz), CAD at 573 g override, PROVISIONAL
ARM_X        = 0.095                        # m, half of 190 mm (fore-aft), CAD
ARM_Y        = 0.130                        # m, half of 260 mm (left-right), CAD
# Rotor order / spin directions follow the Iris (Pegasus' PX4 motor order):
# rotor0 front-right, rotor1 back-left, rotor2 front-left, rotor3 back-right.
ROTOR_XY     = ((ARM_X, -ARM_Y), (-ARM_X, ARM_Y), (ARM_X, ARM_Y), (-ARM_X, -ARM_Y))
ROT_DIR      = [-1, -1, 1, 1]
# Each rotor link's own mass and inertia, set explicitly (the Iris USD leaves
# them to PhysX's density default). Small, so Pegasus' spinning prop animation
# carries almost no angular momentum; MASS / INERTIA_DIAG stay the totals.
ROTOR_MASS   = 0.005                        # kg, prop + bell (est.)
ROTOR_INERTIA = (1.0e-6, 1.0e-6, 2.0e-6)    # kg m^2 about the rotor's own axes (est.)
K_F          = 4.383e-6                     # N/(rad/s)^2, measured: hover logs 2026-11-03, 5,407 RPM
K_M          = 5.65e-8                      # N m/(rad/s)^2, (est.) torque/thrust 0.0129 m x K_F
TAU_MOTOR    = 0.035                        # s, first-order rotor lag, measured (30-40 ms)
# VOXL ESC: RPM = 2000 + 10000 u while armed (VOXL_ESC_RPM_MIN/MAX, measured).
# Pegasus' PX4 backend computes w = INPUT_SCALING * (u + 0) + ZERO_POSITION_ARMED.
INPUT_SCALING       = 1047.20               # rad/s per unit PX4 output (10,000 RPM)
ZERO_POSITION_ARMED = 209.44                # rad/s, 2,000 RPM idle when armed
# Clip range of the thrust curve. W_MIN is 0, not the ESC's 209.44: Pegasus
# sends 0 when disarmed and the motors must then stop; armed, the backend never
# commands less than ZERO_POSITION_ARMED anyway.
W_MIN        = 0.0                          # rad/s
W_MAX        = 1256.64                      # rad/s = 12,000 RPM (ESC max)
DRAG_COEFF   = [0.226, 0.136, 0.0]          # N/(m/s), body FLU, (est.)

# PX4 SITL parameters (the same set is px4/starling2max_sitl_params.sh;
# control/test_vehicles.py checks both against the profile). Thrust model from
# the drone (THR_MDL_FAC inferred from MPC_THR_MAX 0.6 = 9,628 RPM,
# STARLING2MAX_DYNAMICS.md 4.2). Control allocation in FRD (x forward, y right),
# PX4 motor order 0 front-right, 1 back-left, 2 front-left, 3 back-right;
# KM = +-kappa = K_M / K_F (> 0 = CCW). The drone's own CA_ROTORn_PX/PY are
# +-0.085 / +-0.0625 m (STARLING2_DYNAMICS.md item 4a); SITL uses the measured
# airframe, which is what Pegasus flies. The drone's CA_ROTORn_KM is +-0.05
# (PX4's default allocation ratio, logged, STARLING2_DYNAMICS.md S7c); SITL uses
# the model's kappa so PX4 allocates yaw for the physics Pegasus simulates.
PX4_SITL_PARAMS = {
    "THR_MDL_FAC": 0.9, "MPC_THR_HOVER": 0.14, "MPC_THR_MAX": 0.6, "MPC_THR_MIN": 0.04,
    "MPC_USE_HTE": 1,
    "CA_ROTOR_COUNT": 4,
    "CA_ROTOR0_PX": 0.095, "CA_ROTOR0_PY": 0.13, "CA_ROTOR0_KM": 0.0129,
    "CA_ROTOR1_PX": -0.095, "CA_ROTOR1_PY": -0.13, "CA_ROTOR1_KM": 0.0129,
    "CA_ROTOR2_PX": 0.095, "CA_ROTOR2_PY": -0.13, "CA_ROTOR2_KM": -0.0129,
    "CA_ROTOR3_PX": -0.095, "CA_ROTOR3_PY": 0.13, "CA_ROTOR3_KM": -0.0129,
    "COM_LOW_BAT_ACT": 0, "MC_AIRMODE": 1,   # sim only: no battery model; attitude authority at low thrust
    # Rate-loop gains. SITL's Iris gains (P 0.15, I 0.2, D 0.003) act ~3.4x (roll)
    # and ~5.0x (pitch) harder on the Starling: angular acceleration per unit of
    # normalized torque, 4 * T_max * arm / I, is 1041 / 1067 rad/s^2 here against
    # 307 / 215 for the Iris. With them it went into a +-270 deg/s pitch
    # oscillation at liftoff (2026-11-07). Scaled by those ratios, it hovered at
    # 2.51 m within 5 mm, body rates within 0.25 deg/s. Replace with the real
    # drone's gains from its parameter file when we have it.
    "MC_ROLLRATE_P": 0.044, "MC_ROLLRATE_I": 0.06, "MC_ROLLRATE_D": 0.0009,
    "MC_PITCHRATE_P": 0.03, "MC_PITCHRATE_I": 0.04, "MC_PITCHRATE_D": 0.0006,
}


class LaggedQuadraticThrustCurve(QuadraticThrustCurve):
    """
    QuadraticThrustCurve whose rotor speeds follow the commanded speed through
    a first-order lag, dw/dt = (w_ref - w) / TAU_MOTOR (exact discretization
    for the physics step dt). Thrust and yaw moment then come from the lagged
    speed through the parent's update. With tau = 0 it is the parent.
    (As in spawn_starling2max.py on the vehicle/starling2max-sim branch.)
    """

    def __init__(self, config={}, tau=TAU_MOTOR):
        super().__init__(config=config)
        self._tau = float(tau)
        n = config.get("num_rotors", 4)
        self._lag_ref = [0.0] * n
        self._lag_w = [0.0] * n

    def set_input_reference(self, input_reference):
        self._lag_ref = [float(w) for w in input_reference]

    def update(self, state, dt):
        if self._tau > 0.0 and dt > 0.0:
            a = 1.0 - math.exp(-dt / self._tau)
            self._lag_w = [w + a * (r - w) for w, r in zip(self._lag_w, self._lag_ref)]
        else:
            self._lag_w = list(self._lag_ref)
        super().set_input_reference(self._lag_w)
        return super().update(state, dt)


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

    ROTOR_XY is in the rigid body's frame ({DRONE_PATH}/body: x forward,
    y left), the frame PX4's CA_ROTORn_PX/PY describe. localPos0 is in the
    frame of the joint's body0, which for the Iris is the mesh prim
    {DRONE_PATH}/body/body, rotated 90 deg about z from the rigid body.
    Writing ROTOR_XY into localPos0 unconverted put every rotor 90 deg CCW
    from where PX4 mixes it (2026-11-07: roll and pitch swapped through the
    motors, and the drone flipped on takeoff). So both the rotor prims and the
    anchors are converted through the stage transforms here.

    Prop colliders are disabled. The rotors keep the Iris' 10-inch props, and
    at the Starling spacing the front and rear props on each side overlap
    (hubs 0.19 m apart, props 0.247 m across). Pegasus spins the rotor joints
    for the animation, so the counter-rotating blades hit each other and spun
    the drone at -200 deg/s on the ground as soon as it armed (2026-11-07).
    Thrust is applied as forces at the rotor prims, so the props need no
    collision. Each rotor gets an explicit ROTOR_MASS / ROTOR_INERTIA.

    Mass: the body gets MASS minus the rotors' own masses, and the inertia
    INERTIA_DIAG minus the rotors' point-mass (parallel-axis) contribution, so
    the whole vehicle matches MASS / INERTIA_DIAG.
    """
    rotor_paths = [f"{DRONE_PATH}/rotor{i}" for i in range(4)]
    rotor_mass = 0.0
    rotor_inertia = [0.0, 0.0, 0.0]

    # Frames: points are written p_frame * M (Gf row-vector convention), so
    # body-frame point -> world is p * body_w, and world -> X's local is
    # p * inverse(X_w).
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    body_prim = stage.GetPrimAtPath(f"{DRONE_PATH}/body")
    body_w = xc.GetLocalToWorldTransform(body_prim) if body_prim.IsValid() else Gf.Matrix4d(1.0)
    body_w_inv = body_w.GetInverse()

    for i, path in enumerate(rotor_paths):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            print(f"[starling] WARNING: {path} not found; rotor {i} keeps the Iris arm")
            continue
        x, y = ROTOR_XY[i]
        # The rotor's current position in the body frame (keep its height).
        p_body = body_w_inv.Transform(xc.GetLocalToWorldTransform(prim).ExtractTranslation())
        target_w = body_w.Transform(Gf.Vec3d(x, y, p_body[2]))
        parent_w = xc.GetLocalToWorldTransform(prim.GetParent())
        _set_translate(prim, tuple(parent_w.GetInverse().Transform(target_w)))

        # No prop collisions (see the docstring), and a defined rotor mass.
        n_off = 0
        for sub in Usd.PrimRange(prim):
            if sub.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(sub).CreateCollisionEnabledAttr().Set(False)
                n_off += 1
        rotor_mass_api = UsdPhysics.MassAPI.Apply(prim)
        rotor_mass_api.CreateMassAttr().Set(ROTOR_MASS)
        rotor_mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*ROTOR_INERTIA))
        rotor_mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        print(f"[starling] rotor{i}: {n_off} prop collider(s) disabled, mass {ROTOR_MASS} kg")

        m_r = ROTOR_MASS
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
        body0 = [str(t) for t in joint.GetBody0Rel().GetTargets()]
        b0_prim = stage.GetPrimAtPath(body0[0]) if body0 else body_prim
        # body frame -> body0 frame, and back
        body_to_b0 = body_w * xc.GetLocalToWorldTransform(b0_prim).GetInverse()
        b0_to_body = body_to_b0.GetInverse()
        for i, path in enumerate(rotor_paths):
            if path in targets:
                old = Gf.Vec3d(joint.GetLocalPos0Attr().Get() or Gf.Vec3f(0.0, 0.0, 0.0))
                x, y = ROTOR_XY[i]
                old_body = b0_to_body.Transform(old)
                new_b0 = body_to_b0.Transform(Gf.Vec3d(x, y, old_body[2]))
                joint.GetLocalPos0Attr().Set(Gf.Vec3f(new_b0))
                check = b0_to_body.Transform(Gf.Vec3d(joint.GetLocalPos0Attr().Get()))
                print(f"[starling] rotor{i} anchor in body frame ({check[0]:+.3f}, {check[1]:+.3f}, "
                      f"{check[2]:+.3f}) m (written in {b0_prim.GetPath()} as "
                      f"({new_b0[0]:+.3f}, {new_b0[1]:+.3f}, {new_b0[2]:+.3f}))")
                n_joints += 1
    print(f"[starling] rotors moved to x=+-{ARM_X} m, y=+-{ARM_Y} m "
          f"({n_joints} rotor joint anchors updated)")
    if n_joints != 4:
        print("[starling] WARNING: expected 4 rotor joints; check the rotor prims in the "
              "Physics inspector")

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
        # PX4 output u in [0, 1] -> rotor speed, like the VOXL ESC:
        # w = (u + offset) * scaling + zero_position_armed = 209.44 + 1047.20 u rad/s
        "input_offset":        [0.0] * 4,
        "input_scaling":       [INPUT_SCALING] * 4,
        "zero_position_armed": [ZERO_POSITION_ARMED] * 4,
    })

    config = MultirotorConfig()
    # Starling thrust curve: F_i = K_F w_i^2, yaw moment = sum K_M w_i^2 rot_dir_i
    config.thrust_curve = LaggedQuadraticThrustCurve(config={
        "num_rotors":                 4,
        "rotor_constant":             [K_F] * 4,
        "rolling_moment_coefficient": [K_M] * 4,
        "rot_dir":                    ROT_DIR,
        "min_rotor_velocity":         [W_MIN] * 4,
        "max_rotor_velocity":         [W_MAX] * 4,
    }, tau=TAU_MOTOR)
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
    x_h = (hover_w - ZERO_POSITION_ARMED) / INPUT_SCALING   # PX4 motor output at hover
    f = PX4_SITL_PARAMS["THR_MDL_FAC"]
    u_h = f * x_h ** 2 + (1.0 - f) * x_h                    # PX4 normalized hover thrust
    print(f"[init] Starling 2 Max: {MASS} kg, max thrust {4 * K_F * W_MAX ** 2:.2f} N "
          f"(T/W {4 * K_F * W_MAX ** 2 / (MASS * 9.81):.2f} at 12000 RPM), "
          f"hover {hover_w * 60 / (2 * math.pi):.0f} RPM = motor output {x_h:.3f} = hover thrust {u_h:.3f}")
    print("[init] Before takeoff, paste into pxh> (or run px4/starling2max_sitl_params.sh):")
    for k, v in PX4_SITL_PARAMS.items():
        print(f"param set {k} {v}")
    print("[init] Done — start PX4 SITL now")

asyncio.ensure_future(main())
