"""
Vehicle profiles: every airframe-specific number the stack uses, one profile
per drone.

    iris          Pegasus / Isaac Sim Iris. The default, and what every result
                  before 2026-11 was flown with.
    starling2     ModalAI Starling 2 (285 g). workspace/STARLING2_DYNAMICS.md,
                  workspace/spawn_starling.py.
    starling2max  ModalAI Starling 2 Max (573 g). Mass, thrust curve, motor
                  lag and spin directions from hover flights; geometry from
                  the datasheet; yaw moment, inertia and drag are estimates.
                  workspace/STARLING2MAX_DYNAMICS.md derives every number;
                  px4/sysid.py re-fits them from flight logs.

Pick one with the environment variable DRONE_VEHICLE (default "iris"):

    DRONE_VEHICLE=starling2max python3 rl/fly_course.py ...

control/drone_constraints.py and rl/drone_params.QuadParams read the active
profile at import time, so set the variable before starting Python.

ONE VEHICLE PER PROCESS. The MPC, planner and EKF read their limits from
drone_constraints, i.e. from $DRONE_VEHICLE. get_vehicle(name) and
QuadParams.for_vehicle(name) give another vehicle's numbers, but pass them
only to code that takes everything from QuadParams (QuadSim, ThrustMap):
an MPC built with another vehicle's mass would still use this process's
thrust and torque limits.

All vectors are body FLU (x forward, y left, z up). Rotor order and spin
directions follow Pegasus' PX4 motor order: rotor0 front-right, rotor1
back-left, rotor2 front-left, rotor3 back-right.

Plain Python only (no numpy) so it imports anywhere.
"""

from dataclasses import dataclass
import math
import os

ENV_VAR = "DRONE_VEHICLE"
DEFAULT_VEHICLE = "iris"


@dataclass(frozen=True)
class Vehicle:
    name: str
    description: str

    # --- rigid body ---
    mass: float                 # kg, with battery
    inertia_diag: tuple         # kg m^2 (Ixx, Iyy, Izz)
    rotor_xy: tuple             # m, body FLU, Pegasus rotor order
    rotor_z: float              # m
    rot_dir: tuple              # +1 = rotor pushes the airframe CCW (from above) when it speeds up

    # --- Pegasus QuadraticThrustCurve: F_i = k_f w^2, M_z,i = k_m w^2 rot_dir_i ---
    k_f: float                  # N / (rad/s)^2
    k_m: float                  # N m / (rad/s)^2
    w_min: float                # rad/s
    w_max: float                # rad/s

    # --- Pegasus PX4MavlinkBackend: w = px4_w_scale * u + px4_w_offset ---
    px4_w_scale: float
    px4_w_offset: float

    # --- Pegasus LinearDrag: force_body = -drag_coeff * v_body ---
    drag_coeff: tuple           # N / (m/s), body x, y, z

    # --- motor: first-order lag from commanded to actual rotor speed ---
    tau_motor: float            # s; 0 = instantaneous (Pegasus)

    # --- control limits (what the MPC, planner and RL read) ---
    tilt_limit: float           # rad, |roll| and |pitch|
    body_torque_max: tuple      # N m, roll / pitch / yaw (symmetric)
    total_thrust_max: float     # N, a little below 4 k_f w_max^2
    max_velocity: float         # m/s, per axis
    max_accel: float            # m/s^2, horizontal per axis
    vertical_accel: tuple       # m/s^2, (min, max)
    rate_limits: tuple          # rad/s, roll / pitch / yaw
    battery_voltage: tuple      # V, (cutoff under load, full)
    battery_current_max: float  # A
    collision_radius: float     # m, drone treated as a sphere

    # ------------------------------------------------------------------
    def max_rotor_thrust(self) -> float:
        return self.k_f * self.w_max ** 2

    def max_total_thrust(self) -> float:
        return 4.0 * self.max_rotor_thrust()

    def hover_motor_output(self, g: float = 9.81) -> float:
        """Normalised PX4 motor output u in [0, 1] that hovers (Pegasus mapping)."""
        w = math.sqrt(self.mass * g / (4.0 * self.k_f))
        return (w - self.px4_w_offset) / self.px4_w_scale


VEHICLES = {
    # Pegasus Iris. Numbers unchanged from drone_constraints.py / drone_params.py
    # before the profiles existed, so earlier results reproduce.
    #   mass, inertia: iris.usd. Thrust curve, k_m, w range: Pegasus
    #   QuadraticThrustCurve defaults. Drag: Pegasus iris.py.
    #   rotor_xy: PX4 sitl_gazebo iris, as rl/ has always used it. NOTE: the
    #   Iris USD has the long arm along x (about +-0.21 m fore-aft, +-0.13 m
    #   lateral), the transpose of this. Left as is so rl/ results reproduce;
    #   fix it together with a re-run of the QuadSim baselines.
    "iris": Vehicle(
        name="iris",
        description="Pegasus / Isaac Sim Iris (1.5 kg)",
        mass=1.5,
        inertia_diag=(0.029125, 0.029125, 0.055225),
        rotor_xy=((0.13, -0.22), (-0.13, 0.20), (0.13, 0.22), (-0.13, -0.20)),
        rotor_z=0.023,
        rot_dir=(-1.0, -1.0, 1.0, 1.0),
        k_f=8.54858e-6,
        k_m=1e-6,
        w_min=0.0,
        w_max=1100.0,
        px4_w_scale=1000.0,
        px4_w_offset=100.0,
        drag_coeff=(0.50, 0.30, 0.0),
        tau_motor=0.0,
        # Max tilt that holds altitude is arccos(1/2.81) = 1.21 rad; 1.0 leaves margin.
        tilt_limit=1.0,
        # Full saturation from the USD rotor geometry. Values that hold near hover
        # (a motor cannot go below zero thrust) are roll 1.8, pitch 3.1, yaw 0.85.
        body_torque_max=(2.6, 4.4, 2.4),
        total_thrust_max=41.0,          # 4 x 10.34 N
        max_velocity=12.0,              # PX4 default; the planner uses 80 % of it
        max_accel=15.0,                 # about 25.7 possible at 1.21 rad
        vertical_accel=(-9.0, 17.0),    # up (41.4 - 14.7) / 1.5 = 17.8
        rate_limits=(10.0, 10.0, 4.0),
        battery_voltage=(9.9, 12.6),    # 3S
        battery_current_max=45.0,
        collision_radius=0.30,
    ),

    # ModalAI Starling 2. Every number: workspace/STARLING2_DYNAMICS.md section 2.
    # Must match the constants in workspace/spawn_starling.py (control/test_vehicles.py checks).
    "starling2": Vehicle(
        name="starling2",
        description="ModalAI Starling 2 (0.285 kg), STARLING2_DYNAMICS.md",
        mass=0.285,
        inertia_diag=(5.62e-4, 4.79e-4, 9.81e-4),                         # (est.)
        rotor_xy=((0.066, -0.088), (-0.066, 0.088), (0.066, 0.088), (-0.066, -0.088)),
        rotor_z=0.023,
        rot_dir=(-1.0, -1.0, 1.0, 1.0),
        k_f=1.50e-6,                    # 1.815 N per rotor at 1100 rad/s
        k_m=8.29e-8,                    # (est.) kappa = 0.0553 m
        w_min=0.0,
        w_max=1100.0,
        px4_w_scale=1000.0,
        px4_w_offset=100.0,
        drag_coeff=(0.106, 0.064, 0.0),                                   # (est.)
        tau_motor=0.0,
        tilt_limit=1.0,                 # hold-altitude limit 1.176 rad (T/W 2.60)
        body_torque_max=(0.31, 0.23, 0.20),   # saturation 0.319 / 0.2396 / 0.201
        total_thrust_max=7.2,           # 4 x 1.815 = 7.26 N
        max_velocity=12.0,
        max_accel=15.0,                 # g tan(1.0) = 15.3 at the tilt limit
        vertical_accel=(-9.0, 15.0),    # up (7.26 - 2.80) / 0.285 = 15.7
        rate_limits=(10.0, 10.0, 4.0),
        battery_voltage=(6.6, 8.4),     # 2S
        battery_current_max=15.0,       # (est.)
        collision_radius=0.17,          # 0.110 m motor radius + 0.06 m prop
    ),

    # ModalAI Starling 2 Max. workspace/STARLING2MAX_DYNAMICS.md has the derivations;
    # px4/STARLING2MAX_SYSID_FLIGHTS.md the flights that replace the (est.) values.
    # Datasheet: https://docs.modalai.com/starling-2-max-datasheet/ (checked 2026-11-03). (meas.) = identified from two indoor hover flights
    # on 2026-11-03 (PX4 on VOXL 2, Position mode, 231 s of steady hover, C29).
    #   mass 0.573 kg (meas.): weighed with batteries and props.
    #   Geometry: datasheet motor-to-motor 260 mm wide x 190 mm long ->
    #     ly = 0.130 m, lx = 0.095 m, R = 0.161 m. 180 mm tri-blade props.
    #     Hover RPM of front/back and left/right pairs balance within 0.1 %,
    #     so the center of mass is centered (meas.).
    #   Motors: 2203.5 1500 KV on 4S (16.8 V full). Spin (meas.): rotors 0, 1
    #     CCW, rotors 2, 3 CW, which is rot_dir (-1, -1, 1, 1).
    #   k_f = 4.383e-6 (meas.): thrust from the accelerometer vs sum w^2,
    #     hover at 5,407 RPM (4 k_f w^2 = 5.621 N = m g). Only measured near
    #     hover (up to about 5,900 RPM); above that k_f w^2 is extrapolated.
    #   Rotor speed: VOXL ESC maps PX4 output u linearly to 2,000-12,000 RPM,
    #     so w = 1047.2 u + 209.4 rad/s, w_max 1256.6 rad/s (6.92 N per rotor,
    #     T/W 4.9 if the extrapolation holds). Hover is u = 0.341.
    #   PX4 caps collective thrust with MPC_THR_MAX = 0.6, which is 9,628 RPM,
    #     4.46 N per rotor, 17.82 N total (T/W 3.17). The control limits below
    #     use this cap, not the ESC limit.
    #   tau_motor = 0.035 s (meas.): measured RPM lags commanded RPM 30-35 ms.
    #   k_m (est.): yaw maneuvers not flown yet. Iris kappa 0.117 m scaled by
    #     prop diameter 180 / 254 mm -> kappa = 0.0829 m (STARLING2_DYNAMICS.md
    #     4.6). Tri-blade props likely have a somewhat higher C_Q/C_T.
    #   Inertia (est.): Starling 2 values x mass ratio 2.011 x (arm ratio)^2
    #     per axis (ly for Ixx, lx for Iyy, R for Izz).
    #   Drag (est.): Starling 2 values x (R ratio)^2 = 2.14 (area).
    #   Battery: two 2S 18650 packs in series = 4S.
    "starling2max": Vehicle(
        name="starling2max",
        description="ModalAI Starling 2 Max (0.573 kg), hover-identified rotors + estimates",
        mass=0.573,                                                       # (meas.)
        inertia_diag=(2.47e-3, 2.00e-3, 4.23e-3),                         # (est.)
        rotor_xy=((0.095, -0.130), (-0.095, 0.130), (0.095, 0.130), (-0.095, -0.130)),
        rotor_z=0.023,                                                    # (est.)
        rot_dir=(-1.0, -1.0, 1.0, 1.0),                                   # (meas.)
        k_f=4.383e-6,                                                     # (meas.)
        k_m=3.633e-7,                                                     # (est.)
        w_min=209.44,                   # 2,000 RPM, ESC minimum while armed
        w_max=1256.64,                  # 12,000 RPM, ESC limit
        px4_w_scale=1047.20,            # 10,000 RPM per unit of PX4 output
        px4_w_offset=209.44,
        drag_coeff=(0.227, 0.137, 0.0),                                   # (est.)
        tau_motor=0.035,                                                  # (meas.)
        tilt_limit=1.0,                 # hold-altitude limit at the PX4 cap: 1.250 rad
        body_torque_max=(1.15, 0.84, 0.73),   # at 4.46 N per rotor: 1.158 / 0.847 / 0.739
        total_thrust_max=17.8,          # PX4 MPC_THR_MAX 0.6: 4 x 4.46 = 17.82 N
        max_velocity=15.0,              # datasheet cruise 15 m/s (dash 20)
        max_accel=15.0,                 # g tan(1.0) = 15.3 at the tilt limit
        vertical_accel=(-9.0, 21.0),    # up (17.82 - 5.62) / 0.573 = 21.3
        rate_limits=(10.0, 10.0, 4.0),
        battery_voltage=(14.0, 16.8),   # 4S, 3.5 V/cell under load
        battery_current_max=30.0,       # (est.)
        collision_radius=0.25,          # 0.161 m motor radius + 0.09 m prop
    ),
}


def get_vehicle(name: str) -> Vehicle:
    try:
        return VEHICLES[name]
    except KeyError:
        raise ValueError(f"unknown vehicle {name!r}; choose one of {sorted(VEHICLES)}") from None


def active_vehicle() -> Vehicle:
    """The profile named by $DRONE_VEHICLE, or the Iris."""
    return get_vehicle(os.environ.get(ENV_VAR, DEFAULT_VEHICLE).strip() or DEFAULT_VEHICLE)
