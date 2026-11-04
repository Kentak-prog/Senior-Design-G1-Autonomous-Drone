"""
Physical parameters of the quadrotor the RL stack trains and tests on.

Everything the simulator, the rate controller and the PID baseline need to
know about the airframe lives in ONE dataclass, `QuadParams`. Its airframe
defaults come from the active vehicle profile in control/vehicles.py, picked
with the DRONE_VEHICLE environment variable (default "iris": the Pegasus Iris
used in Isaac Sim, workspace/spawn_example.py). QuadParams.for_vehicle(name)
builds the params of a given vehicle regardless of the environment.

CONVENTIONS (shared by every module in rl/)
    World frame: ENU (x east, y north, z up) = Pegasus "map" frame = the frame
        of test_gates.json. Gravity is -z, g = 9.81.
    Body frame:  FLU (x forward, y left, z up). Body rates are in this frame.
    Quaternions: WXYZ, body -> world (same order as vision/camera.quat_to_rot).
    Rotor i spins with direction rot_dir[i]; Pegasus applies the yaw moment
        sum(k_m * w_i^2 * rot_dir_i) about body +z, so rot_dir = +1 pushes the
        airframe counter-clockwise (seen from above) when that rotor speeds up.

WHY these numbers (provenance)
    airframe (mass, inertia, rotors, thrust curve, drag, collision radius):
        the vehicle profile; control/vehicles.py documents each one.
    thrust curve / yaw coefficient / rotor-speed limit: Pegasus
        QuadraticThrustCurve form (thrust = k_f * w^2, w in [0, w_max]).
    PX4 output -> rotor speed: Pegasus PX4MavlinkBackend maps a normalised
        motor output u in [0, 1] to w = 1000*u + 100 rad/s. This is what lets
        us convert a collective-thrust command to a PX4 setpoint later.
    linear drag: Pegasus LinearDrag, force = -coeff * v_body
        (elementwise, body frame), applied in the body frame.
    tau_motor: from the profile. 0 = instantaneous, as Pegasus rotors are
        (Iris); the Starling 2 Max's 35 ms is measured from flight logs.
    rate-controller gains: our own choice, modelled on PX4's rate loop (see
        quad_sim.RateController); not read from PX4.
    TODO verify everything marked above before trusting absolute numbers.
"""

from dataclasses import dataclass, fields
import math
import os
import sys

_CONTROL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "control")
if _CONTROL not in sys.path:
    sys.path.insert(0, _CONTROL)
from vehicles import Vehicle, active_vehicle, get_vehicle   # noqa: E402

_V = active_vehicle()


@dataclass
class QuadParams:
    # --- rigid body (active vehicle profile, control/vehicles.py) ---
    mass: float = _V.mass                    # kg
    inertia_diag: tuple = _V.inertia_diag    # kg m^2 (Ixx, Iyy, Izz)
    gravity: float = 9.81                    # m/s^2

    # Rotor positions in body FLU (x fwd, y left, z up).
    # Index order matches Pegasus' rotor / motor-output order.
    rotor_xy: tuple = _V.rotor_xy
    rotor_z: float = _V.rotor_z

    # --- Pegasus QuadraticThrustCurve ---
    k_f: float = _V.k_f          # thrust per rotor = k_f * w^2   [N / (rad/s)^2]
    k_m: float = _V.k_m          # yaw moment per rotor = k_m * w^2 * rot_dir [N m / (rad/s)^2]
    rot_dir: tuple = _V.rot_dir
    w_min: float = _V.w_min      # rad/s
    w_max: float = _V.w_max      # rad/s

    # --- Pegasus PX4MavlinkBackend: w = px4_w_scale * u + px4_w_offset ---
    px4_w_scale: float = _V.px4_w_scale
    px4_w_offset: float = _V.px4_w_offset

    # --- Pegasus LinearDrag: force_body = -coeff * v_body (elementwise) ---
    drag_coeff: tuple = _V.drag_coeff

    # --- timing ---
    tau_motor: float = _V.tau_motor   # s; 0 = instantaneous (Pegasus); >0 = first-order lag
    physics_dt: float = 1.0 / 250.0   # Pegasus PX4 update rate
    control_dt: float = 1.0 / 50.0    # policy / outer-loop rate

    # --- CTBR inner rate loop (see quad_sim.RateController) ---
    # torque = I * (Kp*(w_ref - w) + Ki*integral(w_ref - w) - Kd*dw/dt)
    # Kp = 40 1/s gives a ~25 ms closed-loop time constant on the ideal plant.
    rate_kp: tuple = (40.0, 40.0, 20.0)
    rate_ki: tuple = (20.0, 20.0, 10.0)     # 1/s^2
    rate_kd: tuple = (0.0, 0.0, 0.0)        # dimensionless (scales effective inertia)
    rate_int_limit: float = 0.5             # rad; anti-windup clamp on the error integral

    # --- command limits ---
    max_rate: tuple = (6.0, 6.0, 3.0)       # rad/s, roll / pitch / yaw
    collision_radius: float = _V.collision_radius   # m, drone treated as a sphere

    # ------------------------------------------------------------------
    @classmethod
    def for_vehicle(cls, name: str, **overrides) -> "QuadParams":
        """Params of the named vehicle profile (ignores $DRONE_VEHICLE).
        Non-airframe fields keep their defaults unless overridden. Only for
        code that reads nothing but QuadParams (QuadSim, ThrustMap); the MPC
        still takes its limits from $DRONE_VEHICLE (control/vehicles.py)."""
        v = get_vehicle(name)
        own = {f.name for f in fields(cls)}
        kw = {f.name: getattr(v, f.name) for f in fields(Vehicle) if f.name in own}
        kw.update(overrides)
        return cls(**kw)

    # ------------------------------------------------------------------
    def max_rotor_thrust(self) -> float:
        return self.k_f * self.w_max ** 2

    def max_total_thrust(self) -> float:
        return 4.0 * self.max_rotor_thrust()

    def hover_thrust(self) -> float:
        return self.mass * self.gravity

    def hover_rotor_speed(self) -> float:
        """Rotor speed [rad/s] at which four equal rotors carry the weight."""
        return math.sqrt(self.hover_thrust() / (4.0 * self.k_f))

    def hover_motor_output(self) -> float:
        """Normalised PX4 motor output u in [0,1] that hovers (about 0.556 for the Iris)."""
        return (self.hover_rotor_speed() - self.px4_w_offset) / self.px4_w_scale

    def px4_output_to_thrust(self, u: float) -> float:
        """Total thrust [N] of four equal rotors at PX4 motor output u.
        Note u = 0 is w = 100 rad/s, i.e. a small nonzero thrust, as in Pegasus."""
        w = self.px4_w_scale * u + self.px4_w_offset
        return 4.0 * self.k_f * w ** 2

    def thrust_to_px4_output(self, T_total: float) -> float:
        """Inverse of px4_output_to_thrust for collective thrust (equal rotors),
        clipped to u in [0, 1]."""
        w = math.sqrt(max(T_total, 0.0) / (4.0 * self.k_f))
        u = (w - self.px4_w_offset) / self.px4_w_scale
        return min(max(u, 0.0), 1.0)
