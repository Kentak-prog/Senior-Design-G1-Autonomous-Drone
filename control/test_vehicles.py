"""Self-tests for the vehicle profiles. Run: python test_vehicles.py

Checks that every profile is physically consistent (limits inside what the
rotors can produce), that the Iris profile still gives the numbers the stack
used before profiles existed, that the Starling 2 Max profile matches
workspace/spawn_starling.py and px4/starling2max_sitl_params.sh, that PX4's
thrust model round-trips, and that DRONE_VEHICLE selects the profile.
"""

import ast
import math
import os
import re
import subprocess
import sys

from vehicles import ENV_VAR, VEHICLES, get_vehicle

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
G = 9.81


def test_profiles_physically_consistent():
    for name, v in VEHICLES.items():
        assert v.name == name, (name, v.name)
        assert v.mass > 0 and all(i > 0 for i in v.inertia_diag), name
        assert len(v.rotor_xy) == 4 and len(v.rot_dir) == 4 and sum(v.rot_dir) == 0, name
        # total thrust limit: at most saturation (the Max's is PX4's cap, well below), above hover with margin
        t_sat = v.max_total_thrust()
        assert v.total_thrust_max <= t_sat, (name, v.total_thrust_max, t_sat)
        assert v.total_thrust_max / (v.mass * G) >= 2.0, name
        # tilt limit below the tilt that still holds altitude at full thrust
        assert v.tilt_limit < math.acos(v.mass * G / v.total_thrust_max), name
        assert 0.0 < v.hover_motor_output() < 1.0, name
        # vertical accel limit up to (T_max - m g) / m
        assert v.vertical_accel[1] <= (v.total_thrust_max - v.mass * G) / v.mass, name
        assert v.w_min <= v.px4_w_offset and v.px4_w_offset + v.px4_w_scale <= v.w_max + 1e-6, name
        assert v.tau_motor >= 0.0, name
        if name == "iris":
            continue   # Iris rotor_xy is transposed vs its USD torques; see vehicles.py
        # body torque limits inside what the rotor geometry can produce: at rotor saturation,
        # or at the per-rotor thrust of a cap well below it (the Max: PX4 MPC_THR_MAX)
        capped = v.total_thrust_max < 0.98 * t_sat
        f = v.total_thrust_max / 4.0 if capped else v.max_rotor_thrust()
        roll_sat = sum(y * f for _, y in v.rotor_xy if y > 0)
        pitch_sat = sum(-x * f for x, _ in v.rotor_xy if x < 0)
        yaw_sat = 2.0 * (v.k_m / v.k_f) * f
        for lim, sat in zip(v.body_torque_max, (roll_sat, pitch_sat, yaw_sat)):
            assert 0.95 * sat <= lim <= sat, (name, lim, sat)
        assert v.body_torque_max[0] <= sum(y for _, y in v.rotor_xy if y > 0) * v.max_rotor_thrust(), name
    print(f"  {sorted(VEHICLES)}: thrust, tilt, torque limits within rotor saturation  OK")


def test_iris_unchanged():
    """The numbers the stack reads, as they were before control/vehicles.py."""
    v = get_vehicle("iris")
    assert v.mass == 1.5 and v.inertia_diag == (0.029125, 0.029125, 0.055225)
    assert v.drag_coeff == (0.50, 0.30, 0.0) and v.k_f == 8.54858e-6 and v.k_m == 1e-6
    assert v.rotor_xy == ((0.13, -0.22), (-0.13, 0.20), (0.13, 0.22), (-0.13, -0.20))
    assert (v.w_max, v.px4_w_scale, v.px4_w_offset) == (1100.0, 1000.0, 100.0)
    assert v.tilt_limit == 1.0 and v.body_torque_max == (2.6, 4.4, 2.4)
    assert v.total_thrust_max == 41.0 and v.max_velocity == 12.0 and v.max_accel == 15.0
    assert v.collision_radius == 0.30
    assert abs(v.hover_motor_output() - 0.556) < 0.001
    print("  iris: mass, inertia, thrust curve, drag, limits as before; hover u = 0.556  OK")


def test_starling2max_matches_hover_flights():
    """2026-11-03 hover flights: 573 g, k_f 4.383e-6, hover 5,407 RPM, ESC 2,000-12,000 RPM,
    PX4 MPC_THR_MAX 0.6 = 9,628 RPM = 4.46 N per rotor, motor lag 35 ms."""
    v = get_vehicle("starling2max")
    rpm = lambda r: r * 2.0 * math.pi / 60.0
    assert abs(4.0 * v.k_f * rpm(5407) ** 2 - v.mass * G) / (v.mass * G) < 1e-3     # hover balances
    assert abs(v.w_min - rpm(2000)) < 0.01 and abs(v.w_max - rpm(12000)) < 0.01
    assert abs(v.hover_motor_output() - (5407 - 2000) / 10000) < 1e-3                # u = 0.341
    assert abs(v.max_rotor_thrust() - 6.92) < 0.01                                  # ESC limit
    assert abs(4.0 * v.k_f * rpm(9628) ** 2 - 17.82) < 0.01 and v.total_thrust_max <= 17.82
    assert v.tau_motor == 0.035 and v.rot_dir == (-1.0, -1.0, 1.0, 1.0)              # 0, 1 CCW; 2, 3 CW
    print(f"  starling2max: m g balanced at 5,407 RPM, hover u = {v.hover_motor_output():.3f}, "
          f"PX4 cap 17.82 N, ESC max {v.max_rotor_thrust():.2f} N per rotor, lag 35 ms  OK")


def _spawn_constants(path):
    """Module-level NAME = literal assignments of a spawn script (it needs Isaac to import)."""
    tree = ast.parse(open(path).read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def test_starling2max_matches_spawn_script():
    # workspace/spawn_starling.py is the Starling 2 Max (C29) since 2026-11-05; the
    # Starling 2 (285 g) profile has no spawn script any more.
    c = _spawn_constants(os.path.join(REPO, "workspace", "spawn_starling.py"))
    v = get_vehicle("starling2max")
    assert c["MASS"] == v.mass and tuple(c["INERTIA_DIAG"]) == v.inertia_diag, (c["INERTIA_DIAG"], v.inertia_diag)
    assert (c["ARM_X"], c["ARM_Y"]) == (v.rotor_xy[2][0], v.rotor_xy[2][1])
    assert tuple(float(d) for d in c["ROT_DIR"]) == v.rot_dir
    assert (c["K_F"], c["K_M"], c["W_MAX"]) == (v.k_f, v.k_m, v.w_max)
    assert c["W_MIN"] == 0.0            # motors stop when disarmed; armed idle is ZERO_POSITION_ARMED
    assert (c["INPUT_SCALING"], c["ZERO_POSITION_ARMED"]) == (v.px4_w_scale, v.px4_w_offset)
    assert c["ZERO_POSITION_ARMED"] == v.w_min
    assert tuple(c["DRAG_COEFF"]) == v.drag_coeff and c["TAU_MOTOR"] == v.tau_motor
    # PX4 SITL: the spawn's printed set and the shell script agree with the profile
    kappa = round(v.k_m / v.k_f, 4)
    sp = c["PX4_SITL_PARAMS"]
    assert sp["THR_MDL_FAC"] == v.px4_thr_mdl_fac
    assert sp["MPC_THR_HOVER"] == round(v.hover_thrust_setpoint(), 2)
    for i, (x, y) in enumerate(v.rotor_xy):                  # profile FLU -> PX4 FRD: PY = -y
        km = kappa if v.rot_dir[i] < 0 else -kappa            # rot_dir -1 = CCW = KM > 0
        assert (sp[f"CA_ROTOR{i}_PX"], sp[f"CA_ROTOR{i}_PY"], sp[f"CA_ROTOR{i}_KM"]) == (x, -y, km), i
    sh = open(os.path.join(REPO, "px4", "starling2max_sitl_params.sh")).read()
    lines = dict(re.findall(r"^\$P set (\S+) (\S+)", sh, flags=re.M))
    assert {k: float(x) for k, x in lines.items()} == {k: float(x) for k, x in sp.items()}, (lines, sp)
    print(f"  starling2max profile == workspace/spawn_starling.py == px4/starling2max_sitl_params.sh "
          f"(kappa {kappa} m)  OK")


def test_thrust_model():
    """PX4 THR_MDL_FAC chain: setpoint <-> motor output <-> newtons (from vehicle/starling2max-sim)."""
    from vehicles import thr_mdl_motor_output, thr_mdl_setpoint
    for f in (0.0, 0.3, 0.9, 1.0):
        for t in (0.0, 0.14, 0.5, 1.0):
            assert abs(thr_mdl_setpoint(thr_mdl_motor_output(t, f), f) - t) < 1e-12, (f, t)
    for name in VEHICLES:
        v = get_vehicle(name)
        T = v.mass * 9.81
        assert abs(v.setpoint_to_thrust(v.hover_thrust_setpoint()) - T) < 1e-9, name
        assert abs(v.setpoint_to_thrust(v.thrust_to_setpoint(0.7 * T)) - 0.7 * T) < 1e-9, name
    iris = get_vehicle("iris")
    assert iris.px4_thr_mdl_fac == 0.0 and iris.hover_thrust_setpoint() == iris.hover_motor_output()
    m = get_vehicle("starling2max")
    assert abs(m.hover_thrust_setpoint() - 0.139) < 1e-3                       # MPC_THR_HOVER
    assert abs(m.setpoint_to_thrust(0.6) - 17.82) < 0.02                        # MPC_THR_MAX 0.6
    print(f"  thrust model: Max hover setpoint {m.hover_thrust_setpoint():.3f}, "
          f"MPC_THR_MAX 0.6 = {m.setpoint_to_thrust(0.6):.2f} N; Iris unchanged  OK")


def _lag_step_response(tau, dt, n):
    """The spawn script's discretization, w += (1 - exp(-dt/tau)) (w_ref - w)."""
    a = 1.0 - math.exp(-dt / tau)
    w, out = 0.0, []
    for _ in range(n):
        w += a * (1.0 - w)
        out.append(w)
    return out


def test_spawn_motor_lag_discretization():
    src = open(os.path.join(REPO, "workspace", "spawn_starling.py")).read()
    assert "a = 1.0 - math.exp(-dt / self._tau)" in src
    dt = 1.0 / 250.0
    resp = _lag_step_response(0.035, dt, 250)
    for k in (5, 9, 25):                # exact at the samples: 1 - exp(-t / tau)
        assert abs(resp[k - 1] - (1.0 - math.exp(-k * dt / 0.035))) < 1e-12
    print("  spawn motor lag: exact first-order step response at 250 Hz  OK")


def test_unknown_vehicle():
    try:
        get_vehicle("iris2")
    except ValueError as e:
        assert "starling2max" in str(e)
    else:
        raise AssertionError("unknown vehicle accepted")
    print("  unknown vehicle -> ValueError listing the profiles  OK")


def _probe(vehicle, code):
    env = dict(os.environ)
    if vehicle is None:
        env.pop(ENV_VAR, None)
    else:
        env[ENV_VAR] = vehicle
    rl = os.path.join(REPO, "rl")
    out = subprocess.run([sys.executable, "-c", f"import sys; sys.path[:0] = [{HERE!r}, {rl!r}]; {code}"],
                         env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_env_selects_profile():
    code = ("from drone_constraints import drone_constraints as c, vehicle; "
            "from drone_params import QuadParams as Q; "
            "print(vehicle.name, c['mass'], c['total_thrust_limits']['max'], Q().mass, Q().k_f)")
    assert _probe(None, code) == "iris 1.5 41.0 1.5 8.54858e-06", _probe(None, code)
    got = _probe("starling2max", code).split()
    v = get_vehicle("starling2max")
    assert got == ["starling2max", str(v.mass), str(v.total_thrust_max), str(v.mass), str(v.k_f)], got
    print("  DRONE_VEHICLE unset -> iris; =starling2max -> constraints and QuadParams follow  OK")


def test_quadparams_for_vehicle():
    code = ("from drone_params import QuadParams as Q; "
            "p = Q.for_vehicle('starling2', control_dt=0.01); "
            "print(p.mass, p.inertia_diag[2], p.collision_radius, p.control_dt, p.rate_kp[0])")
    assert _probe("iris", code) == "0.285 0.000981 0.17 0.01 40.0", _probe("iris", code)
    print("  QuadParams.for_vehicle ignores the env, keeps non-airframe defaults, takes overrides  OK")


if __name__ == "__main__":
    print("test_profiles_physically_consistent"); test_profiles_physically_consistent()
    print("test_iris_unchanged");                 test_iris_unchanged()
    print("test_starling2max_matches_hover_flights"); test_starling2max_matches_hover_flights()
    print("test_starling2max_matches_spawn_script"); test_starling2max_matches_spawn_script()
    print("test_thrust_model");                   test_thrust_model()
    print("test_spawn_motor_lag_discretization"); test_spawn_motor_lag_discretization()
    print("test_unknown_vehicle");                test_unknown_vehicle()
    print("test_env_selects_profile");            test_env_selects_profile()
    print("test_quadparams_for_vehicle");         test_quadparams_for_vehicle()
    print("\nAll tests passed.")
