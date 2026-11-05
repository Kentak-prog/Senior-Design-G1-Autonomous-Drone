#!/bin/sh
# PX4 SITL parameters for the Starling 2 Max in Isaac Sim
# (workspace/spawn_starling2max.py). Run after PX4 SITL has started and
# before takeoff, in the PX4 container (the same px4-<command> clients as
# px4-commander):
#
#     sh px4/starling2max_sitl_params.sh
#
# These make SITL map thrust the way the real drone does. Values marked
# (inferred) / (est.) come from workspace/STARLING2MAX_DYNAMICS.md; replace
# them with the drone's saved parameter file when we have it (section 12).
#
# Not set here, on purpose: the rate-loop gains (MC_*RATE_P/I/D/K). SITL keeps
# the Iris airframe's gains. The Max has about 5x the Iris' roll/pitch
# authority per unit of output relative to its inertia, so if the hover
# oscillates, halve MC_ROLLRATE_P / MC_PITCHRATE_P (and their D) first, then
# take the real drone's gains from its parameter file.
set -e
P=${PX4_PARAM:-px4-param}

# Thrust model (STARLING2MAX_DYNAMICS.md 4.2). Must match the profile's
# px4_thr_mdl_fac, which the attitude bridge and ThrustMap invert.
$P set THR_MDL_FAC 0.9          # (inferred)
$P set MPC_THR_HOVER 0.14       # 5,407 RPM: motor output 0.341 -> setpoint 0.139
$P set MPC_THR_MAX 0.6          # as on the drone: 9,628 RPM, 17.8 N
$P set MPC_USE_HTE 1            # let the hover-thrust estimator refine it

# Control allocation geometry, FRD (x forward, y right), Pegasus / PX4 motor
# order: 0 front-right, 1 back-left, 2 front-left, 3 back-right.
# KM > 0 = CCW (rotors 0, 1), magnitude k_m / k_f = 0.0829 m (est.).
$P set CA_ROTOR_COUNT 4
$P set CA_ROTOR0_PX  0.095
$P set CA_ROTOR0_PY  0.130
$P set CA_ROTOR0_KM  0.0829
$P set CA_ROTOR1_PX -0.095
$P set CA_ROTOR1_PY -0.130
$P set CA_ROTOR1_KM  0.0829
$P set CA_ROTOR2_PX  0.095
$P set CA_ROTOR2_PY -0.130
$P set CA_ROTOR2_KM -0.0829
$P set CA_ROTOR3_PX -0.095
$P set CA_ROTOR3_PY  0.130
$P set CA_ROTOR3_KM -0.0829

# Sim-only settings from the Iris bring-up (no battery model in Pegasus;
# airmode keeps attitude authority at low thrust).
$P set COM_LOW_BAT_ACT 0
$P set MC_AIRMODE 1

$P show THR_MDL_FAC
$P show MPC_THR_HOVER
echo "Starling 2 Max SITL parameters set"
