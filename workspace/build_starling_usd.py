"""
Build the Starling 2 Max (C29) USD that workspace/spawn_starling.py loads through
STARLING_USD, from the STL meshes exported out of the SolidWorks model
(Starling2Max_C29.SLDASM, one STL per part, CAD assembly coordinates in millimetres;
the four props are Propeller_Rotor0..3, in PX4 motor order).

The stage has the structure Pegasus expects of a multirotor (the same as its
Iris USD):

    /vehicle                    Xform, articulation root (default prim)
    /vehicle/body               rigid body: mass, inertia, center of mass
    /vehicle/body/visuals/*     one mesh per CAD part
    /vehicle/body/collisions/*  simple invisible collision shapes
    /vehicle/rotor0..3          rigid bodies at the motor axes, with the prop mesh
    /vehicle/rotorN/jointN      revolute joint about z: body0 = /vehicle/body

Frames. CAD: X = drone left, Y = up, Z = forward, origin on the plate between
the motors. Body (this repo, FLU): x = Z_cad, y = X_cad, z = Y_cad - 10.0 mm.
The origin is the point ModalAI's extrinsics.conf calls `body`: 77.6 mm above
the ground on the landing gear (the feet are at Y_cad = -67.6 mm), which is
also the height of the CAD model's center of mass (10.0 mm, mass_model.py).
In x and y the origin is the center of the four motors; the hover logs put the
center of mass there within a few tenths of a millimetre, so the center of
mass is authored at (0, 0, 0).

Mass and inertia here are the values in spawn_starling.py (MASS, INERTIA_DIAG,
ROTOR_MASS), which rewrites them at spawn anyway; they are authored so the USD
is also right when loaded on its own.

Run with any Python that has pxr and numpy. In the isaac-sim container:

    L=$(ls -d /isaac-sim/extscache/omni.usd.libs-*)
    PYTHONPATH=$L LD_LIBRARY_PATH=$L/bin /isaac-sim/kit/python/bin/python3 \
        workspace/build_starling_usd.py --stl-dir <dir> --out workspace/assets/starling2max_c29.usd
"""
import argparse
import glob
import os
import struct

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

ORIGIN_Y_CAD = 0.010                      # m, body origin height in CAD (see module docstring)
ROTOR_XY = ((0.095, -0.130), (-0.095, 0.130), (0.095, 0.130), (-0.095, -0.130))   # body FLU, PX4 order
ROTOR_Z = 0.03475 - ORIGIN_Y_CAD          # m, prop plane above the body origin (CAD Y = 34.75 mm)
MASS = 0.573
INERTIA_DIAG = (3.333e-3, 2.055e-3, 5.170e-3)
ROTOR_MASS = 0.005
ROTOR_INERTIA = (1.0e-6, 1.0e-6, 2.0e-6)

COLORS = {                                # linear RGB by the suffix (or, battery / propeller, the start) of the part name
    "CarbonPlate": (0.03, 0.03, 0.035), "Plastic": (0.02, 0.02, 0.02), "Motors": (0.05, 0.05, 0.05),
    "PropNuts": (0.55, 0.42, 0.10), "Electronics": (0.02, 0.16, 0.07), "Cameras": (0.04, 0.04, 0.05),
    "Fasteners": (0.45, 0.45, 0.47), "Battery": (0.05, 0.35, 0.12), "Propeller": (0.03, 0.03, 0.03),
}


def read_stl(path):
    """Binary STL -> (points (n, 3) float64, triangles (m, 3) int), vertices welded."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:5] == b"solid" and b"facet" in data[:2000]:
        raise ValueError(f"{path} is an ASCII STL; export binary")
    (n,) = struct.unpack_from("<I", data, 80)
    rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n, offset=84)
    verts = rec["v"].reshape(-1, 3).astype(np.float64)
    uniq, inv = np.unique(np.round(verts, 4), axis=0, return_inverse=True)
    tris = inv.reshape(-1, 3)
    keep = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
    return uniq, tris[keep]


def cad_mm_to_body(p):
    """CAD millimetres (X left, Y up, Z forward) -> body metres (x forward, y left, z up)."""
    p = np.asarray(p, dtype=np.float64) / 1000.0
    return np.stack([p[..., 2], p[..., 0], p[..., 1] - ORIGIN_Y_CAD], axis=-1)


def add_mesh(stage, path, points, tris, color):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in points])
    mesh.CreateFaceVertexCountsAttr([3] * len(tris))
    mesh.CreateFaceVertexIndicesAttr([int(i) for i in tris.reshape(-1)])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    lo, hi = points.min(axis=0), points.max(axis=0)
    mesh.CreateExtentAttr([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))])
    return mesh


def part_name(path):
    """Part name from an STL file name ("<assembly> - <part>-1.STL" when exported from the assembly)."""
    name = os.path.splitext(os.path.basename(path))[0]
    if " - " in name:
        name = name.split(" - ", 1)[1]
    name = name.rsplit("-", 1)[0] if name.rsplit("-", 1)[-1].isdigit() else name
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)


def color_for(name):
    for key, col in COLORS.items():
        if name.endswith(key) or name.startswith(key):
            return col
    return (0.3, 0.3, 0.3)


def collider(stage, path, kind, pos, size, rot_z_deg=0.0, axis="Z"):
    """Invisible collision primitive under the body. size: cube (x, y, z) full sizes; cylinder/capsule (radius, height)."""
    if kind == "cube":
        prim = UsdGeom.Cube.Define(stage, path)
        prim.CreateSizeAttr(1.0)
        scale = Gf.Vec3f(*size)
    else:
        prim = (UsdGeom.Capsule if kind == "capsule" else UsdGeom.Cylinder).Define(stage, path)
        prim.CreateRadiusAttr(size[0])
        prim.CreateHeightAttr(size[1])
        prim.CreateAxisAttr(axis)
        scale = Gf.Vec3f(1.0, 1.0, 1.0)
    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, rot_z_deg))
    xf.AddScaleOp().Set(scale)
    prim.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    return prim


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stl-dir", required=True, help="folder with the STL files exported from SolidWorks (mm, CAD axes)")
    ap.add_argument("--out", required=True, help="output .usd / .usdc / .usda")
    args = ap.parse_args()

    stage = Usd.Stage.CreateNew(args.out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    vehicle = UsdGeom.Xform.Define(stage, "/vehicle")
    stage.SetDefaultPrim(vehicle.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(vehicle.GetPrim())

    body = UsdGeom.Xform.Define(stage, "/vehicle/body")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
    rotor_i = [sum(ROTOR_MASS * y * y for x, y in ROTOR_XY), sum(ROTOR_MASS * x * x for x, y in ROTOR_XY),
               sum(ROTOR_MASS * (x * x + y * y) for x, y in ROTOR_XY)]
    mass.CreateMassAttr(MASS - 4 * ROTOR_MASS)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*[i - r for i, r in zip(INERTIA_DIAG, rotor_i)]))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    UsdGeom.Scope.Define(stage, "/vehicle/body/visuals")
    UsdGeom.Scope.Define(stage, "/vehicle/body/collisions")

    n_tri = 0
    props = {}
    for path in sorted(glob.glob(os.path.join(args.stl_dir, "*.stl")) + glob.glob(os.path.join(args.stl_dir, "*.STL"))):
        name = part_name(path)
        pts, tris = read_stl(path)
        if name.startswith("Propeller_Rotor"):
            props[int(name[len("Propeller_Rotor")])] = (pts, tris)
            continue
        add_mesh(stage, f"/vehicle/body/visuals/{name.replace('S2M_', '')}", cad_mm_to_body(pts), tris, color_for(name))
        n_tri += len(tris)
        print(f"  {name:<34} {len(pts):>7} points {len(tris):>7} triangles")
    if sorted(props) != [0, 1, 2, 3]:
        raise SystemExit("need Propeller_Rotor0..3 STL files in --stl-dir")

    # Collision shapes (body frame, metres): two arm beams motor to motor, four motors, the fuselage with the
    # battery packs, and the four landing legs down to the feet 77.6 mm below the origin.
    z_plate = 0.0025 - ORIGIN_Y_CAD
    ang = np.degrees(np.arctan2(0.130, 0.095))
    collider(stage, "/vehicle/body/collisions/arm_fl_br", "cube", (0, 0, z_plate), (0.34, 0.02, 0.006), ang)
    collider(stage, "/vehicle/body/collisions/arm_fr_bl", "cube", (0, 0, z_plate), (0.34, 0.02, 0.006), -ang)
    collider(stage, "/vehicle/body/collisions/fuselage", "cube", (0.0, 0.0, 0.0195 - ORIGIN_Y_CAD), (0.145, 0.102, 0.071))
    for i, (x, y) in enumerate(ROTOR_XY):
        collider(stage, f"/vehicle/body/collisions/motor{i}", "cylinder", (x, y, 0.01775 - ORIGIN_Y_CAD), (0.0138, 0.0275))
        sx, sy = np.sign(x), np.sign(y)
        collider(stage, f"/vehicle/body/collisions/leg{i}", "capsule",
                 (sx * 0.0923, sy * 0.1246, -0.0338 - ORIGIN_Y_CAD), (0.0055, 0.0566))

    # Rotors: each prop mesh (CAD assembly coordinates) expressed relative to its rotor prim; joint about z.
    for i, (x, y) in enumerate(ROTOR_XY):
        rotor = UsdGeom.Xform.Define(stage, f"/vehicle/rotor{i}")
        UsdGeom.Xformable(rotor).AddTranslateOp().Set(Gf.Vec3d(x, y, ROTOR_Z))
        UsdPhysics.RigidBodyAPI.Apply(rotor.GetPrim())
        rm = UsdPhysics.MassAPI.Apply(rotor.GetPrim())
        rm.CreateMassAttr(ROTOR_MASS)
        rm.CreateDiagonalInertiaAttr(Gf.Vec3f(*ROTOR_INERTIA))
        rm.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
        pp, pt = props[i]
        add_mesh(stage, f"/vehicle/rotor{i}/prop", cad_mm_to_body(pp) - np.array([x, y, ROTOR_Z]), pt, color_for("Propeller"))
        n_tri += len(pt)
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/vehicle/rotor{i}/joint{i}")
        joint.CreateAxisAttr("Z")
        joint.CreateBody0Rel().SetTargets([Sdf.Path("/vehicle/body")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/vehicle/rotor{i}")])
        joint.CreateLocalPos0Attr(Gf.Vec3f(x, y, ROTOR_Z))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    stage.GetRootLayer().Save()
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(vehicle.GetPrim()).ComputeAlignedRange()
    print(f"wrote {args.out}: {n_tri} triangles, visual bounds (m) min {tuple(round(v, 4) for v in bbox.GetMin())} "
          f"max {tuple(round(v, 4) for v in bbox.GetMax())}")


if __name__ == "__main__":
    main()
