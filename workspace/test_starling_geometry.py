"""
Check that spawn_starling.apply_starling_geometry puts each rotor where PX4
mixes it (ROTOR_XY, in the rigid body's frame), without starting Isaac.

Builds an in-memory stage shaped like Pegasus' Iris: /World/Iris/body (rigid
body) with a mesh child body/body rotated 90 deg about z, rotor0..3 siblings,
and revolute joints whose body0 is body/body. That rotation is what put every
rotor 90 deg off on 2026-11-07 (the drone flipped on takeoff). The function is
extracted from spawn_starling.py with ast, so omni / Pegasus are not needed,
only pxr. In the isaac-sim container:

    cd /workspace/Senior-Design-G1-Autonomous-Drone
    L=$(ls -d /isaac-sim/extscache/omni.usd.libs-*)
    PYTHONPATH=$L LD_LIBRARY_PATH=$L/bin /isaac-sim/kit/python/bin/python3 workspace/test_starling_geometry.py

Expect "ALL OK". Optional argument: another spawn script to check.
"""
import ast
import os
import sys
from pxr import Usd, UsdGeom, UsdPhysics, Gf

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "spawn_starling.py")
src = open(path).read()
tree = ast.parse(src)
want = {"_set_translate", "apply_starling_geometry"}
consts = {"DRONE_PATH", "ARM_X", "ARM_Y", "ROTOR_XY", "MASS", "INERTIA_DIAG"}
g = {"Usd": Usd, "UsdGeom": UsdGeom, "UsdPhysics": UsdPhysics, "Gf": Gf, "print": print}
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in consts for t in node.targets):
        exec(compile(ast.Module([node], []), "spawn", "exec"), g)
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in want:
        exec(compile(ast.Module([node], []), "spawn", "exec"), g)
D = g["DRONE_PATH"]

stage = Usd.Stage.CreateInMemory()
UsdGeom.Xform.Define(stage, "/World")
iris = UsdGeom.Xform.Define(stage, D)
UsdGeom.XformCommonAPI(iris).SetTranslate(Gf.Vec3d(0.3, -0.2, 2.5))
UsdGeom.XformCommonAPI(iris).SetRotate(Gf.Vec3f(0, 0, 30))          # some heading
body = UsdGeom.Xform.Define(stage, f"{D}/body")
UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
mesh = UsdGeom.Xform.Define(stage, f"{D}/body/body")
UsdGeom.XformCommonAPI(mesh).SetRotate(Gf.Vec3f(0, 0, 90))           # the Iris mesh frame
iris_xy = [(0.13, -0.22), (-0.13, 0.20), (0.13, 0.22), (-0.13, -0.20)]
for i, (x, y) in enumerate(iris_xy):
    r = UsdGeom.Xform.Define(stage, f"{D}/rotor{i}")
    UsdGeom.XformCommonAPI(r).SetTranslate(Gf.Vec3d(x, y, 0.023))
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{D}/rotor{i}/joint{i}")
    j.CreateBody0Rel().SetTargets([f"{D}/body/body"])
    j.CreateBody1Rel().SetTargets([f"{D}/rotor{i}"])
    # Iris anchor in the mesh frame: rotate the body-frame point by -90 deg
    j.CreateLocalPos0Attr().Set(Gf.Vec3f(y, -x, 0.023))
    j.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))

g["apply_starling_geometry"](stage)

xc = UsdGeom.XformCache(Usd.TimeCode.Default())
bw_inv = xc.GetLocalToWorldTransform(body.GetPrim()).GetInverse()
mesh_to_body = xc.GetLocalToWorldTransform(mesh.GetPrim()) * bw_inv
ok = True
for i, (x, y) in enumerate(g["ROTOR_XY"]):
    j = UsdPhysics.Joint(stage.GetPrimAtPath(f"{D}/rotor{i}/joint{i}"))
    a = mesh_to_body.Transform(Gf.Vec3d(j.GetLocalPos0Attr().Get()))
    p = bw_inv.Transform(xc.GetLocalToWorldTransform(stage.GetPrimAtPath(f"{D}/rotor{i}")).ExtractTranslation())
    good = abs(a[0] - x) < 1e-5 and abs(a[1] - y) < 1e-5 and abs(a[2] - 0.023) < 1e-5 \
        and abs(p[0] - x) < 1e-5 and abs(p[1] - y) < 1e-5 and abs(p[2] - 0.023) < 1e-5
    ok &= good
    print(f"rotor{i}: want ({x:+.3f},{y:+.3f})  anchor in body ({a[0]:+.3f},{a[1]:+.3f},{a[2]:+.3f})  "
          f"prim in body ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})  {'OK' if good else 'WRONG'}")
print("ALL OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
