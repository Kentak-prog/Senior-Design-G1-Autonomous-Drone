"""
Task 1.1 — Attach an RGB camera to the drone in Isaac Sim and get frames,
intrinsics, and the camera's world pose out of it.

Run this INSIDE the Isaac Sim Python environment (script editor or a
standalone app), not with plain python3. The import block handles both the
4.5-era `isaacsim.sensors.camera` namespace and the older
`omni.isaac.sensor` one; check which your container actually has.

The one thing to verify empirically before trusting anything downstream:
point the drone at a gate whose world position you know, run
`debug_check_projection()`, and confirm the projected pixel lands on the gate
in the saved image. If it is mirrored or upside down, the USD -> OpenCV
conversion in camera.py is being applied in the wrong direction.
"""

import numpy as np

try:
    from isaacsim.sensors.camera import Camera          # Isaac Sim 4.5+
except ImportError:  # pragma: no cover - depends on container image
    from omni.isaac.sensor import Camera               # Isaac Sim <= 4.2

import omni.isaac.core.utils.numpy.rotations as rot_utils

from camera import (CameraIntrinsics, usd_camera_pose_to_cv, project_points,
                    invert_transform, transform_points)


# Body -> camera mount. Adjust to wherever the camera physically sits.
# Orientation is given in the USD camera convention (-Z forward), so the
# default below is a forward-facing camera tilted down by `pitch_deg`.
CAMERA_OFFSET_M = (0.10, 0.0, 0.03)
CAMERA_PITCH_DEG = -15.0


def attach_drone_camera(drone_prim_path: str,
                        resolution=(1280, 720),
                        frequency: int = 60,
                        pitch_deg: float = CAMERA_PITCH_DEG) -> Camera:
    """
    Create a camera prim parented to the drone body so it moves with it.

    drone_prim_path is the body prim, e.g. "/World/iris/base_link".
    Returns an initialized Camera. Call world.reset() after this, then
    camera.initialize() before the first get_rgba().
    """
    cam = Camera(
        prim_path=f"{drone_prim_path}/front_camera",
        translation=np.array(CAMERA_OFFSET_M),
        frequency=frequency,
        resolution=resolution,
        orientation=rot_utils.euler_angles_to_quats(
            np.array([0.0, pitch_deg, 0.0]), degrees=True),
    )
    cam.initialize()
    cam.add_motion_vectors_to_frame()  # optional; drop if you do not need it
    return cam


def intrinsics_from_camera(cam: Camera) -> CameraIntrinsics:
    """
    Pull real intrinsics off the camera prim rather than assuming an FOV.
    Isaac exposes focal length and aperture in USD units; only their ratio
    matters, so no unit conversion is needed.
    """
    width, height = cam.get_resolution()
    focal = cam.get_focal_length()
    h_aperture = cam.get_horizontal_aperture()
    v_aperture = cam.get_vertical_aperture()
    return CameraIntrinsics.from_usd_camera(
        width=width, height=height,
        focal_length=focal,
        horizontal_aperture=h_aperture,
        vertical_aperture=v_aperture,
    )


def camera_pose_world(cam: Camera) -> np.ndarray:
    """
    T_world_cam in the OpenCV camera convention, ready to compose with a
    solvePnP result via gate_pose.gate_pose_world().
    """
    position, quat_wxyz = cam.get_world_pose()
    return usd_camera_pose_to_cv(np.asarray(position), np.asarray(quat_wxyz))


def grab_frame(cam: Camera) -> np.ndarray | None:
    """RGB uint8 HxWx3, or None if the sensor has not produced a frame yet."""
    rgba = cam.get_rgba()
    if rgba is None or rgba.size == 0:
        return None
    return rgba[:, :, :3]


def debug_check_projection(cam: Camera, world_point: np.ndarray):
    """
    Sanity check the whole frame chain. Give it a point whose world position
    you know (a gate center), and it prints where that point should appear in
    the image. Overlay it on the saved frame and confirm it lands correctly.
    """
    intr = intrinsics_from_camera(cam)
    T_world_cam = camera_pose_world(cam)
    p_cam = transform_points(invert_transform(T_world_cam), world_point)
    uv = project_points(p_cam, intr)
    print(f"world {np.asarray(world_point).ravel()} -> "
          f"cam {p_cam.ravel()} -> pixel {uv.ravel()}")
    if p_cam[0, 2] <= 0:
        print("  point is BEHIND the camera; check the USD->CV conversion")
    return uv


# ---------------------------------------------------------------------------
# Task 1.2 sketch: synthetic capture loop with ground-truth labels.
# Fill in gate_prim_paths with the gates in your stage and this writes an
# image plus the true corner pixels and true gate pose for every frame.
# ---------------------------------------------------------------------------

def capture_labeled_frame(cam: Camera, T_world_gate: np.ndarray,
                          gate_side: float = 1.5) -> dict | None:
    """One labeled sample: image, true corner pixels, true pose in camera frame."""
    from gate_pose import gate_model_points

    rgb = grab_frame(cam)
    if rgb is None:
        return None
    intr = intrinsics_from_camera(cam)
    T_cam_world = invert_transform(camera_pose_world(cam))
    T_cam_gate = T_cam_world @ T_world_gate
    corners_cam = transform_points(T_cam_gate, gate_model_points(gate_side))
    corners_px = project_points(corners_cam, intr)
    return {
        "rgb": rgb,
        "corners_px": corners_px,
        "T_cam_gate": T_cam_gate,
        "T_world_gate": T_world_gate,
        "intrinsics": intr.to_dict(),
    }
