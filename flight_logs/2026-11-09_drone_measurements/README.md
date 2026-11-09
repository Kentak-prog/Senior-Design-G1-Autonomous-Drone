# Starling 2 Max: drone data captured 2026-11-09

Drone SKU `MRB-D0012-4-V2-C29-T7-M28-X0`: airframe V2, C29 sensors. The CAD model measured earlier is V3/C28.

| File | What it is |
|---|---|
| `extrinsics.conf` | `/etc/modalai/extrinsics.conf` from the drone: sensor positions and orientations |
| `camera_intrinsics_1280x800_fisheye.yml` | One camera calibration (1280×800 fisheye, so an AR0144 tracking camera; exact file name not recorded) |
| `voxl-version.txt` | `voxl-version`: SDK 1.5.1, PX4 1.14 (`voxl-px4 1.14.0-2.0.105`) |
| `voxl-camera-server.conf` | `/etc/modalai/voxl-camera-server.conf`: which cameras run, at what resolution and frame rate |

## Cameras that are running (from `voxl-camera-server.conf`)

| Camera | Sensor | Enabled | Streams | Notes |
|---|---|---|---|---|
| `tracking_front` | AR0144 | yes, 30 fps | raw + processed 1280×800, h265 video | `en_rotate` on (image rotated 180°) |
| `hires_front` | IMX412 | yes, 30 fps | h264 video 1024×768 and 4056×3040, JPEG snapshots | **no uncompressed `preview` stream** (`en_preview` false); lens 3.1 mm, 17 mm in 35 mm terms (about 91° × 75° FOV, estimated) |
| `tof` | PMD (`pmd-tof-liow2`) | yes, 10 fps | depth | `en_rotate` on |
| `tracking_down` | AR0144 | yes, 30 fps | raw + processed 1280×800 | |
| `hires_down` | IMX412 | **no** | | |

`tracking_rear` and `lepton0_raw` appear in `extrinsics.conf` but not here, so the drone isn't running them.

## Simulator camera (decision 2026-11-09: use the color `hires_front` camera)

`workspace/spawn_starling.py` already models `hires_front`: 1024×768, level, at FLU (0.0711, −0.0003, −0.0070) m,
which matches this `extrinsics.conf` (also identical to `workspace/starling2max_c29_extrinsics.conf`).

Its field of view is still uncertain: ModalAI's M0161 page gives a 2.7 mm lens (HFOV 98.7°, used in the sim),
but this camera config's EXIF setting says 3.1 mm (HFOV 90.8°). No `hires_front` calibration was found on the
drone. To settle it, enable the `hires_front` preview stream and calibrate it with `voxl-camera-calibration`.

## Sensor positions in our body frame

ModalAI's `body` frame is FRD (x forward, y right, z down), with its origin at the center of mass. The sim and the
rest of this repo use FLU (x forward, y left, z up), so y and z flip sign. Cameras are given relative to
`imu_apps`, which sits at FRD (0.0303, −0.0057, −0.0116) m in `body` with no rotation, so the two are added.
Rotations are intrinsic XYZ (degrees); the camera's optical axis is its z axis (OpenCV convention).

| Sensor | FRD position (m) | **FLU position (m)** | Looks | Image x / y |
|---|---|---|---|---|
| `hires_front` | (0.0711, 0.0003, 0.0070) | **(0.071, −0.000, −0.007)** | forward, level | right / down |
| `tracking_front` | (0.0693, −0.0057, −0.0110) | **(0.069, 0.006, 0.011)** | forward, level | right / down |
| `tof` | (0.0695, 0.0096, −0.0080) | **(0.070, −0.010, 0.008)** | forward, level | down / left (image rotated 90°) |
| `tracking_rear` | (−0.0617, 0.0103, −0.0080) | (−0.062, −0.010, 0.008) | backward | left / down |
| `tracking_down` | (−0.0585, −0.0127, 0.0215) | (−0.059, 0.013, −0.022) | down | back / left |
| `hires_down` | (−0.0584, 0.0053, 0.0232) | (−0.058, −0.005, −0.023) | down | right / back |
| `lepton0_raw` (thermal) | (−0.0288, 0.0573, 0.0181) | (−0.029, −0.057, −0.018) | down | right / back |
| `imu_apps` | (0.0303, −0.0057, −0.0116) | (0.030, 0.006, 0.012) | — | — |
| `imu_px4` | (0.0051, 0.0080, −0.0116) | (0.005, −0.008, 0.012) | — | — |

- The front cameras and TOF all point **straight ahead with no tilt**. The sim currently tilts its camera 15° up.
- `ground`: when the drone sits on its landing gear, the body origin is **0.0776 m** above the ground.
- The file's comment says this is a one-off C29 layout with the small Starling 2 front camera bezel and a PMD TOF on the front.
- The `lepton0_raw` thermal-camera entry may be left over from a template; check whether the drone has a Lepton.
