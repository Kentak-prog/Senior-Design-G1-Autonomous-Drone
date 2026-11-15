# Camera calibration: Starling 2 Max `hires_front`

| File | What it is |
|---|---|
| `hires_front_1024x768.yml` | **Use this.** Calibration of `hires_front_small_color` (1024×768), 2026-11-13 |
| `hires_front_small_approx.yml` | Superseded guess (EXIF 3.1 mm, zero distortion), kept because `real_gate_01` used it |
| `calibrate_hires.py` | Capture checkerboard frames from the VOXL Portal feed, then solve |
| `make_checkerboard.py`, `checkerboard_9x6_25mm_letter.pdf` | The printed board: 10×7 squares of 25 mm = 9×6 inner corners, US Letter |
| `hires_front_frames/` | The 40 captured frames (local only, gitignored) |

## Result (2026-11-13)

OpenCV fisheye (Kannala-Brandt) model, 39 frames after dropping one outlier, RMS 0.49 px (worst frame 0.85 px):

```
fx 505.40  fy 505.79  cx 518.52  cy 390.42      (1024x768)
D  [-0.0171, -0.00922, 0.00516, -0.00005]
```

- fx matches a **3.1 mm** lens (the `exif_focal_length` in `voxl-camera-server.conf`), not the 2.7 mm on ModalAI's M0161 page.
- About 119° × 88° to the image edges. The pinhole equivalent the sim uses is 90.7° HFOV (`workspace/spawn_starling.py`).
- Seeding the solver at fx 400, 440 or 600 gives the same answer. A bootstrap over frames gives a std of about 4 px in fx (0.8 %) and 1–1.5 px in cx, cy.
- The square size is assumed to be exactly 25.0 mm; the print was not measured. A different size scales PnP range. Re-solve with `--square-mm`, which needs the local frames.

The drone has no `hires_front` calibration of its own. `flight_logs/2026-11-09_drone_measurements/camera_intrinsics_1280x800_fisheye.yml`
is an AR0144 **tracking** camera, not this one.

## Redo it

```bash
adb forward tcp:8080 tcp:80
python3 calibrate_hires.py capture --target 40        # saves a frame when the board is still and in a new pose
python3 calibrate_hires.py solve --square-mm 25.0      # writes hires_front_1024x768.yml
```

Capture writes `hires_front_frames/coverage.png` so you can see which parts of the image still need the board. Cover the
corners and edges, hold the board close (a third to half of the image), and tilt it 20–40°.
