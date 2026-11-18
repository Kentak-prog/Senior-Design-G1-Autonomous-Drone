# Starling 2 Max (VOXL 2) bring-up from a Windows laptop

Step by step: get on the drone over Wi‑Fi, learn what runs on it, then run the gate vision **on the
drone**. No flying anywhere in this guide. Each step ends with a **Check:**; don't go on until it passes.

**Status:** written 2026-11-18, before anyone ran it. Steps 7's ONNX export and benchmark were run on a
Mac; everything on the drone is untried. VOXL service, pipe and command names come from the drone's
package list (`flight_logs/2026-11-09_drone_measurements/voxl-version.txt`, SDK 1.5.1) and ModalAI's
docs: if one differs on the drone, trust the drone and fix this file.

```
Windows laptop ── Wi‑Fi ──► VOXL 2 (the Starling's computer, Linux, ARM)
  PowerShell: ssh, scp          ├─ ModalAI services: camera, IMU, VIO, PX4, VOXL Portal (web page)
  browser: VOXL Portal          └─ Docker container: our vision code (numpy + OpenCV + onnxruntime)
  Python: YOLO export, tests
```

---

## Step 0: tools on the laptop (once)

| Tool | Get it | Used for |
|---|---|---|
| Git for Windows | git-scm.com | the repo |
| Python 3.12 | python.org (tick **Add python.exe to PATH**) | YOLO export, offline tests |
| Android Platform-Tools | Google's "SDK Platform-Tools" zip; unzip to e.g. `C:\platform-tools` | `adb`, only for the one-time USB setup |
| OpenSSH | built into Windows 10/11 (`ssh -V` in PowerShell) | everything else |
| WSL2 + Ubuntu 22.04 (optional) | `wsl --install -d Ubuntu-22.04` | ROS 2 bag replay on the laptop (Step 11) |

Clone the repo, then check the `.gitattributes` at its root is there: it keeps shell scripts and configs
in Linux line endings. Without it, Git on Windows converts them and a script copied to the drone fails
with `/bin/sh^M: not found`.

Windows paths: commands in the repo's other docs like `~/venvs/yolo/bin/python` are Mac/Linux. On
Windows the venv's Python is `.\venv\Scripts\python.exe`. On the drone and in WSL they work as written.

**Check:** `git --version`, `python --version` (3.12) and `ssh -V` all print a version.

## Step 1: Wi‑Fi on the drone (USB, once)

Plug the drone into the laptop with USB‑C and power it on. In PowerShell:
```powershell
cd C:\platform-tools
.\adb.exe devices          # the drone must be listed as "device"
.\adb.exe shell           # a root shell on the drone
```

On the drone:
```bash
ip link                   # need a wlan0 entry; none = no Wi-Fi hardware, get ModalAI's Wi-Fi add-on
voxl-wifi                 # prints its usage; the two modes below
```

Pick a mode:

| Mode | Command (on the drone) | Use it for |
|---|---|---|
| **Station**: the drone joins a network | `voxl-wifi station <SSID> <password>` | development: the drone gets internet (Docker images, pip). Use a phone hotspot or your own router. **Not eduroam**: it needs a per-user login the drone can't do |
| **SoftAP**: the drone is the hotspot | `voxl-wifi softap <SSID>` | the flying field: no router needed, but no internet on the drone |

Reboot (`reboot`), wait about a minute, then find the drone's address:
```powershell
.\adb.exe shell ip addr show wlan0        # the "inet" line, e.g. 192.168.43.57
```
In softAP mode it is usually `192.168.8.1`, with your laptop joined to the drone's network.

**Check:** with the laptop on the same network, `ping <drone-ip>` answers. Unplug the USB cable: from
here on everything is over Wi‑Fi. (To change Wi‑Fi mode later, do it over SSH; if you lock yourself out,
USB + adb still works.)

## Step 2: SSH from PowerShell

```powershell
ssh root@<drone-ip>        # ModalAI's default password is oelinux123
passwd                     # on the drone: change it, and tell the team
exit
```

A key, so you stop typing the password (Windows has no `ssh-copy-id`; this does the same):
```powershell
ssh-keygen -t ed25519      # Enter for every question
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh root@<drone-ip> "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

A short name: create `C:\Users\<you>\.ssh\config` (no extension) with
```
Host starling
    HostName <drone-ip>
    User root
```
In station mode the address can change after a reboot: give the drone a fixed address in your
router/hotspot if it can, or update `HostName`.

**Check:** `ssh starling` logs in with no password. `http://<drone-ip>` in the browser shows VOXL Portal.

## Step 3: look around the drone (nothing changes)

Over `ssh starling`:
```bash
voxl-inspect-services                  # which ModalAI services run, enabled at boot or not
voxl-list-pipes                        # every data stream (MPA pipe) on the drone
voxl-inspect-imu imu_apps              # IMU at full rate; Ctrl-C to stop
voxl-inspect-cam hires_front_small_color   # the camera our gate vision uses (1024x768)
voxl-inspect-qvio                      # the VIO; if nothing, look for a vio / ov pipe in voxl-list-pipes
voxl-cpu-monitor                       # CPU load per core
df -h /data                            # free space: Docker images and bags go on /data
```

Walk the drone slowly around the room with the VIO inspector running and watch the position.

**Check:** you can write down (1) which VIO runs and at what rate, (2) the exact name of the hires camera
pipe, (3) the CPU load at idle, (4) free space on `/data`. Put them in a notes file: they are what every
later step plugs in.

## Step 4: ROS 2 on the drone

```bash
ls /opt/ros                            # expect foxy
source /opt/ros/foxy/setup.bash
voxl-inspect-services | grep mpa-to-ros2
systemctl enable --now voxl-mpa-to-ros2    # only if it wasn't running
ros2 topic list
ros2 topic hz <the hires image topic>
```

**Check:** the camera, IMU and VIO topics are listed with their rates. Add the topic names to the notes:
our nodes take them as parameters (they default to Isaac's `/iris_0/...` and `/drone00/...`).

## Step 5: Docker on the drone

```bash
docker --version
docker info | grep -i "root dir"       # should be on /data, which has the space
docker run --rm arm64v8/hello-world    # needs internet: station mode
```

The ROS version test (our code was tested on ROS 2 Humble; the drone publishes Foxy):
```bash
docker run --rm -it --network host --ipc host ros:humble bash
# inside the container:
source /opt/ros/humble/setup.bash
ros2 topic list
ros2 topic hz <the hires image topic>
exit
```

`--network host` and `--ipc host` let the container see the drone's ROS topics exactly as the drone does.

**Check:** the container sees the topics, with images at their full rate. If `topic list` is empty or
`hz` shows nothing, repeat with `ros:foxy` and note which one works: that is the base for our image.
Also note whether `docker compose version` works (the VOXL's Docker may not have it).

## Step 6: the repo on the drone

If the drone can reach GitHub (station mode, and the repo is public or you have access):
```bash
cd /data && git clone <repo URL> repo && cd repo && git checkout main
```
Otherwise copy just what the drone needs from the laptop, in PowerShell from the repo root:
```powershell
ssh starling "mkdir -p /data/repo"
scp -r vision ros2_nodes control Trajectory drone_msgs starling:/data/repo/
```

**Check:** `ssh starling ls /data/repo/vision/calib` shows `hires_front_1024x768.yml`.

## Step 7: export YOLO to ONNX (on the laptop)

The drone has no Mac GPU and no PyTorch, so the YOLO model goes to ONNX, a plain format that
`onnxruntime` runs on the drone's CPU. In PowerShell from the repo root:
```powershell
python -m venv venv
.\venv\Scripts\pip install ultralytics onnx onnxruntime onnxslim
cd vision\yolo_gate\weights
..\..\..\venv\Scripts\yolo export model=pink_det_v1_yolo11n.pt format=onnx imgsz=480,640
ren pink_det_v1_yolo11n.onnx pink_det_v1_480x640.onnx
..\..\..\venv\Scripts\yolo export model=pink_det_v1_yolo11n.pt format=onnx imgsz=240,320
ren pink_det_v1_yolo11n.onnx pink_det_v1_240x320.onnx
cd ..\..\..
```
Every export writes the same `pink_det_v1_yolo11n.onnx`, hence the rename after each. `480,640` is
height, width: it fits the 1024x768 frame with no padding (the input the `.pt` gets in
`live_gate_test.py`); `240,320` is about 3x less work.

Check the export against the `.pt` on recorded frames (any run's `raw_jpg/`, if the laptop has one):
```powershell
.\venv\Scripts\python vision\yolo_gate\bench_yolo.py --model vision\yolo_gate\weights\pink_det_v1_480x640.onnx --images <a raw_jpg folder> --n 100 --compare-pt vision\yolo_gate\weights\pink_det_v1_yolo11n.pt
```

Measured on a Mac (M-series CPU, 2026-11-18), 100 frames of `hybrid_live_01`:

| Model input | YOLO time per frame (median) | Frames with a box | `.pt` vs `.onnx` |
|---|---|---|---|
| 640x640 | 34 ms (29 fps) | 67 % | identical |
| 480x640 | 25 ms (40 fps) | 73 % | identical (largest box difference 7 px on 1 frame) |
| 240x320 | 8 ms (130 fps) | 83 % | identical (largest 1.7 px) |

More boxes at 240x320 is not automatically better: some may be false. Judge sizes by the hybrid
detector's corners and ranges (Step 9), not by box count. The VOXL's CPU will be slower than the Mac's:
Step 8 measures by how much.

**Check:** the compare line says the box / no-box decisions agree on (nearly) every frame. Copy the
models to the drone:
```powershell
scp vision\yolo_gate\weights\*.onnx starling:/data/repo/vision/yolo_gate/weights/
scp vision\yolo_gate\bench_yolo.py vision\yolo_gate\tflite_gpu.py starling:/data/repo/vision/yolo_gate/
```

## Step 8: benchmark YOLO on the drone

A slim container: Python, numpy, OpenCV and onnxruntime only.
```bash
ssh starling
docker run --rm -it --network host --ipc host --cpuset-cpus 4-6 -v /data/repo:/repo -w /repo ubuntu:22.04 bash
# inside:
apt update && apt install -y python3-pip
pip3 install numpy opencv-python-headless onnxruntime
cd vision/yolo_gate
python3 bench_yolo.py --model weights/pink_det_v1_480x640.onnx --n 100 --threads 3
python3 bench_yolo.py --model weights/pink_det_v1_240x320.onnx --n 100 --threads 3
```
`--cpuset-cpus 4-6` keeps the container on three of the big cores (the QRB5165's cores 4–6 are fast,
7 is the fastest, 0–3 are slow: check against `voxl-cpu-monitor`). Core 7 stays free for the MPC later.
Run it once with the drone idle and once with the VIO running, which is how it will fly.

Without `--images` the benchmark uses random frames: timing only. For detections on the drone, copy a
`raw_jpg/` folder over and pass `--images`.

This is the CPU. The GPU (Adreno 650, through OpenCL + TFLite) has its own container:
`docker/voxl2-tflite-gpu/README.md`. Benchmark both and keep the faster one.

**Check:** write down fps for both sizes. About 10 fps or more with the VIO running: YOLO on the drone
works, go on with it. Below that: run the color detector alone on the drone (Step 9a) and treat YOLO as
an upgrade.

## Step 9: run the gate vision on the drone

The same `live_gate_test.py` as on the laptop, but inside the container on the drone, reading VOXL
Portal at `localhost` (no `--show`: there is no screen; watch Portal in the laptop's browser instead).

**9a. Color only** (numpy + OpenCV, works in the Step 8 container as is):
```bash
cd /repo/vision
python3 live_gate_test.py --detector color --url http://localhost/video_raw/hires_front_small_color \
    --hsv-lo 0 80 50 --hsv-hi 12 255 255 --close-px 31 --gate-side 0.8636 \
    --calib calib/hires_front_1024x768.yml --duration 120 --out-dir live_runs/onboard_color_01
```

**9b. Hybrid** (YOLO box, then color corners), with the exported model: no Ultralytics or PyTorch
needed. The `.onnx` runs in the Step 8 container as is; for the `.tflite` on the GPU, use the
`docker/voxl2-tflite-gpu` container (its README, step 8):
```bash
python3 live_gate_test.py --detector hybrid --model yolo_gate/weights/pink_det_v1_480x640.onnx \
    --threads 3 --gate-side 0.8636 --calib calib/hires_front_1024x768.yml \
    --url http://localhost/video_raw/hires_front_small_color --duration 120 --out-dir live_runs/onboard_hybrid_01
```

Do the tape-measure test while it runs: gate face-on at 1, 2, 3 and 4 m, then turned about 20° and
40°, a few seconds each, and note the times.

Copy the results back (PowerShell):
```powershell
scp -r starling:/data/repo/vision/live_runs/onboard_color_01 vision\live_runs\
```

**Check:** the script's fps on the drone; ranges within a few cm of the tape; results written up in
the run's `README.md` like the earlier runs. The script prints its fps when it ends.

## Step 10: what's left before vision flies (code, not setup)

- **Gate node:** `ros2_nodes/ros_camera_pose_fallback.py` still assumes Isaac. It needs the hybrid
  detector on onnxruntime, fisheye undistortion from `hires_front_1024x768.yml`, the real Starling camera
  mount, the VOXL topic names from Steps 3–4, and gate positions from the VIO pose at each image's time.
- **Image:** a Dockerfile with the base from Step 5 and the packages from Step 8, built on the drone
  (`docker build` in `/data/repo`: the drone is ARM, so it builds natively there; building on an x86
  Windows laptop needs emulation and is slow).
- **Start on boot:** a systemd service on the drone that runs the container.

## Step 11 (optional): WSL2 for replaying data on the laptop

Recorded ROS 2 bags (`ros2 bag record` on the drone, copied back with `scp`) can be replayed on the
laptop in WSL2 with ROS 2 Humble on Ubuntu 22.04, to test the gate node without the drone.

To see the drone's **live** topics from WSL2, WSL must share Windows' network. On Windows 11, create
`C:\Users\<you>\.wslconfig` with
```
[wsl2]
networkingMode=mirrored
```
then `wsl --shutdown` and reopen Ubuntu. Windows Firewall may also need to allow ROS 2's UDP traffic.
Bag replay needs none of this.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `adb devices` shows nothing | another USB‑C cable (some are charge-only); drone fully booted? |
| `ssh: connect ... timed out` | laptop and drone on different networks, or the drone's address changed (Step 1) |
| `docker run` can't pull images | drone in softAP mode (no internet): switch to station mode |
| Humble container sees no topics | `--network host --ipc host` missing, or a Foxy/Humble mismatch: try `ros:foxy` (Step 5) |
| `/bin/sh^M: not found` | a script got Windows line endings: check `.gitattributes`, re-clone or `dos2unix` the file |
| Copying a Docker image through `docker save ... \| ssh ...` gives a corrupt image | PowerShell 5 breaks binary pipes: `docker save -o x.tar`, `scp`, then `docker load -i` on the drone. Better: build on the drone |
| `live_gate_test.py` gets no frames on the drone | the pipe name in `--url` (Step 3), or VOXL Portal not running (`voxl-inspect-services`) |
