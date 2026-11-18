#!/bin/bash
# Run ON THE VOXL 2: start the container with the GPU passed in (only the two GPU device nodes, not
# --privileged), the drone's network and shared memory (ROS 2 topics, VOXL Portal on localhost),
# and the repo mounted at /repo.
#
#   ./run.sh                                  # a shell
#   ./run.sh opencl_check                     # any command
#   CPUS=4-7 REPO=/data/repo ./run.sh ...     # other cores / repo location
set -euo pipefail

IMAGE="${IMAGE:-voxl2-tflite-gpu:latest}"
REPO="${REPO:-/data/repo}"
CPUS="${CPUS:-4-6}"   # big cores; 7 (the fastest) stays free for the MPC

if [ $# -eq 0 ]; then set -- bash; fi

exec docker run --rm -it \
  --device /dev/kgsl-3d0 --device /dev/ion \
  --network host --ipc host \
  --cpuset-cpus "$CPUS" \
  -v "$REPO":/repo -w /repo \
  "$IMAGE" "$@"
