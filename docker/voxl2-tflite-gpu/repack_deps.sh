#!/bin/bash
# Run ON THE VOXL 2 (as root), in this folder, before `docker build`.
# Repacks the drone's installed Adreno GPU / OpenCL user-space driver and the Qualcomm libraries it
# needs into deps/*.deb, which the Dockerfile installs in the image (ModalAI's "OpenCL in Docker"
# recipe). Re-run after a VOXL SDK upgrade: the driver in the image must match the drone's kernel.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/deps"
PKGS=(qti-adreno libcutils0 libsync qti-libion liblog0)

for p in "${PKGS[@]}"; do
  if ! dpkg -s "$p" >/dev/null 2>&1; then
    echo "package $p is not installed on this VOXL: check the name with 'dpkg -l | grep -i $p'" >&2
    exit 1
  fi
done

if ! command -v dpkg-repack >/dev/null; then
  apt-get update && apt-get install -y dpkg-repack
fi

rm -rf "$OUT" && mkdir -p "$OUT"
cd "$OUT"
dpkg-repack "${PKGS[@]}"
ls -la "$OUT"

echo
echo "GPU device nodes (run.sh passes these to the container):"
ls -la /dev/kgsl-3d0 /dev/ion
