#!/bin/bash
# Run ON THE VOXL 2 (as root), in this folder, before `docker build`.
# Bundles the drone's Adreno GPU user-space driver (OpenCL) and the Qualcomm libraries it needs into
# deps/gpu-driver.tar, which the Dockerfile unpacks into the image.
#
# ModalAI's "OpenCL in Docker" recipe repacks these packages as .debs with dpkg-repack. Our drone has
# no dpkg-repack (installing it needs internet), and the packages' architectures (m0054, aarch64)
# would not install in an arm64 Ubuntu image anyway. So this takes the installed files straight from
# the package lists (dpkg -L): no extra tools, no internet. Checked on our Starling 2 Max
# (SDK 1.5.1, 2026-11-18): libOpenCL.so loads libCB, libgsl, libq3dtools_adreno and libllvm-qcom, all
# in qti-adreno, which need libcutils, liblog, libsync and libion (the other four packages) plus
# glib, pcre and zlib (installed from Ubuntu in the Dockerfile).
#
# Re-run after a VOXL SDK upgrade: the driver in the image must match the drone's kernel.
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

rm -rf "$OUT" && mkdir -p "$OUT"
LIST="$OUT/files.txt"
# Files and symlinks only (no directories); not the GPU firmware (the drone's kernel loads it, the
# container never does), udev rules or test binaries.
dpkg -L "${PKGS[@]}" | while read -r f; do
  case "$f" in /lib/firmware/*|/etc/*|/usr/bin/*) continue ;; esac
  if [ -L "$f" ] || [ -f "$f" ]; then echo "${f#/}"; fi
done | sort -u > "$LIST"

tar -C / -cf "$OUT/gpu-driver.tar" -T "$LIST"
echo "$(wc -l < "$LIST") files from ${PKGS[*]} -> $OUT/gpu-driver.tar ($(du -h "$OUT/gpu-driver.tar" | cut -f1))"
grep -q '^usr/lib/libOpenCL.so$' "$LIST" || { echo "libOpenCL.so missing from the bundle" >&2; exit 1; }

echo
echo "GPU device nodes (run.sh passes these to the container):"
ls -la /dev/kgsl-3d0 /dev/ion
