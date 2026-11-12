#!/bin/sh
# usage: xs.sh <xtype args...>  ; always ends with a screenshot to $S/s.png (scaled to 1440 wide crop of kit window)
S=${SCRATCH:-/tmp}
export KUBECONFIG=~/Desktop/FAU/EGN1/kube/jup19-rke2-caai.yaml
kubectl -n fake-isaac exec deploy/isaac-sim -c isaac-sim -- env PYTHONPATH=/workspace/pylib /isaac-sim/kit/python/bin/python3 /tmp/xtype.py "$@" ";;" sleep 0.6 ";;" shot /workspace/race/tmp/s.ppm 0 0 1440 900 >/dev/null && \
kubectl -n fake-isaac exec deploy/isaac-sim -c isaac-sim -- cat /workspace/race/tmp/s.ppm > $S/s.ppm && \
/usr/local/bin/python3.12 -c "import cv2;cv2.imwrite('$S/s.png',cv2.imread('$S/s.ppm'))"
