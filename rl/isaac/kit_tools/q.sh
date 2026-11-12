#!/bin/sh
# q.sh FILE : queue a script for the Isaac poller (atomic)
export KUBECONFIG=~/Desktop/FAU/EGN1/kube/jup19-rke2-caai.yaml
kubectl -n fake-isaac exec -i deploy/isaac-sim -c isaac-sim -- sh -c 'cat > /workspace/race/.q.tmp && mv /workspace/race/.q.tmp /workspace/race/isaac_next.py' < "$1"
