#!/bin/bash
# Pink 36 in gate matrix on the Isaac Starling 2 Max (ros2-eval container).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DRONE_VEHICLE=starling2max
cd /workspace/race/rl/isaac
OUT=../runs/isaac/max_pink2; mkdir -p $OUT
for t in straight3 oval6 slalom3 slalom5; do
  printf 'import os\nos.environ["SPAWN_TRACK"] = "%s_pink36"\nexec(open("/workspace/race/workspace/spawn_track.py").read(), {"__name__": "__main__", "__file__": "/workspace/race/workspace/spawn_track.py"})\n' $t > /workspace/race/.q.tmp
  mv /workspace/race/.q.tmp /workspace/race/isaac_next.py; sleep 5
  timeout 200 /workspace/venvs/race/bin/python recover.py 0 0 2.5 >> $OUT/recover.txt 2>&1
  for c in pid mpc_ctbr; do for r in 1 2 3; do
    timeout 300 /workspace/venvs/race/bin/python runner.py --track ../tracks/${t}_pink36.json --controller $c \
      --origin-offset 0 0 0.05 --thrust-scale 1.0 --max-time 90 --out-dir $OUT/${c}_${t}_r$r > $OUT/${c}_${t}_r$r.txt 2>&1
    echo "$(tail -2 $OUT/${c}_${t}_r$r.txt | head -1)  [$c r$r]" | tee -a $OUT/matrix.txt
  done; done
done
echo DONE | tee -a $OUT/matrix.txt
