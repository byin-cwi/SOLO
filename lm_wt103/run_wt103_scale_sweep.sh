#!/bin/bash
# run_wt103_scale_sweep.sh - WT-103 scale sweep: runs run_wt103_scalept.sh at 64M, 256M, 128M, 340M, 512M in that order
#   (40M is covered by run_wt103_ladder.sh). Finished arms and extracts are skipped, so rerunning resumes.
# Usage: GPUS=4,5,6,7 nohup bash run_wt103_scale_sweep.sh > logs_scalept/sweep_driver.log 2>&1 &
# Knobs: GPUS
# Reference: 40M random readout val ppl 98.88.
cd "$(dirname "$0")"
GPUS=${GPUS:-0,1,2,3}
mkdir -p logs_scalept ckpt_scalept

run () {   # $1=NLAYER $2=DMODEL $3=NHEAD $4=TAGPFX
  echo "======== [$(date '+%F %T')] SCALE $4  (NLAYER=$1 DMODEL=$2 NHEAD=$3) ========"
  GPUS="$GPUS" NLAYER="$1" DMODEL="$2" NHEAD="$3" TAGPFX="$4" bash run_wt103_scalept.sh \
    || echo "[warn] $4 exited non-zero (possibly OOM); continuing with next scale, rerun this script to resume. If a large model OOMs, rerun that scale with BS=32."
}

# 1) 64M / 256M
run 12 512  8  wt103_64m
run 16 1024 16 wt103_256m
# 2) remaining scales (small -> large)
run 16 768  12 wt103_128m
run 24 1024 16 wt103_340m
run 24 1280 16 wt103_512m

echo "[$(date '+%F %T')] SWEEP_ALL_DONE"
echo "==== random FINAL summary across scales ===="
grep -a FINAL logs_scalept/*_random_r3.log 2>/dev/null | sed 's#logs_scalept/##; s/_random_r3.log:/ /'
echo "==== lstale vs bpstale per scale (co-adaptation) ===="
for s in 64m 128m 256m 340m 512m; do
  l=$(grep -a FINAL logs_scalept/wt103_${s}_lstale_r3.log 2>/dev/null | grep -oE 'val ppl: [0-9.]+' | tail -1)
  b=$(grep -a FINAL logs_scalept/wt103_${s}_bpstale_r3.log 2>/dev/null | grep -oE 'val ppl: [0-9.]+' | tail -1)
  echo "  $s: lstale [$l]  vs  bpstale [$b]   (lstale<bpstale => reproduced)"
done
