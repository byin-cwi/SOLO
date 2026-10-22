#!/bin/bash
# run_ablate_md.sh - A/B ablation of the milestone loss: LOSS=md (SOLO milestone loss) vs LOSS=ce (greedy local CE).
#   The arms differ only in the local objective of non-terminal stages. 340M, K=2xR=2, 4 GPUs.
# Usage: bash run_ablate_md.sh <ARM md|ce> [GPUS] [PORT] [SEED]
#   bash run_ablate_md.sh md 0,1,2,3 30021
#   bash run_ablate_md.sh ce 4,5,6,7 30023     # the two arms can run concurrently on different GPU groups
# Overrides: BUDGET (tokens, default 1e9), BIN (default /dev/shm/slim/slim_15b.bin).
# Reference (full 15B, 340M K=2): SOLO terminal bin-val 12.97 / wiki 29.05.
set -e
cd "$(dirname "$0")"
ARM=${1:?md|ce}; G=${2:-0,1,2,3}; PORT=${3:-30021}; SEED=${4:-42}
[ "$ARM" = md ] || [ "$ARM" = ce ] || { echo "ARM must be md or ce"; exit 1; }
BUDGET=${BUDGET:-1000000000}
BIN=${BIN:-/dev/shm/slim/slim_15b.bin}
[ -f "$BIN" ] || { echo "missing $BIN"; exit 1; }
TAG=ablmd_${ARM}_s${SEED}
L=/tmp/slim_logs; mkdir -p $L ckpt

export MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT WORLD_SIZE=4 K=2 \
  CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8 MKL_THREADING_LAYER=GNU \
  NLAYER=24 DMODEL=1024 NHEAD=16 FFN=0 SEQ=2048 BS=8 ACCUM=16 CH=8192 \
  LR=3e-4 LR_MIN=3e-5 WARMUP_TOKENS=5e8 WD=0.01 CLIP=1.0 SEED=$SEED \
  BIN=$BIN TOKENS=$BUDGET COMPILE=2 FAKE_DATA=0 STEPS=0 \
  LOSS=$ARM EVAL_EVERY=200 SAVE_EVERY=100000 \
  OUT=ckpt/$TAG.pt RESUME=ckpt/$TAG.pt
for r in 0 1 2 3; do
  RANK=$r LOCAL_RANK=$r nohup python slim_pipeline_solo.py \
    > $L/${TAG}_r$r.log 2>&1 &
done
echo "launched $TAG (GPUs $G, port $PORT, budget=$BUDGET)"
echo "  curve: grep -a 'eval @' $L/${TAG}_r1.log"
echo "  check: first line prints LOSS=$ARM; ARM=md log has tight/act certificate, empty for ARM=ce"
