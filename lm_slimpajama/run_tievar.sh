#!/bin/bash
# run_tievar.sh - tied-readout ablation arms (340M, K=2x2, 4 GPUs, 1e9 tokens) via slim_pipeline_tievar.py.
#   random: frozen random readout (scale-corrected)
#   affine: tied readout + identity-initialized learnable adapter
#   untie_affine: private readout + the same adapter
#   stale: frozen readout loaded from STALE_FC, no broadcast
# Usage: bash run_tievar.sh <ARM random|affine|untie_affine|stale> [GPUS] [PORT] [SEED]
#   bash run_tievar.sh random 0,1,2,3 30051
#   bash run_tievar.sh affine 4,5,6,7 30053
# Overrides: BUDGET, BIN, STALE_FC (default ckpt/untie_tie_s42.pt.r1), TAG_SUFFIX.
# Reference (s42, 1e9 tok, K=2x2, terminal val ppl): tie 27.60 / untie 28.01.
set -e
cd "$(dirname "$0")"
ARM=${1:?random|affine|untie_affine|stale}; G=${2:-0,1,2,3}; PORT=${3:-30051}; SEED=${4:-42}
case "$ARM" in random|affine|untie_affine|stale) ;; *) echo "ARM must be random|affine|untie_affine|stale"; exit 1;; esac
STALE_FC=${STALE_FC:-ckpt/untie_tie_s42.pt.r1}
[ "$ARM" = stale ] && { [ -f "$STALE_FC" ] || { echo "missing $STALE_FC (set STALE_FC=... to override)"; exit 1; }; }
BUDGET=${BUDGET:-1000000000}
BIN=${BIN:-/dev/shm/slim/slim_15b.bin}
[ -f "$BIN" ] || { echo "missing $BIN (override with BIN=data/slim_15b.bin)"; exit 1; }
TAG=tievar_${ARM}${TAG_SUFFIX:-}_s${SEED}
L=/tmp/slim_logs; mkdir -p $L ckpt

export MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT WORLD_SIZE=4 K=2 \
  CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8 MKL_THREADING_LAYER=GNU \
  NLAYER=24 DMODEL=1024 NHEAD=16 FFN=0 SEQ=2048 BS=8 ACCUM=16 CH=8192 \
  LR=3e-4 LR_MIN=3e-5 WARMUP_TOKENS=5e8 WD=0.01 CLIP=1.0 SEED=$SEED \
  BIN=$BIN TOKENS=$BUDGET COMPILE=2 FAKE_DATA=0 STEPS=0 \
  LOSS=md TIE_MODE=$ARM STALE_FC=$STALE_FC EVAL_EVERY=200 SAVE_EVERY=100000 \
  OUT=ckpt/$TAG.pt RESUME=ckpt/$TAG.pt
for r in 0 1 2 3; do
  RANK=$r LOCAL_RANK=$r nohup python slim_pipeline_tievar.py \
    > $L/${TAG}_r$r.log 2>&1 &
done
echo "launched $TAG (GPUs $G, port $PORT, budget=$BUDGET, TIE_MODE=$ARM)"
echo "  fingerprint: head -3 $L/${TAG}_r0.log"
echo "  curve: grep -a 'eval @' $L/${TAG}_r1.log | tail -2"
