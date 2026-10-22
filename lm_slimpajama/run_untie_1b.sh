#!/bin/bash
# run_untie_1b.sh - 1.3B tie/untie control (UNTIE switch of slim_pipeline_untied.py): 24L d2048, K=2xR=2, 15B tokens.
# Data must be in /dev/shm (memmap over NFS can raise SIGBUS).
# Usage: bash run_untie_1b.sh <tie|untie> [GPUS] [PORT] [SEED]
#   bash run_untie_1b.sh tie   0,1,2,3 29881
#   bash run_untie_1b.sh untie 4,5,6,7 29883
# Overrides: BIN, TIE_MODE (default trained), STALE_FC (required for TIE_MODE=stale), TAG_SUFFIX.
# Reference: tie arm Wiki ppl ~22.29.
set -e
cd "$(dirname "$0")"
ARM=${1:?tie|untie}; G=${2:-0,1,2,3}; PORT=${3:-29881}; SEED=${4:-42}
[ "$ARM" = tie ] || [ "$ARM" = untie ] || { echo "ARM must be tie or untie"; exit 1; }
U=$([ "$ARM" = untie ] && echo 1 || echo 0)
[ "${TIE_MODE:-}" = stale ] && { [ -f "${STALE_FC:-}" ] || { echo "TIE_MODE=stale requires STALE_FC=<existing ckpt>"; exit 1; }; }
BIN=${BIN:-/dev/shm/slim/slim_15b.bin}
[ -f "$BIN" ] || { echo "missing $BIN - first cp data/slim_15b.bin{,.meta.json} /dev/shm/slim/"; exit 1; }
TAG=untie1b_${ARM}${TAG_SUFFIX:-}_s${SEED}
L=/tmp/slim_logs; mkdir -p $L ckpt

export MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT WORLD_SIZE=4 K=2 \
  CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8 MKL_THREADING_LAYER=GNU \
  ARCH=llama NLAYER=24 DMODEL=2048 NHEAD=16 FFN=0 SEQ=2048 V=32000 \
  BS=8 ACCUM=16 CH=8192 LR=3e-4 LR_MIN=3e-5 WARMUP_TOKENS=5e8 WD=0.01 CLIP=1.0 \
  TOKENS=15000000000 SEED=$SEED COMPILE=2 FAKE_DATA=0 STEPS=0 \
  LOSS=md UNTIE=$U TIE_MODE=${TIE_MODE:-trained} STALE_FC=${STALE_FC:-} EVAL_EVERY=1000 SAVE_EVERY=2500 \
  BIN=$BIN CKPT_STAGE_DIR=/dev/shm OUT=ckpt/$TAG.pt RESUME=ckpt/$TAG.pt
for r in 0 1 2 3; do
  RANK=$r LOCAL_RANK=$r taskset -c $((r*16))-$((r*16+15)) nohup python slim_pipeline_untied.py \
      > $L/${TAG}_r$r.log 2>&1 &
done
echo "launched $TAG (GPUs $G, port $PORT, UNTIE=$U); ~20h"
echo "  check: head -3 $L/${TAG}_r0.log (untie non-terminal rank params should be 65.5M larger); tie arm should converge to ~22.29"
echo "  curve: grep -a 'eval @' $L/${TAG}_r1.log | tail -3"
