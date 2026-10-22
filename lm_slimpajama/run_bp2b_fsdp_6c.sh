#!/bin/bash
# run_bp2b_fsdp_6c.sh - 2B BP baseline (FSDP FULL_SHARD, 24L/d2560/20 heads), 15B tokens, from scratch on 6 GPUs.
# Effective batch = NG x BS8 x ACCUM x SEQ2048 (6 GPUs, ACCUM=8: 786,432 tok/step). FSDP shards are tied to the GPU count.
# Usage: [ACCUM=8] bash run_bp2b_fsdp_6c.sh [GPUS] [PORT] [SEED]     # default 2,3,4,5,6,7  30210  42
# Overrides: ACCUM, BIN.
set -e
cd "$(dirname "$0")"
G=${1:-2,3,4,5,6,7}; PORT=${2:-30210}; SEED=${3:-42}
ACCUM=${ACCUM:-8}
TAG=bp2b_fsdp6
BIN=${BIN:-/dev/shm/slim/slim_15b.bin}
[ -f "$BIN" ] || { echo "missing $BIN - first: mkdir -p /dev/shm/slim && cp data/slim_15b.bin /dev/shm/slim/"; exit 1; }
FREE_SHM=$(df -BG /dev/shm | awk 'NR==2{gsub("G","",$4); print $4}')
[ "$FREE_SHM" -ge 40 ] || { echo "/dev/shm has only ${FREE_SHM}GB free < 40GB, not enough to stage checkpoints"; exit 1; }
NG=$(echo "$G" | awk -F, '{print NF}')
EFF=$(( 8 * ACCUM * 2048 * NG ))
L=/tmp/slim_logs; mkdir -p "$L" ckpt
echo "[cfg] $TAG  GPUs=$G (NG=$NG)  ACCUM=$ACCUM  eff_batch=$EFF tok/step  steps≈$(( 15000000000 / EFF ))"

for r in $(seq 0 $((NG-1))); do
  env -i HOME="$HOME" PATH="$PATH" \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_THREADING_LAYER=GNU \
    CUDA_VISIBLE_DEVICES=$G MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT WORLD_SIZE=$NG RANK=$r LOCAL_RANK=$r \
    COMPILE=1 FSDP_SHARD=full FSDP_MP=1 ARCH=llama \
    NLAYER=24 DMODEL=2560 NHEAD=20 FFN=0 SEQ=2048 BS=8 ACCUM=$ACCUM \
    LR=3e-4 LR_MIN=3e-5 WARMUP_TOKENS=5e8 WD=0.01 CLIP=1.0 DROP=0.0 TIE=0 SEED=$SEED \
    EVAL_EVERY=1000 SAVE_EVERY=2500 BIN=$BIN TOKENS=15000000000 \
    OUT=ckpt/$TAG.pt RESUME=ckpt/$TAG.pt CKPT_STAGE_DIR=/dev/shm \
    python -u slim_bp_fsdp.py > "$L/${TAG}_r$r.log" 2>&1 < /dev/null &
done
echo "launched $TAG (GPUs $G, port $PORT, WORLD_SIZE=$NG, eff_batch=$EFF) — tail -f $L/${TAG}_r0.log"
echo "  check: 'FSDP wrap: shard=full' ×$NG · params≈2067M · from scratch (no 'resumed @ step' line in log)"
wait
