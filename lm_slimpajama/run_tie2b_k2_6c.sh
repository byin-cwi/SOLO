#!/bin/bash
# run_tie2b_k2_6c.sh - 2B SOLO K=2 with tied readout (K=2 stages x R=3 replicas = 6 GPUs), 15B tokens, from scratch.
#   Runs slim_pipeline_tievar.py with TIE_MODE=trained; same model, GPUs, eff_batch (786,432) and tokens as run_bp2b_fsdp_6c.sh.
#   TIE_SYNC = readout broadcast period in optimizer steps (1 = every step).
# Usage: [TIE_SYNC=100] [SAVE_EVERY=1000] bash run_tie2b_k2_6c.sh [GPUS] [PORT] [SEED]   # default 2-7  30220  42
# Overrides: TIE_SYNC (default 100), SAVE_EVERY, BIN.
set -e
cd "$(dirname "$0")"
G=${1:-2,3,4,5,6,7}; PORT=${2:-30220}; SEED=${3:-42}
TIE_SYNC=${TIE_SYNC:-100}; SAVE_EVERY=${SAVE_EVERY:-1000}
TAG=tie2b_k2_ts${TIE_SYNC}_6c
BIN=${BIN:-/dev/shm/slim/slim_15b.bin}
[ -f "$BIN" ] || { echo "missing $BIN - first: mkdir -p /dev/shm/slim && cp data/slim_15b.bin /dev/shm/slim/"; exit 1; }
FREE_SHM=$(df -BG /dev/shm | awk 'NR==2{gsub("G","",$4); print $4}')
[ "$FREE_SHM" -ge 25 ] || { echo "/dev/shm has only ${FREE_SHM}GB free < 25GB, not enough to stage shards"; exit 1; }
NG=$(echo "$G" | awk -F, '{print NF}'); K=2; R=$(( NG / K ))
[ $(( K * R )) -eq "$NG" ] || { echo "NG=$NG is not a multiple of K=$K (6 GPUs -> R=3)"; exit 1; }
EFF=$(( 8 * 16 * 2048 * R ))
L=/tmp/slim_logs; mkdir -p "$L" ckpt
echo "[cfg] $TAG  GPUs=$G (NG=$NG = K${K}×R${R})  eff_batch=$EFF  TIE_SYNC=$TIE_SYNC  SAVE_EVERY=$SAVE_EVERY  steps≈$(( 15000000000 / EFF ))"

export MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT WORLD_SIZE=$NG K=$K \
  CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8 MKL_THREADING_LAYER=GNU \
  NLAYER=24 DMODEL=2560 NHEAD=20 FFN=0 SEQ=2048 BS=8 ACCUM=16 CH=8192 \
  LR=3e-4 LR_MIN=3e-5 WARMUP_TOKENS=5e8 WD=0.01 CLIP=1.0 SEED=$SEED \
  BIN=$BIN TOKENS=15000000000 COMPILE=2 FAKE_DATA=0 STEPS=0 \
  LOSS=md TIE_MODE=trained TIE_SYNC=$TIE_SYNC EVAL_EVERY=1000 SAVE_EVERY=$SAVE_EVERY \
  OUT=ckpt/$TAG.pt RESUME=ckpt/$TAG.pt CKPT_STAGE_DIR=/dev/shm
for r in $(seq 0 $((NG-1))); do
  RANK=$r LOCAL_RANK=$r nohup python slim_pipeline_tievar.py > "$L/${TAG}_r$r.log" 2>&1 &
done
echo "launched $TAG (GPUs $G, port $PORT, WORLD_SIZE=$NG, K=$K, R=$R, TIE_SYNC=$TIE_SYNC)"
echo "  check: head -3 $L/${TAG}_r0.log   (expect K=2, R=$R, eff_batch=$EFF)"
echo "  curve: grep -a 'eval @' $L/${TAG}_r1.log | tail   (terminal stage = stage1 of replica 0 = rank1)"
wait
