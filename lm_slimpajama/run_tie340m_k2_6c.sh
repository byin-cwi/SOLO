#!/bin/bash
# run_tie340m_k2_6c.sh - 340M SOLO K=2 with tied readout (K=2 stages x R replicas, default 6 GPUs), from scratch.
#   Runs slim_pipeline_tievar.py with TIE_MODE=trained; eff_batch = 8x16x2048xR (786,432 at R=3).
#   TIE_SYNC = readout broadcast period in optimizer steps (1 = every step).
# Usage: [TIE_SYNC=1] [BUDGET=15000000000] [SAVE_EVERY=1000] bash run_tie340m_k2_6c.sh [GPUS] [PORT] [SEED]
#   TIE_SYNC=1 bash run_tie340m_k2_6c.sh 2,3,4,5,6,7 30230 42
#   TIE_SYNC=1 BUDGET=1000000000 bash run_tie340m_k2_6c.sh 2,3,4,5,6,7 30230 42
#   TIE_SYNC=100 bash run_tie340m_k2_6c.sh 2,3,4,5,6,7 30232 42
# Overrides: TIE_SYNC, BUDGET, SAVE_EVERY, BIN.
set -e
cd "$(dirname "$0")"
G=${1:-2,3,4,5,6,7}; PORT=${2:-30230}; SEED=${3:-42}
TIE_SYNC=${TIE_SYNC:-1}; SAVE_EVERY=${SAVE_EVERY:-1000}; BUDGET=${BUDGET:-15000000000}
NG=$(echo "$G" | awk -F, '{print NF}'); K=2; R=$(( NG / K ))
[ $(( K * R )) -eq "$NG" ] || { echo "NG=$NG is not a multiple of K=$K (4 GPUs -> R=2, 6 GPUs -> R=3)"; exit 1; }
TAG=tie340m_k2_ts${TIE_SYNC}_${NG}c
BIN=${BIN:-/dev/shm/slim/slim_15b.bin}
[ -f "$BIN" ] || { echo "missing $BIN - first: mkdir -p /dev/shm/slim && cp data/slim_15b.bin /dev/shm/slim/"; exit 1; }
EFF=$(( 8 * 16 * 2048 * R ))
L=/tmp/slim_logs; mkdir -p "$L" ckpt
echo "[cfg] $TAG  GPUs=$G (NG=$NG = K${K}×R${R})  eff_batch=$EFF  TIE_SYNC=$TIE_SYNC  BUDGET=$BUDGET  SAVE_EVERY=$SAVE_EVERY  steps≈$(( BUDGET / EFF ))"

export MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT WORLD_SIZE=$NG K=$K \
  CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8 MKL_THREADING_LAYER=GNU \
  NLAYER=24 DMODEL=1024 NHEAD=16 FFN=0 SEQ=2048 BS=8 ACCUM=16 CH=8192 \
  LR=3e-4 LR_MIN=3e-5 WARMUP_TOKENS=5e8 WD=0.01 CLIP=1.0 SEED=$SEED \
  BIN=$BIN TOKENS=$BUDGET COMPILE=2 FAKE_DATA=0 STEPS=0 \
  LOSS=md TIE_MODE=trained TIE_SYNC=$TIE_SYNC EVAL_EVERY=1000 SAVE_EVERY=$SAVE_EVERY \
  OUT=ckpt/$TAG.pt RESUME=ckpt/$TAG.pt CKPT_STAGE_DIR=/dev/shm
for r in $(seq 0 $((NG-1))); do
  RANK=$r LOCAL_RANK=$r nohup python slim_pipeline_tievar.py > "$L/${TAG}_r$r.log" 2>&1 &
done
echo "launched $TAG (GPUs $G, port $PORT, WORLD_SIZE=$NG, K=$K, R=$R, TIE_SYNC=$TIE_SYNC)"
echo "  check: head -3 $L/${TAG}_r0.log   (expect K=2, R=$R, eff_batch=$EFF, TIE_SYNC=$TIE_SYNC)"
echo "  curve: grep -a 'eval @' $L/${TAG}_r1.log | tail   (terminal stage = stage1 of replica 0 = rank1)"
wait
