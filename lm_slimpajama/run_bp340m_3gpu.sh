#!/bin/bash
# run_bp340m_3gpu.sh - 340M BP baseline (DDP, default 3 GPUs), iso-FLOP with the SOLO 340M arm.
# ACCUM = EFF_REF / (BS*SEQ*WORLD); 3 GPUs with EFF_REF=524288 give ACCUM=10 (491,520 tok/step).
# Iso-FLOP tokens: TOKENS = MDP_TOKENS x (1 + FLOP_PCT/100), default 15e9 x 1.13; setting TOKENS overrides this.
# BIN must be the same bin as the SOLO arm and needs BIN.meta.json (from prep_slimpajama.py).
# Usage: [TOKENS=15000000000] [WARMUP_TOKENS=5e8] GPUS=0,1,2 [PORT=29540] bash run_bp340m_3gpu.sh
# Overrides: GPUS, PORT, PY, BS, SEQ, EFF_REF, FLOP_PCT, MDP_TOKENS, TOKENS, WARMUP_TOKENS, BIN, SEED, COMPILE, EVAL_EVERY, LOG_EVERY, SAVE_EVERY.
cd "$(dirname "$0")"
PY=${PY:-python}; GPUS=${GPUS:-0,1,2}; WORLD=$(echo "$GPUS" | awk -F, '{print NF}')
BS=${BS:-8}; SEQ=${SEQ:-2048}; EFF_REF=${EFF_REF:-524288}
ACCUM=$(( EFF_REF / (BS * SEQ * WORLD) )); [ "$ACCUM" -ge 1 ] || { echo "EFF_REF/(BS*SEQ*WORLD)<1"; exit 1; }
EFF=$(( BS * SEQ * ACCUM * WORLD )); WT=${WARMUP_TOKENS:-5e8}
FLOP_PCT=${FLOP_PCT:-13}; MDP_TOKENS=${MDP_TOKENS:-15000000000}   # SOLO FLOP/token overhead over BP, in %
TOKENS=${TOKENS:-$(( MDP_TOKENS * (100 + FLOP_PCT) / 100 ))}
BIN=${BIN:-/dev/shm/slim/slim_15b.bin}; [ -f "$BIN" ] || BIN=data/slim_15b.bin
[ -f "$BIN" ] || { echo "missing $BIN (override with BIN=...); BP must use the same bin as SOLO"; exit 1; }
[ -f "$BIN.meta.json" ] || echo "[warn] missing $BIN.meta.json - slim_bp needs it (produced by prep_slimpajama)"
SEED=${SEED:-42}; TAG=bp340m_${WORLD}gpu_$(( TOKENS/1000000000 ))b_s${SEED}
mkdir -p logs ckpt
echo "[cfg] $TAG GPUS=$GPUS WORLD=$WORLD ACCUM=$ACCUM eff_batch=$EFF (target $EFF_REF) | iso-FLOP: SOLO=$MDP_TOKENS ×(1+${FLOP_PCT}%) => BP TOKENS=$TOKENS WARMUP_TOKENS=$WT opt_steps≈$(( TOKENS/EFF ))"
export WORLD_SIZE=$WORLD MASTER_ADDR=127.0.0.1 MASTER_PORT=${PORT:-29540} CUDA_VISIBLE_DEVICES=$GPUS
COMMON="ARCH=llama NLAYER=24 DMODEL=1024 NHEAD=16 FFN=0 SEQ=$SEQ BS=$BS ACCUM=$ACCUM \
  BIN=$BIN TOKENS=$TOKENS LR=3e-4 LR_MIN=3e-5 WARMUP_TOKENS=$WT WD=0.01 CLIP=1.0 TIE=0 \
  COMPILE=${COMPILE:-1} EVAL_EVERY=${EVAL_EVERY:-1000} LOG_EVERY=${LOG_EVERY:-50} SAVE_EVERY=${SAVE_EVERY:-2000} \
  SEED=$SEED OUT=ckpt/$TAG.pt RESUME=ckpt/$TAG.pt"
for r in $(seq 0 $((WORLD - 1))); do
  env $COMMON RANK=$r LOCAL_RANK=$r "$PY" -u slim_bp.py > logs/${TAG}_r${r}.log 2>&1 &
done
echo "  tail -f logs/${TAG}_r0.log ; check: params≈340M, eff_batch=$EFF, opt_steps≈$(( TOKENS/EFF ))"
wait
