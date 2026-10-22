#!/bin/bash
# run_lwce_4gpu.sh - local-CE control on 4 data-parallel GPUs via run_lwce.py (each GPU holds all 24 layers).
# Effective batch BS*SEQ*ACCUM*4 must match the reference arm; ACCUM = EFF_REF / (BS*SEQ*4).
# Usage:
#   ARM=light   bash run_lwce_4gpu.sh     # HEAD_DEPTH=0, greedy local CE
#   ARM=matched bash run_lwce_4gpu.sh     # HEAD_DEPTH=2, same head as SOLO
#   ARM=mdp     bash run_lwce_4gpu.sh     # LWCE=0, milestone loss, same runner
# Overrides: GPUS, BS, SEQ, K, EFF_REF (default 131072), TOKENS (default 1e9), BIN, SEED, EVAL_EVERY, LOG_EVERY, MASTER_PORT.
cd "$(dirname "$0")"
PY=${PY:-python3}
command -v "$PY" >/dev/null 2>&1 || PY=python
mkdir -p logs ckpt

ARM=${ARM:-light}
case "$ARM" in
  light)   HD=0; LW=1 ;;
  untie-greedy) HD=2; LW=1; export SRC=slim_untied.py UNTIE=1 ;;
  untie-mdp)    HD=2; LW=0; export SRC=slim_untied.py UNTIE=1 ;;
  matched) HD=2; LW=1 ;;
  mdp)     HD=2; LW=0 ;;
  *) echo "ARM must be one of light|matched|mdp|untie-greedy|untie-mdp"; exit 1 ;;
esac

GPUS=${GPUS:-0,1,2,3}
WORLD=4
BS=${BS:-8}; SEQ=${SEQ:-2048}; K=${K:-4}
EFF_REF=${EFF_REF:-131072}                       # effective batch of the reference arm
ACCUM=$(( EFF_REF / (BS * SEQ * WORLD) ))
if [ "$ACCUM" -lt 1 ]; then echo "EFF_REF too small, got ACCUM=$ACCUM"; exit 1; fi
TOKENS=${TOKENS:-1000000000}
STEPS_EXP=$(( TOKENS / EFF_REF ))
TAG=lwce_${ARM}_4gpu

echo "[$(date +%m-%d\ %H:%M)] ARM=$ARM HEAD_DEPTH=$HD LWCE=$LW GPUS=$GPUS"
echo "  BS=$BS SEQ=$SEQ ACCUM=$ACCUM WORLD=$WORLD -> eff_batch=$(( BS*SEQ*ACCUM*WORLD ))"
echo "  TOKENS=$TOKENS -> opt_steps≈$STEPS_EXP   <-- check these three numbers against the reference arm log header line"

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1} MASTER_PORT=${MASTER_PORT:-29744}
export WORLD_SIZE=$WORLD CUDA_VISIBLE_DEVICES=$GPUS
COMMON="SRC=${SRC:-slim_solo.py} UNTIE=${UNTIE:-0} BIN=${BIN:-data/slim_2b.bin} TOKENS=$TOKENS BS=$BS SEQ=$SEQ ACCUM=$ACCUM K=$K \
HEAD_DEPTH=$HD LWCE=$LW EVAL_EVERY=${EVAL_EVERY:-200} LOG_EVERY=${LOG_EVERY:-50} \
SEED=${SEED:-42} OUT=ckpt/${TAG}.pt"

for r in 0 1 2 3; do
  env $COMMON RANK=$r LOCAL_RANK=$r "$PY" -u run_lwce.py > logs/${TAG}_r${r}.log 2>&1 &
done
wait
echo "[$(date +%m-%d\ %H:%M)] $TAG done"
echo "read results from [eval @ step N] val_ppl / exits in logs/${TAG}_r0.log"
