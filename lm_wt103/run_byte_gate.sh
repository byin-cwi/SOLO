#!/bin/bash
# run_byte_gate.sh - byte-level (V=256) WT-103 control: RBOT {0,128} x TIE_AUX {1,0}, K=4 on 4 GPUs (lm_hz_bnk.py).
#
# Objective is pure greedy local CE (TK=0,0,0 => hinge=0, loss = AUX*CE). MAXTOK=1e8 truncates the train set
# (~6.1k steps/epoch). RBOT>0 factorizes the readout DMODEL->RBOT->V; the untied private aux head uses the same rank.
# Model/optimizer variables are literals in COMMON so that variables exported in the outer shell are not inherited.
# HZ_H=1,1,1 is required: the script default 4,3,2 changes the objective.
#
# Usage:
#   bash run_byte_gate.sh prep    # download WT-103 and build the byte cache (no-op if it exists)
#   bash run_byte_gate.sh smoke   # CPU fake data, 2 processes, tie/untie at RBOT=128 (needs torch>=2.0)
#   GPUS=0,1,2,3 bash run_byte_gate.sh run        # 4 runs: rbot{0,128} x tie{1,0}
#   GPUS=0,1,2,3 EP=3 bash run_byte_gate.sh run   # multi-epoch (tag includes ep)
# Knobs: GPUS EP SEED RBOTS STEPS COMPILE SAVE_CKPT PY LOGDIR CKPTDIR DATA PORT0 RANK_STAGGER STALL_MIN FAIL_TAIL
cd "$(dirname "$0")"
PY=${PY:-python}
LOGDIR=${LOGDIR:-logs_bytegate}
CKPTDIR=${CKPTDIR:-ckpt_bytegate}
DATA=${DATA:-./data/wikitext}
mkdir -p "$LOGDIR" "$CKPTDIR" "$DATA"

GPUS=${GPUS:-0,1,2,3}
K=4                                     # number of segments = number of GPUs (one segment per GPU)
EP=${EP:-1}                             # EP, not EPOCHS, so an exported EPOCHS is not inherited
COMMON="DATA=$DATA TOKENIZER=byte DSET=wikitext-103-raw-v1 MAXTOK=100000000 \
NLAYER=8 DMODEL=384 NHEAD=6 SEQ=256 BS=64 LR=3e-4 WARMUP=1000 \
CLIP=0.5 WD=0.0 LS=0.0 DROP=0.1 EPOCHS=$EP SEED=${SEED:-42} \
MODE=mfb BETA=1 PRIOR=unigram STAGED=1 TK=0,0,0 MU=0 AUX=1 KAPPA=0.7 \
HZ_H=1,1,1 BUCKET_BY=h HEAD_DEPTH=2 \
FFN=0 STEPS=${STEPS:-0} LOAD= TIE_SYNC=100 \
COMPILE=${COMPILE:-1}"

case "${1:-run}" in
prep)
  CF="$DATA/byte_cache_wikitext-103-raw-v1.pt"
  if [ -f "$CF" ]; then echo "[prep] byte cache already exists: $CF, skipping"; exit 0; fi
  echo "[prep] download wikitext-103-raw-v1 -> $DATA (byte path needs no tokenizer)"
  HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} $PY - <<PYEOF
import os
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
from datasets import load_dataset
d = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", cache_dir="$DATA")
print({k: len(v) for k, v in d.items()})
PYEOF
  echo "[prep] building byte cache (single-process gloo; stores full uint8 ~0.54GB, exits when done)"
  env $COMMON WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29689 \
      BACKEND=gloo STEPS=1 EPOCHS=1 HEAD_DEPTH=0 TIE_AUX=1 COMPILE=0 \
      BUILD_WORD_CACHE=1 $PY -u lm_hz_bnk.py > "$LOGDIR/prep_cache.log" 2>&1 \
    || { echo "[prep] cache build failed, see $LOGDIR/prep_cache.log"; tail -5 "$LOGDIR/prep_cache.log"; exit 1; }
  ls -la "$DATA"/byte_cache_* 2>/dev/null
  echo "[prep] done"; exit 0 ;;
smoke)
  echo "[smoke] CPU fake data, two processes: tie/untie paths at RBOT=128 (tie includes composite-matrix broadcast)"
  for tie in 1 0; do
    MASTER_ADDR=127.0.0.1 MASTER_PORT=29673 WORLD_SIZE=2 BACKEND=gloo FAKE_DATA=1 VOCAB=256 \
    STEPS=3 EPOCHS=1 NLAYER=4 DMODEL=64 NHEAD=2 FFN=128 SEQ=32 BS=4 LR=3e-4 WARMUP=10 DROP=0.1 \
    HEAD_DEPTH=1 RBOT=128 TIE_AUX=$tie HZ_H=1,1 CLIP=0.5 \
    MODE=mfb BETA=1 PRIOR=unigram STAGED=1 TK=0 MU=0 AUX=1 KAPPA=0.7 BUCKET_BY=h COMPILE=0 \
    bash -c "RANK=0 LOCAL_RANK=0 $PY -u lm_hz_bnk.py > $LOGDIR/smoke_rb128_t${tie}_r0.log 2>&1 & \
             RANK=1 LOCAL_RANK=1 $PY -u lm_hz_bnk.py > $LOGDIR/smoke_rb128_t${tie}_r1.log 2>&1; wait"
    echo "  tie=$tie: $(grep -a 'tie_aux' $LOGDIR/smoke_rb128_t${tie}_r0.log)"
    echo "         $(grep -a FINAL $LOGDIR/smoke_rb128_t${tie}_r1.log | tail -1)"
  done; exit 0 ;;
esac

export CUDA_VISIBLE_DEVICES=$GPUS MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
# One inductor compile worker per rank: the default min(32, cpu_count) across 4 ranks can exhaust host memory.
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
port=${PORT0:-$(( 20000 + RANDOM % 20000 ))}   # random base: a fixed port can be held by TIME_WAIT/leftover processes
echo "[$(date +%H:%M:%S)] byte two-arm N0 gate  EP=$EP  GPUS=$GPUS  LOGDIR=$LOGDIR"
for rb in ${RBOTS:-0 128}; do for tie in 1 0; do
  tag=bytegate_ep${EP}_rbot${rb}_tie${tie}
  if grep -qa FINAL "$LOGDIR/${tag}_r$((K-1)).log" 2>/dev/null; then echo "[skip] $tag"; continue; fi
  pkill -f "lm_hz_bnk.py" 2>/dev/null && { echo "  (cleaned up leftover processes from previous round)"; sleep 5; }
  port=$((port+1))
  for _try in 1 2 3 4 5; do                       # if the port is taken, pick another
    if command -v ss >/dev/null && ss -ltn 2>/dev/null | grep -q ":$port "; then
      echo "  port $port in use, switching to $((port+17))"; port=$((port+17))
    else break; fi
  done
  echo "[$(date +%H:%M:%S)] START $tag (rbot=$rb tie=$tie port=$port)"
  pids=()
  for r in $(seq 0 $((K-1))); do
    env $COMMON WORLD_SIZE=$K MASTER_PORT=$port RBOT=$rb TIE_AUX=$tie \
        TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_r${r} \
        OUT=${SAVE_CKPT:+$CKPTDIR/${tag}.pt} RANK=$r LOCAL_RANK=$r \
        $PY -u lm_hz_bnk.py > "$LOGDIR/${tag}_r${r}.log" 2>&1 &
    pids+=($!)
    sleep ${RANK_STAGGER:-3}
  done
  # Stall watchdog: starts timing after the first heartbeat of the terminal rank (startup + compile can be silent for long).
  ( SM=${STALL_MIN:-30}; LF="$LOGDIR/${tag}_r$((K-1)).log"
    while kill -0 ${pids[0]} 2>/dev/null; do
      sleep 60
      grep -qa "\[ep " "$LF" 2>/dev/null || continue     # training not started yet, do not time
      age=$(( ($(date +%s) - $(stat -c %Y "$LF" 2>/dev/null || echo 0)) / 60 ))
      if [ "$age" -ge "$SM" ]; then
        echo "[STALL] $tag heartbeat stopped for ${age} min -> deadlock, killing whole group" >&2
        kill -9 ${pids[@]} 2>/dev/null; break
      fi
    done ) &
  wd=$!
  left=$K; failed=0
  while [ $left -gt 0 ]; do
    if wait -n; then left=$((left-1)); else rc=$?; failed=1; break; fi
  done
  if [ $failed -eq 1 ]; then
    echo "[FAIL] $tag a rank exited with code ${rc:-?} -> killing remaining processes in this group"
    [ "${rc:-0}" = "137" ] && echo "  ** 137 = SIGKILL: almost certainly host-memory OOM." \
        "try COMPILE=0 first, or check dmesg -T | grep -i 'killed process'"
    for r in $(seq 0 $((K-1))); do
      echo "  -- r$r --"; tail -${FAIL_TAIL:-15} "$LOGDIR/${tag}_r${r}.log" 2>/dev/null
    done
    kill -9 ${pids[@]} 2>/dev/null
  fi
  kill $wd 2>/dev/null; wait 2>/dev/null
  echo "[$(date +%H:%M:%S)] DONE  $tag: $(grep -a FINAL "$LOGDIR/${tag}_r$((K-1)).log" | tail -1)"
done; done
echo "[$(date +%H:%M:%S)] done. See FINAL line in $LOGDIR/bytegate_ep${EP}_*_r$((K-1)).log"
echo "B = ln(ppl_tie/ppl_untie) @ terminal exit"
