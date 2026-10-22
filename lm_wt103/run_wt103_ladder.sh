#!/bin/bash
# run_wt103_ladder.sh - WT-103 readout ladder (lm_hz_tievar.py; 8L d384, word level V=32768, K=4, HD=2, 1ep), arms run in order.
# Arms (positional args): tie | untie | random | affine | untie_affine | stale. stale reads STALE_FC (default: terminal
#   ckpt of the tie arm in CKPTDIR); if that file is missing, a tie_src arm is run first to produce it.
# Usage: GPUS=0,1,2,3 nohup bash run_wt103_ladder.sh random affine stale untie_affine > logs_wt103tie/ladder_driver.log 2>&1 &
# Knobs: GPUS EPOCHS HEAD_DEPTH STALE_FC BS LR WARMUP SEED COMPILE PORT0 RANK_STAGGER PY LOGDIR CKPTDIR DATA
# Reference: tie 70.60 / untie 75.35.
cd "$(dirname "$0")"
PY=${PY:-python}; LOGDIR=${LOGDIR:-logs_wt103tie}; CKPTDIR=${CKPTDIR:-ckpt_wt103tie}; DATA=${DATA:-./data/wikitext}
mkdir -p "$LOGDIR" "$CKPTDIR"
GPUS=${GPUS:-0,1,2,3}; K=4; EP=${EPOCHS:-1}; HD=${HEAD_DEPTH:-2}
STALE_FC=${STALE_FC:-$CKPTDIR/wt103_ep${EP}_hd${HD}_tie1.pt.r$((K-1))}
COMMON="DATA=$DATA TOKENIZER=word DSET=wikitext-103-raw-v1 VMAX=32768 \
NLAYER=8 DMODEL=384 NHEAD=6 SEQ=256 BS=${BS:-64} LR=${LR:-3e-4} WARMUP=${WARMUP:-1000} \
CLIP=0.5 WD=0.0 LS=0.0 DROP=0.1 EPOCHS=$EP SEED=${SEED:-42} \
MODE=mfb BETA=1 PRIOR=unigram STAGED=1 TK=0,0,0 MU=0 AUX=1 KAPPA=0.7 \
HZ_H=1,1,1 BUCKET_BY=h COMPILE=${COMPILE:-1}"
export CUDA_VISIBLE_DEVICES=$GPUS MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
port=${PORT0:-$(( 20000 + RANDOM % 20000 ))}

LOCK="$LOGDIR/.ladder.lock"                       # one driver per LOGDIR
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "[abort] $LOGDIR already has driver PID $(cat "$LOCK") running; not launching again"; exit 1; fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
ARMS=("$@")
if printf '%s\n' "${ARMS[@]}" | grep -qx stale && [ ! -f "$STALE_FC" ]; then
  STALE_FC=$CKPTDIR/wt103_ep${EP}_hd${HD}_tie1_src.pt.r$((K-1))       # extra tie arm under the _src tag that saves the ckpt
  echo "[note] stale source missing, running tie1_src first to write -> $STALE_FC"; ARMS=(tie_src "${ARMS[@]}")
fi

run_arm () {   # $1 = tie|random|affine|untie_affine|stale
  local arm=$1 tie=1 mode=$1 tag
  case $arm in tie|tie_src) mode=trained;; untie) mode=trained; tie=0;; untie_affine) tie=0;; esac
  case $arm in tie) tag=wt103_ep${EP}_hd${HD}_tie1;; untie) tag=wt103_ep${EP}_hd${HD}_tie0;; tie_src) tag=wt103_ep${EP}_hd${HD}_tie1_src;; *) tag=wt103_ep${EP}_hd${HD}_$arm;; esac
  if grep -qa FINAL "$LOGDIR/${tag}_r$((K-1)).log" 2>/dev/null; then echo "[skip] $tag"; return 0; fi
  port=$((port+1))
  echo "[$(date +%H:%M:%S)] START $tag (mode=$mode tie_aux=$tie port=$port)"
  local pids=()
  for r in $(seq 0 $((K-1))); do
    env $COMMON WORLD_SIZE=$K MASTER_PORT=$port HEAD_DEPTH=$HD TIE_AUX=$tie TIE_MODE=$mode STALE_FC=$STALE_FC \
        TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_r${r} OUT=$CKPTDIR/${tag}.pt RANK=$r LOCAL_RANK=$r \
        $PY -u lm_hz_tievar.py > "$LOGDIR/${tag}_r${r}.log" 2>&1 &
    pids+=($!); sleep ${RANK_STAGGER:-3}
  done
  local failed=0
  for p in "${pids[@]}"; do wait "$p" || failed=1; done
  [ $failed -eq 1 ] && { echo "[FAIL] $tag a rank exited non-zero"; kill -9 "${pids[@]}" 2>/dev/null; return 1; }
  echo "[$(date +%H:%M:%S)] DONE  $tag: $(grep -a FINAL "$LOGDIR/${tag}_r$((K-1)).log" | tail -1)"
}
for a in "${ARMS[@]}"; do run_arm "$a" || echo "[warn] $a failed, continuing with next arm"; done
echo LADDER_DONE
