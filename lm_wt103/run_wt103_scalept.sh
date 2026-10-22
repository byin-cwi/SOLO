#!/bin/bash
# run_wt103_scalept.sh - readout ladder plus stale-readout arms at one WT-103 scale point (lm_hz_tievar.py, lm_bp_ddp.py, extract_bpfc.py).
# Runs sequentially (K=4 pipeline on 4 GPUs): tie, untie, random, BP K=1 -> extract BP readout ->
#   local-stale (frozen terminal readout of tie) and BP-stale (frozen terminal readout of BP).
# Protocol as run_wt103_ladder.sh (1ep, word level V=32768, K=4, HD=2); only NLAYER/DMODEL/NHEAD change.
# Tags with a FINAL line are skipped.
# Usage: GPUS=4,5,6,7 TAGPFX=wt103_64m nohup bash run_wt103_scalept.sh > logs_scalept/${TAGPFX:-wt103_64m}_driver.log 2>&1 &
# Knobs (defaults = 64M): NLAYER=12 DMODEL=512 NHEAD=8 (FFN = 4d) TAGPFX=wt103_64m SEED=42 GPUS DATA
#   K=4 HD=2 SEQ=256 BS=64 LR=3e-4 WARMUP=1000 EPOCHS=1 COMPILE PORT0 PY LOGDIR CKPTDIR
# Reference (64M): tie ~59 / untie ~63 / random ~82.
cd "$(dirname "$0")"
PY=${PY:-python}; LOGDIR=${LOGDIR:-logs_scalept}; CKPTDIR=${CKPTDIR:-ckpt_scalept}; DATA=${DATA:-./data/wikitext}
mkdir -p "$LOGDIR" "$CKPTDIR"
NLAYER=${NLAYER:-12}; DMODEL=${DMODEL:-512}; NHEAD=${NHEAD:-8}
K=${K:-4}; HD=${HD:-2}; SEQ=${SEQ:-256}; BS=${BS:-64}; LR=${LR:-3e-4}; WARMUP=${WARMUP:-1000}; EP=${EPOCHS:-1}
SEED=${SEED:-42}; GPUS=${GPUS:-0,1,2,3}; TAGPFX=${TAGPFX:-wt103_64m}
IFS=',' read -ra G <<< "$GPUS"; NG=${#G[@]}
export CUDA_VISIBLE_DEVICES=$GPUS MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
echo "[cfg] $TAGPFX NLAYER=$NLAYER DMODEL=$DMODEL NHEAD=$NHEAD K=$K HD=$HD SEQ=$SEQ BS=$BS EP=$EP SEED=$SEED GPUS=$GPUS"

# COMMON of run_wt103_ladder.sh with the scale as variables
LCOMMON="DATA=$DATA TOKENIZER=word DSET=wikitext-103-raw-v1 VMAX=32768 \
NLAYER=$NLAYER DMODEL=$DMODEL NHEAD=$NHEAD SEQ=$SEQ BS=$BS LR=$LR WARMUP=$WARMUP \
CLIP=0.5 WD=0.0 LS=0.0 DROP=0.1 EPOCHS=$EP SEED=$SEED \
MODE=mfb BETA=1 PRIOR=unigram STAGED=1 TK=0,0,0 MU=0 AUX=1 KAPPA=0.7 HZ_H=1,1,1 BUCKET_BY=h COMPILE=${COMPILE:-1}"

port=${PORT0:-$(( 20000 + RANDOM % 20000 ))}

ladder_arm () {   # $1=tag $2=TIE_AUX $3=TIE_MODE $4=STALE_FC(may be empty)
  local tag=$1 tie=$2 mode=$3 stale=${4:-}
  if grep -qa FINAL "$LOGDIR/${tag}_r$((K-1)).log" 2>/dev/null; then
    echo "[skip] $tag: $(grep -a FINAL "$LOGDIR/${tag}_r$((K-1)).log" | tail -1)"; return 0; fi
  port=$((port+1)); echo "[$(date +%H:%M:%S)] START $tag (mode=$mode tie_aux=$tie)"
  local pids=()
  for r in $(seq 0 $((K-1))); do
    env $LCOMMON WORLD_SIZE=$K MASTER_PORT=$port HEAD_DEPTH=$HD TIE_AUX=$tie TIE_MODE=$mode STALE_FC=$stale \
        TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_sp_${tag}_r${r} OUT=$CKPTDIR/${tag}.pt RANK=$r LOCAL_RANK=$r \
        $PY -u lm_hz_tievar.py > "$LOGDIR/${tag}_r${r}.log" 2>&1 &
    pids+=($!)
  done
  local failed=0; for p in "${pids[@]}"; do wait "$p" || failed=1; done
  [ $failed -eq 1 ] && { echo "[FAIL] $tag; see $LOGDIR/${tag}_r*.log"; kill -9 "${pids[@]}" 2>/dev/null; return 1; }
  echo "[$(date +%H:%M:%S)] DONE  $tag: $(grep -a FINAL "$LOGDIR/${tag}_r$((K-1)).log" | tail -1)"
}

# ---- Phase A: sources (tie / untie) + random + BP ----
ladder_arm "${TAGPFX}_tie"    1 trained || exit 1
ladder_arm "${TAGPFX}_untie"  0 trained || true         # failure does not block
ladder_arm "${TAGPFX}_random" 1 random  || true         # random frozen readout
# BP K=1 (lm_bp_ddp), single process on 1 GPU
bptag=${TAGPFX}_bp
if grep -qa FINAL "$LOGDIR/${bptag}.log" 2>/dev/null; then
  echo "[skip] $bptag: $(grep -a FINAL "$LOGDIR/${bptag}.log" | tail -1)"
else
  echo "[$(date +%H:%M:%S)] START $bptag (BP K=1)"
  env MASTER_PORT=$((port+1)) WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
      DATA=$DATA TOKENIZER=word DSET=wikitext-103-raw-v1 VMAX=32768 \
      NLAYER=$NLAYER DMODEL=$DMODEL NHEAD=$NHEAD FFN=0 SEQ=$SEQ BS=$BS LR=$LR WARMUP=$WARMUP \
      CLIP=0.5 WD=0.0 LS=0.0 DROP=0.1 EPOCHS=$EP SEED=$SEED TIE=1 OUT=$CKPTDIR/${bptag}.pt \
      $PY -u lm_bp_ddp.py > "$LOGDIR/${bptag}.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $bptag: $(grep -a FINAL "$LOGDIR/${bptag}.log" | tail -1)"
fi

# ---- extract BP terminal readout -> flat {fc.weight,fc.bias} (skipped if present) ----
bpfc=$CKPTDIR/${TAGPFX}_bpfc.pt
if [ -f "$bpfc" ]; then echo "[skip] extract (already have $bpfc)"; else
  $PY extract_bpfc.py "$CKPTDIR/${bptag}.pt" "$bpfc" || { echo "[FAIL] extracting BP readout failed"; exit 1; }
fi

# ---- Phase B: two stale arms ----
tie_src=$CKPTDIR/${TAGPFX}_tie.pt.r$((K-1))
[ -f "$tie_src" ] || { echo "[FAIL] missing tie source $tie_src"; exit 1; }
ladder_arm "${TAGPFX}_lstale"  1 stale "$tie_src" || true    # local-stale: own mature readout
ladder_arm "${TAGPFX}_bpstale" 1 stale "$bpfc"    || true    # BP-stale:   imported BP readout

echo "[$(date +%H:%M:%S)] SCALEPT_DONE $TAGPFX"
echo "==== FINAL summary ($TAGPFX) ===="
for a in tie untie random lstale bpstale; do
  f=$LOGDIR/${TAGPFX}_${a}_r$((K-1)).log; grep -a FINAL "$f" 2>/dev/null | sed "s/^/[$a] /" | tail -1
done
grep -a FINAL "$LOGDIR/${bptag}.log" 2>/dev/null | sed 's/^/[bp] /' | tail -1
echo "Compare: [lstale] val ppl vs [bpstale] val ppl (a lower lstale means the own mature readout beats the imported BP readout)."
