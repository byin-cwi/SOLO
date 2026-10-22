#!/bin/bash
# run_wt103_stalebp.sh - BP-stale arm of the WT-103 readout ladder (lm_hz_tievar.py): protocol of run_wt103_ladder.sh
#   (8L d384, word level V=32768, K=4, HD=2, 1ep) with STALE_FC = BP terminal readout (flat {fc.weight,fc.bias}).
# Tag wt103_ep${EP}_hd${HD}_stalebp, separate from the ladder's _stale/_tie1/_tie0 tags.
# Requires a BP checkpoint saved with OUT= and its readout extracted to $STALE_FC with extract_bpfc.py.
# Usage:
#   GPUS=4,5,6,7 STALE_FC=ckpt_wt103tie/bpfc_readout.pt \
#     nohup bash run_wt103_stalebp.sh > logs_wt103tie/stalebp_driver.log 2>&1 &
# Knobs: GPUS STALE_FC EPOCHS HEAD_DEPTH BS LR WARMUP SEED COMPILE PORT0 RANK_STAGGER PY LOGDIR CKPTDIR DATA
cd "$(dirname "$0")"
PY=${PY:-python}; LOGDIR=${LOGDIR:-logs_wt103tie}; CKPTDIR=${CKPTDIR:-ckpt_wt103tie}; DATA=${DATA:-./data/wikitext}
mkdir -p "$LOGDIR" "$CKPTDIR"
GPUS=${GPUS:-0,1,2,3}; K=4; EP=${EPOCHS:-1}; HD=${HEAD_DEPTH:-2}
STALE_FC=${STALE_FC:-$CKPTDIR/bpfc_readout.pt}
[ -f "$STALE_FC" ] || { echo "[abort] missing STALE_FC=$STALE_FC - run extract_bpfc.py first to extract the BP readout"; exit 1; }
# COMMON of run_wt103_ladder.sh
COMMON="DATA=$DATA TOKENIZER=word DSET=wikitext-103-raw-v1 VMAX=32768 \
NLAYER=8 DMODEL=384 NHEAD=6 SEQ=256 BS=${BS:-64} LR=${LR:-3e-4} WARMUP=${WARMUP:-1000} \
CLIP=0.5 WD=0.0 LS=0.0 DROP=0.1 EPOCHS=$EP SEED=${SEED:-42} \
MODE=mfb BETA=1 PRIOR=unigram STAGED=1 TK=0,0,0 MU=0 AUX=1 KAPPA=0.7 \
HZ_H=1,1,1 BUCKET_BY=h COMPILE=${COMPILE:-1}"
export CUDA_VISIBLE_DEVICES=$GPUS MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
port=${PORT0:-$(( 20000 + RANDOM % 20000 ))}

tag=wt103_ep${EP}_hd${HD}_stalebp                 # separate from the local-stale _stale tag
if grep -qa FINAL "$LOGDIR/${tag}_r$((K-1)).log" 2>/dev/null; then
  echo "[skip] $tag already done: $(grep -a FINAL "$LOGDIR/${tag}_r$((K-1)).log" | tail -1)"; exit 0; fi
echo "[$(date +%H:%M:%S)] START $tag  (TIE_MODE=stale from BP readout $STALE_FC, port=$port)"
pids=()
for r in $(seq 0 $((K-1))); do
  env $COMMON WORLD_SIZE=$K MASTER_PORT=$port HEAD_DEPTH=$HD TIE_AUX=1 TIE_MODE=stale STALE_FC=$STALE_FC \
      TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_sbp_r${r} OUT=$CKPTDIR/${tag}.pt RANK=$r LOCAL_RANK=$r \
      $PY -u lm_hz_tievar.py > "$LOGDIR/${tag}_r${r}.log" 2>&1 &
  pids+=($!); sleep ${RANK_STAGGER:-3}
done
failed=0
for p in "${pids[@]}"; do wait "$p" || failed=1; done
[ $failed -eq 1 ] && { echo "[FAIL] $tag a rank exited non-zero; see $LOGDIR/${tag}_r*.log"; kill -9 "${pids[@]}" 2>/dev/null; exit 1; }
echo "[$(date +%H:%M:%S)] DONE  $tag: $(grep -a FINAL "$LOGDIR/${tag}_r$((K-1)).log" | tail -1)"
