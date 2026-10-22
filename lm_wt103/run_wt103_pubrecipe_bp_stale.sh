#!/bin/bash
# run_wt103_pubrecipe_bp_stale.sh - BP and stale-readout arms for the public-recipe single-process table (lm_solo.py, 1 GPU per arm).
#   BP          : MODE=bp, end-to-end single head, same recipe and data pipeline as tie/untie. GPU0.
#   local-stale : terminal readout of the finished tie run, frozen and injected into the aux heads (STALE_FC). GPU1.
#   BP-stale    : BP readout, frozen and injected the same way; runs on GPU0 after BP.
#   In the stale arms the terminal fc trains as usual.
# Requires extract_bpfc.py and the tie checkpoint ckpt_pubsp/wt103sp_k2_tie.pt (from run_wt103_pubrecipe_ksweep_sp.sh).
# Tags with a FINAL line are skipped.
# Usage: GPUS=0,1 nohup bash run_wt103_pubrecipe_bp_stale.sh > logs_pubsp/bp_stale.log 2>&1 &
# Knobs: GPUS EPOCHS SEED TIE_CKPT PY LOGDIR CKPTDIR DATA
cd "$(dirname "$0")"
PY=${PY:-python}; LOGDIR=${LOGDIR:-logs_pubsp}; CKPTDIR=${CKPTDIR:-ckpt_pubsp}; DATA=${DATA:-./data/wikitext}
mkdir -p "$LOGDIR" "$CKPTDIR"
GPUS=${GPUS:-0,1}; EP=${EPOCHS:-30}; SEED=${SEED:-42}
IFS=',' read -ra G <<< "$GPUS"; G0=${G[0]}; G1=${G[1]:-${G[0]}}
TIE_CKPT=${TIE_CKPT:-$CKPTDIR/wt103sp_k2_tie.pt}   # local-stale source: finished tie K=2 run
COMMON="STAGED=1 DATA=$DATA TOKENIZER=bert DSET=wikitext-103-raw-v1 NLAYER=16 DMODEL=410 FFN=2100 NHEAD=10 \
SEQ=256 BS=64 LR=2.5e-4 WD=0.0 DROP=0.1 LS=0.0 HEAD_DEPTH=2 EPOCHS=$EP SEED=$SEED \
BETA=1 PRIOR=unigram MU=0 AUX=1 KAPPA=0.7"
echo "[cfg] BP+stale  GPUS=$GPUS (BP,bpstale->$G0 ; lstale->$G1)  EP=$EP  SEED=$SEED"
echo "[recipe] $COMMON"

done_p(){ grep -qa FINAL "$LOGDIR/$1.log" 2>/dev/null; }
final(){ grep -a FINAL "$LOGDIR/$1.log" 2>/dev/null | tail -1; }

# ---- BP (GPU0, MODE=bp, end-to-end single head) ----
run_bp(){
  local tag=wt103sp_bp
  if done_p $tag; then echo "[skip] $tag: $(final $tag)"; return 0; fi
  echo "[$(date +%H:%M:%S)] START $tag (MODE=bp gpu=$G0)"
  env CUDA_VISIBLE_DEVICES=$G0 $COMMON MODE=bp K=1 OUT=$CKPTDIR/$tag.pt \
      $PY -u lm_solo.py > "$LOGDIR/$tag.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $tag: $(final $tag)"
}

# ---- stale arms (aux heads read a frozen readout; $1=tag $2=STALE_FC $3=gpu) ----
run_stale(){
  local tag=$1 fc=$2 gpu=$3
  if done_p $tag; then echo "[skip] $tag: $(final $tag)"; return 0; fi
  [ -f "$fc" ] || { echo "[FAIL] $tag: missing frozen readout $fc"; return 1; }
  echo "[$(date +%H:%M:%S)] START $tag (STALE_FC=$fc gpu=$gpu)"
  env CUDA_VISIBLE_DEVICES=$gpu $COMMON MODE=mfb K=2 STALE_FC=$fc OUT=$CKPTDIR/$tag.pt \
      $PY -u lm_solo.py > "$LOGDIR/$tag.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $tag: $(final $tag)"
}

# extract readout ($1=in ckpt  $2=out flat fc)
extract(){ [ -f "$2" ] && { echo "[skip] extract, already have $2"; return 0; }; $PY extract_bpfc.py "$1" "$2"; }

# ---- concurrency: BP(G0) and local-stale(G1) start at the same time ----
[ -f "$TIE_CKPT" ] || { echo "[FAIL] missing tie source ckpt $TIE_CKPT (local-stale has no source)"; }
tie_fc=$CKPTDIR/wt103sp_k2_tie_fc.pt
( run_bp ) &  PID_BP=$!
( extract "$TIE_CKPT" "$tie_fc" && run_stale wt103sp_lstale "$tie_fc" "$G1" ) &  PID_LS=$!

# ---- BP done -> extract BP readout -> BP-stale takes over G0 ----
wait $PID_BP
bp_fc=$CKPTDIR/wt103sp_bp_fc.pt
if done_p wt103sp_bp; then
  extract "$CKPTDIR/wt103sp_bp.pt" "$bp_fc" && run_stale wt103sp_bpstale "$bp_fc" "$G0"
else
  echo "[warn] no FINAL for BP, skipping BP-stale (check $LOGDIR/wt103sp_bp.log)"
fi
wait $PID_LS

echo "[$(date +%H:%M:%S)] BP_STALE_DONE"
echo "==== FINAL summary ===="
for a in bp lstale bpstale; do final wt103sp_$a | sed "s/^/[$a] /"; done
echo "Compare: [lstale] val ppl vs [bpstale] val ppl (a lower lstale means the own mature readout beats the imported BP readout)."
echo "Readout: val ppl = FINAL 'best val ppl:'; wppl/mem = last epoch line; train_tokens in the log header 'train_tokens='."
