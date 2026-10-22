#!/bin/bash
# run_wt103_pubrecipe_ksweep_sp.sh - single-process public-recipe WT-103 K sweep (lm_solo.py): {tie, untie} x K in {2,4,8}.
# K is an in-model config (no pipeline, no NCCL); 1 GPU per arm, up to NG arms concurrently.
# Recipe: MODE=mfb, bert tokenizer, 16L d410 FFN2100 10 heads, SEQ256, LR 2.5e-4, WD 0, DROP 0.1, HEAD_DEPTH=2, 30ep, seed 42;
#   BETA=1 PRIOR=unigram MU=0 AUX=1 KAPPA=0.7.
# Tags with a FINAL line are skipped.
# Usage: GPUS=0,1,2,3 nohup bash run_wt103_pubrecipe_ksweep_sp.sh > logs_pubsp/sweep.log 2>&1 &
#   K=2 only: GPUS=0,1 KS=2 bash run_wt103_pubrecipe_ksweep_sp.sh
# Knobs: GPUS KS EPOCHS SEED PY LOGDIR CKPTDIR DATA
# Reference: K=2 tie val ppl ~ 23.61 (subword); BP 22.96.
cd "$(dirname "$0")"
PY=${PY:-python}; LOGDIR=${LOGDIR:-logs_pubsp}; CKPTDIR=${CKPTDIR:-ckpt_pubsp}; DATA=${DATA:-./data/wikitext}
mkdir -p "$LOGDIR" "$CKPTDIR"
GPUS=${GPUS:-0,1,2,3}; KS=${KS:-"2 4 8"}; EP=${EPOCHS:-30}; SEED=${SEED:-42}
IFS=',' read -ra G <<< "$GPUS"; NG=${#G[@]}
COMMON="MODE=mfb STAGED=1 DATA=$DATA TOKENIZER=bert DSET=wikitext-103-raw-v1 NLAYER=16 DMODEL=410 FFN=2100 NHEAD=10 \
SEQ=256 BS=64 LR=2.5e-4 WD=0.0 DROP=0.1 LS=0.0 HEAD_DEPTH=2 EPOCHS=$EP SEED=$SEED \
BETA=1 PRIOR=unigram MU=0 AUX=1 KAPPA=0.7"
echo "[cfg] SP K-sweep KS='$KS' EP=$EP GPUS=$GPUS (NG=$NG)"
echo "[recipe] $COMMON"

arms=()
for K in $KS; do arms+=("wt103sp_k${K}_tie|$K|1" "wt103sp_k${K}_untie|$K|0"); done

launch(){   # $1=tag $2=K $3=TIE_AUX $4=gpu
  local tag=$1 K=$2 tie=$3 gpu=$4
  if grep -qa FINAL "$LOGDIR/${tag}.log" 2>/dev/null; then
    echo "[skip] $tag: $(grep -a FINAL "$LOGDIR/${tag}.log" | tail -1)"; return 0; fi
  echo "[$(date +%H:%M:%S)] START $tag (K=$K tie_aux=$tie gpu=$gpu)"
  env CUDA_VISIBLE_DEVICES=$gpu $COMMON K=$K TIE_AUX=$tie OUT=$CKPTDIR/${tag}.pt \
      $PY -u lm_solo.py > "$LOGDIR/${tag}.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $tag: $(grep -a FINAL "$LOGDIR/${tag}.log" | tail -1)"
}

# concurrent pool: NG arms at once (1 GPU each)
i=0; n=${#arms[@]}
while [ $i -lt $n ]; do
  pids=()
  for ((g=0; g<NG && i<n; g++, i++)); do
    IFS='|' read -r tag K tie <<< "${arms[$i]}"
    ( launch "$tag" "$K" "$tie" "${G[$g]}" ) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
done

echo "[$(date +%H:%M:%S)] SP_SWEEP_DONE"
sumtsv=$LOGDIR/summary.tsv
printf "K\tarm\tval_ppl\tpeak_mem_GB\ttok_s\n" > "$sumtsv"
echo "==== Summary (subword val ppl | peak mem | tok/s from train_tokens/epoch seconds) ===="
for K in $KS; do
  for a in tie untie; do
    f=$LOGDIR/wt103sp_k${K}_${a}.log
    ppl=$(grep -a FINAL "$f" 2>/dev/null | awk -F'val ppl: ' 'NF>1{print $2+0}' | tail -1)
    mem=$(grep -aoE 'mem=[0-9.]+GB' "$f" 2>/dev/null | tail -1 | grep -oE '[0-9.]+')
    ntok=$(grep -aoE 'train_tokens=[0-9]+' "$f" 2>/dev/null | tail -1 | grep -oE '[0-9]+')
    esec=$(grep -aoE '\([0-9]+s\)' "$f" 2>/dev/null | tail -1 | grep -oE '[0-9]+')
    tps=$([ -n "$ntok" ] && [ -n "$esec" ] && [ "$esec" -gt 0 ] && echo $((ntok/esec)) || echo -)
    printf "  K=%-2s %-6s ppl=%-7s mem=%-6s GB  tok/s≈%s\n" "$K" "$a" "${ppl:--}" "${mem:--}" "$tps"
    printf "%s\t%s\t%s\t%s\t%s\n" "$K" "$a" "${ppl:--}" "${mem:--}" "$tps" >> "$sumtsv"
  done
done
echo "TSV: $sumtsv. Reference: K=2 tie ~ 23.61 (subword val). Per K report tie/untie + channel(untie-tie)% + mem/tok-s."
echo "[tok/s note] = train_tokens / last-epoch seconds (includes eval of that epoch, slightly low); for pure training throughput see the step lines in the log."
