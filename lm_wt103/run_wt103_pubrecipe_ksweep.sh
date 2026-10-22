#!/bin/bash
# run_wt103_pubrecipe_ksweep.sh - K sweep x readout ladder (tie, untie, stale, random) for the public-recipe WT-103 config
#   (bert tokenizer, 16L d410 FFN2100 10 heads, SEQ256, LR 2.5e-4, CLIP 0.25, DROP 0.1, WD 0, HEAD_DEPTH=2, 30ep; lm_hz_tievar.py).
# Each rank uses one GPU: CUDA_VISIBLE_DEVICES=<arm's GPU subset>[r % subset size], LOCAL_RANK=0. When K > #GPUs
#   (K=8 on 4 GPUs) there are 2 segments per GPU and adjacent ranks sit on different GPUs; if NCCL hangs, use BACKEND=gloo.
# When K <= NG/2, NG/K arms run concurrently on disjoint GPU subsets (waves {tie,untie}, {stale,random}); stale uses the
#   tie readout, so it runs in the wave after tie. HZ_H/TK are generated from K; NLAYER=16 must be divisible by K.
# Tags with a FINAL line are skipped.
# Usage: GPUS=0,1,2,3 nohup bash run_wt103_pubrecipe_ksweep.sh > logs_pubk/sweep.log 2>&1 &
#   K=8 hang check: KS=8 EPOCHS=1 GPUS=0,1,2,3 bash run_wt103_pubrecipe_ksweep.sh
# Knobs: GPUS KS EPOCHS SEED BACKEND PORT0 PY LOGDIR CKPTDIR DATA
# Reference: K=2 tie val ppl ~ 23.61 (subword); BP 22.96.
cd "$(dirname "$0")"
PY=${PY:-python}; LOGDIR=${LOGDIR:-logs_pubk}; CKPTDIR=${CKPTDIR:-ckpt_pubk}; DATA=${DATA:-./data/wikitext}
mkdir -p "$LOGDIR" "$CKPTDIR"
GPUS=${GPUS:-0,1,2,3}; KS=${KS:-"2 4 8"}; EP=${EPOCHS:-30}; SEED=${SEED:-42}; BACKEND=${BACKEND:-nccl}
IFS=',' read -ra G <<< "$GPUS"; NG=${#G[@]}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1} TORCHINDUCTOR_COMPILE_THREADS=1
port0=${PORT0:-$((20000 + RANDOM % 20000))}
echo "[cfg] pubrecipe K-sweep KS='$KS' EP=$EP GPUS=$GPUS (NG=$NG) SEED=$SEED BACKEND=$BACKEND"

join_k(){ local v=$1 n=$2 o=""; for ((i=0; i<n; i++)); do o+="$v,"; done; echo "${o%,}"; }
subset(){ local s=$1 k=$2; local -a a=("${G[@]:s:k}"); local IFS=,; echo "${a[*]}"; }

launch_arm(){   # $1=tag $2=K $3=TIE_AUX $4=TIE_MODE $5=gpus(comma) $6=port $7=STALE_FC(may be empty)
  local tag=$1 K=$2 tie=$3 mode=$4 gpus=$5 pt=$6 stale=${7:-}
  if grep -qa FINAL "$LOGDIR/${tag}_r$((K-1)).log" 2>/dev/null; then
    echo "[skip] $tag: $(grep -a FINAL "$LOGDIR/${tag}_r$((K-1)).log" | tail -1)"; return 0; fi
  local -a GA; IFS=',' read -ra GA <<< "$gpus"; local ng=${#GA[@]}
  local HZ TK; HZ=$(join_k 1 $((K-1))); TK=$(join_k 0 $((K-1)))
  echo "[$(date +%H:%M:%S)] START $tag (K=$K mode=$mode gpus=$gpus port=$pt $([ $K -gt $ng ] && echo '2 seg/GPU' || echo '1 seg/GPU'))"
  local C="DATA=$DATA TOKENIZER=bert DSET=wikitext-103-raw-v1 NLAYER=16 DMODEL=410 FFN=2100 NHEAD=10 \
SEQ=256 BS=64 LR=2.5e-4 WARMUP=1000 CLIP=0.25 WD=0.0 LS=0.0 DROP=0.1 EPOCHS=$EP SEED=$SEED HEAD_DEPTH=2 \
MODE=mfb BETA=1 PRIOR=unigram STAGED=1 TK=$TK MU=0 AUX=1 KAPPA=0.7 HZ_H=$HZ BUCKET_BY=h COMPILE=1 BACKEND=$BACKEND"
  local pids=()
  for r in $(seq 0 $((K-1))); do
    env $C WORLD_SIZE=$K RANK=$r LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=${GA[$((r % ng))]} MASTER_PORT=$pt \
        TIE_AUX=$tie TIE_MODE=$mode STALE_FC=$stale \
        TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_pk_${tag}_r${r} OUT=$CKPTDIR/${tag}.pt \
        $PY -u lm_hz_tievar.py > "$LOGDIR/${tag}_r${r}.log" 2>&1 &
    pids+=($!)
  done
  local failed=0; for p in "${pids[@]}"; do wait "$p" || failed=1; done
  [ $failed -eq 1 ] && { echo "[FAIL] $tag; see $LOGDIR/${tag}_r*.log (K=8: if NCCL hangs, rerun with BACKEND=gloo)"; kill -9 "${pids[@]}" 2>/dev/null; return 1; }
  echo "[$(date +%H:%M:%S)] DONE  $tag: $(grep -a FINAL "$LOGDIR/${tag}_r$((K-1)).log" | tail -1)"
}

run_K(){   # in arm order tie/untie/stale/random; NG/K concurrent when K<=NG/2
  local K=$1 p=wt103pub_k${K}
  local conc=1; [ "$K" -le "$NG" ] && conc=$((NG / K)); [ "$conc" -lt 1 ] && conc=1
  local names=(tie untie stale random) ties=(1 0 1 1) modes=(trained trained stale random)
  echo "===== [$(date +%H:%M:%S)] K=$K (concurrency=$conc) ====="
  local i=0
  while [ $i -lt 4 ]; do
    local bg=() j=0 base=$((port0 + K * 100 + i * 4))
    while [ $j -lt $conc ] && [ $i -lt 4 ]; do
      local a=${names[$i]} tie=${ties[$i]} mode=${modes[$i]} tag=wt103pub_k${K}_${names[$i]}
      local gsub; if [ "$K" -gt "$NG" ]; then gsub=$GPUS; else gsub=$(subset $((j * K)) $K); fi
      local stale=""
      if [ "$a" = stale ]; then
        stale=$CKPTDIR/${p}_tie.pt.r$((K-1))
        [ -f "$stale" ] || { echo "[warn] missing tie source $stale, skipping K=$K stale"; i=$((i+1)); continue; }
      fi
      ( launch_arm "$tag" "$K" "$tie" "$mode" "$gsub" $((base + j)) "$stale" ) &
      bg+=($!); j=$((j+1)); i=$((i+1))
    done
    for b in "${bg[@]}"; do wait "$b"; done
  done
}

for K in $KS; do run_K "$K"; done

echo "[$(date +%H:%M:%S)] PUBK_SWEEP_DONE"
sumtsv=$LOGDIR/summary.tsv
printf "K\tarm\tval_ppl\tpeak_mem_GB\ttok_s\n" > "$sumtsv"
echo "==== Summary (subword val ppl | peak mem (max over ranks) | tok/s) ===="
for K in $KS; do
  for a in tie untie stale random; do
    ftail=$LOGDIR/wt103pub_k${K}_${a}_r$((K-1)).log
    ppl=$(grep -a FINAL "$ftail" 2>/dev/null | awk -F'val ppl: ' 'NF>1{print $2+0}' | tail -1)
    # mem/tok/s: last epoch line across all rank logs of this arm (mem is a per-rank list; take max = peak)
    eline=$(grep -ahE 'tok/s=[0-9,]+' $LOGDIR/wt103pub_k${K}_${a}_r*.log 2>/dev/null | tail -1)
    tps=$(printf '%s' "$eline" | grep -oE 'tok/s=[0-9,]+' | tail -1 | sed 's#tok/s=##; s/,//g')
    mem=$(printf '%s' "$eline" | grep -oE 'mem=\[[^]]*\]' | grep -oE '[0-9]+\.[0-9]+' | sort -g | tail -1)
    printf "  K=%-2s %-7s ppl=%-7s mem=%-6s GB  tok/s=%s\n" "$K" "$a" "${ppl:--}" "${mem:--}" "${tps:--}"
    printf "%s\t%s\t%s\t%s\t%s\n" "$K" "$a" "${ppl:--}" "${mem:--}" "${tps:--}" >> "$sumtsv"
  done
done
echo "TSV saved: $sumtsv (K / arm / val_ppl / peak_mem_GB / tok_s)"
echo "[mem note] peak_mem = max of per-rank (per-module process) peaks. K<=$NG is 1 module/GPU = true GPU peak;"
echo "          K>$NG (e.g. K=8, 2 modules/GPU): this is the single-module peak; actual GPU usage ~ sum of both modules on the card (~2x), check nvidia-smi for the true value."
echo "          (the single-module peak falling with K is itself the measure of 'memory sharded by K'.)"
echo "Reference: K=2 tie ~ 23.61 (subword val). Per K report channel(untie-tie)/random penalty/stale + mem/tok/s."
