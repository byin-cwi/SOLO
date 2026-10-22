#!/bin/bash
# run_wt103_pubrecipe_tiesync.sh - shared-readout refresh period sweep, TIE_SYNC in {1,10,50,100,200}, on the public-recipe
#   WT-103 config (bert/16L/d410; lm_hz_tievar.py pipeline, one GPU per rank). For a single GPU use run_wt103_pubrecipe_tiesync_sp.sh.
# Recipe = tie arm of run_wt103_pubrecipe_ksweep.sh except EPOCHS=1 (default) and TIE_SYNC per arm.
# Intermediate heads read a broadcast copy of the terminal fc (buffer, no grad); the terminal rank re-broadcasts it every
#   TIE_SYNC steps, after an initial broadcast before training. TIE_SYNC=1 matches the 1-step-lag detach of lm_solo.py.
#   Each broadcast is a collective (a pipeline drain), so tok/s drops as TIE_SYNC shrinks; the summary reports both.
# K=4 by default (4 GPUs, 1 segment each); arms run sequentially on all GPUS. Tags with a FINAL line are skipped.
# Requires $DATA/bert_cache_wikitext-103-raw-v1.pt (same cache as the public-recipe K sweep).
# Usage:
#   GPUS=0,1,2,3 nohup bash run_wt103_pubrecipe_tiesync.sh > logs_pubts/sweep.log 2>&1 &
#   Summary only, after runs finish: SYNCS="1 10 50 100 200" bash run_wt103_pubrecipe_tiesync.sh
# Knobs: SYNCS (default "100 1 10 50 200") K EPOCHS SEED BACKEND GPUS PORT0 RANK_STAGGER FAIL_TAIL DRY=1 (print commands only)
cd "$(dirname "$0")"
PY=${PY:-python}; LOGDIR=${LOGDIR:-logs_pubts}; CKPTDIR=${CKPTDIR:-ckpt_pubts}; DATA=${DATA:-./data/wikitext}
mkdir -p "$LOGDIR" "$CKPTDIR"
GPUS=${GPUS:-0,1,2,3}; K=${K:-4}; SYNCS=${SYNCS:-"100 1 10 50 200"}; EP=${EPOCHS:-1}; SEED=${SEED:-42}
BACKEND=${BACKEND:-nccl}; DRY=${DRY:-0}
IFS=',' read -ra G <<< "$GPUS"; NG=${#G[@]}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1} TORCHINDUCTOR_COMPILE_THREADS=1
port0=${PORT0:-$((20000 + RANDOM % 20000))}
[ $((16 % K)) -eq 0 ] || { echo "[abort] NLAYER=16 must be divisible by K=$K"; exit 1; }
CACHE=$DATA/bert_cache_wikitext-103-raw-v1.pt
[ "$DRY" = 1 ] || [ -f "$CACHE" ] || { echo "[abort] missing bert cache $CACHE (same file as the public-recipe K sweep; build the cache first)"; exit 1; }
echo "[cfg] TIE_SYNC sweep SYNCS='$SYNCS' K=$K EP=$EP GPUS=$GPUS (NG=$NG) SEED=$SEED BACKEND=$BACKEND DRY=$DRY"

join_k(){ local v=$1 n=$2 o=""; for ((i=0; i<n; i++)); do o+="$v,"; done; echo "${o%,}"; }
HZ=$(join_k 1 $((K-1))); TK=$(join_k 0 $((K-1)))
# Same as C in launch_arm of run_wt103_pubrecipe_ksweep.sh (EPOCHS defaults to 1)
COMMON="DATA=$DATA TOKENIZER=bert DSET=wikitext-103-raw-v1 NLAYER=16 DMODEL=410 FFN=2100 NHEAD=10 \
SEQ=256 BS=64 LR=2.5e-4 WARMUP=1000 CLIP=0.25 WD=0.0 LS=0.0 DROP=0.1 EPOCHS=$EP SEED=$SEED HEAD_DEPTH=2 \
MODE=mfb BETA=1 PRIOR=unigram STAGED=1 TK=$TK MU=0 AUX=1 KAPPA=0.7 HZ_H=$HZ BUCKET_BY=h COMPILE=1 BACKEND=$BACKEND"
echo "[recipe] $COMMON"

launch_arm(){   # $1=tag $2=TIE_SYNC $3=port
  local tag=$1 ts=$2 pt=$3
  local flog=$LOGDIR/${tag}_r$((K-1)).log
  if grep -qa FINAL "$flog" 2>/dev/null; then
    echo "[skip] $tag: $(grep -a FINAL "$flog" | tail -1)"; return 0; fi
  echo "[$(date +%H:%M:%S)] START $tag (K=$K TIE_SYNC=$ts gpus=$GPUS port=$pt $([ $K -gt $NG ] && echo '2 seg/GPU' || echo '1 seg/GPU'))"
  local pids=() r cmd
  for r in $(seq 0 $((K-1))); do
    cmd="env $COMMON WORLD_SIZE=$K RANK=$r LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=${G[$((r % NG))]} MASTER_PORT=$pt \
TIE_AUX=1 TIE_MODE=trained TIE_SYNC=$ts TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_ts_${tag}_r${r} OUT=$CKPTDIR/${tag}.pt \
$PY -u lm_hz_tievar.py"
    if [ "$DRY" = 1 ]; then echo "  [dry] $cmd > $LOGDIR/${tag}_r${r}.log"; continue; fi
    $cmd > "$LOGDIR/${tag}_r${r}.log" 2>&1 &
    pids+=($!); sleep "${RANK_STAGGER:-0}"
  done
  [ "$DRY" = 1 ] && return 0
  local failed=0 p; for p in "${pids[@]}"; do wait "$p" || failed=1; done
  if [ $failed -eq 1 ]; then
    echo "[FAIL] $tag a rank exited non-zero -> killing remaining processes of this arm, continuing with next arm; see $LOGDIR/${tag}_r*.log (if NCCL hangs, rerun this arm alone with BACKEND=gloo)"
    for r in $(seq 0 $((K-1))); do echo "  -- r$r --"; tail -${FAIL_TAIL:-8} "$LOGDIR/${tag}_r${r}.log" 2>/dev/null; done
    kill -9 "${pids[@]}" 2>/dev/null; return 1
  fi
  echo "[$(date +%H:%M:%S)] DONE  $tag: $(grep -a FINAL "$flog" | tail -1)"
}

i=0; nfail=0
for ts in $SYNCS; do
  case "$ts" in ''|*[!0-9]*) echo "[abort] TIE_SYNC must be a positive integer, got '$ts'"; exit 1;; esac
  [ "$ts" -ge 1 ] || { echo "[abort] TIE_SYNC must be >=1, got '$ts'"; exit 1; }
  launch_arm "wt103pub_k${K}_ts${ts}" "$ts" $((port0 + i)) || nfail=$((nfail + 1))
  i=$((i + 1))
done
echo "[$(date +%H:%M:%S)] TIESYNC_SWEEP_DONE (failed arms $nfail)"

# ---- Summary: FINAL line of the terminal rank (val/test) + last epoch line (wppl/exits/tok/s/time/memory) ----
parse_arm(){  # $1=TIE_SYNC -> "val wppl exits test tok_s sec mem" (tab-separated, missing = -)
  local f=$LOGDIR/wt103pub_k${K}_ts$1_r$((K-1)).log
  local fl el val test wppl exits tps sec mem
  fl=$(grep -a FINAL "$f" 2>/dev/null | tail -1); el=$(grep -a '^\[epoch' "$f" 2>/dev/null | tail -1)
  val=$(printf '%s' "$fl" | awk -F'val ppl: ' 'NF>1{print $2+0}')
  test=$(printf '%s' "$fl" | grep -oE 'test=[0-9.]+' | cut -d= -f2)
  wppl=$(printf '%s' "$el" | grep -oE 'wppl=[0-9.]+' | head -1 | cut -d= -f2)
  exits=$(printf '%s' "$el" | grep -oE "exits=\[[^]]*\]" | head -1 | sed "s/exits=//; s/'//g; s/ //g")
  tps=$(printf '%s' "$el" | grep -oE 'tok/s=[0-9,]+' | sed 's#tok/s=##; s/,//g')
  sec=$(printf '%s' "$el" | grep -oE '\([0-9]+s\)$' | tr -d '()s')
  mem=$(printf '%s' "$el" | grep -oE 'mem=\[[^]]*\]' | grep -oE '[0-9]+\.[0-9]+' | sort -g | tail -1)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${val:--}" "${wppl:--}" "${exits:--}" "${test:--}" "${tps:--}" "${sec:--}" "${mem:--}"
}
sumtsv=$LOGDIR/summary_k${K}_ep${EP}.tsv
printf "TIE_SYNC\tval_ppl\twppl\texits\ttest_ppl\ttok_s\tsec\tpeak_mem_GB\tdelta_vs_100\n" > "$sumtsv"
base=$(parse_arm 100 | cut -f1)
echo "==== Summary K=$K EP=$EP: subword val ppl | word-level wppl | per-exit val | test | tok/s | time | peak mem (max over ranks) | delta vs TIE_SYNC=100 ===="
for ts in $(printf '%s\n' $SYNCS | sort -n); do
  IFS=$'\t' read -r val wppl exits test tps sec mem <<< "$(parse_arm "$ts")"
  d="-"
  if [ "$val" != "-" ] && [ -n "$base" ] && [ "$base" != "-" ]; then
    d=$(awk -v a="$val" -v b="$base" 'BEGIN{printf "%+.2f (%+.1f%%)", a-b, (a-b)/b*100}'); fi
  printf "  TIE_SYNC=%-4s val=%-7s wppl=%-7s exits=%-28s test=%-7s tok/s=%-8s %6ss  mem=%sGB  Δ=%s\n" \
         "$ts" "$val" "$wppl" "$exits" "$test" "$tps" "$sec" "$mem" "$d"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$ts" "$val" "$wppl" "$exits" "$test" "$tps" "$sec" "$mem" "$d" >> "$sumtsv"
done
echo "TSV saved: $sumtsv"
echo "The tie_sync used by each arm is printed on the 'pipeline K=' line of its r0 log."
