#!/bin/bash
# run_wt103_pubrecipe_tiesync_sp.sh - single-GPU, single-process TIE_SYNC sweep {1,10,50,100,200} on the public-recipe
#   WT-103 config (bert/16L/d410; lm_solo.py).
# Recipe = COMMON of run_wt103_pubrecipe_ksweep_sp.sh except EPOCHS=1 (default) and TIE_SYNC per arm. CLIP/WARMUP use the
#   lm_solo.py defaults 1.0/200; pass CLIP=0.25 WARMUP=1000 to match the pipeline K sweep.
# TIE_SYNC=1: intermediate heads read a detached copy of the terminal fc every step (1-step lag, lm_solo.py default).
# TIE_SYNC=N>1: they read a frozen copy refreshed in place from the terminal fc every N optimizer steps (copied at step 0),
#   matching the pipeline broadcast. There is no broadcast overhead here, so throughput does not depend on N.
#   The params= line of N>1 arms includes the untrained V*d frozen copy.
# K=4 by default; arms run sequentially in ascending TIE_SYNC.
# Tag = wt103sp_k{K}_ts{N}[_s{SEED}][_ep{EPOCHS}] (no suffix for seed 42 / 1ep); the summary also finds the per-directory
#   layout LOGDIR_s{SEED}/wt103sp_k{K}_ts{N}.log. Tags with a FINAL line are skipped.
# Uses $DATA/bert_cache_wikitext-103-raw-v1.pt; if missing, lm_solo.py builds it (needs transformers and the dataset cache).
# Usage:
#   GPU=1 nohup bash run_wt103_pubrecipe_tiesync_sp.sh > logs_pubts/sweep_sp.log 2>&1 &            # seed 42
#   SEED=43 GPU=1 nohup bash run_wt103_pubrecipe_tiesync_sp.sh > logs_pubts/sweep_sp_s43.log 2>&1 &  # extra seed
#   SEEDS="42 43" bash run_wt103_pubrecipe_tiesync_sp.sh        # summary only, incl. the paired multi-seed table
#   EPOCHS=30 SYNCS="1" GPU=1 nohup bash run_wt103_pubrecipe_tiesync_sp.sh > logs_pubts/k4_ts1_ep30.log 2>&1 &   # full-budget single arm
# Knobs: SYNCS K EPOCHS SEED SEEDS GPU CLIP WARMUP FAIL_TAIL PY LOGDIR CKPTDIR DATA DRY=1 (print commands only)
cd "$(dirname "$0")"
PY=${PY:-python}; LOGDIR=${LOGDIR:-logs_pubts}; CKPTDIR=${CKPTDIR:-ckpt_pubts}; DATA=${DATA:-./data/wikitext}
mkdir -p "$LOGDIR" "$CKPTDIR"
GPU=${GPU:-1}; K=${K:-4}; SYNCS=${SYNCS:-"1 10 50 100 200"}; EP=${EPOCHS:-1}; SEED=${SEED:-42}; DRY=${DRY:-0}
SEEDS=${SEEDS:-$SEED}
[ $((16 % K)) -eq 0 ] || { echo "[abort] NLAYER=16 must be divisible by K=$K"; exit 1; }
grep -q 'TIE_SYNC = int(E("TIE_SYNC"' lm_solo.py 2>/dev/null || { echo "[abort] lm_solo.py in this directory has no TIE_SYNC knob"; exit 1; }
EXTRA=""; [ -n "$CLIP" ] && EXTRA="$EXTRA CLIP=$CLIP"; [ -n "$WARMUP" ] && EXTRA="$EXTRA WARMUP=$WARMUP"
# COMMON of run_wt103_pubrecipe_ksweep_sp.sh (EPOCHS defaults to 1; EXTRA is set only when CLIP/WARMUP are given)
COMMON="MODE=mfb STAGED=1 DATA=$DATA TOKENIZER=bert DSET=wikitext-103-raw-v1 NLAYER=16 DMODEL=410 FFN=2100 NHEAD=10 \
SEQ=256 BS=64 LR=2.5e-4 WD=0.0 DROP=0.1 LS=0.0 HEAD_DEPTH=2 EPOCHS=$EP SEED=$SEED \
BETA=1 PRIOR=unigram MU=0 AUX=1 KAPPA=0.7$EXTRA"
echo "[cfg] SP TIE_SYNC sweep SYNCS='$SYNCS' K=$K EP=$EP GPU=$GPU SEED=$SEED (summary SEEDS='$SEEDS') DRY=$DRY"
echo "[recipe] $COMMON"

tag_of(){  # $1=ts $2=seed -> wt103sp_k{K}_ts{N}[_s{seed}][_ep{EP}]  (seed 42 / 1ep get no suffix)
  local t=wt103sp_k${K}_ts$1; [ "$2" = 42 ] || t=${t}_s$2; [ "$EP" = 1 ] || t=${t}_ep$EP; echo "$t"; }
log_of(){  # $1=ts $2=seed -> log path; falls back to LOGDIR_s<seed>/<unsuffixed tag>
  local f=$LOGDIR/$(tag_of "$1" "$2").log
  [ -f "$f" ] || { [ "$EP" = 1 ] && { local g=${LOGDIR}_s$2/wt103sp_k${K}_ts$1.log; [ -f "$g" ] && f=$g; }; }
  echo "$f"; }

launch(){  # $1=TIE_SYNC (seed = $SEED)
  local ts=$1 tag f fchk cmd rc
  tag=$(tag_of "$ts" "$SEED"); f=$LOGDIR/$tag.log; fchk=$(log_of "$ts" "$SEED")
  if grep -qa FINAL "$fchk" 2>/dev/null; then echo "[skip] $tag: $(grep -a FINAL "$fchk" | tail -1)"; return 0; fi
  cmd="env CUDA_VISIBLE_DEVICES=$GPU $COMMON K=$K TIE_AUX=1 TIE_SYNC=$ts OUT=$CKPTDIR/$tag.pt $PY -u lm_solo.py"
  echo "[$(date +%H:%M:%S)] START $tag (K=$K TIE_SYNC=$ts seed=$SEED gpu=$GPU)"
  if [ "$DRY" = 1 ]; then echo "  [dry] $cmd > $f"; return 0; fi
  if $cmd > "$f" 2>&1; then
    echo "[$(date +%H:%M:%S)] DONE  $tag: $(grep -a FINAL "$f" | tail -1)"
  else
    rc=$?; echo "[FAIL] $tag rc=$rc -> continuing with next arm; tail of $f:"; tail -${FAIL_TAIL:-8} "$f"; return 1
  fi
}

nfail=0
for ts in $SYNCS; do
  case "$ts" in ''|*[!0-9]*|0) echo "[abort] TIE_SYNC must be a positive integer, got '$ts'"; exit 1;; esac
  launch "$ts" || nfail=$((nfail + 1))
done
echo "[$(date +%H:%M:%S)] TIESYNC_SP_SWEEP_DONE seed=$SEED (failed arms $nfail)"

# ---- Summary: FINAL line (val/test) + last epoch line (wppl/exits/ens/mem/time) + startup lines (train_tokens, [tie_sync] check) ----
parse_arm(){  # $1=TIE_SYNC $2=seed -> val wppl exits ens test mem sec tok_s sync_chk (tab-separated, missing = -)
  local f fl el val test wppl exits ens mem sec ntok tps chk; f=$(log_of "$1" "$2")
  [ -f "$f" ] || { printf -- '-\t-\t-\t-\t-\t-\t-\t-\t-\n'; return; }
  fl=$(grep -a '^FINAL' "$f" | tail -1); el=$(grep -a '^\[epoch' "$f" | tail -1)
  val=$(printf '%s' "$fl" | awk -F'val ppl: ' 'NF>1{print $2+0}')
  test=$(printf '%s' "$fl" | grep -oE 'test=[0-9.]+' | cut -d= -f2)
  wppl=$(printf '%s' "$el" | grep -oE 'wppl=[0-9.]+' | head -1 | cut -d= -f2)
  exits=$(printf '%s' "$el" | grep -oE "exits=\[[^]]*\]" | head -1 | sed "s/exits=//; s/'//g; s/ //g")
  ens=$(printf '%s' "$el" | grep -oE ' ens=[0-9.]+' | head -1 | cut -d= -f2)
  mem=$(printf '%s' "$el" | grep -oE 'mem=[0-9.]+GB' | grep -oE '[0-9.]+')
  sec=$(printf '%s' "$el" | grep -oE '\([0-9]+s\)$' | tr -d '()s')
  ntok=$(grep -aoE 'train_tokens=[0-9]+' "$f" | tail -1 | grep -oE '[0-9]+')
  tps=-; [ -n "$ntok" ] && [ -n "$sec" ] && [ "$sec" -gt 0 ] && tps=$((ntok / sec))
  chk=$(grep -a '^\[tie_sync\]' "$f" | grep -oE 'every [0-9]+ opt' | grep -oE '[0-9]+' | tail -1)
  [ -z "$chk" ] && { [ "$1" = 1 ] && chk=live || chk=MISSING; }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${val:--}" "${wppl:--}" "${exits:--}" "${ens:--}" "${test:--}" "${mem:--}" "${sec:--}" "$tps" "$chk"
}
delta(){ [ "$1" != - ] && [ -n "$2" ] && [ "$2" != - ] && awk -v a="$1" -v b="$2" 'BEGIN{printf "%+.2f (%+.1f%%)", a-b, (a-b)/b*100}' || echo "-"; }
SORTED=$(printf '%s\n' $SYNCS | sort -n)

# (a) single-seed table for the current seed
sumtsv=$LOGDIR/summary_sp_k${K}_ep${EP}$([ "$SEED" = 42 ] || echo "_s$SEED").tsv
printf "TIE_SYNC\tval_ppl\twppl\texits\tens\ttest_ppl\tpeak_mem_GB\tsec\ttok_s\tsync_chk\tdelta_vs_1\n" > "$sumtsv"
base=$(parse_arm 1 "$SEED" | cut -f1)
echo "==== Summary seed=$SEED K=$K EP=$EP: subword val ppl | word-level wppl | per-exit | ens | test | peak mem | time | tok/s~train_tokens/epoch sec | refresh period checked in log | delta vs TIE_SYNC=1 ===="
for ts in $SORTED; do
  IFS=$'\t' read -r val wppl exits ens test mem sec tps chk <<< "$(parse_arm "$ts" "$SEED")"
  d=$(delta "$val" "$base")
  printf "  TIE_SYNC=%-4s val=%-7s wppl=%-7s exits=%-28s ens=%-7s test=%-7s mem=%-5sGB %6ss tok/s≈%-7s sync=%-7s Δ=%s\n" \
         "$ts" "$val" "$wppl" "$exits" "$ens" "$test" "$mem" "$sec" "$tps" "$chk" "$d"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$ts" "$val" "$wppl" "$exits" "$ens" "$test" "$mem" "$sec" "$tps" "$chk" "$d" >> "$sumtsv"
done
echo "TSV saved: $sumtsv"

# (b) paired multi-seed table (SEEDS with >=2 seeds): per-seed val, mean +/- half-range, paired delta vs same-seed TIE_SYNC=1, test mean
stats(){ awk -v vals="$1" 'BEGIN{n=split(vals,a," "); k=0; s=0; mn=1e18; mx=-1e18
  for(i=1;i<=n;i++){ if(a[i]=="-"||a[i]=="") continue; k++; s+=a[i]; if(a[i]+0<mn)mn=a[i]+0; if(a[i]+0>mx)mx=a[i]+0 }
  if(k==0){printf "-"; exit}; if(k>1) printf "%.2f ± %.2f", s/k, (mx-mn)/2; else printf "%.2f", s/k }'; }
if [ "$(echo $SEEDS | wc -w)" -ge 2 ]; then
  seedtsv=$LOGDIR/summary_sp_k${K}_ep${EP}_seeds.tsv
  { printf "TIE_SYNC"; for s in $SEEDS; do printf "\tval_s%s" "$s"; done; printf "\tval_mean_hr"
    for s in $SEEDS; do printf "\tdelta_s%s" "$s"; done; printf "\tdelta_mean_hr\ttest_mean_hr\n"; } > "$seedtsv"
  echo; echo "==== Multi-seed paired summary SEEDS='$SEEDS' (val ppl; mean +- half-range; delta = paired diff vs TIE_SYNC=1 within the same seed) ===="
  for ts in $SORTED; do
    vals=""; tests=""; dels=""; vcells=""; dcells=""
    for s in $SEEDS; do
      v=$(parse_arm "$ts" "$s" | cut -f1); t=$(parse_arm "$ts" "$s" | cut -f5); b=$(parse_arm 1 "$s" | cut -f1)
      dv=-; [ "$v" != - ] && [ "$b" != - ] && dv=$(awk -v a="$v" -v b="$b" 'BEGIN{printf "%+.2f", a-b}')
      vals="$vals $v"; tests="$tests $t"; dels="$dels $dv"
      vcells="$vcells s$s=$v"; dcells="$dcells s$s=$dv"
    done
    printf "  TIE_SYNC=%-4s val:%-28s mean %-14s | Δ_vs_1:%-24s mean %-14s | test mean %s\n" \
           "$ts" "$vcells" "$(stats "$vals")" "$dcells" "$(stats "$dels")" "$(stats "$tests")"
    { printf "%s" "$ts"; for v in $vals; do printf "\t%s" "$v"; done; printf "\t%s" "$(stats "$vals")"
      for dv in $dels; do printf "\t%s" "$dv"; done; printf "\t%s\t%s\n" "$(stats "$dels")" "$(stats "$tests")"; } >> "$seedtsv"
  done
  echo "TSV saved: $seedtsv"
fi
echo "sync column = period read from the [tie_sync] log line (the TIE_SYNC=1 arm has no such line and is recorded as live); MISSING means TIE_SYNC was not enabled for that arm."
