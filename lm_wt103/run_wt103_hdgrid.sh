#!/bin/bash
# run_wt103_hdgrid.sh - aux-head depth HD x scale sweep on WT-103 (lm_hz_tievar.py).
#
# Grid: HD in {0,1,2,3,5} x scale in {40m,128m,256m} x arm in {tie=SOLO, untie=PRIV}, plus a noise-floor rerun
#   (NOISE_SEED) of NOISE_HDS at NOISE_SCALE. HD = number of full Blocks in the aux head of each non-final module.
# Protocol: word level V=32768, 1 epoch, K=4, TIE_SYNC=1; other hyperparameters as LCOMMON in run_wt103_scalept.sh.
# SUMMARY reports terminal val ppl, C = PRIV - SOLO, delta ppl per head layer, and per-exit best_exits.
# Cells with a FINAL line are skipped, so rerunning resumes. SAVE=0 by default (no checkpoints).
# Two parallel streams on 8 GPUs (noise cells run only in the stream whose SCALES contains NOISE_SCALE):
#     tmux new-session -d -s hdA 'GPUS=0,1,2,3 SCALES="40m 128m" bash run_wt103_hdgrid.sh'
#     tmux new-session -d -s hdB 'GPUS=4,5,6,7 SCALES="256m" bash run_wt103_hdgrid.sh'
#
# Usage:
#   bash run_wt103_hdgrid.sh SMOKE          # smoke test (FAKE_DATA, STEPS=3, no official log)
#   GPUS=4,5,6,7 nohup bash run_wt103_hdgrid.sh > logs_hdgrid/driver.log 2>&1 &
#   bash run_wt103_hdgrid.sh PROGRESS       # done/running/stalled/pending matrix, live processes, failures
#   bash run_wt103_hdgrid.sh SUMMARY        # tables from cells already on disk
# Knobs: SCALES HDS ARMS NOISE_HDS NOISE_SCALE SEED0 NOISE_SEED K TIE_SYNC SEQ BS LR WARMUP EPOCHS SAVE GPUS PORT0 COMPILE DRY=1
# Reference: BP terminal val ppl (BPref column) 40m 67.08, 128m 44.46, 256m 40.47.
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}; LOGDIR=${LOGDIR:-logs_hdgrid}; CKPTDIR=${CKPTDIR:-ckpt_hdgrid}; DATA=${DATA:-./data/wikitext}
mkdir -p "$LOGDIR" "$CKPTDIR"
SCALES=${SCALES:-"40m 128m 256m"}
HDS=${HDS:-"0 1 2 3 5"}
NOISE_HDS=${NOISE_HDS:-"0 2"}          # noise floor reruns only these columns at the cheapest scale
NOISE_SCALE=${NOISE_SCALE:-40m}
SEED0=${SEED0:-42}; NOISE_SEED=${NOISE_SEED:-43}
K=${K:-4}; TIE_SYNC=${TIE_SYNC:-1}; SEQ=${SEQ:-256}; BS=${BS:-64}
LR=${LR:-3e-4}; WARMUP=${WARMUP:-1000}; EP=${EPOCHS:-1}
GPUS=${GPUS:-0,1,2,3}; SAVE=${SAVE:-0}; DRY=${DRY:-0}
ARMS=${ARMS:-"tie untie"}   # ARMS=tie runs the SOLO arm only
export CUDA_VISIBLE_DEVICES=$GPUS MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
port=${PORT0:-$(( 20000 + RANDOM % 20000 ))}

# Scale table: NLAYER DMODEL NHEAD
cfg_for () {
  case "$1" in
    40m)  echo "8 384 6"    ;;
    128m) echo "16 768 12"  ;;
    256m) echo "16 1024 16" ;;
    340m) echo "24 1024 16" ;;
    512m) echo "24 1280 16" ;;
    *) echo ""; return 1    ;;
  esac
}

# LCOMMON of run_wt103_scalept.sh with scale/HD/seed/TIE_SYNC as variables
common_for () {   # $1=NLAYER $2=DMODEL $3=NHEAD $4=SEED
  echo "DATA=$DATA TOKENIZER=word DSET=wikitext-103-raw-v1 VMAX=32768 \
NLAYER=$1 DMODEL=$2 NHEAD=$3 SEQ=$SEQ BS=$BS LR=$LR WARMUP=$WARMUP \
CLIP=0.5 WD=0.0 LS=0.0 DROP=0.1 EPOCHS=$EP SEED=$4 \
MODE=mfb BETA=1 PRIOR=unigram STAGED=1 TK=0,0,0 MU=0 AUX=1 KAPPA=0.7 HZ_H=1,1,1 BUCKET_BY=h COMPILE=${COMPILE:-1}"
}

run_cell () {   # $1=scale $2=hd $3=arm(tie|untie) $4=seed
  local scale=$1 hd=$2 armname=$3 seed=$4
  local cfg nl dm nh tag last tie_aux out t0 t1
  cfg=$(cfg_for "$scale") || { echo "[FAIL] unknown scale $scale"; return 1; }
  read -r nl dm nh <<< "$cfg"
  tag="wt103_${scale}_hd${hd}_${armname}"
  [ "$seed" != "$SEED0" ] && tag="${tag}_s${seed}"
  last="$LOGDIR/${tag}_r$((K-1)).log"
  if grep -qa FINAL "$last" 2>/dev/null; then
    echo "[skip] $tag: $(grep -a FINAL "$last" | tail -1)"; return 0; fi
  tie_aux=1; [ "$armname" = "untie" ] && tie_aux=0
  out=""; [ "$SAVE" = "1" ] && out="$CKPTDIR/${tag}.pt"
  if [ "$DRY" = "1" ]; then
    echo "[dry] $tag  NLAYER=$nl DMODEL=$dm NHEAD=$nh HD=$hd TIE_AUX=$tie_aux SEED=$seed"; return 0; fi
  port=$((port+1)); t0=$(date +%s)
  echo "[$(date '+%F %T')] START $tag (NLAYER=$nl DMODEL=$dm HD=$hd tie_aux=$tie_aux seed=$seed port=$port)"
  local pids=() r
  for r in $(seq 0 $((K-1))); do
    env $(common_for "$nl" "$dm" "$nh" "$seed") \
        WORLD_SIZE=$K MASTER_PORT=$port HEAD_DEPTH=$hd TIE_AUX=$tie_aux TIE_MODE=trained TIE_SYNC=$TIE_SYNC \
        TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_hd_${tag}_r${r} OUT="$out" RANK=$r LOCAL_RANK=$r \
        $PY -u lm_hz_tievar.py > "$LOGDIR/${tag}_r${r}.log" 2>&1 &
    pids+=($!)
  done
  local failed=0 p
  for p in "${pids[@]}"; do wait "$p" || failed=1; done
  t1=$(date +%s)
  if [ $failed -eq 1 ]; then
    echo "[FAIL] $tag ($(( (t1-t0)/60 )) min); see tail of $LOGDIR/${tag}_r*.log (on OOM, rerun this cell with BS=32 for this scale)"
    kill -9 "${pids[@]}" 2>/dev/null; return 1
  fi
  echo "[$(date '+%F %T')] DONE  $tag ($(( (t1-t0)/60 )) min): $(grep -a FINAL "$last" | tail -1)"
}

ppl_of () {   # $1=tag -> terminal val ppl (empty = not run)
  grep -a FINAL "$LOGDIR/$1_r$((K-1)).log" 2>/dev/null | tail -1 | grep -oE 'val ppl: [0-9.]+' | tail -1 | awk '{print $3}'
}

cell_state () {   # $1=tag -> DONE | RUN | STALE | PEND
  local f="$LOGDIR/$1_r$((K-1)).log" age now mt
  [ -f "$f" ] || { echo PEND; return; }
  grep -qa FINAL "$f" && { echo DONE; return; }
  now=$(date +%s); mt=$(stat -c %Y "$f" 2>/dev/null || stat -f %m "$f" 2>/dev/null || echo 0)
  age=$(( now - mt ))
  [ "$age" -lt 900 ] && echo RUN || echo STALE
}

progress () {     # matrix + counts + live processes + failures
  local s hd a tag st total=0 nd=0 nr=0 ns=0 np=0 noise_here
  case " $SCALES " in *" $NOISE_SCALE "*) noise_here=1 ;; *) noise_here=0 ;; esac
  echo "==================== Progress (D=done R=running S=stalled>15min .=not started) ===================="
  for a in tie untie; do
    echo "---- $([ "$a" = tie ] && echo 'SOLO (tie)' || echo 'PRIV (untie)') ----"
    printf "%-10s" "scale"; for hd in $HDS; do printf "%8s" "HD=$hd"; done; echo
    for s in $SCALES; do
      printf "%-10s" "$s"
      for hd in $HDS; do
        st=$(cell_state "wt103_${s}_hd${hd}_${a}"); total=$((total+1))
        case $st in DONE) nd=$((nd+1)); printf "%8s" "D";; RUN) nr=$((nr+1)); printf "%8s" "R";;
                    STALE) ns=$((ns+1)); printf "%8s" "S";; *) np=$((np+1)); printf "%8s" ".";; esac
      done; echo
    done
  done
  if [ "$noise_here" = 1 ]; then
    printf "%-10s" "${NOISE_SCALE}_s${NOISE_SEED}"
    for hd in $NOISE_HDS; do for a in tie untie; do
      st=$(cell_state "wt103_${NOISE_SCALE}_hd${hd}_${a}_s${NOISE_SEED}"); total=$((total+1))
      case $st in DONE) nd=$((nd+1)); printf " hd%s/%s:D" "$hd" "$a";; RUN) nr=$((nr+1)); printf " hd%s/%s:R" "$hd" "$a";;
                  STALE) ns=$((ns+1)); printf " hd%s/%s:S" "$hd" "$a";; *) np=$((np+1)); printf " hd%s/%s:." "$hd" "$a";; esac
    done; done; echo
  fi
  echo
  echo "Count: $nd/$total done | $nr running | $ns stalled | $np not started"
  echo "Alive: lm_hz_tievar processes $(pgrep -fc lm_hz_tievar.py 2>/dev/null || echo 0) (should be $K per stream) | tmux: $(tmux ls 2>/dev/null | tr '\n' ' ' || echo 'none')"
  echo "Finished: $(grep -ah HDGRID_DONE "$LOGDIR"/driver*.log 2>/dev/null | wc -l) drivers printed HDGRID_DONE"
  local fails; fails=$(grep -ah "^\[FAIL\]" "$LOGDIR"/driver*.log 2>/dev/null)
  [ -n "$fails" ] && { echo "Failed:"; echo "$fails" | sed 's/^/  /'; }
  local run_tags; run_tags=$(for s in $SCALES; do for hd in $HDS; do for a in tie untie; do
      tag="wt103_${s}_hd${hd}_${a}"; [ "$(cell_state "$tag")" = RUN ] && echo "$tag"; done; done; done)
  if [ -n "$run_tags" ]; then
    echo "Running cells (last line):"
    echo "$run_tags" | while read -r tag; do
      echo "  $tag: $(tail -1 "$LOGDIR/${tag}_r$((K-1)).log" 2>/dev/null | cut -c1-140)"; done
  fi
  [ $((nd)) -eq $((total)) ] && echo "==> all $total cells done, run SUMMARY to produce the table."
}

summary () {
  echo
  echo "==================== HD x scale summary (terminal val ppl, TIE_SYNC=$TIE_SYNC, 1ep) ===================="
  local s hd a tag v line
  for a in tie untie; do
    [ "$a" = tie ] && echo "---- SOLO (tie, shared readout) ----" || echo "---- PRIV (untie, private readout) ----"
    printf "%-8s" "scale"; for hd in $HDS; do printf "%10s" "HD=$hd"; done; printf "%10s\n" "BPref"
    for s in $SCALES; do
      printf "%-8s" "$s"
      for hd in $HDS; do v=$(ppl_of "wt103_${s}_hd${hd}_${a}"); printf "%10s" "${v:---}"; done
      case "$s" in 40m) printf "%10s\n" 67.08;; 128m) printf "%10s\n" 44.46;; 256m) printf "%10s\n" 40.47;;
                   340m) printf "%10s\n" 39.51;; 512m) printf "%10s\n" 36.81;; *) printf "%10s\n" "--";; esac
    done
    echo
  done
  echo "---- channel C = PRIV - SOLO (absolute / relative%) ----"
  printf "%-8s" "scale"; for hd in $HDS; do printf "%14s" "HD=$hd"; done; echo
  for s in $SCALES; do
    printf "%-8s" "$s"
    for hd in $HDS; do
      local t u; t=$(ppl_of "wt103_${s}_hd${hd}_tie"); u=$(ppl_of "wt103_${s}_hd${hd}_untie")
      if [ -n "$t" ] && [ -n "$u" ]; then
        printf "%14s" "$(awk -v t="$t" -v u="$u" 'BEGIN{printf "%+.2f/%+.1f%%", u-t, 100*(u-t)/t}')"
      else printf "%14s" "--"; fi
    done; echo
  done
  echo
  echo "---- per-layer head exchange rate delta ppl/layer (SOLO, adjacent HD diff / layer diff; vs BP trunk -0.12 ppl/layer) ----"
  printf "%-8s" "scale"; local prev
  for hd in $HDS; do printf "%12s" "→HD=$hd"; done; echo
  for s in $SCALES; do
    printf "%-8s" "$s"; prev=""; prev_hd=""
    for hd in $HDS; do
      local v; v=$(ppl_of "wt103_${s}_hd${hd}_tie")
      if [ -n "$prev" ] && [ -n "$v" ] && [ -n "$prev_hd" ]; then
        printf "%12s" "$(awk -v a="$prev" -v b="$v" -v da="$prev_hd" -v db="$hd" 'BEGIN{printf "%+.3f", (b-a)/(db-da)}')"
      else printf "%12s" "--"; fi
      [ -n "$v" ] && { prev=$v; prev_hd=$hd; }
    done; echo
  done
  echo
  echo "---- noise floor (${NOISE_SCALE}, seed $SEED0 vs $NOISE_SEED) ----"
  for hd in $NOISE_HDS; do for a in tie untie; do
    local v1 v2; v1=$(ppl_of "wt103_${NOISE_SCALE}_hd${hd}_${a}"); v2=$(ppl_of "wt103_${NOISE_SCALE}_hd${hd}_${a}_s${NOISE_SEED}")
    [ -n "$v1" ] && [ -n "$v2" ] && echo "  hd$hd $a: $v1 vs $v2  (|Δ|=$(awk -v a="$v1" -v b="$v2" 'BEGIN{printf "%.2f", (a>b)?a-b:b-a}'))"
  done; done
  echo
  echo "---- per-exit profile (best_exits) ----"
  for s in $SCALES; do for hd in $HDS; do for a in tie untie; do
    tag="wt103_${s}_hd${hd}_${a}"; line=$(grep -a FINAL "$LOGDIR/${tag}_r$((K-1)).log" 2>/dev/null | tail -1)
    [ -n "$line" ] && echo "  $tag: ${line#*best_exits=}"
  done; done; done
}

smoke () {   # tiny fake-data run of each HD branch; no official log
  echo "[smoke] tiny K=$K, HD=0/2/5 x tie/untie, FAKE_DATA, STEPS=3"
  local hd a p=$((port+500)) pids=() r ok=1
  for hd in 0 2 5; do for a in tie untie; do
    local tie_aux=1; [ "$a" = untie ] && tie_aux=0
    p=$((p+1)); pids=()
    for r in $(seq 0 $((K-1))); do
      env DATA=$DATA TOKENIZER=word DSET=wikitext-103-raw-v1 VMAX=512 FAKE_DATA=1 STEPS=3 \
          NLAYER=4 DMODEL=64 NHEAD=2 SEQ=32 BS=4 LR=3e-4 WARMUP=1 CLIP=0.5 WD=0.0 LS=0.0 DROP=0.1 \
          EPOCHS=1 SEED=$SEED0 MODE=mfb BETA=1 PRIOR=unigram STAGED=1 TK=0,0,0 MU=0 AUX=1 KAPPA=0.7 \
          HZ_H=1,1,1 BUCKET_BY=h COMPILE=0 WORLD_SIZE=$K MASTER_PORT=$p HEAD_DEPTH=$hd \
          TIE_AUX=$tie_aux TIE_MODE=trained TIE_SYNC=$TIE_SYNC OUT="" RANK=$r LOCAL_RANK=$r \
          $PY -u lm_hz_tievar.py > "$LOGDIR/smoke_hd${hd}_${a}_r${r}.log" 2>&1 &
      pids+=($!)
    done
    local failed=0 q; for q in "${pids[@]}"; do wait "$q" || failed=1; done
    if [ $failed -eq 1 ]; then echo "  [FAIL] hd=$hd $a - see $LOGDIR/smoke_hd${hd}_${a}_r*.log"; ok=0
    else echo "  [ok]   hd=$hd $a: $(grep -a FINAL "$LOGDIR/smoke_hd${hd}_${a}_r$((K-1)).log" | tail -1)"; fi
  done; done
  [ $ok = 1 ] && echo "[smoke] all green, grid can be launched." || echo "[smoke] failures found, fix before launching."
}

case "${1:-grid}" in
  SMOKE|smoke)       smoke; exit 0 ;;
  SUMMARY|summary)   summary; exit 0 ;;
  PROGRESS|progress) progress; exit 0 ;;
esac

echo "[cfg] SCALES='$SCALES' HDS='$HDS' K=$K TIE_SYNC=$TIE_SYNC SEQ=$SEQ BS=$BS EPOCHS=$EP GPUS=$GPUS SAVE=$SAVE"
# Order: all cells at NOISE_SCALE (both seeds) first, then the remaining scales small to large. A failed cell does not block.
case " $SCALES " in *" $NOISE_SCALE "*) noise_here=1 ;; *) noise_here=0 ;; esac
if [ "$noise_here" = 1 ]; then
  for hd in $HDS; do
    for a in $ARMS; do run_cell "$NOISE_SCALE" "$hd" "$a" "$SEED0" || true; done
  done
  for hd in $NOISE_HDS; do
    for a in $ARMS; do run_cell "$NOISE_SCALE" "$hd" "$a" "$NOISE_SEED" || true; done
  done
else
  echo "[note] NOISE_SCALE=$NOISE_SCALE not in SCALES, this stream skips the noise floor (handled by the stream running that scale)"
fi
for s in $SCALES; do
  [ "$s" = "$NOISE_SCALE" ] && [ "$noise_here" = 1 ] && continue
  for hd in $HDS; do
    for a in $ARMS; do run_cell "$s" "$hd" "$a" "$SEED0" || true; done
  done
done

echo "[$(date '+%F %T')] HDGRID_DONE"
summary
