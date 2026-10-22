#!/bin/bash
# run_wt103_2x2.sh - sharing x content 2x2 readout control on WT-103 (lm_hz_rot.py): 4 arms x HD {0,2}.
#
#                    content = trained W_T        content = random
#   shared coords    tie    (TIE_MODE=trained)    rshare (TIE_MODE=random_shared)
#   per-head coords  rot    (TIE_MODE=rot)        rand   (TIE_MODE=random)
#
#   sharing = 1/2[(rot-tie) + (rand-rshare)]   content = 1/2[(rshare-tie) + (rand-rot)]
#   interaction = (tie + rand) - (rshare + rot)
#
#   rot: head k uses W_T R_k with R_k a fixed random orthogonal matrix, so range(W_T R_k) = range(W_T).
#
# Protocol as run_wt103_hdgrid.sh (word level V=32768, 1ep, K=4, TIE_SYNC=1, SEQ=256 BS=64).
# SAVE=1 by default because align_probe.py needs the checkpoints; SAVE=0 if only ppl is needed.
#
# Usage:
#   bash run_wt103_2x2.sh SMOKE              # tiny smoke test (FAKE_DATA, each arm once)
#   GPUS=4,5,6,7 nohup bash run_wt103_2x2.sh > logs_2x2/driver.log 2>&1 &
#   bash run_wt103_2x2.sh PROGRESS | SUMMARY # SUMMARY prints the 2x2 and the main-effect decomposition
# Knobs: SCALE HDS ARMS SEED K TIE_SYNC SAVE GPUS PORT0 NCCL_TIMEOUT_H COMPILE TRAIN DRY=1
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}; TRAIN=${TRAIN:-lm_hz_rot.py}
LOGDIR=${LOGDIR:-logs_2x2}; CKPTDIR=${CKPTDIR:-ckpt_2x2}; DATA=${DATA:-./data/wikitext}
mkdir -p "$LOGDIR" "$CKPTDIR"
SCALE=${SCALE:-40m}; HDS=${HDS:-"0 2"}; ARMS=${ARMS:-"tie rot rshare rand"}
K=${K:-4}; TIE_SYNC=${TIE_SYNC:-1}; SEQ=${SEQ:-256}; BS=${BS:-64}
LR=${LR:-3e-4}; WARMUP=${WARMUP:-1000}; EP=${EPOCHS:-1}; SEED=${SEED:-42}
GPUS=${GPUS:-0,1,2,3}; SAVE=${SAVE:-1}; DRY=${DRY:-0}
TO_H=${NCCL_TIMEOUT_H:-3}   # process-group timeout in hours (lm_hz_rot.py default 0.5)
export CUDA_VISIBLE_DEVICES=$GPUS MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
port=${PORT0:-$(( 24000 + RANDOM % 4000 ))}

cfg_for () {                      # NLAYER DMODEL NHEAD
  case "$1" in
    40m)  echo "8 384 6"    ;;  64m)  echo "12 512 8"  ;;
    128m) echo "16 768 12"  ;;  256m) echo "16 1024 16" ;;
    *) echo ""; return 1    ;;
  esac
}
arm_cfg () {                      # TIE_AUX TIE_MODE  (arm name -> two env vars)
  case "$1" in
    tie)    echo "1 trained"       ;;   # shared coords + trained content (= SOLO in hdgrid)
    rot)    echo "1 rot"           ;;   # per-head rotation + trained content
    rshare) echo "1 random_shared" ;;   # shared coords + random content
    rand)   echo "1 random"        ;;   # independent random per head (= RAND)
    priv)   echo "0 trained"       ;;   # private trained readout (= PRIV; supplies ckpts for the probe)
    *) echo ""; return 1           ;;
  esac
}

run_cell () {                     # $1=hd $2=arm
  local hd=$1 armname=$2 cfg nl dm nh ac tie_aux mode tag last out t0 t1
  cfg=$(cfg_for "$SCALE") || { echo "[FAIL] unknown scale $SCALE"; return 1; }
  ac=$(arm_cfg "$armname") || { echo "[FAIL] unknown arm $armname"; return 1; }
  read -r nl dm nh <<< "$cfg"; read -r tie_aux mode <<< "$ac"
  tag="wt103_${SCALE}_hd${hd}_${armname}"
  [ "$SEED" != "42" ] && tag="${tag}_s${SEED}"
  last="$LOGDIR/${tag}_r$((K-1)).log"
  if grep -qa FINAL "$last" 2>/dev/null; then
    echo "[skip] $tag: $(grep -a FINAL "$last" | tail -1)"; return 0; fi
  out=""; [ "$SAVE" = "1" ] && out="$CKPTDIR/${tag}.pt"
  if [ "$DRY" = "1" ]; then
    echo "[dry] $tag  NLAYER=$nl DMODEL=$dm HD=$hd TIE_AUX=$tie_aux TIE_MODE=$mode SAVE=${SAVE}"; return 0; fi
  port=$((port+1)); t0=$(date +%s)
  echo "[$(date '+%F %T')] START $tag (HD=$hd tie_aux=$tie_aux mode=$mode port=$port)"
  local pids=() r
  for r in $(seq 0 $((K-1))); do
    env DATA=$DATA TOKENIZER=word DSET=wikitext-103-raw-v1 VMAX=32768 \
        NLAYER=$nl DMODEL=$dm NHEAD=$nh SEQ=$SEQ BS=$BS LR=$LR WARMUP=$WARMUP \
        CLIP=0.5 WD=0.0 LS=0.0 DROP=0.1 EPOCHS=$EP SEED=$SEED \
        MODE=mfb BETA=1 PRIOR=unigram STAGED=1 TK=0,0,0 MU=0 AUX=1 KAPPA=0.7 \
        HZ_H=1,1,1 BUCKET_BY=h COMPILE=${COMPILE:-1} NCCL_TIMEOUT_H=$TO_H \
        WORLD_SIZE=$K MASTER_PORT=$port HEAD_DEPTH=$hd TIE_AUX=$tie_aux TIE_MODE=$mode TIE_SYNC=$TIE_SYNC \
        TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_2x2_${tag}_r${r} OUT="$out" RANK=$r LOCAL_RANK=$r \
        $PY -u "$TRAIN" > "$LOGDIR/${tag}_r${r}.log" 2>&1 &
    pids+=($!)
  done
  local failed=0 p; for p in "${pids[@]}"; do wait "$p" || failed=1; done
  t1=$(date +%s)
  if [ $failed -eq 1 ]; then
    echo "[FAIL] $tag ($(( (t1-t0)/60 )) min); see tail of $LOGDIR/${tag}_r*.log"; kill -9 "${pids[@]}" 2>/dev/null; return 1; fi
  echo "[$(date '+%F %T')] DONE  $tag ($(( (t1-t0)/60 )) min): $(grep -a FINAL "$last" | tail -1)"
}

ppl_of () { grep -a FINAL "$LOGDIR/wt103_${SCALE}_hd$1_$2_r$((K-1)).log" 2>/dev/null | tail -1 \
            | grep -oE 'val ppl: [0-9.]+' | tail -1 | awk '{print $3}'; }

summary () {
  echo
  echo "======== 2x2 sharing x content ($SCALE, K=$K, TIE_SYNC=$TIE_SYNC, 1ep, seed $SEED) ========"
  local hd t r s w
  for hd in $HDS; do
    t=$(ppl_of "$hd" tie); r=$(ppl_of "$hd" rot); s=$(ppl_of "$hd" rshare); w=$(ppl_of "$hd" rand)
    echo "---- HD=$hd ----"
    printf "%-18s %14s %14s\n" "" "content=trained" "content=random"
    printf "%-18s %14s %14s\n" "shared coords"   "${t:---}"  "${s:---}"
    printf "%-18s %14s %14s\n" "per-head coords" "${r:---}"  "${w:---}"
    if [ -n "$t" ] && [ -n "$r" ] && [ -n "$s" ] && [ -n "$w" ]; then
      awk -v t="$t" -v r="$r" -v s="$s" -v w="$w" 'BEGIN{
        sh=((r-t)+(w-s))/2; ct=((s-t)+(w-r))/2; ix=(t+w)-(s+r);
        printf "  main effect sharing %+0.2f ppl (%+0.1f%%)   content %+0.2f ppl (%+0.1f%%)   interaction %+0.2f\n",
               sh, 100*sh/t, ct, 100*ct/t, ix;
        printf "  decomp rot-tie %+0.2f | rand-rshare %+0.2f | rshare-tie %+0.2f | rand-rot %+0.2f\n",
               r-t, w-s, s-t, w-r}'
    else echo "  (incomplete)"; fi
    p=$(ppl_of "$hd" priv); [ -n "$p" ] && echo "  reference PRIV=$p (private trained readout)"
  done
  echo
  echo
  echo "---- per-exit profile ----"
  for hd in $HDS; do for a in $(printf "%s\n" $ARMS priv | awk '!seen[$0]++'); do
    f="$LOGDIR/wt103_${SCALE}_hd${hd}_${a}_r$((K-1)).log"
    [ -f "$f" ] && echo "  hd$hd $a: $(grep -a FINAL "$f" | tail -1 | sed 's/.*best_exits=//; s/ test=.*//')"
  done; done
  echo
  echo "---- ckpt (for align_probe) ----"; ls -1 "$CKPTDIR"/*.r$((K-1)) 2>/dev/null | sed 's/^/  /' || echo "  (none; run with SAVE=0)"
}

progress () {
  local hd a f n=0 d=0
  echo "==== Progress ($SCALE) ===="
  for hd in $HDS; do for a in $ARMS; do
    f="$LOGDIR/wt103_${SCALE}_hd${hd}_${a}_r$((K-1)).log"; n=$((n+1))
    if grep -qa FINAL "$f" 2>/dev/null; then d=$((d+1)); echo "  hd$hd $a: DONE $(grep -a FINAL "$f"|tail -1|grep -oE 'val ppl: [0-9.]+')"
    elif [ -f "$f" ]; then echo "  hd$hd $a: RUN  $(tail -1 "$f" | cut -c1-90)"
    else echo "  hd$hd $a: ."; fi
  done; done
  echo "  $d/$n done | lm_hz_rot processes: $(pgrep -fc lm_hz_rot.py 2>/dev/null || echo 0)"
}

smoke () {
  echo "[smoke] tiny K=$K, each of four arms once (FAKE_DATA, STEPS=3, no official log)"
  local a ac tie_aux mode p=$((port+700)) r pids ok=1
  for a in $ARMS; do
    ac=$(arm_cfg "$a") || { echo "  [FAIL] unknown arm $a"; ok=0; continue; }
    read -r tie_aux mode <<< "$ac"; p=$((p+1)); pids=()
    for r in $(seq 0 $((K-1))); do
      env FAKE_DATA=1 STEPS=3 VMAX=512 DATA=$DATA TOKENIZER=word DSET=wikitext-103-raw-v1 \
          NLAYER=4 DMODEL=64 NHEAD=2 SEQ=32 BS=4 LR=3e-4 WARMUP=1 CLIP=0.5 WD=0.0 LS=0.0 DROP=0.1 \
          EPOCHS=1 SEED=$SEED MODE=mfb BETA=1 PRIOR=unigram STAGED=1 TK=0,0,0 MU=0 AUX=1 KAPPA=0.7 \
          HZ_H=1,1,1 BUCKET_BY=h COMPILE=0 NCCL_TIMEOUT_H=$TO_H WORLD_SIZE=$K MASTER_PORT=$p HEAD_DEPTH=2 \
          TIE_AUX=$tie_aux TIE_MODE=$mode TIE_SYNC=$TIE_SYNC OUT="" RANK=$r LOCAL_RANK=$r \
          $PY -u "$TRAIN" > "$LOGDIR/smoke_${a}_r${r}.log" 2>&1 &
      pids+=($!)
    done
    local failed=0 q; for q in "${pids[@]}"; do wait "$q" || failed=1; done
    if [ $failed -eq 1 ]; then echo "  [FAIL] $a - see $LOGDIR/smoke_${a}_r*.log"; ok=0
    else echo "  [ok]   $a ($mode): $(grep -a FINAL "$LOGDIR/smoke_${a}_r$((K-1)).log" | tail -1)"; fi
  done
  [ $ok = 1 ] && echo "[smoke] all green. Note: at K=4 rand and rshare share the same seed on rank0, differences start at rank1."
}

case "${1:-grid}" in
  SMOKE|smoke)       smoke; exit 0 ;;
  SUMMARY|summary)   summary; exit 0 ;;
  PROGRESS|progress) progress; exit 0 ;;
esac

echo "[cfg] SCALE=$SCALE HDS='$HDS' ARMS='$ARMS' K=$K TIE_SYNC=$TIE_SYNC SAVE=$SAVE GPUS=$GPUS TRAIN=$TRAIN NCCL_TIMEOUT_H=$TO_H"
for hd in $HDS; do
  for a in $ARMS; do run_cell "$hd" "$a" || true; done
done
echo "[$(date '+%F %T')] 2X2_DONE"
summary
