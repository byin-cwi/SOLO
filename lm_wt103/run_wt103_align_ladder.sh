#!/bin/bash
# run_wt103_align_ladder.sh - online gradient-alignment probe on the WT-103 readout ladder (lm_solo.py, single process,
#   1 GPU per arm; 8L d384, word level V=32768, K=4, HD=2, 1ep). Every PROBE_EVERY optimizer steps it measures
#   cos(g_hat_k, g_k^BP) and the error terms E_read/E_resid/E_path on a fixed val batch at the three boundaries.
# Arms:
#   solo     = shared live readout, refreshed every step (TIE_AUX=1 TIE_SYNC=1)
#   priv     = private learnable readout (TIE_AUX=0)
#   rand     = frozen random readout with std(W) scale (TIE_MODE=random)
#   conv_loc = frozen final fc of the same-seed solo run (STALE_FC; starts after solo finishes)
#   conv_bp  = frozen BP readout (STALE_FC=$BPFC, shared across seeds; skipped if the file is missing)
# Recipe = single-process equivalent of run_wt103_ladder.sh COMMON (TK=0,0,0 => local loss = AUX x CE).
# Requires the pipeline vocab cache $DATA/word_cache_wikitext-103-raw-v1_32768.pt (build with lm_hz_tievar.py and
#   BUILD_WORD_CACHE=1) so that rows match W*_BP. Arms with a FINAL line are skipped. Probe JSONL = logs_align/<tag>.align.jsonl.
# Usage: GPUS=0,1,2,3 SEEDS="42 43 44" nohup bash run_wt103_align_ladder.sh > logs_align/driver.log 2>&1 &
#   Plot: python make_fig_align_time.py logs_align
# Knobs: GPUS SEEDS ARMS BPFC EPOCHS PROBE_EVERY PROBE_BS POLL STAGGER FAIL_TAIL DRY=1 (print only)
cd "$(dirname "$0")"
PY=${PY:-python}; LOGDIR=${LOGDIR:-logs_align}; CKPTDIR=${CKPTDIR:-ckpt_align}; DATA=${DATA:-./data/wikitext}
mkdir -p "$LOGDIR" "$CKPTDIR"
GPUS=${GPUS:-0,1,2,3}; SEEDS=${SEEDS:-"42 43 44"}; ARMS=${ARMS:-"solo priv rand conv_loc conv_bp"}
EP=${EPOCHS:-1}; PE=${PROBE_EVERY:-500}; PB=${PROBE_BS:-16}; DRY=${DRY:-0}; POLL=${POLL:-60}
BPFC=${BPFC:-ckpt_wt103tie/bpfc_readout.pt}
IFS=',' read -ra G <<< "$GPUS"; NG=${#G[@]}
CACHE=$DATA/word_cache_wikitext-103-raw-v1_32768.pt
[ "$DRY" = 1 ] || [ -f "$CACHE" ] || { echo "[abort] missing pipeline vocab cache $CACHE (build it first with lm_hz_tievar.py and BUILD_WORD_CACHE=1); a vocab built by a single process would misalign rows with W*_BP"; exit 1; }
grep -q 'PROBE_EVERY = int(E("PROBE_EVERY"' lm_solo.py 2>/dev/null || { echo "[abort] lm_solo.py has no PROBE_EVERY probe"; exit 1; }
# run_wt103_ladder.sh COMMON without pipeline-only HZ_H/COMPILE, plus K/STAGED/WORD_CACHE/probe
COMMON="MODE=mfb STAGED=1 DATA=$DATA TOKENIZER=word DSET=wikitext-103-raw-v1 VMAX=32768 WORD_CACHE=1 \
NLAYER=8 DMODEL=384 NHEAD=6 SEQ=256 BS=64 LR=3e-4 WARMUP=1000 CLIP=0.5 WD=0.0 LS=0.0 DROP=0.1 EPOCHS=$EP \
K=4 HEAD_DEPTH=2 BETA=1 PRIOR=unigram TK=0,0,0 MU=0 AUX=1 KAPPA=0.7 BUCKET_BY=h PROBE_EVERY=$PE PROBE_BS=$PB"
echo "[cfg] align ladder GPUS=$GPUS SEEDS='$SEEDS' ARMS='$ARMS' EP=$EP PROBE_EVERY=$PE PROBE_BS=$PB BPFC=$BPFC DRY=$DRY"
echo "[recipe] $COMMON"

tag_of(){ echo "wt103al_$1_s$2"; }
done_tag(){ grep -qa FINAL "$LOGDIR/$1.log" 2>/dev/null; }
dep_ok(){  # $1=arm $2=seed
  case $1 in conv_loc) [ -f "$CKPTDIR/$(tag_of solo $2).pt" ] || done_tag "$(tag_of solo $2)";; conv_bp) [ -f "$BPFC" ];; *) return 0;; esac; }
arm_env(){ # $1=arm $2=seed
  case $1 in solo) echo "TIE_AUX=1 TIE_SYNC=1";; priv) echo "TIE_AUX=0";; rand) echo "TIE_AUX=1 TIE_MODE=random TIE_SYNC=1";;
    conv_loc) echo "TIE_AUX=1 STALE_FC=$CKPTDIR/solo_fc_s$2.pt";; conv_bp) echo "TIE_AUX=1 STALE_FC=$BPFC";;
    *) echo "[abort] unknown arm $1" >&2; exit 1;; esac; }
LAUNCH_PID=""
launch(){  # $1=arm $2=seed $3=gpu  (non-DRY: launches in background, LAUNCH_PID=pid)
  local arm=$1 seed=$2 gpu=$3 tag f cmd; tag=$(tag_of $arm $seed); f=$LOGDIR/$tag.log
  cmd="env CUDA_VISIBLE_DEVICES=$gpu $COMMON SEED=$seed $(arm_env $arm $seed) OUT=$CKPTDIR/$tag.pt PROBE_OUT=$LOGDIR/$tag.align.jsonl $PY -u lm_solo.py"
  if [ "$DRY" = 1 ]; then echo "  [dry] gpu$gpu $tag: $cmd > $f"; return 0; fi
  if [ "$arm" = conv_loc ] && [ ! -f "$CKPTDIR/solo_fc_s$seed.pt" ]; then   # extract the final fc of the same-seed solo -> flat {fc.weight,fc.bias}
    $PY extract_bpfc.py "$CKPTDIR/$(tag_of solo $seed).pt" "$CKPTDIR/solo_fc_s$seed.pt" >> "$LOGDIR/extract_s$seed.log" 2>&1 || { echo "[FAIL] extracting solo_fc_s$seed failed, see $LOGDIR/extract_s$seed.log"; return 1; }
  fi
  rm -f "$LOGDIR/$tag.align.jsonl"          # probe log is written from scratch
  $cmd > "$f" 2>&1 &
  LAUNCH_PID=$!
}

JOBS=(); for s in $SEEDS; do for a in $ARMS; do JOBS+=("$a|$s"); done; done
[ " $ARMS " = *" conv_bp "* ] || true
[ -f "$BPFC" ] || { echo "[note] no $BPFC: conv_bp arm will be skipped (set BPFC=... if needed)"; }
BUSY_PID=(); BUSY_TAG=(); for ((i=0; i<NG; i++)); do BUSY_PID[$i]=""; BUSY_TAG[$i]=""; done
pending=("${JOBS[@]}")
while :; do
  for ((i=0; i<NG; i++)); do p=${BUSY_PID[$i]}
    if [ -n "$p" ] && ! kill -0 "$p" 2>/dev/null; then wait "$p" 2>/dev/null; t=${BUSY_TAG[$i]}
      if done_tag "$t"; then echo "[$(date +%H:%M:%S)] DONE  $t (gpu${G[$i]}): $(grep -a FINAL $LOGDIR/$t.log | tail -1 | cut -c1-90)  probes=$(wc -l < $LOGDIR/$t.align.jsonl 2>/dev/null)"
      else echo "[FAIL] $t (gpu${G[$i]}) no FINAL; tail:"; tail -${FAIL_TAIL:-6} "$LOGDIR/$t.log"; fi
      BUSY_PID[$i]=""; BUSY_TAG[$i]=""; fi
  done
  new=(); launched=0
  for j in "${pending[@]}"; do a=${j%%|*}; s=${j##*|}; t=$(tag_of $a $s)
    if done_tag "$t"; then echo "[skip] $t: $(grep -a FINAL $LOGDIR/$t.log | tail -1 | cut -c1-60)"; continue; fi
    if [ "$a" = conv_bp ] && [ ! -f "$BPFC" ]; then continue; fi
    if ! dep_ok "$a" "$s"; then new+=("$j"); continue; fi
    slot=-1; for ((i=0; i<NG; i++)); do [ -z "${BUSY_PID[$i]}" ] && { slot=$i; break; }; done
    [ $slot -lt 0 ] && { new+=("$j"); continue; }
    if [ "$DRY" = 1 ]; then launch "$a" "$s" "${G[$slot]}"; BUSY_PID[$slot]=dry; BUSY_TAG[$slot]=$t; launched=1; continue; fi   # DRY also occupies slots to show the first-wave GPU assignment
    if launch "$a" "$s" "${G[$slot]}"; then BUSY_PID[$slot]=$LAUNCH_PID; BUSY_TAG[$slot]=$t; launched=1
      echo "[$(date +%H:%M:%S)] START $t gpu${G[$slot]} pid=$LAUNCH_PID"; sleep ${STAGGER:-5}
    else echo "[warn] launching $t failed (dependency files not ready?), retrying next round"; new+=("$j"); fi
  done
  pending=("${new[@]}")
  busy=0; for ((i=0; i<NG; i++)); do [ -n "${BUSY_PID[$i]}" ] && busy=1; done
  if [ "$DRY" = 1 ]; then [ ${#pending[@]} -gt 0 ] && echo "[dry] queued after first wave (waiting for a free GPU or for dependency solo final state/BPFC): ${pending[*]}"; break; fi
  [ ${#pending[@]} -eq 0 ] && [ $busy -eq 0 ] && break
  if [ $busy -eq 0 ] && [ $launched -eq 0 ] && [ ${#pending[@]} -gt 0 ]; then echo "[abort] no launchable arm and no running arm, unmet dependencies: ${pending[*]} (solo arm failed?)"; break; fi
  sleep "$POLL"
done
echo "[$(date +%H:%M:%S)] ALIGN_LADDER_DONE"
echo "==== Summary (terminal val ppl | exits | probe points | last-point cos per boundary) ===="
for s in $SEEDS; do for a in $ARMS; do t=$(tag_of $a $s); f=$LOGDIR/$t.log; j=$LOGDIR/$t.align.jsonl
  [ -f "$f" ] || continue
  fl=$(grep -a '^FINAL' "$f" | tail -1); val=$(printf '%s' "$fl" | awk -F'val ppl: ' 'NF>1{print $2+0}')
  ex=$(grep -a '^\[epoch' "$f" | tail -1 | grep -oE "exits=\[[^]]*\]" | sed "s/exits=//; s/'//g; s/ //g")
  last=$( [ -f "$j" ] && tail -1 "$j" | $PY -c "import sys,json; r=json.loads(sys.stdin.read()); print(' '.join(f\"k{x['k']}={x['cos']:.3f}\" for x in r['rows']))" 2>/dev/null )
  printf "  %-22s val=%-7s exits=%-26s probes=%-3s last_cos: %s\n" "$t" "${val:--}" "${ex:--}" "$( [ -f "$j" ] && wc -l < "$j" || echo 0)" "${last:--}"
done; done
echo "Plot: $PY make_fig_align_time.py $LOGDIR   (fig_align_time.{pdf,png}, fig_align_terms.png, last-point table)"
