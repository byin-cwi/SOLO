#!/bin/bash
# run_wt103_vocab_sweep.sh - vocabulary sweep at fixed d (lm_hz_tievar.py): VMAX in {8k,32k,65k,131k} x {tie, untie, random}.
# Fixed NLAYER=12 DMODEL=512 NHEAD=8, K=4, HD=2, 1ep (the 64M config of run_wt103_scalept.sh); V/d = 16/64/128/256.
# WT-103 has ~267k unique words, so no VMAX is UNK-saturated.
# The summary reports relative quantities: channel% = (untie-tie)/untie, random penalty% = (random-untie)/untie.
# Tags with a FINAL line are skipped.
# Usage: GPUS=4,5,6,7 nohup bash run_wt103_vocab_sweep.sh > logs_vocab/sweep.log 2>&1 &
# Knobs: GPUS VOCABS NLAYER DMODEL NHEAD BS SEED COMPILE PORT0 PY DATA
# Reference: V=32k (64M config) tie ~59 / untie ~63 / random ~82.
cd "$(dirname "$0")"
PY=${PY:-python}; LOGDIR=logs_vocab; CKPTDIR=ckpt_vocab; DATA=${DATA:-./data/wikitext}
mkdir -p "$LOGDIR" "$CKPTDIR"
NLAYER=${NLAYER:-12}; DMODEL=${DMODEL:-512}; NHEAD=${NHEAD:-8}; K=4; HD=2; SEQ=256; BS=${BS:-64}
SEED=${SEED:-42}; GPUS=${GPUS:-0,1,2,3}; VOCABS=${VOCABS:-"8192 32768 65536 131072"}; COMPILE=${COMPILE:-1}
IFS=',' read -ra G <<< "$GPUS"
export CUDA_VISIBLE_DEVICES=$GPUS MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export TORCHINDUCTOR_COMPILE_THREADS=1
port=${PORT0:-$((20000 + RANDOM % 20000))}
echo "[cfg] vocab-sweep d=$DMODEL L=$NLAYER K=$K BS=$BS VOCABS='$VOCABS' GPUS=$GPUS"

arm () {   # $1=tag $2=VMAX $3=TIE_AUX $4=TIE_MODE
  local tag=$1 vmax=$2 tie=$3 mode=$4
  if grep -qa FINAL "$LOGDIR/${tag}_r$((K-1)).log" 2>/dev/null; then
    echo "[skip] $tag: $(grep -a FINAL "$LOGDIR/${tag}_r$((K-1)).log" | tail -1)"; return 0; fi
  port=$((port+1)); echo "[$(date +%H:%M:%S)] START $tag (V=$vmax mode=$mode tie_aux=$tie)"
  local C="DATA=$DATA TOKENIZER=word DSET=wikitext-103-raw-v1 VMAX=$vmax \
NLAYER=$NLAYER DMODEL=$DMODEL NHEAD=$NHEAD SEQ=$SEQ BS=$BS LR=3e-4 WARMUP=1000 \
CLIP=0.5 WD=0.0 LS=0.0 DROP=0.1 EPOCHS=1 SEED=$SEED \
MODE=mfb BETA=1 PRIOR=unigram STAGED=1 TK=0,0,0 MU=0 AUX=1 KAPPA=0.7 HZ_H=1,1,1 BUCKET_BY=h COMPILE=$COMPILE"
  local pids=()
  for r in $(seq 0 $((K-1))); do
    env $C WORLD_SIZE=$K MASTER_PORT=$port HEAD_DEPTH=$HD TIE_AUX=$tie TIE_MODE=$mode \
        TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_voc_${tag}_r${r} OUT=$CKPTDIR/${tag}.pt RANK=$r LOCAL_RANK=$r \
        $PY -u lm_hz_tievar.py > "$LOGDIR/${tag}_r${r}.log" 2>&1 &
    pids+=($!)
  done
  local failed=0; for p in "${pids[@]}"; do wait "$p" || failed=1; done
  [ $failed -eq 1 ] && { echo "[FAIL] $tag; see $LOGDIR/${tag}_r*.log (V=131k: on OOM rerun with BS=32)"; kill -9 "${pids[@]}" 2>/dev/null; return 1; }
  echo "[$(date +%H:%M:%S)] DONE  $tag: $(grep -a FINAL "$LOGDIR/${tag}_r$((K-1)).log" | tail -1)"
}

for V in $VOCABS; do
  vk=$((V/1024))k
  arm "wt103_v${vk}_tie"    $V 1 trained || true
  arm "wt103_v${vk}_untie"  $V 0 trained || true
  arm "wt103_v${vk}_random" $V 1 random  || true
done

echo "[$(date +%H:%M:%S)] VOCAB_SWEEP_DONE"
echo "==== Summary: V/d vs channel% / random penalty% (compare only relative values across V) ===="
getp(){ grep -a FINAL "$1" 2>/dev/null | awk -F'val ppl: ' '{print $2+0}' | tail -1; }
for V in $VOCABS; do
  vk=$((V/1024))k
  t=$(getp "$LOGDIR/wt103_v${vk}_tie_r$((K-1)).log")
  u=$(getp "$LOGDIR/wt103_v${vk}_untie_r$((K-1)).log")
  r=$(getp "$LOGDIR/wt103_v${vk}_random_r$((K-1)).log")
  awk -v v=$V -v d=$DMODEL -v t="$t" -v u="$u" -v r="$r" 'BEGIN{
    printf "V=%-6d V/d=%-6.1f | tie=%-7s untie=%-7s random=%-7s", v, v/d, (t==""?"-":t), (u==""?"-":u), (r==""?"-":r);
    if(t>0&&u>0) printf " | channel%%=%+.2f", (u-t)/u*100;
    if(r>0&&u>0) printf " | randompen%%=%+.2f", (r-u)/u*100;
    print ""}'
done
