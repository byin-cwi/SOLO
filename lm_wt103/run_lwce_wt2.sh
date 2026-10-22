#!/bin/bash
# run_lwce_wt2.sh - WT-2 greedy local CE control (lm_solo.py): HEAD_DEPTH {0 (light), 2 (matched)} x SEED {42,1,2}.
#
# TK=0 MU=0 AUX=1 disables milestones: with MODE=mfb BETA=1, log_m = log_anchor = log q.detach(), so ell_m == ell_q,
# the hinge is 0 and loss = AUX * ell_q, i.e. pure segment-wise greedy local CE.
# LS / SEED / EPOCHS must match the mf4h arm being compared against.
#
# Usage: nohup bash run_lwce_wt2.sh > lwce_wt2.out 2>&1 < /dev/null &
# Knobs: PY
cd "$(dirname "$0")"
PY=${PY:-python3}
command -v "$PY" >/dev/null 2>&1 || PY=python
mkdir -p logs ckpt

BASE="MODE=mfb BETA=1 PRIOR=unigram K=2 EPOCHS=30 NLAYER=8 DMODEL=384 NHEAD=6 \
DSET=wikitext-2-raw-v1 TOKENIZER=word SEQ=256 BS=32 LR=3e-4 WD=0.1 DROP=0.2 LS=0.1"
LWCE="TK=0 MU=0 AUX=1"        # disables milestones (see header)

for seed in 42 1 2; do
  for arm in light:0 matched:2; do
    name=${arm%%:*}; hd=${arm##*:}
    tag=lwce_${name}_s${seed}
    echo "[$(date +%m-%d\ %H:%M)] start $tag (HEAD_DEPTH=$hd)"
    env $BASE $LWCE HEAD_DEPTH=$hd SEED=$seed CKPT=ckpt/${tag}.pt \
        "$PY" -u lm_solo.py > logs/${tag}.log 2>&1 \
      || echo "[$(date +%H:%M)] $tag FAILED -- tail logs/${tag}.log"
    echo "[$(date +%m-%d\ %H:%M)] done $tag: $(grep -a FINAL logs/${tag}.log | tail -1)"
  done
done
echo "lwce wave complete"
echo "Sanity check: act in the log should stay 0.00; the tight column is meaningless (promise is still logged against unigram), ignore it"
