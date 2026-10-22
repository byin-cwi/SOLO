#!/bin/bash
# run_ladder_abl.sh - readout-ladder driver.
# Phase P1 runs the tie / untie / random arms (+ bp with WITH_BP=1); phase P2 runs stale, which loads the
# fc from the same-seed tie checkpoint. SCRIPT selects the backbone: ./vision_ladder_r32.py (default) or
# ./vision_ladder_r50.py; both take the same env interface and print the same FINAL line.
#
# Env (defaults):
#   SCRIPT=./vision_ladder_r32.py  DATASET=c100  EPOCHS=200  BATCH=128
#   LABEL_SMOOTH=0  TERM_LS=$LABEL_SMOOTH  K_MODULES=15  HEAD_DEPTH=4  HEAD_WIDEN=2  TIE_DIM=64
#   AUTOAUG=1  GS_MODE=jacobi  LR=0.1  WD=5e-4  DATA_ROOT=./data  PYBIN=python
#   SEEDS="42"  GPUS="0,1,2,3"  WITH_BP=0  TAGPFX=abl_${DATASET}
# Runs NG=len(GPUS) jobs at a time, one per GPU. Tags whose log already has a FINAL line are skipped,
# so the script can be re-run to resume. Checkpoints go to ckpt/ and logs to logs/ in this directory.
#
# Usage: DATASET=c100 DATA_ROOT=<data dir> GPUS=0,1,2,3 bash run_ladder_abl.sh
cd "$(dirname "$0")"; mkdir -p logs ckpt
SCRIPT=${SCRIPT:-./vision_ladder_r32.py}
DATASET=${DATASET:-c100}; EPOCHS=${EPOCHS:-200}; BATCH=${BATCH:-128}
LS=${LABEL_SMOOTH:-0}; TLS=${TERM_LS:-$LS}
K_MODULES=${K_MODULES:-15}; HEAD_DEPTH=${HEAD_DEPTH:-4}; HEAD_WIDEN=${HEAD_WIDEN:-2}; TIE_DIM=${TIE_DIM:-64}
AUTOAUG=${AUTOAUG:-1}; GS_MODE=${GS_MODE:-jacobi}; LR=${LR:-0.1}; WD=${WD:-5e-4}; DATA_ROOT=${DATA_ROOT:-./data}
SEEDS=${SEEDS:-42}; GPUS=${GPUS:-0,1,2,3}; WITH_BP=${WITH_BP:-0}; TAGPFX=${TAGPFX:-abl_${DATASET}}
IFS=',' read -ra G <<< "$GPUS"; NG=${#G[@]}
C="EPOCHS=$EPOCHS BATCH=$BATCH LR=$LR WD=$WD K_MODULES=$K_MODULES HEAD_DEPTH=$HEAD_DEPTH HEAD_WIDEN=$HEAD_WIDEN \
TIE_DIM=$TIE_DIM GS_MODE=$GS_MODE AUTOAUG=$AUTOAUG DATA_ROOT=$DATA_ROOT DATASET=$DATASET LABEL_SMOOTH=$LS TERM_LS=$TLS"
echo "[cfg] SCRIPT=$SCRIPT DATASET=$DATASET EPOCHS=$EPOCHS BATCH=$BATCH K=$K_MODULES TIE_DIM=$TIE_DIM LS=$LS/$TLS \
AUTOAUG=$AUTOAUG SEEDS='$SEEDS' GPUS=$GPUS(NG=$NG) WITH_BP=$WITH_BP TAGPFX=$TAGPFX"

# preflight checks: python with torch/torchvision, backbone script, data directory
PYBIN=${PYBIN:-python}
command -v "$PYBIN" >/dev/null 2>&1 || { echo "[preflight] python not found (PYBIN=$PYBIN); activate a conda env with torchvision first, or set PYBIN=<path>"; exit 1; }
"$PYBIN" -c "import torch, torchvision" 2>/dev/null || { echo "[preflight] $PYBIN lacks torch/torchvision - activate the right env or set PYBIN"; exit 1; }
[ -f "$SCRIPT" ] || { echo "[preflight] backbone script not found SCRIPT=$SCRIPT (should be in this directory)"; exit 1; }
if [ "$DATASET" = tiny ]; then
  [ -d "$DATA_ROOT/tiny-imagenet-200" ] || { echo "[preflight] missing $DATA_ROOT/tiny-imagenet-200/ - set DATA_ROOT correctly (must contain tiny-imagenet-200/)"; exit 1; }
else
  [ -d "$DATA_ROOT" ] || { echo "[preflight] DATA_ROOT=$DATA_ROOT does not exist (CIFAR downloads here, must be writable)"; exit 1; }
fi
echo "[preflight] OK: $PYBIN + torch/torchvision / $SCRIPT / DATA_ROOT=$DATA_ROOT"

run_wave () {                          # args: "tag ENV1=.. ENV2=.." entries, run NG at a time
  local jobs=("$@") i=0
  while [ $i -lt ${#jobs[@]} ]; do
    local pids=()
    for ((g=0; g<NG && i<${#jobs[@]}; g++, i++)); do
      set -- ${jobs[$i]}; local tag=$1; shift
      if grep -qa FINAL "logs/${tag}.log" 2>/dev/null; then echo "[skip] $tag"; continue; fi
      env CUDA_VISIBLE_DEVICES=${G[$g]} $C "$@" CKPT=ckpt/${tag}.pt \
          "$PYBIN" -u "$SCRIPT" > "logs/${tag}.log" 2>&1 &
      pids+=($!); echo "[$(date +%H:%M)] START $tag gpu${G[$g]}"
    done
    [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
  done
}

P1=()
for s in $SEEDS; do
  P1+=("${TAGPFX}_tie_s${s} SEED=${s} TIE=1")
  P1+=("${TAGPFX}_untie_s${s} SEED=${s} TIE=0")
  P1+=("${TAGPFX}_rand_s${s} SEED=${s} TIE=1 TIE_MODE=random TIE_SCALE=1")
  [ "$WITH_BP" = 1 ] && P1+=("${TAGPFX}_bp_s${s} SEED=${s} MODE=bp")
done
echo "[$(date +%H:%M)] === P1 tie/untie/random$([ "$WITH_BP" = 1 ] && echo /bp) (${#P1[@]}) ==="; run_wave "${P1[@]}"

P2=()
for s in $SEEDS; do
  src=ckpt/${TAGPFX}_tie_s${s}.pt
  [ -f "$src" ] && P2+=("${TAGPFX}_stale_s${s} SEED=${s} TIE=1 TIE_MODE=stale STALE_FC=$src") \
                || echo "[warn] missing $src, skipping stale s${s}"
done
echo "[$(date +%H:%M)] === P2 stale (${#P2[@]}) ==="; run_wave "${P2[@]}"
echo "[$(date +%H:%M)] ALL_DONE $TAGPFX"; grep -aH FINAL logs/${TAGPFX}_*.log 2>/dev/null
