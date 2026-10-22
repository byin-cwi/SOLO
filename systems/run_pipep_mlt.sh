#!/bin/bash
# run_pipep_mlt.sh: fixed global batch of G sequences, sweep over the microbatches per replica m; compares SOLO hybrid
# layouts with all BP 3D-parallel schedules. Default shapes: L=24 d=2048 T=2048 V=32768, H in {2, 1}.
#
# Per cell: replicas R are set by the arm, microbatch size MB = G/(m*R) (cells where this is not an integer are skipped
#   with the reason printed). pipep.py gets M = m*R (microbatches over all replicas), so every arm processes G rows per
#   step (checked through rows_per_iter).
# Arms (default 9):
#   solo_kr2x4  SOLO K=2 modules x R=4 replicas   solo_kr4x2  K=4 x R=2      solo  K=p=8 (R=1)
#   bp          BP PP8 1F1B                        bp_zb  zero-bubble (v=VCHUNKS)   bp_il  interleaved (v=VCHUNKS)
#   bp_pd4      BP PP4 x DP2                       bp_pd2  PP2 x DP4               bp_dp  pure DDP (PP1 x DP8)
#   bp_il / bp_zb are framework schedules and require M divisible by p; other cells are skipped.
# H: HS order (default 2 then 1); BP arms do not depend on H and run only for the first H (BP_EVERY_H=1 runs them for
#   every H, as repeated measurements).
#
# Usage: bash run_pipep_mlt.sh 0,1,2,3,4,5,6,7           # first argument = GPUS (or GPUS=... env)
#   Options: G=32 MLS="1 2 4 8" HS="2 1" ARMS="..." ITERS=10 WARM=5 TIE_SYNC=50 VCHUNKS=2 CE_MODE=compile
#            NLAYER=24 DMODEL=2048 SEQ=2048 VMAX=32768 LOGDIR_PREFIX=logs_pipep_mlt DRY=1 (only list the cells)
#   At the end summarize_pipep.py --mlt prints, per m, each arm's tok/s relative to the best BP schedule, with peak
#   memory and communication per step. Smaller G (e.g. G=8, G=4) skips bp_dp and large-R cells; run those separately.
cd "$(dirname "$0")"
if [ -z "${PIPEP_LOCKED:-}" ] && command -v flock >/dev/null 2>&1; then
  exec 9>"$(pwd)/.pipep.lock"
  if ! flock -n 9; then
    echo "[abort] a pipep experiment is already running (lock $(pwd)/.pipep.lock). Stop it or wait for it to finish:"
    ps -ef | grep -E "run_pipep|pipep.py" | grep -v grep | sed 's/^/  /'
    exit 1
  fi
  export PIPEP_LOCKED=1
fi
GPUS=${1:-${GPUS:-0,1,2,3,4,5,6,7}}
IFS=',' read -ra ALL <<< "$GPUS"; P=${#ALL[@]}
G=${G:-32}; MLS=${MLS:-"1 2 4 8"}; HS=${HS:-"2 1"}
ARMS=${ARMS:-"solo_kr2x4 solo_kr4x2 solo bp bp_zb bp_il bp_pd4 bp_pd2 bp_dp"}
export NLAYER=${NLAYER:-24} DMODEL=${DMODEL:-2048} SEQ=${SEQ:-2048} VMAX=${VMAX:-32768}
export ITERS=${ITERS:-10} WARM=${WARM:-5} VCHUNKS=${VCHUNKS:-2} TIE_SYNC=${TIE_SYNC:-50} CE_MODE=${CE_MODE:-compile}
export BACKEND=${BACKEND:-nccl}
PREFIX=${LOGDIR_PREFIX:-logs_pipep_mlt}_G${G}
BP_EVERY_H=${BP_EVERY_H:-0}; DRY=${DRY:-0}
echo "[cfg] GPUS=$GPUS p=$P G=$G MLS='$MLS' HS='$HS' ARMS='$ARMS' L=$NLAYER d=$DMODEL T=$SEQ V=$VMAX" \
     "ITERS=$ITERS WARM=$WARM TIE_SYNC=$TIE_SYNC v=$VCHUNKS BP_EVERY_H=$BP_EVERY_H"

# Preflight: all listed GPUs must be idle. SKIP_PREFLIGHT=1 skips it.
if [ "$DRY" != 1 ] && [ "${SKIP_PREFLIGHT:-0}" != 1 ] && command -v nvidia-smi >/dev/null 2>&1; then
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', *' -v list=",$GPUS," 'index(list, ","$1",") && $2 > 1024 {printf "  GPU %s: %s MiB used\n", $1, $2}')
  if [ -n "$busy" ]; then
    echo "[abort] the following GPUs are not idle:"; echo "$busy"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv 2>/dev/null | sed 's/^/  /'
    exit 1
  fi
  echo "[preflight] GPU $GPUS idle"
fi
if [ "$DRY" != 1 ]; then echo "--- probe (schedules available on this machine)"; PROBE=1 ${PY:-python} pipep.py 2>&1 | sed 's/^/  /'; fi

replicas(){  # arm -> replica count R
  case $1 in
    solo_kr*x*) echo "${1#*x}" ;;
    bp_pd*)     echo $((P / ${1#bp_pd})) ;;
    bp_dp|solo_dp) echo "$P" ;;
    *)          echo 1 ;;
  esac
}
is_bp(){ case $1 in bp*) return 0 ;; *) return 1 ;; esac; }

first_h=1; n_run=0; n_skip=0; n_fail=0
for H in $HS; do
  LOGDIR=${PREFIX}_H$H; mkdir -p "$LOGDIR"
  echo "===== H=$H  LOGDIR=$LOGDIR  $(date '+%F %T') ====="
  for m in $MLS; do for a in $ARMS; do
    if is_bp "$a" && [ "$BP_EVERY_H" != 1 ] && [ "$first_h" != 1 ]; then continue; fi   # BP arms do not depend on H
    R=$(replicas "$a"); M=$((m * R))
    if [ $((G % M)) -ne 0 ]; then
      echo "[skip] $a m=$m: G=$G not divisible by m*R=$M (MB not an integer)"; n_skip=$((n_skip + 1)); continue
    fi
    MB=$((G / M))
    case $a in bp_il|bp_zb)
      if [ $((M % P)) -ne 0 ]; then
        echo "[skip] $a m=$m: framework interleaved/zero-bubble schedules require M=$M divisible by p=$P"
        n_skip=$((n_skip + 1)); continue
      fi ;;
    esac
    echo "--- $a  m=$m  R=$R  M=$M  MB=$MB  H=$H  $(date '+%T')"
    if [ "$DRY" = 1 ]; then continue; fi
    GPUS=$GPUS ARMS=$a MS=$M MB=$MB HEAD_DEPTH=$H LOGDIR=$LOGDIR FAIL_FAST=0 NO_TABLE=1 SKIP_PROBE=1 \
      bash run_pipep.sh
    # a cell succeeded iff all P rank logs contain a PIPEP line (run_pipep.sh returns 0 on failure with FAIL_FAST=0)
    n_ok=$(grep -l '^PIPEP' "$LOGDIR/${a}_p${P}_M${M}_v${VCHUNKS}".r*.log 2>/dev/null | wc -l | tr -d " ")
    if [ "$n_ok" -eq "$P" ]; then n_run=$((n_run + 1)); else n_fail=$((n_fail + 1)); echo "[fail] $a m=$m H=$H ($n_ok/$P ranks reported)"; fi
  done; done
  first_h=0
done
echo "===== done  $(date '+%F %T')  ran $n_run cells, skipped $n_skip, failed $n_fail ====="
[ "$DRY" = 1 ] && exit 0
echo
echo "================================ summary (SOLO vs best BP) ================================"
${PY:-python} summarize_pipep.py --mlt $(for H in $HS; do printf '%s_H%s ' "$PREFIX" "$H"; done)
