#!/bin/bash
# run_pipep_multi.sh: on one set of GPUs, compare SOLO vs BP throughput, memory and communication at equal pipeline depth p.
#
# Default grid (12 cells):
#   ARMS  bp (BP+PP, 1F1B)  solo (SOLO+PP, shared readout broadcast every 50 steps)  bp_dp (BP+DDP reference)
#   PS    2 and the GPU count N
#   MS    24 72 (divisible by 2, 6, 8 as the DP arms require)
# Prints a summary table at the end (summarize_pipep.py).
#
# Usage: GPUS=0,1,2,3,4,5 nohup bash run_pipep_multi.sh > pipep_multi.log 2>&1 &
#   More arms via ARMS (bp_rc bp_il bp_zb solo_dp solo@N solo_priv); ITERS=10 WARM=5 by default.
#   BCAST_SWEEP=1 adds the readout-broadcast sweep at p=max(PS); EXTRA_V4=1 adds interleaved v=4. Both default to 0.
cd "$(dirname "$0")"
# Mutex: one pipep experiment at a time (concurrent runs distort throughput and memory). The top-level script holds
# the lock and exports PIPEP_LOCKED=1 so child scripts do not lock again. Skipped without flock (e.g. macOS).
if [ -z "${PIPEP_LOCKED:-}" ] && command -v flock >/dev/null 2>&1; then
  exec 9>"$(pwd)/.pipep.lock"
  if ! flock -n 9; then
    echo "[abort] a pipep experiment is already running (lock $(pwd)/.pipep.lock). Stop it or wait for it to finish:"
    ps -ef | grep -E "run_pipep|pipep.py" | grep -v grep | sed 's/^/  /'
    exit 1
  fi
  export PIPEP_LOCKED=1
fi
GPUS=${GPUS:?set GPUS, e.g. GPUS=0,1,2,3,4,5}
IFS=',' read -ra ALL <<< "$GPUS"; N=${#ALL[@]}
PS=$(printf '%s\n' ${PS:-2 $N} | awk -v n="$N" '$1>=2 && $1<=n && !seen[$1]++' | tr '\n' ' ')
ARMS=${ARMS:-"bp solo bp_dp"}; MS=${MS:-"24 72"}
export ITERS=${ITERS:-10} WARM=${WARM:-5}
PMAX=$(printf '%s\n' $PS | sort -n | tail -1)
echo "[cfg] GPUS=$GPUS  N=$N  PS='$PS'  ARMS='$ARMS'  MS='$MS'  ITERS=${ITERS:-20} WARM=${WARM:-10}"

# interleaved / zero-bubble may require M divisible by p
case " $ARMS " in *" bp_il"*|*" bp_zb"*)
  for p in $PS; do for m in $MS; do
    [ $((m % p)) -eq 0 ] || echo "[warn] M=$m is not divisible by p=$p, bp_il/bp_zb may fail for this cell"
  done; done ;;
esac

# Pre-launch check: all listed GPUs must be idle. SKIP_PREFLIGHT=1 skips it.
if [ "${SKIP_PREFLIGHT:-0}" != 1 ] && command -v nvidia-smi >/dev/null 2>&1; then
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', *' -v list=",$GPUS," 'index(list, ","$1",") && $2 > 1024 {printf "  GPU %s: %s MiB used\n", $1, $2}')
  if [ -n "$busy" ]; then
    echo "[abort] the following GPUs are not idle:"; echo "$busy"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv 2>/dev/null | sed 's/^/  /'
    exit 1
  fi
  echo "[preflight] GPU $GPUS idle"
fi

sub(){ local p=$1; (IFS=,; echo "${ALL[*]:0:$p}"); }

for p in $PS; do
  G=$(sub "$p")
  echo "===== p=$p  GPUS=$G  $(date '+%F %T') ====="
  GPUS=$G ARMS="$ARMS" MS="$MS" LOGDIR=${LOGDIR_PREFIX:-logs_pipep_p}$p bash run_pipep.sh \
    || { echo "===== p=$p failed, stopping ====="; exit 1; }
done

G=$(sub "$PMAX")
if [ "${BCAST_SWEEP:-0}" = 1 ]; then
  echo "===== extra  p=$PMAX readout broadcast sweep  $(date '+%F %T') ====="
  GPUS=$G ARMS="solo solo@1 solo@100 solo_priv bp" MS=24 LOGDIR=logs_pipep_p${PMAX}_bcast \
    bash run_pipep.sh || exit 1
fi

if [ "${EXTRA_V4:-0}" = 1 ]; then
  echo "===== extra  p=$PMAX interleaved v=4  $(date '+%F %T') ====="
  GPUS=$G ARMS="bp_il" VCHUNKS=4 MS="$MS" LOGDIR=logs_pipep_p${PMAX}_v4 bash run_pipep.sh || exit 1
fi
echo "===== done  $(date '+%F %T') ====="
echo
echo "================================ summary ================================"
EXTRA_DIRS=""
[ "${BCAST_SWEEP:-0}" = 1 ] && EXTRA_DIRS="$EXTRA_DIRS logs_pipep_p${PMAX}_bcast"
[ "${EXTRA_V4:-0}" = 1 ] && EXTRA_DIRS="$EXTRA_DIRS logs_pipep_p${PMAX}_v4"
python summarize_pipep.py $(for p in $PS; do printf '%s%s ' "${LOGDIR_PREFIX:-logs_pipep_p}" "$p"; done) $EXTRA_DIRS
