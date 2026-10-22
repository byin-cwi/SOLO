#!/bin/bash
# run_pipep.sh: compare SOLO and BP throughput, memory and communication under the same pipeline parallelism (PP):
# same p, split, microbatch count and microbatch size. Reports
#   throughput: tok/s; the readout broadcast of tied solo is amortized over TIE_SYNC.
#   memory:     peak, split into state (params + optimizer state + buffers) and an activation estimate
#               (peak - state - gradients).
#   comm:       pipeline P2P bytes and the tied-solo readout broadcast bytes, listed separately.
#
# Arms (labels in ARMS):
#   solo       shared readout, broadcast every TIE_SYNC steps (default TIE_SYNC=50)
#   solo@N     same with TIE_SYNC=N (e.g. solo@1 solo@50 solo@100)
#   solo_priv  private readout per stage, no broadcast
#   bp         standard 1F1B
#   bp_rc      standard 1F1B + per-block activation recomputation
#   bp_fw      standard 1F1B via PyTorch Schedule1F1B
#   bp_il      interleaved 1F1B (VCHUNKS virtual stages per GPU), PyTorch reference implementation
#   bp_il_rc   interleaved + activation recomputation
#   bp_zb      zero-bubble, PyTorch reference implementation
#   bp_dp      BP + data parallel (DDP): p full replicas, M/p microbatches each
#   bp_dp_rc   BP + DDP + activation recomputation
#   solo_dp    SOLO + DDP: K=p modules per replica, one DDP per module, shared readout copied on the GPU
#   solo_kr<K>x<R>  SOLO with K modules x R replicas (K*R = p), one module per GPU, DDP across replicas; e.g. solo_kr2x4
#   bp_pd<D>   BP with pipeline depth D x data parallel p/D, 1F1B + DDP; e.g. bp_pd4 (D=p is bp, D=1 is bp_dp)
#   Data-parallel arms require M divisible by the replica count (the default MS satisfies this).
#
# Usage: GPUS=0,1,2,3 ARMS="solo bp bp_rc bp_il" bash run_pipep.sh
#   Options: MS="8 16 32 64" VCHUNKS=2 TIE_SYNC=50 NLAYER=96 DMODEL=1024 SEQ=1024 VMAX=8192 MB=4
#            HEAD_DEPTH=1 ITERS=20 WARM=10 CE_MODE=compile LOGDIR=logs_pipep FAIL_FAST=1 DRY=1 LINK=nvlink
#            SKIP_PROBE=1 (no probe) NO_TABLE=1 (no summary table; used by run_pipep_mlt.sh)
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
PY=${PY:-python}; GPUS=${GPUS:-0,1}; DRY=${DRY:-0}
ARMS=${ARMS:-"solo bp bp_rc bp_il"}; MS=${MS:-"8 16 32 64"}; V=${VCHUNKS:-2}
TS=${TIE_SYNC:-50}
L=${NLAYER:-96}; D=${DMODEL:-1024}; T=${SEQ:-1024}; VOC=${VMAX:-8192}
MB=${MB:-4}; H=${HEAD_DEPTH:-1}; ITERS=${ITERS:-20}; WARM=${WARM:-10}; CE=${CE_MODE:-compile}
LOGDIR=${LOGDIR:-logs_pipep}; mkdir -p "$LOGDIR"
# Interconnect: nvlink (as is) | socket (NCCL over TCP loopback) | 10g/5g/1g/500m (socket + tc tbf rate limit).
# Rate-limited modes need root and tc and only emulate a link on one node; nominal rate != effective bandwidth.
LINK=${LINK:-nvlink}; LINK_IF=${LINK_IF:-lo}; TC_ON=0; LINK_TAG=""
tc_clear(){ [ "$TC_ON" = 1 ] && { tc qdisc del dev "$LINK_IF" root 2>/dev/null; TC_ON=0; }; }
trap tc_clear EXIT INT TERM
SOCK_ENV="NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=$LINK_IF"
case $LINK in
  nvlink) LINK_ENV="" ;;
  socket) LINK_ENV="$SOCK_ENV"; LINK_TAG="_socket" ;;
  *g|*m)  LINK_ENV="$SOCK_ENV"; LINK_TAG="_$LINK"
          command -v tc >/dev/null 2>&1 || { echo "[abort] LINK=$LINK requires tc"; exit 1; }
          [ "$(id -u)" = 0 ] || { echo "[abort] LINK=$LINK requires root"; exit 1; }
          case $LINK in *g) RB=$(( ${LINK%g} * 1000000000 ));; *m) RB=$(( ${LINK%m} * 1000000 ));; esac
          BURST=$(( RB / 8 / 1000 * 3 / 2 )); [ "$BURST" -lt 32768 ] && BURST=32768
          tc qdisc del dev "$LINK_IF" root 2>/dev/null
          tc qdisc add dev "$LINK_IF" root tbf rate "${LINK%[gm]}${LINK##*[0-9]}bit" \
            burst "$BURST" latency 200ms || { echo "[abort] tc qdisc add failed"; exit 1; }
          TC_ON=1 ;;
  *) echo "[abort] unknown LINK=$LINK (nvlink|socket|<n>g|<n>m)"; exit 1 ;;
esac
[ -n "$LINK_ENV" ] && echo "[link] $LINK  env: $LINK_ENV$( [ "$TC_ON" = 1 ] && echo "  tbf burst=$BURST" )"
FAIL_FAST=${FAIL_FAST:-1}
IFS=',' read -ra G <<< "$GPUS"; P=${#G[@]}
COMMON="NLAYER=$L DMODEL=$D SEQ=$T VMAX=$VOC MB=$MB HEAD_DEPTH=$H ITERS=$ITERS WARM=$WARM \
CE_MODE=$CE VCHUNKS=$V BACKEND=${BACKEND:-nccl}"

echo "[cfg] p=$P GPUS=$GPUS L=$L d=$D T=$T V=$VOC MB=$MB H=$H v=$V TIE_SYNC=$TS ARMS='$ARMS' MS='$MS' LINK=$LINK"
echo "[layers/stage] $((L / P))"
if [ "${SKIP_PROBE:-0}" != 1 ]; then
  echo "--- probe (schedules available on this machine)"
  PROBE=1 $PY pipep.py 2>&1 | sed 's/^/  /'
fi

arm_env(){  # label -> pipep.py environment variables
  case $1 in
    solo)      echo "ARM=solo SOLO_READOUT=tied TIE_SYNC=$TS" ;;
    solo@*)    echo "ARM=solo SOLO_READOUT=tied TIE_SYNC=${1#solo@}" ;;
    solo_priv) echo "ARM=solo SOLO_READOUT=priv" ;;
    bp)        echo "ARM=bp" ;;
    bp_rc)     echo "ARM=bp RECOMPUTE=full" ;;
    bp_fw)     echo "ARM=bp_fw" ;;
    bp_il)     echo "ARM=bp_il" ;;
    bp_il_rc)  echo "ARM=bp_il RECOMPUTE=full" ;;
    bp_zb)     echo "ARM=bp_zb" ;;
    bp_dp)     echo "ARM=bp_dp" ;;
    bp_dp_rc)  echo "ARM=bp_dp RECOMPUTE=full" ;;
    solo_dp)   echo "ARM=solo_dp SOLO_READOUT=tied TIE_SYNC=$TS" ;;
    solo_kr*x*) local kr=${1#solo_kr}; local k=${kr%x*} r=${kr#*x}
               [ $((k * r)) -eq "$P" ] || { echo "[abort] $1: K*R=$((k * r)) != GPU count $P" >&2; return 1; }
               echo "ARM=solo_kr KR_K=$k SOLO_READOUT=tied TIE_SYNC=$TS" ;;
    bp_pd*)    local d=${1#bp_pd}
               [ $((P % d)) -eq 0 ] || { echo "[abort] $1: D=$d does not divide GPU count $P" >&2; return 1; }
               echo "ARM=bp_pd PD_D=$d" ;;
    *)         return 1 ;;
  esac
}
for a in $ARMS; do arm_env "$a" >/dev/null || { echo "[abort] unknown arm label: $a"; exit 1; }; done

free_port(){  # let the kernel pick a free port
  $PY -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}
show_fail(){  # $1=tag  print the first real error from every rank's log, rank 0 first
  local tag=$1 f
  for f in $(ls "$LOGDIR/$tag".r*.log 2>/dev/null | sort -V); do
    echo "    --- $(basename "$f")"
    grep -m3 -nE "Error|Traceback|OutOfMemory|Address already in use|RuntimeError|AssertionError|Killed" "$f" \
      | grep -v "may indicate a possible application crash" | sed 's/^/      /'
    tail -3 "$f" | sed 's/^/      | /'
  done
}
run_arm(){  # $1=label $2=M -> echoes the PIPEP json of every rank
  local label=$1 m=$2 tag="${1//@/at}_p${P}_M${2}_v${V}${LINK_TAG}"
  [ "$DRY" = 1 ] && { echo "  [dry] $tag: $(arm_env "$label")" >&2; return 0; }
  local port; port=$(free_port)
  local cmd=""
  for r in $(seq 0 $((P-1))); do
    cmd="$cmd RANK=$r CUDA_VISIBLE_DEVICES=${G[$r]} $PY -u pipep.py > $LOGDIR/$tag.r$r.log 2>&1 &"
  done
  env MASTER_ADDR=127.0.0.1 MASTER_PORT=$port WORLD_SIZE=$P $COMMON $LINK_ENV $(arm_env "$label") \
    ARM_LABEL=$label${LINK_TAG//_/@} M=$m bash -c "$cmd wait"
  cat "$LOGDIR/$tag".r*.log 2>/dev/null | grep -ah '^PIPEP'
}

RES=$(mktemp)
for m in $MS; do for a in $ARMS; do
  out=$(run_arm "$a" "$m")
  [ "$DRY" = 1 ] && continue
  n=$(printf '%s\n' "$out" | grep -c '^PIPEP')
  if [ "$n" -ne "$P" ]; then
    echo "[FAIL] $a M=$m (only $n/$P ranks reported); first error in each rank's log:"
    show_fail "${a//@/at}_p${P}_M${m}_v${V}"
    if [ "$FAIL_FAST" = 1 ]; then echo "[stop] FAIL_FAST=1, exiting"; rm -f "$RES"; exit 1; fi
    continue
  fi
  printf '%s\n' "$out" >> "$RES"
  echo "  done $a M=$m"
done; done
[ "$DRY" = 1 ] && { rm -f "$RES"; exit 0; }
[ "${NO_TABLE:-0}" = 1 ] && { rm -f "$RES"; exit 0; }

echo
$PY - "$RES" "$ARMS" <<'PYX'
import json, sys
rows = [json.loads(l[6:]) for l in open(sys.argv[1]) if l.startswith("PIPEP ")]
order = sys.argv[2].split()
by = {}
for r in rows:
    by.setdefault((r["M"], r["label"]), []).append(r)
P = rows[0]["p"]
ms = sorted({k[0] for k in by})


def summarize(rs):
    top = max(rs, key=lambda x: x["peak_gb"])       # memory split taken from the rank with the highest peak
    return dict(tok=min(x["tok_per_s"] for x in rs), raw=min(x["tok_per_s_raw"] for x in rs),
                v=rs[0]["v"], arm=rs[0]["arm"], rows=rs[0].get("rows_per_iter"),
                peak=top["peak_gb"], state=top["state_gb"], act=top["act_gb_est"], rank=top["rank"],
                p2p=rs[0]["mb_p2p_analytic"], bcast=rs[0]["mb_bcast_analytic"],
                ar=rs[0].get("mb_allreduce_analytic", 0.0), par=rs[0].get("parallel", "pp"),
                bms=max(x["bcast_ms"] for x in rs), ts=rs[0]["tie_sync"])


def bubble(e, m):
    return {"solo": "0", "bp": f"{(P - 1) / (m + P - 1):.1%}", "bp_fw": f"{(P - 1) / (m + P - 1):.1%}",
            "bp_il": f"{(P - 1) / (e['v'] * m + P - 1):.1%}", "bp_zb": "~0",
            "bp_dp": "-", "solo_dp": "-", "solo_kr": "0", "bp_pd": "-"}[e["arm"]]


print(f"Table 1  throughput (p={P}; slowest rank; tied solo readout broadcast amortized over TIE_SYNC)")
print(f"      vs SOLO uses SOLO under the same parallelism as baseline: PP arms vs solo, DP arms vs solo_dp")
print(f"{'M':>3} {'arm':<10} {'par':>4} {'v':>2} {'tok/s':>8} {'vs SOLO':>8} {'bubble':>7} {'bcast ms':>9}")
TAB2 = []
for m in ms:
    ent = {lab: summarize(by[(m, lab)]) for lab in order
           if (m, lab) in by and len(by[(m, lab)]) == P}
    rws = {a: e["rows"] for a, e in ent.items() if e["rows"] is not None}
    if len(set(rws.values())) > 1:
        print(f"  [ABORT] M={m}: rows_per_iter differ {rws}, row dropped")
        continue
    def base_for(par):
        if par == "dp":
            return ent.get("solo_dp")
        if "solo" in ent:
            return ent["solo"]
        return next((ent[a] for a in ent if a.startswith("solo") and ent[a]["par"] == "pp"), None)
    for lab, e in ent.items():
        b = base_for(e["par"])
        vs = f"{b['tok'] / e['tok']:.3f}" if b else "-"
        bms = f"{e['bms']:.1f}" if e["ts"] else "-"
        print(f"{m:>3} {lab:<10} {e['par']:>4} {e['v']:>2} {e['tok']:>8} {vs:>8} {bubble(e, m):>7} {bms:>9}")
    print()
    TAB2.append((m, ent, base_for))

print(f"Table 2  memory and communication (memory from the rank with the highest peak; act = peak - state - grad, estimate)")
print(f"{'M':>3} {'arm':<10} {'par':>4} {'peak GB':>7} {'stateGB':>7} {'act GB':>7} {'P2P MB':>8} "
      f"{'bcast MB':>8} {'AllRed MB':>10} {'total MB':>9}")
for m, ent, base_for in TAB2:
    for lab, e in ent.items():
        tot = e["p2p"] + e["bcast"] + e["ar"]
        print(f"{m:>3} {lab:<10} {e['par']:>4} {e['peak']:>7.2f} {e['state']:>7.2f} {e['act']:>7.2f} "
              f"{e['p2p']:>8.1f} {e['bcast']:>8.2f} {e['ar']:>10.1f} {tot:>9.1f}")
    print()

print("Legend:")
print("  vs SOLO > 1 means SOLO is faster (solo_tok / arm_tok). Baseline is solo for PP arms and solo_dp for DP arms,")
print("  so each ratio compares SOLO and BP under one parallelism. Across parallelisms compare tok/s and memory directly.")
print("  Comm: PP arms = inter-stage P2P (+ solo readout broadcast); DP arms = gradient all-reduce (ring, fp32, by params).")
print("        solo_dp all-reduces the aux-head parameters in addition to bp_dp. Smaller TIE_SYNC = more broadcast (solo@1 etc.).")
print("  Memory is from the torch allocator and excludes NCCL buffers and the CUDA context.")
PYX
rm -f "$RES"
