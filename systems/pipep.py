"""pipep.py - pipeline-throughput A/B for any p: SOLO forward streaming vs BP schedules.
All arms share Block/Head/optimizer/data/timing.

Arms (ARM):
  solo     forward-only stream: rank k receives h -> its blocks -> sends h to k+1 -> local head + local loss
           -> local backward -> local step. No gradient communication, no global barrier, no bubble.
  bp       standard 1F1B, hand-written (warmup p-1-rank / steady 1F1B / cooldown). At most p microbatches in flight.
  bp_fw    the same 1F1B via torch.distributed.pipelining Schedule1F1B (v=1, same split as bp). The framework sends
           stage outputs in their own dtype (fp32 residual stream under autocast), so its P2P bytes are about 2x bp's.
  bp_il    interleaved 1F1B, VCHUNKS virtual stages per GPU (ScheduleInterleaved1F1B): bubble (p-1)/(v*m), P2P bytes x v.
  bp_zb    zero-bubble (ScheduleInterleavedZeroBubble / ScheduleZBVZeroBubble): bubble ~0, activation memory ~x2.
  bp_dp    BP + DDP: p full replicas, each runs M/p microbatches.
  solo_dp  SOLO + DDP: each replica holds DP_K (default p) modules, one DDP per module; shared readout copied locally.
  solo_kr  SOLO with K=KR_K modules x R replicas (K*R = WORLD_SIZE), rank = replica*K + module, one module per GPU.
           Gradients of the same module are all-reduced across replicas (DDP); only detached activations cross modules;
           the shared readout is broadcast within each replica every TIE_SYNC steps. R=1 equals solo.
  bp_pd    BP with pipeline depth D=PD_D x data parallel WORLD_SIZE/D (1F1B + DDP). D=WORLD_SIZE equals bp, D=1 equals bp_dp.

Options (env, default): NLAYER=24 DMODEL=1024 SEQ=1024 MB=4 VMAX=8192 HEAD_DEPTH=1 VCHUNKS=2 AUX=1.0 CE_MODE=compile
  M=8           microbatches per iteration, summed over replicas
  SPLIT=""      layers per stage, e.g. "6,6,6,6"; empty = even split
  SOLO_READOUT=tied  aux heads read a detached copy of the terminal readout, broadcast every TIE_SYNC=50 steps;
               priv  private trainable readout per stage, no broadcast
  TIE_ASYNC=1   async broadcast on a separate process group, double-buffered; 0 = synchronous broadcast
  TIE_LAG=auto  iterations between issuing and waiting for the async broadcast; auto = ceil((K-1)/M_local)
  RECOMPUTE=none|full  ITERS=20 WARM=10 BACKEND=nccl (gloo = CPU smoke test) OUT_JSON=""
  Self-checks: TIE_CHECK=1 DP_CHECK=1 PARAM_CHECK=1.

Output: one line per rank, "PIPEP <json>": tok/s (tied solo: broadcast amortized over TIE_SYNC), peak / state /
estimated activation memory, and analytic communication bytes per iteration.

Probe (10 s): PROBE=1 python pipep.py   # prints the torch version and available schedules; no processes launched
Launch: see run_pipep.sh (one grid) / run_pipep_mlt.sh (fixed global-batch sweep)
"""
import contextlib
import json
import math
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

E = os.environ.get

# ---- probe: no distributed init, report framework capabilities -------------------------
if int(E("PROBE", "0")):
    print(f"torch {torch.__version__}")
    try:
        import torch.distributed.pipelining as pp
        names = [n for n in dir(pp) if n.startswith("Schedule")]
        print("torch.distributed.pipelining available, schedules:", ", ".join(sorted(names)))
        print("  1F1B        :", "ok" if hasattr(pp, "Schedule1F1B") else "missing")
        print("  Interleaved :", "ok" if hasattr(pp, "ScheduleInterleaved1F1B") else "missing")
        zb = [n for n in names if "Zero" in n or "ZB" in n]
        print("  ZeroBubble  :", ", ".join(zb) if zb else "missing (bp_zb arm will be skipped)")
        import inspect                                              # constructor signatures (version differences)
        for n in ("PipelineStage", "Schedule1F1B", "ScheduleInterleaved1F1B"):
            if hasattr(pp, n):
                c = getattr(pp, n)
                try:
                    sig = str(inspect.signature(c))
                except (TypeError, ValueError):
                    sig = "(?)"
                print(f"  {n}{sig}  bases: {[b.__name__ for b in c.__mro__[1:-1]]}")
    except Exception as e:                                          # noqa: BLE001
        print(f"torch.distributed.pipelining unavailable: {type(e).__name__}: {e}")
        print("  => bp_il / bp_zb cannot run; solo and bp are unaffected (hand-written)")
    raise SystemExit(0)

ARM = E("ARM", "solo")
assert ARM in ("solo", "bp", "bp_fw", "bp_il", "bp_zb", "bp_dp", "solo_dp", "solo_kr", "bp_pd"), ARM
DP_ARM = ARM in ("bp_dp", "solo_dp")   # data parallel: p full replicas, M/p microbatches each, gradient all-reduce
HAND = ARM in ("solo", "bp", "solo_kr", "bp_pd")   # hand-written pipeline arms (solo/bp = the R=1 case)
SOLO_LIKE = ARM in ("solo", "solo_kr")
KR_K = int(E("KR_K", "2"))             # solo_kr: modules per replica K
PD_D = int(E("PD_D", "4"))             # bp_pd: pipeline stages per replica D
NLAYER = int(E("NLAYER", "24"))
DMODEL = int(E("DMODEL", "1024"))
NHEAD = int(E("NHEAD", str(max(1, int(E("DMODEL", "1024")) // 64))))
FFN = int(E("FFN", "0")) or 4 * DMODEL
SEQ = int(E("SEQ", "1024"))
MB = int(E("MB", "4"))
M = int(E("M", "8"))
VMAX = int(E("VMAX", "8192"))
HEAD_DEPTH = int(E("HEAD_DEPTH", "1"))
VCHUNKS = int(E("VCHUNKS", "2"))       # bp_il / bp_zb: virtual stages per GPU v
DROP = float(E("DROP", "0.1"))
LS = float(E("LS", "0.0"))
AUX = float(E("AUX", "1.0"))
LR = float(E("LR", "3e-4"))
CLIP = float(E("CLIP", "0.5"))
CE_MODE = E("CE_MODE", "compile")
assert CE_MODE in ("autocast", "bf16", "compile"), CE_MODE
SPLIT = E("SPLIT", "")                 # "6,6,6,6": layers per stage; empty = even split
SOLO_READOUT = E("SOLO_READOUT", "tied")  # readout of the solo aux heads:
    # tied = aux heads read a detached copy of the terminal readout, broadcast by the last stage every TIE_SYNC steps;
    # priv = private trainable readout per stage, no broadcast
assert SOLO_READOUT in ("tied", "priv"), SOLO_READOUT
TIE_SYNC = int(E("TIE_SYNC", "50"))    # tied: broadcast period (optimizer steps)
assert TIE_SYNC >= 1, TIE_SYNC
TIE_ASYNC = int(E("TIE_ASYNC", "1"))   # 1 = async broadcast (separate process group + double buffer + lagged wait); 0 = sync
TIE_LAG = E("TIE_LAG", "auto")         # iterations between async issue and wait/swap; auto = ceil((K-1)/M_local)
RECOMPUTE = E("RECOMPUTE", "none")     # none | full: full = checkpoint every trunk block (store only block inputs)
assert RECOMPUTE in ("none", "full"), RECOMPUTE
ARM_LABEL = E("ARM_LABEL", "")         # launcher label for variants of one ARM (e.g. solo@1 / solo@50)
ITERS = int(E("ITERS", "20"))
WARM = int(E("WARM", "10"))
SEED = int(E("SEED", "42"))
OUT_JSON = E("OUT_JSON", "")
BACKEND = E("BACKEND", "nccl")

RANK = int(os.environ["RANK"])
P = int(os.environ["WORLD_SIZE"])
assert P >= 2, P
torch.manual_seed(SEED)
CUDA = BACKEND == "nccl" and torch.cuda.is_available()
dev = "cuda" if CUDA else "cpu"
if CUDA:
    torch.cuda.set_device(0)
dist.init_process_group(BACKEND, rank=RANK, world_size=P)
V = VMAX
AC = dict(device_type=dev, dtype=torch.bfloat16, enabled=CUDA)

# ---- layout: NSTAGE stages per replica x R replicas, rank = replica * NSTAGE + stage ------------------------
# solo/bp/framework arms: NSTAGE=p, R=1. bp_dp/solo_dp: NSTAGE=1, R=p. solo_kr/bp_pd: NSTAGE=K or D.
if ARM == "solo_kr":
    NSTAGE = KR_K
elif ARM == "bp_pd":
    NSTAGE = PD_D
elif DP_ARM:
    NSTAGE = 1
else:
    NSTAGE = P
assert 1 <= NSTAGE <= P and P % NSTAGE == 0, f"stages per replica {NSTAGE} must divide the GPU count {P}"
R = P // NSTAGE                                   # data-parallel replicas
REP, STAGE = RANK // NSTAGE, RANK % NSTAGE
assert M % R == 0, f"microbatch count M={M} must be divisible by R={R} (replica count)"
M_LOCAL = M // R                                  # microbatches per replica per iteration
IS_FIRST, IS_LAST = STAGE == 0, STAGE == NSTAGE - 1


# ----------------------------------------------------------------- CE
def _ce_none_raw(z, y):
    return F.cross_entropy(z, y, reduction="none")


def _ce_mean_raw(z, y, ls):
    return F.cross_entropy(z, y, label_smoothing=ls)


_CE_NONE, _CE_MEAN = _ce_none_raw, _ce_mean_raw
if CE_MODE == "compile":
    _CE_NONE, _CE_MEAN = torch.compile(_ce_none_raw), torch.compile(_ce_mean_raw)
_FELL_BACK = False


def ce_loss(z, y, ls=0.0, none=False):
    global _CE_NONE, _CE_MEAN, _FELL_BACK
    if CE_MODE == "autocast":
        return F.cross_entropy(z, y, reduction="none") if none \
            else F.cross_entropy(z, y, label_smoothing=ls)
    with torch.autocast(dev, enabled=False):
        try:
            return _CE_NONE(z, y) if none else _CE_MEAN(z, y, ls)
        except Exception as e:                                   # noqa: BLE001
            if CE_MODE != "compile" or _FELL_BACK:
                raise
            _FELL_BACK = True
            _CE_NONE, _CE_MEAN = _ce_none_raw, _ce_mean_raw
            print(f"[r{RANK}][ce] compile failed, falling back to bf16: {type(e).__name__}", flush=True)
            return _CE_NONE(z, y) if none else _CE_MEAN(z, y, ls)


# ----------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(DMODEL)
        self.attn = nn.MultiheadAttention(DMODEL, NHEAD, dropout=DROP, batch_first=True)
        self.ln2 = nn.LayerNorm(DMODEL)
        self.mlp = nn.Sequential(nn.Linear(DMODEL, FFN), nn.GELU(),
                                 nn.Linear(FFN, DMODEL), nn.Dropout(DROP))

    def forward(self, x, m):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=m, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class Head(nn.Module):
    """tied=True: no trainable readout; reads the w_sync/b_sync buffers (copy of the terminal readout written by the
    broadcast). Buffers are HDT (bf16 on CUDA), not optimized, no grad_W. The async broadcast writes w_next/b_next, which
    are swapped with w_sync/b_sync after wait (double buffer: the broadcast never writes a buffer a forward is reading)."""
    def __init__(self, depth=0, tied=False):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(depth)])
        self.ln = nn.LayerNorm(DMODEL)
        self.tied = tied
        if tied:
            self.fc = None
            self.register_buffer("w_sync", torch.zeros(V, DMODEL, dtype=HDT), persistent=False)
            self.register_buffer("b_sync", torch.zeros(V, dtype=HDT), persistent=False)
            self.register_buffer("w_next", torch.zeros(V, DMODEL, dtype=HDT), persistent=False)
            self.register_buffer("b_next", torch.zeros(V, dtype=HDT), persistent=False)
        else:
            self.fc = nn.Linear(DMODEL, V)

    def swap_sync(self):
        self.w_sync, self.w_next = self.w_next, self.w_sync
        self.b_sync, self.b_next = self.b_next, self.b_sync

    def forward(self, h, m):
        for b in self.blocks:
            h = b(h, m)
        h = self.ln(h)
        return F.linear(h, self.w_sync, self.b_sync) if self.tied else self.fc(h)



# ----------------------------------------------------------------- stage split
def even_split(total, n):
    q, r = divmod(total, n)
    return [q + (1 if i < r else 0) for i in range(n)]


if SPLIT:
    SIZES = [int(s) for s in SPLIT.split(",")]
    assert len(SIZES) == NSTAGE and sum(SIZES) == NLAYER, (SIZES, NSTAGE, NLAYER)
else:
    SIZES = even_split(NLAYER, NSTAGE)


class Chunk(nn.Module):
    """A contiguous trunk segment. first also holds emb/pos; last also holds the readout head.
    Non-last solo stages hold a HEAD_DEPTH-deep aux head.
    forward takes and returns a single tensor, as torch.distributed.pipelining requires."""
    def __init__(self, n_blocks, first, last, aux_depth=None, tied=False):
        super().__init__()
        self.first, self.last = first, last
        if first:
            self.emb = nn.Embedding(V, DMODEL)
            self.pos = nn.Embedding(SEQ, DMODEL)
            nn.init.normal_(self.emb.weight, std=0.02)
            nn.init.normal_(self.pos.weight, std=0.02)
            self.drop = nn.Dropout(DROP)
        self.blocks = nn.ModuleList([Block() for _ in range(n_blocks)])
        self.head = None if aux_depth is None else Head(0 if last else aux_depth, tied=tied)

    def trunk(self, x, m):
        if self.first:
            p = torch.arange(x.size(1), device=x.device)
            h = self.drop(self.emb(x) + self.pos(p)[None])
        else:
            h = x
        for b in self.blocks:
            if RECOMPUTE == "full" and torch.is_grad_enabled():
                h = torch.utils.checkpoint.checkpoint(b, h, m, use_reentrant=False)
            else:
                h = b(h, m)
        return h

    def forward(self, x):
        """Path used by torch.distributed.pipelining: the last stage returns (n,V) logits, the others return h."""
        m = _mask(x.size(1))
        with torch.autocast(**AC):
            h = self.trunk(x, m)
            return self.head(h, m).reshape(-1, V) if self.last else h


_MASK_CACHE = {}


def _mask(T):
    k = (T, dev)
    if k not in _MASK_CACHE:
        _MASK_CACHE[k] = torch.full((T, T), float("-inf"), device=dev).triu(1)
    return _MASK_CACHE[k]


# ----------------------------------------------------------------- data (same seed and order on all ranks)
_g = torch.Generator().manual_seed(SEED)
_r = torch.arange(1, V + 1, dtype=torch.double)
_p = (_r ** -1.0)
_p = (_p / _p.sum()).float()
_BATCHES = [(torch.multinomial(_p, MB * (SEQ + 1), replacement=True, generator=_g)
             .view(MB, SEQ + 1)) for _ in range(8)]
_BATCHES = [(b[:, :SEQ].to(dev), b[:, 1:].to(dev).reshape(-1)) for b in _BATCHES]
# torch.distributed.pipelining's schedule.step(x) takes the whole global batch and splits it into n_microbatches,
# so framework arms are fed (M*MB, SEQ) to get MB rows per microbatch.
FRAMEWORK = ARM in ("bp_fw", "bp_il", "bp_zb")
if FRAMEWORK:
    _gb = torch.multinomial(_p, M * MB * (SEQ + 1), replacement=True,
                            generator=_g).view(M * MB, SEQ + 1)
    GBATCH = (_gb[:, :SEQ].to(dev), _gb[:, 1:].to(dev))          # y stays (M*MB, SEQ); the schedule splits rows
    del _gb
del _r, _p
MASK = _mask(SEQ)
HSHAPE = (MB, SEQ, DMODEL)
HDT = torch.bfloat16 if CUDA else torch.float32
BYTES_PER_MSG = MB * SEQ * DMODEL * HDT.itemsize     # hand-written arms: inter-stage messages are sent as HDT
FW_MSG_BYTES = None                                  # framework arms: set by build_framework from the measured dtype
_FW_ROWS = [0]                                       # framework arms: rows seen by the last-stage loss_fn (self-check)


def batch_for(i):
    """Microbatch i of replica REP; replicas take different microbatches."""
    return _BATCHES[(i * R + REP) % len(_BATCHES)]


_wait = 0.0
_nbytes = 0


def _recv(buf, src):
    global _wait
    t = time.time()
    dist.recv(buf, src=src)
    _wait += time.time() - t


# In-flight sends are reclaimed lazily across iterations: post waits for the oldest one only above cap;
# drain_inflight empties the queue at the end of warmup, at the end of the timing window, and before exit.
_INFLIGHT = []


def post(t, dst, cap=2):
    global _nbytes
    _INFLIGHT.append((dist.isend(t, dst=dst), t))
    _nbytes += BYTES_PER_MSG
    while len(_INFLIGHT) > cap:
        _INFLIGHT.pop(0)[0].wait()


def drain_inflight():
    while _INFLIGHT:
        _INFLIGHT.pop(0)[0].wait()


# ----------------------------------------------------------------- hand-written arms
class StageModule(nn.Module):
    """One stage of a hand-written pipeline arm, a single nn.Module so it can be wrapped by DDP when R>1
    (the DDP reducer requires the forward to go through its __call__).
    solo-like: forward(x, send) = trunk -> send(h) (early, so downstream can start) -> [wait for a due readout
               broadcast] -> head -> logits.
    bp-like:   forward(x) = trunk -> last-stage head -> logits; non-last stages return h (as Chunk.forward)."""
    def __init__(self, chunk):
        super().__init__()
        self.chunk = chunk

    def forward(self, x, send=None):
        with torch.autocast(**AC):
            h = self.chunk.trunk(x, MASK)
            if send is not None:
                send(h)
            if SOLO_LIKE:
                tie_wait_if_due()
                return self.chunk.head(h, MASK).reshape(-1, V)
            return self.chunk.head(h, MASK).reshape(-1, V) if self.chunk.last else h


def sched_solo():
    """Forward-only stream: recv h -> local trunk -> send h downstream (early) -> local head + local loss -> local backward.
    No gradient communication. With R>1 (solo_kr) the first M_LOCAL-1 microbatches accumulate under DDP no_sync and the
    last backward triggers this stage's gradient all-reduce (as in solo_dp)."""
    buf = None if IS_FIRST else torch.empty(HSHAPE, device=dev, dtype=HDT)
    send = None if IS_LAST else (lambda h: post(h.detach().to(HDT).contiguous(), RANK + 1))
    for i in range(M_LOCAL):
        x, y = batch_for(i)
        if IS_FIRST:
            inp = x
        else:
            _recv(buf, RANK - 1)
            inp = buf.to(torch.float32, copy=True)   # no gradient is sent back
        with contextlib.ExitStack() as st:
            if DDP_MOD is not None and i < M_LOCAL - 1:
                st.enter_context(DDP_MOD.no_sync())
            z = MOD(inp, send)
        with torch.autocast(**AC):
            lk = ce_loss(z, y, ls=LS) if IS_LAST else AUX * ce_loss(z, y, none=True).float().mean()
        (lk / M_LOCAL).backward()


def sched_bp_1f1b():
    """Standard 1F1B: (p-1-rank) warmup forwards, then interleaved 1F1B, then cooldown. At most p microbatches in flight.
    Gradient scaling: the last stage runs (loss/M).backward(), so gradients received upstream already include 1/M.
    R>1 (bp_pd): all forwards run under no_sync (DDP does not prepare the reducer in forward); prepare_for_backward is
    called by hand before the last microbatch's backward, whose hooks launch bucketed all-reduces overlapped with it."""
    buf = None if IS_FIRST else torch.empty(HSHAPE, device=dev, dtype=HDT)
    gbuf = None if IS_LAST else torch.empty(HSHAPE, device=dev, dtype=HDT)

    def do_fwd(i):
        x, y = batch_for(i)
        if IS_FIRST:
            leaf, inp = None, x
        else:
            _recv(buf, RANK - 1)
            leaf = buf.to(torch.float32, copy=True).requires_grad_(True)
            inp = leaf
        with contextlib.ExitStack() as st:
            if DDP_MOD is not None:
                st.enter_context(DDP_MOD.no_sync())
            out = MOD(inp)
        if IS_LAST:
            with torch.autocast(**AC):
                out = ce_loss(out, y, ls=LS) / M_LOCAL
        else:
            post(out.to(HDT).contiguous(), RANK + 1)
        return leaf, out

    def do_bwd(item, last):
        leaf, out = item
        if DDP_MOD is not None and last:
            DDP_MOD.reducer.prepare_for_backward([])   # as torch.distributed.pipelining combines with DDP
        if IS_LAST:
            out.backward()
        else:
            _recv(gbuf, RANK + 1)      # not counted: the sender's post already counted this message
            out.backward(gbuf.float())
        if leaf is not None:
            post(leaf.grad.to(HDT).contiguous(), RANK - 1, cap=1)

    nw = min(NSTAGE - 1 - STAGE, M_LOCAL)
    q = []                                   # (microbatch index, (leaf, out)); backward of index M_LOCAL-1 triggers the all-reduce
    for i in range(nw):
        q.append((i, do_fwd(i)))
    for i in range(nw, M_LOCAL):
        q.append((i, do_fwd(i)))
        j, item = q.pop(0)
        do_bwd(item, last=(j == M_LOCAL - 1))
    while q:
        j, item = q.pop(0)
        do_bwd(item, last=(j == M_LOCAL - 1))


# ----------------------------------------------------------------- framework arms (bp_fw / interleaved / zero-bubble)
def build_framework():
    """Returns (schedule, params, n_virtual, schedule name), using the torch.distributed.pipelining reference schedules."""
    import torch.distributed.pipelining as pp
    global FW_MSG_BYTES
    v = 1 if ARM == "bp_fw" else VCHUNKS
    ns = v * P
    assert ns <= NLAYER, f"virtual stages {ns} > layers {NLAYER}"
    sizes_v = SIZES if v == 1 else even_split(NLAYER, ns)          # v=1: same split as hand-written bp (incl. SPLIT)
    my_js = [c * P + RANK for c in range(v)]
    mods = [Chunk(sizes_v[j], first=(j == 0), last=(j == ns - 1),
                  aux_depth=(0 if j == ns - 1 else None)).to(dev) for j in my_js]
    try:
        stages = [pp.PipelineStage(mods[c], my_js[c], ns, torch.device(dev)) for c in range(v)]
    except TypeError as e:                                           # signature differs across torch versions
        raise SystemExit(f"[abort] PipelineStage signature mismatch: {e}\n"
                         f"        run PROBE=1 python pipep.py to see what this torch provides") from e

    # Actual inter-stage message bytes: the framework sends stage outputs in their own dtype. Rank 0 runs one forward
    # on one row to measure bytes per element and broadcasts it; the analytic communication uses it.
    code = torch.zeros(1, dtype=torch.long, device=dev)
    if IS_FIRST:
        with torch.no_grad():
            code[0] = mods[0](GBATCH[0][:1]).element_size()
    dist.broadcast(code, src=0)
    FW_MSG_BYTES = MB * SEQ * DMODEL * int(code.item())

    def loss_fn(z, y):
        # z: (rows*SEQ, V) logits of this microbatch; y: (rows, SEQ) its labels (the schedule splits along dim 0)
        _FW_ROWS[0] += y.numel() // SEQ
        return ce_loss(z, y.reshape(-1), ls=LS)

    if ARM == "bp_il":
        cls = getattr(pp, "ScheduleInterleaved1F1B", None)
        if cls is None:
            raise SystemExit("[abort] this torch has no ScheduleInterleaved1F1B (PROBE=1 lists available schedules)")
    elif ARM == "bp_fw":
        cls = getattr(pp, "Schedule1F1B", None)
        if cls is None:
            raise SystemExit("[abort] this torch has no Schedule1F1B (PROBE=1 lists available schedules)")
    else:
        cls = next((getattr(pp, n) for n in
                    ("ScheduleInterleavedZeroBubble", "ScheduleZBVZeroBubble", "ScheduleZeroBubble")
                    if hasattr(pp, n)), None)
        if cls is None:
            raise SystemExit("[abort] this torch has no zero-bubble schedule (PROBE=1 lists available schedules); "
                             "the bp_zb arm needs roughly torch >= 2.5")
    assert (M * MB) % M == 0, (M, MB)
    # The constructor takes a single stage or a stage list depending on the torch version; with v=1 try a single stage
    # first, then a list, and exit with both errors if both fail.
    errs = []
    for arg in ([stages[0], stages] if v == 1 else [stages]):
        try:
            sched = cls(arg, n_microbatches=M, loss_fn=loss_fn)
            break
        except (TypeError, AttributeError, AssertionError) as e:
            errs.append(f"passing {type(arg).__name__}: {type(e).__name__}: {e}")
    else:
        raise SystemExit(f"[abort] constructing {cls.__name__} failed (PROBE=1 shows signatures):\n  " + "\n  ".join(errs))
    params = [q for mo in mods for q in mo.parameters()]
    return sched, params, v, type(sched).__name__


def sched_framework(sched):
    x, y = GBATCH
    if IS_FIRST and IS_LAST:
        sched.step(x, target=y, losses=[])
    elif IS_FIRST:
        sched.step(x)
    elif IS_LAST:
        sched.step(target=y, losses=[])
    else:
        sched.step()


# ----------------------------------------------------------------- data-parallel arms (bp_dp / solo_dp)
class SoloDPModule(nn.Module):
    """One SOLO module under data parallelism; forward returns (h, logits). Logits come from the aux head on non-last
    modules and from the terminal head on the last. Each module has its own DDP, so its gradient all-reduce overlaps
    with the following modules' compute during per-module backward, as BP+DDP overlaps all-reduce with backward."""
    def __init__(self, chunk):
        super().__init__()
        self.chunk = chunk

    def forward(self, x):
        with torch.autocast(**AC):
            h = self.chunk.trunk(x, MASK)
            z = self.chunk.head(h, MASK).reshape(-1, V)
        return h, z


def sync_local_readout():
    """solo_dp shared readout: copy this replica's terminal readout into the aux heads' buffers (same GPU, no communication)."""
    fc = DP_CHUNKS[-1].head.fc
    with torch.no_grad():
        for c in DP_CHUNKS[:-1]:
            c.head.w_sync.copy_(fc.weight)
            c.head.b_sync.copy_(fc.bias)


def sched_dp():
    """Each replica runs M/p microbatches of size MB; the first M/p-1 accumulate under no_sync and the last backward
    triggers the all-reduce. Replicas take different microbatches."""
    for i in range(M_LOCAL):
        x, y = batch_for(i)
        last = i == M_LOCAL - 1
        with contextlib.ExitStack() as st:
            if not last:
                for d in DDP_MODS:
                    st.enter_context(d.no_sync())
            if ARM == "bp_dp":
                z = DDP_MODS[0](x)
                with torch.autocast(**AC):
                    loss = ce_loss(z, y, ls=LS) / M_LOCAL
                loss.backward()
            else:
                inp = x
                for k, d in enumerate(DDP_MODS):
                    h, z = d(inp)
                    with torch.autocast(**AC):
                        if k == len(DDP_MODS) - 1:
                            lk = ce_loss(z, y, ls=LS)
                        else:
                            lk = AUX * ce_loss(z, y, none=True).float().mean()
                    (lk / M_LOCAL).backward()      # per-module backward frees this module's graph
                    inp = h.detach()


# ----------------------------------------------------------------- build
NVIRT, SCHED_NAME = 1, ARM
DP_TIED = False
TIED = False
DDP_MOD = None                     # DDP wrapper of this stage for hand-written arms with R>1 (None for R=1)
MOD = None
TIE_GROUP, TIE_SRC = None, None    # tied readout broadcast: process group (within a replica) and source rank (its last stage)
STAGE_GROUPS = REP_GROUPS = None
if HAND:
    from torch.nn.parallel import DistributedDataParallel as DDP
    TIED = SOLO_LIKE and SOLO_READOUT == "tied"
    chunk = Chunk(SIZES[STAGE], IS_FIRST, IS_LAST,
                  aux_depth=(0 if IS_LAST else (HEAD_DEPTH if SOLO_LIKE else None)),
                  tied=(TIED and not IS_LAST)).to(dev)
    MOD = StageModule(chunk)
    if R > 1:
        # one group per stage across replicas (gradient all-reduce), one per replica across stages (readout broadcast);
        # all ranks create the groups in the same order.
        STAGE_GROUPS = [dist.new_group([r_ * NSTAGE + s_ for r_ in range(R)]) for s_ in range(NSTAGE)]
        REP_GROUPS = [dist.new_group([r_ * NSTAGE + s_ for s_ in range(NSTAGE)]) for r_ in range(R)]
        ddp_kw = dict(broadcast_buffers=False, gradient_as_bucket_view=True,
                      process_group=STAGE_GROUPS[STAGE])
        if CUDA:
            ddp_kw["device_ids"] = [0]
        DDP_MOD = DDP(MOD, **ddp_kw)
        MOD = DDP_MOD
        SCHED_NAME = f"{'solo' if SOLO_LIKE else '1f1b'}_{NSTAGE}x{R}"
        TIE_GROUP, TIE_SRC = REP_GROUPS[REP], REP * NSTAGE + NSTAGE - 1
    elif TIED:
        TIE_GROUP, TIE_SRC = dist.new_group(list(range(P))), P - 1   # separate communicator: does not serialize with P2P
    PARAMS = list(chunk.parameters())
    SCHED = None
elif DP_ARM:
    from torch.nn.parallel import DistributedDataParallel as DDP
    ddp_kw = dict(broadcast_buffers=False, gradient_as_bucket_view=True)
    if CUDA:
        ddp_kw["device_ids"] = [0]
    if ARM == "bp_dp":
        DP_CHUNKS = [Chunk(NLAYER, True, True, aux_depth=0).to(dev)]
        DDP_MODS = [DDP(DP_CHUNKS[0], **ddp_kw)]
        SCHED_NAME = "ddp"
    else:
        DP_K = int(E("DP_K", str(P)))     # number of modules, default p (as SOLO+PP)
        sizes_dp = even_split(NLAYER, DP_K)
        DP_TIED = SOLO_READOUT == "tied"
        DP_CHUNKS = [Chunk(sizes_dp[k], k == 0, k == DP_K - 1,
                           aux_depth=(0 if k == DP_K - 1 else HEAD_DEPTH),
                           tied=(DP_TIED and k < DP_K - 1)).to(dev) for k in range(DP_K)]
        DDP_MODS = [DDP(SoloDPModule(c), **ddp_kw) for c in DP_CHUNKS]
        SCHED_NAME = f"ddp_x{DP_K}"
    PARAMS = [q for c in DP_CHUNKS for q in c.parameters()]
    SCHED = None
    chunk = None
else:
    SCHED, PARAMS, NVIRT, SCHED_NAME = build_framework()
    chunk = None
opt = torch.optim.AdamW(PARAMS, lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
nparam = sum(q.numel() for q in PARAMS)
N_TRAIN = sum(q.numel() for q in PARAMS if q.requires_grad)
# Trainable parameters of one replica (all-reduce accounting): hand-written arms hold one stage per GPU, so sum and divide by R.
if HAND and R > 1:
    _n = torch.tensor([float(N_TRAIN)], dtype=torch.float64, device=dev)
    dist.all_reduce(_n)
    N_REPLICA = int(_n.item()) // R
else:
    N_REPLICA = N_TRAIN

# Analytic cross-device traffic (global, per iteration). Each forward microbatch crosses (v*NSTAGE - 1) stage boundaries
# and BP sends as many gradients back. Tied solo adds the readout broadcast: every TIE_SYNC steps the last stage of each
# replica sends the terminal readout (V*d + V values) to the other NSTAGE-1 ranks of that replica.
HOPS = NVIRT * NSTAGE - 1
MSG_BYTES = FW_MSG_BYTES if FRAMEWORK else BYTES_PER_MSG
BYTES_P2P = M * HOPS * MSG_BYTES * (1 if SOLO_LIKE else 2)        # M = microbatches over all replicas
BYTES_BCAST = R * (NSTAGE - 1) * (V * DMODEL + V) * HDT.itemsize / TIE_SYNC if TIED else 0.0
# Ring all-reduce: each rank sends 2(R-1)/R gradient copies, 2(R-1) summed over R replicas; fp32 gradients.
BYTES_ALLREDUCE = 2 * (R - 1) * N_REPLICA * 4 if R > 1 else 0
BYTES_ANALYTIC = BYTES_P2P + BYTES_BCAST + BYTES_ALLREDUCE
GRAD_GB = N_TRAIN * 4 / 2 ** 30          # fp32 parameters, gradients of the same size
ROWS_PER_ITER = M * MB          # must be equal across arms; the launcher checks it
if TIE_LAG == "auto":
    TIE_LAG_N = math.ceil((NSTAGE - 1) / M_LOCAL)   # iterations the first stage runs before the last stage's step completes
else:
    TIE_LAG_N = int(TIE_LAG)
TIE_LAG_N = max(0, min(TIE_LAG_N, TIE_SYNC - 1))    # wait before the next broadcast is issued (0 when TIE_SYNC=1)


_step = 0
_bcast_times = []
_tie_pending = None          # (work_w, work_b, issue_step) of an issued async broadcast not yet waited
_tie_wait_s = 0.0            # CPU seconds spent in wait inside the timing window (real blocking under gloo, ~0 under nccl)


def _tie_src():
    """Last stage: HDT copy of the terminal readout (copy=True: a step during the send does not change in-flight data)."""
    return (chunk.head.fc.weight.detach().to(dtype=HDT, copy=True),
            chunk.head.fc.bias.detach().to(dtype=HDT, copy=True))


def tie_drain():
    """Wait for a pending async broadcast; non-last stages swap the double buffer."""
    global _tie_pending, _tie_wait_s
    if _tie_pending is None:
        return
    hw, hb, _s = _tie_pending
    t = time.time()
    hw.wait()
    hb.wait()
    if not IS_LAST:
        chunk.head.swap_sync()
    _tie_wait_s += time.time() - t
    _tie_pending = None


def tie_wait_if_due():
    """Call before the first head use: wait and swap if a broadcast was issued at least TIE_LAG_N iterations ago."""
    if _tie_pending is not None and _step >= _tie_pending[2] + TIE_LAG_N:
        tie_drain()


def tie_broadcast():
    """Synchronous broadcast: terminal readout -> w_sync/b_sync of the other ranks of the replica. Collective: all ranks of
    the group call it at the same step. Synchronizes before and after and returns the wall time (s). Used at step 0, for
    the fallback measurement when the window has no broadcast, for the self-check, and with TIE_ASYNC=0."""
    tie_drain()
    if IS_LAST:
        w, b = _tie_src()
    else:
        w, b = chunk.head.w_sync, chunk.head.b_sync
    if CUDA:
        torch.cuda.synchronize()
    t = time.time()
    dist.broadcast(w, src=TIE_SRC, group=TIE_GROUP)
    dist.broadcast(b, src=TIE_SRC, group=TIE_GROUP)
    if CUDA:
        torch.cuda.synchronize()
    return time.time() - t


def tie_broadcast_async():
    """Async broadcast: returns after issuing and writes w_next/b_next; tie_wait_if_due waits and swaps after TIE_LAG_N
    iterations. Returns the CPU seconds of the issue (~0 under nccl; any non-overlapped cost stays in the wall time)."""
    global _tie_pending
    tie_drain()
    if IS_LAST:
        w, b = _tie_src()
    else:
        w, b = chunk.head.w_next, chunk.head.b_next
    t = time.time()
    hw = dist.broadcast(w, src=TIE_SRC, group=TIE_GROUP, async_op=True)
    hb = dist.broadcast(b, src=TIE_SRC, group=TIE_GROUP, async_op=True)
    _tie_pending = (hw, hb, _step)
    return time.time() - t


def one_iter():
    """Returns the seconds spent on the readout broadcast in this step (0 if none), removed from timing and amortized over TIE_SYNC."""
    global _step
    opt.zero_grad(set_to_none=True)
    if SOLO_LIKE:
        sched_solo()
    elif ARM in ("bp", "bp_pd"):
        sched_bp_1f1b()
    elif DP_ARM:
        sched_dp()
    else:
        sched_framework(SCHED)
    torch.nn.utils.clip_grad_norm_(PARAMS, CLIP)
    opt.step()
    _step += 1
    if DP_TIED and _step % TIE_SYNC == 0:
        sync_local_readout()
    if TIED and _step % TIE_SYNC == 0:
        return tie_broadcast_async() if TIE_ASYNC else tie_broadcast()
    return 0.0


if RANK == 0:
    print(f"[cfg] ARM={ARM}({SCHED_NAME}) label={ARM_LABEL or ARM} p={P} stages={NSTAGE} replicas={R} "
          f"M_local={M_LOCAL} L={NLAYER}({SIZES}) v={NVIRT} "
          f"d={DMODEL} T={SEQ} V={V} MB={MB} M={M} H={HEAD_DEPTH} CE_MODE={CE_MODE} ITERS={ITERS} "
          f"readout={SOLO_READOUT if ARM in ('solo', 'solo_dp', 'solo_kr') else '-'} "
          f"tie_sync={TIE_SYNC if (TIED or DP_TIED) else '-'} "
          f"tie_async={(TIE_ASYNC, TIE_LAG_N) if TIED else '-'} "
          f"recompute={RECOMPUTE}", flush=True)
if TIED:
    tie_broadcast()                      # step-0 copy = terminal readout at init (not timed)
if DP_TIED:
    sync_local_readout()
for _ in range(WARM):
    one_iter()
drain_inflight()
if CUDA:
    torch.cuda.synchronize()
dist.barrier()

# State memory = parameters + optimizer state + buffers (measured with gradients cleared);
# peak - state - gradients ~= activations and temporaries.
opt.zero_grad(set_to_none=True)
if CUDA:
    torch.cuda.synchronize()
STATE_GB = torch.cuda.memory_allocated() / 2 ** 30 if CUDA else 0.0

_wait = 0.0
_nbytes = 0
_tie_wait_s = 0.0
if CUDA:
    torch.cuda.reset_peak_memory_stats()
t_bcast = 0.0
n_bcast = 0
_FW_ROWS[0] = 0
t0 = time.time()
for _ in range(ITERS):
    dt = one_iter()
    if dt > 0:
        t_bcast += dt
        n_bcast += 1
drain_inflight()                         # drain in-flight sends at the end of the window
if CUDA:
    torch.cuda.synchronize()
el = time.time() - t0
dist.barrier()
mem = torch.cuda.max_memory_allocated() / 2 ** 30 if CUDA else 0.0
tok = ITERS * M * MB * SEQ

# Framework self-check: rows processed by the last-stage loss_fn per iteration must equal M*MB (a wrongly shaped
# global batch makes the schedule do less work without failing). The measured value goes into rows_per_iter;
# the launcher drops the row when arms disagree.
ROWS_SEEN = ROWS_PER_ITER
if FRAMEWORK:
    _n = torch.tensor([_FW_ROWS[0]], dtype=torch.long, device=dev)
    dist.broadcast(_n, src=P - 1)
    ROWS_SEEN = int(_n.item()) // ITERS
    if ROWS_SEEN != ROWS_PER_ITER and RANK == 0:
        print(f"[ABORT] {ARM}: last stage processed {ROWS_SEEN} rows per iteration, expected {ROWS_PER_ITER}", flush=True)

# Broadcast timing: the number of broadcasts inside the window depends on its alignment with TIE_SYNC, so the measured
# broadcast time is removed and one broadcast per TIE_SYNC steps is added back (el_eff). If the window has no broadcast,
# one is measured after it. With TIE_ASYNC=0 that measurement misses the pre-broadcast alignment wait (throughput high by
# about (p-1)/(3*M*TIE_SYNC); use ITERS >= TIE_SYNC to include it). With TIE_ASYNC=1 the removed time is the CPU time of
# issue + wait; the non-overlapped part stays in the wall time.
if TIED:
    tie_drain()
    t_bcast += _tie_wait_s
    if n_bcast == 0:
        if RANK == 0:
            print(f"[warn] no readout broadcast in the timing window (ITERS={ITERS} < TIE_SYNC={TIE_SYNC}): the alignment"
                  f" wait before broadcasts is not counted, throughput high by about {(NSTAGE - 1) / (3 * M_LOCAL * TIE_SYNC):.2%}", flush=True)
        t_one = tie_broadcast()
    else:
        t_one = t_bcast / n_bcast
    el_eff = (el - t_bcast) + ITERS * t_one / TIE_SYNC
else:
    t_one = 0.0
    el_eff = el
PARALLEL = "dp" if DP_ARM else ("ppdp" if (HAND and R > 1) else "pp")
rec = dict(arm=ARM, label=ARM_LABEL or ARM, sched=SCHED_NAME, rank=RANK, p=P, v=NVIRT, L=NLAYER,
           split=SIZES, d=DMODEL, T=SEQ, V=V, MB=MB, M=M, H=HEAD_DEPTH, ce_mode=CE_MODE, iters=ITERS,
           stages=NSTAGE, replicas=R, M_local=M_LOCAL, rep=REP, stage=STAGE,
           readout=(SOLO_READOUT if ARM in ("solo", "solo_dp", "solo_kr") else None),
           tie_sync=(TIE_SYNC if TIED else None), tie_async=(bool(TIE_ASYNC) if TIED else None),
           tie_lag=(TIE_LAG_N if TIED else None), parallel=PARALLEL,
           recompute=RECOMPUTE,
           params_M=round(nparam / 1e6, 1), s_per_iter=round(el_eff / ITERS, 5),
           tok_per_s=round(tok / el_eff),                        # tied solo: broadcast amortized over TIE_SYNC
           tok_per_s_raw=round(tok / el),                        # raw wall time of the timing window
           bcast_ms=round(1000 * t_one, 2), n_bcast_in_window=n_bcast,
           wait_frac=round(_wait / el, 3),
           peak_gb=round(mem, 2), state_gb=round(STATE_GB, 2), grad_gb=round(GRAD_GB, 2),
           act_gb_est=round(max(mem - STATE_GB - GRAD_GB, 0.0), 2),
           mb_sent_per_iter=round(_nbytes / ITERS / 2 ** 20, 1),   # P2P bytes sent by this rank
           mb_p2p_analytic=round(BYTES_P2P / 2 ** 20, 2),
           mb_bcast_analytic=round(BYTES_BCAST / 2 ** 20, 2),
           mb_allreduce_analytic=round(BYTES_ALLREDUCE / 2 ** 20, 1),
           mb_per_iter_analytic=round(BYTES_ANALYTIC / 2 ** 20, 2), hops=HOPS,
           msg_mb=(None if DP_ARM else round(MSG_BYTES / 2 ** 20, 2)),
           rows_per_iter=ROWS_SEEN)
# wait_frac is not meaningful under nccl (dist.recv returns on enqueue). mb_sent_per_iter counts bytes sent by this rank
# (sum over ranks = global traffic, cf. mb_per_iter_analytic); it is 0 for framework arms, use the analytic value.
if TIED and int(E("TIE_CHECK", "0")):
    # Self-check: broadcast once more, then every rank reports a checksum of its readout copy; all must equal the last
    # stage's fc (solo_kr: last stages agree across replicas through DDP, so all must equal rank P-1).
    tie_broadcast()
    if IS_LAST:
        ref = chunk.head.fc.weight.detach().to(HDT)
    else:
        ref = chunk.head.w_sync
    cs = torch.tensor([ref.double().sum().item(), ref.double().abs().sum().item()], device=dev,
                      dtype=torch.float64)
    allcs = [torch.zeros_like(cs) for _ in range(P)]
    dist.all_gather(allcs, cs)
    if RANK == 0:
        ok = all(torch.equal(c, allcs[-1]) for c in allcs)
        print(f"[tie_check] readout copy checksums equal on all ranks: {ok}  {[round(c[1].item(), 4) for c in allcs]}",
              flush=True)
if DP_ARM and int(E("DP_CHECK", "0")):
    # Self-check: replicas read different microbatches, so parameters stay equal only if the gradient all-reduce happened.
    # Reported per module to locate an unsynchronized DDP in solo_dp.
    sums = torch.tensor([sum(q.detach().double().sum().item() for q in c.parameters())
                         for c in DP_CHUNKS], dtype=torch.float64, device=dev)
    allsums = [torch.zeros_like(sums) for _ in range(P)]
    dist.all_gather(allsums, sums)
    if RANK == 0:
        same = [all(abs(a[k].item() - allsums[0][k].item()) <= 1e-9 * max(1.0, abs(allsums[0][k].item()))
                    for a in allsums) for k in range(len(DP_CHUNKS))]
        print(f"[dp_check] parameters equal across replicas, per module: {same}", flush=True)
if HAND and R > 1 and int(E("DP_CHECK", "0")):
    # Self-check (solo_kr / bp_pd): the R replicas of each stage must have equal parameter checksums; reported per stage.
    s = torch.tensor([float(STAGE), sum(q.detach().double().sum().item() for q in chunk.parameters())],
                     dtype=torch.float64, device=dev)
    alls = [torch.zeros_like(s) for _ in range(P)]
    dist.all_gather(alls, s)
    if RANK == 0:
        by_stage = {}
        for a in alls:
            by_stage.setdefault(int(a[0].item()), []).append(a[1].item())
        same = [all(abs(x - v[0]) <= 1e-9 * max(1.0, abs(v[0])) for x in v)
                for _k, v in sorted(by_stage.items())]
        print(f"[dp_check] parameters equal across replicas, per stage (0..{NSTAGE - 1}): {same}", flush=True)
if int(E("PARAM_CHECK", "0")):
    # Self-check (regression): sum and abs-sum of this rank's trainable parameters after the timing window,
    # for bitwise comparison of two code versions or code paths.
    _s = sum(q.detach().double().sum().item() for q in PARAMS)
    _a = sum(q.detach().double().abs().sum().item() for q in PARAMS)
    print(f"[param_check] rank={RANK} sum={_s:.12e} abs={_a:.12e}", flush=True)
print("PIPEP " + json.dumps(rec, ensure_ascii=False), flush=True)
if OUT_JSON:
    with open(f"{OUT_JSON}.r{RANK}", "w") as f:
        json.dump(rec, f)
drain_inflight()
dist.destroy_process_group()
