"""pipe2.py - two-GPU pipeline-throughput A/B: solo forward streaming vs BP 1F1B. Self-contained; both arms share
Block/Head/optimizer/data/timing and differ only in the execution pattern.

  ARM=solo  rank0: emb -> blocks[:s] -> isend(h.detach()) -> local head0+AUX*CE -> backward -> (accum)
            rank1: recv(h) -> blocks[s:] -> terminal head -> CE -> backward -> (accum)
            Only forward activations cross GPUs: no gradient communication, no global barrier, no bubble.
  ARM=bp    rank0: fwd -> isend(h) -> ... -> recv(grad_h) -> backward     (1F1B, interleaved)
            rank1: recv(h) -> fwd -> loss -> backward -> isend(grad_h)
            Bubble beta = (p-1)/(m+p-1) = 1/(m+1); cross-GPU traffic is bidirectional.

Both arms accumulate over M microbatches and step once per iteration (one iteration = one global batch); no all-reduce.

Metrics: steady-state tokens/s, per-rank wait fraction, per-rank peak memory, cross-GPU bytes per iteration.
Data: synthetic Zipf, same seed and order on both ranks (rank1 takes y locally).
Options: NLAYER=24 DMODEL=1024 SEQ=1024 MB=4 M=8 VMAX=8192 HEAD_DEPTH=1 AUX=1.0 CE_MODE=compile SPLIT="" (e.g. "12,12")
         STANDALONE=0 ITERS=20 WARM=10 BACKEND=nccl (gloo = CPU smoke test) OUT_JSON="".

Launch (2 GPUs):
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29570 WORLD_SIZE=2 ARM=solo M=8 \
  bash -c 'RANK=0 CUDA_VISIBLE_DEVICES=0 python -u pipe2.py & RANK=1 CUDA_VISIBLE_DEVICES=1 python -u pipe2.py; wait'
"""
import json
import math
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

E = os.environ.get
ARM = E("ARM", "solo")
assert ARM in ("solo", "bp"), ARM
NLAYER = int(E("NLAYER", "24"))
DMODEL = int(E("DMODEL", "1024"))
NHEAD = int(E("NHEAD", str(max(1, int(E("DMODEL", "1024")) // 64))))
FFN = int(E("FFN", "0")) or 4 * DMODEL
SEQ = int(E("SEQ", "1024"))
MB = int(E("MB", "4"))                 # microbatch size (rows)
M = int(E("M", "8"))                   # microbatches per iteration (= m of 1F1B)
VMAX = int(E("VMAX", "8192"))
HEAD_DEPTH = int(E("HEAD_DEPTH", "1"))
DROP = float(E("DROP", "0.1"))
LS = float(E("LS", "0.0"))
AUX = float(E("AUX", "1.0"))
LR = float(E("LR", "3e-4"))
CLIP = float(E("CLIP", "0.5"))
CE_MODE = E("CE_MODE", "compile")
assert CE_MODE in ("autocast", "bf16", "compile"), CE_MODE
SPLIT = E("SPLIT", "")                 # "12,12"; empty = even split
STANDALONE = int(E("STANDALONE", "0"))  # 1 = no communication, each rank runs only its own stage =>
                                       # pure compute time per stage; pipeline upper bound = tok/max(t0,t1)
ITERS = int(E("ITERS", "20"))
WARM = int(E("WARM", "10"))
SEED = int(E("SEED", "42"))
OUT_JSON = E("OUT_JSON", "")

RANK = int(os.environ["RANK"])
WORLD = int(os.environ.get("WORLD_SIZE", "2"))
assert WORLD == 2, "this script supports p=2 only"
BACKEND = E("BACKEND", "nccl")         # gloo = CPU smoke test (logic runs without GPUs)
torch.manual_seed(SEED)
CUDA = BACKEND == "nccl" and torch.cuda.is_available()
dev = "cuda" if CUDA else "cpu"
if CUDA:
    torch.cuda.set_device(0)           # each process sees only one GPU (CUDA_VISIBLE_DEVICES)
dist.init_process_group(BACKEND, rank=RANK, world_size=WORLD)
V = VMAX
AC = dict(device_type=dev, dtype=torch.bfloat16, enabled=CUDA)


# ----------------------------------------------------------------- CE (same semantics as lm_solo)
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


# ----------------------------------------------------------------- model (same architecture as lm_solo.py)
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
    def __init__(self, depth=0):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(depth)])
        self.ln = nn.LayerNorm(DMODEL)
        self.fc = nn.Linear(DMODEL, V)

    def forward(self, h, m):
        for b in self.blocks:
            h = b(h, m)
        return self.fc(self.ln(h))


class Stage(nn.Module):
    """The segment held by this rank. rank0 additionally holds emb/pos; in solo, rank0 also holds the aux head."""
    def __init__(self, n_blocks, first, head_depth, has_head):
        super().__init__()
        self.first = first
        if first:
            self.emb = nn.Embedding(V, DMODEL)
            self.pos = nn.Embedding(SEQ, DMODEL)
            nn.init.normal_(self.emb.weight, std=0.02)
            nn.init.normal_(self.pos.weight, std=0.02)
            self.drop = nn.Dropout(DROP)
        self.blocks = nn.ModuleList([Block() for _ in range(n_blocks)])
        self.head = Head(head_depth) if has_head else None

    def trunk(self, x, m):
        if self.first:
            p = torch.arange(x.size(1), device=x.device)
            h = self.drop(self.emb(x) + self.pos(p)[None])
        else:
            h = x
        for b in self.blocks:
            h = b(h, m)
        return h


if SPLIT:
    sizes = [int(s) for s in SPLIT.split(",")]
    assert len(sizes) == 2 and sum(sizes) == NLAYER, (sizes, NLAYER)
else:
    sizes = [NLAYER // 2, NLAYER - NLAYER // 2]
first = RANK == 0
# solo: both stages have a head (rank0 = aux head, depth HEAD_DEPTH; rank1 = terminal head, depth 0)
# bp:   only rank1 has the terminal head
has_head = True if ARM == "solo" else (RANK == 1)
hd = HEAD_DEPTH if (ARM == "solo" and RANK == 0) else 0
stage = Stage(sizes[RANK], first, hd, has_head).to(dev)
opt = torch.optim.AdamW(stage.parameters(), lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
nparam = sum(p.numel() for p in stage.parameters())

# ----------------------------------------------------------------- data (same seed and order on both ranks)
_g = torch.Generator().manual_seed(SEED)
_r = torch.arange(1, V + 1, dtype=torch.double)
_p = (_r ** -1.0)
_p = (_p / _p.sum()).float()
_BATCHES = [(torch.multinomial(_p, MB * (SEQ + 1), replacement=True, generator=_g)
             .view(MB, SEQ + 1)) for _ in range(8)]
_BATCHES = [(b[:, :SEQ].to(dev), b[:, 1:].to(dev).reshape(-1)) for b in _BATCHES]
del _r, _p
MASK = torch.full((SEQ, SEQ), float("-inf"), device=dev).triu(1)
_HFAKE = torch.randn(MB, SEQ, DMODEL, device=dev) * 0.02   # fake input for rank1 under STANDALONE
HSHAPE = (MB, SEQ, DMODEL)
HDT = torch.bfloat16 if CUDA else torch.float32   # bf16 activations across GPUs (fp32 for the gloo smoke test)
BYTES_PER_MSG = MB * SEQ * DMODEL * HDT.itemsize

# ----------------------------------------------------------------- timing
_wait = 0.0


def _recv(buf, src):
    global _wait
    t = time.time()
    dist.recv(buf, src=src)
    _wait += time.time() - t


def one_iter():
    """One iteration = M microbatches; both arms step once at the end. Returns cross-GPU bytes.

    All sends are isend with a deferred wait: a blocking send would deadlock with the peer's send
    (r0 waits for isend(h_{i+1}) while r1 is stuck on isend(g_i) waiting for r0 to post recv).
    An in-flight cap of 2 avoids the deadlock and lets both stages run ahead (same for both arms).
    """
    global _wait
    opt.zero_grad(set_to_none=True)
    nbytes = 0
    inflight = []                      # [(handle, tensor)] -- hold the tensor so it is not freed

    if STANDALONE:                     # pure local-stage compute, no dist calls at all
        for i in range(M):
            x, y = _BATCHES[i % len(_BATCHES)]
            inp = x if first else _HFAKE
            with torch.autocast(**AC):
                h = stage.trunk(inp, MASK)
                if stage.head is not None:
                    z = stage.head(h, MASK).reshape(-1, V)
                    lk = (AUX * ce_loss(z, y, none=True).float().mean()) if (ARM == "solo" and first) \
                        else ce_loss(z, y, ls=LS)
                else:
                    lk = h.float().sum() * 0.0
            (lk / M).backward()
        torch.nn.utils.clip_grad_norm_(stage.parameters(), CLIP)
        opt.step()
        return 0

    def post(t, dst, cap=2):
        nonlocal nbytes
        inflight.append((dist.isend(t, dst=dst), t))
        nbytes += BYTES_PER_MSG
        while len(inflight) > cap:
            inflight.pop(0)[0].wait()

    if ARM == "solo":
        if RANK == 0:
            for i in range(M):
                x, y = _BATCHES[i % len(_BATCHES)]
                with torch.autocast(**AC):
                    h = stage.trunk(x, MASK)
                post(h.detach().to(HDT).contiguous(), 1)     # send first so downstream can start immediately
                with torch.autocast(**AC):
                    z = stage.head(h, MASK).reshape(-1, V)
                    lk = AUX * ce_loss(z, y, none=True).float().mean()
                (lk / M).backward()
        else:
            buf = torch.empty(HSHAPE, device=dev, dtype=HDT)
            for i in range(M):
                _, y = _BATCHES[i % len(_BATCHES)]
                _recv(buf, 0)
                h_in = buf.float()                            # solo: no gradient is sent back
                with torch.autocast(**AC):
                    h = stage.trunk(h_in, MASK)
                    z = stage.head(h, MASK).reshape(-1, V)
                    lk = ce_loss(z, y, ls=LS)
                (lk / M).backward()
    else:  # bp 1F1B
        if RANK == 0:
            pend = []                                          # forward graphs already sent, waiting for the returned gradient
            gbuf = torch.empty(HSHAPE, device=dev, dtype=HDT)
            for i in range(M):
                x, _ = _BATCHES[i % len(_BATCHES)]
                with torch.autocast(**AC):
                    h = stage.trunk(x, MASK)
                post(h.to(HDT).contiguous(), 1)
                pend.append(h)
                if i >= 1:                                     # 1F1B: one backward received per forward sent
                    _recv(gbuf, 1)
                    nbytes += BYTES_PER_MSG
                    pend.pop(0).backward(gbuf.float() / M)
            while pend:                                        # cooldown
                _recv(gbuf, 1)
                nbytes += BYTES_PER_MSG
                pend.pop(0).backward(gbuf.float() / M)
        else:
            buf = torch.empty(HSHAPE, device=dev, dtype=HDT)
            for i in range(M):
                _, y = _BATCHES[i % len(_BATCHES)]
                _recv(buf, 0)
                h_in = buf.float().requires_grad_(True)
                with torch.autocast(**AC):
                    h = stage.trunk(h_in, MASK)
                    z = stage.head(h, MASK).reshape(-1, V)
                    lk = ce_loss(z, y, ls=LS)
                (lk / M).backward()
                post(h_in.grad.to(HDT).contiguous(), 0, cap=1)
    for hnd, _t in inflight:
        hnd.wait()
    torch.nn.utils.clip_grad_norm_(stage.parameters(), CLIP)
    opt.step()
    return nbytes


if RANK == 0:
    print(f"[cfg] ARM={ARM} L={NLAYER}({sizes}) d={DMODEL} T={SEQ} V={V} MB={MB} M={M} "
          f"H={HEAD_DEPTH} CE_MODE={CE_MODE} ITERS={ITERS} WARM={WARM} STANDALONE={STANDALONE}",
          flush=True)
for _ in range(WARM):
    one_iter()
if CUDA:
    torch.cuda.synchronize()
dist.barrier()
_wait = 0.0
if CUDA:
    torch.cuda.reset_peak_memory_stats()
t0 = time.time()
nb = 0
for _ in range(ITERS):
    nb += one_iter()
if CUDA:
    torch.cuda.synchronize()
el = time.time() - t0
dist.barrier()
tok = ITERS * M * MB * SEQ
mem = torch.cuda.max_memory_allocated() / 2 ** 30 if CUDA else 0.0
rec = dict(arm=ARM, rank=RANK, L=NLAYER, split=sizes, d=DMODEL, T=SEQ, V=V, MB=MB, M=M,
           H=HEAD_DEPTH, ce_mode=CE_MODE, iters=ITERS, params_M=round(nparam / 1e6, 1),
           s_per_iter=round(el / ITERS, 5), tok_per_s=round(tok / el), wait_frac=round(_wait / el, 3),
           peak_gb=round(mem, 2), mb_per_iter=round(nb / ITERS / 2 ** 20, 1),
           standalone=STANDALONE)
# Under nccl, dist.recv returns on enqueue, so wait_frac is only meaningful under gloo; use STANDALONE=1 for stage imbalance.
print("PIPE2 " + json.dumps(rec, ensure_ascii=False), flush=True)
if OUT_JSON:
    with open(f"{OUT_JSON}.r{RANK}", "w") as f:
        json.dump(rec, f)
dist.destroy_process_group()
