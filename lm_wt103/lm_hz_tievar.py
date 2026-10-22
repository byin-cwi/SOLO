"""lm_hz_tievar.py - SOLO module-parallel pipeline with multi-horizon milestones and aux-readout variants (TIE_MODE).

Pipelined execution of the lm_solo.py recipe MODE=mfb, BETA=1, PRIOR=unigram, TIE_AUX=1: same math, one module per
process/GPU. Between GPUs there is only a one-way forward activation stream: no gradient communication and no global
optimizer barrier.

Topology (K = WORLD_SIZE GPUs):
  rank 0:   data -> emb+pos -> group 0 -> head 0 -> isend(h, y) -> local milestone loss + step -> next batch
  rank k:   irecv -> group k -> head k -> isend -> local loss + step
  rank K-1: irecv -> group K-1 -> terminal CE + step; prints the epoch line (per-exit ppl via all_gather)
Each rank backpropagates and steps right after sending, so a module consumes activations produced by upstream
parameters from one step earlier. With TIE_AUX=1 the terminal fc is broadcast every TIE_SYNC steps and the
intermediate heads read it as a buffer without grad (detach semantics). No ensemble ppl is computed.

Multi-horizon milestones: intermediate module k sets a milestone for each offset D = 1..H_k (H_k from HZ_H).
  * D=1 is the single-horizon milestone; HZ_H all 1 gives exactly the single-horizon baseline.
  * D>=2: a learned offset embedding e_D is added before the head blocks, the readout is computed on a random
    HZ_RHO fraction of positions, and the loss is weighted HZ_GAMMA^(D-1) and added on top (not normalized).
  * The ratchet is indexed per (D, bucket); a D>=2 bucket updates only with >= HZ_NMIN samples.
  * HZ_CE=1: control arm with plain CE (no milestone) for D>=2.

Aux-readout variants TIE_MODE (default trained; all keep the zero cross-module gradient):
  trained      : heads read the broadcast copy of the terminal fc (refreshed every TIE_SYNC steps).
  random       : one frozen N(0,1) readout per head (seed includes RANK); the broadcast carries only std(W_terminal),
                 used to rescale it.
  affine       : trained + an identity-initialized learnable DMODEL->DMODEL adapter per head before the readout.
  untie_affine : private readout (forces TIE_AUX=0) + the same adapter.
  stale        : frozen terminal fc/bias loaded from STALE_FC=<{OUT}.r{K-1}>, never refreshed.

Env (defaults): TOKENIZER=word, DSET=wikitext-2-raw-v1, VMAX=32768, NLAYER=8, DMODEL=384, NHEAD=6,
  FFN=0 (0 = 4*DMODEL), SEQ=256, DROP=0.1, BS=32, LR=3e-4, WARMUP=1000, CLIP=0.5, WD=0.0, LS=0.0, EPOCHS=10,
  STEPS=0 (>0 caps steps/epoch), SEED=42, DATA=./data/wikitext, OUT= (saves <OUT>.r<RANK>), LOAD= (reads <LOAD>.r<RANK>),
  BETA=1.0, BUCKET_BY=ellpi, KAPPA=0.9, CFLOOR=0.9, NBUCKET=12, LAM0=0.1, MU=0.3, AUX=0.1, TK=(k+1)/K, HEAD_DEPTH=0,
  SPLIT= (layers per module), TIE_AUX=1, TIE_MODE=trained, STALE_FC=, TIE_SYNC=100, HZ_H=4,3,2, HZ_GAMMA=0.5, HZ_RHO=0.5,
  HZ_CE=0, HZ_NMIN=64, LOG_EVERY=100 (heartbeat steps), COMPILE=0 (1 = torch.compile trunk/heads; same math and ckpt
  format), NCCL_TIMEOUT_H=0.5, BACKEND=nccl|gloo, FAKE_DATA=0 (random tokens, VOCAB=28996).
  Requires RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT (LOCAL_RANK optional).
Data: run dl_wikitext.py once. TOKENIZER=word reads DATA/word_cache_<DSET>_<VMAX>.pt (build it once with
  BUILD_WORD_CACHE=1, which exits after building); TOKENIZER=bert needs DATA/bert_cache_<DSET>.pt (built by lm_solo.py).
Limitations: fixed recipe (mfb, BETA=1, unigram prior); no ACCUM; the variable-length tail window is dropped
  (fixed-length send/recv).

CPU two-process smoke test (gloo, fake data, ~1 minute):
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29650 WORLD_SIZE=2 BACKEND=gloo FAKE_DATA=1 \
  STEPS=3 EPOCHS=1 NLAYER=4 DMODEL=64 NHEAD=2 FFN=128 SEQ=32 BS=4 HEAD_DEPTH=1 \
  bash -c 'RANK=0 python -u lm_hz_tievar.py & RANK=1 python -u lm_hz_tievar.py; wait'

Full training (K=2, two GPUs):
  cd <dir> && export CUDA_VISIBLE_DEVICES=0,1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29651 \
    WORLD_SIZE=2 DATA=./data/wikitext TOKENIZER=bert DSET=wikitext-103-raw-v1 \
    NLAYER=16 DMODEL=410 FFN=2100 NHEAD=10 SEQ=256 LR=2.5e-4 WARMUP=1000 CLIP=0.25 \
    WD=0.0 LS=0.0 DROP=0.1 EPOCHS=10 SEED=42 KAPPA=0.7 AUX=0.3 HEAD_DEPTH=2 BS=64 \
    OUT=ckpt/pipe_k2.pt && \
  { RANK=0 LOCAL_RANK=0 nohup python -u lm_hz_tievar.py > logs/pipe_k2_r0.log 2>&1 & \
    RANK=1 LOCAL_RANK=1 nohup python -u lm_hz_tievar.py > logs/pipe_k2_r1.log 2>&1 & }
"""
import math
import os
import time

import torch
import torch.distributed as dist
from datetime import timedelta
import torch.nn as nn
import torch.nn.functional as F


def E(k, d):
    return os.environ.get(k, d)


# ---------------------------------------------------------------- config
TOKENIZER = E("TOKENIZER", "word")
DSET = E("DSET", "wikitext-2-raw-v1")
NLAYER = int(E("NLAYER", "8"))
DMODEL = int(E("DMODEL", "384"))
NHEAD = int(E("NHEAD", "6"))
FFN = int(E("FFN", "0"))
SEQ = int(E("SEQ", "256"))
DROP = float(E("DROP", "0.1"))
BS = int(E("BS", "32"))
LR = float(E("LR", "3e-4"))
WARMUP = int(E("WARMUP", "1000"))
CLIP = float(E("CLIP", "0.5"))
WD = float(E("WD", "0.0"))
LS = float(E("LS", "0.0"))
EPOCHS = int(E("EPOCHS", "10"))
SEED = int(E("SEED", "42"))
DATA = E("DATA", "./data/wikitext")
OUT = E("OUT", "")
LOAD = E("LOAD", "")
STEPS = int(E("STEPS", "0"))
CH = 2048
BETA = float(E("BETA", "1.0"))
BUCKET_BY = E("BUCKET_BY", "ellpi")
KAPPA = float(E("KAPPA", "0.9"))
CFLOOR = float(E("CFLOOR", "0.9"))
NBUCKET = int(E("NBUCKET", "12"))
LAM0 = float(E("LAM0", "0.1"))
MU = float(E("MU", "0.3"))
AUX = float(E("AUX", "0.1"))
HEAD_DEPTH = int(E("HEAD_DEPTH", "0"))
SPLIT = E("SPLIT", "")
TIE_AUX = int(E("TIE_AUX", "1"))
TIE_MODE = E("TIE_MODE", "trained")   # trained | random | affine | untie_affine | stale
assert TIE_MODE in ("trained", "random", "affine", "untie_affine", "stale"), TIE_MODE
if TIE_MODE == "untie_affine":
    TIE_AUX = 0                       # private readout + adapter
STALE_FC = E("STALE_FC", "")
TIE_SYNC = int(E("TIE_SYNC", "100"))
HZ_H = [int(v) for v in E("HZ_H", "4,3,2").split(",")]   # horizon depth of each intermediate module
HZ_GAMMA = float(E("HZ_GAMMA", "0.5"))
HZ_RHO = float(E("HZ_RHO", "0.5"))
HZ_CE = int(E("HZ_CE", "0"))          # 1 = plain CE for D>=2 (control)
HZ_NMIN = int(E("HZ_NMIN", "64"))     # min samples for a D>=2 ratchet bucket update
FAKE = E("FAKE_DATA", "0") == "1"

RANK = int(os.environ["RANK"])            # explicit RANK required (manual or torchrun)
WORLD = int(os.environ["WORLD_SIZE"])
K = WORLD
nccl_ok = torch.cuda.is_available() and dist.is_nccl_available()
BACKEND = E("BACKEND", "nccl" if nccl_ok else "gloo")
use_cuda = BACKEND == "nccl" and torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", RANK)))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
dev = "cuda" if use_cuda else "cpu"
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")  # raise NCCL errors instead of hanging
_TO_H = float(E("NCCL_TIMEOUT_H", "0.5"))   # hours
dist.init_process_group(BACKEND, timeout=timedelta(hours=_TO_H))
torch.manual_seed(SEED + RANK)            # independent RNG stream per rank (dropout); rank 0 drives the data order

# ---------------------------------------------------------------- data (all ranks read the same)
if FAKE:
    V = int(E("VOCAB", "28996"))
    g = torch.Generator().manual_seed(SEED)
    data = {"train": torch.randint(0, V, (200_000,), generator=g),
            "validation": torch.randint(0, V, (20_000,), generator=g),
            "test": torch.randint(0, V, (20_000,), generator=g)}
    n_words = {}
else:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from datasets import load_dataset  # noqa: E402

    ds = load_dataset("Salesforce/wikitext", DSET, cache_dir=DATA)
    if TOKENIZER == "bert":
        cache_f = os.path.join(DATA, f"bert_cache_{DSET}.pt")
        assert os.path.exists(cache_f), \
            "build the bert token cache with lm_solo.py (TOKENIZER=bert) before running the pipeline"
        blob = torch.load(cache_f)
        data, n_words = blob["data"], blob["n_words"]
        V = 28996
    else:
        VMAX = int(E("VMAX", "32768"))
        _cf = os.path.join(DATA, f"word_cache_{DSET}_{VMAX}.pt")
        if os.path.exists(_cf):
            _blob = torch.load(_cf)
            data, V, n_words = _blob["data"], _blob["V"], _blob.get("n_words", {})
            if RANK == 0:
                print(f"[data] word cache hit {_cf}", flush=True)
        else:
            assert E("BUILD_WORD_CACHE", "0") == "1", (
                f"missing word-level token cache {_cf} => build it first by running this script with BUILD_WORD_CACHE=1."
                " WT-103 has ~103M words; tokenizing it once per rank x arm (4 x 4) is pure waste,"
                " and concurrent writes to the same cache from multiple processes race.")
            from collections import Counter
            print("[data] building word cache (WT-103 takes a few minutes)...", flush=True)
            cnt = Counter()
            for t in ds["train"]["text"]:
                cnt.update(t.split())
            vocab = {w: i + 2 for i, (w, _) in enumerate(cnt.most_common(VMAX - 2))}
            V = len(vocab) + 2

            def enc(split):
                ids = []
                for t in ds[split]["text"]:
                    ids += [vocab.get(w, 1) for w in t.split()]
                return torch.tensor(ids, dtype=torch.long)

            data = {s: enc(s) for s in ("train", "validation", "test")}
            n_words = {}
            torch.save(dict(data=data, V=V, n_words=n_words), _cf)
            print(f"[data] saved {_cf}  V={V} "
                  f"train={data['train'].numel():,}", flush=True)
            raise SystemExit(0)                    # prep mode: exit after building, no training

_cnt = torch.bincount(data["train"], minlength=V).double()
_p = 0.95 * (_cnt / _cnt.sum()) + 0.05 / V
LOG_UNI = _p.log().float().to(dev)
H_UNI = float(-(_p * _p.log()).sum())
del _cnt, _p

# module partition
if SPLIT:
    sizes = [int(s) for s in SPLIT.split(",")]
    assert sum(sizes) == NLAYER and len(sizes) == K
else:
    per = NLAYER // K
    sizes = [per] * (K - 1) + [NLAYER - per * (K - 1)]
bounds = [0]
for s in sizes:
    bounds.append(bounds[-1] + s)
MY_BLOCKS = list(range(bounds[RANK], bounds[RANK + 1]))
TK = ([float(v) for v in E("TK", "").split(",") if v]
      or [(k + 1) / K for k in range(K - 1)])
EDGES = torch.linspace(0, math.log(V) if BUCKET_BY == "h" else 14.0, NBUCKET + 1, device=dev)


# ---------------------------------------------------------------- model pieces
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(DMODEL)
        self.attn = nn.MultiheadAttention(DMODEL, NHEAD, dropout=DROP, batch_first=True)
        self.ln2 = nn.LayerNorm(DMODEL)
        fw = FFN if FFN else 4 * DMODEL
        self.mlp = nn.Sequential(nn.Linear(DMODEL, fw), nn.GELU(),
                                 nn.Linear(fw, DMODEL), nn.Dropout(DROP))

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class Stage(nn.Module):
    """The part held by this GPU: (rank0: embedding) + this group's blocks + (intermediate: deep head / tied readout | last: terminal head)."""
    def __init__(self):
        super().__init__()
        if RANK == 0:
            self.emb = nn.Embedding(V, DMODEL)
            self.pos = nn.Embedding(SEQ, DMODEL)
            nn.init.normal_(self.emb.weight, std=0.02)
            nn.init.normal_(self.pos.weight, std=0.02)
            self.drop = nn.Dropout(DROP)
        self.blocks = nn.ModuleList([Block() for _ in MY_BLOCKS])
        if RANK < K - 1:
            self.hblocks = nn.ModuleList([Block() for _ in range(HEAD_DEPTH)])
            self.hln = nn.LayerNorm(DMODEL)
            myh = HZ_H[RANK] if RANK < len(HZ_H) else 1
            if myh > 1:                                   # offset embeddings for D=2..H (added before the head blocks)
                self.off_emb = nn.Parameter(torch.randn(myh - 1, DMODEL) * 0.02)
            if TIE_AUX:                                   # read-only copy (buffer without grad = detach)
                self.register_buffer("w_t", torch.zeros(V, DMODEL))
                self.register_buffer("b_t", torch.zeros(V))
            else:
                self.hfc = nn.Linear(DMODEL, V)
            if TIE_MODE in ("affine", "untie_affine"):    # identity-initialized learnable adapter
                self.adapter = nn.Linear(DMODEL, DMODEL, bias=True)
                nn.init.eye_(self.adapter.weight)
                nn.init.zeros_(self.adapter.bias)
            if TIE_MODE == "random":                      # frozen random readout (unit variance), rescaled by the broadcast std
                g = torch.Generator().manual_seed(SEED * 1000 + RANK)
                self.register_buffer("w_rand", torch.randn(V, DMODEL, generator=g))
                self.w_t.copy_(self.w_rand * 0.02)
            if TIE_MODE == "stale":                       # frozen trained terminal readout, never refreshed
                assert STALE_FC, "TIE_MODE=stale requires STALE_FC=<{OUT}.r{K-1} path>"
                _sd = torch.load(STALE_FC, map_location="cpu")
                _sd = _sd.get("stage", _sd) if isinstance(_sd, dict) else _sd
                _w = next((v for k, v in _sd.items() if k.endswith("fc.weight")
                           and tuple(v.shape) == (V, DMODEL)), None)
                _b = next((v for k, v in _sd.items() if k.endswith("fc.bias")
                           and tuple(v.shape) == (V,)), None)
                assert _w is not None and _b is not None, \
                    f"STALE_FC has no ({V},{DMODEL}) fc.weight/bias; sample keys {list(_sd)[:6]}"
                self.w_t.copy_(_w.float()); self.b_t.copy_(_b.float())
                print(f"[rank {RANK}] stale readout loaded from {STALE_FC} "
                      f"(std {float(_w.float().std()):.4f})", flush=True)
        else:
            self.ln = nn.LayerNorm(DMODEL)
            self.fc = nn.Linear(DMODEL, V)

    def trunk(self, h, mask):
        for b in self.blocks:
            h = b(h, mask)
        return h

    def head_feats(self, h, mask, d_idx):
        """Head blocks -> LN features for offset index d_idx (0: D=1 as is; >=1: adds e_D first)."""
        if d_idx > 0:
            h = h + self.off_emb[d_idx - 1][None, None, :]
        for b in self.hblocks:
            h = b(h, mask)
        return self.hln(h)

    def aux_readout(self, f):
        if TIE_MODE in ("affine", "untie_affine"):
            f = self.adapter(f)
        if TIE_AUX:
            return F.linear(f, self.w_t, self.b_t)
        return self.hfc(f)

    def head_out(self, h, mask):
        if RANK < K - 1:
            f = self.head_feats(h, mask, 0)
            return self.aux_readout(f)
        return self.fc(self.ln(h))


def floored_log(y_flat, lam_vec):
    n = y_flat.size(0)
    off = torch.log((1 - torch.exp(-lam_vec)) / (V - 1)).unsqueeze(1).expand(n, V).clone()
    off.scatter_(1, y_flat[:, None], -lam_vec[:, None])
    return off


MYH = HZ_H[RANK] if RANK < len(HZ_H) else 1               # horizon depth of this module
lam_tab = torch.full((max(MYH, 1), NBUCKET), -1.0, device=dev)  # (D, bucket) ratchet
ach_tab = torch.full((max(MYH, 1), NBUCKET), -1.0, device=dev)
lam_row, ach_row = lam_tab[0], ach_tab[0]                 # D=1 rows, used by module_loss


def module_loss(zk, y_flat):
    """Unigram-only version of lm_solo.py module_loss; this rank is module RANK (RANK<K-1)."""
    n = y_flat.numel()
    t = TK[RANK]
    acc = torch.zeros((), device=dev)
    _agg = None
    for lo in range(0, n, CH):
        sl = slice(lo, min(lo + CH, n))
        yc = y_flat[sl]
        with torch.no_grad():
            log_pi = LOG_UNI[None, :].expand(yc.size(0), V)
            Hpi = torch.full((yc.size(0),), H_UNI, device=dev)
            ell_pi = -LOG_UNI[yc]
            key = Hpi if BUCKET_BY == "h" else ell_pi
            bucket = torch.bucketize(key, EDGES[1:-1])
            hiq = key >= torch.quantile(key, 0.75)
            ach = -F.log_softmax(zk[sl].detach().float(), -1) \
                .gather(1, yc[:, None]).squeeze(1)
            for b in torch.unique(bucket):
                mb = bucket == b
                a = ach[mb].mean()
                if ach_row[b] < 0:
                    ach_row[b] = a
                ach_row[b] = 0.99 * ach_row[b] + 0.01 * a
                init = torch.clamp(CFLOOR * key[mb].mean(), min=LAM0)
                cand = torch.clamp(KAPPA * ach_row[b], min=LAM0)
                lam_row[b] = torch.min(init, cand) if lam_row[b] < 0 \
                    else torch.min(lam_row[b], cand)
            lam_vec = lam_row[bucket]
            log_self = F.log_softmax(zk[sl].detach().float(), -1)
            log_anchor = log_self if BETA >= 1.0 else \
                ((1 - BETA) * log_pi + BETA * log_self)
            log_anchor = log_anchor - log_anchor.logsumexp(-1, keepdim=True)
            ell_anchor = ell_pi
            log_delta = floored_log(yc, lam_vec)
            log_m = F.log_softmax((1 - t) * log_anchor + t * log_delta, -1)
            ell_m = -log_m.gather(1, yc[:, None]).squeeze(1)
            promise = (1 - t) * ell_anchor + t * lam_vec
        logq = F.log_softmax(zk[sl].float(), -1)
        ell_q = -logq.gather(1, yc[:, None]).squeeze(1)
        active = (ell_q > ell_m).float().detach()
        hinge = (ell_q - ell_m).clamp_min(0).sum()
        hell = (1 - (0.5 * (logq + log_m)).exp().sum(-1)).clamp_min(0)
        acc = acc + hinge + MU * (hell * active).sum() + AUX * ell_q.sum()
        with torch.no_grad():          # accumulate on GPU: one host sync per step
            _s = torch.stack([ell_q.sum(), promise.sum(), ell_q[hiq].sum(),
                              promise[hiq].sum(), active.sum()])
            _agg = _s if _agg is None else _agg + _s
    tn, td, hn, hd, ac = (_agg.tolist() if _agg is not None else [0.0] * 5)
    return acc / n, dict(tight_num=tn / n, tight_den=td / n,
                         hi_num=hn, hi_den=max(hd, 1e-9), act=ac / n)


# ---------------------------------------------------------------- setup
stage = Stage().to(dev)
if int(E("COMPILE", "0")):                   # compile trunk/heads (math unchanged, ckpt format unchanged)
    _t_eager, _h_eager = stage.trunk, stage.head_out
    try:
        stage.trunk = torch.compile(stage.trunk, dynamic=False)
        stage.head_out = torch.compile(stage.head_out, dynamic=False)
        print(f"[rank {RANK}] COMPILE=1 enabled", flush=True)
    except Exception as _e:                  # noqa: BLE001
        stage.trunk, stage.head_out = _t_eager, _h_eager
        print(f"[rank {RANK}] COMPILE failed -> eager: {type(_e).__name__}", flush=True)
n_par = sum(p.numel() for p in stage.parameters()) / 1e6
print(f"[rank {RANK}] blocks={MY_BLOCKS} params={n_par:.1f}M "
      f"({'terminal' if RANK == K - 1 else 'milestone'})", flush=True)
if LOAD and os.path.exists(f"{LOAD}.r{RANK}"):
    stage.load_state_dict(torch.load(f"{LOAD}.r{RANK}", map_location=dev)["stage"])
    print(f"[rank {RANK}] loaded {LOAD}.r{RANK}", flush=True)

opt = torch.optim.AdamW(stage.parameters(), lr=LR, weight_decay=WD,
                        betas=(0.9, 0.95), fused=use_cuda)
L_train = data["train"].numel() // BS
starts_all = [i for i in range(0, L_train - 2, SEQ) if L_train - 1 - i >= SEQ]
n_steps = min(len(starts_all), STEPS) if STEPS else len(starts_all)
total_steps = max(1, EPOCHS * n_steps)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min((s + 1) / WARMUP, 0.5 * (1 + math.cos(math.pi * s / total_steps))))
WIRE_DT = torch.bfloat16 if use_cuda else torch.float32
mask = torch.full((SEQ, SEQ), float("-inf"), device=dev).triu(1)


HZ_STAT = [0.0] * max(MYH - 1, 1)
HZ_CNT = 0


def module_loss_h(zk, yc_all, d):
    """D>=2 milestones on already-subsampled positions: same structure as module_loss, (d, bucket) ratchet,
    a bucket updates only with >= HZ_NMIN samples."""
    t = TK[RANK]
    n = yc_all.numel()
    acc = torch.zeros((), device=dev)
    a_sum = 0.0
    for lo in range(0, n, CH):
        sl = slice(lo, min(lo + CH, n))
        yc = yc_all[sl]
        with torch.no_grad():
            ell_pi = -LOG_UNI[yc]
            bucket = torch.bucketize(ell_pi, EDGES[1:-1])
            ach = -F.log_softmax(zk[sl].detach().float(), -1) \
                .gather(1, yc[:, None]).squeeze(1)
            for b in torch.unique(bucket):
                mb = bucket == b
                if int(mb.sum()) < HZ_NMIN:
                    continue
                a = ach[mb].mean()
                if ach_tab[d, b] < 0:
                    ach_tab[d, b] = a
                ach_tab[d, b] = 0.99 * ach_tab[d, b] + 0.01 * a
                init = torch.clamp(CFLOOR * ell_pi[mb].mean(), min=LAM0)
                cand = torch.clamp(KAPPA * ach_tab[d, b], min=LAM0)
                lam_tab[d, b] = torch.min(init, cand) if lam_tab[d, b] < 0 \
                    else torch.min(lam_tab[d, b], cand)
            lam_vec = torch.where(lam_tab[d][bucket] > 0, lam_tab[d][bucket],
                                  torch.clamp(CFLOOR * ell_pi, min=LAM0))
            log_self = F.log_softmax(zk[sl].detach().float(), -1)
            off = torch.log((1 - torch.exp(-lam_vec)) / (V - 1)) \
                .unsqueeze(1).expand(yc.size(0), V).clone()
            off.scatter_(1, yc[:, None], -lam_vec[:, None])
            log_m = F.log_softmax((1 - t) * log_self + t * off, -1)
            ell_m = -log_m.gather(1, yc[:, None]).squeeze(1)
        logq = F.log_softmax(zk[sl].float(), -1)
        ell_q = -logq.gather(1, yc[:, None]).squeeze(1)
        active = (ell_q > ell_m).float().detach()
        hinge = (ell_q - ell_m).clamp_min(0).sum()
        hell = (1 - (0.5 * (logq + log_m)).exp().sum(-1)).clamp_min(0)
        acc = acc + hinge + MU * (hell * active).sum() + AUX * ell_q.sum()
        a_sum += float(active.sum())
    return acc / n, a_sum / n


def hz_extra(h, y, loss):
    """Add the D>=2 horizon terms on top of the D=1 loss."""
    global HZ_CNT
    if RANK >= K - 1 or MYH <= 1:
        return loss
    for d in range(1, MYH):
        D = d + 1
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_cuda):
            fD = stage.head_feats(h, mask, d)
            TT = fD.size(1) - (D - 1)
            ff = fD[:, :TT].reshape(-1, DMODEL)
        yD = y[:, D - 1:].reshape(-1)
        nsub = max(min(CH, ff.size(0)), int(HZ_RHO * ff.size(0)))
        idx = torch.randperm(ff.size(0), device=dev)[:nsub]
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_cuda):
            zD = stage.aux_readout(ff[idx])
        if HZ_CE:
            lD = F.cross_entropy(zD.float(), yD[idx])
            aD = -1.0
        else:
            lD, aD = module_loss_h(zD, yD[idx], d)
        loss = loss + (HZ_GAMMA ** d) * lD
        HZ_STAT[d - 1] += aD
    HZ_CNT += 1
    return loss


def tie_broadcast():
    if not TIE_AUX or K == 1 or TIE_MODE == "stale":
        return
    if TIE_MODE == "random":                          # send only std(W_terminal), not the content
        s = (stage.fc.weight.data.float().std().reshape(1) if RANK == K - 1
             else torch.zeros(1, device=dev))
        dist.broadcast(s, K - 1)
        if RANK < K - 1:
            stage.w_t.copy_(stage.w_rand * s.to(stage.w_t.dtype))
        return
    w = stage.fc.weight.data.contiguous() if RANK == K - 1 else stage.w_t
    b = stage.fc.bias.data.contiguous() if RANK == K - 1 else stage.b_t
    dist.broadcast(w, K - 1)
    dist.broadcast(b, K - 1)


def val_starts(split):
    Lv = data[split].numel() // BS
    return [i for i in range(0, Lv - 2, SEQ) if Lv - 1 - i >= SEQ]


@torch.no_grad()
def pipe_eval(split):
    """Pipeline evaluation: each rank accumulates the NLL of its own exit locally, gathered via all_gather; no ens."""
    stage.eval()
    mat = batch_mat(split)
    h_buf = torch.empty(BS, SEQ, DMODEL, dtype=WIRE_DT, device=dev)
    y_buf = torch.empty(BS, SEQ, dtype=torch.long, device=dev)
    nll, ntok = torch.zeros((), device=dev, dtype=torch.float64), 0
    for i in val_starts(split):
        if RANK == 0:
            x = mat[:, i:i + SEQ].to(dev)
            y = mat[:, i + 1:i + 1 + SEQ].to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_cuda):
                h = stage.drop(stage.emb(x) + stage.pos(torch.arange(SEQ, device=dev))[None])
                h = stage.trunk(h, mask)
                z = stage.head_out(h, mask)
            if K > 1:
                dist.send(h.to(WIRE_DT).contiguous(), RANK + 1)
                dist.send(y.contiguous(), RANK + 1)
        else:
            dist.recv(h_buf, RANK - 1)
            dist.recv(y_buf, RANK - 1)
            y = y_buf
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_cuda):
                h = stage.trunk(h_buf, mask)
                z = stage.head_out(h, mask)
            if RANK < K - 1:
                dist.send(h.to(WIRE_DT).contiguous(), RANK + 1)
                dist.send(y.contiguous(), RANK + 1)
        y_flat = y.reshape(-1)
        zf = z.reshape(-1, V)
        for lo in range(0, y_flat.numel(), CH):
            sl = slice(lo, min(lo + CH, y_flat.numel()))
            lp = F.log_softmax(zf[sl].float(), -1)
            nll += -lp.gather(1, y_flat[sl][:, None]).sum().double()
        ntok += y_flat.numel()
    stage.train()
    packs = [None] * K
    dist.all_gather_object(packs, dict(nll=float(nll), n=ntok))
    ppls = [math.exp(p["nll"] / max(p["n"], 1)) for p in packs]
    wppl = None
    if TOKENIZER == "bert" and split in n_words and RANK == K - 1:
        cov = ntok / max(len(data[split]) - 1, 1)
        wppl = math.exp(nll / (n_words[split] * cov))
    return ppls, wppl


def batch_mat(split):
    return batchify_ids(data[split])


def batchify_ids(ids):
    n = len(ids) // BS
    return ids[: n * BS].view(BS, n)


# ---------------------------------------------------------------- train
if RANK == 0:
    print(f"pipeline K={K} split={sizes} steps/ep={n_steps} vocab={V} "
          f"tie_aux={TIE_AUX} tie_mode={TIE_MODE} tie_sync={TIE_SYNC} backend={BACKEND}", flush=True)
tie_broadcast()

h_buf = torch.empty(BS, SEQ, DMODEL, dtype=WIRE_DT, device=dev)
y_buf = torch.empty(BS, SEQ, dtype=torch.long, device=dev)
best, best_exits = float("inf"), None
LOG_EVERY = int(E("LOG_EVERY", "100"))     # step-level heartbeat interval (terminal rank)
_EMA = [None]
gstep = 0
for ep in range(1, EPOCHS + 1):
    t0 = time.time()
    agg = dict(tight_num=0.0, tight_den=0.0, act=0.0, cnt=0)
    if RANK == 0:
        mat = batch_mat("train")
        order = [starts_all[j] for j in torch.randperm(len(starts_all)).tolist()][:n_steps]
    for s in range(n_steps):
        send_reqs = []
        if RANK == 0:
            i = order[s]
            x = mat[:, i:i + SEQ].to(dev)
            y = mat[:, i + 1:i + 1 + SEQ].to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_cuda):
                h = stage.drop(stage.emb(x) + stage.pos(torch.arange(SEQ, device=dev))[None])
                h = stage.trunk(h, mask)
                z = stage.head_out(h, mask)
            if K > 1:                                     # send first, then learn: transfer overlaps the local backward
                hs = h.detach().to(WIRE_DT).contiguous()
                ys = y.contiguous()
                send_reqs = [dist.isend(hs, RANK + 1), dist.isend(ys, RANK + 1)]
            loss, dg = module_loss(z.reshape(-1, V), y.reshape(-1))
            loss = hz_extra(h, y, loss)
        else:
            dist.recv(h_buf, RANK - 1)
            dist.recv(y_buf, RANK - 1)
            y = y_buf.clone()
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_cuda):
                h = stage.trunk(h_buf, mask)
                z = stage.head_out(h, mask)
            if RANK < K - 1:
                hs = h.detach().to(WIRE_DT).contiguous()
                ys = y.contiguous()
                send_reqs = [dist.isend(hs, RANK + 1), dist.isend(ys, RANK + 1)]
                loss, dg = module_loss(z.reshape(-1, V), y.reshape(-1))
                loss = hz_extra(h, y, loss)
            else:
                loss = F.cross_entropy(z.reshape(-1, V), y.reshape(-1), label_smoothing=LS)
                dg = None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(stage.parameters(), CLIP)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        for r in send_reqs:
            r.wait()
        gstep += 1
        if RANK == K - 1:                      # EMA of the terminal CE, kept on GPU
            _d = loss.detach().float()
            _EMA[0] = _d if _EMA[0] is None else 0.98 * _EMA[0] + 0.02 * _d
            if (s + 1) % LOG_EVERY == 0:
                _el = max(time.time() - t0, 1e-9)
                _em = float(_EMA[0])
                print(f"[ep {ep} step {s+1}/{n_steps}] loss={_em:.3f} "
                      f"ppl~{math.exp(min(_em, 20)):.1f} "
                      f"tok/s={(s+1)*BS*SEQ/_el:,.0f} "
                      f"eta={(n_steps-s-1)*_el/(s+1)/60:.0f}min", flush=True)
        if TIE_AUX and gstep % TIE_SYNC == 0:
            tie_broadcast()
        if dg is not None:
            agg["tight_num"] += dg["tight_num"]; agg["tight_den"] += dg["tight_den"]
            agg["act"] += dg["act"]; agg["cnt"] += 1
    ppls, wppl = pipe_eval("validation")
    packs = [None] * K
    mem = torch.cuda.max_memory_allocated() / 2**30 if use_cuda else 0.0
    tight = agg["tight_num"] / max(agg["tight_den"], 1e-9) if agg["cnt"] else 0.0
    act = agg["act"] / max(agg["cnt"], 1)
    lam_m = float(lam_row.clamp_min(0).mean())
    hz_val = [round(v / max(HZ_CNT, 1), 2) for v in HZ_STAT] if (RANK < K - 1 and MYH > 1) else []
    for _i in range(len(HZ_STAT)):
        HZ_STAT[_i] = 0.0
    HZ_CNT = 0
    dist.all_gather_object(packs, dict(tight=tight, act=act, lam=lam_m, mem=mem, hz=hz_val))
    if RANK == K - 1:
        best = min(best, ppls[-1])
        best_exits = ppls if best_exits is None else [min(a, b) for a, b in zip(best_exits, ppls)]
        dt = time.time() - t0
        tput = n_steps * BS * SEQ / max(dt, 1e-9)
        wstr = f" wppl={wppl:.2f}" if wppl else ""
        exits_s = [f"{p:.1f}" for p in ppls]
        tight_s = [f"{p['tight']:.2f}" for p in packs[:-1]]
        act_s = [f"{p['act']:.2f}" for p in packs[:-1]]
        lam_s = [f"{p['lam']:.2f}" for p in packs[:-1]]
        mem_s = [f"{p['mem']:.1f}" for p in packs]
        hz_s = [p['hz'] for p in packs[:-1]]
        print(f"[epoch {ep:3d}/{EPOCHS}] val_ppl={ppls[-1]:.2f} best={best:.2f}{wstr} "
              f"exits={exits_s} | tight={tight_s} act={act_s} lam={lam_s} "
              f"hz={hz_s} mem={mem_s}GB tok/s={tput:,.0f} ({dt:.0f}s)", flush=True)

tp, twppl = pipe_eval("test")
if RANK == K - 1:
    tw = f" test_wppl={twppl:.2f}" if twppl else ""
    print(f"FINAL best val ppl: {best:.2f} best_exits={[f'{b:.2f}' for b in best_exits]} "
          f"test={tp[-1]:.2f} test_exits={[f'{p:.2f}' for p in tp]}{tw} "
          f"(pipeline K={K}, seed={SEED})", flush=True)
if OUT:
    torch.save({"stage": stage.state_dict()}, f"{OUT}.r{RANK}")
dist.barrier()
dist.destroy_process_group()
