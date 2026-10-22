"""slim_pipeline_replay.py: SOLO pipeline (pure pipeline, K = WORLD_SIZE) with local activation replay.

Each GPU keeps a ring buffer of its own inputs (stage 0: tokens, others: received activations).
After each fresh optimizer step it runs REPLAY_R extra local optimizer steps of ACCUM micro-batches
sampled from the buffer, with the full local loss and no cross-GPU communication. The LR schedule
advances only on fresh steps; the ratchet floor is updated by replay steps as well. The fresh data
stream uses the same rng as slim_pipeline_solo.py with R=1; replay sampling uses a separate rng.

Pipeline and local objective are as in slim_pipeline_solo.py (explicit milestone implementation here,
no hybrid data parallelism). Effective batch = BS*SEQ*ACCUM.

Launch (4 GPUs):
  export MASTER_ADDR=127.0.0.1 MASTER_PORT=29703 WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
    BIN=data/slim_2b.bin TOKENS=1000000000 BS=8 ACCUM=32 OUT=ckpt/slim_pipe_shake.pt
  then start one process per RANK=0..3 (LOCAL_RANK = RANK).
CPU smoke test (two processes RANK=0/1):
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29704 WORLD_SIZE=2 BACKEND=gloo FAKE_DATA=1 \
  STEPS=3 NLAYER=4 DMODEL=64 NHEAD=2 SEQ=64 BS=2 ACCUM=2 HEAD_DEPTH=1 EVAL_EVERY=2 \
  VAL_TOKENS=10000 LOG_EVERY=1 TIE_SYNC=2

Environment (in addition to the slim_solo.py model/data/optimizer variables):
  TIE_SYNC=100     opt steps between broadcasts of the terminal readout to the aux heads
  REPLAY_R=2       replay optimizer steps after each fresh step (0 = off)
  REPLAY_BUF=64    ring buffer capacity in micro-batches

Checkpoints: OUT.r{rank} holds stage weights, step, lam, ach (no optimizer state).
RESUME loads the stage weights only; training restarts from step 0.
"""
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def E(k, d):
    return os.environ.get(k, d)


NLAYER = int(E("NLAYER", "24"))
DMODEL = int(E("DMODEL", "1024"))
NHEAD = int(E("NHEAD", "16"))
FFN = int(E("FFN", "0"))
if FFN == 0:
    FFN = 256 * ((int(DMODEL * 4 * 2 / 3) + 255) // 256)
SEQ = int(E("SEQ", "2048"))
BS = int(E("BS", "8"))
ACCUM = int(E("ACCUM", "32"))
LR = float(E("LR", "3e-4"))
LR_MIN = float(E("LR_MIN", "3e-5"))
WARMUP = int(E("WARMUP", "1000"))
WARMUP_TOKENS = int(float(E("WARMUP_TOKENS", "5e8")))
WD = float(E("WD", "0.01"))
CLIP = float(E("CLIP", "1.0"))
SEED = int(E("SEED", "42"))
BIN = E("BIN", "data/slim_2b.bin")
TOKENS = int(float(E("TOKENS", "0")))
VAL_TOKENS = int(float(E("VAL_TOKENS", "5e6")))
OUT = E("OUT", "")
RESUME = E("RESUME", "")
SAVE_EVERY = int(E("SAVE_EVERY", "2000"))
EVAL_EVERY = int(E("EVAL_EVERY", "1000"))
LOG_EVERY = int(E("LOG_EVERY", "50"))
STEPS = int(E("STEPS", "0"))
FAKE = E("FAKE_DATA", "0") == "1"
SPLIT = E("SPLIT", "")
HEAD_DEPTH = int(E("HEAD_DEPTH", "2"))
KAPPA = float(E("KAPPA", "0.7"))
CFLOOR = float(E("CFLOOR", "0.9"))
NBUCKET = int(E("NBUCKET", "12"))
LAM0 = float(E("LAM0", "0.1"))
MU = float(E("MU", "0.3"))
AUX = float(E("AUX", "0.3"))
CH = int(E("CH", "2048"))
TIE_SYNC = int(E("TIE_SYNC", "100"))
REPLAY_R = int(E("REPLAY_R", "2"))
REPLAY_BUF = int(E("REPLAY_BUF", "64"))

RANK = int(os.environ["RANK"])
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
dist.init_process_group(BACKEND)
torch.manual_seed(SEED + RANK)
WIRE_DT = torch.bfloat16 if use_cuda else torch.float32

if FAKE:
    V = 32000
    g = torch.Generator().manual_seed(SEED)
    ids = torch.randint(0, V, (2_000_000,), generator=g, dtype=torch.int32).numpy()
else:
    meta = json.load(open(BIN + ".meta.json"))
    V = meta["vocab_size"]
    ids = np.memmap(BIN, dtype=np.uint16, mode="r", shape=(meta["tokens"],))
N_total = len(ids)
N_train = N_total - VAL_TOKENS
BUDGET = TOKENS if TOKENS > 0 else N_train
EFF = BS * SEQ * ACCUM
opt_steps = max(1, BUDGET // EFF)
if STEPS:
    opt_steps = min(opt_steps, STEPS)
if WARMUP_TOKENS > 0:
    WARMUP = max(1, WARMUP_TOKENS // EFF)

_cnt = np.bincount(np.asarray(ids[:min(N_train, 2_000_000_000)]), minlength=V)
_p = 0.95 * (_cnt / _cnt.sum()) + 0.05 / V
LOG_UNI = torch.from_numpy(np.log(_p)).float().to(dev)
del _cnt, _p

if SPLIT:
    sizes = [int(s) for s in SPLIT.split(",")]
    assert sum(sizes) == NLAYER and len(sizes) == K
else:
    per = NLAYER // K
    sizes = [per] * (K - 1) + [NLAYER - per * (K - 1)]
bounds = [0]
for s_ in sizes:
    bounds.append(bounds[-1] + s_)
MY_BLOCKS = list(range(bounds[RANK], bounds[RANK + 1]))
TK = ([float(v) for v in E("TK", "").split(",") if v]
      or [(k + 1) / K for k in range(K - 1)])
EDGES = torch.linspace(0, 14.0, NBUCKET + 1, device=dev)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


_HD = DMODEL // NHEAD
_inv = 1.0 / (10000.0 ** (torch.arange(0, _HD, 2).float() / _HD))
_fr = torch.outer(torch.arange(SEQ).float(), _inv)
ROPE_COS = torch.cat([_fr.cos(), _fr.cos()], dim=-1)[None, None]
ROPE_SIN = torch.cat([_fr.sin(), _fr.sin()], dim=-1)[None, None]


def rope(x):
    T = x.size(2)
    cos = ROPE_COS[:, :, :T].to(x.device, x.dtype)
    sin = ROPE_SIN[:, :, :T].to(x.device, x.dtype)
    x1, x2 = x[..., :_HD // 2], x[..., _HD // 2:]
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


class LlamaBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = RMSNorm(DMODEL)
        self.qkv = nn.Linear(DMODEL, 3 * DMODEL, bias=False)
        self.o = nn.Linear(DMODEL, DMODEL, bias=False)
        self.n2 = RMSNorm(DMODEL)
        self.gate = nn.Linear(DMODEL, FFN, bias=False)
        self.up = nn.Linear(DMODEL, FFN, bias=False)
        self.down = nn.Linear(FFN, DMODEL, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.n1(x)).split(C, dim=2)
        q = rope(q.view(B, T, NHEAD, _HD).transpose(1, 2))
        k = rope(k.view(B, T, NHEAD, _HD).transpose(1, 2))
        v = v.view(B, T, NHEAD, _HD).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, C))
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class Stage(nn.Module):
    def __init__(self):
        super().__init__()
        if RANK == 0:
            self.emb = nn.Embedding(V, DMODEL)
            nn.init.normal_(self.emb.weight, std=0.02)
        self.blocks = nn.ModuleList([LlamaBlock() for _ in MY_BLOCKS])
        if RANK < K - 1:
            self.hblocks = nn.ModuleList([LlamaBlock() for _ in range(HEAD_DEPTH)])
            self.hnorm = RMSNorm(DMODEL)
            self.register_buffer("w_t", torch.zeros(V, DMODEL))   # detached copy of the terminal readout
        else:
            self.ln = RMSNorm(DMODEL)
            self.fc = nn.Linear(DMODEL, V, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def trunk(self, h):
        for b in self.blocks:
            h = b(h)
        return h

    def head_out(self, h):
        if RANK < K - 1:
            for b in self.hblocks:
                h = b(h)
            return F.linear(self.hnorm(h), self.w_t)
        return self.fc(self.ln(h))


stage = Stage().to(dev)
n_par = sum(p.numel() for p in stage.parameters()) / 1e6
print(f"[rank {RANK}] blocks={MY_BLOCKS} params={n_par:.1f}M "
      f"({'terminal' if RANK == K - 1 else 'milestone'})", flush=True)
if RESUME and os.path.exists(f"{RESUME}.r{RANK}"):
    ck = torch.load(f"{RESUME}.r{RANK}", map_location=dev)
    stage.load_state_dict(ck["stage"])
    print(f"[rank {RANK}] resumed", flush=True)

opt = None
decay, nodecay = [], []
for n_, p in stage.named_parameters():
    (decay if p.dim() >= 2 else nodecay).append(p)
opt = torch.optim.AdamW([{"params": decay, "weight_decay": WD},
                         {"params": nodecay, "weight_decay": 0.0}],
                        lr=LR, betas=(0.9, 0.95))


def lr_factor(s):
    if s < WARMUP:
        return (LR_MIN + (LR - LR_MIN) * (s + 1) / WARMUP) / LR
    prog = (s - WARMUP) / max(opt_steps - WARMUP, 1)
    return (LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * prog))) / LR


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
lam_row = torch.full((NBUCKET,), -1.0, device=dev)
ach_row = torch.full((NBUCKET,), -1.0, device=dev)


def _bucket_stats(bucket, vals):
    cnt = torch.zeros(NBUCKET, device=dev).scatter_add_(0, bucket, torch.ones_like(vals))
    ssum = torch.zeros(NBUCKET, device=dev).scatter_add_(0, bucket, vals)
    present = cnt > 0
    mean = torch.where(present, ssum / cnt.clamp_min(1), torch.zeros_like(ssum))
    return mean, present


def module_loss(zk, y_flat):
    n = y_flat.numel()
    t = TK[RANK]
    acc = torch.zeros((), device=dev)
    tn = td = hn = hd = ac = 0.0
    for lo in range(0, n, CH):
        sl = slice(lo, min(lo + CH, n))
        yc = y_flat[sl]
        with torch.no_grad():
            ell_pi = -LOG_UNI[yc]
            bucket = torch.bucketize(ell_pi, EDGES[1:-1])
            hiq = ell_pi >= torch.quantile(ell_pi, 0.75)
            ach = -F.log_softmax(zk[sl].detach().float(), -1) \
                .gather(1, yc[:, None]).squeeze(1)
            a_mean, present = _bucket_stats(bucket, ach)
            key_mean, _ = _bucket_stats(bucket, ell_pi)
            fresh = (ach_row < 0) & present
            ach_row[fresh] = a_mean[fresh]
            upd = present & ~fresh
            ach_row[upd] = 0.99 * ach_row[upd] + 0.01 * a_mean[upd]
            init = torch.clamp(CFLOOR * key_mean, min=LAM0)
            cand = torch.clamp(KAPPA * ach_row, min=LAM0)
            lfresh = (lam_row < 0) & present
            lam_row[lfresh] = torch.min(init, cand)[lfresh]
            lupd = present & ~lfresh
            lam_row[lupd] = torch.min(lam_row, cand)[lupd]
            lam_vec = lam_row[bucket]
            log_self = F.log_softmax(zk[sl].detach().float(), -1)
            off = torch.log((1 - torch.exp(-lam_vec)) / (V - 1)) \
                .unsqueeze(1).expand(yc.size(0), V).clone()
            off.scatter_(1, yc[:, None], -lam_vec[:, None])
            log_m = F.log_softmax((1 - t) * log_self + t * off, -1)
            ell_m = -log_m.gather(1, yc[:, None]).squeeze(1)
            promise = (1 - t) * ell_pi + t * lam_vec
        logq = F.log_softmax(zk[sl].float(), -1)
        ell_q = -logq.gather(1, yc[:, None]).squeeze(1)
        active = (ell_q > ell_m).float().detach()
        hinge = (ell_q - ell_m).clamp_min(0).sum()
        hell = (1 - (0.5 * (logq + log_m)).exp().sum(-1)).clamp_min(0)
        acc = acc + hinge + MU * (hell * active).sum() + AUX * ell_q.sum()
        with torch.no_grad():
            tn += ell_q.sum().item(); td += promise.sum().item()
            hn += ell_q[hiq].sum().item(); hd += promise[hiq].sum().item()
            ac += active.sum().item()
    return acc / n, dict(tight_num=tn / n, tight_den=td / n, act=ac / n)


def tie_broadcast():
    w = stage.fc.weight.data.contiguous() if RANK == K - 1 else stage.w_t
    dist.broadcast(w, K - 1)


def hh_cast(t):
    return t


def val_iter():
    i = N_train
    while i + SEQ + 1 <= N_total:
        yield i
        i += SEQ


@torch.no_grad()
def pipe_eval():
    stage.eval()
    h_buf = torch.empty(BS, SEQ, DMODEL, dtype=WIRE_DT, device=dev)
    y_buf = torch.empty(BS, SEQ, dtype=torch.long, device=dev)
    nll, ntok = 0.0, 0
    for i in val_iter():
        if RANK == 0:
            x = torch.from_numpy(np.asarray(ids[i:i + SEQ], dtype=np.int64)) \
                .repeat(BS, 1).to(dev)          # replicated to fill the batch, only row 0 is counted
            y = torch.from_numpy(np.asarray(ids[i + 1:i + 1 + SEQ], dtype=np.int64)) \
                .repeat(BS, 1).to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_cuda):
                h = stage.trunk(stage.emb(x))
                z = stage.head_out(h)
            if K > 1:
                dist.send(h.to(WIRE_DT).contiguous(), RANK + 1)
                dist.send(y.contiguous(), RANK + 1)
        else:
            dist.recv(h_buf, RANK - 1)
            dist.recv(y_buf, RANK - 1)
            y = y_buf
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_cuda):
                h = stage.trunk(h_buf)
                z = stage.head_out(h)
            if RANK < K - 1:
                dist.send(h.to(WIRE_DT).contiguous(), RANK + 1)
                dist.send(y.contiguous(), RANK + 1)
        yf = y[0].reshape(-1)
        lp = F.log_softmax(z[0].reshape(-1, V).float(), -1)
        nll += -lp.gather(1, yf[:, None]).sum().item()
        ntok += SEQ
    stage.train()
    packs = [None] * K
    dist.all_gather_object(packs, dict(nll=nll, n=ntok))
    return [math.exp(p["nll"] / max(p["n"], 1)) for p in packs]


rng = np.random.default_rng(SEED)            # fresh sampling on rank 0
rng_rp = np.random.default_rng(SEED * 7 + 13 * RANK + 1)   # replay sampling
RBUF = []                                     # (input, y) ring buffer, local to this GPU
h_buf = torch.empty(BS, SEQ, DMODEL, dtype=WIRE_DT, device=dev)
y_buf = torch.empty(BS, SEQ, dtype=torch.long, device=dev)
tie_broadcast()
if RANK == 0:
    print(f"pipe/llama vocab={V} budget={BUDGET:,} eff_batch={EFF:,} "
          f"opt_steps={opt_steps:,} warmup={WARMUP} K={K} split={sizes} "
          f"hd={HEAD_DEPTH}", flush=True)

t0 = time.time()
best = float("inf")
agg = dict(tight_num=0.0, tight_den=0.0, act=0.0, cnt=0)
gmicro = 0
for step in range(opt_steps):
    for _ in range(ACCUM):
        send_reqs = []
        if RANK == 0:
            ix = rng.integers(0, N_train - SEQ - 1, size=BS)
            x = torch.from_numpy(np.stack(
                [np.asarray(ids[i:i + SEQ], dtype=np.int64) for i in ix])).to(dev)
            y = torch.from_numpy(np.stack(
                [np.asarray(ids[i + 1:i + 1 + SEQ], dtype=np.int64) for i in ix])).to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_cuda):
                h = stage.trunk(stage.emb(x))
                z = stage.head_out(h)
            if K > 1:
                hs = h.detach().to(WIRE_DT).contiguous()
                ys = y.contiguous()
                send_reqs = [dist.isend(hs, RANK + 1), dist.isend(ys, RANK + 1)]
            loss, dg = module_loss(z.reshape(-1, V), y.reshape(-1))
        else:
            dist.recv(h_buf, RANK - 1)
            dist.recv(y_buf, RANK - 1)
            y = y_buf.clone()
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_cuda):
                h = stage.trunk(h_buf)
                z = stage.head_out(h)
            if RANK < K - 1:
                hs = h.detach().to(WIRE_DT).contiguous()
                ys = y.contiguous()
                send_reqs = [dist.isend(hs, RANK + 1), dist.isend(ys, RANK + 1)]
                loss, dg = module_loss(z.reshape(-1, V), y.reshape(-1))
            else:
                loss = F.cross_entropy(z.reshape(-1, V), y.reshape(-1))
                dg = None
        (loss / ACCUM).backward()
        for r_ in send_reqs:
            r_.wait()
        if REPLAY_R > 0:                       # rank 0 stores tokens, others store activations
            item = (x, y) if RANK == 0 else (h_buf.clone(), y.clone())
            if len(RBUF) >= REPLAY_BUF:
                RBUF.pop(0)
            RBUF.append(item)
        gmicro += 1
        if dg is not None:
            agg["tight_num"] += dg["tight_num"]; agg["tight_den"] += dg["tight_den"]
            agg["act"] += dg["act"]; agg["cnt"] += 1
    torch.nn.utils.clip_grad_norm_(stage.parameters(), CLIP)
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)
    if REPLAY_R > 0 and len(RBUF) >= ACCUM:    # local replay steps: no communication, no sched.step()
        for _r in range(REPLAY_R):
            for _m in range(ACCUM):
                j = int(rng_rp.integers(0, len(RBUF)))
                ib, yb = RBUF[j]
                with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_cuda):
                    hh = stage.trunk(stage.emb(ib) if RANK == 0 else hh_cast(ib))
                    zz = stage.head_out(hh)
                if RANK < K - 1:
                    rl, _ = module_loss(zz.reshape(-1, V), yb.reshape(-1))
                else:
                    rl = F.cross_entropy(zz.reshape(-1, V), yb.reshape(-1))
                (rl / ACCUM).backward()
            torch.nn.utils.clip_grad_norm_(stage.parameters(), CLIP)
            opt.step()
            opt.zero_grad(set_to_none=True)
    s1 = step + 1
    if s1 % TIE_SYNC == 0:
        tie_broadcast()
    if s1 % LOG_EVERY == 0:
        mem = torch.cuda.max_memory_allocated() / 2**30 if use_cuda else 0.0
        tight = agg["tight_num"] / max(agg["tight_den"], 1e-9) if agg["cnt"] else 0.0
        act = agg["act"] / max(agg["cnt"], 1)
        packs = [None] * K
        dist.all_gather_object(packs, dict(tight=tight, act=act, mem=mem))
        agg = dict(tight_num=0.0, tight_den=0.0, act=0.0, cnt=0)
        if RANK == K - 1:
            dt = time.time() - t0
            tput = s1 * EFF / max(dt, 1e-9)
            tight_s = [f"{p['tight']:.2f}" for p in packs[:-1]]
            act_s = [f"{p['act']:.2f}" for p in packs[:-1]]
            mem_s = [f"{p['mem']:.1f}" for p in packs]
            print(f"[step {s1:6d}/{opt_steps}] seen={s1*EFF/1e6:.0f}M "
                  f"lr={sched.get_last_lr()[0]:.2e} tight={tight_s} act={act_s} "
                  f"mem={mem_s}GB tok/s={tput:,.0f}", flush=True)
    if s1 % EVAL_EVERY == 0 or s1 == opt_steps:
        ppls = pipe_eval()
        if RANK == K - 1:
            best = min(best, ppls[-1])
            print(f"[eval @ step {s1}] val_ppl={ppls[-1]:.2f} best={best:.2f} "
                  f"exits={[f'{p:.1f}' for p in ppls]}", flush=True)
    if OUT and (s1 % SAVE_EVERY == 0 or s1 == opt_steps):
        torch.save(dict(stage=stage.state_dict(), step=s1,
                        lam=lam_row, ach=ach_row), f"{OUT}.r{RANK}")

if RANK == K - 1:
    print(f"FINAL best val ppl: {best:.2f} (slim-pipeline K={K}, seed={SEED})", flush=True)
dist.barrier()
dist.destroy_process_group()
