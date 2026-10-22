"""slim_untied.py - untied-readout control of slim_solo.py (SOLO-P local training on SlimPajama).
UNTIE=1: each intermediate head learns its own vocabulary projection (aux_fc); UNTIE=0 is identical to slim_solo.py.

Architecture, data and schedule as in slim_bp.py (ARCH=llama); only the training method differs:
  * the NLAYER blocks are split into K gradient-isolated modules (detach at boundaries); each module runs its
    backward right after its forward, so live activations are O(L/K);
  * intermediate module k (aux head of HEAD_DEPTH blocks, output q) minimizes the milestone loss
      m = softmax((1 - t_k) log sg(q) + t_k log o),  o_y = e^{-lam},  o_j = (1 - e^{-lam}) / (V - 1),
      L = (ell_q - ell_m)_+ + MU * 1[ell_q > ell_m] * (1 - sum_j sqrt(q_j m_j)) + AUX * ell_q,
    with ell = -log p(y), sg = stop-gradient, t_k from TK (default (k+1)/K), and lam a ratchet floor
    per unigram-NLL bucket (NBUCKET buckets): lam <- min(lam, max(KAPPA * EMA(ell_q), LAM0)), first value
    capped by max(CFLOOR * unigram NLL, LAM0). The terminal module uses standard CE. The milestone loss is
    exact over the full vocabulary (fp32, CH-row chunks);
  * TIE_AUX (UNTIE=0): the aux readout is the detached terminal head weight (no gradient into fc);
  * DDP: manual gradient allreduce after the staged backward; effective batch = BS*SEQ*ACCUM*WORLD.

Single GPU:
  env CUDA_VISIBLE_DEVICES=0 BIN=data/slim_2b.bin TOKENS=1000000000 BS=8 ACCUM=8 \
    K=4 UNTIE=1 OUT=ckpt/slim_untied_shake.pt nohup python -u slim_untied.py > logs/slim_untied_shake.log &
4 GPUs:
  export MASTER_ADDR=127.0.0.1 MASTER_PORT=29702 WORLD_SIZE=4 ... BS=8 ACCUM=8
CPU smoke test: FAKE_DATA=1 STEPS=3 NLAYER=4 DMODEL=64 NHEAD=2 SEQ=64 BS=2 ACCUM=2 K=2 \
  HEAD_DEPTH=1 EVAL_EVERY=2 VAL_TOKENS=10000 UNTIE=1 python -u slim_untied.py
env: same as slim_bp (BIN TOKENS VAL_TOKENS NLAYER DMODEL NHEAD FFN SEQ BS ACCUM LR LR_MIN
     WARMUP WARMUP_TOKENS WD CLIP SEED OUT RESUME SAVE_EVERY EVAL_EVERY LOG_EVERY STEPS
     FAKE_DATA) + K SPLIT HEAD_DEPTH KAPPA CFLOOR NBUCKET LAM0 MU AUX TK CH UNTIE"""
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
ACCUM = int(E("ACCUM", "16"))
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
# SOLO knobs
K = int(E("K", "4"))
SPLIT = E("SPLIT", "")
HEAD_DEPTH = int(E("HEAD_DEPTH", "2"))
UNTIE = E("UNTIE", "0") == "1"   # 1 = each intermediate head learns its own readout
KAPPA = float(E("KAPPA", "0.7"))
CFLOOR = float(E("CFLOOR", "0.9"))
NBUCKET = int(E("NBUCKET", "12"))
LAM0 = float(E("LAM0", "0.1"))
MU = float(E("MU", "0.3"))
AUX = float(E("AUX", "0.3"))
CH = int(E("CH", "2048"))

RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
DDP_ON = WORLD > 1
dev = "cuda" if torch.cuda.is_available() else "cpu"
if DDP_ON:
    if dev == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", RANK)))
    dist.init_process_group("nccl" if (dev == "cuda" and dist.is_nccl_available())
                            else "gloo")
torch.manual_seed(SEED)                    # same init on all ranks (manual DDP semantics)
if dev == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
EFF = BS * SEQ * ACCUM * WORLD
opt_steps = max(1, BUDGET // EFF)
if STEPS:
    opt_steps = min(opt_steps, STEPS)
if WARMUP_TOKENS > 0:
    WARMUP = max(1, WARMUP_TOKENS // EFF)

# unigram prior (for bucket keys/floor init/promise; counted from the training pool)
_cnt = np.bincount(np.asarray(ids[:min(N_train, 2_000_000_000)]), minlength=V)
_p = 0.95 * (_cnt / _cnt.sum()) + 0.05 / V
LOG_UNI = torch.from_numpy(np.log(_p)).float().to(dev)
del _cnt, _p

# module split and t grid
if SPLIT:
    sizes = [int(s) for s in SPLIT.split(",")]
    assert sum(sizes) == NLAYER and len(sizes) == K
else:
    per = NLAYER // K
    sizes = [per] * (K - 1) + [NLAYER - per * (K - 1)]
bounds = [0]
for s_ in sizes:
    bounds.append(bounds[-1] + s_)
GROUPS = [list(range(bounds[k], bounds[k + 1])) for k in range(K)]
TK = ([float(v) for v in E("TK", "").split(",") if v]
      or [(k + 1) / K for k in range(K - 1)])
EDGES = torch.linspace(0, 14.0, NBUCKET + 1, device=dev)
if RANK == 0:
    print(f"solo/llama vocab={V} train_pool={N_train:,} budget={BUDGET:,} "
          f"eff_batch={EFF:,} (world={WORLD}) opt_steps={opt_steps:,} warmup={WARMUP} "
          f"K={K} split={sizes} hd={HEAD_DEPTH} kappa={KAPPA} aux={AUX} mu={MU}",
          flush=True)


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


class SOLOLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, DMODEL)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.blocks = nn.ModuleList([LlamaBlock() for _ in range(NLAYER)])
        self.hblocks = nn.ModuleList(
            [nn.ModuleList([LlamaBlock() for _ in range(HEAD_DEPTH)])
             for _ in range(K - 1)])
        self.hnorm = nn.ModuleList([RMSNorm(DMODEL) for _ in range(K - 1)])
        if UNTIE:
            self.aux_fc = nn.ModuleList(
                [nn.Linear(DMODEL, V, bias=False) for _ in range(K - 1)])
        self.ln = RMSNorm(DMODEL)
        self.fc = nn.Linear(DMODEL, V, bias=False)   # terminal head, untied from emb
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def head_k(self, h, k):
        for b in self.hblocks[k]:
            h = b(h)
        phi = self.hnorm[k](h)
        if UNTIE:
            return self.aux_fc[k](phi)
        return F.linear(phi, self.fc.weight.detach())    # TIE_AUX: detached terminal readout

    def forward_all(self, x):
        outs = []
        h = self.emb(x)
        for k, g_ in enumerate(GROUPS):
            if k > 0:
                h = h.detach()
            for i in g_:
                h = self.blocks[i](h)
            if k < K - 1:
                outs.append(self.head_k(h, k))
        outs.append(self.fc(self.ln(h)))
        return outs


model = SOLOLM().to(dev)
n_par = sum(p.numel() for p in model.parameters()) / 1e6
if RANK == 0:
    print(f"params={n_par:.1f}M ({NLAYER}L/d{DMODEL}/ffn{FFN} + {K-1}x{HEAD_DEPTH} deep heads, "
          f"tie_aux, terminal untied)", flush=True)

lam_b = torch.full((K - 1, NBUCKET), -1.0, device=dev)
ach_b = torch.full((K - 1, NBUCKET), -1.0, device=dev)


def _bucket_stats(bucket, vals):
    """Vectorized bucket mean: returns (NBUCKET,) means and occurrence mask."""
    cnt = torch.zeros(NBUCKET, device=dev).scatter_add_(
        0, bucket, torch.ones_like(vals))
    ssum = torch.zeros(NBUCKET, device=dev).scatter_add_(0, bucket, vals)
    present = cnt > 0
    mean = torch.where(present, ssum / cnt.clamp_min(1), torch.zeros_like(ssum))
    return mean, present


def module_loss(zk, k, y_flat):
    """Milestone loss for module k with the bucketed ratchet floor (same semantics as lm_wt103/lm_solo.py).
    Returns (mean loss, diagnostics (sum ell_q, sum promise, same on the top-quartile unigram NLL, n_active, n))."""
    n = y_flat.numel()
    t = TK[k]
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
            row_a = ach_b[k]
            fresh = (row_a < 0) & present
            row_a[fresh] = a_mean[fresh]
            upd = present & ~fresh
            row_a[upd] = 0.99 * row_a[upd] + 0.01 * a_mean[upd]
            init = torch.clamp(CFLOOR * key_mean, min=LAM0)
            cand = torch.clamp(KAPPA * row_a, min=LAM0)
            row_l = lam_b[k]
            lfresh = (row_l < 0) & present
            row_l[lfresh] = torch.min(init, cand)[lfresh]
            lupd = present & ~lfresh
            row_l[lupd] = torch.min(row_l, cand)[lupd]
            lam_vec = row_l[bucket]
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
    return acc / n, (tn, td, hn, hd, ac, n)


def staged_step(x, y):
    """Per module: forward -> local loss -> immediate backward (frees the graph) -> detach and pass down."""
    y_flat = y.reshape(-1)
    diags = []
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        h = model.emb(x)
    h_in = h
    for k, g_ in enumerate(GROUPS):
        if k > 0:
            h_in = h_in.detach()
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            for i in g_:
                h_in = model.blocks[i](h_in)
            if k < K - 1:
                zk = model.head_k(h_in, k).reshape(-1, V)
                loss_k, dg = module_loss(zk, k, y_flat)
            else:
                zk = model.fc(model.ln(h_in)).reshape(-1, V)
                loss_k = F.cross_entropy(zk, y_flat)
                dg = None
        saved = h_in
        (loss_k / ACCUM).backward()
        h_in = saved.detach() if k < K - 1 else saved
        if dg is not None:
            diags.append(dg)
        del zk, loss_k
    return diags, None


def grad_allreduce():
    if not DDP_ON:
        return
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(WORLD)


decay, nodecay = [], []
for n_, p in model.named_parameters():
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
start_step = 0
if RESUME and os.path.exists(RESUME):
    ck = torch.load(RESUME, map_location=dev)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"])
    lam_b.copy_(ck["lam_b"]); ach_b.copy_(ck["ach_b"])
    start_step = ck["step"]
    if RANK == 0:
        print(f"resumed @ step {start_step}", flush=True)

rng = np.random.default_rng(SEED + RANK * 9973 + start_step)


def get_batch():
    ix = rng.integers(0, N_train - SEQ - 1, size=BS)
    x = np.stack([np.asarray(ids[i:i + SEQ], dtype=np.int64) for i in ix])
    y = np.stack([np.asarray(ids[i + 1:i + 1 + SEQ], dtype=np.int64) for i in ix])
    return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)


@torch.no_grad()
def evaluate():
    if RANK != 0:
        return None
    model.eval()
    nll = [0.0] * K
    ntok = 0
    i = N_train
    while i + SEQ + 1 <= N_total:
        x = torch.from_numpy(np.asarray(ids[i:i + SEQ], dtype=np.int64))[None].to(dev)
        y = torch.from_numpy(np.asarray(ids[i + 1:i + 1 + SEQ], dtype=np.int64))[None].to(dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            outs = model.forward_all(x)
        yf = y.reshape(-1)
        for j, z in enumerate(outs):
            lp = F.log_softmax(z.reshape(-1, V).float(), -1)
            nll[j] += -lp.gather(1, yf[:, None]).sum().item()
        ntok += SEQ
        i += SEQ
    model.train()
    return [math.exp(v / max(ntok, 1)) for v in nll]


t0, ema, seen = time.time(), None, start_step * BS * SEQ * ACCUM
best = float("inf")
agg = [0.0] * 5
agg_n = 0
for step in range(start_step, opt_steps):
    for _ in range(ACCUM):
        x, y = get_batch()
        diags, _ = staged_step(x, y)
        seen += BS * SEQ
        for k_ in range(K - 1):
            for j in range(5):
                agg[j] += diags[k_][j] / (K - 1)
        agg_n += 1
        with torch.no_grad():
            pass
    grad_allreduce()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)
    s1 = step + 1
    if RANK == 0 and s1 % LOG_EVERY == 0:
        dt = time.time() - t0
        mem = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0.0
        dtok = (seen - start_step * BS * SEQ * ACCUM) * WORLD
        n_ = agg[5 - 5] and 1
        tight = agg[0] / max(agg[1], 1e-9)
        act = agg[4] / max(agg_n * BS * SEQ, 1)
        lam_m = float(lam_b.clamp_min(0).mean())
        print(f"[step {s1:6d}/{opt_steps}] seen={seen*WORLD/1e6:.0f}M "
              f"lr={sched.get_last_lr()[0]:.2e} tight={tight:.2f} act={act:.2f} "
              f"lam={lam_m:.2f} tok/s={dtok/dt:,.0f} mem={mem:.1f}GB", flush=True)
        agg = [0.0] * 5
        agg_n = 0
    if s1 % EVAL_EVERY == 0 or s1 == opt_steps:
        ppls = evaluate()
        if RANK == 0:
            best = min(best, ppls[-1])
            print(f"[eval @ step {s1}] val_ppl={ppls[-1]:.2f} best={best:.2f} "
                  f"exits={[f'{p:.1f}' for p in ppls]}", flush=True)
        if DDP_ON:
            dist.barrier()
    if RANK == 0 and OUT and (s1 % SAVE_EVERY == 0 or s1 == opt_steps):
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), lam_b=lam_b, ach_b=ach_b,
                        step=s1), OUT)

if RANK == 0:
    print(f"FINAL best val ppl: {best:.2f} (solo/llama K={K}, {n_par:.0f}M, "
          f"budget={BUDGET:,} tokens, world={WORLD}, seed={SEED})", flush=True)
if DDP_ON:
    dist.barrier()
    dist.destroy_process_group()
