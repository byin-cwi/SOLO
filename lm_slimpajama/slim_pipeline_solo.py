"""slim_pipeline_solo.py: SOLO module-parallel pipeline on SlimPajama (llama blocks).

One process and one module per GPU. Stage 0 = embedding + group 0 + aux head, middle stages relay,
the last stage computes the terminal CE. Only a one-way forward activation stream (h in bf16, y)
crosses GPUs; there is no gradient communication. Each stage consumes activations produced by the
upstream parameters of the previous step (Jacobi lag). Params and optimizer state are sharded by stage.

Hybrid parallelism: K stages x R = WORLD // K replicas, rank = replica*K + stage. Each replica has its
own data stream (SEED + replica*9973); at the end of each step gradients are all-reduced within a stage
(SUM/R) and the ratchet floor is synced (ach = mean over valid, lam = min). Effective batch
= BS*SEQ*ACCUM*R (BS=8 ACCUM=32 -> 524k tokens/step).

Local objective of non-terminal stage k (t = TK[k], default (k+1)/K):
  loss = hinge(ell_q - ell_m) + MU * Hellinger * [active] + AUX * ell_q
  where ell_q is the CE of the aux head (readout tied to the terminal head, broadcast every TIE_SYNC
  steps) and the milestone m_y has promise (1-t)*ell_pi + t*lam, ell_pi = -log unigram(y), lam = per
  entropy-bucket ratchet floor. _ms_b implements this with three full-matrix reductions using the
  row-constant invariance of log_softmax (checked against the explicit form in ms_identity_test.py).
  LOSS=ce replaces it with plain CE on the same aux head (ablation; tight/act columns are then empty).

Launch (4 GPUs):
  export MASTER_ADDR=127.0.0.1 MASTER_PORT=29703 WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
    BIN=data/slim_2b.bin TOKENS=1000000000 BS=8 ACCUM=32 OUT=ckpt/slim_pipe_shake.pt
  then start one process per RANK=0..3 (LOCAL_RANK = RANK). Add K=2 for 2 stages x 2 replicas.
CPU smoke test (two processes RANK=0/1):
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29704 WORLD_SIZE=2 BACKEND=gloo FAKE_DATA=1 \
  STEPS=3 NLAYER=4 DMODEL=64 NHEAD=2 SEQ=64 BS=2 ACCUM=2 HEAD_DEPTH=1 EVAL_EVERY=2 \
  VAL_TOKENS=10000 LOG_EVERY=1 TIE_SYNC=2

Environment (in addition to the slim_solo.py model/data/optimizer variables):
  K=WORLD_SIZE     number of pipeline stages
  SPLIT=""         comma list of blocks per stage (default: even split)
  TIE_SYNC=100     opt steps between broadcasts of the terminal readout to the aux heads
  DUP_DATA=0       1 = all replicas see the same data (R>1 then reproduces R=1 exactly)
  LOSS=md          md = milestone loss | ce = greedy local CE
  MS_W=1           weight of the milestone term
  COMPILE=0        1 = torch.compile the milestone loss; 2 = also compile trunk/head_out
  CKPT_STAGE_DIR=/dev/shm   staging dir for checkpoints (moved to OUT by a background thread)
  DRIFT_T=""       e.g. "10,100": log r_T = |W(t)-W(t-T)|_F/|W(t-T)|_F of the terminal readout
  Post-training from a BP checkpoint (all off by default):
  FREEZE=0         freeze the trunk for the first N opt steps (terminal stage fully frozen)
  HEAD_LR_MULT=1   aux head LR multiplier (capped at the base LR)
  FREEZE_STAGES="" stages whose trunk is frozen for the whole run
  TRUNK_LR_MULT=1  per-stage trunk LR multiplier (single value or comma list indexed by stage)
  L2SP=0           grad += L2SP*(p - p0), pulls the trunk toward the RESUME weights
  UNI_BIN=""       bin used to estimate the unigram LOG_UNI (default: training pool)
  RATCHET_FROM=""  ckpt prefix to restore lam/ach from on a weights-only warm start
  FANCHOR=0        feature anchor FANCHOR*mean((h-h_base)^2)/mean(h_base^2), h_base from a frozen
                   copy of the warm-start stage (cf. arXiv 2605.04913); MS_W=0 gives pure preservation

Log line: EMA train loss/ppl, per-stage tight/act certificate stats, busy = compute / step time per GPU,
rel = |dtheta|/|theta_base| of the trunk, gn = pre-clip grad norm; tok/s excludes eval time.

Checkpoints: replica 0 saves OUT.r{stage} (stage weights, AdamW state with 4 param groups
trunk/head x decay/no-decay, step, lam, ach), staged in CKPT_STAGE_DIR and atomically renamed at OUT.
RESUME with a full checkpoint restores everything and fast-forwards the scheduler and the data rng,
so the data stream is identical to an uninterrupted run. A weights-only checkpoint (bp_to_pipe.py)
is a warm start: scheduler fast-forwarded, fresh AdamW and ratchet (unless RATCHET_FROM).
Process groups use a 3h timeout.
"""
import json
import math
import os
import shutil
import threading
import time
from datetime import timedelta

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
FREEZE = int(E("FREEZE", "0"))
HEAD_LR_MULT = float(E("HEAD_LR_MULT", "1"))
FREEZE_STAGES = [int(x) for x in E("FREEZE_STAGES", "").split(",") if x.strip() != ""]
_tlm = [float(x) for x in E("TRUNK_LR_MULT", "1").split(",")]
L2SP = float(E("L2SP", "0"))
UNI_BIN = E("UNI_BIN", "")
RATCHET_FROM = E("RATCHET_FROM", "")
FANCHOR = float(E("FANCHOR", "0"))               # ANCHOR is the L2SP parameter snapshot
MS_W = float(E("MS_W", "1"))
LOSS = E("LOSS", "md")
assert LOSS in ("md", "ce"), LOSS
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
COMPILE = int(E("COMPILE", "0"))

RANK = int(os.environ["RANK"])
WORLD = int(os.environ["WORLD_SIZE"])
K = int(E("K", str(WORLD)))
assert WORLD % K == 0, (WORLD, K)
R = WORLD // K
REP, STAGE = RANK // K, RANK % K            # rank = replica*K + stage
DUP_DATA = E("DUP_DATA", "0") == "1"
nccl_ok = torch.cuda.is_available() and dist.is_nccl_available()
BACKEND = E("BACKEND", "nccl" if nccl_ok else "gloo")
use_cuda = BACKEND == "nccl" and torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", RANK)))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
dev = "cuda" if use_cuda else "cpu"
dist.init_process_group(BACKEND, timeout=timedelta(hours=3))
_TO = timedelta(hours=3)                    # new_group does not inherit the default-group timeout
STAGE_GROUPS = [dist.new_group([r_ * K + s_ for r_ in range(R)], timeout=_TO) for s_ in range(K)]
REP_GROUPS = [dist.new_group([r_ * K + s_ for s_ in range(K)], timeout=_TO) for r_ in range(R)]
CKPT_STAGE_DIR = E("CKPT_STAGE_DIR", "/dev/shm")
torch.manual_seed(SEED + STAGE)              # same init for all replicas of a stage
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
EFF = BS * SEQ * ACCUM * R
opt_steps = max(1, BUDGET // EFF)
if STEPS:
    opt_steps = min(opt_steps, STEPS)
if WARMUP_TOKENS > 0:
    WARMUP = max(1, WARMUP_TOKENS // EFF)

if UNI_BIN:
    _um = json.load(open(UNI_BIN + ".meta.json"))
    assert _um["vocab_size"] == V, (_um["vocab_size"], V)
    _uids = np.memmap(UNI_BIN, dtype=np.uint16, mode="r", shape=(_um["tokens"],))
    _cnt = np.bincount(np.asarray(_uids[:min(_um["tokens"], 50_000_000)]), minlength=V)
    del _uids
else:
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
MY_BLOCKS = list(range(bounds[STAGE], bounds[STAGE + 1]))
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
        if STAGE == 0:
            self.emb = nn.Embedding(V, DMODEL)
            nn.init.normal_(self.emb.weight, std=0.02)
        self.blocks = nn.ModuleList([LlamaBlock() for _ in MY_BLOCKS])
        if STAGE < K - 1:
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
        if STAGE < K - 1:
            for b in self.hblocks:
                h = b(h)
            return F.linear(self.hnorm(h), self.w_t)
        return self.fc(self.ln(h))


stage = Stage().to(dev)
n_par = sum(p.numel() for p in stage.parameters()) / 1e6
print(f"[rank {RANK}] rep={REP} stage={STAGE} blocks={MY_BLOCKS} params={n_par:.1f}M "
      f"({'terminal' if STAGE == K - 1 else 'milestone'})", flush=True)
RES_CK = None
if RESUME and os.path.exists(f"{RESUME}.r{STAGE}"):
    RES_CK = torch.load(f"{RESUME}.r{STAGE}", map_location=dev)
    stage.load_state_dict(RES_CK["stage"])

ref_stage = None
if FANCHOR > 0 and STAGE < K - 1:                # frozen copy of the warm-start stage
    import copy
    ref_stage = copy.deepcopy(stage).eval()
    for _p in ref_stage.parameters():
        _p.requires_grad_(False)
    print(f"[rank {RANK}] feature-anchor ON: FANCHOR={FANCHOR} MS_W={MS_W}", flush=True)


def anchor_loss(h, x_or_hin):
    """Scale-free MSE between h and the reference stage's trunk output on the same input."""
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=use_cuda):
        hb = ref_stage.trunk(ref_stage.emb(x_or_hin) if STAGE == 0 else x_or_hin)
    return (h.float() - hb.float()).pow(2).mean() / (hb.float().pow(2).mean() + 1e-8)

opt = None
def _is_head(n_):
    return n_.startswith("hblocks") or n_.startswith("hnorm")


decay, nodecay, hdecay, hnodecay = [], [], [], []
for n_, p in stage.named_parameters():
    ((hdecay if p.dim() >= 2 else hnodecay) if _is_head(n_)
     else (decay if p.dim() >= 2 else nodecay)).append(p)
opt = torch.optim.AdamW([{"params": decay, "weight_decay": WD},
                         {"params": nodecay, "weight_decay": 0.0},
                         {"params": hdecay, "weight_decay": WD},
                         {"params": hnodecay, "weight_decay": 0.0}],
                        lr=LR, betas=(0.9, 0.95))
FROZEN = [p for n_, p in stage.named_parameters() if not _is_head(n_)]
TRUNK_MULT = _tlm[STAGE] if len(_tlm) > 1 else _tlm[0]
ANCHOR = [p.detach().clone() for p in FROZEN] if L2SP > 0 else None
BASE0 = [p.detach().float().cpu().clone() for p in FROZEN]            # reference for the rel metric
BASE0_N = math.sqrt(sum(p.pow(2).sum().item() for p in BASE0)) or 1.0


def lr_factor(s):
    if s < WARMUP:
        return (LR_MIN + (LR - LR_MIN) * (s + 1) / WARMUP) / LR
    prog = (s - WARMUP) / max(opt_steps - WARMUP, 1)
    return (LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * prog))) / LR


def lr_factor_trunk(s):
    return lr_factor(s) * TRUNK_MULT


def lr_factor_head(s):
    return min(1.0, lr_factor(s) * HEAD_LR_MULT)


sched = torch.optim.lr_scheduler.LambdaLR(
    opt, [lr_factor_trunk, lr_factor_trunk, lr_factor_head, lr_factor_head])
lam_row = torch.full((NBUCKET,), -1.0, device=dev)
ach_row = torch.full((NBUCKET,), -1.0, device=dev)
start_step = 0
if RES_CK is not None:
    start_step = RES_CK["step"]
    if "opt" in RES_CK:                          # full checkpoint: resume
        opt.load_state_dict(RES_CK["opt"])
        lam_row.copy_(RES_CK["lam"].to(dev))
        ach_row.copy_(RES_CK["ach"].to(dev))
        for _ in range(start_step):
            sched.step()
        print(f"[rank {RANK}] resumed @ step {start_step} (full)", flush=True)
    else:                                        # weights only (bp_to_pipe.py): warm start
        for _ in range(start_step):
            sched.step()
        _rst = "fresh ratchet"
        if RATCHET_FROM and os.path.exists(f"{RATCHET_FROM}.r{STAGE}"):
            _rk = torch.load(f"{RATCHET_FROM}.r{STAGE}", map_location=dev)
            lam_row.copy_(_rk["lam"].to(dev))
            ach_row.copy_(_rk["ach"].to(dev))
            del _rk
            _rst = f"ratchet<-{RATCHET_FROM}"
        elif RATCHET_FROM:
            raise FileNotFoundError(f"RATCHET_FROM={RATCHET_FROM}.r{STAGE} does not exist")
        print(f"[rank {RANK}] WARM-START weights-only @ step {start_step} "
              f"(fresh opt, {_rst}, sched fast-forwarded, freeze={FREEZE})", flush=True)
    del RES_CK


def _bucket_stats(bucket, vals):
    cnt = torch.zeros(NBUCKET, device=dev).scatter_add_(0, bucket, torch.ones_like(vals))
    ssum = torch.zeros(NBUCKET, device=dev).scatter_add_(0, bucket, vals)
    present = cnt > 0
    mean = torch.where(present, ssum / cnt.clamp_min(1), torch.zeros_like(ssum))
    return mean, present


def _ms_a(zc, yc):
    """Chunk part A: log_softmax (with grad) and the achieved loss -log q_y (no grad)."""
    logq = F.log_softmax(zc.float(), -1)
    ach = -logq.detach().gather(1, yc[:, None]).squeeze(1)
    return logq, ach


def _ms_b(logq, yc, lam_vec, t):
    """Chunk part B, equal to the explicit implementation (ms_identity_test.py).
    The milestone offset is a row constant c except in the target column, so with ls = logq.detach():
      log_m = log_softmax((1-t)*ls + delta*e_y),  delta = t*(-lam - c)
      LSE   = S + log(exp(d+delta) - expm1(d)),   S = logsumexp((1-t)*ls),  d = (1-t)*ls_y - S
      ell_m = LSE - (1-t)*ls_y - delta
      sum_j exp(.5(logq_j + log_m_j)) = exp(-LSE/2) * [G + g_y*expm1(delta/2)],
      G = sum_j exp(.5*logq_j + .5*(1-t)*ls_j)
    G and g_y use an alpha-mix so the gradient flows only through .5*logq_j
    (value equals sum_j exp((1-t/2)*logq_j))."""
    ls = logq.detach()
    with torch.no_grad():
        c = torch.log((1 - torch.exp(-lam_vec)) / (V - 1))
        delta = t * (-lam_vec - c)
        s_y = (1 - t) * ls.gather(1, yc[:, None]).squeeze(1)
        S = torch.logsumexp((1 - t) * ls, -1)
        d = s_y - S
        LSE = S + torch.log(torch.exp(d + delta) - torch.expm1(d))
        ell_m = LSE - s_y - delta
    ell_q = -logq.gather(1, yc[:, None]).squeeze(1)
    active = (ell_q > ell_m).float()
    hinge = (ell_q - ell_m).clamp_min(0).sum()
    a = 1 - t / 2
    alpha = 0.5 / a
    Gc = torch.exp(a * logq).sum(-1)
    G = alpha * Gc + (1 - alpha) * Gc.detach()
    gyc = torch.exp(-a * ell_q)
    g_y = alpha * gyc + (1 - alpha) * gyc.detach()
    H = torch.exp(-0.5 * LSE) * (G + g_y * torch.expm1(0.5 * delta))
    hell = (1 - H).clamp_min(0)
    loss = hinge + MU * (hell * active).sum() + AUX * ell_q.sum()
    return loss, ell_q.detach(), active


if COMPILE and STAGE < K - 1:
    _ms_a_eager, _ms_b_eager = _ms_a, _ms_b
    try:                                       # warm-up with real shapes, fall back to eager on failure
        _ms_a = torch.compile(_ms_a, dynamic=False)
        _ms_b = torch.compile(_ms_b, dynamic=False)
        _zc = torch.randn(CH, V, device=dev, dtype=torch.bfloat16 if use_cuda else torch.float32,
                          requires_grad=True)
        _yc = torch.randint(0, V, (CH,), device=dev)
        _lq, _ = _ms_a(_zc, _yc)
        _l, _, _ = _ms_b(_lq, _yc, torch.full((CH,), 2.0, device=dev), TK[STAGE])
        _l.backward()
        del _zc, _yc, _lq, _l
        print(f"[rank {RANK}] COMPILE=1 warm-up ok", flush=True)
    except Exception as _e:                    # noqa: BLE001
        print(f"[rank {RANK}] COMPILE FAILED -> eager: {type(_e).__name__}: {str(_e)[:200]}",
              flush=True)
        _ms_a, _ms_b = _ms_a_eager, _ms_b_eager


def module_loss(zk, y_flat):
    """Returns (loss, stats[5]); stats = (tight_num, tight_den, hi_num, hi_den, act) as GPU tensors.
    No host sync: the ratchet update uses torch.where."""
    n = y_flat.numel()
    t = TK[STAGE]
    acc = torch.zeros((), device=dev)
    st = torch.zeros(5, device=dev)
    for lo in range(0, n, CH):
        sl = slice(lo, min(lo + CH, n))
        yc = y_flat[sl]
        logq, ach = _ms_a(zk[sl], yc)
        with torch.no_grad():
            ell_pi = -LOG_UNI[yc]
            bucket = torch.bucketize(ell_pi, EDGES[1:-1])
            hiq = ell_pi >= torch.quantile(ell_pi, 0.75)
            a_mean, present = _bucket_stats(bucket, ach)
            key_mean, _ = _bucket_stats(bucket, ell_pi)
            fresh = (ach_row < 0) & present
            upd = present & ~fresh
            ach_row.copy_(torch.where(fresh, a_mean,
                          torch.where(upd, 0.99 * ach_row + 0.01 * a_mean, ach_row)))
            init = torch.clamp(CFLOOR * key_mean, min=LAM0)
            cand = torch.clamp(KAPPA * ach_row, min=LAM0)
            lfresh = (lam_row < 0) & present
            lupd = present & ~lfresh
            lam_row.copy_(torch.where(lfresh, torch.min(init, cand),
                          torch.where(lupd, torch.min(lam_row, cand), lam_row)))
            lam_vec = lam_row[bucket]
            promise = (1 - t) * ell_pi + t * lam_vec
        l_c, ell_q, active = _ms_b(logq, yc, lam_vec, t)
        acc = acc + l_c
        with torch.no_grad():
            st += torch.stack([ell_q.sum(), promise.sum(), ell_q[hiq].sum(),
                               promise[hiq].sum(), active.sum()])
    return acc / n, st / n


def tie_broadcast():
    w = stage.fc.weight.data.contiguous() if STAGE == K - 1 else stage.w_t
    dist.broadcast(w, REP * K + K - 1, group=REP_GROUPS[REP])


def dp_sync_grads():
    """Average gradients over the replicas of this stage."""
    for p in stage.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=STAGE_GROUPS[STAGE])
            p.grad.div_(R)


def dp_sync_ratchet():
    """Sync the ratchet across replicas: ach = mean over valid buckets (-1 = unseen), lam = min."""
    valid = (ach_row >= 0).float()
    num = torch.where(valid > 0, ach_row, torch.zeros_like(ach_row))
    dist.all_reduce(num, group=STAGE_GROUPS[STAGE])
    dist.all_reduce(valid, group=STAGE_GROUPS[STAGE])
    ach_row.copy_(torch.where(valid > 0, num / valid.clamp_min(1), torch.full_like(ach_row, -1.0)))
    tmp = torch.where(lam_row < 0, torch.full_like(lam_row, float("inf")), lam_row)
    dist.all_reduce(tmp, op=dist.ReduceOp.MIN, group=STAGE_GROUPS[STAGE])
    lam_row.copy_(torch.where(torch.isinf(tmp), torch.full_like(lam_row, -1.0), tmp))


def val_iter():
    i = N_train
    while i + SEQ + 1 <= N_total:
        yield i
        i += SEQ


@torch.no_grad()
def pipe_eval():
    """Batched val, BS sequences per pipeline pass; the last chunk is padded with its last sequence
    and only valid rows are counted. Every rank derives the chunk sizes from val_iter."""
    stage.eval()
    h_buf = torch.empty(BS, SEQ, DMODEL, dtype=WIRE_DT, device=dev)
    y_buf = torch.empty(BS, SEQ, dtype=torch.long, device=dev)
    nll, ntok = torch.zeros((), device=dev, dtype=torch.float64), 0
    starts = list(val_iter())
    for j in range(0, len(starts), BS):
        ck = starts[j:j + BS]
        nv = len(ck)
        ck = ck + [ck[-1]] * (BS - nv)
        if STAGE == 0:
            x = torch.from_numpy(np.stack(
                [np.asarray(ids[i:i + SEQ], dtype=np.int64) for i in ck])).to(dev)
            y = torch.from_numpy(np.stack(
                [np.asarray(ids[i + 1:i + 1 + SEQ], dtype=np.int64) for i in ck])).to(dev)
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
            if STAGE < K - 1:
                dist.send(h.to(WIRE_DT).contiguous(), RANK + 1)
                dist.send(y.contiguous(), RANK + 1)
        yf = y[:nv].reshape(-1)
        lp = F.log_softmax(z[:nv].reshape(-1, V).float(), -1)
        nll += -lp.gather(1, yf[:, None]).sum().double()
        ntok += nv * SEQ
    stage.train()
    packs = [None] * WORLD
    dist.all_gather_object(packs, dict(nll=nll.item(), n=ntok))
    return [math.exp(p["nll"] / max(p["n"], 1)) for p in packs[:K]]   # K exits of replica 0


if COMPILE >= 2:                              # state_dict format unchanged
    _trunk_eager, _head_eager = stage.trunk, stage.head_out
    try:
        stage.trunk = torch.compile(stage.trunk, dynamic=False)
        stage.head_out = torch.compile(stage.head_out, dynamic=False)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_cuda):
            _hin = (stage.emb(torch.randint(0, V, (BS, SEQ), device=dev)) if STAGE == 0
                    else torch.randn(BS, SEQ, DMODEL, device=dev, dtype=WIRE_DT))
            _z = stage.head_out(stage.trunk(_hin))
        _z.float().mean().backward()
        stage.zero_grad(set_to_none=True)
        del _hin, _z
        if use_cuda:
            torch.cuda.empty_cache()
        print(f"[rank {RANK}] COMPILE=2 warm-up ok", flush=True)
    except Exception as _e:                    # noqa: BLE001
        print(f"[rank {RANK}] COMPILE=2 FAILED -> eager blocks: {type(_e).__name__}: "
              f"{str(_e)[:200]}", flush=True)
        stage.trunk, stage.head_out = _trunk_eager, _head_eager
        stage.zero_grad(set_to_none=True)

rng = np.random.default_rng(SEED if DUP_DATA else SEED + REP * 9973)
for _ in range(start_step * ACCUM):          # resume: fast-forward the data stream
    rng.integers(0, N_train - SEQ - 1, size=BS)
h_buf = torch.empty(BS, SEQ, DMODEL, dtype=WIRE_DT, device=dev)
y_buf = torch.empty(BS, SEQ, dtype=torch.long, device=dev)
if R > 1:                                    # replica 0 is authoritative
    for p_ in stage.parameters():
        dist.broadcast(p_.data, STAGE, group=STAGE_GROUPS[STAGE])
DRIFT_T = [int(v) for v in E("DRIFT_T", "").split(",") if v.strip()]
DRIFT_DEV = E("DRIFT_DEV", "")             # snapshot device: "" = same as W, "cpu" to offload
DRIFT_LOG = E("DRIFT_OUT", "") or (f"{OUT}.drift.jsonl" if OUT else "drift.jsonl")
_DRIFT = {}                                # T -> [snapshot, step of snapshot]


def drift_log(s1):
    """Terminal stage of replica 0. At every multiple of T, log the relative Frobenius change of fc.weight
    over the last T opt steps (the first multiple after start/resume only takes the snapshot)."""
    W = stage.fc.weight.detach()
    recs = []
    for T in DRIFT_T:
        if s1 % T:
            continue
        snap = _DRIFT.get(T)
        if snap is None or snap[1] + T != s1:
            _DRIFT[T] = [W.to(DRIFT_DEV or W.device, copy=True), s1]
            continue
        Wd = W.to(snap[0].device).float()
        d = (Wd - snap[0].float()).norm().item()
        n = snap[0].float().norm().item()
        recs.append(dict(step=s1, T=T, r=d / max(n, 1e-30), dnorm=d, wnorm=n))
        snap[0].copy_(Wd.to(snap[0].dtype)); snap[1] = s1
    if recs:
        lr = float(sched.get_last_lr()[0])
        with open(DRIFT_LOG, "a") as f:
            for r in recs:
                f.write(json.dumps(dict(r, lr=lr, warmup_steps=int(WARMUP), tokens_per_step=int(EFF))) + "\n")
        print(f"[drift @ step {s1}] " + "  ".join(f"T={r['T']}: r={r['r']:.3e}" for r in recs) + f"  lr={lr:.2e}", flush=True)


tie_broadcast()
if RANK == 0:
    print(f"pipe/llama LOSS={LOSS} vocab={V} budget={BUDGET:,} eff_batch={EFF:,} "
          f"opt_steps={opt_steps:,} warmup={WARMUP} K={K} R={R} split={sizes} "
          f"hd={HEAD_DEPTH}", flush=True)

t0 = time.time()
ema = None
best = float("inf")
_ck_thread = None
agg = torch.zeros(5, device=dev)
agg_n = 0
gmicro = 0


def _ev():
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


for step in range(start_step, opt_steps):
    ev_pairs = []                                # per-microbatch (compute start, compute end) events
    ev_step0 = _ev() if use_cuda else None
    for _ in range(ACCUM):
        send_reqs = []
        if STAGE == 0:
            ix = rng.integers(0, N_train - SEQ - 1, size=BS)
            x = torch.from_numpy(np.stack(
                [np.asarray(ids[i:i + SEQ], dtype=np.int64) for i in ix])).to(dev)
            y = torch.from_numpy(np.stack(
                [np.asarray(ids[i + 1:i + 1 + SEQ], dtype=np.int64) for i in ix])).to(dev)
            e0 = _ev() if use_cuda else None
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_cuda):
                h = stage.trunk(stage.emb(x))
                z = stage.head_out(h)
            if K > 1:
                hs = h.detach().to(WIRE_DT).contiguous()
                ys = y.contiguous()
                send_reqs = [dist.isend(hs, RANK + 1), dist.isend(ys, RANK + 1)]
            if LOSS == "ce":
                loss = F.cross_entropy(z.reshape(-1, V), y.reshape(-1))
                dg = None
            else:
                loss, dg = module_loss(z.reshape(-1, V), y.reshape(-1))
            if ref_stage is not None:
                loss = MS_W * loss + FANCHOR * anchor_loss(h, x)
        else:
            dist.recv(h_buf, RANK - 1)
            dist.recv(y_buf, RANK - 1)
            e0 = _ev() if use_cuda else None
            y = y_buf.clone()
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_cuda):
                h = stage.trunk(h_buf)
                z = stage.head_out(h)
            if STAGE < K - 1:
                hs = h.detach().to(WIRE_DT).contiguous()
                ys = y.contiguous()
                send_reqs = [dist.isend(hs, RANK + 1), dist.isend(ys, RANK + 1)]
                if LOSS == "ce":
                    loss = F.cross_entropy(z.reshape(-1, V), y.reshape(-1))
                    dg = None
                else:
                    loss, dg = module_loss(z.reshape(-1, V), y.reshape(-1))
                if ref_stage is not None:
                    loss = MS_W * loss + FANCHOR * anchor_loss(h, h_buf)
            else:
                loss = F.cross_entropy(z.reshape(-1, V), y.reshape(-1))
                dg = None
                ld = loss.detach().float()          # fp32 EMA (a bf16 EMA stalls)
                ema = ld if ema is None else 0.98 * ema + 0.02 * ld
        (loss / ACCUM).backward()
        if use_cuda:
            ev_pairs.append((e0, _ev()))
        for r_ in send_reqs:
            r_.wait()
        gmicro += 1
        if dg is not None:
            agg += dg
            agg_n += 1
    if R > 1:
        dp_sync_grads()
        if STAGE < K - 1:
            dp_sync_ratchet()
    if FREEZE > 0 and step < start_step + FREEZE:
        for p_ in FROZEN:
            p_.grad = None
    elif FREEZE > 0 and step == start_step + FREEZE and RANK == 0:
        print(f"[freeze] trunk unfrozen @ step {step}", flush=True)
    if STAGE in FREEZE_STAGES:
        for p_ in FROZEN:
            p_.grad = None
    elif L2SP > 0:
        for p_, p0_ in zip(FROZEN, ANCHOR):
            if p_.grad is not None:
                p_.grad.add_(p_.detach() - p0_, alpha=L2SP)
    gnorm = float(torch.nn.utils.clip_grad_norm_(stage.parameters(), CLIP))
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)
    s1 = step + 1
    if s1 % TIE_SYNC == 0:
        tie_broadcast()
    if DRIFT_T and STAGE == K - 1 and REP == 0:
        drift_log(s1)
    if s1 % LOG_EVERY == 0:
        mem = torch.cuda.max_memory_allocated() / 2**30 if use_cuda else 0.0
        busy = 0.0
        if use_cuda:
            ev_end = _ev()
            torch.cuda.synchronize()
            tot = ev_step0.elapsed_time(ev_end)
            busy = sum(a.elapsed_time(b) for a, b in ev_pairs) / max(tot, 1e-6)
        a5 = agg.tolist()
        tight = a5[0] / max(a5[1], 1e-9) if agg_n else 0.0
        act = a5[4] / max(agg_n, 1)
        packs = [None] * WORLD
        with torch.no_grad():                # rel = |dtheta| / |theta_base| over the trunk
            rel = math.sqrt(sum((p.detach().float().cpu() - b).pow(2).sum().item()
                                for p, b in zip(FROZEN, BASE0))) / BASE0_N
        dist.all_gather_object(packs, dict(tight=tight, act=act, mem=mem, busy=busy,
                                           rel=rel, gn=gnorm))
        agg.zero_()
        agg_n = 0
        if RANK == K - 1:                    # terminal stage of replica 0
            pk = packs[:K]
            dt = time.time() - t0
            tput = (s1 - start_step) * EFF / max(dt, 1e-9)
            tight_s = [f"{p['tight']:.2f}" for p in pk[:-1]]
            act_s = [f"{p['act']:.2f}" for p in pk[:-1]]
            mem_s = [f"{p['mem']:.1f}" for p in packs]
            busy_s = [f"{p['busy']:.2f}" for p in packs]
            rel_s = [f"{p['rel']:.2e}" for p in packs]
            gn_s = [f"{p['gn']:.2f}" for p in packs]
            em = float(ema)
            print(f"[step {s1:6d}/{opt_steps}] loss={em:.3f} ppl~{math.exp(em):.1f} "
                  f"seen={s1*EFF/1e6:.0f}M "
                  f"lr={sched.get_last_lr()[0]:.2e} tight={tight_s} act={act_s} "
                  f"busy={busy_s} mem={mem_s}GB tok/s={tput:,.0f} "
                  f"rel={rel_s} gn={gn_s}", flush=True)
    if s1 % EVAL_EVERY == 0 or s1 == opt_steps:
        _te = time.time()
        ppls = pipe_eval()
        t0 += time.time() - _te                  # exclude eval time from tok/s
        if RANK == K - 1:
            best = min(best, ppls[-1])
            print(f"[eval @ step {s1}] val_ppl={ppls[-1]:.2f} best={best:.2f} "
                  f"exits={[f'{p:.1f}' for p in ppls]}", flush=True)
    if OUT and REP == 0 and (s1 % SAVE_EVERY == 0 or s1 == opt_steps):
        _final = f"{OUT}.r{STAGE}"
        _obj = dict(stage=stage.state_dict(), opt=opt.state_dict(), step=s1,
                    lam=lam_row, ach=ach_row)
        if _ck_thread is not None:                # wait for the previous move
            _ck_thread.join()
        if os.path.isdir(CKPT_STAGE_DIR):
            _loc = os.path.join(CKPT_STAGE_DIR, os.path.basename(_final) + f".{os.getpid()}.tmp")
            torch.save(_obj, _loc)
            def _ship(src=_loc, dst=_final):      # background copy to OUT, atomic rename
                shutil.copyfile(src, dst + ".tmp")
                os.replace(dst + ".tmp", dst)
                os.remove(src)
            _ck_thread = threading.Thread(target=_ship, daemon=False)
            _ck_thread.start()
        else:
            torch.save(_obj, _final + ".tmp")
            os.replace(_final + ".tmp", _final)
        del _obj

if _ck_thread is not None:
    _ck_thread.join()
if RANK == K - 1:
    print(f"FINAL best val ppl: {best:.2f} (slim-pipeline K={K} x R={R}, seed={SEED})", flush=True)
dist.barrier()
dist.destroy_process_group()
