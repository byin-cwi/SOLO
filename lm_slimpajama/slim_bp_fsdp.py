"""slim_bp_fsdp.py - FSDP version of slim_bp.py (SlimPajama BP baseline, token-budgeted).
Differences from slim_bp.py: FullyShardedDataParallel instead of DDP (auto-wrap per transformer block,
use_orig_params=True), with the optimizer created after wrapping. FSDP_SHARD=full (ZeRO-3, default) | grad_op (ZeRO-2);
FSDP_MP=1 (default) = bf16 gather + fp32 reduce, 0 = fp32 throughout.
  1) Sharded checkpoint/resume: each rank saves {OUT}.shard.r{rank} (SHARDED_STATE_DICT, model + optimizer + scheduler
     + step; staged via CKPT_STAGE_DIR). Resume requires the same world size; the data stream is reseeded with
     SEED + rank*9973 + start_step.
  2) Final step: FULL_STATE_DICT (offload_to_cpu, rank0_only) weights are saved to {OUT} as a slim_bp-format
     dict(model=sd, step), usable by export_hf.py SRC=bp.
  3) The FSDP forward is collective, so all ranks run evaluate(); rank0 prints.
  state_dict / clipping / eval use the uncompiled FSDP root (fsdp_net); training uses the compiled one.
COMPILE=1: torch.compile the training forward (falls back to eager on failure).

Two architectures:
  ARCH=llama  Transformer++ (fla-org/flame transformer_340M config): RoPE(theta=1e4) + SwiGLU
              (intermediate = int(4d*2/3) rounded up to a multiple of 256) + RMSNorm(1e-6), no position table,
              untied weights; defaults wd=0.01, lr 3e-5 -> 3e-4 -> 3e-5 (warmup counted in tokens).
  ARCH=gelu   pre-LN + GELU MLP + learned position embedding, tied weights by default.
Data: uint16 memmap from prep_slimpajama.py (BIN + BIN.meta.json); random windows, TOKENS budget, no epochs;
the last VAL_TOKENS are held out for validation.

CPU smoke test: FAKE_DATA=1 ARCH=llama STEPS=3 NLAYER=2 DMODEL=64 NHEAD=2 SEQ=64 BS=2 \
  ACCUM=2 EVAL_EVERY=2 VAL_TOKENS=10000 python -u slim_bp_fsdp.py
env: ARCH BIN TOKENS VAL_TOKENS NLAYER DMODEL NHEAD FFN SEQ BS ACCUM LR LR_MIN
     WARMUP WARMUP_TOKENS WD CLIP DROP TIE SEED OUT RESUME SAVE_EVERY EVAL_EVERY
     LOG_EVERY STEPS FAKE_DATA COMPILE FSDP_SHARD FSDP_MP CKPT_STAGE_DIR
     (distributed: RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR/PORT)"""
import json
import math
import os
import shutil
import threading
import time

from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def E(k, d):
    return os.environ.get(k, d)


ARCH = E("ARCH", "llama")                  # llama | gelu
NLAYER = int(E("NLAYER", "24"))
DMODEL = int(E("DMODEL", "1024"))
NHEAD = int(E("NHEAD", "16"))
FFN = int(E("FFN", "0"))                   # 0 = auto: llama uses the fla formula, gelu uses 4d
if FFN == 0:
    FFN = (256 * ((int(DMODEL * 4 * 2 / 3) + 255) // 256)) if ARCH == "llama" else 4 * DMODEL
SEQ = int(E("SEQ", "2048"))
BS = int(E("BS", "8"))
ACCUM = int(E("ACCUM", "16"))
LR = float(E("LR", "3e-4"))
LR_MIN = float(E("LR_MIN", "3e-5" if ARCH == "llama" else "0"))
WARMUP = int(E("WARMUP", "1000"))          # optimizer steps; overridden by the token-converted value when WARMUP_TOKENS>0
WARMUP_TOKENS = int(float(E("WARMUP_TOKENS", "5e8" if ARCH == "llama" else "0")))
WD = float(E("WD", "0.01" if ARCH == "llama" else "0.1"))
CLIP = float(E("CLIP", "1.0"))
DROP = float(E("DROP", "0.0"))
TIE = int(E("TIE", "0" if ARCH == "llama" else "1"))
SEED = int(E("SEED", "42"))
BIN = E("BIN", "data/slim_2b.bin")
TOKENS = int(float(E("TOKENS", "0")))      # training budget; 0 = entire bin minus val
VAL_TOKENS = int(float(E("VAL_TOKENS", "5e6")))
OUT = E("OUT", "")
RESUME = E("RESUME", "")
SAVE_EVERY = int(E("SAVE_EVERY", "2000"))  # optimizer steps
EVAL_EVERY = int(E("EVAL_EVERY", "1000"))
LOG_EVERY = int(E("LOG_EVERY", "50"))
STEPS = int(E("STEPS", "0"))               # >0: cap the number of optimizer steps
FAKE = E("FAKE_DATA", "0") == "1"
COMPILE = int(E("COMPILE", "0"))
FSDP_SHARD = E("FSDP_SHARD", "full")       # full=ZeRO-3 | grad_op=ZeRO-2
FSDP_MP = int(E("FSDP_MP", "1"))           # 1 = bf16 gather + fp32 reduce; 0 = fp32 throughout
CKPT_STAGE_DIR = E("CKPT_STAGE_DIR", "/dev/shm")   # local staging dir; a background thread copies to OUT
_ck_thread = None

RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
DDP_ON = WORLD > 1
dev = "cuda" if torch.cuda.is_available() else "cpu"
if DDP_ON:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    if dev == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", RANK)))
    dist.init_process_group("nccl" if (dev == "cuda" and torch.distributed.is_nccl_available())
                            else "gloo", timeout=timedelta(hours=3))
torch.manual_seed(SEED)
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
assert N_train > SEQ + 1
BUDGET = TOKENS if TOKENS > 0 else N_train
EFF = BS * SEQ * ACCUM * WORLD
opt_steps = max(1, BUDGET // EFF)
if STEPS:
    opt_steps = min(opt_steps, STEPS)
if WARMUP_TOKENS > 0:
    WARMUP = max(1, WARMUP_TOKENS // EFF)
if RANK == 0:
    print(f"arch={ARCH} vocab={V} bin_tokens={N_total:,} train_pool={N_train:,} "
          f"budget={BUDGET:,} eff_batch={EFF:,} tok/step (world={WORLD}) "
          f"opt_steps={opt_steps:,} warmup={WARMUP} lr={LR}->{LR_MIN} wd={WD} "
          f"ffn={FFN} tie={TIE}", flush=True)


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
_t = torch.arange(SEQ).float()
_fr = torch.outer(_t, _inv)
ROPE_COS = torch.cat([_fr.cos(), _fr.cos()], dim=-1)[None, None]   # (1,1,T,hd)
ROPE_SIN = torch.cat([_fr.sin(), _fr.sin()], dim=-1)[None, None]


def rope(x):
    """x: (B,H,T,hd); rotate-half RoPE (theta=1e4, as in fla/llama)."""
    T = x.size(2)
    cos, sin = ROPE_COS[:, :, :T].to(x.device, x.dtype), ROPE_SIN[:, :, :T].to(x.device, x.dtype)
    x1, x2 = x[..., :_HD // 2], x[..., _HD // 2:]
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


class Block(nn.Module):
    """ARCH=gelu: pre-LN + GELU MLP."""
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(DMODEL)
        self.qkv = nn.Linear(DMODEL, 3 * DMODEL, bias=False)
        self.proj = nn.Linear(DMODEL, DMODEL, bias=False)
        self.ln2 = nn.LayerNorm(DMODEL)
        self.mlp = nn.Sequential(nn.Linear(DMODEL, FFN), nn.GELU(),
                                 nn.Linear(FFN, DMODEL), nn.Dropout(DROP))
        self.hd = _HD

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q = q.view(B, T, NHEAD, self.hd).transpose(1, 2)
        k = k.view(B, T, NHEAD, self.hd).transpose(1, 2)
        v = v.view(B, T, NHEAD, self.hd).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=DROP if self.training else 0.0)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.ln2(x))


class LlamaBlock(nn.Module):
    """ARCH=llama: RMSNorm + RoPE attention + SwiGLU (Transformer++)."""
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


class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, DMODEL)
        nn.init.normal_(self.emb.weight, std=0.02)
        if ARCH == "gelu":
            self.pos = nn.Embedding(SEQ, DMODEL)
            nn.init.normal_(self.pos.weight, std=0.02)
        self.drop = nn.Dropout(DROP)
        cls = LlamaBlock if ARCH == "llama" else Block
        self.blocks = nn.ModuleList([cls() for _ in range(NLAYER)])
        self.ln = RMSNorm(DMODEL) if ARCH == "llama" else nn.LayerNorm(DMODEL)
        self.fc = nn.Linear(DMODEL, V, bias=False)
        if TIE:
            self.fc.weight = self.emb.weight
        if ARCH == "llama":                     # fla/llama convention: linear layers N(0,0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        h = self.emb(x)
        if ARCH == "gelu":
            h = h + self.pos(torch.arange(x.size(1), device=x.device))[None]
        h = self.drop(h)
        for b in self.blocks:
            h = b(h)
        return self.fc(self.ln(h))


model = LM().to(dev)
n_par = sum(p.numel() for p in model.parameters()) / 1e6
if RANK == 0:
    print(f"params={n_par:.1f}M (arch={ARCH}, {NLAYER}L/d{DMODEL}/ffn{FFN}, tie={TIE})",
          flush=True)

IS_FSDP = DDP_ON
net = model
if IS_FSDP:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
    _strategy = {"full": ShardingStrategy.FULL_SHARD,
                 "grad_op": ShardingStrategy.SHARD_GRAD_OP}[FSDP_SHARD]
    _mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                         buffer_dtype=torch.bfloat16) if FSDP_MP else None
    net = FSDP(model, auto_wrap_policy=ModuleWrapPolicy({LlamaBlock, Block}),
               sharding_strategy=_strategy, mixed_precision=_mp,
               device_id=(torch.cuda.current_device() if dev == "cuda"
                          else torch.device("cpu")),
               use_orig_params=True)
    if RANK == 0:
        print(f"FSDP wrap: shard={FSDP_SHARD} mp={'bf16-gather/fp32-reduce' if FSDP_MP else 'fp32'} "
              f"unit=transformer block", flush=True)

decay, nodecay = [], []
for n, p in net.named_parameters():        # create the optimizer after FSDP (required by flat shards)
    (decay if p.dim() >= 2 else nodecay).append(p)
opt = torch.optim.AdamW([{"params": decay, "weight_decay": WD},
                         {"params": nodecay, "weight_decay": 0.0}],
                        lr=LR, betas=(0.9, 0.95))
def lr_factor(s):
    """Linear warmup LR_MIN -> LR, then cosine LR -> LR_MIN."""
    if s < WARMUP:
        return (LR_MIN + (LR - LR_MIN) * (s + 1) / WARMUP) / LR
    prog = (s - WARMUP) / max(opt_steps - WARMUP, 1)
    return (LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * prog))) / LR


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
start_step = 0
if IS_FSDP and RESUME and all(os.path.exists(f"{RESUME}.shard.r{r}") for r in range(WORLD)):
    _ck = torch.load(f"{RESUME}.shard.r{RANK}", map_location="cpu", weights_only=False)
    with FSDP.state_dict_type(net, StateDictType.SHARDED_STATE_DICT):
        net.load_state_dict(_ck["model"])
        _osd = FSDP.optim_state_dict_to_load(net, opt, _ck["opt"])
        opt.load_state_dict(_osd)
    sched.load_state_dict(_ck["sched"])
    start_step = _ck["step"]
    del _ck, _osd
    if RANK == 0:
        print(f"resumed @ step {start_step} (FSDP sharded, world={WORLD} must match the one used when saving)",
              flush=True)
elif IS_FSDP and RESUME:
    if RANK == 0:
        print(f"WARM: incomplete {RESUME}.shard.r0..r{WORLD-1}, training from scratch", flush=True)
if (not IS_FSDP) and RESUME and os.path.exists(RESUME):
    ck = torch.load(RESUME, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    start_step = ck["step"]
    if "opt" in ck:
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        if RANK == 0:
            print(f"resumed @ step {start_step}", flush=True)
    else:                                     # weights-only warm start: fresh optimizer state, scheduler fast-forward
        for _ in range(start_step):
            sched.step()
        if RANK == 0:
            print(f"WARM-START @ step {start_step} (weights only, fresh Adam, sched fast-forwarded)",
                  flush=True)

fsdp_net = net                                 # uncompiled FSDP root: for state_dict/clipping/eval
if COMPILE:
    _net_eager = net
    try:
        net = torch.compile(net, dynamic=False)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            _z = net(torch.randint(0, V, (BS, SEQ), device=dev))
        _z.float().mean().backward()
        model.zero_grad(set_to_none=True)
        del _z
        if dev == "cuda":
            torch.cuda.empty_cache()
        if RANK == 0:
            print("COMPILE=1 warm-up ok", flush=True)
    except Exception as _e:                    # noqa: BLE001
        print(f"[rank {RANK}] COMPILE FAILED -> eager: {type(_e).__name__}: {str(_e)[:200]}",
              flush=True)
        net = _net_eager
        model.zero_grad(set_to_none=True)
rng = np.random.default_rng(SEED + RANK * 9973 + start_step)


def get_batch():
    ix = rng.integers(0, N_train - SEQ - 1, size=BS)
    x = np.stack([np.asarray(ids[i:i + SEQ], dtype=np.int64) for i in ix])
    y = np.stack([np.asarray(ids[i + 1:i + 1 + SEQ], dtype=np.int64) for i in ix])
    return torch.from_numpy(x).to(dev, non_blocking=True), \
        torch.from_numpy(y).to(dev, non_blocking=True)


@torch.no_grad()
def evaluate():
    _m = fsdp_net if IS_FSDP else model
    _m.eval()
    nll, n = 0.0, 0
    starts = [i for i in range(N_train, N_total - SEQ, SEQ)]
    for j in range(0, len(starts), BS):
        ck = starts[j:j + BS]
        x = torch.from_numpy(np.stack(
            [np.asarray(ids[i:i + SEQ], dtype=np.int64) for i in ck])).to(dev)
        y = torch.from_numpy(np.stack(
            [np.asarray(ids[i + 1:i + 1 + SEQ], dtype=np.int64) for i in ck])).to(dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            z = _m(x)
        lp = F.log_softmax(z.reshape(-1, V).float(), -1)
        nll += -lp.gather(1, y.reshape(-1)[:, None]).sum().item()
        n += y.numel()
    _m.train()
    return math.exp(nll / max(n, 1))


t0, ema, seen = time.time(), None, start_step * BS * SEQ * ACCUM
best = float("inf")
for step in range(start_step, opt_steps):
    for _ in range(ACCUM):
        x, y = get_batch()
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            z = net(x)
            loss = F.cross_entropy(z.reshape(-1, V), y.reshape(-1))
        (loss / ACCUM).backward()
        seen += BS * SEQ
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
    if IS_FSDP:
        fsdp_net.clip_grad_norm_(CLIP)
    else:
        torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)
    s1 = step + 1
    if RANK == 0 and s1 % LOG_EVERY == 0:
        dt = time.time() - t0
        mem = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0.0
        dtok = (seen - start_step * BS * SEQ * ACCUM) * WORLD
        print(f"[step {s1:6d}/{opt_steps}] loss={ema:.3f} ppl~{math.exp(ema):.1f} "
              f"seen={seen*WORLD/1e6:.0f}M lr={sched.get_last_lr()[0]:.2e} "
              f"tok/s={dtok/dt:,.0f} mem={mem:.1f}GB", flush=True)
    if s1 % EVAL_EVERY == 0 or s1 == opt_steps:
        if IS_FSDP:
            vp = evaluate()                    # the FSDP forward is a collective op, all ranks run it
            best = min(best, vp)
            if RANK == 0:
                print(f"[eval @ step {s1}] val_ppl={vp:.2f} best={best:.2f}", flush=True)
        elif RANK == 0:
            vp = evaluate()
            best = min(best, vp)
            print(f"[eval @ step {s1}] val_ppl={vp:.2f} best={best:.2f}", flush=True)
    if IS_FSDP and OUT and (s1 % SAVE_EVERY == 0 or s1 == opt_steps):
        with FSDP.state_dict_type(fsdp_net, StateDictType.SHARDED_STATE_DICT):
            _msd = fsdp_net.state_dict()
            _osd = FSDP.optim_state_dict(fsdp_net, opt)
        _obj = dict(model=_msd, opt=_osd, sched=sched.state_dict(), step=s1)
        _dst = f"{OUT}.shard.r{RANK}"
        if _ck_thread is not None:
            _ck_thread.join()
        if os.path.isdir(CKPT_STAGE_DIR):
            _loc = os.path.join(CKPT_STAGE_DIR,
                                os.path.basename(_dst) + f".{os.getpid()}.tmp")
            torch.save(_obj, _loc)
            def _ship(src=_loc, dst=_dst):
                shutil.copyfile(src, dst + ".tmp")
                os.replace(dst + ".tmp", dst)
                os.remove(src)
            _ck_thread = threading.Thread(target=_ship, daemon=False)
            _ck_thread.start()
        else:
            torch.save(_obj, _dst + ".tmp")
            os.replace(_dst + ".tmp", _dst)
        del _obj, _msd, _osd
        if s1 == opt_steps:                    # final step: full model export in slim_bp format
            with FSDP.state_dict_type(fsdp_net, StateDictType.FULL_STATE_DICT,
                                      FullStateDictConfig(offload_to_cpu=(dev == "cuda"), rank0_only=True)):
                _fsd = fsdp_net.state_dict()
            if RANK == 0:
                torch.save(dict(model=_fsd, step=s1), OUT + ".tmp")
                os.replace(OUT + ".tmp", OUT)
                print(f"FULL export -> {OUT}", flush=True)
            del _fsd
    if RANK == 0 and OUT and not IS_FSDP and (s1 % SAVE_EVERY == 0 or s1 == opt_steps):
        _obj = dict(model=model.state_dict(), opt=opt.state_dict(),
                    sched=sched.state_dict(), step=s1)
        if _ck_thread is not None:
            _ck_thread.join()
        if os.path.isdir(CKPT_STAGE_DIR):
            _loc = os.path.join(CKPT_STAGE_DIR, os.path.basename(OUT) + f".{os.getpid()}.tmp")
            torch.save(_obj, _loc)
            def _ship(src=_loc, dst=OUT):
                shutil.copyfile(src, dst + ".tmp")
                os.replace(dst + ".tmp", dst)
                os.remove(src)
            _ck_thread = threading.Thread(target=_ship, daemon=False)
            _ck_thread.start()
        else:
            torch.save(_obj, OUT + ".tmp")
            os.replace(OUT + ".tmp", OUT)
        del _obj

if _ck_thread is not None:
    _ck_thread.join()

if RANK == 0:
    print(f"FINAL best val ppl: {best:.2f} (bp/{ARCH}, {n_par:.0f}M, "
          f"budget={BUDGET:,} tokens, world={WORLD}, seed={SEED})", flush=True)
if DDP_ON:
    dist.barrier()
    dist.destroy_process_group()
