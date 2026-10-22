"""lm_bp_ddp.py - BP baseline (end-to-end backprop) with DDP data parallelism over WORLD_SIZE GPUs, single file.
Same protocol as lm_solo.py MODE=bp: the global batch BS is split evenly across ranks, so optimizer steps and the
LR schedule match single-GPU training at the same BS. Gradient all-reduce is done by DDP. rank 0 evaluates and prints.

Env (defaults): TOKENIZER=word (word | bert; bert needs DATA/bert_cache_<DSET>.pt), DSET=wikitext-2-raw-v1,
  VMAX=32768 (word vocab), NLAYER=8, DMODEL=384, NHEAD=6, FFN=0 (0 = 4*DMODEL), SEQ=256, DROP=0.1,
  BS=64 (global batch; must be divisible by WORLD_SIZE), LR=3e-4, WARMUP=1000, CLIP=0.5, WD=0.0, LS=0.0 (label smoothing),
  EPOCHS=10, STEPS=0 (0 = full epoch), SEED=42, TIE=1 (tie fc to emb), DATA=./data/wikitext, OUT= (checkpoint path),
  BACKEND=nccl|gloo, FAKE_DATA=0 (1 = random tokens, vocab VOCAB=28996). Requires RANK, WORLD_SIZE, MASTER_ADDR/PORT.

Launch (2 GPUs):
  cd <dir> && export CUDA_VISIBLE_DEVICES=0,1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29660 \
    WORLD_SIZE=2 DATA=./data/wikitext TOKENIZER=bert DSET=wikitext-103-raw-v1 \
    NLAYER=16 DMODEL=410 FFN=2100 NHEAD=10 SEQ=256 LR=2.5e-4 WARMUP=1000 CLIP=0.25 \
    WD=0.0 LS=0.0 DROP=0.1 EPOCHS=10 SEED=42 TIE=1 BS=64 OUT=ckpt/bp_ddp.pt && \
  { RANK=0 LOCAL_RANK=0 nohup python -u lm_bp_ddp.py > logs/bp_ddp_r0.log 2>&1 & \
    RANK=1 LOCAL_RANK=1 nohup python -u lm_bp_ddp.py > logs/bp_ddp_r1.log 2>&1 & }
CPU smoke test: MASTER_ADDR=127.0.0.1 MASTER_PORT=29661 WORLD_SIZE=2 BACKEND=gloo FAKE_DATA=1 \
  STEPS=3 EPOCHS=1 NLAYER=4 DMODEL=64 NHEAD=2 FFN=128 SEQ=32 BS=8 VOCAB=500 \
  bash -c 'RANK=0 python -u lm_bp_ddp.py & RANK=1 python -u lm_bp_ddp.py; wait'
"""
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


def E(k, d):
    return os.environ.get(k, d)


TOKENIZER = E("TOKENIZER", "word")
DSET = E("DSET", "wikitext-2-raw-v1")
NLAYER = int(E("NLAYER", "8"))
DMODEL = int(E("DMODEL", "384"))
NHEAD = int(E("NHEAD", "6"))
FFN = int(E("FFN", "0"))
SEQ = int(E("SEQ", "256"))
DROP = float(E("DROP", "0.1"))
BS = int(E("BS", "64"))                    # global batch
LR = float(E("LR", "3e-4"))
WARMUP = int(E("WARMUP", "1000"))
CLIP = float(E("CLIP", "0.5"))
WD = float(E("WD", "0.0"))
LS = float(E("LS", "0.0"))
EPOCHS = int(E("EPOCHS", "10"))
SEED = int(E("SEED", "42"))
DATA = E("DATA", "./data/wikitext")
OUT = E("OUT", "")
STEPS = int(E("STEPS", "0"))
TIE = int(E("TIE", "1"))
CH = 2048
FAKE = E("FAKE_DATA", "0") == "1"

RANK = int(os.environ["RANK"])
WORLD = int(os.environ["WORLD_SIZE"])
assert BS % WORLD == 0, "global batch must be divisible by the number of GPUs"
BS_L = BS // WORLD
nccl_ok = torch.cuda.is_available() and dist.is_nccl_available()
BACKEND = E("BACKEND", "nccl" if nccl_ok else "gloo")
use_cuda = BACKEND == "nccl" and torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", RANK)))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
dev = "cuda" if use_cuda else "cpu"
dist.init_process_group(BACKEND)
torch.manual_seed(SEED)                    # same init on all ranks (DDP semantics)

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
        assert os.path.exists(cache_f), "build the bert token cache first (see README)"
        blob = torch.load(cache_f)
        data, n_words = blob["data"], blob["n_words"]
        V = 28996
    else:
        from collections import Counter
        cnt = Counter()
        for t in ds["train"]["text"]:
            cnt.update(t.split())
        VMAX = int(E("VMAX", "32768"))
        vocab = {w: i + 2 for i, (w, _) in enumerate(cnt.most_common(VMAX - 2))}
        V = len(vocab) + 2

        def enc(split):
            ids = []
            for t in ds[split]["text"]:
                ids += [vocab.get(w, 1) for w in t.split()]
            return torch.tensor(ids, dtype=torch.long)

        data = {s: enc(s) for s in ("train", "validation", "test")}
        n_words = {}


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


class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, DMODEL)
        self.pos = nn.Embedding(SEQ, DMODEL)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.drop = nn.Dropout(DROP)
        self.blocks = nn.ModuleList([Block() for _ in range(NLAYER)])
        self.ln = nn.LayerNorm(DMODEL)
        self.fc = nn.Linear(DMODEL, V)
        if TIE:
            self.fc.weight = self.emb.weight

    def forward(self, x, mask):
        h = self.drop(self.emb(x) + self.pos(torch.arange(x.size(1), device=x.device))[None])
        for b in self.blocks:
            h = b(h, mask)
        return self.fc(self.ln(h))


model = LM().to(dev)
n_par = sum(p.numel() for p in model.parameters()) / 1e6
print(f"[rank {RANK}] bp-ddp params={n_par:.1f}M world={WORLD} bs/rank={BS_L}", flush=True)
ddp = DDP(model, device_ids=[torch.cuda.current_device()] if use_cuda else None)
opt = torch.optim.AdamW(ddp.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.95))
L_train = data["train"].numel() // BS
starts_all = [i for i in range(0, L_train - 2, SEQ) if L_train - 1 - i >= SEQ]
n_steps = min(len(starts_all), STEPS) if STEPS else len(starts_all)
total_steps = max(1, EPOCHS * n_steps)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min((s + 1) / WARMUP, 0.5 * (1 + math.cos(math.pi * s / total_steps))))
mask = torch.full((SEQ, SEQ), float("-inf"), device=dev).triu(1)


def batchify(ids, bs):
    n = len(ids) // bs
    return ids[: n * bs].view(bs, n)


@torch.no_grad()
def evaluate(split):
    if RANK != 0:
        return None, None
    ddp.eval()
    mat = batchify(data[split], BS)
    Lv = mat.size(1)
    nll, ntok = 0.0, 0
    for i in [j for j in range(0, Lv - 2, SEQ) if Lv - 1 - j >= SEQ]:
        x = mat[:, i:i + SEQ].to(dev)
        y = mat[:, i + 1:i + 1 + SEQ].to(dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_cuda):
            z = model(x, mask)
        y_flat = y.reshape(-1)
        zf = z.reshape(-1, V)
        for lo in range(0, y_flat.numel(), CH):
            sl = slice(lo, min(lo + CH, y_flat.numel()))
            lp = F.log_softmax(zf[sl].float(), -1)
            nll += -lp.gather(1, y_flat[sl][:, None]).sum().item()
        ntok += y_flat.numel()
    ddp.train()
    ppl = math.exp(nll / max(ntok, 1))
    wppl = None
    if TOKENIZER == "bert" and split in n_words:
        cov = ntok / max(len(data[split]) - 1, 1)
        wppl = math.exp(nll / (n_words[split] * cov))
    return ppl, wppl


best = float("inf")
gstep = 0
for ep in range(1, EPOCHS + 1):
    t0 = time.time()
    mat = batchify(data["train"], BS)
    g = torch.Generator().manual_seed(SEED * 1000 + ep)   # same batch order on all ranks
    order = [starts_all[j] for j in torch.randperm(len(starts_all), generator=g).tolist()][:n_steps]
    rows = slice(RANK * BS_L, (RANK + 1) * BS_L)
    for i in order:
        x = mat[rows, i:i + SEQ].to(dev)
        y = mat[rows, i + 1:i + 1 + SEQ].to(dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_cuda):
            z = ddp(x, mask)
            loss = F.cross_entropy(z.reshape(-1, V), y.reshape(-1), label_smoothing=LS)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ddp.parameters(), CLIP)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        gstep += 1
    ppl, wppl = evaluate("validation")
    mem = torch.cuda.max_memory_allocated() / 2**30 if use_cuda else 0.0
    packs = [None] * WORLD
    dist.all_gather_object(packs, dict(mem=mem))
    if RANK == 0:
        best = min(best, ppl)
        dt = time.time() - t0
        tput = n_steps * BS * SEQ / max(dt, 1e-9)
        wstr = f" wppl={wppl:.2f}" if wppl else ""
        mem_s = [f"{p['mem']:.1f}" for p in packs]
        print(f"[epoch {ep:3d}/{EPOCHS}] val_ppl={ppl:.2f} best={best:.2f}{wstr} "
              f"mem={mem_s}GB tok/s={tput:,.0f} ({dt:.0f}s)", flush=True)
    dist.barrier()

tp, tw = evaluate("test")
if RANK == 0:
    tws = f" test_wppl={tw:.2f}" if tw else ""
    print(f"FINAL best val ppl: {best:.2f} test={tp:.2f}{tws} "
          f"(bp-ddp, world={WORLD}, seed={SEED})", flush=True)
    if OUT:
        torch.save({"model": model.state_dict()}, OUT)
dist.barrier()
dist.destroy_process_group()
