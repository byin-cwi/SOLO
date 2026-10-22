"""lm_solo.py - single-process SOLO trainer for decoder-only LMs on WikiText.

The transformer (TOKENIZER=word | bert) is split into K modules. In local arms the input of every module k > 0 is
detached, so no gradient crosses a module boundary. Each intermediate module trains its own readout head; the terminal
module trains CE with label smoothing LS. Local arms also train an input-anchored prior head on the detached embedding;
its predictions enter the targets only in mf/mfb.

Arms (MODE):
  bp     end-to-end CE
  dsn    deeply-supervised control: same aux heads (own fc), full backprop, terminal CE + AUX * aux CE
  proto  class-prototype milestones (ablation)
  mf     Milestone Flow (default): input-prior path + entropy-bucket ratchet floors
  mfb    mf with a BETA-mix of the detached own prediction into the prior

Intermediate objective (mf/mfb/proto), per position with label y and t = TK[k]:
  m    = softmax((1 - t) log pi + t log delta_lam),  delta_lam = exp(-lam) on y, the rest spread uniformly
  loss = relu(l_q - l_m) + MU * (1 - sum_v sqrt(q_v m_v)) * [l_q > l_m] + AUX * l_q
pi = smoothed prior (class-prototype row for proto); l_q, l_m = CE of head q and milestone m on y. With FLOOR=bucket,
lam is a per-bucket floor (buckets of H(pi), or of the prior loss with BUCKET_BY=ellpi) initialised at CFLOOR * H(pi)
and ratcheted down to KAPPA * EMA(achieved CE), never below LAM0; FLOOR=global uses lam = LAM0.

Aux-head readout options (all keep the zero cross-module gradient):
  TIE_AUX=1        aux heads read through a detached copy of the terminal fc (no own fc); 0 = own fc per head
  TIE_SYNC=1       N > 1: aux heads read a frozen copy of the terminal fc refreshed every N opt steps
  TIE_MODE=trained random: frozen N(0,1) dictionary rescaled to std(W_terminal) every TIE_SYNC opt steps
  STALE_FC=        path to a frozen imported readout (flat {fc.weight[, fc.bias]}); overrides TIE_SYNC / TIE_MODE
  PRIOR=bigram     bigram: linear on emb | conv: + causal depthwise conv | unigram: static smoothed table (STAGED=1 only)

Env (defaults):
  MODE=mf K=2 EPOCHS=15 BS=32 SEQ=256 DMODEL=384 NLAYER=8 NHEAD=6 FFN=0 (0 = 4*DMODEL) LR=3e-4 WD=0.1 DROP=0.2
  LS=0.1 CLIP=1.0 WARMUP=200 ACCUM=1 SEED=42 STEPS=0 (>0 caps steps/epoch) TIE=0 (tie fc to emb; bp only)
  LAM0=0.1 MU=0.3 AUX=0.1 BETA=0.0 TK=(k+1)/K (comma list) KAPPA=0.9 CFLOOR=0.9 NBUCKET=12 EMA_M=0.99 (proto)
  FLOOR=bucket for mf/mfb, else global; BUCKET_BY=h; HEAD_DEPTH=0 (transformer blocks inside intermediate heads)
  SPLIT= (e.g. "6,2": layers per module) CARRY=0 (detached head-feature carry into the next module)
  MILESTONE_AFTER=0 (epochs of plain local CE first) STAGED=0 (stage-wise backward, mf/mfb; peak activations ~1/K)
  DATA=./data/wikitext DSET=wikitext-2-raw-v1 TOKENIZER=word VMAX=32768 TRAIN_TOKENS=0 (>0 caps train tokens)
  OUT= (checkpoint path) LOAD= (warm restart from an OUT checkpoint, weights only)
Data: HF datasets offline from DATA. The word vocab is built from the train split unless WORD_CACHE=1 and
  DATA/word_cache_<DSET>_<VMAX>.pt exists (the lm_hz_*.py vocab, required for readouts trained there);
  TOKENIZER=bert builds or reads DATA/bert_cache_<DSET>.pt.
Diagnostics:
  PROBE_EVERY=0  N > 0: at step 0, every N opt steps and at the end, log cos(g_hat_k, g_k^BP) and its error terms on
                 the first PROBE_BS=16 x SEQ validation tokens -> PROBE_OUT (default <OUT>.align.jsonl); side-effect free
  DRIFT_T=       comma list of T: log r_T(t) = |W(t) - W(t-T)| / |W(t-T)| of the terminal fc -> DRIFT_OUT
                 (default <OUT>.drift.jsonl); DRIFT_DEV = device for the snapshots

Logged per epoch: val ppl (final head), per-exit ppls, multi-exit ensemble ppl, promise tightness per intermediate
module (overall and top-entropy quartile), hinge active rate, prior ppl, mean floors.
"""
import json
import math
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.nn as nn
import torch.nn.functional as F

CH = 2048  # position chunk for loss/milestone computation (memory bound)

# ----------------------------------------------------------------------- config
E = os.environ.get
MODE = E("MODE", "mf")
K = int(E("K", "2"))
EPOCHS = int(E("EPOCHS", "15"))
BS = int(E("BS", "32"))
SEQ = int(E("SEQ", "256"))
DMODEL = int(E("DMODEL", "384"))
NLAYER = int(E("NLAYER", "8"))
NHEAD = int(E("NHEAD", "6"))
LR = float(E("LR", "3e-4"))
WD = float(E("WD", "0.1"))
DROP = float(E("DROP", "0.2"))
LS = float(E("LS", "0.1"))          # terminal label smoothing (all arms)
LAM0 = float(E("LAM0", "0.1"))      # global floor lower bound
MU = float(E("MU", "0.3"))
AUX = float(E("AUX", "0.1"))
BETA = float(E("BETA", "0.0"))      # self-mix weight (mfb)
PRIOR = E("PRIOR", "bigram")        # bigram: linear on emb | conv: +causal conv | unigram: static table
TIE_AUX = int(E("TIE_AUX", "1"))    # aux heads read out through detached terminal fc (no own fc)
STALE_FC = E("STALE_FC", "")        # frozen imported readout for aux heads; terminal fc still trains
TIE_SYNC = int(E("TIE_SYNC", "1"))  # refresh period (opt steps) of the aux heads' copy of the terminal fc;
                                    # 1 = live detached read; ignored with STALE_FC
assert TIE_SYNC >= 1, TIE_SYNC
TIE_MODE = E("TIE_MODE", "trained")  # trained | random
assert TIE_MODE in ("trained", "random"), TIE_MODE
WORD_CACHE = int(E("WORD_CACHE", "0"))    # 1 = use the pipeline word cache (+ its vocab) if the file exists
PROBE_EVERY = int(E("PROBE_EVERY", "0"))  # >0 = alignment probe every N opt steps (+ step 0 and end); 0 = off
PROBE_BS = int(E("PROBE_BS", "16"))       # probe batch = first PROBE_BS rows x SEQ tokens of validation
PROBE_OUT = E("PROBE_OUT", "")            # jsonl path (default <OUT>.align.jsonl or probe_align.jsonl)
BUCKET_BY = E("BUCKET_BY", "h")     # h: H(prior) | ellpi: prior loss on true label (label-aware)
FLOOR = E("FLOOR", "bucket" if MODE in ("mf", "mfb") else "global")
HEAD_DEPTH = int(E("HEAD_DEPTH", "0"))  # transformer blocks inside intermediate heads
SPLIT = E("SPLIT", "")                  # e.g. "6,2": layers per module (overrides even split)
CARRY = int(E("CARRY", "0"))            # detached head-feature carry into next module
MILESTONE_AFTER = int(E("MILESTONE_AFTER", "0"))  # epochs of plain-local-CE warmup first
CUR_EP = 1
EMA_M = float(E("EMA_M", "0.99"))   # prototype EMA (proto)
KAPPA = float(E("KAPPA", "0.9"))    # floor ratchet vs achieved EMA
CFLOOR = float(E("CFLOOR", "0.9"))  # floor init factor on H(prior)
NBUCKET = int(E("NBUCKET", "12"))
VMAX = int(E("VMAX", "32768"))
SEED = int(E("SEED", "42"))
STEPS = int(E("STEPS", "0"))        # >0: cap steps/epoch (smoke)
DATA = E("DATA", "./data/wikitext")
DSET = E("DSET", "wikitext-2-raw-v1")
TRAIN_TOKENS = int(E("TRAIN_TOKENS", "0"))  # >0: cap train tokens (WT-103 subset regime)
TOKENIZER = E("TOKENIZER", "word")          # word | bert (bert-base-cased)
FFN = int(E("FFN", "0"))                    # FFN width; 0 = 4*DMODEL
TIE = int(E("TIE", "0"))                    # tie final head to embedding (bp anchor only)
CLIP = float(E("CLIP", "1.0"))              # grad clip norm
WARMUP = int(E("WARMUP", "200"))            # LR warmup steps
ACCUM = int(E("ACCUM", "1"))                # gradient accumulation
STAGED = int(E("STAGED", "0"))              # stage-wise backward: peak activations ~L/K
OUT = E("OUT", "")

torch.manual_seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------------- data
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from datasets import load_dataset  # noqa: E402

ds = load_dataset("Salesforce/wikitext", DSET, cache_dir=DATA)

n_words = {}  # raw word counts per split (for word-level ppl under subword tokenizers)

if TOKENIZER == "bert":
    from transformers import AutoTokenizer  # noqa: E402
    cache_f = os.path.join(DATA, f"bert_cache_{DSET}.pt")
    tk = AutoTokenizer.from_pretrained("bert-base-cased")
    V = tk.vocab_size
    if os.path.exists(cache_f):
        blob = torch.load(cache_f)
        data, n_words = blob["data"], blob["n_words"]
    else:
        data = {}
        for s in ("train", "validation", "test"):
            lines = [l.replace("<unk>", "[UNK]") for l in ds[s]["text"] if l.strip()]
            n_words[s] = sum(len(l.split()) for l in lines)
            ids = []
            B_LINES = 2000
            for i in range(0, len(lines), B_LINES):
                enc = tk(lines[i:i + B_LINES], add_special_tokens=False)["input_ids"]
                for e in enc:
                    ids.extend(e)
                    ids.append(tk.sep_token_id)
            data[s] = torch.tensor(ids, dtype=torch.long)
        torch.save({"data": data, "n_words": n_words}, cache_f)
    if TRAIN_TOKENS:
        data["train"] = data["train"][:TRAIN_TOKENS]
else:
    _wc = os.path.join(DATA, f"word_cache_{DSET}_{VMAX}.pt")
    if WORD_CACHE and os.path.exists(_wc):
        # word cache of the lm_hz_*.py scripts: {data: {split: ids}, V, n_words}; vocab unk=1, words from 2,
        # no <eos>. Readouts trained by those scripts (STALE_FC) index this vocab.
        _blob = torch.load(_wc)
        data, V = _blob["data"], int(_blob["V"])
        if TRAIN_TOKENS:
            data["train"] = data["train"][:TRAIN_TOKENS]
        print(f"[data] word cache hit {_wc} (pipeline vocab V={V})", flush=True)
    else:
        def tokenize(split):
            toks = []
            for line in ds[split]["text"]:
                w = line.split()
                if w:
                    toks.extend(w)
                    toks.append("<eos>")
            return toks

        train_toks = tokenize("train")
        if TRAIN_TOKENS:
            train_toks = train_toks[:TRAIN_TOKENS]
        from collections import Counter  # noqa: E402

        cnt = Counter(train_toks)
        vocab = ["<unk>"] + [w for w, _ in cnt.most_common(VMAX - 1)]
        stoi = {w: i for i, w in enumerate(vocab)}
        V = len(vocab)

        def encode(split):
            toks = train_toks if split == "train" else tokenize(split)
            return torch.tensor([stoi.get(w, 0) for w in toks], dtype=torch.long)

        data = {s: encode(s) for s in ("train", "validation", "test")}
print(f"vocab={V} train_tokens={len(data['train'])} val={len(data['validation'])}", flush=True)


def batchify(ids, bs):
    n = len(ids) // bs
    return ids[: n * bs].view(bs, n)


def windows(mat, shuffle):
    L = mat.size(1)
    starts = list(range(0, L - 2, SEQ))
    if shuffle:
        starts = [starts[i] for i in torch.randperm(len(starts)).tolist()]
    for i in starts:
        seq = min(SEQ, L - 1 - i)
        yield mat[:, i : i + seq].to(dev), mat[:, i + 1 : i + 1 + seq].to(dev)


# ------------------------------------------------------------------------ model
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(DMODEL)
        self.attn = nn.MultiheadAttention(DMODEL, NHEAD, dropout=DROP, batch_first=True)
        self.ln2 = nn.LayerNorm(DMODEL)
        fw = FFN if FFN else 4 * DMODEL
        self.mlp = nn.Sequential(
            nn.Linear(DMODEL, fw), nn.GELU(), nn.Linear(fw, DMODEL),
            nn.Dropout(DROP))

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class Head(nn.Module):
    """Per-position readout; depth > 0 adds causal transformer blocks (intermediate heads only)."""
    def __init__(self, depth=0):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(depth)])
        self.ln = nn.LayerNorm(DMODEL)
        self.fc = nn.Linear(DMODEL, V)
        self.tied_src = None  # [terminal fc] (plain list, unregistered) when TIE_AUX

    def forward(self, h, mask=None, ret_h=False):
        for b in self.blocks:
            h = b(h, mask)
        h = self.ln(h)
        if self.fc is not None:
            z = self.fc(h)
        else:  # detached shared readout: no grad ever flows into the terminal fc from here
            tf = self.tied_src[0]
            z = F.linear(h, tf.weight.detach(),
                         None if tf.bias is None else tf.bias.detach())
        return (z, h) if ret_h else z


class PriorHead(nn.Module):
    """Input-anchored prior: linear on the (detached) embedding stream; PRIOR=conv adds a
    causal depthwise conv (receptive field 4 tokens). No trunk gradients either way."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(DMODEL, DMODEL, 4, padding=3, groups=DMODEL) \
            if PRIOR == "conv" else None
        self.ln = nn.LayerNorm(DMODEL)
        self.fc = nn.Linear(DMODEL, V)

    def forward(self, h):
        if self.conv is not None:
            c = self.conv(h.transpose(1, 2))[:, :, : h.size(1)].transpose(1, 2)
            h = h + F.gelu(c)
        return self.fc(self.ln(h))


class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, DMODEL)
        self.pos = nn.Embedding(SEQ, DMODEL)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.drop = nn.Dropout(DROP)
        self.blocks = nn.ModuleList([Block() for _ in range(NLAYER)])
        if SPLIT:
            sizes = [int(s) for s in SPLIT.split(",")]
            assert sum(sizes) == NLAYER and len(sizes) == K
            bounds = [0]
            for s in sizes:
                bounds.append(bounds[-1] + s)
            self.groups = [list(range(bounds[k], bounds[k + 1])) for k in range(K)]
        else:
            per = NLAYER // K
            self.groups = [list(range(k * per, (k + 1) * per if k < K - 1 else NLAYER))
                           for k in range(K)]
        n_heads = K if MODE != "bp" else 1
        self.heads = nn.ModuleList(
            [Head(HEAD_DEPTH if k < n_heads - 1 else 0) for k in range(n_heads)])
        self.prior = PriorHead() if (MODE not in ("bp", "dsn") and PRIOR != "unigram") \
            else None
        if STALE_FC and MODE not in ("bp", "dsn") and n_heads > 1:
            # aux heads read a frozen imported readout; gradients reach the head features, never the readout
            _blob = torch.load(STALE_FC, map_location="cpu")
            _w = _blob["fc.weight"].float()
            _b = _blob.get("fc.bias", None)
            assert tuple(_w.shape) == (V, DMODEL), \
                f"STALE_FC fc.weight {tuple(_w.shape)} != (V,DMODEL)=({V},{DMODEL})"
            self.stale_fc = nn.Linear(DMODEL, V, bias=_b is not None)  # registered => moved by .to(dev)
            with torch.no_grad():
                self.stale_fc.weight.copy_(_w)
                if _b is not None:
                    self.stale_fc.bias.copy_(_b.float())
            for _p in self.stale_fc.parameters():
                _p.requires_grad_(False)                 # grad stays None => AdamW/clip skip it
            for hd in self.heads[:-1]:
                del hd.fc
                hd.fc = None
                hd.tied_src = [self.stale_fc]
            print(f"[stale] aux heads read frozen readout from {STALE_FC} "
                  f"W{tuple(_w.shape)} std={float(_w.std()):.4f}", flush=True)
        elif TIE_AUX and MODE not in ("bp", "dsn") and n_heads > 1:
            src = self.heads[-1].fc
            if TIE_MODE == "random":
                # frozen N(0,1) dictionary (own generator, global RNG untouched); sync_fc.weight =
                # w_rand * std(W_terminal), rescaled every TIE_SYNC opt steps; no bias
                _g = torch.Generator().manual_seed(SEED * 1000 + 7)
                self.register_buffer("w_rand", torch.randn(V, DMODEL, generator=_g), persistent=False)
                with torch.random.fork_rng(devices=[]):
                    self.sync_fc = nn.Linear(DMODEL, V, bias=False)
                with torch.no_grad():
                    self.sync_fc.weight.copy_(self.w_rand * src.weight.std())
                self.sync_fc.weight.requires_grad_(False)
                src = self.sync_fc
                print(f"[tie_random] aux heads read a frozen random readout re-scaled to std(W) every "
                      f"{TIE_SYNC} opt steps (copy adds {V * DMODEL / 1e6:.1f}M untrained params to "
                      f"the params= line)", flush=True)
            elif TIE_SYNC > 1:
                # frozen copy of the terminal fc (copy 0 = init), refreshed in place every TIE_SYNC opt steps.
                # fork_rng keeps the CPU RNG stream (init / shuffle order) identical to TIE_SYNC=1.
                with torch.random.fork_rng(devices=[]):
                    self.sync_fc = nn.Linear(DMODEL, V, bias=src.bias is not None)
                with torch.no_grad():
                    self.sync_fc.weight.copy_(src.weight)
                    if src.bias is not None:
                        self.sync_fc.bias.copy_(src.bias)
                for _p in self.sync_fc.parameters():
                    _p.requires_grad_(False)
                src = self.sync_fc
                print(f"[tie_sync] aux heads read a frozen copy of the terminal fc refreshed every "
                      f"{TIE_SYNC} opt steps (copy adds {V * DMODEL / 1e6:.1f}M untrained params to "
                      f"the params= line)", flush=True)
            for hd in self.heads[:-1]:
                del hd.fc          # drop the aux vocab projection (V*d params each)
                hd.fc = None
                hd.tied_src = [src]
        self.carry = nn.ModuleList([nn.Linear(DMODEL, DMODEL) for _ in range(K - 1)]) \
            if (CARRY and MODE != "bp") else None
        if TIE and MODE == "bp":  # bp only; tying would couple modules
            self.heads[0].fc.weight = self.emb.weight

    @torch.no_grad()
    def refresh_sync_fc(self):
        tf = self.heads[-1].fc
        if TIE_MODE == "random":                     # rescale only; content frozen
            self.sync_fc.weight.copy_(self.w_rand * tf.weight.std())
            return
        self.sync_fc.weight.copy_(tf.weight)
        if tf.bias is not None:
            self.sync_fc.bias.copy_(tf.bias)

    def trunk_in(self, x):
        p = torch.arange(x.size(1), device=x.device)
        return self.drop(self.emb(x) + self.pos(p)[None])

    def forward_all(self, x):
        h = self.trunk_in(x)
        mask = torch.full((x.size(1), x.size(1)), float("-inf"), device=x.device).triu(1)
        outs, h_in, prev_hh = [], h, None
        for k, g in enumerate(self.groups):
            if MODE not in ("bp", "dsn") and k > 0:
                h_in = h_in.detach()
                if self.carry is not None and prev_hh is not None:
                    h_in = h_in + self.carry[k - 1](prev_hh.detach())
            for i in g:
                h_in = self.blocks[i](h_in, mask)
            if MODE != "bp":
                z, prev_hh = self.heads[k](h_in, mask, ret_h=True)
                outs.append(z)
        if MODE == "bp":
            outs.append(self.heads[0](h_in, mask))
        return outs, h


LOG_UNI, H_UNI = None, None
if PRIOR == "unigram":
    assert STAGED, "unigram prior is implemented on the staged path only"
    _cnt = torch.bincount(data["train"], minlength=V).double()
    _p = 0.95 * (_cnt / _cnt.sum()) + 0.05 / V     # same smoothing as the learned-prior path
    LOG_UNI = _p.log().float().to(dev)
    H_UNI = float(-(_p * _p.log()).sum())
    del _cnt, _p

model = LM().to(dev)
LOAD = E("LOAD", "")  # weights only; optimizer/scheduler/ratchet floors restart fresh
if LOAD:
    _sd = torch.load(LOAD, map_location=dev)["model"]
    if hasattr(model, "sync_fc"):   # a TIE_SYNC=1 ckpt has no sync_fc.* -> seed the copy from its terminal fc
        _sd = dict(_sd)
        _tk = f"heads.{len(model.heads) - 1}.fc"
        _sd.setdefault("sync_fc.weight", _sd[f"{_tk}.weight"])
        if model.sync_fc.bias is not None:
            _sd.setdefault("sync_fc.bias", _sd[f"{_tk}.bias"])
    else:                           # a TIE_SYNC>1 ckpt carries sync_fc.* -> drop (no such module here)
        _sd = {k: v for k, v in _sd.items() if not k.startswith("sync_fc.")}
    model.load_state_dict(_sd)
    print(f"loaded {LOAD}", flush=True)
print(f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M mode={MODE} K={K}", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.95))
steps_per_epoch = max(1, (data["train"].size(0) // BS - 1) // SEQ)
if STEPS:
    steps_per_epoch = min(steps_per_epoch, STEPS)
total_steps = max(1, EPOCHS * steps_per_epoch // ACCUM)  # optimizer steps
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min((s + 1) / WARMUP, 0.5 * (1 + math.cos(math.pi * s / max(total_steps, 1)))))

# t-grid for intermediate modules (the terminal module trains CE+LS)
TK = ([float(v) for v in E("TK", "").split(",") if v]
      or [(k + 1) / K for k in range(K - 1)])

# persistent state
proto = None
if MODE == "proto":
    proto = torch.full((V, V), 1.0 / V, device=dev, dtype=torch.float16)  # per interface K=2
lam_b = torch.full((max(K - 1, 1), NBUCKET), -1.0, device=dev)            # ratchet floors
ach_b = torch.full((max(K - 1, 1), NBUCKET), -1.0, device=dev)            # achieved-CE EMA
EDGES = torch.linspace(0, math.log(V) if BUCKET_BY == "h" else 14.0,
                       NBUCKET + 1, device=dev)


def floored_log(y_flat, lam_vec):
    """log delta_y^lam, uniform off-label spread; lam_vec per position."""
    n = y_flat.size(0)
    off = torch.log((1 - torch.exp(-lam_vec)) / (V - 1)).unsqueeze(1).expand(n, V).clone()
    off.scatter_(1, y_flat[:, None], -lam_vec[:, None])
    return off


def local_losses(outs, y):
    """Returns (loss, diag): terminal CE(+LS), prior CE, and chunked milestone losses for
    intermediate modules. All (n,V) fp32 work is chunked at CH positions (memory bound)."""
    y_flat = y.reshape(-1)
    n = y_flat.numel()
    diag = dict(tight_num=[0.0] * (K - 1), tight_den=[0.0] * (K - 1),
                hi_num=[0.0] * (K - 1), hi_den=[0.0] * (K - 1),
                act=[0.0] * (K - 1), n=n, prior_loss=0.0)
    loss = F.cross_entropy(outs[-1].reshape(-1, V), y_flat, label_smoothing=LS)
    if MODE == "bp":
        return loss, diag
    if MODE == "dsn":  # deeply-supervised control: same aux heads, full backprop, aux CE
        for k in range(K - 1):
            loss = loss + AUX * F.cross_entropy(outs[k].reshape(-1, V), y_flat)
        return loss, diag
    prior_logits = model.prior(model.trunk_in_detached).reshape(-1, V)
    prior_loss = F.cross_entropy(prior_logits, y_flat)
    loss = loss + prior_loss
    diag["prior_loss"] = prior_loss.item()
    warm = CUR_EP <= MILESTONE_AFTER  # plain local CE during warmup epochs
    for k in range(K - 1):
        zk = outs[k].reshape(-1, V)
        if warm:
            loss = loss + F.cross_entropy(zk, y_flat)
            continue
        t = TK[k]
        acc = torch.zeros((), device=dev)
        tn = td = hn = hd = ac = 0.0
        for lo in range(0, n, CH):
            sl = slice(lo, min(lo + CH, n))
            yc = y_flat[sl]
            with torch.no_grad():
                pi = 0.95 * F.softmax(prior_logits[sl].float(), -1) + 0.05 / V
                log_pi = pi.log()
                Hpi = -(pi * log_pi).sum(-1)
                ell_pi = -log_pi.gather(1, yc[:, None]).squeeze(1)
                key = Hpi if BUCKET_BY == "h" else ell_pi
                bucket = torch.bucketize(key, EDGES[1:-1])
                hiq = key >= torch.quantile(key, 0.75)
                # floors: bucket ratchet (default for mf/mfb; FLOOR=bucket enables for proto)
                if FLOOR == "bucket":
                    ach = -F.log_softmax(zk[sl].detach().float(), -1) \
                        .gather(1, yc[:, None]).squeeze(1)
                    for b in torch.unique(bucket):
                        mb = bucket == b
                        a = ach[mb].mean()
                        if ach_b[k, b] < 0:
                            ach_b[k, b] = a
                        ach_b[k, b] = 0.99 * ach_b[k, b] + 0.01 * a
                        init = torch.clamp(CFLOOR * key[mb].mean(), min=LAM0)
                        cand = torch.clamp(KAPPA * ach_b[k, b], min=LAM0)
                        lam_b[k, b] = torch.min(init, cand) if lam_b[k, b] < 0 \
                            else torch.min(lam_b[k, b], cand)
                    lam_vec = lam_b[k][bucket]
                else:
                    lam_vec = torch.full_like(ell_pi, LAM0)
                if MODE == "proto":
                    anchor = proto[yc].float().clamp_min(1e-8)
                    log_anchor = anchor.log() - anchor.sum(-1, keepdim=True).log()
                    ell_anchor = -log_anchor.gather(1, yc[:, None]).squeeze(1)
                else:  # mf / mfb
                    log_anchor = log_pi
                    if MODE == "mfb" and BETA > 0:
                        log_self = F.log_softmax(zk[sl].detach().float(), -1)
                        log_anchor = (1 - BETA) * log_pi + BETA * log_self
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
            if MODE == "proto":  # prototype EMA from this module's own predictions
                with torch.no_grad():
                    q_det = logq.detach().exp()
                    uy, inv = torch.unique(yc, return_inverse=True)
                    S = torch.zeros(uy.size(0), V, device=dev)
                    S.index_add_(0, inv, q_det)
                    cnts = torch.bincount(inv, minlength=uy.size(0)).clamp_min(1)[:, None]
                    proto[uy] = (EMA_M * proto[uy].float() + (1 - EMA_M) * S / cnts).half()
            with torch.no_grad():
                tn += ell_q.sum().item(); td += promise.sum().item()
                hn += ell_q[hiq].sum().item(); hd += promise[hiq].sum().item()
                ac += active.sum().item()
        loss = loss + acc / n
        diag["tight_num"][k] = tn / n; diag["tight_den"][k] = td / n
        diag["hi_num"][k] = hn; diag["hi_den"][k] = max(hd, 1e-9)
        diag["act"][k] = ac / n
    return loss, diag


def module_loss(zk, k, y_flat, prior_logits):
    """Milestone loss of intermediate module k (mf/mfb), chunked; same math as local_losses, used by the
    staged path. Returns (loss_sum, diag_k)."""
    assert MODE in ("mf", "mfb"), "staged path supports mf/mfb only"
    n = y_flat.numel()
    t = TK[k]
    acc = torch.zeros((), device=dev)
    tn = td = hn = hd = ac = 0.0
    for lo in range(0, n, CH):
        sl = slice(lo, min(lo + CH, n))
        yc = y_flat[sl]
        with torch.no_grad():
            if prior_logits is None:  # unigram: position-independent prior row
                log_pi = LOG_UNI[None, :].expand(yc.size(0), V)
                Hpi = torch.full((yc.size(0),), H_UNI, device=dev)
                ell_pi = -LOG_UNI[yc]
            else:
                pi = 0.95 * F.softmax(prior_logits[sl].float(), -1) + 0.05 / V
                log_pi = pi.log()
                Hpi = -(pi * log_pi).sum(-1)
                ell_pi = -log_pi.gather(1, yc[:, None]).squeeze(1)
            key = Hpi if BUCKET_BY == "h" else ell_pi
            bucket = torch.bucketize(key, EDGES[1:-1])
            hiq = key >= torch.quantile(key, 0.75)
            if FLOOR == "bucket":
                ach = -F.log_softmax(zk[sl].detach().float(), -1) \
                    .gather(1, yc[:, None]).squeeze(1)
                for b in torch.unique(bucket):
                    mb = bucket == b
                    a = ach[mb].mean()
                    if ach_b[k, b] < 0:
                        ach_b[k, b] = a
                    ach_b[k, b] = 0.99 * ach_b[k, b] + 0.01 * a
                    init = torch.clamp(CFLOOR * key[mb].mean(), min=LAM0)
                    cand = torch.clamp(KAPPA * ach_b[k, b], min=LAM0)
                    lam_b[k, b] = torch.min(init, cand) if lam_b[k, b] < 0 \
                        else torch.min(lam_b[k, b], cand)
                lam_vec = lam_b[k][bucket]
            else:
                lam_vec = torch.full_like(ell_pi, LAM0)
            log_anchor = log_pi
            if MODE == "mfb" and BETA > 0:
                log_self = F.log_softmax(zk[sl].detach().float(), -1)
                log_anchor = (1 - BETA) * log_pi + BETA * log_self
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
        with torch.no_grad():
            tn += ell_q.sum().item(); td += promise.sum().item()
            hn += ell_q[hiq].sum().item(); hd += promise[hiq].sum().item()
            ac += active.sum().item()
    diag_k = dict(tight_num=tn / n, tight_den=td / n,
                  hi_num=hn, hi_den=max(hd, 1e-9), act=ac / n)
    return acc, diag_k


def staged_step(x, y):
    """Stage-wise train step: forward module k -> local loss -> immediate backward
    (frees module k's graph) -> detach -> module k+1. Peak activation memory ~L/K of the
    joint path; identical math to local_losses (losses are module-separable by detach)."""
    y_flat = y.reshape(-1)
    n = y_flat.numel()
    mask = torch.full((x.size(1), x.size(1)), float("-inf"), device=x.device).triu(1)
    diag = dict(tight_num=[0.0] * (K - 1), tight_den=[0.0] * (K - 1),
                hi_num=[0.0] * (K - 1), hi_den=[0.0] * (K - 1),
                act=[0.0] * (K - 1), n=n, prior_loss=0.0)
    h = model.trunk_in(x)
    if model.prior is None:  # unigram: no prior head, no prior params, no prior backward
        pl_det = None
        diag["prior_loss"] = float((-LOG_UNI[y_flat]).mean())
    else:
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            prior_logits = model.prior(h.detach()).reshape(-1, V)
            prior_loss = F.cross_entropy(prior_logits, y_flat)
        (prior_loss / ACCUM).backward()
        diag["prior_loss"] = prior_loss.item()
        pl_det = prior_logits.detach()
        del prior_logits, prior_loss
    h_in = h
    for k, g in enumerate(model.groups):
        if k > 0:
            h_in = h_in.detach()
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            for i in g:
                h_in = model.blocks[i](h_in, mask)
            if k < K - 1:
                zk = model.heads[k](h_in, mask).reshape(-1, V)
                lk, dk = module_loss(zk, k, y_flat, pl_det)
                loss_k = lk / n
            else:
                zk = model.heads[k](h_in, mask).reshape(-1, V)
                loss_k = F.cross_entropy(zk, y_flat, label_smoothing=LS)
        if k < K - 1:
            for key in ("tight_num", "tight_den", "hi_num", "hi_den", "act"):
                diag[key][k] = dk[key]
            saved = h_in
            (loss_k / ACCUM).backward()
            h_in = saved.detach()
        else:
            (loss_k / ACCUM).backward()
        del zk, loss_k
    return diag


@torch.no_grad()
def evaluate(split):
    model.eval()
    mat = batchify(data[split], BS)
    tot = [0.0] * (K if MODE != "bp" else 1)
    tot_ens, n = 0.0, 0
    for x, y in windows(mat, shuffle=False):
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            outs, _ = model.forward_all(x)
        y_flat = y.reshape(-1)
        for lo in range(0, y_flat.numel(), CH):
            sl = slice(lo, min(lo + CH, y_flat.numel()))
            yc = y_flat[sl]
            lp_sum = None
            for i, z in enumerate(outs):
                lp = F.log_softmax(z.reshape(-1, V)[sl].float(), -1)
                tot[i] += -lp.gather(1, yc[:, None]).sum().item()
                lp_sum = lp if lp_sum is None else lp_sum + lp
            lse = F.log_softmax(lp_sum / len(outs), -1)
            tot_ens += -lse.gather(1, yc[:, None]).sum().item()
        n += y_flat.numel()
    model.train()
    ppl = [math.exp(t / n) for t in tot]
    wppl = None
    if TOKENIZER == "bert" and split in n_words:
        cov = n / max(len(data[split]) - 1, 1)
        wppl = math.exp(tot[-1] / (n_words[split] * cov))
    return ppl, math.exp(tot_ens / n), wppl


# ---------------------------------------------------------------- terminal-readout drift r_T(t)
DRIFT_T = [int(v) for v in E("DRIFT_T", "").split(",") if v.strip()]
DRIFT_DEV = E("DRIFT_DEV", "")
DRIFT_LOG = E("DRIFT_OUT", "") or (f"{OUT}.drift.jsonl" if OUT else "drift.jsonl")
_DRIFT = {}


def drift_log(s1):
    """Every multiple of T (opt steps): relative Frobenius change of the terminal fc.weight over the last T steps;
    the first multiple only takes the snapshot. Reads W only; no effect on training."""
    W = model.heads[-1].fc.weight.detach()
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
                f.write(json.dumps(dict(r, lr=lr, warmup_steps=int(WARMUP), tokens_per_step=int(BS * SEQ * ACCUM))) + "\n")
        print(f"[drift @ step {s1}] " + "  ".join(f"T={r['T']}: r={r['r']:.3e}" for r in recs) + f"  lr={lr:.2e}", flush=True)


# ---------------------------------------------------------------- alignment probe (gradient-error decomposition)
PROBE_ON = bool(PROBE_EVERY) and len(model.heads) > 1
PROBE_LOG = PROBE_OUT or (f"{OUT}.align.jsonl" if OUT else "probe_align.jsonl")
_PROBE_XY = []


def align_probe(opt_step):
    """Log cos(g_hat_k, g_k^BP) at every boundary k < K-1 on a fixed validation batch, with the error decomposition.
    One non-detached fp32 forward in the current train/eval mode (one dropout draw shared by all gradients):
      g_hat_k = dCE_k/dh_k   local direction (exact for TK=0; for TK>0 the CE component of l_k; AUX scale dropped)
      g_bp_k  = dCE_K/dh_k   terminal loss (with LS) backpropagated through modules k+1..K
      E_read  = J^T (W_k - W)^T d_k,  E_resid = J^T W^T (d_k - d_K),  E_path = J^T W^T d_K - g_bp_k,
      d_j = dCE_j/dz_j, J = dphi_k/dh_k (VJPs through the head features), random-readout scale folded into W_k.
    g_hat - g_bp = E_read + E_resid + E_path; idr is the relative residual of this identity.
    Side-effect free: autograd.grad only, RNG state restored, no ratchet / prototype updates."""
    if not _PROBE_XY:
        v = data["validation"]
        n = PROBE_BS * SEQ
        assert v.numel() > n + 1, "validation split too small for PROBE_BS x SEQ"
        _PROBE_XY.append((v[:n].view(PROBE_BS, SEQ).to(dev), v[1:n + 1].view(PROBE_BS, SEQ).to(dev)))
    x, y = _PROBE_XY[0]
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if dev == "cuda" else None
    y_flat = y.reshape(-1)
    mask = torch.full((SEQ, SEQ), float("-inf"), device=dev).triu(1)
    h_in, prev_hh = model.trunk_in(x), None
    hs, zs, phis = [], [], []
    for k, g in enumerate(model.groups):
        if k > 0 and model.carry is not None and prev_hh is not None:
            h_in = h_in + model.carry[k - 1](prev_hh.detach())      # as forward_all, without detaching h_in
        for i in g:
            h_in = model.blocks[i](h_in, mask)
        z, phi = model.heads[k](h_in, mask, ret_h=True)
        hs.append(h_in); zs.append(z); phis.append(phi); prev_hh = phi
    ce = [F.cross_entropy(z.reshape(-1, V).float(), y_flat) for z in zs[:-1]]
    ce_K = F.cross_entropy(zs[-1].reshape(-1, V).float(), y_flat, label_smoothing=LS)
    g_bp = torch.autograd.grad(ce_K, hs[:-1], retain_graph=True)
    d_K = torch.autograd.grad(ce_K, zs[-1], retain_graph=True)[0].float()
    W = model.heads[-1].fc.weight.detach().float()
    rows = []
    for k in range(K - 1):
        g_hat, d_k = torch.autograd.grad(ce[k], [hs[k], zs[k]], retain_graph=True)
        d_k = d_k.float()
        hd = model.heads[k]
        Wk = (hd.fc.weight if hd.fc is not None else hd.tied_src[0].weight).detach().float()
        vjp = lambda c: torch.autograd.grad(phis[k], hs[k], grad_outputs=c.to(phis[k].dtype),
                                            retain_graph=True)[0]
        e_read = vjp(d_k @ (Wk - W))
        e_resid = vjp((d_k - d_K) @ W)
        e_path = vjp(d_K @ W) - g_bp[k]
        diff = g_hat - g_bp[k]
        idr = ((diff - (e_read + e_resid + e_path)).norm() / diff.norm().clamp_min(1e-30)).item()
        gh, gb = g_hat.reshape(-1, DMODEL).float(), g_bp[k].reshape(-1, DMODEL).float()
        cos_pos = F.cosine_similarity(gh, gb, dim=-1)
        rows.append(dict(k=k, cos=F.cosine_similarity(gh.reshape(1, -1), gb.reshape(1, -1)).item(),
                         cos_pos=cos_pos.mean().item(), cos_pos_med=cos_pos.median().item(),
                         g_hat=g_hat.norm().item(), g_bp=g_bp[k].norm().item(), diff=diff.norm().item(),
                         e_read=e_read.norm().item(), e_resid=e_resid.norm().item(),
                         e_path=e_path.norm().item(), idr=idr, ce_k=ce[k].item()))
    rec = dict(step=opt_step, ce_K=ce_K.item(), rows=rows,
               arm=dict(mode=MODE, tie_aux=TIE_AUX, tie_mode=TIE_MODE, tie_sync=TIE_SYNC,
                        stale=bool(STALE_FC), seed=SEED, K=K, hd=HEAD_DEPTH))
    with open(PROBE_LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[probe @ {opt_step}] " + "  ".join(
        f"k{r['k']}: cos={r['cos']:.3f} (pos {r['cos_pos']:.3f}) |E|=[{r['e_read']:.2e} "
        f"{r['e_resid']:.2e} {r['e_path']:.2e}] idr={r['idr']:.1e}" for r in rows), flush=True)
    del hs, zs, phis, g_bp, d_K, ce, ce_K
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)


if PROBE_ON:
    print(f"[probe] every {PROBE_EVERY} opt steps on validation[: {PROBE_BS}x{SEQ}] -> {PROBE_LOG}", flush=True)
    align_probe(0)
if DRIFT_T:
    print(f"[drift] r_T(t) of the terminal readout every T in {DRIFT_T} opt steps -> {DRIFT_LOG}", flush=True)

best, best_ens = float("inf"), float("inf")
best_exits = None
step = 0
for ep in range(1, EPOCHS + 1):
    CUR_EP = ep
    t0 = time.time()
    mat = batchify(data["train"], BS)
    agg = None
    for x, y in windows(mat, shuffle=True):
        if STEPS and step % steps_per_epoch == 0 and step > 0 and step // steps_per_epoch >= ep:
            break
        if STAGED and MODE in ("mf", "mfb"):
            diag = staged_step(x, y)
        else:
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                model.trunk_in_detached = model.trunk_in(x).detach()
                outs, _ = model.forward_all(x)
                loss, diag = local_losses(outs, y)
            (loss / ACCUM).backward()
        step += 1
        if step % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            if (TIE_SYNC > 1 or TIE_MODE == "random") and hasattr(model, "sync_fc") \
                    and (step // ACCUM) % TIE_SYNC == 0:
                model.refresh_sync_fc()
            if PROBE_ON and (step // ACCUM) % PROBE_EVERY == 0:
                align_probe(step // ACCUM)
            if DRIFT_T:
                drift_log(step // ACCUM)
        if agg is None:
            agg = {k: (list(v) if isinstance(v, list) else v) for k, v in diag.items()}
            agg["cnt"] = 1
        else:
            for k in ("tight_num", "tight_den", "hi_num", "hi_den", "act"):
                agg[k] = [a + b for a, b in zip(agg[k], diag[k])]
            agg["prior_loss"] += diag["prior_loss"]
            agg["cnt"] += 1
        if STEPS and agg["cnt"] >= steps_per_epoch:
            break
    ppls, ens, wppl = evaluate("validation")
    best = min(best, ppls[-1]); best_ens = min(best_ens, ens)
    best_exits = ppls if best_exits is None else [min(b, p) for b, p in zip(best_exits, ppls)]
    wppl_str = f" wppl={wppl:.2f}" if wppl else ""
    c = agg["cnt"]
    tight = [f"{agg['tight_num'][k]/max(agg['tight_den'][k],1e-9):.2f}" for k in range(K - 1)]
    hi = [f"{agg['hi_num'][k]/max(agg['hi_den'][k],1e-9):.2f}" for k in range(K - 1)]
    act = [f"{agg['act'][k]/c:.2f}" for k in range(K - 1)]
    lam_str = " ".join(f"{v:.2f}" for v in lam_b.clamp_min(0).mean(1).tolist()) \
        if MODE in ("mf", "mfb") else "-"
    print(f"[epoch {ep:3d}/{EPOCHS}] val_ppl={ppls[-1]:.2f} best={best:.2f}{wppl_str} "
          f"exits={[f'{p:.1f}' for p in ppls]} ens={ens:.2f} best_ens={best_ens:.2f} | "
          f"tight={tight} hiQ_tight={hi} act={act} "
          f"prior_ppl={math.exp(agg['prior_loss']/max(c,1)):.1f} lam={lam_str} "
          f"mem={torch.cuda.max_memory_allocated()/2**30:.2f}GB "
          f"({time.time()-t0:.0f}s)", flush=True)

if PROBE_ON and (step // ACCUM) % PROBE_EVERY != 0:
    align_probe(step // ACCUM)

tp, te, twppl = evaluate("test")
tw = f" test_wppl={twppl:.2f}" if twppl else ""
print(f"FINAL best val ppl: {best:.2f} best_ens={best_ens:.2f} "
      f"best_exits={[f'{b:.2f}' for b in best_exits]} "
      f"test={tp[-1]:.2f} test_exits={[f'{p:.2f}' for p in tp]} test_ens={te:.2f}{tw} "
      f"(MODE={MODE}, K={K}, seed={SEED})", flush=True)
if OUT:
    torch.save({"model": model.state_dict()}, OUT)
