"""export_hf.py - training ckpt -> HuggingFace LlamaForCausalLM directory.
SRC=pipe reads {CKPT}.r0..r{K-1}. EXIT=-1 (default) assembles the terminal model (rank0 emb + all blocks + last-rank
ln/fc); EXIT=k (0..K-2) exports the k-th intermediate exit (trunk + aux head blocks). Exit readout priority:
hfc.weight (separate readout, TIE_AUX=0) > aux_fc.weight (private readout, UNTIE=1) > w_t (tied copy).
SPLIT=a,b,c,d non-uniform split (same semantics as the training scripts).

Usage:
  SRC=bp   CKPT=ckpt/bp15b.pt   OUTDIR=hf_out/bp15b   python export_hf.py
  SRC=pipe CKPT=ckpt/solo15b.pt OUTDIR=hf_out/solo15b K=4 python export_hf.py

Evaluation (lm-eval-harness):
  pip install lm-eval
  lm_eval --model hf --model_args pretrained=hf_out/bp15b,dtype=bfloat16 \
    --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge \
    --batch_size 32 --output_path evals/bp15b

Before writing, a minimal fp32 replica of the training forward (RMSNorm / rotate-half RoPE theta=1e4 / SwiGLU) is
compared to the HF model on the same weights; max|dlogit| > 2e-3 aborts the export (typical difference ~1e-5).

env: SRC CKPT OUTDIR K EXIT=-1 HEAD_DEPTH=2 SPLIT="" NLAYER=24 DMODEL=1024 NHEAD=16 FFN=0(auto fla formula) SEQ=2048
     V=32000 TOK=mistralai/Mistral-7B-v0.1 SKIP_TOK=0 HF_ENDPOINT(default huggingface.co)
"""
import math
import os

os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("HF_DATASETS_OFFLINE", None)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM


def E(k, d):
    return os.environ.get(k, d)


SRC = E("SRC", "bp")
CKPT = E("CKPT", "ckpt/bp15b.pt")
OUTDIR = E("OUTDIR", "hf_out/model")
K = int(E("K", "4"))
NLAYER = int(E("NLAYER", "24"))
DMODEL = int(E("DMODEL", "1024"))
NHEAD = int(E("NHEAD", "16"))
FFN = int(E("FFN", "0"))
if FFN == 0:
    FFN = ((int(4 * DMODEL * 2 / 3) + 255) // 256) * 256   # fla formula (d1024 -> 2816)
SEQ = int(E("SEQ", "2048"))
V = int(E("V", "32000"))
TOK = E("TOK", "mistralai/Mistral-7B-v0.1")
SKIP_TOK = E("SKIP_TOK", "0") == "1"
HD = DMODEL // NHEAD

# ---------------------------------------------------------------- read training ckpt -> flat dict
if SRC == "bp":
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
    print(f"bp ckpt: {CKPT} step={torch.load(CKPT, map_location='cpu', weights_only=False).get('step', '?')}")
elif SRC == "pipe":
    EXIT = int(E("EXIT", "-1"))               # -1=terminal; k=0..K-2 -> k-th intermediate exit
    HEAD_DEPTH = int(E("HEAD_DEPTH", "2"))
    SPLIT = E("SPLIT", "")
    if SPLIT:                                # layers per module, comma-separated
        sizes = [int(v) for v in SPLIT.split(",")]
        assert sum(sizes) == NLAYER and len(sizes) == K, (sizes, NLAYER, K)
    else:
        per = NLAYER // K
        sizes = [per] * (K - 1) + [NLAYER - per * (K - 1)]
    bounds = [0]
    for s_ in sizes:
        bounds.append(bounds[-1] + s_)
    last = K - 1 if EXIT < 0 else EXIT
    sd = {}
    for r in range(last + 1):
        ck = torch.load(f"{CKPT}.r{r}", map_location="cpu", weights_only=False)
        st = ck["stage"]
        if r == 0:
            sd["emb.weight"] = st["emb.weight"]
        for j in range(sizes[r]):
            g = bounds[r] + j
            for suf in ("n1.weight", "qkv.weight", "o.weight", "n2.weight",
                        "gate.weight", "up.weight", "down.weight"):
                sd[f"blocks.{g}.{suf}"] = st[f"blocks.{j}.{suf}"]
        if EXIT < 0 and r == K - 1:
            sd["ln.weight"] = st["ln.weight"]
            sd["fc.weight"] = st["fc.weight"]
        if EXIT == r < K - 1:                 # exit model: aux head blocks appended after the backbone
            for j in range(HEAD_DEPTH):
                g = bounds[r + 1] + j
                for suf in ("n1.weight", "qkv.weight", "o.weight", "n2.weight",
                            "gate.weight", "up.weight", "down.weight"):
                    sd[f"blocks.{g}.{suf}"] = st[f"hblocks.{j}.{suf}"]
            sd["ln.weight"] = st["hnorm.weight"]
            for _k in ("hfc.weight", "aux_fc.weight", "w_t"):
                if _k in st:
                    sd["fc.weight"] = st[_k]; print(f"  exit {r} readout <- {_k}"); break
            else:
                raise SystemExit(f"rank{r} has no readout key (hfc.weight / aux_fc.weight / w_t): {sorted(st)[:8]}...")
        print(f"  rank{r}: blocks {bounds[r]}..{bounds[r+1]-1} step={ck.get('step','?')}")
    if EXIT >= 0:
        NLAYER = bounds[EXIT + 1] + HEAD_DEPTH
        print(f"  exit {EXIT}: trunk {bounds[EXIT+1]} layers + head {HEAD_DEPTH} layers -> {NLAYER}-layer model")
else:
    raise SystemExit(f"SRC={SRC} unknown (bp|pipe)")
assert sd["emb.weight"].shape == (V, DMODEL), sd["emb.weight"].shape
assert f"blocks.{NLAYER-1}.qkv.weight" in sd, "layer count does not match NLAYER"

# ---------------------------------------------------------------- HF model + weight mapping
cfg = LlamaConfig(vocab_size=V, hidden_size=DMODEL, intermediate_size=FFN,
                  num_hidden_layers=NLAYER, num_attention_heads=NHEAD,
                  num_key_value_heads=NHEAD, max_position_embeddings=SEQ,
                  rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=False,
                  attention_bias=False, mlp_bias=False,
                  bos_token_id=1, eos_token_id=2)
hf = LlamaForCausalLM(cfg)
new = {"model.embed_tokens.weight": sd["emb.weight"],
       "model.norm.weight": sd["ln.weight"],
       "lm_head.weight": sd["fc.weight"]}
for i in range(NLAYER):
    qkv = sd[f"blocks.{i}.qkv.weight"]                     # (3C, C) row order q,k,v
    new[f"model.layers.{i}.self_attn.q_proj.weight"] = qkv[:DMODEL]
    new[f"model.layers.{i}.self_attn.k_proj.weight"] = qkv[DMODEL:2 * DMODEL]
    new[f"model.layers.{i}.self_attn.v_proj.weight"] = qkv[2 * DMODEL:]
    new[f"model.layers.{i}.self_attn.o_proj.weight"] = sd[f"blocks.{i}.o.weight"]
    new[f"model.layers.{i}.mlp.gate_proj.weight"] = sd[f"blocks.{i}.gate.weight"]
    new[f"model.layers.{i}.mlp.up_proj.weight"] = sd[f"blocks.{i}.up.weight"]
    new[f"model.layers.{i}.mlp.down_proj.weight"] = sd[f"blocks.{i}.down.weight"]
    new[f"model.layers.{i}.input_layernorm.weight"] = sd[f"blocks.{i}.n1.weight"]
    new[f"model.layers.{i}.post_attention_layernorm.weight"] = sd[f"blocks.{i}.n2.weight"]
miss, unexp = hf.load_state_dict(new, strict=False)
miss = [m for m in miss if "rotary" not in m and "inv_freq" not in m]
assert not miss and not unexp, (miss, unexp)
hf.eval()

# ---------------------------------------------------------------- cross-check: minimal replica of the training forward
class _RMS(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        x = x.float()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


_inv = 1.0 / (10000.0 ** (torch.arange(0, HD, 2).float() / HD))


def _rope(x):
    T = x.size(2)
    fr = torch.outer(torch.arange(T).float(), _inv)
    cos = torch.cat([fr.cos(), fr.cos()], -1)[None, None]
    sin = torch.cat([fr.sin(), fr.sin()], -1)[None, None]
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


def _ref_forward(x):
    h = F.embedding(x, sd["emb.weight"].float())
    for i in range(NLAYER):
        w = {k.split(f"blocks.{i}.")[1]: v.float() for k, v in sd.items()
             if k.startswith(f"blocks.{i}.")}
        B, T, C = h.shape
        n1 = h.float() * torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True) + 1e-6) \
            * w["n1.weight"]
        q, k_, v = F.linear(n1, w["qkv.weight"]).split(C, dim=2)
        q = _rope(q.view(B, T, NHEAD, HD).transpose(1, 2))
        k_ = _rope(k_.view(B, T, NHEAD, HD).transpose(1, 2))
        v = v.view(B, T, NHEAD, HD).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k_, v, is_causal=True)
        h = h + F.linear(a.transpose(1, 2).reshape(B, T, C), w["o.weight"])
        n2 = h.float() * torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True) + 1e-6) \
            * w["n2.weight"]
        h = h + F.linear(F.silu(F.linear(n2, w["gate.weight"]))
                         * F.linear(n2, w["up.weight"]), w["down.weight"])
    h = h.float() * torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True) + 1e-6) \
        * sd["ln.weight"].float()
    return F.linear(h, sd["fc.weight"].float())


torch.manual_seed(0)
x = torch.randint(0, V, (2, min(128, SEQ)))
with torch.no_grad():
    ours = _ref_forward(x)
    theirs = hf.float()(x).logits
d = (ours - theirs).abs().max().item()
rel = d / ours.abs().max().item()
print(f"parity: max|Δlogit|={d:.2e} (rel {rel:.2e})")
assert d < 2e-3, "parity check failed - convention mismatch, refusing to export"

# ---------------------------------------------------------------- write to disk
os.makedirs(OUTDIR, exist_ok=True)
hf.save_pretrained(OUTDIR, safe_serialization=True)
if not SKIP_TOK:
    AutoTokenizer.from_pretrained(TOK, token=os.environ.get("HF_TOKEN") or None) \
        .save_pretrained(OUTDIR)
n = sum(p.numel() for p in hf.parameters())
print(f"DONE: {OUTDIR} ({n/1e6:.1f}M params) - see the file header comment for eval commands")
