"""bp_to_pipe.py - split a BP ckpt (slim_bp format) into pipeline per-stage ckpts.
Backbone blocks are assigned per stage; rank0 holds emb; the last stage holds ln/fc; aux heads of intermediate
stages are freshly initialized (N(0,0.02)) and w_t is a copy of the terminal fc. Output dict(stage=..., step=...)
has no optimizer state; slim_pipeline_solo.py treats it as a weights-only warm start.
STEP: -1 (default) inherits the step count of the BP ckpt; N >= 0 overrides it (0 = fresh schedule).
Usage: SRC=ckpt/bp15b.pt K=2 NLAYER=24 DMODEL=1024 NHEAD=16 FFN=0 HEAD_DEPTH=2 STEP=-1 OUT=ckpt/bp_pipe.pt python bp_to_pipe.py"""
import os
import torch

E = lambda k, d: os.environ.get(k, d)
SRC, OUT = E("SRC", ""), E("OUT", "")
K = int(E("K", "2"))
NL = int(E("NLAYER", "24"))
DM = int(E("DMODEL", "1024"))
NH = int(E("NHEAD", "16"))
FFN = int(E("FFN", "0"))
HD = int(E("HEAD_DEPTH", "2"))
SEED = int(E("SEED", "42"))
if FFN == 0:
    FFN = ((int(4 * DM * 2 / 3) + 255) // 256) * 256
assert SRC and OUT
ck = torch.load(SRC, map_location="cpu")
sd, step = ck["model"], ck["step"]
_ov = int(E("STEP", "-1"))
if _ov >= 0:
    step = _ov
V = sd["fc.weight"].shape[0]
per = NL // K
sizes = [per] * (K - 1) + [NL - per * (K - 1)]
bounds = [0]
for z in sizes:
    bounds.append(bounds[-1] + z)
g = torch.Generator().manual_seed(SEED)


def fresh_block(pref, out):
    """Fresh LlamaBlock params (n1/qkv/o/n2/gate/up/down, N(0,0.02), norm weights 1)."""
    out[pref + "n1.weight"] = torch.ones(DM)
    out[pref + "n2.weight"] = torch.ones(DM)
    for nm, shp in (("qkv.weight", (3 * DM, DM)), ("o.weight", (DM, DM)),
                    ("gate.weight", (FFN, DM)), ("up.weight", (FFN, DM)), ("down.weight", (DM, FFN))):
        out[pref + nm] = torch.randn(shp, generator=g) * 0.02


for r in range(K):
    st = {}
    if r == 0:
        st["emb.weight"] = sd["emb.weight"]
    for j in range(sizes[r]):
        gl = bounds[r] + j
        for suf in ("n1.weight", "qkv.weight", "o.weight", "n2.weight",
                    "gate.weight", "up.weight", "down.weight"):
            st[f"blocks.{j}.{suf}"] = sd[f"blocks.{gl}.{suf}"]
    if r < K - 1:                              # intermediate stage: fresh aux head + copy of the tied readout
        for h in range(HD):
            fresh_block(f"hblocks.{h}.", st)
        st["hnorm.weight"] = torch.ones(DM)
        st["w_t"] = sd["fc.weight"].clone()
    else:
        st["ln.weight"] = sd["ln.weight"]
        st["fc.weight"] = sd["fc.weight"]
    torch.save(dict(stage=st, step=step), f"{OUT}.r{r}.tmp")   # two-phase write: all .tmp first
    n = sum(v.numel() for v in st.values())
    print(f"  rank{r}: blocks {bounds[r]}..{bounds[r+1]-1} +{'fresh head' if r < K-1 else 'ln/fc'} "
          f"({n/1e6:.1f}M) step={step}")
for r in range(K):                             # then rename, never leaving a partial set of .r files
    os.replace(f"{OUT}.r{r}.tmp", f"{OUT}.r{r}")
print(f"DONE: {OUT}.r0..r{K-1} (V={V}, ffn={FFN})")
