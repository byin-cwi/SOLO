"""hybrid_convert.py - assemble pipeline per-stage ckpts into a slim_bp-format warm-start ckpt (weights + step only).
rank0 emb + all blocks in global layer order + last-stage ln/fc; aux heads are dropped. Output dict(model=..., step=...);
slim_bp.py RESUME treats a ckpt without opt/sched as a warm start (fresh optimizer state, scheduler fast-forward).
Usage: SRC=ckpt/xxx.pt K=2 [STEP=N] OUT=ckpt/hybrid_ws.pt python hybrid_convert.py
env: STEP (default: step stored in the ckpt), NLAYER=24, SPLIT="" (must match training)"""
import os
import torch

E = lambda k, d: os.environ.get(k, d)
SRC = E("SRC", "")
OUT = E("OUT", "")
K = int(E("K", "4"))
NLAYER = int(E("NLAYER", "24"))
SPLIT = E("SPLIT", "")
assert SRC and OUT
if SPLIT:
    sizes = [int(v) for v in SPLIT.split(",")]
    assert sum(sizes) == NLAYER and len(sizes) == K
else:
    per = NLAYER // K
    sizes = [per] * (K - 1) + [NLAYER - per * (K - 1)]
bounds = [0]
for z in sizes:
    bounds.append(bounds[-1] + z)
sd, step = {}, None
for r in range(K):
    ck = torch.load(f"{SRC}.r{r}", map_location="cpu")
    st, step = ck["stage"], ck["step"]
    if r == 0:
        sd["emb.weight"] = st["emb.weight"]
    for j in range(sizes[r]):
        g = bounds[r] + j
        for suf in ("n1.weight", "qkv.weight", "o.weight", "n2.weight",
                    "gate.weight", "up.weight", "down.weight"):
            sd[f"blocks.{g}.{suf}"] = st[f"blocks.{j}.{suf}"]
    if r == K - 1:
        sd["ln.weight"] = st["ln.weight"]
        sd["fc.weight"] = st["fc.weight"]
    print(f"  rank{r}: blocks {bounds[r]}..{bounds[r+1]-1} step={step}")
step = int(E("STEP", str(step)))
n = sum(v.numel() for v in sd.values())
torch.save(dict(model=sd, step=step), OUT)
print(f"DONE: {OUT} ({n/1e6:.1f}M params, warm-start @ step {step}, no opt/sched)")
