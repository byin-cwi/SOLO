"""bench_parts.py - time the SOLO training step part by part: cost constants b (one Block fwd+bwd) and
r (one readout GEMM+CE fwd+bwd), per-position subsampling of the aux readout, and CE implementation variants.

Self-contained (no data, does not import lm_solo). Default shapes follow ../lm_wt103/run_wt103_align_ladder.sh.
Usage: python bench_parts.py   /   D=768 V=50257 python bench_parts.py
env: D V BS SEQ NHEAD FFN L K HEAD_DEPTH ITER WARM RHOS FCHUNKS
"""
import os, time, math
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch, torch.nn as nn, torch.nn.functional as F

E = os.environ.get
D = int(E("D", "384")); V = int(E("V", "32768")); BS = int(E("BS", "64")); T = int(E("SEQ", "256"))
NHEAD = int(E("NHEAD", "6")); FFN = int(E("FFN", "0")) or 4 * D
L = int(E("L", "8")); K = int(E("K", "4")); HD = int(E("HEAD_DEPTH", "2"))
ITER = int(E("ITER", "20")); WARM = int(E("WARM", "5"))
RHOS = [float(v) for v in E("RHOS", "1.0,0.5,0.25,0.1").split(",")]
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
n = BS * T
AC = dict(device_type=dev, dtype=torch.bfloat16, enabled=dev == "cuda")


class Block(nn.Module):
    """Same architecture as lm_solo.Block."""
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, NHEAD, dropout=0.1, batch_first=True)
        self.ln2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(nn.Linear(D, FFN), nn.GELU(), nn.Linear(FFN, D), nn.Dropout(0.1))

    def forward(self, x, m):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=m, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


mask = torch.full((T, T), float("-inf"), device=dev).triu(1)
blk = Block().to(dev)
W = torch.randn(V, D, device=dev) * (D ** -0.5)        # shared readout (TIE_AUX): aux heads read its detached copy
Wt = nn.Parameter(W.clone())                            # terminal readout: trainable
y = torch.randint(0, V, (n,), device=dev)


def timeit(fn, label):
    for _ in range(WARM):
        fn()
    if dev == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(ITER):
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time(); fn()
        if dev == "cuda":
            torch.cuda.synchronize()
        ts.append(time.time() - t0)
    ts.sort()
    ms = ts[len(ts) // 2] * 1e3
    mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev == "cuda" else 0.0
    print(f"  {label:<34} {ms:8.2f} ms   peak {mem:5.2f} GB")
    return ms


def fresh_h(grad=True):
    return torch.randn(BS, T, D, device=dev, requires_grad=grad)


def f_block():
    h = fresh_h()
    with torch.autocast(**AC):
        o = blk(h, mask)
    o.sum().backward()
    blk.zero_grad(set_to_none=True)


def f_gemm():                       # forward GEMM only, no CE, no backward => pure matmul lower bound
    h = fresh_h(False)
    with torch.autocast(**AC):
        F.linear(h, W).reshape(-1, V)


def f_ce_only():                    # CE fwd+bwd with given logits => separates CE from GEMM
    z = torch.randn(n, V, device=dev, dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast(**AC):
        l = F.cross_entropy(z, y)
    l.backward()


def f_readout_aux(rho=1.0):         # aux readout: W detached => backward reaches only h
    h = fresh_h()
    with torch.autocast(**AC):
        hh = h.reshape(-1, D)
        if rho < 1.0:
            idx = torch.randperm(n, device=dev)[: max(1, int(rho * n))]
            hh, yy = hh[idx], y[idx]
        else:
            yy = y
        l = F.cross_entropy(F.linear(hh, W), yy)
    l.backward()


def f_readout_term():               # terminal readout: W trainable => backward reaches h and W
    h = fresh_h()
    with torch.autocast(**AC):
        l = F.cross_entropy(F.linear(h.reshape(-1, D), Wt), y)
    l.backward()
    Wt.grad = None


print(f"[cfg] dev={dev} d={D} V={V} bs={BS} seq={T} n={n} L={L} K={K} head_depth={HD} "
      f"iter={ITER}", flush=True)
if dev == "cuda":
    print(f"[gpu] {torch.cuda.get_device_name(0)}", flush=True)
print("Components:")
b = timeit(f_block, "Block fwd+bwd")
g = timeit(f_gemm, "readout GEMM fwd only")
c = timeit(f_ce_only, "CE fwd+bwd (logits given)")
ra = timeit(f_readout_aux, "aux readout GEMM+CE fwd+bwd (W detach)")
rt = timeit(f_readout_term, "term readout GEMM+CE fwd+bwd (W trainable)")
print("P2-a subsampled aux readout (unbiased: sampled by position):")
sub = {}
for rho in RHOS:
    sub[rho] = timeit(lambda r=rho: f_readout_aux(r), f"  rho={rho:<5}")

# ------------------------------------------------------------------ CE variants: time + gradient accuracy
print("\nCE variants (logits given, fwd+bwd; accuracy vs fp32 reference):")
z0 = torch.randn(n, V, device=dev, dtype=torch.bfloat16)
zr = z0.float().detach().requires_grad_(True)
F.cross_entropy(zr, y).backward()
GREF = zr.grad.clone()
LREF = float(F.cross_entropy(z0.float(), y))
del zr
if dev == "cuda":
    torch.cuda.empty_cache()


def _ce_acc(fn):
    """Run once; return (loss relative error, grad relative error) vs the fp32 reference."""
    z = z0.detach().requires_grad_(True)
    l = fn(z, y)
    l.backward()
    dl = abs(float(l) - LREF) / max(abs(LREF), 1e-30)
    dg = (z.grad.float() - GREF).abs().max().item() / max(GREF.abs().max().item(), 1e-30)
    return dl, dg


def _ce_autocast(z, yy):
    with torch.autocast(**AC):
        return F.cross_entropy(z, yy)


CE_VARIANTS = {"autocast fp32 (current)": _ce_autocast}
CE_VARIANTS["bf16 direct (no upcast)"] = lambda z, yy: F.cross_entropy(z, yy)
try:
    _cc = torch.compile(lambda z, yy: F.cross_entropy(z, yy))
    _cc(z0[:8].detach().requires_grad_(True), y[:8]).backward()   # trigger compilation
    CE_VARIANTS["torch.compile CE"] = _cc
except Exception as e:                                            # noqa: BLE001
    print(f"  [skip] torch.compile unavailable: {type(e).__name__}: {str(e)[:70]}")

for name, fn in CE_VARIANTS.items():
    dl, dg = _ce_acc(fn)
    def _run(f=fn):
        z = z0.detach().requires_grad_(True)
        f(z, y).backward()
    ms = timeit(_run, f"{name}")
    print(f"  {'':<34} {'':>8}      loss rel={dl:.2e}  grad rel={dg:.2e}")

# ------------------------------------------------------------------ readout level: the three CE_MODE options of lm_solo.py
print("\nThree CE implementations for the aux readout (linear+CE fwd+bwd, W detach):")
_h0 = torch.randn(BS, T, D, device=dev)


def _ro(mode):
    def f():
        h = _h0.detach().requires_grad_(True)
        with torch.autocast(**AC):
            z = F.linear(h.reshape(-1, D), W)          # bf16 GEMM (autocast)
            if mode == "autocast":
                l = F.cross_entropy(z, y)
            else:
                with torch.autocast(device_type=dev, enabled=False):
                    l = (F.cross_entropy if mode == "bf16" else _CE_C)(z, y)
        l.backward()
    return f


_CE_C = None
RO = {"autocast (current)": "autocast", "bf16 direct": "bf16"}
try:
    _CE_C = torch.compile(lambda z, yy: F.cross_entropy(z, yy))
    _ro("compile")()                                    # trigger compilation
    RO["torch.compile"] = "compile"
except Exception as e:                                  # noqa: BLE001
    print(f"  [skip] torch.compile unavailable: {type(e).__name__}: {str(e)[:70]}")
ro_ms = {}
for nm, md in RO.items():
    ro_ms[nm] = timeit(_ro(md), nm)

# ------------------------------------------------------------------ fused linear+CE (fused_ce.py)
print("\nfused linear+CE (aux head, W detach) [rejected: loses to torch.compile, kept for record only]:")
try:
    from fused_ce import fused_linear_ce
    hr2 = torch.randn(n, D, device=dev)
    hf = hr2.float().detach().requires_grad_(True)      # fp32 reference (outside autocast)
    F.cross_entropy(F.linear(hf, W.float()), y).backward()
    GH_REF = hf.grad.clone()
    for chunk in [int(v) for v in E("FCHUNKS", "8192,4096,2048").split(",")]:
        for sz in (1, 0):
            h = hr2.detach().requires_grad_(True)
            with torch.autocast(**AC):                  # call under autocast, otherwise the GEMM runs in fp32
                ell = fused_linear_ce(h, W, None, y, chunk=chunk, save_z=sz)
            ell.mean().backward()
            dg = (h.grad - GH_REF).abs().max().item() / max(GH_REF.abs().max().item(), 1e-30)
            def _run(c=chunk, s=sz):
                hh = hr2.detach().requires_grad_(True)
                with torch.autocast(**AC):
                    e_ = fused_linear_ce(hh, W, None, y, chunk=c, save_z=s)
                e_.mean().backward()
            ms = timeit(_run, f"  chunk={chunk} save_z={sz}")
            print(f"  {'':<34} {'':>8}      grad_h rel={dg:.2e}   (reference: aux readout {ra:.2f} ms)")
except Exception as e:                                            # noqa: BLE001
    print(f"  [skip] {type(e).__name__}: {e}")

print("\nBreakdown:")
print(f"  GEMM fwd share of aux readout {100*g/ra:5.1f}%   (CE and its backward = {ra-g:.2f} ms)")
print(f"  b = {b:.2f} ms/Block,  r_aux = {ra:.2f} ms,  r_term = {rt:.2f} ms,  r_aux/b = {ra/b:.1f}")
nb = L + (K - 1) * HD
pred = nb * b + (K - 1) * ra + rt
print(f"\npred solo_fast = {nb}·b + {K-1}·r_aux + r_term = {pred:.1f} ms")
print(f"pred bp        = {L}·b + r_term                = {L*b+rt:.1f} ms")
for rho in RHOS:
    if rho < 1.0:
        p = nb * b + (K - 1) * sub[rho] + rt
        print(f"pred solo_fast + P2-a(rho={rho}) = {p:.1f} ms   (saves {pred-p:.1f} ms, "
              f"{100*(pred-p)/pred:.0f}%)")
if len(ro_ms) > 1:
    base = ro_ms["autocast (current)"]
    print()
    for nm, ms in ro_ms.items():
        if nm == "autocast (current)":
            continue
        d = base - ms                                    # saved per readout
        print(f"pred CE_MODE={'bf16' if 'bf16' in nm else 'compile'}: "
              f"solo_fast {pred - 4*d:6.1f} ms   bp {L*b+rt - d:5.1f} ms   "
              f"ratio {(pred-4*d)/(L*b+rt-d):.2f}x   (per-readout saving {d:.1f} ms x 4)")
print("\nPredictions assume additive components (no kernel fusion or launch overhead).")
