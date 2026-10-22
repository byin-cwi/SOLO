"""mb_scan.py - single-GPU scan over microbatch size MB: per-token throughput and MFU of one pipeline stage's trunk.

Same Block as pipep.py (nn.MultiheadAttention + GELU MLP + LayerNorm + dropout, bf16 autocast); only fwd+bwd of
NLAYER blocks is timed (no readout head, no optimizer).
Training FLOPs per token = 6 * 12 d^2 * NLAYER + 12 * NLAYER * d * T; MFU denominator = PEAK_TFLOPS (A100 bf16: 312).

Usage (one idle GPU, about 2-5 minutes):
  CUDA_VISIBLE_DEVICES=0 python mb_scan.py
  Options: NLAYERS="12 48" MBS="1 2 4 8 16 32" DMODEL=1024 SEQ=1024 DROP=0.1 ITERS=20 WARM=5 PEAK_TFLOPS=312
Column tok/s/MB4 = per-token throughput relative to MB=4; values clearly above 1 mean MB=4 does not saturate the GPU.
"""
import os
import time

import torch
import torch.nn as nn

E = os.environ.get
D = int(E("DMODEL", "1024"))
T = int(E("SEQ", "1024"))
NHEAD = int(E("NHEAD", str(max(1, D // 64))))
DROP = float(E("DROP", "0.1"))
ITERS = int(E("ITERS", "20"))
WARM = int(E("WARM", "5"))
PEAK = float(E("PEAK_TFLOPS", "312")) * 1e12
NLAYERS = [int(x) for x in E("NLAYERS", "12 48").split()]
MBS = [int(x) for x in E("MBS", "1 2 4 8 16 32").split()]
CUDA = torch.cuda.is_available()
dev = "cuda" if CUDA else "cpu"
AC = dict(device_type=dev, dtype=torch.bfloat16, enabled=CUDA)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, NHEAD, dropout=DROP, batch_first=True)
        self.ln2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D), nn.Dropout(DROP))

    def forward(self, x, m):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=m, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


def sync():
    if CUDA:
        torch.cuda.synchronize()


mask = torch.full((T, T), float("-inf"), device=dev).triu(1)
print(f"device={torch.cuda.get_device_name(0) if CUDA else 'cpu'} torch={torch.__version__} "
      f"d={D} T={T} heads={NHEAD} drop={DROP}")
print(f"{'layers':>6} {'MB':>4} {'ms/iter':>9} {'tok/s':>9} {'tok/s/MB4':>10} {'MFU':>7} {'peak GB':>8}")
for L in NLAYERS:
    model = nn.ModuleList([Block() for _ in range(L)]).to(dev)
    flops_tok = 6 * 12 * D * D * L + 12 * L * D * T
    base = None
    for mb in MBS:
        try:
            if CUDA:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            x = torch.randn(mb, T, D, device=dev, requires_grad=True)
            for i in range(WARM + ITERS):
                if i == WARM:
                    sync()
                    t0 = time.time()
                with torch.autocast(**AC):
                    h = x
                    for b in model:
                        h = b(h, mask)
                h.float().pow(2).mean().backward()
                x.grad = None
                for q in model.parameters():
                    q.grad = None
            sync()
            dt = (time.time() - t0) / ITERS
        except RuntimeError as e:                    # skip larger MB when out of memory
            if "out of memory" not in str(e):
                raise
            print(f"{L:>6} {mb:>4}   OOM")
            del x
            break
        tok = mb * T / dt
        if mb == 4:
            base = tok
        mfu = tok * flops_tok / PEAK
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if CUDA else 0.0
        rel = f"{tok / base:.3f}" if base else "-"
        print(f"{L:>6} {mb:>4} {1000 * dt:>9.1f} {tok:>9.0f} {rel:>10} {mfu:>7.1%} {peak:>8.2f}")
        del x
    del model
print("tok/s/MB4: per-token throughput relative to MB=4 (rows with MB<4 show '-' until MB=4 is measured).")
