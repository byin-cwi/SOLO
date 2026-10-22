"""fused_ce.py - row-chunked linear+CE for aux heads that read a detached shared readout.

  ell_i = logsumexp_v (h_i W^T + b)_v - (h_i W^T + b)_{y_i}
Rows are independent, so row chunking is exact (each row's logsumexp covers the full V); only the summation order differs.
Logits stay in bf16 (F.cross_entropy under autocast upcasts them to fp32); only the logsumexp accumulation is fp32.
Backward rewrites z in place into q = softmax(z), then one GEMM gives grad_h; grad_W is skipped when W needs no gradient.

  SAVE_Z=1 (default): forward saves z (bf16, n*V*2 bytes), backward does not recompute => 2 GEMMs.
  SAVE_Z=0          : backward recomputes z => 3 GEMMs, peak memory only chunk*V.
  FCHUNK (default 4096): rows per chunk.

Self-test: python fused_ce.py   (compares loss and grad_h against the fp32 reference)
"""
import os
import torch
import torch.nn.functional as F


class _FusedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, W, b, y, chunk, save_z):
        # backward runs outside the autocast context: cast W/b to h's dtype once and reuse the copy in backward.
        if W.dtype != h.dtype:
            W = W.to(h.dtype)
        if b is not None and b.dtype != h.dtype:
            b = b.to(h.dtype)
        n = h.shape[0]
        ell = torch.empty(n, device=h.device, dtype=torch.float32)
        lse = torch.empty(n, device=h.device, dtype=torch.float32)
        zs = [] if save_z else None
        with torch.no_grad():
            for i in range(0, n, chunk):
                j = min(i + chunk, n)
                z = F.linear(h[i:j], W, b)                      # (c,V) bf16
                m = z.amax(-1, keepdim=True)
                s = (z - m).exp().sum(-1, dtype=torch.float32)  # fp32 accumulation
                l = m.squeeze(-1).float() + s.log()
                lse[i:j] = l
                ell[i:j] = l - z.gather(1, y[i:j, None]).squeeze(1).float()
                if save_z:
                    zs.append(z)
        ctx.save_for_backward(h, W, y, lse)
        ctx.b, ctx.chunk, ctx.zs = b, chunk, zs
        return ell

    @staticmethod
    def backward(ctx, g):
        h, W, y, lse = ctx.saved_tensors
        b, chunk, zs = ctx.b, ctx.chunk, ctx.zs
        n = h.shape[0]
        need_h, need_W = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        gh = torch.empty_like(h) if need_h else None
        gW = torch.zeros(W.shape, device=W.device, dtype=torch.float32) if need_W else None
        for c, i in enumerate(range(0, n, chunk)):
            j = min(i + chunk, n)
            z = zs[c] if zs is not None else F.linear(h[i:j], W, b)
            z = z.sub_(lse[i:j, None].to(z.dtype)).exp_()       # in place: z -> q = softmax(z)
            z.scatter_add_(1, y[i:j, None], torch.full_like(z[:, :1], -1.0))
            z.mul_(g[i:j, None].to(z.dtype))
            if need_h:
                gh[i:j] = (z @ W).to(gh.dtype)
            if need_W:
                gW.addmm_(z.t().float(), h[i:j].float())
            if zs is not None:
                zs[c] = None
        ctx.zs = None
        return gh, (gW.to(W.dtype) if gW is not None else None), None, None, None, None


def fused_linear_ce(h, W, b, y, chunk=None, save_z=None):
    """Per-position CE, shape (n,) fp32, differentiable w.r.t. h. h: (n,d)  W: (V,d)  y: (n,)
    The grad_W path uses an fp32 GEMM, so a trainable (terminal) head should use F.cross_entropy instead."""
    chunk = int(os.environ.get("FCHUNK", "4096")) if chunk is None else chunk
    save_z = int(os.environ.get("SAVE_Z", "1")) if save_z is None else int(save_z)
    # Replicate autocast for linear by hand (nn.LayerNorm outputs fp32 under autocast): cast h to the autocast
    # dtype and W follows in forward; otherwise the GEMM runs in fp32/TF32.
    if h.is_cuda and torch.is_autocast_enabled():
        try:
            dt = torch.get_autocast_dtype("cuda")
        except (AttributeError, TypeError):
            dt = torch.get_autocast_gpu_dtype()
        if h.dtype != dt:
            h = h.to(dt)
    return _FusedLinearCE.apply(h, W, b, y, chunk, bool(save_z))


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n, d, V = 2048, 128, 1024
    torch.manual_seed(0)
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    h0 = torch.randn(n, d, device=dev, dtype=dt) * 0.5
    W0 = torch.randn(V, d, device=dev, dtype=dt) * d ** -0.5
    b0 = torch.randn(V, device=dev, dtype=dt) * 0.1
    y = torch.randint(0, V, (n,), device=dev)
    for save_z in (1, 0):
        h = h0.clone().requires_grad_(True)
        ell = fused_linear_ce(h, W0, b0, y, chunk=512, save_z=save_z)
        ell.mean().backward()
        hr = h0.float().clone().requires_grad_(True)
        er = F.cross_entropy(F.linear(hr, W0.float(), b0.float()), y, reduction="none")
        er.mean().backward()
        de = (ell.float() - er).abs().max().item() / max(er.abs().max().item(), 1e-30)
        dg = (h.grad.float() - hr.grad).abs().max().item() / max(hr.grad.abs().max().item(), 1e-30)
        print(f"save_z={save_z}  ell rel={de:.3e}   grad_h rel={dg:.3e}   "
              f"(dtype={dt}, reference is fp32)")
