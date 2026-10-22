"""rho.py - closed-form model of the solo/BP work ratio, and the throughput-win condition under multi-GPU pipelining.

Counted in units of one forward (b = 12d^2+2Td is the per-token MACs of one Block, r = Vd is one full-vocabulary readout):
    Block fwd+bwd        = 3b
    aux readout fwd+bwd      = 2r   (shared readout is detached, no grad_W)
    terminal readout fwd+bwd = 3r
    W_BP   = 3Lb + 3r
    W_solo = 3(L + (K-1)H)b + (2K+1)r
Dividing by 3Ld and letting gamma = V / (L(12d+2T)) (weight of the readout relative to the trunk):

    rho = [1 + (K-1)H/L + (2K+1)gamma/3] / (1 + gamma)

As gamma -> 0 (deep/wide model, V fixed), rho -> 1 + (K-1)H/L.

Multi-GPU: p=K-stage pipeline; BP with 1F1B (m microbatches) has bubble beta=(K-1)/(m+K-1); solo streams forward with no bubble.
    solo wins  <=>  rho < 1/(1-beta) = 1 + (K-1)/m
As gamma->0 the (K-1) cancels on both sides, reducing to the K-independent  m < L/H.

The model counts FLOPs only; a bandwidth-bound readout (e.g. CE_MODE=compile) makes measured rho lower.

Usage: python rho.py                      # default phase diagram
       python rho.py L=24 d=1024 T=1024 V=32768 K=2 H=2 m=8
"""
import sys


def gamma(L, d, T, V):
    return V / (L * (12 * d + 2 * T))


def rho(L, d, T, V, K, H):
    g = gamma(L, d, T, V)
    return (1 + (K - 1) * H / L + (2 * K + 1) * g / 3) / (1 + g)


def head_cost_factor(rho_s, d, T):
    """Cost factor of a single-block aux head at query sparsity rate rho_s (SPARSE_Q).
    q/o projections and the MLP scale with rho_s; k,v projections do not (keys/values still cover the full sequence); attention scales with rho_s.
        c = [(2 + 10*rho_s) d^2 + 2*rho_s*T*d] / (12 d^2 + 2 T d)
    """
    return ((2 + 10 * rho_s) * d * d + 2 * rho_s * T * d) / (12 * d * d + 2 * T * d)


def rho_sparse(L, d, T, V, K, H, rho_s):
    """rho under query sparsity. For H>=2 only the last block is sparse (the first H-1 blocks run dense to stay exact per position);
    the aux readout scales exactly linearly with rho_s. With rho_s=1 this reduces to rho()."""
    g = gamma(L, d, T, V)
    head = (K - 1) * ((H - 1) + head_cost_factor(rho_s, d, T)) / L if H >= 1 else 0.0
    return (1 + head + (3 + 2 * (K - 1) * rho_s) * g / 3) / (1 + g)


def flops_per_token(L, d, T, V, K=1, H=0, solo=False):
    """Training FLOPs per token, split into trunk / aux blocks / readout (1 MAC = 2 FLOPs).
    Embedding lookup counts no FLOPs. K and H are ignored when solo=False."""
    b, r = 12 * d * d + 2 * T * d, V * d
    trunk = 3 * L * b
    aux_blocks = 3 * (K - 1) * H * b if solo else 0
    readout = ((2 * K + 1) if solo else 3) * r
    return dict(trunk=2 * trunk, aux=2 * aux_blocks, readout=2 * readout,
                total=2 * (trunk + aux_blocks + readout))


def mfu(tok_per_s, flops_tok, n_gpu, peak_tflops=312.0):
    """Model FLOPs utilization. tok_per_s is the total throughput over all GPUs."""
    return tok_per_s * flops_tok / (n_gpu * peak_tflops * 1e12)


def flops_table(configs):
    """configs: (name, L, d, T, V, K, H) -> print a breakdown table of per-token FLOPs and rho."""
    print(f"{'config':<22} {'GFLOPs/tok BP':>13} {'trunk':>7} {'readout':>8} "
          f"{'GFLOPs/tok SOLO':>15} {'aux blk':>8} {'readout':>8} {'rho':>6}")
    for name, L, d, T, V, K, H in configs:
        bp = flops_per_token(L, d, T, V)
        so = flops_per_token(L, d, T, V, K, H, solo=True)
        print(f"{name:<22} {bp['total'] / 1e9:>13.2f} {bp['trunk'] / 1e9:>7.2f} "
              f"{bp['readout'] / 1e9:>8.3f} {so['total'] / 1e9:>15.2f} {so['aux'] / 1e9:>8.2f} "
              f"{so['readout'] / 1e9:>8.3f} {so['total'] / bp['total']:>6.3f}")


def threshold(K, m):
    """Tolerance the 1F1B bubble gives solo: solo wins <=> rho < 1 + (K-1)/m."""
    return 1 + (K - 1) / m


def main():
    kw = dict(L=24, d=1024, T=1024, V=32768, K=2, H=2, m=8)
    for a in sys.argv[1:]:
        k, _, v = a.partition("=")
        assert k in kw, f"unknown argument {k}; available: {list(kw)}"
        kw[k] = int(v)
    L, d, T, V, K, H, m = (kw[x] for x in ("L", "d", "T", "V", "K", "H", "m"))
    g = gamma(L, d, T, V)
    r = rho(L, d, T, V, K, H)
    t = threshold(K, m)
    print(f"L={L} d={d} T={T} V={V} K={K} H={H} m={m}")
    print(f"  gamma = {g:.4f}   (readout/trunk)")
    print(f"  rho   = {r:.4f}   = [1 + {(K-1)*H/L:.4f}(aux heads) + {(2*K+1)*g/3:.4f}(readout)] / {1+g:.4f}")
    print(f"  thresh= {t:.4f}   (1F1B m={m} bubble tolerance)   =>  {'solo wins' if r < t else 'BP wins'} "
          f"{abs(r-t)/t*100:.1f}%")
    print(f"  asymptote (gamma->0): rho -> {1 + (K-1)*H/L:.4f};  criterion m < L/H = {L/max(H,1e-9):.1f}"
          if H else "  asymptote (gamma->0): rho -> 1.0000 (H=0: no aux-head cost)")
    print(f"\n  SPARSE_Q ladder @ H={H}:")
    print("    rho_s   c(rho_s)     rho    m*")
    for rs in (1.0, 0.5, 0.25, 0.125):
        rr = rho_sparse(L, d, T, V, K, H, rs)
        ms_ = (K - 1) / (rr - 1) if rr > 1 else float("inf")
        print(f"    {rs:<7} {head_cost_factor(rs, d, T):<10.3f} {rr:<7.4f} {ms_:>5.1f}")
    print(f"\n(H, m) phase diagram @ L={L} d={d} T={T} V={V} K={K}   [✓ = solo wins, value = rho]")
    ms = [2, 4, 8, 16, 32]
    print("  H \\ m  " + "".join(f"{x:>9}" for x in ms) + "     rho")
    for h in (0, 1, 2, 3):
        rh = rho(L, d, T, V, K, h)
        cells = "".join(f"{('✓' if rh < threshold(K, x) else '✗'):>9}" for x in ms)
        print(f"  {h:<6}" + cells + f"  {rh:>8.4f}")
    print("  thresh" + "".join(f"{threshold(K, x):>9.4f}" for x in ms))


if __name__ == "__main__":
    main()
