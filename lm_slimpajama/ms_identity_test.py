"""Identity test for the milestone loss: explicit reference form (explicit off matrix + two log_softmax) vs the
contracted form (row-constant invariance + logsumexp correction + Hellinger reduction), for values and gradients
w.r.t. z. fp64 should agree to ~1e-12; fp32 shows the rounding magnitude."""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
V, N = 3001, 64
MU, AUX = 0.3, 0.3


def v4_loss(z, y, lam, t):
    with torch.no_grad():
        log_self = F.log_softmax(z.detach(), -1)
        off = torch.log((1 - torch.exp(-lam)) / (V - 1)).unsqueeze(1).expand(N, V).clone()
        off.scatter_(1, y[:, None], -lam[:, None])
        log_m = F.log_softmax((1 - t) * log_self + t * off, -1)
        ell_m = -log_m.gather(1, y[:, None]).squeeze(1)
    logq = F.log_softmax(z, -1)
    ell_q = -logq.gather(1, y[:, None]).squeeze(1)
    active = (ell_q > ell_m).float().detach()
    hinge = (ell_q - ell_m).clamp_min(0).sum()
    hell = (1 - (0.5 * (logq + log_m)).exp().sum(-1)).clamp_min(0)
    return hinge + MU * (hell * active).sum() + AUX * ell_q.sum(), ell_m, hell, active


def v5_loss(z, y, lam, t):
    logq = F.log_softmax(z, -1)                                   # the only log_softmax
    ls = logq.detach()
    with torch.no_grad():
        c = torch.log((1 - torch.exp(-lam)) / (V - 1))
        delta = t * (-lam - c)                                     # offset of the target column relative to the row constant
        s_y = (1 - t) * ls.gather(1, y[:, None]).squeeze(1)
        S = torch.logsumexp((1 - t) * ls, -1)                      # reduction 1
        d = s_y - S                                                # <= 0
        LSE = S + torch.log(torch.exp(d + delta) - torch.expm1(d))
        ell_m = LSE - s_y - delta
    ell_q = -logq.gather(1, y[:, None]).squeeze(1)
    active = (ell_q > ell_m).float()
    hinge = (ell_q - ell_m).clamp_min(0).sum()
    # Hellinger: sum_j exp(.5 logq_j + .5 log_m_j) = e^{-LSE/2} [ G + g_y (e^{delta/2}-1) ],
    # G = sum_j exp(u_j), u_j = .5 logq_j + .5 (1-t) ls_j; gradient flows only through .5 logq_j.
    a = 1 - t / 2                                                  # numerically u_j = a*logq_j
    alpha = 0.5 / a                                                # value unchanged, gradient x alpha*a = .5
    Gc = torch.exp(a * logq).sum(-1)                               # reduction 2 (with gradient)
    G = alpha * Gc + (1 - alpha) * Gc.detach()
    gyc = torch.exp(a * ell_q.neg())                               # exp(a*logq_y)
    g_y = alpha * gyc + (1 - alpha) * gyc.detach()
    H = torch.exp(-0.5 * LSE) * (G + g_y * torch.expm1(0.5 * delta))
    hell = (1 - H).clamp_min(0)
    return hinge + MU * (hell * active).sum() + AUX * ell_q.sum(), ell_m, hell, active


for dt in (torch.float64, torch.float32):
    for t in (0.25, 0.5, 0.75):
        z = (torch.randn(N, V) * 3).to(dt)
        # extreme rows: confidently correct / confidently wrong
        z[0, 5] += 30; z[1, 7] += 30
        y = torch.randint(0, V, (N,)); y[0] = 5; y[1] = 9
        lam = (torch.rand(N) * 12 + 0.1).to(dt)   # cover the large-lam regime where delta < 0
        za = z.clone().requires_grad_(True); zb = z.clone().requires_grad_(True)
        L4, m4, h4, a4 = v4_loss(za, y, lam, t)
        L5, m5, h5, a5 = v5_loss(zb, y, lam, t)
        L4.backward(); L5.backward()
        rel = lambda p, q: ((p - q).abs().max() / (q.abs().max() + 1e-30)).item()
        print(f"{str(dt):15s} t={t}: loss rel {rel(L5, L4):.2e} | ell_m {rel(m5, m4):.2e} | "
              f"hell {rel(h5, h4):.2e} | active eq {(a4 == a5).all().item()} | "
              f"grad rel {rel(zb.grad, za.grad):.2e}")
