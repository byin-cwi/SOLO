"""align_probe.py - cosine between local gradients and BP gradients, per module boundary, for any number of arms.

    parameter space   cos_par(k) = cos(grad_{theta_k} l_k, grad_{theta_k} L), theta_k = trunk params of module k
                                   (k=0 includes emb/pos); one scalar per batch
    activation space  cos_act(k) = cos(grad_{h_k} l_k, grad_{h_k} L), per position
l_k = local CE of the module-k readout, L = terminal CE of the assembled network.

Arms: ARMS + CKPT_FMT. The readout type is inferred from the checkpoint:
    w_t present         -> tied readout (tie / rshare / rand / rot)
    hfc.weight present  -> private readout (untie / priv)
    rot present         -> fixed orthogonal R_k applied before the readout (TIE_MODE=rot of lm_hz_rot.py)
All arms must share model size and HD (asserted). Paired differences are relative to REF (default: first arm).

Env (defaults): ARMS="tie rot rshare rand", CKPT_FMT=ckpt_2x2/wt103_40m_hd0_{arm}.pt, REF=<first arm>,
  DATA=./data/wikitext, DSET=wikitext-103-raw-v1, VMAX=32768 (reads word_cache_<DSET>_<VMAX>.pt), NHEAD=6, K=4,
  BS=16, NBATCH=32, NBOOT=1000 (bootstrap resamples), TAG=align, SEED=0, FAKE_DATA=0 (1 = random tokens), DEV.

Usage:
  ARMS="tie rot rshare rand" CKPT_FMT="ckpt_2x2/wt103_40m_hd0_{arm}.pt" \
  DATA=./data/wikitext DSET=wikitext-103-raw-v1 VMAX=32768 NHEAD=6 K=4 BS=16 NBATCH=32 \
  TAG=2x2_hd0 python -u align_probe.py
Smoke test (no data needed):  FAKE_DATA=1 ... python -u align_probe.py
Output: align_<TAG>.json and tables on stdout.
"""
import glob, json, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

E = os.environ.get
DEV = E("DEV", "cuda:0" if torch.cuda.is_available() else "cpu")
DATA = E("DATA", "./data/wikitext"); DSET = E("DSET", "wikitext-103-raw-v1")
VMAX = int(E("VMAX", "32768")); NHEAD = int(E("NHEAD", "6")); K = int(E("K", "4"))
BS = int(E("BS", "16")); NBATCH = int(E("NBATCH", "32")); NBIN = int(E("NBIN", "10"))
NBOOT = int(E("NBOOT", "1000")); TAG = E("TAG", "align"); SEED = int(E("SEED", "0"))
FAKE = E("FAKE_DATA", "0") == "1"
ARMS = E("ARMS", "tie rot rshare rand").split()
CKPT_FMT = E("CKPT_FMT", "ckpt_2x2/wt103_40m_hd0_{arm}.pt")
REF = E("REF", ARMS[0])
assert REF in ARMS, f"REF={REF} is not in ARMS"


def load_stages(prefix):
    fs = sorted(glob.glob(prefix + ".r*"), key=lambda p: int(p.rsplit(".r", 1)[1]))
    assert len(fs) == K, f"{prefix}: {len(fs)} segments != K={K}"
    out = []
    for p in fs:
        sd = torch.load(p, map_location="cpu"); sd = sd.get("stage", sd)
        out.append({(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()})
    return out


SD = {a: load_stages(CKPT_FMT.format(arm=a)) for a in ARMS}
s0 = SD[REF][0]
DMODEL = s0["emb.weight"].shape[1]; V = s0["emb.weight"].shape[0]; SEQ = s0["pos.weight"].shape[0]
FW = s0["blocks.0.mlp.0.weight"].shape[0]
NBLK = [1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks.")) for sd in SD[REF]]
HD = 1 + max((int(k.split(".")[1]) for k in SD[REF][0] if k.startswith("hblocks.")), default=-1)
IS_TIE = {a: ("w_t" in SD[a][0]) for a in ARMS}
HAS_ROT = {a: ("rot" in SD[a][0]) for a in ARMS}
for a in ARMS:
    assert SD[a][0]["emb.weight"].shape == s0["emb.weight"].shape, f"{a}: model size mismatch"
    hd_a = 1 + max((int(k.split(".")[1]) for k in SD[a][0] if k.startswith("hblocks.")), default=-1)
    assert hd_a == HD, f"{a}: HD={hd_a} != {HD}"
print(f"[cfg] DMODEL={DMODEL} V={V} SEQ={SEQ} FW={FW} NHEAD={NHEAD} K={K} blocks={NBLK} hd={HD} "
      f"NBATCH={NBATCH} BS={BS} ref={REF} TAG={TAG}")
for a in ARMS:
    print(f"       arm {a:<8} readout={'tied(w_t)' if IS_TIE[a] else 'private(hfc)'}"
          f"{'  +rot' if HAS_ROT[a] else ''}")


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(DMODEL)
        self.attn = nn.MultiheadAttention(DMODEL, NHEAD, dropout=0.0, batch_first=True)
        self.ln2 = nn.LayerNorm(DMODEL)
        self.mlp = nn.Sequential(nn.Linear(DMODEL, FW), nn.GELU(), nn.Linear(FW, DMODEL), nn.Dropout(0.0))

    def forward(self, x, mask):
        h = self.ln1(x); a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a; return x + self.mlp(self.ln2(x))


class FullLM(nn.Module):
    def __init__(self, sds, is_tie, has_rot):
        super().__init__()
        self.emb = nn.Embedding(V, DMODEL); self.pos = nn.Embedding(SEQ, DMODEL)
        self.trunks = nn.ModuleList([nn.ModuleList([Block() for _ in range(NBLK[k])]) for k in range(K)])
        self.hblocks = nn.ModuleList(); self.hln = nn.ModuleList(); self.readout = nn.ModuleList()
        for k in range(K - 1):
            self.hblocks.append(nn.ModuleList([Block() for _ in range(HD)]))
            self.hln.append(nn.LayerNorm(DMODEL)); self.readout.append(nn.Linear(DMODEL, V, bias=True))
        self.ln = nn.LayerNorm(DMODEL); self.fc = nn.Linear(DMODEL, V)
        with torch.no_grad():
            self.emb.weight.copy_(sds[0]["emb.weight"]); self.pos.weight.copy_(sds[0]["pos.weight"])
            for k in range(K):
                for j in range(NBLK[k]): self._ld(self.trunks[k][j], sds[k], f"blocks.{j}.")
                if k < K - 1:
                    for j in range(HD): self._ld(self.hblocks[k][j], sds[k], f"hblocks.{j}.")
                    self._ld(self.hln[k], sds[k], "hln.")
                    if is_tie:
                        self.readout[k].weight.copy_(sds[k]["w_t"]); self.readout[k].bias.copy_(sds[k]["b_t"])
                    else:
                        self.readout[k].weight.copy_(sds[k]["hfc.weight"]); self.readout[k].bias.copy_(sds[k]["hfc.bias"])
                else:
                    self._ld(self.ln, sds[k], "ln."); self._ld(self.fc, sds[k], "fc.")
        for k in range(K - 1):                    # fixed orthogonal R_k (rot arm), not a parameter
            self.register_buffer(f"rot{k}", sds[k]["rot"].clone() if has_rot else torch.empty(0))
        self.has_rot = has_rot

    @staticmethod
    def _ld(mod, sd, pre): mod.load_state_dict({k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)})

    def trunk(self, k, h, m):
        for b in self.trunks[k]: h = b(h, m)
        return h

    def aux_logits(self, k, h, m):
        f = h
        for b in self.hblocks[k]: f = b(f, m)
        f = self.hln[k](f)
        if self.has_rot: f = f @ getattr(self, f"rot{k}")   # same order as aux_readout in lm_hz_rot.py
        return self.readout[k](f)

    def terminal(self, h): return self.fc(self.ln(h))

    def module_params(self, k):
        """Trunk parameters of module k (plus emb/pos for k=0); head blocks are excluded."""
        ps = [p for b in self.trunks[k] for p in b.parameters()]
        if k == 0: ps = [self.emb.weight, self.pos.weight] + ps
        return ps


mask = torch.full((SEQ, SEQ), float("-inf"), device=DEV).triu(1)


def get_val():
    if FAKE:
        g = torch.Generator().manual_seed(SEED)
        mat = torch.randint(0, V, (BS, SEQ * (NBATCH + 1)), generator=g)
        return mat, [i * SEQ for i in range(NBATCH)]
    blob = torch.load(os.path.join(DATA, f"word_cache_{DSET}_{VMAX}.pt"))
    d = blob["data"]; val = d["validation"] if isinstance(d, dict) and "validation" in d else d
    if isinstance(val, dict): val = val.get("validation", next(iter(val.values())))
    val = torch.as_tensor(val).long(); n = val.numel() // BS
    mat = val[: n * BS].view(BS, n)
    return mat, [i for i in range(0, n - 1 - SEQ, SEQ)][:NBATCH]


models = {a: FullLM(SD[a], IS_TIE[a], HAS_ROT[a]).to(DEV).eval() for a in ARMS}
mat, starts = get_val()
print(f"[data] {'FAKE ' if FAKE else ''}batches={len(starts)}")


def flat(gs, ps):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, ps)])


PAR = {a: [[] for _ in range(K - 1)] for a in ARMS}     # cos_par[arm][k] -> per-batch
ACT = {a: [[] for _ in range(K - 1)] for a in ARMS}     # cos_act[arm][k] -> per-position
TRG = {a: [[] for _ in range(K - 1)] for a in ARMS}
GN = {a: [[] for _ in range(K - 1)] for a in ARMS}      # |g_local| / |g_global|, parameter space
for a, m in models.items():
    for si in starts:
        x = mat[:, si:si + SEQ].to(DEV); y = mat[:, si + 1:si + 1 + SEQ].to(DEV)
        h = m.emb(x) + m.pos(torch.arange(SEQ, device=DEV))[None]
        hs = []
        for k in range(K):
            h = m.trunk(k, h, mask); hs.append(h)
        L = F.cross_entropy(m.terminal(hs[K - 1]).reshape(-1, V), y.reshape(-1))
        for k in range(K - 1):
            zk = m.aux_logits(k, hs[k], mask); qk = F.softmax(zk, -1)
            ell = F.cross_entropy(zk.reshape(-1, V), y.reshape(-1))
            gl = torch.autograd.grad(ell, hs[k], retain_graph=True)[0]
            gg = torch.autograd.grad(L, hs[k], retain_graph=True)[0]
            ACT[a][k].append(F.cosine_similarity(gl.reshape(-1, DMODEL), gg.reshape(-1, DMODEL), -1)
                             .detach().cpu().numpy())
            TRG[a][k].append((1 - qk.pow(2).sum(-1)).reshape(-1).detach().cpu().numpy())
            ps = m.module_params(k)
            pl = flat(torch.autograd.grad(ell, ps, retain_graph=True, allow_unused=True), ps)
            pg = flat(torch.autograd.grad(L, ps, retain_graph=True, allow_unused=True), ps)
            PAR[a][k].append(float(F.cosine_similarity(pl, pg, 0)))
            GN[a][k].append(float(pl.norm() / (pg.norm() + 1e-12)))
PAR = {a: [np.array(v) for v in PAR[a]] for a in ARMS}
ACT = {a: [np.concatenate(v) for v in ACT[a]] for a in ARMS}
TRG = {a: [np.concatenate(v) for v in TRG[a]] for a in ARMS}
GN = {a: [np.array(v) for v in GN[a]] for a in ARMS}

rng = np.random.default_rng(0)


def boot_ci(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2: return (float(x.mean()) if len(x) else 0.0), 0.0, 0.0
    bm = x[rng.integers(0, len(x), size=(NBOOT, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))


res = {"tag": TAG, "arms": ARMS, "ref": REF, "hd": HD, "K": K, "nbatch": len(starts),
       "is_tie": IS_TIE, "has_rot": HAS_ROT, "par": {}, "act": {}, "gnorm_ratio": {},
       "par_vs_ref": {}, "act_vs_ref": {}}
for a in ARMS:
    res["par"][a] = [boot_ci(PAR[a][k]) for k in range(K - 1)]
    res["act"][a] = [boot_ci(ACT[a][k][:: max(1, len(ACT[a][k]) // 20000)]) for k in range(K - 1)]
    res["gnorm_ratio"][a] = [float(GN[a][k].mean()) for k in range(K - 1)]
    if a != REF:                                   # paired: same batches / positions
        res["par_vs_ref"][a] = [boot_ci(PAR[a][k] - PAR[REF][k]) for k in range(K - 1)]
        res["act_vs_ref"][a] = [boot_ci((ACT[a][k] - ACT[REF][k])[:: max(1, len(ACT[a][k]) // 20000)])
                                for k in range(K - 1)]
res["par_all"] = {a: [float(PAR[a][k].mean()) for k in range(K - 1)] for a in ARMS}

out = f"align_{TAG}.json"; json.dump(res, open(out, "w"), indent=1)

W = max(8, max(len(a) for a in ARMS) + 1)
print(f"\n==== parameter space cos(∇_θk ℓ_k, ∇_θk L)  [mean, 95% bootstrap CI over {len(starts)} batches] ====")
print(f"{'arm':<{W}}" + "".join(f"{'module '+str(k):>26}" for k in range(K - 1)))
for a in ARMS:
    row = "".join(f"{f'{m:+.3f} [{lo:+.3f},{hi:+.3f}]':>26}" for m, lo, hi in res["par"][a])
    print(f"{a:<{W}}" + row)
print(f"\n==== activation space cos(∇_hk ℓ_k, ∇_hk L)  [per-position mean] ====")
print(f"{'arm':<{W}}" + "".join(f"{'module '+str(k):>26}" for k in range(K - 1)))
for a in ARMS:
    row = "".join(f"{f'{m:+.3f} [{lo:+.3f},{hi:+.3f}]':>26}" for m, lo, hi in res["act"][a])
    print(f"{a:<{W}}" + row)
print(f"\n==== paired difference vs {REF} (parameter space; <0 = less BP-like than {REF}) ====")
for a in ARMS:
    if a == REF: continue
    row = "".join(f"{f'{m:+.3f} [{lo:+.3f},{hi:+.3f}]':>26}" for m, lo, hi in res["par_vs_ref"][a])
    print(f"{a:<{W}}" + row)
print(f"\n|∇ℓ|/|∇L| (parameter space, mean): " + "  ".join(
    f"{a}={['%.2f' % v for v in res['gnorm_ratio'][a]]}" for a in ARMS))
print(f"\n[dump] {out}")
