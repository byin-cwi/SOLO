"""lens_probe.py - functional logit lens: compare each arm with BP per depth, each network using its own terminal readout.

Reads the single-process align-ladder checkpoints of lm_solo.py (CKPTDIR/wt103al_<arm>_s<SEED>.pt, arms
solo/priv/rand/conv_loc/conv_bp) and a BP checkpoint. Only trunk + terminal readout are loaded
(heads.{K-1}.ln/fc for arms, top-level ln/fc for BP). Per arm and boundary k:
  kl_armbp = KL(p_arm || p_BP), kl_bparm = KL(p_BP || p_arm), agree_bp = top-1 agreement
  lens_ppl = ppl of the arm's terminal readout applied to h_k
  cka_raw / cka_z = linear CKA(h_k^arm, h_k^BP) on raw / per-feature standardized activations

Env (defaults): SEED=42, CKPTDIR=ckpt_align, BP=ckpt_wt103tie/wt103_bp.pt, DATA=./data/wikitext,
  DSET=wikitext-103-raw-v1, VMAX=32768 (reads word_cache_<DSET>_<VMAX>.pt), NHEAD=6, K=4, BS=16, NBATCH=32,
  OUTDIR=., TAG=align_s<SEED>, DEV.
Usage: SEED=42 CKPTDIR=ckpt_align BP=ckpt_wt103tie/wt103_bp.pt DATA=./data/wikitext DSET=wikitext-103-raw-v1 \
      VMAX=32768 NHEAD=6 K=4 BS=16 NBATCH=32 OUTDIR=logs_lens python -u lens_probe.py
Output: OUTDIR/lens_results_<TAG>.json
"""
import os, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

E = os.environ.get
DEV = E("DEV", "cuda:0" if torch.cuda.is_available() else "cpu")
DATA = E("DATA", "./data/wikitext"); DSET = E("DSET", "wikitext-103-raw-v1")
VMAX = int(E("VMAX", "32768")); NHEAD = int(E("NHEAD", "6")); K = int(E("K", "4"))
BS = int(E("BS", "16")); NBATCH = int(E("NBATCH", "32")); SEED = E("SEED", "42")
CKPTDIR = E("CKPTDIR", "ckpt_align"); BP = E("BP", "ckpt_wt103tie/wt103_bp.pt")
OUTDIR = E("OUTDIR", "."); os.makedirs(OUTDIR, exist_ok=True); TAG = E("TAG", f"align_s{SEED}")
ARMS = {"SOLO": "solo", "PRIV": "priv", "RAND": "rand", "CONV_loc": "conv_loc", "CONV_bp": "conv_bp"}
PFX = {a: f"{CKPTDIR}/wt103al_{s}_s{SEED}.pt" for a, s in ARMS.items()}

def load_flat(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob.get("model", blob) if isinstance(blob, dict) else blob
    sd = sd.get("stage", sd) if isinstance(sd, dict) else sd
    return {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}

SDs = {a: load_flat(p) for a, p in PFX.items()}
BPsd = load_flat(BP)
s0 = SDs["SOLO"]
DMODEL = s0["emb.weight"].shape[1]; V = s0["emb.weight"].shape[0]; SEQ = s0["pos.weight"].shape[0]
FFN = s0["blocks.0.mlp.0.weight"].shape[0]
NLAYER = 1 + max(int(k.split(".")[1]) for k in s0 if k.startswith("blocks."))
per = NLAYER // K; CUM = [per * (k + 1) for k in range(K)]
print(f"[cfg] DMODEL={DMODEL} V={V} SEQ={SEQ} FFN={FFN} NLAYER={NLAYER} K={K} cum={CUM} NHEAD={NHEAD} SEED={SEED} TAG={TAG}")

class Block(nn.Module):                        # same as the lm_solo.py Block
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(DMODEL)
        self.attn = nn.MultiheadAttention(DMODEL, NHEAD, dropout=0.0, batch_first=True)
        self.ln2 = nn.LayerNorm(DMODEL)
        self.mlp = nn.Sequential(nn.Linear(DMODEL, FFN), nn.GELU(), nn.Linear(FFN, DMODEL), nn.Dropout(0.0))
    def forward(self, x, mask):
        h = self.ln1(x); a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a; return x + self.mlp(self.ln2(x))

def term_prefix(sd):                           # arm: heads.{K-1}., BP: top-level (or heads.0.)
    for lnp, fcp in [(f"heads.{K-1}.ln.", f"heads.{K-1}.fc."), ("ln.", "fc."), ("heads.0.ln.", "heads.0.fc.")]:
        if any(k.startswith(fcp) for k in sd) and any(k.startswith(lnp) for k in sd):
            return lnp, fcp
    raise KeyError(f"terminal ln/fc not found; sample keys {list(sd)[:8]}")

class LensLM(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.emb = nn.Embedding(V, DMODEL); self.pos = nn.Embedding(SEQ, DMODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(NLAYER)])
        self.ln = nn.LayerNorm(DMODEL); self.fc = nn.Linear(DMODEL, V)
        lnp, fcp = term_prefix(sd)
        with torch.no_grad():
            self.emb.weight.copy_(sd["emb.weight"]); self.pos.weight.copy_(sd["pos.weight"])
            for j in range(NLAYER):
                self.blocks[j].load_state_dict({k[len(f"blocks.{j}."):]: v for k, v in sd.items() if k.startswith(f"blocks.{j}.")})
            self.ln.load_state_dict({k[len(lnp):]: v for k, v in sd.items() if k.startswith(lnp)})
            self.fc.load_state_dict({k[len(fcp):]: v for k, v in sd.items() if k.startswith(fcp)})
    def hiddens(self, x, mask):
        h = self.emb(x) + self.pos(torch.arange(SEQ, device=x.device))[None]; hs = []; nxt = 0
        for j in range(NLAYER):
            h = self.blocks[j](h, mask)
            if j + 1 == CUM[nxt]: hs.append(h); nxt += 1
        return hs
    def readout(self, h): return self.fc(self.ln(h))

mask = torch.full((SEQ, SEQ), float("-inf"), device=DEV).triu(1)

def get_val():
    blob = torch.load(os.path.join(DATA, f"word_cache_{DSET}_{VMAX}.pt"), weights_only=False)
    d = blob["data"]; val = d["validation"] if isinstance(d, dict) and "validation" in d else d
    if isinstance(val, dict): val = val.get("validation", next(iter(val.values())))
    val = torch.as_tensor(val).long(); n = val.numel() // BS
    mat = val[: n * BS].view(BS, n)
    return mat, [i for i in range(0, n - 1 - SEQ, SEQ)][:NBATCH]

models = {a: LensLM(SDs[a]).to(DEV).eval() for a in ARMS}
bp = LensLM(BPsd).to(DEV).eval()
mat, starts = get_val()
print(f"[data] batches={len(starts)}")

@torch.no_grad()
def run():
    res = {a: {"kl_armbp": [], "kl_bparm": [], "agree_bp": [], "lens_ppl": []} for a in models}
    for a, m in models.items():
        acc = {k: dict(klab=0., klba=0., agr=0., lens=0., n=0) for k in range(K)}
        for si in starts:
            x = mat[:, si:si + SEQ].to(DEV); y = mat[:, si + 1:si + 1 + SEQ].to(DEV); yf = y.reshape(-1); nt = yf.numel()
            ha = m.hiddens(x, mask); hbp = bp.hiddens(x, mask)
            for k in range(K):
                za = m.readout(ha[k]).reshape(-1, V); zb = bp.readout(hbp[k]).reshape(-1, V)
                la = F.log_softmax(za, -1); lb = F.log_softmax(zb, -1); pa = la.exp(); pb = lb.exp()
                acc[k]["klab"] += (pa * (la - lb)).sum().item()
                acc[k]["klba"] += (pb * (lb - la)).sum().item()
                acc[k]["agr"] += (za.argmax(-1) == zb.argmax(-1)).float().sum().item()
                acc[k]["lens"] += F.cross_entropy(za, yf).item() * nt
                acc[k]["n"] += nt
        for k in range(K):
            c = acc[k]; n = c["n"]
            res[a]["kl_armbp"].append(c["klab"] / n); res[a]["kl_bparm"].append(c["klba"] / n)
            res[a]["agree_bp"].append(c["agr"] / n); res[a]["lens_ppl"].append(float(np.exp(c["lens"] / n)))
    return res

@torch.no_grad()
def cka_pass():
    d = DMODEL
    st = {a: [dict(sxx=torch.zeros(d, d, device=DEV), syy=torch.zeros(d, d, device=DEV),
                   syx=torch.zeros(d, d, device=DEV), sx=torch.zeros(d, device=DEV),
                   sy=torch.zeros(d, device=DEV), n=0) for _ in range(K)] for a in models}
    for si in starts:
        x = mat[:, si:si + SEQ].to(DEV); Hb = [h.reshape(-1, d).float() for h in bp.hiddens(x, mask)]
        for a, m in models.items():
            ha = m.hiddens(x, mask)
            for k in range(K):
                X = ha[k].reshape(-1, d).float(); Y = Hb[k]; s = st[a][k]
                s["sxx"] += X.T @ X; s["syy"] += Y.T @ Y; s["syx"] += Y.T @ X
                s["sx"] += X.sum(0); s["sy"] += Y.sum(0); s["n"] += X.shape[0]
    def cka(s, z):
        n = s["n"]
        cxx = s["sxx"] - torch.outer(s["sx"], s["sx"]) / n
        cyy = s["syy"] - torch.outer(s["sy"], s["sy"]) / n
        cyx = s["syx"] - torch.outer(s["sy"], s["sx"]) / n
        if z:
            dx = cxx.diag().clamp_min(1e-8).sqrt(); dy = cyy.diag().clamp_min(1e-8).sqrt()
            cyx = cyx / torch.outer(dy, dx); cxx = cxx / torch.outer(dx, dx); cyy = cyy / torch.outer(dy, dy)
        return (cyx.pow(2).sum() / (cxx.norm() * cyy.norm() + 1e-12)).item()
    return {a: [cka(st[a][k], False) for k in range(K)] for a in models}, {a: [cka(st[a][k], True) for k in range(K)] for a in models}

res = run()
ckr, ckz = cka_pass()
for a in models: res[a]["cka_raw"] = ckr[a]; res[a]["cka_z"] = ckz[a]
res["_cfg"] = {"arms": ARMS, "cum": CUM, "K": K, "SEED": SEED, "BP": BP, "NBATCH": len(starts)}
out = os.path.join(OUTDIR, f"lens_results_{TAG}.json"); json.dump(res, open(out, "w"), indent=1)
print(f"\n[dump] {out}")
hdr = f"{'arm':>9} | " + " | ".join(f"k={k}" for k in range(K))
for metric in ["kl_armbp", "agree_bp", "lens_ppl", "cka_raw"]:
    print(f"\n== {metric} ==\n{hdr}")
    for a in models: print(f"{a:>9} | " + " | ".join(f"{v:8.3f}" for v in res[a][metric]))
