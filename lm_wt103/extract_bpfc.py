"""extract_bpfc.py - extract the terminal readout (fc.weight, fc.bias) from a lm_bp_ddp.py checkpoint
into a flat dict {"fc.weight", "fc.bias"}, the format read by the TIE_MODE=stale loader in lm_hz_tievar.py.
With TIE=1, fc.weight and emb.weight are the same tensor; emb.weight is used if fc.weight is absent.

Usage:
  python extract_bpfc.py <in_bp_ckpt> <out_readout>
  python extract_bpfc.py ckpt_wt103tie/wt103_bp.pt ckpt_wt103tie/bpfc_readout.pt
"""
import sys
import torch

IN, OUT = sys.argv[1], sys.argv[2]
blob = torch.load(IN, map_location="cpu")
sd = blob.get("model", blob) if isinstance(blob, dict) else blob
sd = sd.get("stage", sd) if isinstance(sd, dict) else sd


def pick(*suffixes):
    for suf in suffixes:
        for k, v in sd.items():
            if k.endswith(suf) and torch.is_tensor(v):
                return k, v
    return None, None


kw, W = pick("fc.weight", "emb.weight")
kb, B = pick("fc.bias")
assert W is not None, f"fc.weight/emb.weight not found; sample keys {list(sd)[:8]}"
if B is None:                                     # loader requires a (V,) bias
    B = torch.zeros(W.shape[0]); kb = "(zeros fallback)"

out = {"fc.weight": W.float().contiguous(), "fc.bias": B.float().contiguous()}
torch.save(out, OUT)
print(f"[extract] W<-{kw} {tuple(W.shape)} std={float(W.float().std()):.4f}  |  "
      f"b<-{kb} {tuple(B.shape)}  ->  {OUT}")
print(f"[check] for the stale loader assertion: fc.weight needs (V,DMODEL)={tuple(W.shape)}, fc.bias needs ({W.shape[0]},) => must match the (V,DMODEL) of this size rung")
