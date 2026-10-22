"""run_lwce.py - runner for the local-CE control.

Sets the environment, installs observation hooks, then runs slim_solo.py (or SRC) unchanged via runpy;
training math, random streams and checkpoint format are unchanged.
  1) Config guard: with LWCE=1, TK must be all zeros and MU must be 0 (FORCE=1 overrides).
  2) Terminal CE: F.cross_entropy is called with grad only for the terminal module's training loss
     (module_loss does not use it; evaluate() runs under no_grad); its EMA (0.98/0.02) is kept on GPU.
  3) Inserts loss= / ppl~ into [step ...] lines; all other output passes through unchanged.

Usage (light head):
  CUDA_VISIBLE_DEVICES=0 BIN=data/slim_2b.bin TOKENS=1000000000 BS=8 ACCUM=8 \
    K=4 HEAD_DEPTH=0 EVAL_EVERY=200 OUT=ckpt/lwce_light_1b.pt \
    nohup python -u run_lwce.py > logs/lwce_light_1b.log 2>&1 &
Matched head: HEAD_DEPTH=2. Milestone-loss arm through the same runner: LWCE=0. SRC=slim_untied.py for the untied control.
"""
import builtins
import math
import os
import runpy
import sys

_print_raw = print

LWCE = os.environ.get("LWCE", "1") == "1"
if LWCE:                                   # milestone off: t=0 => ell_m == ell_q => hinge=0
    os.environ.setdefault("TK", ",".join(["0"] * (int(os.environ.get("K", "4")) - 1)))
    os.environ.setdefault("MU", "0")
    os.environ.setdefault("AUX", "1")

TK = [float(v) for v in os.environ.get("TK", "").split(",") if v]
MU = float(os.environ.get("MU", "0.3"))
if LWCE and os.environ.get("FORCE") != "1":
    bad = []
    if not TK or any(t != 0.0 for t in TK):
        bad.append(f"TK={os.environ.get('TK','<unset>')} (must be all zeros, length K-1)")
    if MU != 0.0:
        bad.append(f"MU={MU} (must be 0)")
    if float(os.environ.get("AUX", "0.3")) <= 0:
        bad.append("AUX must be > 0 (it is the CE weight)")
    if bad:
        sys.exit("[run_lwce] config is not pure local CE, refusing to start:\n  - " + "\n  - ".join(bad))

import torch.nn.functional as F                                    # noqa: E402
import torch.distributed as dist                                  # noqa: E402
from datetime import timedelta                                    # noqa: E402

# rank-0-only evaluation can exceed the default 600s NCCL watchdog while other ranks wait in barrier
_ipg = dist.init_process_group


def _ipg_hook(*a, **k):
    if "timeout" not in k:
        k["timeout"] = timedelta(hours=3)
        _print_raw("[run_lwce] init_process_group timeout -> 3h (keeps the watchdog from killing at eval boundaries)")
    return _ipg(*a, **k)


dist.init_process_group = _ipg_hook

_ce, _st = F.cross_entropy, {"ema": None}


def _ce_hook(*a, **k):
    out = _ce(*a, **k)
    if out.requires_grad and out.dim() == 0:        # terminal CE of the training path (eval runs under no_grad)
        d = out.detach().float()
        _st["ema"] = d if _st["ema"] is None else 0.98 * _st["ema"] + 0.02 * d
    return out


F.cross_entropy = _ce_hook
_print = builtins.print


def _print_hook(*a, **k):
    if a and isinstance(a[0], str) and a[0].startswith("[step ") and _st["ema"] is not None:
        em = float(_st["ema"])
        a = (a[0].replace("] ", f"] loss={em:.3f} ppl~{math.exp(em):.1f} ", 1),) + a[1:]
    _print(*a, **k)


builtins.print = _print_hook
if os.environ.get("RANK", "0") == "0":
    _print(f"[run_lwce] LWCE={int(LWCE)} TK={os.environ.get('TK')} MU={os.environ.get('MU')} "
               f"AUX={os.environ.get('AUX')} HEAD_DEPTH={os.environ.get('HEAD_DEPTH','2')} "
           f"-> terminal CE hook installed, slim_solo.py runs unchanged", flush=True)

_here = os.path.dirname(os.path.abspath(__file__))
_SRC = os.environ.get("SRC", "slim_solo.py")
_print_raw(f"[run_lwce] SRC={_SRC}")
runpy.run_path(os.path.join(_here, _SRC), run_name="__main__")
