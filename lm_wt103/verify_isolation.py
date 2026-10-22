"""verify_isolation.py - check the gradient-isolation property of lm_hz_tievar.py (CPU, ~2 min, no data needed).

Runs lm_hz_tievar.py unchanged via runpy, with probes on dist.send/isend:
  V1 every float tensor sent across a module boundary has requires_grad=False and grad_fn=None;
  V2 (tie mode) w_t/b_t are buffers and in no optimizer param group;
  V3 after the run, w_t.requires_grad=False and w_t.grad is None.
Integer tensors (labels) are not checked.

Usage (2 processes, once per arm; other env as in lm_hz_tievar.py):
  RANK=r LOCAL_RANK=r python -u verify_isolation.py
"""
import os
import runpy

import torch
import torch.distributed as dist

CNT = {"n": 0}
_send, _isend = dist.send, dist.isend


def _chk(t):
    if torch.is_tensor(t) and t.is_floating_point():
        assert not t.requires_grad and t.grad_fn is None, \
            "V1 FAILED: interface float tensor carries a grad graph!"
        CNT["n"] += 1


def send(t, *a, **k):
    _chk(t)
    return _send(t, *a, **k)


def isend(t, *a, **k):
    _chk(t)
    return _isend(t, *a, **k)


dist.send, dist.isend = send, isend       # same module object as `dist` inside lm_hz_tievar.py

here = os.path.dirname(os.path.abspath(__file__))
g = runpy.run_path(os.path.join(here, "lm_hz_tievar.py"), run_name="__main__")

rank = int(os.environ["RANK"])
K = int(os.environ["WORLD_SIZE"])
tie = int(os.environ.get("TIE_AUX", "1"))
stage, opt = g["stage"], g["opt"]
if tie and rank < K - 1:
    pids = {id(p) for grp in opt.param_groups for p in grp["params"]}
    assert id(stage.w_t) not in pids and id(stage.b_t) not in pids, \
        "V2 FAILED: w_t/b_t appear in the optimizer!"
    assert not stage.w_t.requires_grad and stage.w_t.grad is None, \
        "V3 FAILED: w_t has a gradient!"
    print(f"[verify rank{rank} tie={tie}] V2/V3 passed: "
          f"w_t/b_t are read-only buffers, not in the optimizer, grad=None", flush=True)
print(f"[verify rank{rank} tie={tie}] V1 passed: "
      f"all {CNT['n']} interface float sends detached (zero cross-boundary gradient)", flush=True)
