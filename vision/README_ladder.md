# Vision readout ladder

The ladder measures how the choice of readout for the local heads affects accuracy, by training the same network with different readout arms.

## Arms

- `random`: frozen random readout of the same shape (`TIE=1 TIE_MODE=random TIE_SCALE=1`).
- `untie`: each head has its own private readout (`TIE=0`).
- `tie`: heads use a detached copy of the live terminal readout `fc.weight` (`TIE=1`).
- `stale`: frozen terminal readout loaded from a trained checkpoint (`TIE=1 TIE_MODE=stale STALE_FC=<ckpt>`).
- `bp`: end-to-end backprop baseline, only with `WITH_BP=1` (`MODE=bp`).

## Files

- `run_ladder_abl.sh`: two-phase driver. P1 runs random / untie / tie (and bp) in parallel, one job per GPU; P2 runs stale using the same-seed tie checkpoint.
- `vision_ladder_r32.py`: ResNet-32 ladder for `DATASET=c10|c100|tiny` (default backbone).
- `vision_ladder_r50.py`: ResNet-50 ladder; the readout acts on the 2048-d terminal features. Self-check with `DRYRUN=1`.

Both .py files are self-contained and need only torch, torchvision, numpy and PIL.

## Commands

```bash
D=./data   # data directory
# (1) CIFAR-100, R32: 200 epochs, batch 128, AutoAugment, label smoothing 0.1, K=15
DATA_ROOT=$D DATASET=c100 EPOCHS=200 BATCH=128 LABEL_SMOOTH=0.1 SEEDS=42 GPUS=0,1,2,3 \
  TAGPFX=c100sota_r32 nohup bash run_ladder_abl.sh > logs/drv_c100sota_r32.log 2>&1 &
# (2) Tiny-ImageNet, R32: 100 epochs, batch 128, AutoAugment, one-hot labels, K=15
DATA_ROOT=$D DATASET=tiny EPOCHS=100 BATCH=128 LABEL_SMOOTH=0 SEEDS=42 GPUS=0,1,2,3 \
  TAGPFX=tiny_r32 nohup bash run_ladder_abl.sh > logs/drv_tiny_r32.log 2>&1 &
# (3) Tiny-ImageNet, R50: 100 epochs, batch 64, AutoAugment, one-hot labels, K=16, head depth 2 / widen 1
DATA_ROOT=$D SCRIPT=./vision_ladder_r50.py DATASET=tiny EPOCHS=100 BATCH=64 \
  HEAD_DEPTH=2 HEAD_WIDEN=1 K_MODULES=16 LABEL_SMOOTH=0 SEEDS=42 GPUS=0,1,2,3 \
  TAGPFX=tiny_r50 nohup bash run_ladder_abl.sh > logs/drv_tiny_r50.log 2>&1 &
```

Each command uses all GPUs listed in `GPUS`; run them one after another, or give each a disjoint GPU set (e.g. `GPUS=0,1` and `GPUS=2,3`). Read results with:

```bash
grep -aH FINAL logs/<TAGPFX>_*.log
```

## Main options

| Variable | Default | Meaning |
|---|---|---|
| `SCRIPT` | `./vision_ladder_r32.py` | backbone script |
| `DATASET` | `c100` | `c10`, `c100` or `tiny` |
| `EPOCHS` / `BATCH` | `200` / `128` | training length and batch size |
| `K_MODULES` | `15` | number of modules |
| `HEAD_DEPTH` / `HEAD_WIDEN` | `4` / `2` | auxiliary head size |
| `TIE_DIM` | `64` | readout input dimension (forced to 2048 by the R50 script) |
| `LABEL_SMOOTH` / `TERM_LS` | `0` / `LABEL_SMOOTH` | label smoothing of intermediate heads / terminal head |
| `SEEDS` | `42` | space-separated seeds |
| `GPUS` | `0,1,2,3` | GPUs; one job per GPU |
| `WITH_BP` | `0` | also run the bp arm |
| `TAGPFX` | `abl_${DATASET}` | prefix of log and checkpoint names |
| `DATA_ROOT` | `./data` | data directory |
| `PYBIN` | `python` | Python interpreter |

## Notes

- `DATA_ROOT` must contain `tiny-imagenet-200/` for `DATASET=tiny`; for CIFAR it must exist and be writable (CIFAR is downloaded there).
- Label smoothing is set only through `LABEL_SMOOTH` (intermediate heads) and `TERM_LS` (terminal head, defaults to `LABEL_SMOOTH`); `LS=` has no effect. The `ls=` / `term_ls=` fields on the `device=` line of each log show the values in use.
- With the R50 backbone, run the tie arm with `LABEL_SMOOTH=0.1 TERM_LS=0.1`; with one-hot labels it can collapse to chance level.
- The stale arm loads its readout from the tie checkpoint of the same `TAGPFX` and seed, so it inherits any failure of that run. To use another readout (e.g. from an untie run), run the backbone script directly with `TIE=1 TIE_MODE=stale STALE_FC=ckpt/<tag>.pt`.
- Tags whose log already contains a FINAL line are skipped, so a driver command can be re-run to resume.
