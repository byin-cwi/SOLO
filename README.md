# SOLO

SOLO splits a network into K gradient-isolated modules. Modules exchange forward activations only, never gradients. Each intermediate module carries an auxiliary head trained with a local objective, and the auxiliary heads compute their vocabulary/class readout through a detached copy of the terminal readout weight. This repository contains the training, evaluation, and measurement code for all main experiments in the paper.

[中文说明](README.zh-CN.md)

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.12, CUDA 12.x, and A100-80GB GPUs. All training scripts are configured through environment variables. Multi-GPU scripts initialize from `RANK` / `WORLD_SIZE` / `LOCAL_RANK` / `MASTER_ADDR` / `MASTER_PORT`, so they can be launched with `torchrun` or by setting the variables per rank.

## Layout

| Directory | Contents |
|---|---|
| `lm_slimpajama/` | SlimPajama 340M / 1.3B pretraining (GLA Table-2 protocol), shared-readout and untied controls, local-objective attribution, local activation replay, 2B FSDP baseline, HuggingFace export |
| `lm_wt103/` | WikiText-103 readout ladder (tie / untie / random / stale), scale sweep, K sweep under a published recipe, byte-level controls, gradient-alignment and logit-lens probes, machine check of gradient isolation |
| `vision/` | CIFAR-10/100, Tiny-ImageNet, ImageNet-1k (ResNet-50/101), VGG-11 / ViT-Tiny, DSN control, vision readout ladder, two-GPU deployment |
| `systems/` | Throughput, memory, and communication of SOLO vs. BP under the same parallelism (pipeline parallel and data parallel) |

`lm_slimpajama/CODEMAP.md` documents the SlimPajama files, configurations, and reproduction commands in detail. Each script's header lists its usage and all environment variables.

## Quick self-check (CPU, no data needed)

```bash
cd lm_slimpajama
python ms_identity_test.py        # contracted vs. explicit milestone loss (fp64 error ~1e-16)

# two-process K=2 pipeline, synthetic data, 3 steps
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29704 WORLD_SIZE=2 BACKEND=gloo FAKE_DATA=1 \
  STEPS=3 NLAYER=4 DMODEL=64 NHEAD=2 SEQ=64 BS=2 ACCUM=2 HEAD_DEPTH=1 EVAL_EVERY=2 \
  VAL_TOKENS=10000 LOG_EVERY=1 TIE_SYNC=2 CKPT_STAGE_DIR=/tmp
RANK=0 python -u slim_pipeline_solo.py & RANK=1 python -u slim_pipeline_solo.py; wait

cd ../lm_wt103                    # zero cross-module gradients: every boundary send is detached,
                                  # and the readout copy is not in the optimizer
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29771 WORLD_SIZE=2 BACKEND=gloo FAKE_DATA=1 \
  STEPS=3 EPOCHS=1 NLAYER=4 DMODEL=64 NHEAD=2 FFN=128 SEQ=32 BS=4 HEAD_DEPTH=1
RANK=0 LOCAL_RANK=0 python -u verify_isolation.py & RANK=1 LOCAL_RANK=1 python -u verify_isolation.py; wait
```

## Reproduction

### SlimPajama (`lm_slimpajama/`)

Data preparation, training, export, and evaluation commands are in `lm_slimpajama/CODEMAP.md`. Reference values (lm-eval-harness word-level WikiText perplexity / six-task zero-shot mean):

| Scale | BP | SOLO K=2 | SOLO K=4 |
|---|---|---|---|
| 340M / 15B tokens | 28.04 / 41.5 | 29.05 / 41.5 | 31.19 / 41.0 |
| 1.3B / 15B tokens | 21.61 / 45.9 | 22.29 / 45.4 | 23.59 / 45.0 |

### WikiText-103 (`lm_wt103/`)

```bash
cd lm_wt103
DATA=./data/wikitext python dl_wikitext.py                       # download WikiText-2/103
# before the first pipeline run, run lm_hz_tievar.py once with BUILD_WORD_CACHE=1 to build the word cache (exits when done)
GPUS=0,1,2,3 bash run_wt103_ladder.sh tie untie random stale    # readout ladder, 8L/d384, V=32768, K=4
GPUS=0,1,2,3 bash run_wt103_scale_sweep.sh                       # 64M–512M scale sweep
GPUS=0,1,2,3 bash run_wt103_pubrecipe_ksweep.sh                  # K sweep under a published recipe (16L/d410, BERT tokenizer)
```

The published recipe uses the bert-base-cased tokenizer; the required `bert_cache_*.pt` is built by the first run of `lm_solo.py` with `TOKENIZER=bert`.

Reference values for the readout ladder (terminal validation perplexity): tie 70.60, untie 75.35. Published recipe, K=2 tie: about 23.61. For the other scripts (`run_wt103_hdgrid.sh` head-depth grid, `run_wt103_2x2.sh` coordinate-frame 2×2, `run_wt103_vocab_sweep.sh` vocabulary sweep, `run_byte_gate.sh` byte level, `run_wt103_align_ladder.sh` alignment probe) see the file headers.

### Vision (`vision/`)

```bash
cd vision
python solo_cifar.py                                              # CIFAR, default MODE=md
MODE=bp python solo_cifar.py ; MODE=dsn python solo_cifar.py      # BP and DSN controls
DATA_ROOT=./data/imagenet ARCH=r50in MODE=solo GS_MODE=jacobi SCHED=s3 STRIDED_HEAD=1 \
  HEAD_DEPTH=3 HEAD_WIDEN=2 EMA_M=0.9 python imagenet_solo.py    # ImageNet-1k R50
DATA_ROOT=./data/imagenet ARCH=r50in MODE=bp EPOCHS=90 BATCH=256 python imagenet_solo.py
DATASET=c100 DATA_ROOT=./data GPUS=0,1,2,3 bash run_ladder_abl.sh # vision readout ladder
```

The ImageNet directory must contain `train/` and `val/` in class-folder layout. ImageNet R50 reference values: BP 76.49, SOLO (split [5|11]) 76.27. For Tiny-ImageNet, R101, VGG/ViT, and two-GPU deployment see the headers of `tiny_solo.py`, `imagenet101_solo.py`, `vgg_vit_solo.py`, `imagenet_standalone*.py`, and `imagenet_pipeline_solo.py`.

### Systems (`systems/`)

```bash
cd systems
GPUS=0,1,2,3,4,5,6,7 bash run_pipep_multi.sh    # SOLO+PP vs BP+PP (1F1B) vs BP+DDP; prints a summary table at the end
bash run_pipep_mlt.sh 0,1,2,3,4,5,6,7            # fixed global batch G, sweep over microbatches per replica m
```

`run_pipep_mlt.sh` (8 GPUs) compares the SOLO hybrid layouts (`solo_kr2x4`, `solo_kr4x2`, `solo`) with every BP 3D-parallel schedule (1F1B, interleaved, zero-bubble, PP x DP, DDP) at a fixed global batch, and prints each arm's throughput relative to the best BP schedule together with peak memory and communication per step. Options: `G=32 MLS="1 2 4 8" HS="2 1" DRY=1`.

## Implementation notes

- **Trainers and the paper's numbers.** Each line of experiments keeps a single final trainer. Some numbers in the paper's main tables were produced by earlier implementations of the same algorithm; later changes concern I/O, resumption, compilation, and parallel layout, not the training math. Evidence of equivalence: the identity regression in `ms_identity_test.py` (fp64 error ~1e-16), and three repeats of 340M K=4 across compile and micro-batch settings at 31.22±0.04. `torch.compile` does not guarantee bitwise reproducibility.
- **Shared-readout refresh interval.** `TIE_SYNC` defaults to 100 optimizer steps, matching the paper's main tables. This interval carries a measurable readout-staleness cost; 50 or less is recommended for new experiments.
- **Local objective in the WikiText-103 scripts.** Scripts that set `TK=0` explicitly (`run_wt103_ladder.sh`, `run_wt103_scalept.sh`, `run_wt103_stalebp.sh`, `run_wt103_vocab_sweep.sh`, `run_wt103_hdgrid.sh`, `run_wt103_2x2.sh`, `run_wt103_align_ladder.sh`, `run_wt103_pubrecipe_ksweep.sh`, `run_wt103_pubrecipe_tiesync.sh`, `run_byte_gate.sh`, `run_lwce_wt2.sh`) use `MODE=mfb BETA=1 MU=0 AUX=1`. In this setting the milestone m equals the module's own detached prediction, the hinge value is identically 0, and the intermediate objective reduces to local CE with the shared readout. Because `clamp_min` passes gradient at 0, positions where floating-point rounding makes ℓ_q−ℓ_m non-negative (about 3–5%) receive a doubled auxiliary gradient; loss values are unaffected. The `lm_solo.py` scripts that leave `TK` unset (`run_wt103_pubrecipe_*_sp.sh`, `run_wt103_pubrecipe_bp_stale.sh`) use the default `t_k=(k+1)/K`. The SlimPajama line uses `t_k=(k+1)/K`, `MU=0.3`, `AUX=0.3`, i.e. the full milestone objective.

## License

MIT, see `LICENSE`.
