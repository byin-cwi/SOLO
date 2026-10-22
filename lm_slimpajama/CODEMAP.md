# SlimPajama code map

Two scales, 340M and 1.3B, each trained on 15B tokens under the GLA Table-2 protocol. The workflow is data preparation, training (BP or the SOLO pipeline), export to HuggingFace, and evaluation with lm-eval-harness.

## Files

| Role | File |
|---|---|
| BP baseline (single GPU or DDP) | `slim_bp.py` |
| SOLO pipeline (shared readout, K gradient-isolated stages, optional data parallelism within each stage) | `slim_pipeline_solo.py` |
| Shared-readout variants (`TIE_MODE` selects trained / random / other readout controls) | `slim_pipeline_tievar.py` |
| Untied control (each auxiliary head has a private readout) | `slim_pipeline_untied.py` |
| Single-process SOLO and untied versions (340M four-arm attribution, driven by `run_lwce.py`) | `slim_solo.py`, `slim_untied.py`, `run_lwce.py` |
| Local activation replay | `slim_pipeline_replay.py` |
| 2B BP (FSDP / ZeRO-3) | `slim_bp_fsdp.py` |
| Checkpoint export to HuggingFace (with per-logit parity check) | `export_hf.py` |
| Split a BP checkpoint into pipeline stages / merge stages back into BP | `bp_to_pipe.py`, `hybrid_convert.py` |
| Data pre-tokenization | `prep_slimpajama.py` |
| Regression test for the contracted milestone loss | `ms_identity_test.py` |

## Model and protocol

All arms use Transformer++ (Llama architecture: RMSNorm, rotate-half RoPE, SwiGLU, untied embeddings), the Mistral-32k vocabulary, and SEQ=2048.

| Scale | NLAYER | DMODEL | NHEAD | FFN | Parameters |
|---|---|---|---|---|---|
| 340M | 24 | 1024 | 16 | 0 (auto, 2816) | ≈374M |
| 1.3B | 24 | 2048 | 16 | 0 (auto, 5632) | ≈1.36B |

Optimization: `LR=3e-4 LR_MIN=3e-5 WARMUP_TOKENS=5e8 WD=0.01 CLIP=1.0`, effective batch 524,288 tokens per step, `TOKENS=15000000000`, `opt_steps=28,610`.

The effective batch is `BS×SEQ×ACCUM×#GPUs` for BP and `BS×SEQ×ACCUM×R` for the pipeline, where `R = WORLD_SIZE / K` (K is not multiplied in). Micro-batch settings per arm:

| Scale | Arm | GPUs | Micro-batch |
|---|---|---|---|
| 340M | BP | 4 | `BS=8 ACCUM=8` |
| | K=2×R=2 | 4 | `K=2 BS=8 ACCUM=16` |
| | K=4 | 4 | `K=4 BS=8 ACCUM=32` |
| 1.3B | BP | 4 | `BS=4 ACCUM=16` |
| | K=2×R=2 | 4 | `K=2 BS=8 ACCUM=16` |
| | K=4 | 4 | `K=4 BS=16 ACCUM=16` |

## Local objective of the SOLO pipeline

Ranks are numbered `replica×K + stage`. The auxiliary head of intermediate stage k computes its readout with `w_t`, a detached copy of the terminal readout weight that is refreshed every `TIE_SYNC` optimizer steps. Only forward activations cross stage boundaries; no gradients do.

The intermediate-stage loss (`LOSS=md`) uses `t_k = (k+1)/K` at every position:

```
m      = softmax((1 - t)·log sg(q) + t·log δ_y^λ)
loss   = (ℓ_q − ℓ_m)_+ + MU·H²(q, m)·1[ℓ_q > ℓ_m] + AUX·ℓ_q
```

`δ_y^λ` puts probability `e^{-λ}` on the label and spreads the rest uniformly. `λ` is a bucketed ratchet floor: positions are split into `NBUCKET` buckets by the label's unigram surprisal `−log π(y)`; each bucket starts at `min(CFLOOR·mean(−log π(y)), KAPPA·EMA(achieved CE))` and is then non-increasing, bounded below by `LAM0`. `LOSS=ce` replaces the intermediate objective with plain CE (milestone ablation). The terminal stage always uses standard CE.

## Key environment variables

| Variable | Default | Meaning |
|---|---|---|
| `BIN` | `data/slim_2b.bin` | Pre-tokenized bin; keep it on a RAM disk |
| `TOKENS` | 0 (whole bin) | Training token budget |
| `K` | `WORLD_SIZE` | Number of pipeline stages |
| `SPLIT` | empty (even split) | Layers per stage, e.g. `14,10` |
| `HEAD_DEPTH` | 2 | Transformer blocks inside each auxiliary head |
| `TIE_SYNC` | 100 | Refresh interval of the shared-readout copy (optimizer steps). The paper's main table uses 100; it carries a measurable staleness cost, so 50 or less is recommended for new experiments |
| `MU` / `AUX` / `KAPPA` / `CFLOOR` / `LAM0` / `NBUCKET` | 0.3 / 0.3 / 0.7 / 0.9 / 0.1 / 12 | Milestone-loss parameters |
| `LOSS` | `md` | `md` milestone objective; `ce` greedy local CE |
| `COMPILE` | 0 | 1 fuses the milestone loss; 2 additionally compiles the trunk and head_out |
| `OUT` / `RESUME` | empty | Checkpoint prefix (saved per stage as `.r{stage}`); point `RESUME` at the same prefix to resume |
| `CKPT_STAGE_DIR` | `/dev/shm` | Checkpoints are written here first, then moved atomically to `OUT` in the background |
| `SAVE_EVERY` / `EVAL_EVERY` / `LOG_EVERY` | 2000 / 1000 / 50 | In optimizer steps |
| `UNI_BIN` / `RATCHET_FROM` | empty | Sources of the unigram table and ratchet state for warm starts |
| `FREEZE` / `FREEZE_STAGES` / `HEAD_LR_MULT` / `TRUNK_LR_MULT` / `L2SP` / `FANCHOR` / `MS_W` | off | Freezing, learning-rate multipliers, and retention anchors for warm starts and post-training |
| `DRIFT_T` | empty | Log the relative drift of the terminal readout every T steps |

`slim_bp.py` accepts the model, optimization, and checkpoint subset of these variables, plus `ARCH` (default `llama`), `DROP`, and `TIE`.

## Reproduction

### Data

```bash
TOKENS=15000000000 OUTBIN=data/slim_15b.bin python prep_slimpajama.py
mkdir -p /dev/shm/slim && cp data/slim_15b.bin data/slim_15b.bin.meta.json /dev/shm/slim/
```

### Training (1.3B shown; for 340M drop the model-shape variables and use the micro-batch table above)

```bash
COMMON="NLAYER=24 DMODEL=2048 NHEAD=16 FFN=0 SEQ=2048 TOKENS=15000000000 BIN=/dev/shm/slim/slim_15b.bin"

# BP
env $COMMON BS=4 ACCUM=16 COMPILE=1 OUT=ckpt/bp1b.pt RESUME=ckpt/bp1b.pt \
  torchrun --nproc_per_node=4 --master_port=29878 slim_bp.py

# SOLO K=2×R=2
env $COMMON K=2 BS=8 ACCUM=16 COMPILE=2 OUT=ckpt/solo1b_k2x2.pt RESUME=ckpt/solo1b_k2x2.pt \
  torchrun --nproc_per_node=4 --master_port=29874 slim_pipeline_solo.py

# SOLO K=4
env $COMMON K=4 BS=16 ACCUM=16 COMPILE=2 OUT=ckpt/solo1b_k4.pt RESUME=ckpt/solo1b_k4.pt \
  torchrun --nproc_per_node=4 --master_port=29694 slim_pipeline_solo.py
```

After launch, check `eff_batch=524,288`, `opt_steps=28,610`, and the parameter count in the log. After a crash, re-issue the same command to resume; weights, Adam moments, step, scheduler, ratchet floors, and data RNG are all restored. Pipeline ranks print `resumed @ step N (full)` and BP prints `resumed @ step N`. If a pipeline rank prints `WARM-START weights-only`, the checkpoint lacks optimizer state and the run is treated as a weights-only warm start; stop and investigate before continuing a production run.

### Evaluation

```bash
# BP
SRC=bp CKPT=ckpt/bp1b.pt OUTDIR=hf_out/bp1b NLAYER=24 DMODEL=2048 NHEAD=16 python export_hf.py
# Pipeline terminal exit (EXIT=-1) or an intermediate exit (EXIT=0..K-2)
SRC=pipe CKPT=ckpt/solo1b_k4.pt K=4 EXIT=-1 OUTDIR=hf_out/solo1b_k4 NLAYER=24 DMODEL=2048 NHEAD=16 python export_hf.py

lm_eval --model hf --model_args pretrained=hf_out/solo1b_k4,dtype=bfloat16 \
  --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge \
  --batch_size 16 --output_path evals/solo1b_k4
```

Before writing, the exporter compares the HF model against a minimal re-implementation of the training forward pass and refuses to save if `max|Δlogit| > 2e-3`. `eval_untie340m_k4.sh` is a complete multi-exit evaluation script for 340M K=4.

## Operational notes

- Keep the bin on `/dev/shm`. A memmap on NFS can raise SIGBUS when NFS stalls, and the training process then exits silently.
- Write logs to local disk (e.g. `/tmp`), not NFS.
- Process groups and subgroups use a 3h timeout, and checkpoints are staged through `CKPT_STAGE_DIR` and moved asynchronously, so slow writes do not trigger the NCCL watchdog.
