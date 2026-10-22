# SlimPajama 代码地图

340M 与 1.3B 两个规模，各 15B tokens，GLA Table-2 协议。流程为数据准备、训练（BP 或 SOLO 流水线）、导出 HuggingFace、lm-eval-harness 评测。

## 文件

| 角色 | 文件 |
|---|---|
| BP 基线（单卡或 DDP） | `slim_bp.py` |
| SOLO 流水线（共享读出，K 段梯度隔离，段内可数据并行） | `slim_pipeline_solo.py` |
| 共享读出变体（`TIE_MODE` 切 trained / random 等读出对照） | `slim_pipeline_tievar.py` |
| 解绑对照（每个辅助头私有读出） | `slim_pipeline_untied.py` |
| 单进程 SOLO 与解绑版（340M 四臂归因，由 `run_lwce.py` 驱动） | `slim_solo.py`，`slim_untied.py`，`run_lwce.py` |
| 局部激活重放 | `slim_pipeline_replay.py` |
| 2B BP（FSDP / ZeRO-3） | `slim_bp_fsdp.py` |
| ckpt 导出 HuggingFace（带逐 logit 对拍） | `export_hf.py` |
| BP ckpt 切成流水线分段 / 分段合回 BP | `bp_to_pipe.py`，`hybrid_convert.py` |
| 数据预分词 | `prep_slimpajama.py` |
| 里程碑损失收缩式恒等式回归测试 | `ms_identity_test.py` |

## 模型与协议

所有臂共用 Transformer++（Llama 结构：RMSNorm、rotate-half RoPE、SwiGLU，无权重绑定）、Mistral-32k 词表、SEQ=2048。

| 规模 | NLAYER | DMODEL | NHEAD | FFN | 参数量 |
|---|---|---|---|---|---|
| 340M | 24 | 1024 | 16 | 0（自动 2816） | ≈374M |
| 1.3B | 24 | 2048 | 16 | 0（自动 5632） | ≈1.36B |

优化：`LR=3e-4 LR_MIN=3e-5 WARMUP_TOKENS=5e8 WD=0.01 CLIP=1.0`，有效批 524,288 tok/步，`TOKENS=15000000000`，`opt_steps=28,610`。

有效批在 BP 臂为 `BS×SEQ×ACCUM×卡数`，在流水线臂为 `BS×SEQ×ACCUM×R`，其中 `R = WORLD_SIZE / K`，不乘 K。各臂的微批取法：

| 规模 | 臂 | 卡数 | 微批 |
|---|---|---|---|
| 340M | BP | 4 | `BS=8 ACCUM=8` |
| | K=2×R=2 | 4 | `K=2 BS=8 ACCUM=16` |
| | K=4 | 4 | `K=4 BS=8 ACCUM=32` |
| 1.3B | BP | 4 | `BS=4 ACCUM=16` |
| | K=2×R=2 | 4 | `K=2 BS=8 ACCUM=16` |
| | K=4 | 4 | `K=4 BS=16 ACCUM=16` |

## SOLO 流水线的局部目标

rank 编号为 `副本×K + 段`。中间段 k 的辅助头读出通过终端读出权重的 detach 副本 `w_t` 计算，每 `TIE_SYNC` 个优化步刷新一次。跨段只有前向激活，没有梯度。

中间段损失 (`LOSS=md`) 对每个位置取 `t_k = (k+1)/K`：

```
m      = softmax((1 - t)·log sg(q) + t·log δ_y^λ)
loss   = (ℓ_q − ℓ_m)_+ + MU·H²(q, m)·1[ℓ_q > ℓ_m] + AUX·ℓ_q
```

`δ_y^λ` 在标签处概率为 `e^{-λ}`，其余均分。`λ` 为分桶棘轮地板：按标签的 unigram 惊奇度 `−log π(y)` 分 `NBUCKET` 桶，每桶取 `min(CFLOOR·mean(−log π(y)), KAPPA·EMA(达成 CE))`，此后单调不增，下界 `LAM0`。`LOSS=ce` 把中间段换成普通 CE，用于里程碑消融。终端段始终是标准 CE。

## 关键环境变量

| 变量 | 默认 | 含义 |
|---|---|---|
| `BIN` | `data/slim_2b.bin` | 预分词 bin，必须放内存盘 |
| `TOKENS` | 0（bin 全部） | 训练 token 预算 |
| `K` | `WORLD_SIZE` | 流水线段数 |
| `SPLIT` | 空（均分） | 各段层数，如 `14,10` |
| `HEAD_DEPTH` | 2 | 辅助头内 transformer 块数 |
| `TIE_SYNC` | 100 | 共享读出副本刷新间隔（优化步）。论文主表用 100；它带来可测的陈旧代价，新实验建议 50 或更小 |
| `MU` / `AUX` / `KAPPA` / `CFLOOR` / `LAM0` / `NBUCKET` | 0.3 / 0.3 / 0.7 / 0.9 / 0.1 / 12 | 里程碑损失参数 |
| `LOSS` | `md` | `md` 里程碑；`ce` 贪心局部 CE |
| `COMPILE` | 0 | 1 融合里程碑损失；2 再编译 trunk 与 head_out |
| `OUT` / `RESUME` | 空 | 存档前缀（按段存 `.r{stage}`）；续训指向同一前缀 |
| `CKPT_STAGE_DIR` | `/dev/shm` | 存档先写内存盘，后台原子搬到 `OUT` |
| `SAVE_EVERY` / `EVAL_EVERY` / `LOG_EVERY` | 2000 / 1000 / 50 | 优化步 |
| `UNI_BIN` / `RATCHET_FROM` | 空 | 热启动时 unigram 与棘轮状态的来源 |
| `FREEZE` / `FREEZE_STAGES` / `HEAD_LR_MULT` / `TRUNK_LR_MULT` / `L2SP` / `FANCHOR` / `MS_W` | 关 | 热启动与后训练用的冻结、学习率倍数和保持锚 |
| `DRIFT_T` | 空 | 记录终端读出每 T 步的相对漂移 |

`slim_bp.py` 的变量是上表中与模型、优化、存档相关的子集，另有 `ARCH`（默认 `llama`）、`DROP`、`TIE`。

## 复现命令

### 数据

```bash
TOKENS=15000000000 OUTBIN=data/slim_15b.bin python prep_slimpajama.py
mkdir -p /dev/shm/slim && cp data/slim_15b.bin data/slim_15b.bin.meta.json /dev/shm/slim/
```

### 训练（以 1.3B 为例，340M 去掉模型形状变量并按上表改微批）

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

启动后核对日志中的 `eff_batch=524,288`、`opt_steps=28,610` 和参数量。崩溃后原样重发同一条命令即可续训，恢复内容包括权重、Adam 动量、步数、调度器、棘轮地板和数据 RNG。流水线各 rank 应打印 `resumed @ step N (full)`，BP 打印 `resumed @ step N`；若流水线打印 `WARM-START weights-only`，说明 ckpt 缺优化器态，调度器会按仅权重热启动处理，正式续训时需要中止排查。

### 评测

```bash
# BP
SRC=bp CKPT=ckpt/bp1b.pt OUTDIR=hf_out/bp1b NLAYER=24 DMODEL=2048 NHEAD=16 python export_hf.py
# 流水线终端 (EXIT=-1) 或中间出口 (EXIT=0..K-2)
SRC=pipe CKPT=ckpt/solo1b_k4.pt K=4 EXIT=-1 OUTDIR=hf_out/solo1b_k4 NLAYER=24 DMODEL=2048 NHEAD=16 python export_hf.py

lm_eval --model hf --model_args pretrained=hf_out/solo1b_k4,dtype=bfloat16 \
  --tasks wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge \
  --batch_size 16 --output_path evals/solo1b_k4
```

导出前会用训练前向的最小复刻与 HF 模型逐 logit 对拍，`max|Δ|>2e-3` 时拒绝落盘。`eval_untie340m_k4.sh` 是 340M K=4 多出口评测的完整脚本。

## 运行注意

- bin 必须放在 `/dev/shm`。NFS 上的 memmap 在 NFS 抖动时会触发 SIGBUS，训练进程会无声退出。
- 日志写本地盘（如 `/tmp`），不写 NFS。
- 进程组与子组 timeout 均为 3h，存档经 `CKPT_STAGE_DIR` 暂存后异步搬运，避免慢写触发 NCCL 看门狗。
