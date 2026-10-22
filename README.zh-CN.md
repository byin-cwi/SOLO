# SOLO

[English](README.md)

SOLO 把网络切成 K 个梯度隔离的模块：模块之间只传前向激活，不传梯度；每个中间模块挂一个辅助头做局部更新，辅助头的词表/类别读出共享终端读出权重的 detach 副本。本仓库包含论文全部主实验的训练、评测与测量代码。

## 安装

```bash
pip install -r requirements.txt
```

实测环境为 Python 3.12、CUDA 12.x、A100-80GB。所有训练脚本都用环境变量配置，多卡脚本按 `RANK` / `WORLD_SIZE` / `LOCAL_RANK` / `MASTER_ADDR` / `MASTER_PORT` 初始化，可用 `torchrun` 或逐 rank 设环境变量启动。

## 目录

| 目录 | 内容 |
|---|---|
| `lm_slimpajama/` | SlimPajama 340M / 1.3B 预训练（GLA Table-2 协议）、共享读出与解绑对照、局部目标归因、局部激活重放、2B FSDP 基线、HuggingFace 导出 |
| `lm_wt103/` | WikiText-103 上的读出阶梯（tie / untie / random / stale）、规模扫描、公开配方 K 扫、字节级对照、梯度对齐与 logit-lens 探针、隔离性机器验证 |
| `vision/` | CIFAR-10/100、Tiny-ImageNet、ImageNet-1k（ResNet-50/101）、VGG-11 / ViT-Tiny，DSN 对照，视觉读出阶梯，双卡部署 |
| `systems/` | 同一并行方式下 SOLO 与 BP 的吞吐、显存、通信对比（流水线并行与数据并行） |

`lm_slimpajama/CODEMAP.zh-CN.md` 详细说明 SlimPajama 线的文件、配置与复现命令。各脚本文件头列出其用法与全部环境变量。

## 快速自检（CPU，无需数据）

```bash
cd lm_slimpajama
python ms_identity_test.py        # 里程碑损失收缩式与显式实现的恒等性 (fp64 误差约 1e-16)

# 双进程 K=2 流水线, 假数据, 3 步
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29704 WORLD_SIZE=2 BACKEND=gloo FAKE_DATA=1 \
  STEPS=3 NLAYER=4 DMODEL=64 NHEAD=2 SEQ=64 BS=2 ACCUM=2 HEAD_DEPTH=1 EVAL_EVERY=2 \
  VAL_TOKENS=10000 LOG_EVERY=1 TIE_SYNC=2 CKPT_STAGE_DIR=/tmp
RANK=0 python -u slim_pipeline_solo.py & RANK=1 python -u slim_pipeline_solo.py; wait

cd ../lm_wt103                    # 零跨模块梯度的机器验证 (接口发送全部 detach, 读出副本不在优化器)
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29771 WORLD_SIZE=2 BACKEND=gloo FAKE_DATA=1 \
  STEPS=3 EPOCHS=1 NLAYER=4 DMODEL=64 NHEAD=2 FFN=128 SEQ=32 BS=4 HEAD_DEPTH=1
RANK=0 LOCAL_RANK=0 python -u verify_isolation.py & RANK=1 LOCAL_RANK=1 python -u verify_isolation.py; wait
```

## 复现

### SlimPajama（`lm_slimpajama/`）

数据准备、训练、导出与评测命令见 `lm_slimpajama/CODEMAP.zh-CN.md`。参考值（lm-eval-harness 词级 WikiText 困惑度 / 六任务零样本均值）：

| 规模 | BP | SOLO K=2 | SOLO K=4 |
|---|---|---|---|
| 340M / 15B tokens | 28.04 / 41.5 | 29.05 / 41.5 | 31.19 / 41.0 |
| 1.3B / 15B tokens | 21.61 / 45.9 | 22.29 / 45.4 | 23.59 / 45.0 |

### WikiText-103（`lm_wt103/`）

```bash
cd lm_wt103
DATA=./data/wikitext python dl_wikitext.py                       # 下载 WikiText-2/103
# 首次运行流水线前, 用 BUILD_WORD_CACHE=1 运行 lm_hz_tievar.py 构建词级缓存 (建完即退)
GPUS=0,1,2,3 bash run_wt103_ladder.sh tie untie random stale    # 读出阶梯, 8L/d384, V=32768, K=4
GPUS=0,1,2,3 bash run_wt103_scale_sweep.sh                       # 64M–512M 规模扫描
GPUS=0,1,2,3 bash run_wt103_pubrecipe_ksweep.sh                  # 公开配方 (16L/d410, bert 分词) 的 K 扫
```

公开配方使用 bert-base-cased 分词，所需的 `bert_cache_*.pt` 由 `lm_solo.py` 以 `TOKENIZER=bert` 首次运行时构建。

读出阶梯参考值（终端验证困惑度）：tie 70.60，untie 75.35。公开配方 K=2 tie 参考值约 23.61。其余脚本（`run_wt103_hdgrid.sh` 头深度网格、`run_wt103_2x2.sh` 坐标系 2×2、`run_wt103_vocab_sweep.sh` 词表扫描、`run_byte_gate.sh` 字节级、`run_wt103_align_ladder.sh` 对齐探针）的用法见各自文件头。

### 视觉（`vision/`）

```bash
cd vision
python solo_cifar.py                                              # CIFAR, 默认 MODE=md
MODE=bp python solo_cifar.py ; MODE=dsn python solo_cifar.py      # BP 与 DSN 对照
DATA_ROOT=./data/imagenet ARCH=r50in MODE=solo GS_MODE=jacobi SCHED=s3 STRIDED_HEAD=1 \
  HEAD_DEPTH=3 HEAD_WIDEN=2 EMA_M=0.9 python imagenet_solo.py    # ImageNet-1k R50
DATA_ROOT=./data/imagenet ARCH=r50in MODE=bp EPOCHS=90 BATCH=256 python imagenet_solo.py
DATASET=c100 DATA_ROOT=./data GPUS=0,1,2,3 bash run_ladder_abl.sh # 视觉读出阶梯
```

ImageNet 目录需含 `train/` 与 `val/` 的类目录布局。ImageNet R50 参考值：BP 76.49，SOLO（切分 [5|11]）76.27。Tiny-ImageNet、R101、VGG/ViT、双卡部署的用法见 `tiny_solo.py`、`imagenet101_solo.py`、`vgg_vit_solo.py`、`imagenet_standalone*.py`、`imagenet_pipeline_solo.py` 文件头。

### 系统测量（`systems/`）

```bash
cd systems
GPUS=0,1,2,3,4,5,6,7 bash run_pipep_multi.sh    # SOLO+PP vs BP+PP (1F1B) vs BP+DDP, 结束时打印汇总表
bash run_pipep_mlt.sh 0,1,2,3,4,5,6,7            # 固定全局批 G, 扫描每副本微批数 m
```

`run_pipep_mlt.sh`（8 卡）在固定全局批下比较 SOLO 混合布局（`solo_kr2x4`、`solo_kr4x2`、`solo`）与 BP 的全部三维并行调度（1F1B、interleaved、zero-bubble、PP×DP、DDP），输出各臂相对最优 BP 调度的吞吐，以及每步峰值显存和通信量。选项：`G=32 MLS="1 2 4 8" HS="2 1" DRY=1`。

## 实现说明

- **训练器与论文数字的关系。** 本仓库每条线只保留一个最终训练器。论文主表的部分数字由同一算法的早期实现产出，后续改动只涉及 I/O、恢复、编译与并行形态，不改训练数学。等价性证据：`ms_identity_test.py` 的恒等式回归（fp64 误差约 1e-16），以及 340M K=4 在不同编译与微批设置下三次重复 31.22±0.04。`torch.compile` 不保证逐位一致。
- **共享读出刷新间隔。** `TIE_SYNC` 默认 100 个优化步，与论文主表一致。该间隔带来可测的读出陈旧代价，新实验建议设为 50 或更小。
- **WikiText-103 脚本的局部目标。** 显式设 `TK=0` 的脚本（`run_wt103_ladder.sh`、`run_wt103_scalept.sh`、`run_wt103_stalebp.sh`、`run_wt103_vocab_sweep.sh`、`run_wt103_hdgrid.sh`、`run_wt103_2x2.sh`、`run_wt103_align_ladder.sh`、`run_wt103_pubrecipe_ksweep.sh`、`run_wt103_pubrecipe_tiesync.sh`、`run_byte_gate.sh`、`run_lwce_wt2.sh`）配合 `MODE=mfb BETA=1 MU=0 AUX=1`。此时里程碑 m 等于本模块 detach 后的预测，hinge 值恒为 0，中间模块的局部目标退化为局部 CE 加共享读出。由于 `clamp_min` 在 0 处传递梯度，ℓ_q−ℓ_m 的浮点舍入为非负的位置（约 3–5%）辅助梯度加倍，损失值不受影响。未设 `TK` 的 `lm_solo.py` 脚本（`run_wt103_pubrecipe_*_sp.sh`、`run_wt103_pubrecipe_bp_stale.sh`）使用默认 `t_k=(k+1)/K`。SlimPajama 线使用 `t_k=(k+1)/K`、`MU=0.3`、`AUX=0.3`，是完整的里程碑目标。

## 许可

MIT，见 `LICENSE`。
