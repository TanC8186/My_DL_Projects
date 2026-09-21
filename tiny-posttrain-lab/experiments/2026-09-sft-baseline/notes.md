# Experiment: 2026-09-sft-baseline

## 目标

在 RTX 4060 Laptop 8GB 上用 TRL `SFTTrainer` 跑通 Qwen2.5-0.5B-Instruct 的 LoRA SFT，建立后训练实验基线。

## 状态

- [x] 环境安装完成
- [x] 数据确认（alpaca 或本地 jsonl）
- [x] 训练跑通
- [x] 记录显存峰值
- [x] 训练前后生成对比
- [x] 写出结论与局限

## 环境

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- VRAM: 8 GB
- Python: 3.10.18（`E:\My_DL_Projects\tiny-posttrain-lab\.venv`，基于 conda `dl_env`）
- torch / transformers / trl / peft:
  - torch 2.5.1+cu124（CUDA available）
  - transformers 4.55.2
  - datasets 3.6.0
  - accelerate 1.6.0
  - peft 0.17.1
  - trl 0.21.0

## 配置摘要

- config: `configs/sft_qwen05b_8g.yaml`
- model: `E:/My_DL_Projects/qwen/Qwen2___5-0___5B-Instruct`
- data: `yahma/alpaca-cleaned`（max_samples=2000）
- seq_length: 512
- batch / accum: 1 / 8
- LoRA r / alpha: 16 / 32
- dtype: bfloat16 + gradient checkpointing

## 显存

- peak_allocated (GB): **1.87**
- peak_reserved (GB): **4.89**
- 是否 OOM: 否
- 采集方式：`scripts/train_sft.py` 在模型加载后 `torch.cuda.reset_peak_memory_stats()`，训练结束后读 `max_memory_allocated()` / `max_memory_reserved()`

### 为什么 allocated 1.87 GB 而 reserved 4.89 GB？

PyTorch 的 **caching allocator** 会一次性向驱动多申请一些显存缓存起来，避免反复 `cudaMalloc`：

- `allocated` = 真正被张量占用的字节
- `reserved` = 进程已向驱动申请的字节（含缓存块与碎片）
- `nvidia-smi` 看到的是 **reserved** 级别

**结论：判断"会不会 OOM"要看 reserved，而不是 allocated。** 4.89 / 8 GB = 61%，余量比 allocated 给人的印象小得多。只看 1.87 GB 会严重低估风险。

### 构成拆解（0.5B / bf16 / LoRA r=16 / seq=512 / bs=1 / grad ckpt）

| 部分 | 估算 | 说明 |
|------|------|------|
| 底座权重 | 0.5B × 2 B ≈ 1.0 GB | 冻结，无梯度/优化器状态 |
| LoRA 参数 + 梯度 + AdamW (m, v) | ~10 MB | r=16，4 个投影层 |
| 激活值 | 受 gradient checkpointing 显著压缩 | 用重算换显存 |
| 合计 allocated | 1.87 GB | 与估算一致 |

### 复现性验证

同一 config、同一 seed 连跑两次：

| 运行 | train_loss | runtime | samples/s |
|------|-----------|---------|-----------|
| 首次 | 1.391932 | 349.19 s | 5.728 |
| 重跑（采集显存） | 1.392413 | 344.71 s | 5.802 |

loss 相差 0.0005（0.03%），可复现。

## 结果

| 指标 | 值 |
|------|----|
| train_loss | 1.3919 |
| first_log_loss (step 5) | 1.6731 |
| last_log_loss (step 250) | 1.5654 |
| runtime | 349.19 s (~5.8 min) |
| steps | 250 |
| samples/s | 5.728 |
| output | `experiments/2026-09-sft-baseline/output/final` |

### 样例对比（生成证据）

脚本：`python scripts/generate.py --config configs\sft_qwen05b_8g.yaml --compare --max-new-tokens 128`
完整输出：`generation_compare.md`（4 条 prompt，base vs SFT+LoRA 并排）

**观察 1 — 风格被改写（符合预期）**

alpaca-cleaned 是 `instruction → output` 的干练问答，SFT 后模型去掉了对话式开场白：

| Prompt 1「Give three tips for staying healthy.」开头 | |
|---|---|
| Base | `Certainly! Here are three tips to help you stay healthy:` |
| SFT+LoRA | `1. Exercise regularly: Regular physical activity is crucial ...` |

Base 保留 Instruct 模型的礼貌前缀与长段落；SFT 后直接进入编号列表。说明 LoRA 确实学到了数据分布，而不只是"没坏"。

**观察 2 — 能力退化（重要，别忽略）**

Prompt 4「List the planets in the solar system in order.」：

- Base：Mercury / Venus / Earth / Mars / Jupiter / Saturn / Uranus / Neptune — 8 个全对
- SFT+LoRA：只列 7 项，且第 3 项写成 `Earth (Mars)`，Mars 位置错误

Prompt 2「Explain what a GPU is to a 10-year-old.」SFT 后也更含糊（`super power computer in your home`）。

这不是 bug，是 **0.5B 容量 + 2000 样本 + 1 epoch** 的典型后果：数据量和模型容量都不足以在不损失通用能力的前提下重塑风格，LoRA r=16 的更新足以扰动原本就脆弱的常识。

**观察 3 — LoRA 推理显存开销可忽略**

| | peak_allocated | peak_reserved |
|---|---|---|
| Base 推理 | 0.94 GB | 1.00 GB |
| SFT+LoRA 推理 | 0.95 GB | 1.00 GB |

adapter 额外约 10 MB（r=16，4 个投影层）。

**注意**：以上是 `temperature=0.7` 单次采样的结果，**不构成统计结论**。要下判断需 greedy 解码 + 更多 prompt + 重复运行。本表只记录"看到了什么"。

## 问题记录

1. 系统默认 Python 无 torch；实际使用 Anaconda `dl_env` 创建项目 venv（`--system-site-packages`）。
2. `dl_env` 内 TensorFlow 与 NumPy 2.x 冲突：`import peft/trl` 时 transformers 会尝试 import TF 并崩溃/报错。
   - 解决：训练前设置 `USE_TF=0`、`USE_FLAX=0`。
3. transformers 5.8.1 + trl 1.13.0 在 Py3.13 上 `from trl import SFTTrainer` 触发 access violation。
   - 解决：venv 内固定 `transformers==4.55.2`、`trl==0.21.0`、`peft==0.17.1`。
4. 国内网络下载 HF 数据集建议设置 `HF_ENDPOINT=https://hf-mirror.com`。
5. `scripts/generate.py` 初版把模型留在了 CPU：`from_pretrained` 不带 `device_map` 时默认加载到 CPU；训练脚本没暴露这个问题，因为 `Trainer` 内部会 `model.to(device)`。修复：显式 `model.to("cuda")`。
   - 症状是 `torch.cuda.max_memory_allocated()` 报 **0.0 GB** —— 显存指标本身成了 bug 探测器。
6. `.venv` 用 `--system-site-packages` 会暴露 conda `dl_env` 的 TensorFlow（用 NumPy 1.x 编译，当前 NumPy 2.2.6），而 `peft → transformers → image_transforms` 的**惰性导入链**会把 TF 拉起来并崩溃。需 `USE_TF=0`、`USE_FLAX=0`，见 `scripts/env.ps1`。

## 结论与局限

- 8GB 笔记本 GPU 上，Qwen2.5-0.5B + LoRA r=16 + seq512 + bs1/accum8 的 SFT 基线已跑通。
- 1 epoch / 2000 条 alpaca 样本约 6 分钟；loss 从 ~1.67 降到最终 train_loss 1.39（末段 log 在 1.27–1.57 波动）。
- 生成对比显示：SFT **成功重塑了输出风格**（去掉客套前缀、转为编号列表），但**同时出现通用能力退化**（太阳系列表漏项并错位）。两者都是真实结果，只报前者就是选择性汇报。
- `train_loss` 下降 ≠ 效果变好：本次 loss 降了，但事实性问答反而更差。这正说明**需要 held-out 评测**，而不是只看训练 loss。
- 局限：仅 0.5B、仅 2000 条 alpaca、单次采样、无 held-out 指标，结论不能外推到 1.5B/3B 或工业规模训练。
