# tiny-posttrain-lab

面向**单卡 8GB GPU**（RTX 4060 Laptop 级）的 LLM 后训练实验库：SFT / DPO / GRPO 最小可复现配置 + 实验记录结构，用于大模型预训练 / 后训练实习的作品集与面试。

English: [README.md](./README.md)

## 要解决的问题

想投大模型预训练/后训练实习时，本地往往只有一张笔记本显卡。工业框架按集群设计，公开教程又常缺「8G 能跑通」的完整配置。本仓库收集**实测过的最小配置**和清晰的实验目录，方便你把学习过程变成**可展示的开源项目**。

## 本地硬件（实测目标）

| 项目 | 内容 |
|------|------|
| GPU | NVIDIA RTX 4060 Laptop |
| 显存 | 8 GB |
| 模型 | 本地已有 Qwen2.5-0.5B / 1.5B / 3B-Instruct（`E:\My_DL_Projects\qwen\`） |
| 框架 | Hugging Face TRL（源码参考：`../references/trl`） |

## 快速开始

```powershell
cd E:\My_DL_Projects\tiny-posttrain-lab
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt

python scripts\train_sft.py --config configs\sft_qwen05b_8g.yaml
```

Windows 上 `bitsandbytes` 若安装失败，可改用 WSL2，或在配置里关闭 4bit、对 0.5B 使用 bf16 + LoRA。

## 目录说明

| 路径 | 用途 |
|------|------|
| `configs/` | 面向 8G 的 YAML |
| `scripts/` | 训练入口 |
| `src/my_lab/` | 数据、reward、显存估算等可复用代码 |
| `experiments/` | 每次实验一个文件夹（笔记 + 结果） |
| `docs/` | 精读笔记、显存指南 |
| `tests/` | CPU 单测 |
| `data/` | 小样例与下载说明（不提交大文件） |

## 结果

> 第一次跑通后填表。不要在没有结果时宣称「效果更好」。

| 实验 | 模型 | 方法 | 显存峰值 | 指标 | 备注 |
|------|------|------|----------|------|------|
| 2026-09-sft-baseline | Qwen2.5-0.5B-Instruct | SFT+LoRA r=16 | allocated 1.87 / reserved 4.89 GB | train_loss 1.39 / 345s | 首条基线；含 base vs LoRA 生成对比；见 `experiments/2026-09-sft-baseline/notes.md` |

## 局限

- 结论仅适用于小模型（≤1.5B–3B），不能外推到 70B+ 工业训练。
- verl / OpenRLHF 等分布式 RL 在本地 8G **不作为可运行目标**，只在 `docs/` 中精读。
- 笔记本 GPU 会降频，耗时不能和数据中心卡对比。

## 建议路线（与实习准备对齐）

1. **本周**：跑通 SFT（0.5B），把显存、loss、样例生成写入 `experiments/`
2. **随后**：DPO 小规模 → GRPO + 可验证 reward
3. **并行**：精读 `../references/trl` 与 `docs/` 中的 verl/torchtitan 笔记
4. **投递前**：补全结果表、局限、面试问答

## 许可

本仓库自有代码使用 Apache-2.0。`references/` 下克隆的第三方项目遵循其上游许可证。
