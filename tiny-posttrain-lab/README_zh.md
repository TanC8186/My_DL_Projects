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
| `docs/` | 精读笔记、显存指南、[逐文件代码梳理](./docs/walkthrough/00-index.md) |
| `tests/` | CPU 单测 |
| `data/` | 小样例与下载说明（不提交大文件） |

## 文档导航

| 想看什么 | 去哪 |
|---------|------|
| **每个文件的每段代码在干什么** | [`docs/walkthrough/`](./docs/walkthrough/00-index.md)（10 篇，从 00 开始读） |
| TRL 源码怎么算 SFT loss / DPO loss / GRPO advantage | [`docs/trl-reading-notes.md`](./docs/trl-reading-notes.md) |
| 8G 显存实测数据与 OOM 排查顺序 | [`docs/8gb-memory-guide.md`](./docs/8gb-memory-guide.md) |
| 某个实验的具体结论与局限 | `experiments/*/notes.md` |
| **怎么判断训练是不是真的在工作** | [`walkthrough/09`](./docs/walkthrough/09-experiments.md) |

## 结果

> 第一次跑通后填表。不要在没有结果时宣称「效果更好」。

| 实验 | 模型 | 方法 | 显存峰值 | 指标 | 备注 |
|------|------|------|----------|------|------|
| 2026-09-sft-baseline | Qwen2.5-0.5B-Instruct | SFT+LoRA r=16 | allocated 1.87 / reserved 4.89 GB | train_loss 1.39 / 345s | 首条基线；含 base vs LoRA 生成对比；见 `experiments/2026-09-sft-baseline/notes.md` |
| 2026-09-dpo-from-sft | Qwen2.5-0.5B-Instruct | DPO（续 SFT adapter） | allocated 2.76 / reserved 3.56 GB | accuracy 0.455（随机=0.5）/ 336s | 未产生可测偏好提升；**并发现一个静默训练失效的 bug**，见 `experiments/2026-09-dpo-from-sft/notes.md` |
| 2026-09-grpo-math | Qwen2.5-0.5B-Instruct | GRPO + 可验证 reward | allocated 1.22 / reserved 1.49 GB | held-out 88/100 vs 基座 87/100 / 286s | 链路跑通但**净 +1 题属噪声**（churn）；**发现 gradient checkpointing 会破坏 GRPO 生成**，见 `experiments/2026-09-grpo-math/notes.md` |

## 局限

- 结论仅适用于小模型（≤1.5B–3B），不能外推到 70B+ 工业训练。
- verl / OpenRLHF 等分布式 RL 在本地 8G **不作为可运行目标**，只在 `docs/` 中精读。
- 笔记本 GPU 会降频，耗时不能和数据中心卡对比。

## 建议路线（与实习准备对齐）

1. ~~**本周**：跑通 SFT（0.5B），把显存、loss、样例生成写入 `experiments/`~~ ✅
2. ~~**随后**：DPO 小规模 → GRPO + 可验证 reward~~ ✅
3. ~~**并行**：精读 `../references/trl`~~ ✅ → 结论见 [`docs/trl-reading-notes.md`](./docs/trl-reading-notes.md)
4. **投递前**：把下面的「能讲什么」练成口述，并准备被追问的细节

## 能讲什么（面试素材）

这些不是"我跑了个脚本"，而是**有实测证据的具体结论**：

**机制层面**（对照 TRL 源码，含行号，见 `docs/trl-reading-notes.md`）

- SFT 的 loss 默认算在**全部 token** 上（含 prompt）。`completion_only_loss` 只是开关，
  没有 `completion_mask`/`assistant_masks` 列时它什么都不 mask
- LoRA 省的是**梯度和优化器状态**（约 5 GB → 12 MB），权重本身不省。
  能精确手算参数量：GQA 下 `k_proj`/`v_proj` 的 `out` 是 `num_kv_heads × head_dim`
- DPO 的 `logps` 是**按 token 求和**，长度有偏；靠两次相减抵消模型无关部分
- `ref_model=None` 时参考模型是**原始底座**，不是 SFT 模型（`disable_adapter()` 禁用所有 LoRA）
- GRPO 的 advantage 是**组内相对** `(r_i − mean)/std`，所以组内奖励全同时无梯度

**工程层面**（三个真实 bug，都有完整证据链）

- **静默失效**：`PeftModel.from_pretrained` 的 `is_trainable` 默认 `False` → 优化器为空 →
  权重永不更新，但 `loss` 在降、`grad_norm` 非零。靠 SHA256 比对产物才发现
- **框架陷阱**：TRL 开了 gradient checkpointing 却不在生成前关掉，污染 GRPO 生成，
  奖励恒 0。用"只改模型状态"的二分法定位
- **训练/评测不一致**：一个脚本套 chat template、另一个没套，不报错但结果无法解释

**显存层面**（`docs/8gb-memory-guide.md`）

- `reserved` 才是 OOM 判据（SFT：allocated 1.87 但 reserved 4.89 GB）
- `reserved` 随运行时长漂移，跨实验比显存**必须固定步数**，否则结论是假的

## 许可

本仓库自有代码使用 Apache-2.0。`references/` 下克隆的第三方项目遵循其上游许可证。
