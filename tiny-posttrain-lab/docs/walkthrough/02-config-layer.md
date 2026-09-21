# 02 · 配置层

这一篇讲「参数从哪来」。核心思想：**配置是数据，不是代码。**

| 文件 | 作用 |
|------|------|
| `src/my_lab/config.py` | 20 行，读 YAML + 建目录 |
| `src/my_lab/__init__.py` | 包导出，定义公开 API |
| `configs/sft_qwen05b_8g.yaml` | SFT 全部参数 |
| `configs/dpo_qwen05b_8g.yaml` | DPO 全部参数 |
| `configs/grpo_qwen05b_8g.yaml` | GRPO 全部参数 |

---

## 1. `src/my_lab/config.py` —— 只有两个函数

```python
"""Load YAML training configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return cfg


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
```

### `from __future__ import annotations`

放在每个文件的第一行（docstring 之后）。作用：**把所有类型注解变成字符串，延迟求值。**

为什么需要？看 `def load_config(path: str | Path) -> dict[str, Any]:`

- `str | Path` 是 PEP 604 语法，**Python 3.10 才支持**
- 但有了 `from __future__ import annotations`，注解不会被求值，只是存成字符串
- 于是即使在不支持该语法的版本上，**导入这个模块也不会报错**（只有真正调用
  `typing.get_type_hints()` 时才会）

这是「让代码兼容性更宽」的标准手法，代价几乎为零。

### `load_config` 逐行

```python
path = Path(path)
```
**先把入参统一成 `Path` 对象。** 这样调用方传字符串、传 `Path` 都能工作。
这是 Python 里处理路径的惯例：函数内部只面对一种类型。

```python
if not path.exists():
    raise FileNotFoundError(f"Config not found: {path}")
```
**早失败，且报错信息里带上路径。** 如果不检查，`yaml.safe_load` 会抛出
一个 `FileNotFoundError` 但信息里只有文件名，你在大项目里得猜是哪个目录。
**报错信息要包含足够定位问题的上下文** —— 这是一条被低估的工程原则。

```python
with path.open("r", encoding="utf-8") as f:
```
- `with`：上下文管理器，保证异常时文件句柄也关闭
- `encoding="utf-8"`：**显式指定编码**。Windows 上默认编码是 GBK/cp936，
  而 YAML 里有中文注释（比如 `# 8GB-friendly SFT baseline`）→ 不指定会 `UnicodeDecodeError`

```python
cfg = yaml.safe_load(f)
```
**为什么是 `safe_load` 而不是 `load`？**

`yaml.load()` 可以构造任意 Python 对象，包括执行任意代码的 payload。这是**已知的安全漏洞**。
`safe_load()` 只支持基础类型（dict / list / str / int / float / bool / None）。

即使配置文件是你自己写的，也应该用 `safe_load` —— 因为配置可能来自别处（比如别人提的 PR）。

```python
if not isinstance(cfg, dict):
    raise ValueError(f"Config must be a mapping: {path}")
```
**校验顶层结构。** 如果 YAML 内容是一个列表或纯字符串（比如文件被误截断），
后续 `cfg["model"]` 会报一个晦涩的 `TypeError`。这里提前拦住并给出清晰信息。

### 注意：**只校验到这里为止**

这个函数**不检查** `model` / `data` / `training` 这些 key 是否存在。
为什么？

看调用方（`train_sft.py`）：

```python
model_cfg = cfg["model"]      # KeyError 会自动抛出，信息里有 key 名
data_cfg = cfg["data"]
train_cfg = cfg["training"]
```

`cfg["model"]` 缺 key 时抛 `KeyError: 'model'`，**这个信息已经足够清晰了**。
再加一层校验只是重复劳动。

而叶子字段全部走 `.get(key, default)`：

```python
per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1))
```

**默认值放在代码里，而不是 YAML 里。** 这是一个重要设计选择：

| 方案 | 效果 |
|------|------|
| YAML 里写全所有字段 | 配置文件 100 行长，改一个参数要在噪声里找 |
| **YAML 只写关心的，代码给默认值** ✅ | 配置文件聚焦"这次实验和上次有什么不同" |

所以 `sft_qwen05b_8g.yaml` 里没有 `max_grad_norm` 也能跑（代码里默认 `1.0`），
但写了就覆盖。**配置文件的信噪比因此高很多。**

### `ensure_dir` 逐行

```python
p = Path(path)
p.mkdir(parents=True, exist_ok=True)
return p
```

- `parents=True`：递归创建中间目录（相当于 `mkdir -p`）。没有它，`experiments/a/b/output`
  在 `experiments/a` 不存在时会失败。
- `exist_ok=True`：目录已存在不报错。没有它，第二次跑同一个实验会 `FileExistsError`。
- `return p`：**返回 `Path` 对象**，这样调用方可以链式使用：

```python
out_dir = ensure_dir(train_cfg["output_dir"])
...
trainer.save_model(str(out_dir / "final"))
```

如果返回 `None`，调用方就得自己再 `Path(...)` 一次。

---

## 2. `src/my_lab/__init__.py` —— 定义公开 API

```python
"""tiny-posttrain-lab reusable helpers."""

from my_lab.rewards import accuracy_reward, extract_final_number
from my_lab.memory import estimate_training_vram_gb
from my_lab.vram import (
    cuda_available,
    format_peak_vram,
    peak_vram_gb,
    reset_peak_stats,
)

__all__ = [
    "accuracy_reward",
    "extract_final_number",
    "estimate_training_vram_gb",
    "cuda_available",
    "format_peak_vram",
    "peak_vram_gb",
    "reset_peak_stats",
]
```

### 它在做什么

把子模块的函数**提升到包级别**，于是两种写法都可行：

```python
from my_lab import accuracy_reward          # 简洁（推荐）
from my_lab.rewards import accuracy_reward  # 冗长但明确
```

### 为什么写绝对导入 `from my_lab.rewards import ...`

而不是相对导入 `from .rewards import ...`？

两者都能工作。绝对导入的好处是**在报错栈和 grep 结果里能看到完整路径**，
不用心算"这个 `.` 是几级"。在这个规模的包里，可读性优先。

### `__all__` 的作用

1. **声明公开 API**：`from my_lab import *` 只会导入 `__all__` 里的名字
2. **给工具看**：IDE 补全、文档生成、linter 的未使用导入检查都读它
3. **作为契约**：`tests/test_project_files.py` 就验证这些函数可调用

**注意它导入了什么、没导入什么**：`config.py` 的 `load_config` / `ensure_dir` **不在**
`__all__` 里。这是刻意的 —— 那两个是脚本内部用的工具，不算"库的公开能力"。

---

## 3. 三个 YAML 配置

### 3.1 共同结构

三个文件都是同一套骨架：

```yaml
experiment_id: <实验标识>        # 用于日志和 notes 对应
method: sft|dpo|grpo            # 方法名
notes: <一句话说明>               # 写给人看的
[optional] start_from: <adapter 路径>

model:    { ... }               # 加载什么模型、什么精度
lora:     { ... }               # adapter 怎么配
data:     { ... }               # 用什么数据、多长
training / grpo: { ... }        # 训练超参
output_dir: <产物路径>           # （仅 GRPO 放在顶层，另两个在 training 下）
```

> **注意一个不一致**：SFT/DPO 的 `output_dir` 在 `training:` 下，
> GRPO 的在**顶层**。这是因为 GRPO 脚本写的是
> `cfg.get("output_dir", ...)`，而 SFT/DPO 写的是 `train_cfg["output_dir"]`。
> 这是历史遗留，不是设计 —— **真实项目里这种小不一致很常见，读代码时要以代码为准。**

### 3.2 `sft_qwen05b_8g.yaml` 逐段

```yaml
model:
  model_name_or_path: E:/My_DL_Projects/qwen/Qwen2___5-0___5B-Instruct
  torch_dtype: bfloat16
  load_in_4bit: false
  attn_implementation: sdpa
```

| 字段 | 值 | 为什么 |
|------|-----|--------|
| `model_name_or_path` | 本地绝对路径 | **避免每次重新下载**。注意 `Qwen2___5` 里的三下划线 —— 这是 HuggingFace 把 `.` 替换成 `_` 的目录命名规则（`Qwen2.5` → `Qwen2___5`，因为 `.` 在路径里不安全） |
| `torch_dtype` | `bfloat16` | 2 字节/参数。0.5B 权重 = 1 GB。对比 fp32 省一半显存 |
| `load_in_4bit` | `false` | 0.5B bf16 本来就塞得下，不需要量化；且 `bitsandbytes` 在 Windows 上常装不上 |
| `attn_implementation` | `sdpa` | PyTorch 内置的高效 attention 实现，**零安装成本**。备选 `flash_attention_2` 需要编译，见 01 篇 |

**`bfloat16` 和 `float16` 的区别（面试常问）**：

- `float16`：1 符号位 + 5 指数位 + 10 尾数位。**动态范围窄**（最大约 65504），
  训练中梯度容易溢出成 `inf`
- `bfloat16`：1 符号位 + **8 指数位** + 7 尾数位。指数位和 fp32 一样多，
  **动态范围和 fp32 相同**，代价是精度低（7 位尾数 vs 10 位）

训练用 bf16 是因为**动态范围比精度重要** —— 梯度/激活值的量级跨度很大，
溢出会直接毁掉训练；精度低一点只是噪声大一点。而推理可以用 fp16（无反向传播，不易溢出）。

```yaml
lora:
  enabled: true
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]
```

| 字段 | 含义 |
|------|------|
| `r` | LoRA 的秩。`r=16` 表示用两个 16 列的矩阵 `A[r,in]`、`B[out,r]` 近似权重更新 |
| `alpha` | 缩放系数。实际更新量 = `(alpha/r) * B @ A`。这里 `alpha/r = 2` |
| `dropout` | LoRA 分支上的 dropout，正则化用。0.05 是轻微正则 |
| `target_modules` | 给哪些层加 adapter。这里四个 attention 投影层全加 |

**`alpha` 为什么是 `r` 的两倍？**

LoRA 前向是 `h = Wx + (alpha/r) * BAx`。`alpha/r` 是缩放因子。
原作者建议 `alpha = 2r` 起步，让缩放因子为 2。实践中 `alpha/r ∈ [1, 2]` 都常见。
**关键认知：`alpha` 的绝对值和 `r` 的绝对值都不重要，重要的是 `alpha/r` 的比值**，
以及 `r` 决定的可表达能力。改 `r` 时通常同步改 `alpha` 保持比值。

**`target_modules` 为什么选这四个？**

Transformer 每层的 attention 有 4 个线性投影：Q、K、V、O。
全部加上 adapter 是常见默认。也有只加 `q_proj,v_proj` 的省参数版本。
**不加 MLP 层**（`gate_proj`/`up_proj`/`down_proj`）是权衡：MLP 参数量是 attention 的约 2.7 倍，
加上会让 LoRA 参数量翻几倍。任务越难、数据越多，才越值得加 MLP。

实测：这四个模块在 Qwen2.5-0.5B 上是 **2,162,688 个参数 = 全模型的 0.436%**。
精确推导见 [03 篇](./03-helpers.md)。

```yaml
data:
  dataset_name: yahma/alpaca-cleaned
  dataset_split: train
  max_samples: 2000
  max_seq_length: 512
  packing: false
```

| 字段 | 含义 |
|------|------|
| `dataset_name` | HuggingFace 数据集 ID。也可以改用 `dataset_path` 指向本地 jsonl |
| `dataset_split` | 用哪个 split |
| `max_samples` | **只取前 2000 条**。目的是"边学边跑"，让一次实验只要 6 分钟 |
| `max_seq_length` | 512。**8GB 上最关键的旋钮** —— 激活值显存大致随序列长度线性增长 |
| `packing` | `false`。见下 |

**`packing` 是什么？**

`packing=True` 会把多条短样本**拼接到一条长序列**里（比如 5 条 100 token 的样本
拼成一条 500 token），填满 `max_seq_length`，避免 padding 浪费。

- 好处：**训练吞吐显著提高**（没有 padding 的无效计算）
- 坏处：**不同样本的 token 会在同一个 attention 窗口里互相看见**（除非用
  `position_ids` 和 attention mask 正确隔离，TRL 会处理，但仍是需要注意的点）

这里设 `false` 是为了**结果好解释** —— 8G 场景下首要目标是"跑通且能理解"，
不是"最快"。

```yaml
training:
  output_dir: E:/My_DL_Projects/tiny-posttrain-lab/experiments/2026-09-sft-baseline/output
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4
  lr_scheduler_type: cosine
  warmup_ratio: 0.03
  logging_steps: 5
  save_steps: 200
  save_total_limit: 2
  gradient_checkpointing: true
  bf16: true
  optim: adamw_torch
  max_grad_norm: 1.0
  report_to: []
  seed: 42
  resume_from_checkpoint: null
```

逐字段：

| 字段 | 值 | 解释 |
|------|-----|------|
| `output_dir` | 绝对路径 | 产物落点。`ensure_dir` 会创建它 |
| `num_train_epochs` | 1 | 数据只过一遍。小数据 + 多 epoch 容易过拟合 |
| `per_device_train_batch_size` | 1 | **8G 上必须从 1 开始** |
| `gradient_accumulation_steps` | 8 | 见下 |
| `learning_rate` | 2e-4 | **LoRA 的学习率比全参微调高 10~100 倍**（全参典型 1e-5 ~ 2e-5）。原因：只训练少量新参数，需要更大的步长才能有效更新 |
| `lr_scheduler_type` | cosine | 余弦退火，从 lr 平滑降到 ~0 |
| `warmup_ratio` | 0.03 | 前 3% 的步数线性升 lr，避免一开始步长过大破坏预训练权重 |
| `logging_steps` | 5 | 每 5 步打一次日志 |
| `save_steps` | 200 | 每 200 步存一个 checkpoint |
| `save_total_limit` | 2 | 只保留最近 2 个，防止磁盘爆掉 |
| `gradient_checkpointing` | true | 见下 |
| `bf16` | true | 混合精度训练 |
| `optim` | adamw_torch | 用 PyTorch 原生 AdamW（不用 `bitsandbytes` 的 8-bit 版本，避免额外依赖） |
| `max_grad_norm` | 1.0 | 梯度裁剪阈值 |
| `report_to` | `[]` | 不上报任何平台。想看曲线改 `["tensorboard"]` |
| `seed` | 42 | 随机种子。**实测同 seed 重跑 loss 差 0.03%**，说明可复现 |
| `resume_from_checkpoint` | null | 断点续训。`null` 表示从头开始 |

#### 深入：`batch_size × accum = 有效批量`

```text
per_device_train_batch_size = 1        ← 一次 forward 只处理 1 条样本
gradient_accumulation_steps  = 8       ← 累积 8 次梯度才更新一次参数

有效批量 = 1 × 8 = 8 条样本
250 步 = 2000 条 / 8
```

**为什么这么设？** 因为显存限制的是"一次 forward 能塞多少"，
不是"一个优化步能用多少数据"。

```text
显存受限 → batch_size = 1
           ↓
但 batch_size=1 的梯度噪声极大、训练不稳定
           ↓
用 accum 累积 8 步梯度再更新，等效于 batch_size=8
           ↓
代价：训练变慢（8 次 forward 才更新一次），但显存不变
```

**这是一个纯用时间换显存的技巧**，也是 8G 场景最常用的手段。

#### 深入：`gradient_checkpointing` 用计算换显存

正常反向传播需要保存**每一层**的中间激活值（因为链式法则要用）。
一个 512 token × 896 hidden × 24 层的模型，激活值可能比权重还大。

`gradient_checkpointing=True` 的做法：

```text
前向：只保存每一层的"输入"，不保存中间激活
反向：需要某层的激活时，从最近的 checkpoint 重新前向算一遍
```

- **省显存**：激活值从 O(层数) 降到 O(sqrt(层数)) 或 O(1)（取决于实现）
- **代价**：多一次前向，训练时间增加约 20~30%

本项目实测：SFT 训练峰值 `allocated 1.87 GB`，其中底座权重就占约 1.0 GB，
说明激活值确实被压得很低。

> ⚠️ **重要**：这个开关在 GRPO 里**必须关掉**，原因见 [06 篇](./06-train-grpo.md) ——
> 那是一个让训练完全失效的 bug。

#### 深入：`max_grad_norm` 梯度裁剪

```python
# 伪代码：Trainer 内部
total_norm = sqrt(sum(p.grad.norm() ** 2 for p in params))
if total_norm > max_grad_norm:
    scale = max_grad_norm / total_norm
    for p in params:
        p.grad *= scale
```

**为什么要裁剪？** 单个异常样本（比如一条特别长或特别怪的文本）可能产生巨大的梯度，
一步就把模型推离好区域。裁剪把梯度向量的**整体范数**限制在 1.0 以内，
方向不变、幅度受限。

注意它裁的是**所有参数梯度的联合范数**，不是逐个参数裁。
所以 `clip_grad_norm_` 返回的正是这个范数 —— 这就是日志里 `grad_norm` 的来源。

**这个值后来成了重要线索**：DPO 实验里 `grad_norm` 非零但权重没变（05 篇），
GRPO 实验里 `loss` 为 0 但 `grad_norm` 到 67（06 篇）。

### 3.3 `dpo_qwen05b_8g.yaml` 的差异部分

```yaml
start_from: E:/My_DL_Projects/tiny-posttrain-lab/experiments/2026-09-sft-baseline/output/final
```

**这是 DPO 与 SFT 最大的结构差异。** 它让 DPO 从 SFT 的 adapter 继续训练，
形成真实的 `SFT → DPO` 流水线，而不是从零开始一个无关的 adapter。

脚本里的解析逻辑：

```python
adapter = args.adapter or cfg.get("start_from")
```

**优先级：命令行 > 配置文件。** 命令行用于临时实验，配置文件用于记录"这次实验到底怎么跑的"。

```yaml
lora:
  # Used only when `start_from` is absent (fresh adapter).
  enabled: true
```

**这个 `lora` 块在有 `start_from` 时是被忽略的**（脚本里 `peft_config = None`）。
保留它有两个作用：1) 作为"没有 start_from 时的默认值"；2) 让配置结构统一。

```yaml
data:
  dataset_name: trl-lib/ultrafeedback_binarized
  max_prompt_length: 256
  max_length: 512             # prompt + completion budget
```

**DPO 的数据格式要求**：需要 `chosen` / `rejected` 两列（可选 `prompt`）。
`ultrafeedback_binarized` 的列是 `['chosen', 'rejected', 'score_chosen', 'score_rejected']`
—— **没有 `prompt` 列**，prompt 藏在 chosen/rejected 的消息列表里。
TRL 会用 `maybe_extract_prompt` 自动抽出来。

`max_length` 是 `prompt + completion` 的总预算，而 `max_completion_length` 默认
是 `None`，此时 TRL 用 `max_length - max_prompt_length = 512 - 256 = 256`。

```yaml
training:
  learning_rate: 5.0e-6
  beta: 0.1                   # DPO temperature
  loss_type: sigmoid
```

- `learning_rate: 5e-6`：**比 SFT 的 2e-4 低 40 倍**。DPO 的 loss 是 log-ratio 形式，
  对 lr 更敏感，太大容易训崩
- `beta: 0.1`：**DPO 的"温度"**。它控制对参考模型的偏离惩罚强度。
  `beta` 越大，越严格地贴近参考模型；越小，越允许偏离。
  论文默认 0.1
- `loss_type: sigmoid`：标准 DPO loss。其他选项见 [05 篇](./05-train-dpo.md)

```yaml
dpo_notes:
  why_cheaper_than_ppo: No critic / no separate reward model training loop; still memory-heavy due to policy+ref forward.
  if_oom: reduce max_length, enable load_in_4bit, or shrink LoRA rank.
```

**这两个 key 是纯文档，代码从不读取它们。** 把"为什么 DPO 比 PPO 省"和
"OOM 了怎么办"写在配置里，好处是**看到配置就看到了上下文**，不用翻别处。

> 这种"配置里放注释性字段"的做法有争议（有人认为是污染配置）。
> 在这个项目里它是可接受的，因为 YAML 的读者主要是人。

### 3.4 `grpo_qwen05b_8g.yaml` 的差异部分

```yaml
start_from: null
```

**显式写 `null`**，而不是省略。虽然 `cfg.get("start_from")` 两种情况都返回 `None`，
但写出来是**声明意图**："这里可以接一个 adapter，但本次实验故意不接。"

文件头注释解释了为什么从原始权重起步：

```yaml
# Starts from the base instruct weights on purpose: this isolates GRPO's own
# contribution. Chaining after SFT/DPO would confound "did GRPO help" with
# "was the starting adapter already better at math".
```

**这是实验设计层面的决定，写在配置里最合适** —— 因为读配置的人一定会问这个问题。

```yaml
lora:
  r: 8                        # smaller rank to save VRAM during RL
  alpha: 16
  dropout: 0.0
```

**为什么 `r` 从 16 降到 8？** GRPO 要同时保存"当前策略"和"旧策略"的 log-probs，
还要跑参考模型（如果 `beta>0`）。显存压力比 SFT 大，所以缩小 adapter。

`dropout: 0.0` —— **RL 场景通常关掉 dropout**。原因是重要性采样比 `π_θ/π_old`
要求两次前向在同一状态下可比，dropout 的随机性会引入额外噪声。

```yaml
data:
  dataset_path: .../math_train.jsonl
  eval_path: .../math_eval.jsonl
  max_samples: 200
  max_prompt_length: 128
  max_completion_length: 192
```

`max_completion_length` 的注释很重要：

```yaml
  # Completions longer than this are truncated, so the model never emits its
  # final number and scores 0. That punishes rambling, which is the intent,
  # but it also adds noise. Raise it if the reward curve looks erratic.
```

**截断会直接变成 0 奖励** —— 因为 reward 函数取的是"文本里最后一个数字"，
被截断的文本里没有正确答案。这在设计上是双刃剑：既惩罚啰嗦，也引入噪声。
（实测：修复 bug 后平均生成长度只有 4.6 token，所以 192 完全够用。）

```yaml
grpo:
  num_generations: 4
  temperature: 1.0
  beta: 0.0
  reward_function: accuracy
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  learning_rate: 1.0e-5
  num_train_epochs: 2
```

| 字段 | 值 | 解释 |
|------|-----|------|
| `num_generations` | 4 | **同一个 prompt 采样 4 次**。这是 GRPO 的核心 —— 组内比较需要多个样本 |
| `temperature` | 1.0 | 采样温度。GRPO 强制 `do_sample=True`（不采样就没有组内多样性） |
| `beta` | 0.0 | **KL 惩罚系数**。0 = 不要参考模型，省一次前向。代价是策略可以漂移较远 |
| `reward_function` | `accuracy` | 指向 `my_lab/rewards.py:accuracy_reward` |
| `gradient_accumulation_steps` | 4 | **不能随便设**，见下 |
| `learning_rate` | 1e-5 | 比 DPO 的 5e-6 高，比 SFT 的 2e-4 低 |
| `num_train_epochs` | 2 | 过两遍。因为生成快（关掉 gradient checkpointing 后），多一遍换更多优化步 |

#### 为什么 `accum` 必须 ≥ `num_generations`

配置里的注释：

```yaml
  # TRL requires generation_batch_size (= batch x accum) to be divisible by
  # num_generations, so accum must be >= 4 here.
```

TRL 内部约束：

```python
if self.generation_batch_size % self.num_generations != 0:
    raise ValueError(
        f"generation_batch_size ({self.generation_batch_size}) must be divisible "
        f"by num_generations ({self.num_generations})."
    )
```

其中 `generation_batch_size = per_device_train_batch_size × gradient_accumulation_steps`。

**这个约束的由来**：TRL 一次生成 `generation_batch_size` 条样本，然后按
`num_generations` 分组算 advantage。如果总量不能被组大小整除，就会出现"半个组"，
无法计算组内均值。

实测报错：`batch=1, accum=2` 时 `generation_batch_size=2`，`num_generations=4` → `2 % 4 ≠ 0` → 报错。

**这个坑的实际教训**：调 `num_generations` 时必须同时检查 `accum`。
配置注释里直接写了这条约束，避免下次再撞。

#### 优化步数的算术

```yaml
  # With 2 epochs that is 200 prompts x 2 / 4 = 100 optimizer steps.
```

```text
200 条 prompt × 2 个 epoch = 400 次 micro-step
400 / accum(4) = 100 次优化器更新
```

**为什么要在注释里算这个？** 因为"100 次更新"是判断实验是否可能有效果的关键数字。
GRPO 实验的结论就是"100 次更新里只有约 18 次有梯度信号"，
如果连总步数都不知道，这个分析根本做不出来。

```yaml
  gradient_checkpointing: false
```

后面跟着 10 行注释解释为什么**必须**是 false。这是全项目最重要的一条配置注释，
内容见 [06 篇](./06-train-grpo.md)。

```yaml
if_oom:
  - lower num_generations to 2
  - lower max_completion_length
  - enable load_in_4bit
  - reduce per_device_train_batch_size to 1 (already set)
  - use smaller model or shorter prompts

note_on_gradient_checkpointing:
  Do NOT turn gradient_checkpointing back on for GRPO. ...
```

**`if_oom` 是排查清单**（有序的、从代价最小的开始），**`note_on_gradient_checkpointing`
是警告**。两者都是给未来的自己看的。

注意 `if_oom` 里有一条 `reduce per_device_train_batch_size to 1 (already set)`
—— 括号里标注"已经是最小值了"，避免你在这条上浪费时间。

---

## 本篇小结：配置层的设计原则

1. **配置是数据。** 脚本读 YAML，YAML 不含逻辑。改参数不用改代码。
2. **默认值放代码，YAML 只写差异。** 配置文件的信噪比决定它可读性。
3. **只校验到该校验的地方。** 顶层类型校验（防 YAML 结构错误），
   叶子字段靠 `.get(default)`。`cfg["model"]` 的 `KeyError` 信息已经够清楚。
4. **注释解释"为什么"，而不是"是什么"。** `r: 16` 不需要注释，
   但"为什么 GRPO 的 gradient_checkpointing 必须是 false"必须写。
5. **把约束写进配置。** `accum ≥ num_generations` 这种 TRL 内部约束，
   写在 YAML 注释里比写在文档里更容易被看到。
6. **允许配置里放纯文档字段。** `dpo_notes` / `if_oom` 代码不读，
   但它们让配置自解释。
7. **以代码为准。** 三个配置的 `output_dir` 位置不一致（两个在 `training` 下、
   一个在顶层）—— 读配置时要对着脚本确认，不能靠猜。
