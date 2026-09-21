# 04 · SFT 训练脚本 `scripts/train_sft.py`

这是三条链路里最简单的一条，也是理解另外两条的基础。
**先把它彻底读懂，DPO/GRPO 只是在它的骨架上加东西。**

---

## 0. 全局结构

```text
train_sft.py
├── 引导段：把 src/ 加入 sys.path（让 my_lab 可导入）
├── build_lora_config(cfg)              → peft.LoraConfig 或 None
├── load_dataset_from_cfg(data_cfg)     → datasets.Dataset
└── main()
    ├── 读配置、建输出目录、打印摘要
    ├── 【模型段】tokenizer + model（+ 可选 4bit / gradient checkpointing）
    ├── 【数据段】加载 + 格式化 + 打印列名
    ├── 【训练段】组装 SFTConfig → SFTTrainer → train()
    └── 【落盘段】存模型、存 metrics、打印显存提示
```

**注意 `main()` 里的 import 都写在函数内部**（`import torch`、`from transformers import ...`）。
原因：`torch` 的导入要几秒，放在函数里可以让 `--help` 和参数错误**秒退**，
不用等导入。这是 CLI 脚本的常见优化。

---

## 1. 引导段：让 `my_lab` 可导入

```python
"""Minimal SFT entrypoint for 8GB laptop configs (TRL SFTTrainer)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without install: add src/ to path
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from my_lab.config import ensure_dir, load_config  # noqa: E402
from my_lab.vram import format_peak_vram, peak_vram_gb, reset_peak_stats  # noqa: E402
```

### 逐行

```python
_ROOT = Path(__file__).resolve().parents[1]
```

- `__file__`：当前脚本的路径。可能是相对的（取决于怎么调用）
- `.resolve()`：转成**绝对路径**并解析符号链接。必须做，否则
  `cd` 到别处再调用会算错
- `.parents[1]`：往上两级。`parents[0]` 是 `scripts/`，`parents[1]` 是项目根

```text
E:/My_DL_Projects/tiny-posttrain-lab/scripts/train_sft.py
                                              ↓ .parents[0]
                            E:/My_DL_Projects/tiny-posttrain-lab/scripts
                                              ↓ .parents[1]
                            E:/My_DL_Projects/tiny-posttrain-lab
```

```python
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
```

- `sys.path` 是 Python 的模块搜索路径列表
- `insert(0, ...)` 插到**最前面**，保证我们的 `my_lab` 优先于任何同名包
- **`if ... not in sys.path` 的判断是必要的**：如果脚本被重复导入，
  不判断会插入多个重复项

**为什么要手动加 path？** 因为项目用 src-layout 且没有 `pip install -e .`。
`my_lab` 不在 `sys.path` 上，所以必须手动加。详见 [01 篇](./01-environment.md)。

```python
from my_lab.config import ensure_dir, load_config  # noqa: E402
```

**`# noqa: E402`** = "module level import not at top of file"。这是**故意的**，
因为必须在 `sys.path` 修改之后才能导入。告诉 linter 别报警。

---

## 2. `build_lora_config` —— 把 YAML 变成 peft 对象

```python
def build_lora_config(cfg: dict):
    from peft import LoraConfig

    lora = cfg.get("lora") or {}
    if not lora.get("enabled", False):
        return None
    return LoraConfig(
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=list(lora.get("target_modules", ["q_proj", "v_proj"])),
        task_type="CAUSAL_LM",
    )
```

### `cfg.get("lora") or {}` 这个惯用法

```python
lora = cfg.get("lora") or {}
```

**双重保护**：

- `cfg.get("lora")` → 如果 key 不存在，返回 `None`（不抛异常）
- `or {}` → 如果上面得到 `None`（或空 dict），用 `{}` 替代

这样后面的 `lora.get(...)` 永远安全。**等价于**：

```python
lora = cfg.get("lora")
if lora is None:
    lora = {}
```

但一行搞定。这是 Python 处理可选嵌套配置的标准写法。

### 返回 `None` 表示"不用 LoRA"

```python
if not lora.get("enabled", False):
    return None
```

返回 `None` 而不是 `LoraConfig()` 空对象，是因为**下游 TRL 用 `None` 表示"全参微调"**：

```python
trainer = SFTTrainer(
    ...
    peft_config=build_lora_config(cfg),   # None 或 LoraConfig
)
```

这是 TRL/PEFT 的约定：`peft_config=None` → 全参训练。

### `task_type="CAUSAL_LM"` 是必须的

它告诉 PEFT 目标是**自回归语言模型**，PEFT 会据此决定：

- 哪些模块是"目标模块"（做名字匹配）
- 前向传播怎么插入 adapter

**如果写成 `SEQ_CLS`（序列分类）会静默产生错误行为** —— 不会报错，但 adapter 挂错位置。

### 参数类型转换

```python
r=int(lora.get("r", 16)),
lora_alpha=int(lora.get("alpha", 32)),
lora_dropout=float(lora.get("dropout", 0.05)),
```

**为什么要 `int()` / `float()` 转换？**

因为 **YAML 的类型推断有时不符合预期**：

```yaml
r: 16            → int ✅
r: 16.0          → float ❌  LoraConfig 期望 int
lora_dropout: 0.05  → float ✅
```

显式转换保证类型正确。**代价是 `int(0.9)` 会静默截断成 `0`** ——
但在这个场景里配置文件是人写的，可接受。

### `target_modules` 的默认值和 YAML 不同

```python
target_modules=list(lora.get("target_modules", ["q_proj", "v_proj"])),
```

**代码默认只有 2 个模块，YAML 里写的是 4 个。** 这是"代码给保守默认，配置给具体值"的体现：
如果不写 `target_modules`，用最小的 LoRA（省显存）；写了就用写的。

---

## 3. `load_dataset_from_cfg` —— 两种数据来源

```python
def load_dataset_from_cfg(data_cfg: dict):
    from datasets import load_dataset

    if data_cfg.get("dataset_path"):
        path = Path(data_cfg["dataset_path"])
        if not path.exists():
            raise FileNotFoundError(
                f"dataset_path not found: {path}\n"
                f"Create a small jsonl or switch to dataset_name in the YAML."
            )
        if path.suffix == ".jsonl":
            ds = load_dataset("json", data_files=str(path), split="train")
        else:
            ds = load_dataset(str(path))
            if isinstance(ds, dict):
                ds = ds[list(ds.keys())[0]]
    else:
        name = data_cfg["dataset_name"]
        split = data_cfg.get("dataset_split", "train")
        ds = load_dataset(name, split=split)

    max_samples = data_cfg.get("max_samples")
    if max_samples:
        ds = ds.select(range(min(int(max_samples), len(ds))))
    return ds
```

### 3.1 两条分支：本地文件 vs HF 数据集

```text
if data_cfg.get("dataset_path"):     ← 有本地路径 → 走本地
else:                                 ← 否则用 dataset_name 联网/缓存加载
```

**优先级：`dataset_path` > `dataset_name`。** 这样配置里可以两个都留着，
注释掉其中一个来切换（`sft_qwen05b_8g.yaml` 里就是这么写的）。

### 3.2 本地文件的分支

```python
if path.suffix == ".jsonl":
    ds = load_dataset("json", data_files=str(path), split="train")
```

**`.jsonl` 用 `load_dataset("json", ...)`**，因为 jsonl（每行一个 JSON）
是 `datasets` 的 `json` loader 支持的格式。

```python
else:
    ds = load_dataset(str(path))
    if isinstance(ds, dict):
        ds = ds[list(ds.keys())[0]]
```

**非 jsonl（比如一个目录，里面是 parquet）** 用通用加载。

```python
if isinstance(ds, dict):
    ds = ds[list(ds.keys())[0]]
```

**这里处理一个 `datasets` 的 API 不一致：**

- `load_dataset(name, split="train")` → 返回 `Dataset`
- `load_dataset(name)` → 返回 `DatasetDict`（一个 dict：`{"train": ..., "test": ...}`）

当没指定 `split` 时返回的是 dict，需要取出第一个 split。
`list(ds.keys())[0]` 拿第一个 key 名，再取值。

> **更好的写法**是 `ds = ds["train"]`，但那样如果数据集没有 `train` split 就会
> `KeyError`。用第一个 key 更宽容，代价是**可能取到 `test` 而不是 `train`**。
> 这是简化带来的隐患 —— 生产代码里应该显式指定。

### 3.3 报错信息里带上建议

```python
raise FileNotFoundError(
    f"dataset_path not found: {path}\n"
    f"Create a small jsonl or switch to dataset_name in the YAML."
)
```

**好的报错 = 问题 + 解决方向。** 第二行直接告诉用户"要么建文件，要么改配置"。
对比一下只写 `raise FileNotFoundError(path)` —— 后者你要自己去想该怎么办。

### 3.4 `max_samples` 的双重保护

```python
max_samples = data_cfg.get("max_samples")
if max_samples:
    ds = ds.select(range(min(int(max_samples), len(ds))))
```

```python
min(int(max_samples), len(ds))
```

**为什么需要 `min`？** 如果配置写 `max_samples: 2000` 但数据集只有 1500 条，
`ds.select(range(2000))` 会抛 `IndexError`。`min` 保证不会越界。

**`if max_samples:` 的假值陷阱**：`max_samples: 0` 会被当成"不限制"（因为 `0` 是假值）。
在这个场景里 `0` 没有意义，所以不是问题。但如果配置想表达"取 0 条"就做不到 ——
**这是"用假值判断代替 `is None`"的典型代价。**

---

## 4. `main()` 逐段

### 4.1 参数与配置

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="tiny-posttrain-lab SFT")
    parser.add_argument(
        "--config",
        default=str(_ROOT / "configs" / "sft_qwen05b_8g.yaml"),
        help="Path to YAML config",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    out_dir = ensure_dir(train_cfg["output_dir"])
```

**`default` 指向项目内的配置**，所以不带参数直接 `python scripts/train_sft.py` 就能跑。

**注意 `cfg["model"]` 用的是 `[]` 而不是 `.get()`** —— 这三个是**必需字段**，
缺了就应该立刻失败（`KeyError: 'model'` 信息足够清楚）。见 [02 篇](./02-config-layer.md)。

**`out_dir = ensure_dir(...)`** 立刻创建输出目录。这样后面的
`trainer.save_model(str(out_dir / "final"))` 不需要再建目录。

```python
    print(f"[info] experiment: {cfg.get('experiment_id', 'unnamed')}")
    print(f"[info] model: {model_cfg['model_name_or_path']}")
    print(f"[info] output: {out_dir}")
```

**启动时打印三个关键信息**：实验名（对应 `notes.md`）、模型路径（确认加载对了）、
输出目录（确认产物落点）。日志里 `[info]` 前缀方便 grep。

### 4.2 模型段

```python
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTConfig, SFTTrainer

    model_name = model_cfg["model_name_or_path"]
    dtype_name = str(model_cfg.get("torch_dtype", "bfloat16")).lower()
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype_name, torch.bfloat16)
```

#### dtype 的字符串→对象映射

**为什么要用 dict 查表而不是 `if/elif`？**

```python
# 查表版（现在的写法）
torch_dtype = {"bfloat16": torch.bfloat16, ...}.get(dtype_name, torch.bfloat16)

# if/elif 版
if dtype_name == "bfloat16":
    torch_dtype = torch.bfloat16
elif dtype_name == "float16":
    ...
```

查表版**更短、更好扩展**（加一个新 dtype 只改一行），而且 `.get(name, default)`
天然处理了"未知值"的情况（回退到 bf16）。

**`str(...).lower()` 的作用**：容错。YAML 里写 `BFloat16`、`bfloat16`、`Bfloat16`
都能匹配。

#### ⚠️ 未知 dtype 会静默回退

```python
}.get(dtype_name, torch.bfloat16)
```

如果配置里写 `torch_dtype: float8`（拼错或不存在），**不会报错**，
会静默用 bf16。这是一个**潜在陷阱** —— 用户以为用了 fp16，实际是 bf16。

**更严谨的写法**是显式报错：

```python
if dtype_name not in _DTYPE_MAP:
    raise ValueError(f"Unknown torch_dtype: {dtype_name}")
```

这是"宽容 vs 严格"的经典权衡。这里选了宽容（为了少写样板代码），
但**知道这个取舍比不知道好**。

#### tokenizer 的设置

```python
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
```

**`trust_remote_code=True`**：允许加载模型仓库里的自定义代码（比如 Qwen 的
特殊实现）。**注意这是安全风险** —— 它会执行远程代码。对可信的官方模型
（Qwen、Llama）可以接受。

**`pad_token` 的兜底：**

```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

**为什么需要？** 很多 decoder-only 模型（包括 Qwen）**没有专门的 pad token** ——
它们训练时不需要 padding（一条一条训练）。

但 HuggingFace 的 Trainer 需要**批量**处理，所以必须有 pad token。
用 `eos_token` 代替是最常见的做法（因为它在词表里已存在，不会增加 embedding 大小）。

**代价**：pad token 和 eos token 是同一个 id，模型可能分不清"这是填充"还是"这是结束"。
在 SFT 里问题不大，因为 `labels` 里 pad 位置会被设成 `-100`（不参与 loss）。

**`padding_side = "right"`** —— 这是 SFT 与推理的**关键区别**：

| 场景 | padding 侧 | 为什么 |
|------|-----------|--------|
| **训练（SFT）** | `right` | 序列右对齐，`labels` 里 pad 位置的 `-100` 在尾部，shift 后不会污染前面的有效 token |
| **推理（生成）** | `left` | 生成从序列**末尾**继续。左对齐保证所有序列的"最后一个真实 token"位置一致，生成才对齐 |

**如果推理时用 `right` 会怎样？** 不同长度的序列 padding 后，短的序列右侧是 pad，
生成会从 pad 之后开始 → 输出错位。**这是很容易犯的错**，
所以推理脚本（`generate.py` / `eval_math.py`）里都是 `padding_side = "left"`。

#### 模型加载

```python
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }
    if model_cfg.get("load_in_4bit"):
        model_kwargs["load_in_4bit"] = True
        model_kwargs["device_map"] = "auto"
    attn = model_cfg.get("attn_implementation")
    if attn:
        model_kwargs["attn_implementation"] = attn

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
```

**用 `model_kwargs` 字典累积参数再 `**` 展开**，而不是写两个分支的
`from_pretrained` 调用。好处是**条件参数只影响一行**，主调用保持简洁。

**`device_map="auto"` 只和 4bit 一起用。** 为什么？

`load_in_4bit` 需要 `bitsandbytes` 的量化层，它要求用 `device_map` 来分配。
而**非量化场景下加 `device_map` 会带来副作用**：

```python
# 加了 device_map="auto" 的效果：
# - 模型被自动分配到可用设备
# - 但 transformers 会注入 accelerate 的 hooks
# - 训练时这些 hooks 可能干扰 Trainer 的设备管理
```

**本项目的策略**：非量化时**不加** `device_map`，让模型留在 CPU，
由 `Trainer` 自己 `.to(device)`。这是最稳的做法。

> **这个决定后来暴露了一个 bug**：`generate.py` 没有 Trainer，
> 所以模型真的留在 CPU 上了，导致显存报告 `0.0 GB`。
> 详见 [07 篇](./07-infer-eval.md) —— **同一个设计在训练脚本里正确，在推理脚本里是 bug。**

#### gradient checkpointing 与 `use_cache`

```python
    model.config.use_cache = False
    if train_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    reset_peak_stats()
```

**`model.config.use_cache = False` 是必须的。** 原因：

- `use_cache=True` 时，模型保存 KV cache 供**自回归生成**用
- 但训练时**不需要生成**，而且 KV cache 和 gradient checkpointing **冲突**
  （checkpointing 要求重新前向，而 cache 假设前向只做一次）
- transformers 会打警告：`use_cache=True is incompatible with gradient checkpointing`

**这行代码是训练脚本的标配**，几乎所有官方示例都有。

**`gradient_checkpointing_enable()`** 是 PyTorch/transformers 的 API，
把每层的 forward 包一层 `torch.utils.checkpoint`。原理见 [02 篇](./02-config-layer.md)。

> ⚠️ **同一个开关在 GRPO 里会引发灾难**，见 [06 篇](./06-train-grpo.md)。

**`reset_peak_stats()` 放在这里**（模型加载后、训练前）：
让显存峰值计数器从"模型已加载"的状态开始。见 [03 篇](./03-helpers.md) 的讨论。

### 4.3 数据段

```python
    dataset = load_dataset_from_cfg(data_cfg)
    print(f"[info] dataset rows: {len(dataset)}")
    print(f"[info] columns: {dataset.column_names}")
```

**打印 `column_names` 极其重要** —— 因为后面的格式化逻辑依赖列名，
而且 SFT 的 loss 掩码行为也依赖列名（见下）。

实测输出：`['instruction', 'input', 'output']`（alpaca-cleaned）。

#### 把 alpaca 格式转成纯文本

```python
    # Alpaca-style prompt — adjust if your data uses chat messages only
    def format_example(example: dict) -> dict:
        instr = example.get("instruction") or example.get("prompt") or ""
        inp = example.get("input") or ""
        out = example.get("output") or example.get("response") or example.get("completion") or ""
        if inp:
            text = (
                f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
            )
        else:
            text = f"### Instruction:\n{instr}\n\n### Response:\n{out}"
        return {"text": text}

    # If dataset already has "text", keep it; else try format
    if "text" not in dataset.column_names:
        dataset = dataset.map(format_example)
```

**这段代码在做什么**：把结构化的 `{instruction, input, output}` 拼成**一个字符串**。

**为什么要拼成字符串？**

因为 TRL 的 `SFTTrainer` 有两条路径：

| 路径 | 触发条件 | 需要什么 |
|------|---------|---------|
| **语言建模路径** | 数据集有 `text` 字段 | 一整段训练文本 |
| **对话路径** | 数据集有 `messages` 或 `prompt`+`completion` | 结构化消息 |

这里走的是**语言建模路径**（配置里 `dataset_text_field="text"`），
所以需要把结构化数据拍平成文本。

**`.get(k1) or .get(k2) or ""` 的多级兜底**：

```python
instr = example.get("instruction") or example.get("prompt") or ""
```

**用 `or` 串联而不是 `if/elif`**，是因为这里只是想"取第一个非空的"。
`or` 短路特性天然实现这个语义。**注意**：如果 `instruction` 是空字符串 `""`（假值），
会继续尝试 `prompt` —— 这**正是想要的**行为。

**`map(format_example)` 的行为**：

```python
dataset = dataset.map(format_example)
```

- 对每一条样本调用 `format_example`
- **新增** `text` 字段，**保留**原有字段（`instruction`/`input`/`output` 都还在）

所以最终的 `column_names` 是 `['instruction', 'input', 'output', 'text']`。

**`if "text" not in dataset.column_names`** 保证幂等：
如果数据集本来就有 `text` 字段（比如 GRPO 的数据），不做任何处理。

#### ⚠️ 这段代码决定了 loss 算在哪些 token 上

这是一个**容易被忽略但面试常问**的连锁反应：

```text
数据集列 = ['instruction', 'input', 'output', 'text']    ← 注意：没有 'prompt' 列
        ↓
TRL 内部：if args.completion_only_loss is None:
             self.completion_only_loss = "prompt" in dataset_sample
        ↓
"prompt" not in columns  →  completion_only_loss = False
        ↓
TRL 的 DataCollator：
   labels = input_ids                    ← 默认等于输入
   labels[padding] = -100                ← 只 mask padding
   # completion_only_loss 为 False，所以不 mask prompt
        ↓
结论：loss 算在【全部 token】上，包括 "### Instruction:" 这些指令文本
```

**也就是说，模型在同时学习"预测指令"和"预测回答"。**

这不是 bug，是默认行为。但**它意味着**：

1. loss 里混入了"预测指令"的成分，跨实验比较 loss 要小心
2. 想让模型只学回答，需要：
   - 数据改成 `prompt` + `completion` 两列（对话式），或
   - 数据带上 `assistant_masks` 列（由 chat template 生成）

**`completion_only_loss` 只是一个开关，不是掩码生成器** ——
即使设成 `True`，如果没有 `completion_mask`/`assistant_masks` 列，它也不会 mask 任何东西。
（TRL 源码：`if self.completion_only_loss and "completion_mask" in examples[0]:` —— 注意这个 `and`。）

### 4.4 训练段

```python
    sft_kwargs = dict(
        output_dir=str(out_dir),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "cosine")),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        logging_steps=int(train_cfg.get("logging_steps", 5)),
        save_steps=int(train_cfg.get("save_steps", 200)),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        bf16=bool(train_cfg.get("bf16", True)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        optim=str(train_cfg.get("optim", "adamw_torch")),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        report_to=list(train_cfg.get("report_to") or []),
        seed=int(train_cfg.get("seed", 42)),
        packing=bool(data_cfg.get("packing", False)),
        max_length=int(data_cfg.get("max_seq_length", 512)),
        dataset_text_field="text",
    )
```

#### 为什么先建 dict 再传给 `SFTConfig(**sft_kwargs)`

**关键在第 4.5 节的 `try/except`。** 建 dict 让它可以被"过滤后重传"：

```python
sft_config = TrainingArguments(**{k: v for k, v in sft_kwargs.items() if k != "max_length"})
```

如果直接写 `SFTConfig(output_dir=..., num_train_epochs=..., ...)` 就没有这个灵活性。

#### 每个字段的类型转换

```python
num_train_epochs=float(...)
per_device_train_batch_size=int(...)
bf16=bool(...)
report_to=list(...)
```

**统一做类型转换的原因**：YAML 的类型推断可能给出意外结果。

- `num_train_epochs: 1` → YAML 给 `int`，但 HF 期望 `float` → `float()`
- `bf16: true` → YAML 给 `bool`，但如果写成 `bf16: "true"` → 字符串，
  `bool("true")` 是 `True`，`bool("false")` **也是 `True`**（非空字符串都是真）⚠️

> **`bool()` 的陷阱**：`bool("false") == True`。所以如果用户在 YAML 里
> 把布尔值加了引号，`bool()` 会静默给出错误结果。
> 更严谨的做法是解析字符串，但这里没做 —— **知道有这个坑比不知道好**。

```python
report_to=list(train_cfg.get("report_to") or [])
```

**`or []` 的双重作用**：

1. YAML 里 `report_to:` （空值）→ 解析成 `None` → `or []` 变成 `[]`
2. YAML 里 `report_to: []` → 空列表是假值 → `or []` 也变成 `[]`

**为什么 HF 要求 `list` 而不是 `None`？** 因为 `TrainingArguments` 内部会遍历它。
传 `None` 会 `TypeError`。

```python
dataset_text_field="text",
```

**这个参数是硬编码的，不来自配置。** 因为上面的 `format_example` 就是写死生成 `text` 字段的。
两边必须一致 —— 如果 `format_example` 改成返回 `{"content": text}`，
这里也要改成 `"content"`。

**硬编码在这里是可接受的**，因为两个地方在同一个文件里，改的时候容易看到。

#### 字段名的一个版本陷阱

```python
max_length=int(data_cfg.get("max_seq_length", 512)),
```

**YAML 里叫 `max_seq_length`，TRL 参数叫 `max_length`。**

这个改名是 TRL 版本演进的结果：早期 `SFTTrainer` 的构造参数叫 `max_seq_length`，
后来统一到 `SFTConfig` 后叫 `max_length`。

**所以配置文件和代码用的是不同的名字** —— 读代码时必须注意这种映射，
否则会以为 `max_seq_length` 没生效。

### 4.5 兼容旧版 TRL 的 `try/except`

```python
    try:
        sft_config = SFTConfig(**sft_kwargs)
    except TypeError:
        # Older TRL: pass TrainingArguments + max_seq_length to trainer
        sft_config = TrainingArguments(**{k: v for k, v in sft_kwargs.items() if k != "max_length"})
```

**这段代码在做什么？**

- **新版 TRL（0.21.0）**：有 `SFTConfig` 类，接受 `max_length` 等所有参数
- **旧版 TRL**：没有 `SFTConfig`，只能用 `TrainingArguments`，
  而 `TrainingArguments` **不认识** `max_length` / `dataset_text_field` / `packing` 这些
  SFT 专属参数 → 会抛 `TypeError`

所以：**先试新版，失败就退到旧版**。

#### ⚠️ 这段代码有几个真实问题

**问题 1：过滤不完整。** 只过滤了 `max_length`，但 `dataset_text_field` 和 `packing`
也是 `TrainingArguments` 不认识的。旧版路径下仍会 `TypeError`。

**问题 2：`TypeError` 捕获过宽。** 如果 `SFTConfig` 因为**别的原因**抛
`TypeError`（比如某个参数值类型不对），也会被误判成"版本不兼容"，
然后走旧版路径，产生更难理解的报错。

**问题 3：静默降级。** 走到旧版路径时**没有任何警告**。用户不知道
`max_length` 被丢弃了。

**更稳妥的写法**是用特性探测而不是异常捕获：

```python
try:
    from trl import SFTConfig
    sft_config = SFTConfig(**sft_kwargs)
except ImportError:
    sft_config = TrainingArguments(**filtered_kwargs)
```

**为什么这段代码还留在这里？** 因为它是"防御性兼容"，实际运行时
（TRL 0.21.0）永远走 `try` 分支，`except` 是死代码。
**这是一段技术债** —— 在笔记里标注它比假装它正确要好。

### 4.6 构造 Trainer

```python
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=build_lora_config(cfg),
        processing_class=tokenizer,
    )
```

| 参数 | 说明 |
|------|------|
| `model` | 已经加载好的模型（CPU 上，Trainer 会搬到 GPU） |
| `args` | `SFTConfig`，包含所有超参 |
| `train_dataset` | 已经带 `text` 字段的 Dataset |
| `peft_config` | `LoraConfig` 或 `None`。**TRL 会自己调用 `get_peft_model`** |
| `processing_class` | tokenizer |

#### 关键：`peft_config` 传进来之后发生了什么

**这是理解 LoRA 训练的关键一步。** 当 `peft_config` 非 `None` 时，TRL 内部会：

```text
1. 把 base model 包成 PeftModel（调用 peft.get_peft_model）
        ↓
2. 冻结 base model 的所有参数（requires_grad = False）
        ↓
3. 只让 LoRA 参数 requires_grad = True
        ↓
4. 训练结束后，save_model 只保存 adapter（8.5 MB），不保存底座（1 GB）
```

**所以 `train_sft.py` 里完全没有"冻结底座"的代码** —— 那是 TRL 做的。
**这就是为什么读脚本看不出来 LoRA 到底怎么生效的**：真正的魔法在 TRL 内部。

实测结果（本项目打印）：`trainable parameters: 2,162,688`（占 0.436%）。

> **对比 DPO/GRPO**：那两个脚本在**传了 adapter 路径**时把 `peft_config` 设成 `None`
> （因为 adapter 已经手动挂上去了），此时 TRL 不再包装模型。
> **这个差异是 DPO 那个静默 bug 的根源**，见 [05 篇](./05-train-dpo.md)。

#### `processing_class` 而不是 `tokenizer`

**注意参数名。** 旧版 TRL 用 `tokenizer=...`，新版改用 `processing_class=...`。

改名原因：新版本要同时支持纯文本模型（tokenizer）和多模态模型
（processor，包含图像处理器）。`processing_class` 是更通用的名字。

**这个改名意味着**：照抄旧教程的 `tokenizer=tokenizer` 在新版上会
`TypeError: unexpected keyword argument`。

### 4.7 训练与落盘

```python
    result = trainer.train(
        resume_from_checkpoint=train_cfg.get("resume_from_checkpoint") or None
    )
    vram = peak_vram_gb()
    print(f"[info] peak VRAM: {format_peak_vram()}")
    trainer.save_model(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))
```

#### `resume_from_checkpoint` 的处理

```python
resume_from_checkpoint=train_cfg.get("resume_from_checkpoint") or None
```

**为什么要 `or None`？**

因为 YAML 里 `resume_from_checkpoint: null` 解析成 `None`，
但如果是空值也可能解析成 `None`。`or None` 保证：

- `None` → `None`（从头训练）
- `"path/to/ckpt"` → 用该路径续训

**注意：`trainer.train()` 的这个参数不接受 `False`**，必须是路径或 `None`。
如果配置里写 `resume_from_checkpoint: false`，`False or None` → `None`，
意外地正确处理了。**这是 `or None` 顺手解决的一个边界情况。**

#### 先取 VRAM 再 save_model

```python
vram = peak_vram_gb()           # ← 先读
print(...)
trainer.save_model(...)         # ← 后存
```

**顺序重要吗？** 在这里影响很小（保存模型不会显著增加显存）。
但**逻辑上应该先读峰值**，因为保存可能触发额外的显存分配
（序列化时可能复制张量），污染峰值数字。

**这是一条通用原则：测量要在被测行为结束后立刻做，不要插入其他操作。**

#### 保存的两个东西

```python
trainer.save_model(str(out_dir / "final"))
tokenizer.save_pretrained(str(out_dir / "final"))
```

**为什么 tokenizer 也要存？**

因为推理时要保证**用的是同一个 tokenizer**。如果底座模型更新了 tokenizer
（比如新增了特殊 token），用新 tokenizer 加载旧 adapter 会产生 id 错位。

**存 tokenizer 的代价**：11 MB（`tokenizer.json` 就有 11 MB）。
**收益**：可复现性。

实测产物：

```text
output/final/
├── adapter_config.json          # 0.9 KB  ← LoRA 配置（r/alpha/target_modules）
├── adapter_model.safetensors    # 8.5 MB  ← LoRA 权重
├── tokenizer.json               # 11 MB
├── tokenizer_config.json
├── chat_template.jinja          # ← chat 模板，推理必需
├── vocab.json / merges.txt
└── ...
```

> **注意 `chat_template.jinja`** —— 它决定了 `apply_chat_template` 的行为。
> 存下来保证推理时用的模板和训练时一致。**这个文件在 GRPO 那个 bug 里是关键角色。**

### 4.8 写 metrics 与显存提示

```python
    metrics_path = out_dir / "train_metrics.txt"
    metrics_text = str(result.metrics) + "\n" + format_peak_vram() + "\n"
    metrics_path.write_text(metrics_text, encoding="utf-8")
    print(f"[done] metrics -> {metrics_path}")
    print(f"[done] model   -> {out_dir / 'final'}")

    if vram:
        notes_hint = out_dir.parent / "notes.md"
        print(
            f"[hint] record into {notes_hint}:\n"
            f"       peak_allocated={vram['peak_allocated_gb']} GB, "
            f"peak_reserved={vram['peak_reserved_gb']} GB"
        )
```

#### `result.metrics` 里有什么

`trainer.train()` 返回的 `TrainResult.metrics` 是 HF 的标准指标字典：

```python
{'train_runtime': 344.7117,
 'train_samples_per_second': 5.802,
 'train_steps_per_second': 0.725,
 'total_flos': 697581793920000.0,
 'train_loss': 1.3924129104614258}
```

**关键认知：这里没有 loss 曲线，只有最终平均值。** 想看曲线要：

- 解析 `train_log.txt`（里面有每 `logging_steps` 的 `{'loss': ..., 'grad_norm': ...}`）
- 或开 `report_to: ["tensorboard"]`

#### 把显存和 metrics 写在同一个文件

```python
metrics_text = str(result.metrics) + "\n" + format_peak_vram() + "\n"
```

**为什么合并？** 因为它们是同一件事的两半 ——
"这次训练的结果"包括"指标"和"资源消耗"。分开写两个文件更容易漏掉一个。

**这个文件从 gitignore 里被豁免出来**（见 [01 篇](./01-environment.md)），
因为它只有 0.2 KB 且是唯一的原始事实记录。

#### `[hint]` 打印的设计意图

```text
[hint] record into .../notes.md:
       peak_allocated=1.87 GB, peak_reserved=4.89 GB
```

**脚本没法自己写 `notes.md`**（那是人写的解读），但它可以：

1. 告诉你该写哪个文件
2. **把要填的数字直接打印出来**（省得你去日志里翻）

**这是一个"降低记录成本"的设计** —— 记录的成本越低，越可能真的被记录。
实测这个提示直接导致了 `notes.md` 里显存一节被正确填写。

#### `if vram:` 的意义

```python
if vram:
    ...
```

`vram` 是 `peak_vram_gb()` 的返回 —— **无 CUDA 时是空 dict**（假值）。
所以这个判断自动处理了"没有 GPU"的情况：不打印无意义的提示。

**这就是 [03 篇](./03-helpers.md)里"返回空 dict 而不是抛异常"的价值体现。**

---

## 5. 实测结果与解读

跑完一次 SFT 的完整输出：

```text
[info] experiment: 2026-09-sft-baseline
[info] model: E:/My_DL_Projects/qwen/Qwen2___5-0___5B-Instruct
[info] output: .../experiments/2026-09-sft-baseline/output
[info] dataset rows: 2000
[info] columns: ['instruction', 'input', 'output']
{'loss': 1.6729, 'grad_norm': 1.26, 'learning_rate': 0.0001, 'epoch': 0.02}
...
[info] peak VRAM: peak_allocated=1.87 GB, peak_reserved=4.89 GB
[done] metrics -> .../output/train_metrics.txt
[done] model   -> .../output/final
```

### 怎么判断这次训练是健康的

| 指标 | 健康范围 | 本次实测 | 说明 |
|------|---------|---------|------|
| `loss` 起点 | 应接近 `ln(vocab_size)` 附近 | 1.67 | Qwen 词表 ~150k，`ln(150000)≈11.9`。**1.67 远低于此**，因为这是 **Instruct 模型**（已经预训练+指令微调过），不是随机初始化 |
| `loss` 趋势 | 应下降 | 1.67 → 1.39 | ✅ 降了 17% |
| `grad_norm` | 非零、不应持续暴涨 | 0.66 ~ 1.32 | ✅ 稳定，且 `max_grad_norm=1.0` 生效（超过就被裁） |
| `learning_rate` | warmup 后按 cosine 下降 | 0.0001 → ... → 0 | ✅ |
| 显存 | 不 OOM | 1.87/4.89 GB | ✅ |

**注意 `grad_norm` 大部分在 1.0 附近** —— 这说明**梯度裁剪在频繁生效**。
这是正常的（`max_grad_norm=1.0` 是常见默认值），但值得知道：
如果 `grad_norm` 长期等于 1.0 且 loss 不降，说明 lr 太大。

### ⚠️ loss 下降 ≠ 效果变好

这是本项目 SFT 实验最重要的结论。实测的 base vs LoRA 生成对比：

| Prompt | Base | SFT+LoRA |
|--------|------|----------|
| "List the planets..." | 8 个行星全对 | **只列 7 项，且 `Earth (Mars)` 错位** |

**loss 从 1.67 降到 1.39，但事实性问答变差了。**

原因分析（写在 `notes.md` 里）：

- **风格上成功了**：SFT 去掉了 "Certainly! Here are three tips..." 这类客套前缀，
  说明 LoRA 确实学到了 alpaca 数据的分布（`instruction → output` 都是干练问答）
- **能力上退化了**：0.5B 的容量 + 2000 条数据 + 1 epoch，
  不足以在不损失通用能力的前提下重塑风格。LoRA `r=16` 的更新
  足以扰动原本就脆弱的常识

**所以"train_loss 下降"是一个弱指标。** 真正的效果要看 held-out 评测
（→ [07 篇](./07-infer-eval.md)）。

---

## 本篇小结：SFT 脚本的设计要点

1. **引导段是 src-layout 的代价。** 每个脚本重复 4 行 path 操作，
   换来"不需要安装就能跑"。
2. **import 放在函数内。** 让 CLI 的参数错误能秒退，不等 torch 导入。
3. **配置给具体值，代码给保守默认。** `build_lora_config` 的默认
   `["q_proj","v_proj"]` 比 YAML 的 4 个模块更省。
4. **`padding_side` 训练用 `right`、推理用 `left`。** 弄反会导致生成错位。
5. **`use_cache=False` + `gradient_checkpointing_enable()` 是训练标配。**
6. **`completion_only_loss` 是开关不是掩码生成器。** 没有
   `prompt` / `completion_mask` / `assistant_masks` 列时，
   loss 算在**全部 token** 上（本项目就是这种情况）。
7. **LoRA 的"冻结底座"是 TRL 做的，脚本里看不到。** 传 `peft_config` 给它就行。
8. **`processing_class` 是新版参数名**，旧教程的 `tokenizer=` 会报错。
9. **`try/except TypeError` 兼容旧版是一段技术债**，过滤不完整、捕获过宽、静默降级。
10. **测量要紧接着被测行为。** 先读 VRAM 再 save_model。
11. **`[hint]` 把要填的数字直接打印出来。** 降低记录成本 = 提高记录率。
12. **loss 下降不等于效果变好。** 本次 loss 降了 17%，但事实问答退化。
