# 07 · 推理、评测与数据生成

这一篇讲三个脚本。它们回答一个训练脚本无法回答的问题：
**"训练完了，到底有没有变好？"**

| 文件 | 职责 |
|------|------|
| `scripts/generate.py` | 生成 + base-vs-adapter **定性**对比 |
| `scripts/eval_math.py` | held-out **定量**评测 |
| `scripts/make_math_dataset.py` | 可复现的算术数据集生成器 |

---

## 0. 为什么需要单独一篇讲"评测"

### 训练脚本给出的指标是**不充分**的

`train_sft.py` 输出的是 `train_loss`。它的问题：

| 问题 | 说明 |
|------|------|
| **只是训练集上的平均** | 不知道有没有过拟合 |
| **不反映任务能力** | loss 下降可能只是学会了"预测指令文本"（见 [04 篇](./04-train-sft.md)） |
| **没有基线对比** | 1.39 是好是坏？不知道 |

**实测证据**（SFT 实验）：`train_loss` 从 1.67 降到 1.39（降 17%），
但 base vs LoRA 的生成对比显示**事实性问答变差了**（太阳系漏项错位）。

**所以：训练 loss 下降 ≠ 效果变好。必须有独立评测。**

### 两种评测：定性 vs 定量

| | 定性（`generate.py`） | 定量（`eval_math.py`） |
|---|---|---|
| 输出 | 人读的文本对比 | 一个数字（准确率） |
| 样本量 | 4 条 prompt | 100 条 |
| 能发现 | "风格变了"、"开始胡说" | "提升了 1%，但标准误 3.4%" |
| 不能发现 | 统计显著性 | 具体的退化模式 |
| 成本 | 1 分钟 | 3 分钟 |

**两者都要。** 只有定量会漏掉"为什么错"；只有定性会得出不靠谱的结论。

---

## 1. `scripts/generate.py` —— 生成与对比

### 1.1 整体结构

```text
generate.py
├── 引导段
├── DEFAULT_PROMPTS          ← 内置的 4 条 prompt
├── parse_args()             ← 参数解析
├── resolve_prompts(args)    ← 三种 prompt 来源
├── default_adapter_path(cfg)← 从配置推断 adapter 位置
├── load_model(...)          ← 加载 + 设备放置 + 可选 adapter
├── generate_one(...)        ← 单条生成
├── run_model(...)           ← 一次完整运行（含显存统计）
├── write_report(...)        ← 写 markdown
└── main()                   ← 单次 or --compare 两次
```

### 1.2 内置 prompt 的设计

```python
DEFAULT_PROMPTS = [
    "Give three tips for staying healthy.",
    "Explain what a GPU is to a 10-year-old.",
    "用一句话解释什么是大语言模型的后训练。",
    "List the planets in the solar system in order.",
]
```

**四条 prompt 各有一个目的：**

| # | Prompt | 测什么 |
|---|--------|--------|
| 1 | "Give three tips for staying healthy." | **风格**。base 会说 "Certainly! Here are three tips..."，SFT 后会直接上编号列表 |
| 2 | "Explain what a GPU is to a 10-year-old." | **解释能力 + 受众适配**。0.5B 在这里容易退化 |
| 3 | 中文提问 | **多语言保持**。SFT 数据是英文 alpaca，看中文能力有没有被破坏 |
| 4 | "List the planets in the solar system in order." | **事实记忆**。这是一个有确定答案的知识题，**实测在这里发现了退化** |

**第 4 条是关键。** 前三条只能看出"风格变了"，第 4 条能看出"**知识被破坏了**"：

```text
Base：     Mercury / Venus / Earth / Mars / Jupiter / Saturn / Uranus / Neptune   ← 8 个全对
SFT+LoRA： Mercury / Venus / Earth (Mars) / Jupiter / Saturn / Uranus / Neptune   ← 只 7 项，且错位
```

**设计 prompt 时要包含"有确定答案的知识题"** —— 否则只能看到风格变化，
看不到能力退化。这是一个容易忽略的评测设计要点。

### 1.3 `--compare` 的核心逻辑

```python
    if args.compare:
        adapter_path = Path(args.adapter) if args.adapter else default_adapter_path(cfg)
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"Adapter not found: {adapter_path}\n"
                f"Train first, or pass --adapter / a different --config."
            )
        runs.append(run_model("Base (no adapter)", model_name, cfg, None, prompts, args))
        runs.append(run_model("SFT + LoRA", model_name, cfg, str(adapter_path), prompts, args))
        header.append(f"- adapter: `{adapter_path}`")
        header.append(f"- VRAM base: {runs[0]['vram']}")
        header.append(f"- VRAM adapter: {runs[1]['vram']}")
```

#### `default_adapter_path(cfg)` —— 从配置推断产物位置

```python
def default_adapter_path(cfg: dict) -> Path:
    """`<training.output_dir>/final` — where train_*.py saves the adapter."""
    out_dir = (cfg.get("training") or {}).get("output_dir")
    if not out_dir:
        raise ValueError("config has no training.output_dir; pass --adapter explicitly")
    return Path(out_dir) / "final"
```

**它知道训练脚本的产物约定**：`<training.output_dir>/final`。
所以 `--compare` 不带 `--adapter` 也能自动找到。

**`(cfg.get("training") or {}).get("output_dir")`** —— 链式安全访问：

```python
cfg.get("training")              # → dict 或 None
       or {}                      # → dict（如果上一步是 None）
            .get("output_dir")    # → str 或 None
```

**注意报错信息给出解决方向**：`pass --adapter explicitly`。

#### 为什么 `run_model` 被调用两次而不是一次跑完

```python
runs.append(run_model("Base (no adapter)", ...))
runs.append(run_model("SFT + LoRA", ...))
```

**因为两个模型不能同时驻留显存**（虽然 0.5B 能塞下，但要留余量）。
`run_model` 结尾会释放：

```python
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
```

**这三行的顺序很重要：**

```text
del model, tokenizer      → 删除 Python 引用，对象可被 GC 回收
gc.collect()              → 强制 Python 垃圾回收（立即释放）
torch.cuda.empty_cache()  → 把 caching allocator 的缓存还给驱动
```

**为什么需要 `gc.collect()`？** 因为 PyTorch 张量可能形成**引用环**
（比如 `model` ↔ `optimizer` 互指），单纯 `del` 不会立即回收。
`gc.collect()` 强制处理引用环。

**为什么需要 `torch.cuda.empty_cache()`？** 因为即使 Python 对象被回收，
**显存块只是归还给 allocator 的池子，没有还给驱动**（见 [03 篇](./03-helpers.md)）。
`empty_cache()` 才真正释放。

**没有这三行会怎样？** 第二次 `run_model` 加载模型时，
显存池里还留着第一次的缓存块，可能 OOM。

### 1.4 `load_model` —— 设备放置的坑

```python
def load_model(model_name: str, cfg: dict, adapter: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = cfg.get("model") or {}
    dtype_name = str(model_cfg.get("torch_dtype", "bfloat16")).lower()
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype_name, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"          # ← 注意这里是 left

    model_kwargs: dict = {"trust_remote_code": True, "torch_dtype": torch_dtype}
    attn = model_cfg.get("attn_implementation")
    if attn:
        model_kwargs["attn_implementation"] = attn

    if model_cfg.get("load_in_4bit"):
        model_kwargs["load_in_4bit"] = True
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Without a Trainer, from_pretrained stays on CPU. Move it unless the
    # quantized path already placed it via device_map="auto".
    if "device_map" not in model_kwargs:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"[info] device: {device}")

    model.config.use_cache = True            # ← 推理时开启
    model.eval()

    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model.eval()

    return model, tokenizer
```

#### ⚠️ 这里曾经有一个 bug：模型留在 CPU 上

**原始版本没有这段：**

```python
    if "device_map" not in model_kwargs:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
```

**后果**：`from_pretrained` 不带 `device_map` 时**默认加载到 CPU**。
`generate.py` 没有 `Trainer`（Trainer 会自己 `.to(device)`），
所以模型真的留在 CPU 上运行。

**症状：**

```text
[info] [Model] peak_allocated=0.0 GB, peak_reserved=0.0 GB
```

**显存指标报 `0.0 GB`** —— 因为根本没有用 GPU。

**这个 bug 的价值在于**：**显存指标本身成了 bug 探测器**。

```text
如果没打印显存 → 你会以为推理"就是有点慢"
打印了显存 → 0.0 GB 立刻暴露"没上 GPU"
```

**这就是"记录资源消耗"的额外价值** —— 它不只是给 notes 填数字，
它还是一个**运行时的健全性检查**。

#### `if "device_map" not in model_kwargs` 的判断逻辑

```python
if "device_map" not in model_kwargs:
    model = model.to(device)
```

**为什么用"检查字典里有没有这个 key"而不是"检查配置的 `load_in_4bit`"？**

因为 `device_map` 是**实际传给 `from_pretrained` 的参数**。
判断"这个参数传了没有"比"推断这个参数该不该传"更可靠。

**这是一个"判断事实而不是判断意图"的好例子。**

#### 4bit 路径为什么不手动 `.to()`

```python
if model_cfg.get("load_in_4bit"):
    model_kwargs["load_in_4bit"] = True
    model_kwargs["device_map"] = "auto"
```

**`bitsandbytes` 的 4bit 量化层必须配合 `device_map` 使用** ——
它要在加载时决定每个层放哪个设备，不支持事后 `.to()`。

**如果强行 `.to("cuda")` 会怎样？** 可能报错，也可能静默产生错误行为。
所以这里明确跳过。

#### `use_cache = True` 在推理时是对的

```python
model.config.use_cache = True
```

**与训练脚本相反**（训练时 `False`）。原因：

| 场景 | `use_cache` | 为什么 |
|------|------------|--------|
| 训练 | `False` | KV cache 与 gradient checkpointing 冲突；且训练不需要自回归生成 |
| **推理** | `True` | **生成时必须用**，否则每个 token 重算整条序列 → O(n²) |

**这正是 GRPO 那个 bug 的另一面** —— TRL 因为 gradient checkpointing
强制 `use_cache=False`，导致生成慢 17 倍。

#### `model.eval()` 调用两次

```python
    model.eval()
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
        model.eval()          # ← 第二次
```

**为什么第二次也要调？**

因为 `PeftModel.from_pretrained(...)` **返回一个新的 `PeftModel` 包装对象**，
它的 `training` 状态可能被重置为 `True`（默认）。

**`model.eval()` 的作用**：关闭 dropout、让 LayerNorm 用运行统计量。
**如果忘了调，推理会有随机性** —— 而且不会报错，只是结果不稳定。

> **`PeftModel.from_pretrained` 在这里没有传 `is_trainable=True`，
> 这是对的** —— 推理不需要训练 adapter。**对比 `train_dpo.py` 里的同类调用，
> 那里必须传 `is_trainable=True`**（见 [05 篇](./05-train-dpo.md)）。
> **同一个 API，两种场景，默认值刚好一个对一个错。**

### 1.5 `generate_one` —— 生成参数的正确姿势

```python
def generate_one(
    model, tokenizer, prompt: str, max_new_tokens: int,
    do_sample: bool, temperature: float, top_p: float,
) -> str:
    import torch

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    gen_kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
        # Pass explicit no-op values when greedy so the model's saved
        # generation_config (temperature/top_p/top_k) does not trigger warnings.
        "temperature": temperature if do_sample else 1.0,
        "top_p": top_p if do_sample else 1.0,
        "top_k": 0 if not do_sample else 50,
    }

    with torch.inference_mode():
        output = model.generate(**inputs, **gen_kwargs)

    new_tokens = output[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
```

#### chat template 的正确使用

```python
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

**三个参数各自的含义：**

| 参数 | 值 | 含义 |
|------|-----|------|
| `messages` | `[{"role":"user","content":...}]` | 对话历史 |
| `tokenize=False` | 返回**字符串**而不是 token id | 因为后面还要手动 `tokenizer(text, ...)` 加上 `return_tensors` |
| `add_generation_prompt=True` | **在末尾追加 `<\|im_start\|>assistant\n`** | 告诉模型"轮到你回答了" |

**`add_generation_prompt=True` 是最容易被忘的参数。**

没有它，模板渲染出的文本以 `<|im_end|>` 结尾：

```text
没有 add_generation_prompt：
  ...<|im_start|>user\n问题<|im_end|>\n
        ↑ 模型看到这里，不知道要回答，可能继续生成下一个 user 消息

有 add_generation_prompt：
  ...<|im_start|>user\n问题<|im_end|>\n<|im_start|>assistant\n
                                              ↑ 明确知道该它说话了
```

**实测证据**（GRPO 的排查过程）：渲染出的 prompt 末尾确实是
`<|im_start|>assistant\n`，所以格式是对的。**这个验证是排查 GRPO bug 的关键一步。**

#### `torch.inference_mode()` vs `torch.no_grad()`

```python
with torch.inference_mode():
    output = model.generate(**inputs, **gen_kwargs)
```

| | `no_grad()` | `inference_mode()` |
|---|---|---|
| 关闭梯度追踪 | ✅ | ✅ |
| 额外优化 | 无 | **关闭版本计数、允许更多内存复用** |
| 限制 | 无 | **产生的张量不能在之后用于自动微分** |

**推理用 `inference_mode()` 更快。** 代价是那些张量变成"inference tensor"，
不能参与后续的 autograd。

**⚠️ 这正好解释了 GRPO 那个 bug 的一部分**：
`enable_input_require_grads()` 的 hook 在 `inference_mode` 下试图设置
`requires_grad_(True)`，这个组合会产生异常行为。

#### 提取新生成的 token

```python
    new_tokens = output[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
```

**关键：切片掉输入部分。**

`model.generate()` 返回的是 **prompt + 新生成** 的完整序列：

```text
inputs["input_ids"]         → [1, 46]         （prompt，46 个 token）
output                      → [1, 46 + N]     （prompt + 生成）
output[0][46:]              → [N]             （只有生成部分）
```

**`output[0]`**：`generate` 返回 `[batch, seq_len]`，取第 0 个样本。

**`skip_special_tokens=True`**：解码时跳过 `<|im_end|>`、`<|endoftext|>` 等特殊 token。
**如果为 `False`，输出里会出现这些标记**，读起来很乱。

**`.strip()`**：去掉首尾空白（生成的第一个 token 常常是空格）。

#### 为什么 greedy 时要传"无操作值"

```python
        # Pass explicit no-op values when greedy so the model's saved
        # generation_config (temperature/top_p/top_k) does not trigger warnings.
        "temperature": temperature if do_sample else 1.0,
        "top_p": top_p if do_sample else 1.0,
        "top_k": 0 if not do_sample else 50,
```

**背景**：Qwen 的 `generation_config.json` 里存了默认的
`temperature=0.7, top_p=0.8, top_k=20`。

**`transformers` 会在 `do_sample=False` 时校验这些参数**，并打警告：

```text
The following generation flags are not valid and may be ignored:
['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
```

**问题**：这个警告**每次 `generate` 调用都打一次**。跑 100 条评测 = 100 行警告噪声，
把有用信息淹没了。

**解法：显式传"无操作值"。**

| 参数 | greedy 时传什么 | 为什么是无操作 |
|------|----------------|---------------|
| `temperature` | `1.0` | 采样的温度，greedy 不用。1.0 是"不缩放"的中性值 |
| `top_p` | `1.0` | 核采样阈值，1.0 表示"不截断" |
| `top_k` | `0` | **0 在 transformers 里表示"禁用 top-k 过滤"** |

**为什么这样就消除警告？** 因为校验逻辑是
"`do_sample=False` 且这些值不是中性值 → 警告"。
现在它们是中性值，校验通过。

**这是一个"用正确的值消除噪声"而不是"屏蔽警告"的做法** ——
如果直接 `warnings.filterwarnings("ignore")`，会连同真正的警告一起屏蔽。

> **注意 `top_k=0` 的特殊性**：在 transformers 里，`top_k=0` 和 `top_k=None`
> 都表示禁用，但**只有显式传 `0` 才会覆盖模型自带的 `top_k=20`**。

### 1.6 `write_report` —— 输出 markdown

```python
def write_report(path: Path, header: list[str], prompts: list[str], runs: list[dict]) -> None:
    lines = ["# Generation report", "", *header, ""]
    for i, prompt in enumerate(prompts, 1):
        lines += [f"## Prompt {i}", "", f"> {prompt}", ""]
        for run in runs:
            lines += [f"**{run['label']}**", "", run["outputs"][i - 1] or "_(empty)_", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] report -> {path}")
```

**`["# Generation report", "", *header, ""]`** —— 列表解包。
`*header` 把 header 列表的元素展开插入。

**`run["outputs"][i - 1] or "_(empty)_"`** ——
如果输出是空字符串（模型立刻吐 EOS），显示 `_(empty)_` 而不是空白。

**为什么输出 markdown 而不是 JSON？** 因为这份报告是**给人读的**。
markdown 可以直接在 GitHub / 编辑器里渲染，而且 diff 友好
（改了哪条 prompt 的输出一眼能看出来）。

### 1.7 实测输出

`experiments/2026-09-sft-baseline/generation_compare.md` 的片段：

```markdown
# Generation report

- model: `E:/My_DL_Projects/qwen/Qwen2___5-0___5B-Instruct`
- max_new_tokens: 128
- decoding: sample(t=0.7, top_p=0.9)
- seed: 42
- adapter: `.../2026-09-sft-baseline/output/final`
- VRAM base: peak_allocated=0.94 GB, peak_reserved=1.0 GB
- VRAM adapter: peak_allocated=0.95 GB, peak_reserved=1.0 GB

## Prompt 1

> Give three tips for staying healthy.

**Base (no adapter)**

Certainly! Here are three tips to help you stay healthy:
...

**SFT + LoRA**

1. Exercise regularly: Regular physical activity is crucial ...
```

**注意 VRAM 那一行**：base 0.94 GB → adapter 0.95 GB，**只多 10 MB**。
这印证了"LoRA 推理开销可忽略"（见 [03 篇](./03-helpers.md)）。

---

## 2. `scripts/eval_math.py` —— held-out 定量评测

### 2.1 与 `generate.py` 的关系

**大量代码重复**（`load_model`、`generate_all` vs `generate_one`、
`del/gc/empty_cache`）。**为什么没有抽成共享模块？**

**这是刻意的重复**，理由：

| 抽共享 | 不抽（现状） |
|--------|-------------|
| 改一处两边生效 | 两个脚本可以独立演化 |
| 需要设计抽象接口 | 各自可以自由改动 |
| 但一旦需求分化就要加参数 | 无耦合 |

**两个脚本的需求确实分化了**：

- `generate.py`：要 `--compare` 两次运行、要 4 条 prompt 的并排输出
- `eval_math.py`：要批量 100 条、要算 reward、要写失败案例

**如果强行抽共享**，`load_model` 会需要一堆参数来适配两种场景，
反而更难读。

> **这是一条工程判断**：**"两处相似"不等于"应该抽象"**。
> 抽象的成本是引入耦合，收益要到第三处、第四处才体现。
> **代码重复有时候是更便宜的选择** —— 前提是**知道自己在重复**。

### 2.2 数据加载与 `--limit`

```python
def load_rows(path: Path, limit: int | None) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows
```

**与 `train_grpo.py` 的加载逻辑几乎一样**（同样手写 jsonl 解析）。
**加上两个校验**：

- 文件不存在 → `FileNotFoundError`
- 文件为空 → `ValueError`

**`--limit` 的用途**：快速验证。跑 10 条看看脚本有没有问题，
再跑全量 100 条。**这个"先小后大"的习惯能省很多时间。**

### 2.3 `--greedy` 的默认值设计

```python
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="greedy decoding (deterministic, the honest default for reporting)",
    )
```

**帮助文本里的 "the honest default for reporting" 值得注意。**

**为什么报告应该用 greedy？**

| | 采样（`--greedy` 不传） | 贪心（`--greedy`） |
|---|---|---|
| 结果 | 每次跑不一样 | **确定性的，可复现** |
| 反映 | 一次随机抽样的运气 | 模型最可能的输出 |
| 适合 | 看多样性、看风格 | **报告数字** |

**如果用采样报告准确率**：跑两次得到 87% 和 90%，你无法判断是模型变了
还是运气。**greedy 消除这个变量。**

**`action="store_true"`**：这个参数是**开关**（出现即 `True`），不需要值。
`--greedy` 就是 `True`，不传就是 `False`。

**注意默认值是"采样"而不是"greedy"** —— 因为 `store_true` 的默认是 `False`。
**这在语义上有点别扭**（"greedy 是推荐做法，但不是默认"）。

**更好的设计**是 `--sample` 开关（默认 greedy），
或者用 `argparse.BooleanOptionalAction` 提供 `--greedy/--no-greedy`。
**这是一个可以改进的点。**

### 2.4 `model.generation_config.top_k = 0` 的另一种解法

```python
    model.config.use_cache = True
    # Qwen ships top_k=20 in generation_config.json; transformers warns about it on
    # every greedy call. Neutralise it here (sampling overrides it explicitly).
    model.generation_config.top_k = 0
    model.eval()
```

**这是与 `generate.py` 不同的解法：**

| 脚本 | 解法 |
|------|------|
| `generate.py` | 每次调用 `generate` 时显式传 `top_k=0` |
| `eval_math.py` | **加载后直接改模型的 `generation_config`** |

**后者的好处**：改一次，之后所有 `generate` 调用都干净。
**前者的好处**：不改动模型状态（更"纯"）。

**两种都有效。** 这里用后者是因为 `eval_math.py` 要跑 100 次生成，
在循环外改一次更自然。

**注意注释说 "sampling overrides it explicitly"** ——
采样时 `generate_all` 里传了 `top_k=50`，会覆盖这个 `0`。

### 2.5 批量生成

```python
def generate_all(model, tokenizer, prompts: list[str], args: argparse.Namespace) -> list[str]:
    import torch

    outputs = []
    for i, prompt in enumerate(prompts, 1):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        gen_kwargs: dict = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": not args.greedy,
            "temperature": args.temperature if not args.greedy else 1.0,
            "top_p": 1.0,
            "top_k": 0 if args.greedy else 50,
        }
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        outputs.append(
            tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()
        )
        if i % 25 == 0 or i == len(prompts):
            print(f"[info] generated {i}/{len(prompts)}")
    return outputs
```

#### 逐条生成而不是批量

**为什么不用 `tokenizer(prompts, padding=True)` 一次生成一批？**

**三个原因：**

1. **显存**：批量生成需要 padding 到最长，浪费显存。8G 上不值得冒险
2. **正确性**：批量生成需要正确的 `attention_mask` 处理 padding，
   否则 padding 位置会影响结果（虽然 transformers 会处理，但风险更高）
3. **可解释**：逐条生成时，出问题能精确定位是哪一条

**代价**：速度慢（没有并行）。但 100 条 × 0.23 s ≈ 23 s，完全可接受。

#### 进度打印的粒度

```python
        if i % 25 == 0 or i == len(prompts):
            print(f"[info] generated {i}/{len(prompts)}")
```

**每 25 条打一次，且最后一条必打。**

- **为什么不全打？** 100 条打 100 行日志，噪声太大
- **为什么不只打最后？** 长任务中间看不到进度会让人焦虑，也不知道有没有卡住
- **`or i == len(prompts)`**：保证最后一条一定有输出（即使 100 不是 25 的倍数）

**这是一个小而重要的可用性细节。**

### 2.6 计算 reward 并统计

```python
    outputs = generate_all(model, tokenizer, [r["prompt"] for r in rows], args)
    rewards = accuracy_reward(outputs, [str(r["answer"]) for r in rows])
    vram = format_peak_vram()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    correct = sum(rewards)
    total = len(rewards)
    accuracy = correct / total
```

**`accuracy_reward(outputs, answers)`** —— **直接复用训练时的同一个 reward 函数**。

**这是关键设计**：评测指标和训练奖励**必须是同一个函数**，
否则"训练奖励上升但评测下降"就无法解释。

**`sum(rewards)`** —— `rewards` 是 `[1.0, 0.0, 1.0, ...]`，
求和即正确数。

**先释放模型再统计** —— 因为统计不需要 GPU，早释放让后续操作（写文件）
不会因为显存不足失败（虽然不太可能，但这是好习惯）。

### 2.7 失败案例的收集

```python
    failures = [
        (rows[i]["prompt"], rows[i]["answer"], outputs[i], extract_final_number(outputs[i]))
        for i, r in enumerate(rewards)
        if r < 1.0
    ]
```

**收集 4 元组：`(prompt, 标准答案, 完整输出, 抽取到的数字)`。**

**为什么要 `extract_final_number(outputs[i])`？**

**这是诊断 reward 函数本身是否有问题的关键。**

```text
如果 gold=81 但 pred=1.0  → 说明 reward 函数【读错了】（模型可能答对了但格式不对）
如果 gold=81 但 pred=27.0 → 说明模型【真的答错了】
```

**这两种情况的处理方式完全不同**：
前者要改 reward 函数，后者要改模型。**只看"正确率"无法区分。**

**实测发现**：

```text
[miss] gold=63 pred=83.0 | '83'                      ← 模型真的答错
[miss] gold=81 pred=1.0  | 'To solve the expression \(9 \times 9\), ...'  ← reward 读错了！
```

**第 2 条暴露了 `extract_final_number` 的缺陷**（见 [03 篇](./03-helpers.md) 的 reward misspecification 讨论）。

### 2.8 报告格式

```python
    lines = ["# Math evaluation", "", *header, f"## Failures ({len(failures)})", ""]
    for prompt, gold, out, pred in failures:
        lines += [
            f"### gold={gold} pred={pred}",
            "",
            f"> {prompt}",
            "",
            "```text",
            out[:600],
            "```",
            "",
        ]
```

**标题里直接写 `gold=` 和 `pred=`** —— 这样一眼能看出是"读错了"还是"答错了"。

**`out[:600]` 截断** —— 防止一个超长输出（比如模型写了几千 token 的推理）
把报告撑爆。600 字符足够看到模式。

**用 ```` ```text ```` 代码块包裹** —— 保留原始格式（换行、LaTeX），
不渲染成 markdown。

### 2.9 实测输出对比

**基座（`eval_base.md`）：**

```text
=== accuracy: 87.0/100 = 0.8700 ===
=== VRAM: peak_allocated=0.94 GB, peak_reserved=0.99 GB ===
[miss] gold=280 pred=270.0 | '270'
[miss] gold=513 pred=523.0 | '523'
[miss] gold=65 pred=70.0 | '70'
[miss] gold=369 pred=269.0 | '269'
[miss] gold=340 pred=230.0 | '230'
```

**GRPO 后（`eval_after.md`）：**

```text
=== accuracy: 88.0/100 = 0.8800 ===
[miss] gold=280 pred=270.0 | '270 pages are left.'
[miss] gold=513 pred=523.0 | '523'
[miss] gold=308 pred=288.0 | '288'
[miss] gold=340 pred=330.0 | '330'
[miss] gold=48 pred=56.0 | '56'
```

**注意失败模式**：全是两位数乘法（280→270、513→523、369→269）。
**这直接定位了模型的弱点** —— 如果只看 87% 这个数字，你不会知道问题出在哪。

**还注意到一条**：`'270 pages are left.'` —— 模型**没有遵守"只输出数字"**，
但 reward 仍然正确抽取了 270。**说明 reward 的"取最后一个数字"策略在这种情况下是有效的。**

### 2.10 失败集合的比对（分析的关键一步）

**单独看两次评测的准确率（87% vs 88%）信息量很小。** 真正的分析要比对失败集合：

```powershell
function Get-Fails($p) {
    (Get-Content $p | Select-String -Pattern "^### gold=(\d+) pred=(\S+)") |
        ForEach-Object { $_.Matches[0].Groups[1].Value }
}
$b = Get-Fails eval_base.md
$a = Get-Fails eval_after.md
"fixed by GRPO : $(($b | Where-Object { $a -notcontains $_ }) -join ', ')"
"broken by GRPO: $(($a | Where-Object { $b -notcontains $_ }) -join ', ')"
"still failing : $(($a | Where-Object { $b -contains $_ }) -join ', ')"
```

**结果：**

```text
base failures (13): 280, 513, 65, 369, 340, 121, 73, 260, 129, 624, 27, 69, 229
grpo failures (12): 280, 513, 308, 340, 48, 121, 260, 129, 624, 27, 69, 229

fixed by GRPO : 65, 369, 73
broken by GRPO: 308, 48
still failing : 280, 513, 340, 121, 260, 129, 624, 27, 69, 229
```

**这个分析揭示了"净 +1 是 churn 而不是学习"** ——
如果只是修好了 1 题且没有弄坏任何题，那是真提升；
但"修好 3 题、弄坏 2 题、10 题纹丝不动"说明模型只是**换了几个答案**。

> **这是一条通用的评测方法论**：**看 diff 比看总数更有信息量。**
> 准确率 +1% 可能是"修好 5 题弄坏 4 题"，也可能是"修好 1 题没弄坏"。
> 两者的含义完全不同。

---

## 3. `scripts/make_math_dataset.py` —— 可复现的数据生成

### 3.1 为什么需要生成器而不是手写数据

| | 手写 jsonl | 生成器 |
|---|---|---|
| 答案正确性 | 靠人工检查 | **Python 算出来的，必然正确** |
| 规模 | 写 200 条很痛苦 | 一条命令 |
| 可复现 | 文件本身就是记录 | **需要记录命令行**（所以写进 `data/README.md`） |
| 难度可调 | 手动 | **改一行范围参数** |

**"答案由程序计算"是核心价值。** 对"可验证 reward"的任务来说，
**标准答案错了整个实验就废了**。

### 3.2 九个题目生成器

```python
def _add2(rng: random.Random) -> tuple[str, int]:
    a, b = rng.randint(11, 99), rng.randint(11, 99)
    return f"What is {a} + {b}?", a + b

def _sub2(rng: random.Random) -> tuple[str, int]:
    a, b = rng.randint(11, 99), rng.randint(11, 99)
    if a < b:
        a, b = b, a
    return f"What is {a} - {b}?", a - b
```

#### `_sub2` 里的交换逻辑

```python
    if a < b:
        a, b = b, a
```

**为什么？** 保证 `a >= b`，避免出现负数答案。

**为什么避免负数？** 因为 `_NUMBER_RE = r"-?\d+(?:\.\d+)?"` 虽然支持负号，
但**负号会增加 reward 解析的出错概率**（比如文本里出现 `-` 作为连字符）。
**在生成阶段就避免边界情况，比在 reward 里处理更省事。**

#### 九个生成器的难度分布

| 生成器 | 题型 | 难度 |
|--------|------|------|
| `_add2` | 两位数加法 | 易 |
| `_sub2` | 两位数减法 | 易 |
| `_mul1x2` | 一位 × 两位 | 中 |
| `_mul2x2` | 两位 × 两位 | **难（实测失败集中在这里）** |
| `_div_exact` | 整除 | 中 |
| `_word_pages` | 减法应用题 | 易 |
| `_word_bags` | 乘法应用题 | 中 |
| `_word_speed` | 速度×时间 | 中 |
| `_word_price` | 单价×数量 | 中 |

**实测发现 `_mul2x2` 是模型的主要失败来源**（280→270、513→523、369→269）。
**这验证了"难度分层"的设计是对的** —— 如果全是简单题，
基座准确率会接近 100%，GRPO 就没有学习信号了（见 [06 篇](./06-train-grpo.md)）。

**这也是下一步改进的方向**：加大 `_mul2x2` 的比例，或者引入三位乘法，
把基座准确率压到 40~60%。

#### `_div_exact` 的构造技巧

```python
def _div_exact(rng: random.Random) -> tuple[str, int]:
    b = rng.randint(3, 12)
    q = rng.randint(3, 30)
    return f"Compute {b * q} / {b}.", q
```

**先生成除数 `b` 和商 `q`，再算出被除数 `b*q`。**
这样**保证整除**，答案是整数 `q`。

**如果反过来**（随机生成被除数和除数），大部分除法会有余数，
答案就不是整数，reward 的浮点比较会引入问题。

**"从答案反推题目"是生成可验证数据的常用技巧。**

#### 统一的指令后缀

```python
SUFFIX = "End your response with the number only."
```

**所有 prompt 都以它结尾。** 目的：

1. **明确输出格式要求** —— 提高模型遵守格式的概率（从而让 reward 更容易解析）
2. **统一** —— 避免"有的题要求格式，有的不要求"引入的方差

**实测：模型基本遵守**（输出 `'270'`、`'523'` 这种纯数字），
偶尔多写一点（`'270 pages are left.'`）也不影响 reward 抽取。

### 3.3 `build()` —— 生成去重的池子

```python
def build(n: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    guard = 0
    while len(rows) < n and guard < n * 100:
        guard += 1
        question, answer = rng.choice(GENERATORS)(rng)
        if question in seen:
            continue
        seen.add(question)
        rows.append({"prompt": f"{question} {SUFFIX}", "answer": str(answer)})
    if len(rows) < n:
        raise RuntimeError(f"Only generated {len(rows)} unique rows out of {n}")
    return rows
```

#### `rng.choice(GENERATORS)(rng)` 的语法

```python
question, answer = rng.choice(GENERATORS)(rng)
```

**两步：**

1. `rng.choice(GENERATORS)` → 随机选一个**函数**（比如 `_mul2x2`）
2. `(rng)` → **立即调用它**，把 `rng` 传进去

**等价于：**

```python
gen = rng.choice(GENERATORS)
question, answer = gen(rng)
```

**为什么要传 `rng` 进去而不是让每个生成器自己 `random.randint`？**

因为**这样所有随机性都来自同一个 `Random` 实例**，
只要 `seed` 相同，整个生成过程完全可复现。

**如果每个生成器内部用全局的 `random.randint`**，
那就受全局随机状态影响，无法通过传 `seed` 控制。

**这是"依赖注入"的经典用法** —— 把随机源作为参数传进去，而不是让被调方自己去拿。

#### 去重逻辑

```python
    seen: set[str] = set()
    ...
        if question in seen:
            continue
        seen.add(question)
```

**用 `set` 去重，O(1) 查找。** 如果用 `list`，查找是 O(n)，
生成 300 条时会有 45000 次比较（虽然也能跑，但没必要）。

**为什么需要去重？** 因为随机生成会碰撞。比如
`_add2` 生成 `What is 50 + 50?` 可能出现多次。

**训练集里有重复题会怎样？** 模型可能记住它，扭曲评测。

#### `guard` 防死循环

```python
    guard = 0
    while len(rows) < n and guard < n * 100:
        guard += 1
        ...
    if len(rows) < n:
        raise RuntimeError(f"Only generated {len(rows)} unique rows out of {n}")
```

**为什么需要 `guard`？**

如果请求的 `n` 超过了**可能的题目总数**，循环会**永远转下去**：

```text
假设只想要 1000000 条，但所有生成器加起来只能产生 500000 种不同的题
→ 去重逻辑让 len(rows) 永远到不了 1000000
→ while 条件永远为真 → 死循环
```

**`guard < n * 100`** 给出一个宽松的上限：尝试 `100n` 次还凑不齐就放弃。

**放弃时抛异常而不是静默返回**：

```python
    if len(rows) < n:
        raise RuntimeError(f"Only generated {len(rows)} unique rows out of {n}")
```

**报错信息里有实际数量和期望数量** —— 让你知道差距有多大。

**这是一个"防御性编程"的好例子**：任何 `while` 循环都应该有一个
"最多转多少次"的上限，否则就是在赌它一定会终止。

#### 输出格式

```python
        rows.append({"prompt": f"{question} {SUFFIX}", "answer": str(answer)})
```

**`str(answer)`** —— 转成字符串。因为 jsonl 里存字符串更一致
（虽然 JSON 支持数字，但 `answer` 作为字符串能容纳 `"1/2"` 这类非数字答案）。

**reward 函数里也做了对应处理**：

```python
gold_num = extract_final_number(str(gold))     # ← 再转回数字
```

**存储用字符串，计算用数字** —— 两边显式转换，避免类型混乱。

### 3.4 `split()` —— 保证不重叠的切分

```python
def split(pool: list[dict[str, str]], eval_n: int, seed: int) -> tuple[list, list]:
    """Deterministically split a pool into (eval_rows, train_rows).

    Shuffling one pool before splitting is what guarantees the two files cannot
    overlap; generating them with two independent seeds does not.
    """
    shuffled = list(pool)
    random.Random(seed + 1).shuffle(shuffled)
    if not eval_n:
        return [], shuffled
    return shuffled[:eval_n], shuffled[eval_n:]
```

#### ⚠️ 这是本项目一个真实的错误修复

**原来的做法**（有 bug）：

```powershell
# 生成训练集
python make_math_dataset.py --n 200 --seed 42  --out data/math_train.jsonl
# 生成评测集
python make_math_dataset.py --n 100 --seed 1234 --out data/math_eval.jsonl
```

**看起来合理**（不同 seed 应该产生不同数据），**实测发现 16 条重叠**：

```text
train=200 eval=100 overlapping_prompts=16
```

**为什么？** 因为题目空间有限，两个不同的 seed 仍会撞到相同的题目。

**16 条重叠意味着什么？**

```text
在训练集上见过的 16 道题，也被用来评测
→ 这 16 道题的"正确"可能是【记忆】而不是【泛化】
→ 评测结果被高估
```

**这是数据泄漏（data leakage）的一种。**

#### 修复：单池洗牌后切分

```python
shuffled = list(pool)                    # 复制，不修改原列表
random.Random(seed + 1).shuffle(shuffled)  # 确定性洗牌
return shuffled[:eval_n], shuffled[eval_n:]   # 前 N 条给 eval，其余给 train
```

**为什么这样就保证不重叠？**

因为**两个 split 来自同一个池子的不相交切片**。
数学上：`shuffled[:100]` 和 `shuffled[100:]` 的交集必然是空集。

**而两个独立 seed 生成的两个池子，交集无法保证为空。**

#### `list(pool)` 复制的必要性

```python
shuffled = list(pool)
```

**`random.shuffle()` 是原地操作**（会修改传入的列表）。
如果不复制，调用方的 `pool` 会被打乱 —— **副作用泄漏到调用方**。

**这是一条通用原则**：**函数不应该修改传入的可变对象**，除非名字明确说明
（比如 `list.sort()` 就地排序 vs `sorted()` 返回新列表）。

#### `seed + 1` 的小技巧

```python
random.Random(seed + 1).shuffle(shuffled)
```

**用 `seed + 1` 而不是 `seed`** —— 让洗牌的随机性和生成的随机性**独立**。

**如果都用 `seed`**：生成用的 `Random(seed)` 和洗牌用的 `Random(seed)`
会产生相同的随机序列。虽然在这个场景下不影响正确性，
但**用不同的种子更安全**（避免两个阶段的随机性产生意外的相关性）。

**`+1` 是一个简单但有效的去相关手段。**

#### `if not eval_n` 的边界处理

```python
    if not eval_n:
        return [], shuffled
```

**`eval_n=0` 时返回空 eval + 全部 train。**

**注意这里用的是 `not eval_n` 而不是 `eval_n == 0`** ——
两者在这里等价（`eval_n` 是 int），但 `not` 也覆盖 `None` 的情况（更宽容）。

### 3.5 `main()` 的参数校验

```python
    if args.eval_n and not args.eval_out:
        parser.error("--eval-n requires --eval-out")
    if args.eval_n >= args.n:
        parser.error("--eval-n must be smaller than --n")
```

**用 `parser.error()` 而不是 `raise ValueError`。**

**区别**：

| | `parser.error(msg)` | `raise ValueError(msg)` |
|---|---|---|
| 输出 | 用法说明 + 错误信息 | 只有 traceback |
| 退出码 | 2（约定） | 1 |
| 适合 | **命令行参数错误** | 逻辑错误 |

**参数校验用 `parser.error` 更友好** —— 它会打印 `--help` 摘要，
让用户知道正确用法。

**两条校验的意义**：

1. `--eval-n` 需要 `--eval-out` —— 否则切出来的 eval 集没地方写
2. `--eval-n < --n` —— 否则训练集为空

**这两条都是"防止用户犯低级错误"**，比事后报错更早、更清楚。

### 3.6 `_write()` —— 写出 jsonl

```python
def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
```

#### `newline="\n"` 的重要性

```python
with path.open("w", encoding="utf-8", newline="\n") as f:
```

**Windows 上 Python 默认会把 `\n` 转成 `\r\n`**（文本模式的换行转换）。

**不指定 `newline="\n"` 的后果**：

```text
Windows 生成的文件：每行以 \r\n 结尾
Linux 上读取时：json.loads 能处理（strip 掉 \r）
但 git diff 会显示整个文件被改动（因为行尾变了）
```

**显式 `newline="\n"` 保证跨平台一致**，git diff 干净。

> **这是跨平台项目里必须注意的细节。** 项目的 `.gitignore` 也遇到了
> 同样的问题（git 提示 `LF will be replaced by CRLF`）。

#### `ensure_ascii=False`

```python
json.dumps(row, ensure_ascii=False)
```

**默认 `ensure_ascii=True` 会把非 ASCII 字符转成 `\uXXXX` 转义。**

```python
# ensure_ascii=True（默认）
json.dumps({"prompt": "计算 12 + 30"})
# → '{"prompt": "\\u8ba1\\u7b97 12 + 30"}'

# ensure_ascii=False
json.dumps({"prompt": "计算 12 + 30"}, ensure_ascii=False)
# → '{"prompt": "计算 12 + 30"}'
```

**本项目的数学题是英文的**（`_add2` 等都返回英文），
所以这个参数影响不大。**但 `data/math_sample.jsonl` 里可能有中文，
且保持可读性更好。**

**`ensure_ascii=False` 让文件可读** —— 直接看 jsonl 就知道内容，
不用解码 `\uXXXX`。

### 3.7 命令行与输出

```powershell
python scripts/make_math_dataset.py --n 300 --eval-n 100 --seed 42 `
    --out data/math_train.jsonl --eval-out data/math_eval.jsonl
```

```text
[done] train: 200 rows -> data\math_train.jsonl
[done] eval : 100 rows -> data\math_eval.jsonl
  Compute 28 * 17. End your response with the number only.  => 476
  Compute 5 * 19. End your response with the number only.  => 95
  Compute 21 * 14. End your response with the number only.  => 294
```

**`--n 300 --eval-n 100`** 意味着：生成 300 条的池子，
前 100 条作为 eval，剩下 **200 条作为 train**。

**打印前 3 条做抽样检查** —— 一眼确认格式正确（不用打开文件）。

---

## 本篇小结：推理与评测的设计要点

1. **训练 loss 不是评测。** 必须独立评测才能知道"有没有变好"。
2. **定性和定量都要。** 定性发现"为什么错"，定量判断"是否显著"。
3. **评测 prompt 里要包含有确定答案的知识题。** 否则只能看到风格变化，
   看不到能力退化（本项目靠"太阳系"那题发现了退化）。
4. **`--compare` 要能自动找到产物**（`<output_dir>/final`），
   减少手动指定路径的摩擦。
5. **换模型前要 `del` + `gc.collect()` + `torch.cuda.empty_cache()`。**
   三者缺一不可（引用环 + 缓存池）。
6. **推理时 `use_cache=True`、训练时 `False`。** 弄错会让生成慢 O(n) 倍。
7. **`add_generation_prompt=True` 容易忘。** 没有它，模型不知道要回答。
8. **greedy 时要显式传中性值**（`temperature=1.0, top_p=1.0, top_k=0`），
   否则每次调用都打警告。
9. **报告数字用 greedy**（可复现），看多样性用采样。
10. **记录 `pred` 而不只是对错** —— 区分"reward 读错了"和"模型答错了"。
11. **比对失败集合比看总数更有信息量。** 准确率 +1% 可能是 churn。
12. **答案由程序计算**，不要手写。标准答案错了整个实验就废了。
13. **数据生成用依赖注入的随机源**（传 `rng` 而不是用全局 `random`），
    保证可复现。
14. **`while` 循环必须有上限**（`guard`），否则可能在不可能满足的条件下死循环。
15. **train/eval 必须从同一个池子切分。** 两个独立 seed 会重叠
    （实测 200/100 时有 16 条重叠）。
16. **写文件显式 `newline="\n"`**，避免 Windows 的 CRLF 转换污染 git diff。
