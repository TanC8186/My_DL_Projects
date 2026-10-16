# 03 · 工具层 `src/my_lab/`

这一篇讲三个"可复用逻辑"模块。它们是**全项目唯一被单元测试覆盖的部分**，
也是面试最容易被追问细节的地方。

| 文件 | 职责 | 是否依赖 torch |
|------|------|---------------|
| `rewards.py` | 可验证 reward：抽数字 + 比对 | ❌ 纯 Python |
| `memory.py` | 显存**粗估**（面试口述用） | ❌ 纯 Python |
| `vram.py` | 显存**实测**（CUDA 峰值） | ⚠️ 惰性导入 |

---

## 0. 一个贯穿全篇的设计：**为什么不在模块级 import torch**

先看 `vram.py` 的做法：

```python
"""CUDA peak-memory helpers.

torch is imported lazily inside the functions so this module stays importable
on CPU-only machines and inside the torch-free unit tests.
"""

from __future__ import annotations


def cuda_available() -> bool:
    try:
        import torch          # ← 在函数内部导入，不在文件顶部
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False
```

### 这样做换来什么

`tests/test_vram.py` 可以直接 `from my_lab.vram import peak_vram_gb` 而**不需要装 torch**，
也不需要 GPU。测试在有 GPU 和没有 GPU 的机器上都能跑：

```python
def test_peak_vram_reports_gb_keys_when_cuda_present():
    stats = peak_vram_gb()
    if not stats:
        # CPU-only machine: nothing to assert beyond "no crash".
        return
    ...
```

### 代价是什么

每次调用都执行一次 `import torch`。但 Python 的 `import` 有 `sys.modules` 缓存，
第二次起只是字典查找，**开销可以忽略**。

### 为什么 `try/except Exception` 而不是 `try/except ImportError`

因为 `import torch` 可能抛出**不止 ImportError**：

- `ImportError`：没装
- `OSError`：装了但 CUDA 库加载失败（`libcudart.so` 找不到）
- `RuntimeError`：DLL 初始化失败（Windows 上常见）

**捕获宽泛异常在这里是刻意的**：这个函数的语义是"能不能用 CUDA"，
任何导致不能用的原因都应该返回 `False`，而不是把异常抛给调用方。

### 对比：`memory.py` 干脆完全不碰 torch

```python
"""Rough VRAM heuristics for laptop planning (NOT exact)."""

from __future__ import annotations


def estimate_training_vram_gb(
    model_params_b: float,
    ...
) -> float:
```

**它是纯算术，连 import 都不需要。** 这就是"估算"和"实测"的本质区别：
估算只需要公式，实测需要真的碰到硬件。

---

## 1. `rewards.py` —— 可验证 reward

这是 GRPO 的核心。整份代码：

```python
"""Reward functions for verifiable tasks (math-style accuracy)."""

from __future__ import annotations

import re
from typing import Sequence


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_final_number(text: str) -> float | None:
    """Extract the last number in text; None if not found."""
    if not text:
        return None
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def accuracy_reward(
    completions: Sequence[str],
    answer: Sequence[str],
    **kwargs,
) -> list[float]:
    """
    TRL-style reward: 1.0 if final number matches, else 0.0.

    Args:
        completions: model outputs (strings or OpenAI-style message lists)
        answer: ground-truth answers (string or number-like)
    """
    rewards: list[float] = []
    for completion, gold in zip(completions, answer):
        if isinstance(completion, list):
            # OpenAI chat message list — take last assistant content
            text = ""
            for msg in completion:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    text = msg.get("content") or ""
                elif hasattr(msg, "get") and msg.get("role") == "assistant":
                    text = msg.get("content") or ""
            completion_text = text
        else:
            completion_text = str(completion)

        pred = extract_final_number(completion_text)
        gold_num = extract_final_number(str(gold))
        if pred is None or gold_num is None:
            rewards.append(0.0)
            continue
        rewards.append(1.0 if abs(pred - gold_num) < 1e-4 else 0.0)
    return rewards
```

### 1.1 正则 `r"-?\d+(?:\.\d+)?"` 逐字符拆解

```regex
-?              可选的负号（0 个或 1 个）
\d+             一个或多个数字（整数部分）
(?:\.\d+)?      非捕获组，可选的小数部分
```

逐个细节：

| 写法 | 为什么这么写 |
|------|-------------|
| `-?` | 匹配负数。`?` = 0 或 1 次 |
| `\d+` | `+` = 1 或多次。用 `+` 而非 `*`，保证至少有一位数字（否则 `-` 单独也能匹配） |
| `(?:...)` | **非捕获组**。用 `(...)` 也行，但我们不需要取出这个分组，非捕获组更省内存、语义更清晰 |
| `\.` | **转义的点**。不加反斜杠的 `.` 会匹配任意字符，`12x34` 也会被当成小数 |

**注意它匹配不到的情况**：

- `1,234`（千分位逗号）→ 会匹配到 `234`
- `1e5`（科学计数法）→ 会匹配到 `1` 和 `5`
- `½`（Unicode 分数）→ 匹配不到

**这是刻意的简化。** 数据集里的答案都是普通整数（见 `make_math_dataset.py`），
不需要处理这些边界。

### 1.2 为什么用 `re.compile` 而不是直接 `re.findall(pattern, text)`

```python
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
```

**模块级编译一次，复用。** Python 的 `re` 模块内部有缓存（默认 512 个模式），
所以 `re.findall(pattern, ...)` 也不是每次都重新编译。但：

- 显式编译让**意图更清楚**（这个正则是常量）
- 避免依赖内部缓存的行为
- 在热循环里调用时省一次缓存查找

这是"微小但正确的习惯"，不是性能瓶颈。

### 1.3 `extract_final_number` 逐行

```python
if not text:
    return None
```
**空字符串 / `None` 提前返回。** 注意 `if not text` 同时覆盖了 `""` 和 `None`
两种情况（Python 的假值判断）。如果只写 `if text == ""`，传 `None` 会漏掉。

```python
matches = _NUMBER_RE.findall(text)
if not matches:
    return None
```
`findall` 返回所有匹配的字符串列表。没有匹配 → 返回 `None`（而不是抛异常）。

```python
try:
    return float(matches[-1])
except ValueError:
    return None
```
**`matches[-1]` 取最后一个。** 这是整个 reward 设计的关键假设：
**模型会把最终答案放在最后。**

```python
except ValueError:
```
`float()` 理论上可能失败（虽然正则已经保证了格式）。这个 `except` 是防御性的。
在严格的正则下它是死代码，但**保留它比删掉更安全** —— 万一正则被改宽了。

### 1.4 `accuracy_reward` 逐段

#### 为什么签名是 `(completions, answer, **kwargs)`

这是 TRL 的调用约定。看 TRL 源码（`grpo_trainer.py:1299-1322`）：

```python
keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
reward_kwargs["trainer_state"] = self.state
...
output_reward_func = reward_func(
    prompts=prompts, completions=completions,
    completion_ids=completion_ids_list, **reward_kwargs
)
```

**TRL 把数据集里除 `prompt`/`completion`/`completion_ids` 外的所有列都作为 kwargs 传进来。**
我们的数据集列是 `prompt` 和 `answer`，所以 `answer` 会作为关键字参数传入。

`**kwargs` 用来**吞掉其他参数**（比如 `prompts`、`completion_ids`、`trainer_state`），
避免 `TypeError: unexpected keyword argument`。

#### `zip(completions, answer)`

```python
for completion, gold in zip(completions, answer):
```

**为什么用 `zip` 而不是索引？** 更 Pythonic，且不用管长度。但要注意
**`zip` 会静默截断到较短的那个** —— 如果两个列表长度不等，多出来的会被忽略，
不报错。这在 TRL 的调用场景里不是问题（两者必然等长），
但如果自己写调用要小心。

#### 两种 completion 格式的处理

```python
if isinstance(completion, list):
    # OpenAI chat message list — take last assistant content
    text = ""
    for msg in completion:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            text = msg.get("content") or ""
        elif hasattr(msg, "get") and msg.get("role") == "assistant":
            text = msg.get("content") or ""
    completion_text = text
else:
    completion_text = str(completion)
```

**为什么要处理两种格式？**

- **字符串**：纯文本 completion（本项目 GRPO 用的是这种）
- **消息列表**：对话式 completion，形如
  `[{"role": "user", ...}, {"role": "assistant", "content": "476"}]`

TRL 在**对话式数据集**上会返回后者。这个分支是为了**让 reward 函数两种场景都能用**。

**注意循环里的逻辑**：它**不 break**，而是遍历完取**最后一个** assistant 消息。
这是对的 —— 多轮对话里最后一个 assistant 回复才是最终答案。

```python
if isinstance(msg, dict) and msg.get("role") == "assistant":
    ...
elif hasattr(msg, "get") and msg.get("role") == "assistant":
    ...
```

**这两个分支看起来重复，其实覆盖两种对象：**

| 分支 | 匹配的对象 |
|------|-----------|
| `isinstance(msg, dict)` | 普通 dict |
| `hasattr(msg, "get")` | 任何有 `.get` 方法的对象（比如 `transformers` 的 `BatchEncoding`、自定义 message 类） |

**`elif` 而不是 `if`**：一个 dict 同时满足两个条件，用 `elif` 避免重复执行。

#### 比对逻辑

```python
pred = extract_final_number(completion_text)
gold_num = extract_final_number(str(gold))
if pred is None or gold_num is None:
    rewards.append(0.0)
    continue
rewards.append(1.0 if abs(pred - gold_num) < 1e-4 else 0.0)
```

```python
gold_num = extract_final_number(str(gold))
```

**为什么标准答案也要过一遍 `extract_final_number`？** 因为答案在 jsonl 里是字符串
（`"answer": "42"`），需要转成数字。用同一个函数保证**两边用同一套解析规则** ——
如果只对 pred 解析、对 gold 直接 `float()`，两边规则不一致会导致莫名其妙的 bug。

```python
if pred is None or gold_num is None:
    rewards.append(0.0)
```

**任何一边解析失败都给 0 分。** 这是保守策略：无法判定就算错。
（另一种选择是给 NaN 让 TRL 跳过，但那样样本会被静默丢弃，不利于调试。）

```python
abs(pred - gold_num) < 1e-4
```

**为什么用容差而不是 `==`？**

因为浮点数比较不能用 `==`。`0.1 + 0.2 != 0.3`。而且 `float("42")` 和
`float("42.0")` 虽然相等，但涉及除法运算时（比如 `144/12` 得到 `12.000000000000002`）
直接 `==` 会失败。`1e-4` 是足够宽松又不会误判的容差。

> **面试可讲的点**：这个 reward 的容差是 `1e-4`，对整数答案是安全的。
> 如果任务是"算到小数点后 3 位"，这个容差就太松了（会把 `1.0001` 判成等于 `1.0000`）。

### 1.5 ⚠️ 这个 reward 的真实缺陷：**reward misspecification**

**这是本项目最有价值的"负面知识"之一。**

`extract_final_number` 取的是"文本里最后一个数字"。这依赖一个强假设：
**模型遵守指令，把答案放在最后。**

实测反例（基座模型在 `data/math_sample.jsonl` 上）：

| 题目 | 标准答案 | 模型输出 | 抽取结果 | 判定 |
|------|---------|---------|---------|------|
| `Solve: 9 * 9.` | 81 | `To solve the expression \(9 \times 9\), you simply multiply the two nu...`（被截断） | **1.0** | ❌ 误判 |

模型在写 LaTeX 推理过程 `\(9 \times 9\)`，文本里的数字是 `9`、`9`，
但被截断后的最后一个数字变成了别的。**reward 函数读错了答案。**

**为什么这件事很重要？**

在 RL 里，**reward 就是目标函数**。如果 checker 本身是错的，那么：

```text
GRPO 会朝着"让 checker 给高分"的方向优化，而不是"让答案正确"
```

这叫 **reward hacking / reward misspecification**。这次任务简单、模型也基本听话，
所以问题没爆发；但在真实项目里，一个脆弱的 checker 会让整个 RL 训练跑偏。

**怎么修（如果要继续做）**：

1. **强约束输出格式**：prompt 改成要求 `\boxed{答案}` 或 `Answer: <num>`，
   reward 用正则精确匹配这个模式，而不是"最后一个数字"
2. **多次解析 + 投票**：同时用几种策略（最后一个数字、`answer is X`、最后一个独立行）
3. **先验证 checker 本身**：拿基座模型的输出跑一遍，人工核对解析结果是否合理
   —— 这正是 `scripts/eval_math.py` 里"打印失败案例"的用途

**在笔记里记录这个缺陷，比假装 reward 是完美的更有价值。**

---

## 2. `memory.py` —— 显存粗估

```python
"""Rough VRAM heuristics for laptop planning (NOT exact)."""

from __future__ import annotations


def estimate_training_vram_gb(
    model_params_b: float,
    dtype_bytes: int = 2,
    lora: bool = True,
    gradient_checkpointing: bool = True,
    batch_tokens: int = 512,
    overhead_factor: float = 1.6,
) -> float:
    """
    Ballpark forward+backward VRAM for full/LoRA fine-tuning.

    This is intentionally simple for interview notes and OOM triage.
    Real usage depends on attention, optimizer state, fragmentation, etc.
    """
    # Weights
    weight_gb = model_params_b * dtype_bytes
    if lora:
        # Base frozen; adapters + grads small vs base — keep base weights.
        optimizer_gb = 0.05 * model_params_b  # loose adapter/optimizer estimate
        activation_scale = 0.35 if gradient_checkpointing else 0.9
    else:
        # AdamW m,v ≈ 2x fp32 states often; keep a simplified 2x weights extra
        optimizer_gb = model_params_b * dtype_bytes * 2
        activation_scale = 0.5 if gradient_checkpointing else 1.2

    # Activations scale roughly with batch tokens and model size
    activation_gb = model_params_b * (batch_tokens / 512.0) * activation_scale * 0.15
    total = (weight_gb + optimizer_gb + activation_gb) * overhead_factor
    return round(total, 2)
```

### 2.1 这个函数在算什么

训练显存的三大块：

```text
总显存 ≈ (权重 + 优化器状态 + 激活值) × 开销系数
```

逐项拆解：

#### 权重

```python
weight_gb = model_params_b * dtype_bytes
```

`model_params_b` 单位是 **B（十亿参数）**，`dtype_bytes` 是每个参数的字节数。

- `bfloat16` → 2 字节
- `float32` → 4 字节

**注意单位陷阱**：`0.5B 参数 × 2 字节 = 1.0`，这个数字**恰好**约等于 1 GB
（因为 10⁹ ≈ 2³⁰ = 1.07×10⁹，误差 7%）。所以在面试口述时可以粗略说
"0.5B 的 bf16 权重约 1 GB"。

#### 优化器状态（LoRA vs 全参的天壤之别）

**LoRA 分支：**

```python
optimizer_gb = 0.05 * model_params_b  # loose adapter/optimizer estimate
```

`0.5B × 0.05 = 0.025 GB = 25 MB`。

这个系数是**拍脑袋的经验值**，注释里也承认了（`loose ... estimate`）。
真实值取决于 LoRA 的 `r` 和 `target_modules`。

**用实测数据校准**（这是这个函数存在的意义 —— 用实测反推系数是否合理）：

```text
实测：LoRA r=16 + q/k/v/o_proj 在 0.5B 上 = 2,162,688 参数

AdamW 每个参数需要：参数本身(bf16, 2B) + 梯度(bf16, 2B) + m(fp32, 4B) + v(fp32, 4B) = 12 字节
2,162,688 × 12 = 25,952,256 字节 ≈ 24.8 MB
```

**估算的 25 MB 和精算的 24.8 MB 几乎完全吻合。** 所以 `0.05` 这个系数
对 `r=16` 是准的。

> 想自己复现这个推导，见下面 2.3 节的 LoRA 参数量手算。

**全参分支：**

```python
optimizer_gb = model_params_b * dtype_bytes * 2
```

`0.5B × 2 × 2 = 2.0 GB`。

这个更实在：AdamW 为每个参数维护 `m` 和 `v` 两个状态，**通常用 fp32 存储**
（即使参数是 bf16），所以是 `参数量 × 4 字节 × 2`。而这里写的是
`参数量 × dtype_bytes × 2 = 参数量 × 2 × 2 = 参数量 × 4`，
**低估了一半**（应该是 `参数量 × 8`）。

不过它有 `overhead_factor = 1.6` 在后面兜底，所以最终数字还不算太离谱。

**这正说明了这个函数的定位：`粗估`，只够用来判断"这个配置大概能不能塞进去"，
不能用来做精确规划。** docstring 里写得很清楚：

> This is intentionally simple for interview notes and OOM triage.

#### 激活值

```python
activation_gb = model_params_b * (batch_tokens / 512.0) * activation_scale * 0.15
```

拆开看：

| 因子 | 含义 |
|------|------|
| `model_params_b` | 用参数量代理"模型多大" |
| `(batch_tokens / 512.0)` | 归一化到 512 token 基准。序列越长激活越多（近似线性） |
| `activation_scale` | 是否用 gradient checkpointing 的系数 |
| `0.15` | 经验缩放常数 |

`activation_scale` 的取值：

```python
activation_scale = 0.35 if gradient_checkpointing else 0.9    # LoRA
activation_scale = 0.5  if gradient_checkpointing else 1.2    # 全参
```

**0.35 vs 0.9 意味着 gradient checkpointing 把激活值降到约 39%**。
这个数量级是对的（真实值取决于模型深度和实现）。

**为什么 LoRA 的系数（0.35/0.9）比全参（0.5/1.2）小？**
因为 LoRA 时底座冻结，反向传播不需要穿过整个网络回传梯度到最底层，
中间激活的保留需求更低。

#### 开销系数

```python
total = (...) * overhead_factor      # 1.6
```

**为什么需要 1.6 倍？** 因为上面每一项都低估了：

- PyTorch 的 **caching allocator** 会预留比实际需要更多的显存（碎片）
- attention 的中间矩阵（`[B, heads, T, T]`）在长序列下可能非常大，公式里没单列
- 临时张量、kernel workspace
- 通信 buffer（分布式时）

**这个 1.6 是对"碎片和临时开销"的统一补偿。**

### 2.2 用它来验证实测数据

拿 SFT 实验的实测值反推：

```text
输入：0.5B, bf16, lora=True, gradient_checkpointing=True, batch_tokens=512

weight_gb       = 0.5 × 2 = 1.0
optimizer_gb    = 0.05 × 0.5 = 0.025
activation_gb   = 0.5 × (512/512) × 0.35 × 0.15 = 0.026
小计            = 1.051
total           = 1.051 × 1.6 = 1.68 GB
```

**实测 peak_allocated = 1.87 GB，估算 1.68 GB，误差约 11%。**

对"粗估"来说这个精度相当不错。但**注意它估的是 `allocated` 级别，
而实测的 `reserved` 是 4.89 GB**（差了 2.6 倍）——
**这个函数完全无法预测 `reserved`**，因为那取决于 allocator 的缓存行为，
不是模型的固有属性。

**这就是为什么必须有 `vram.py`（实测）而不能只靠 `memory.py`（估算）。**

### 2.3 附：LoRA 参数量手算（面试高频）

`memory.py` 里的 `0.05` 系数是从哪来的？下面是精确推导。

**Qwen2.5-0.5B 的结构**（来自 `config.json`）：

```text
hidden_size         = 896
num_hidden_layers   = 24
num_attention_heads = 14
num_key_value_heads = 2          ← GQA（分组查询注意力）
head_dim            = 896 / 14 = 64
```

**LoRA 每个矩阵的参数量**：LoRA 把权重更新分解为 `ΔW = B @ A`，其中
`A` 是 `[r, in_features]`、`B` 是 `[out_features, r]`，所以参数量 = `r × (in + out)`。

**逐模块计算**（`r = 16`）：

| 模块 | in_features | out_features | 参数量 |
|------|------------|--------------|--------|
| `q_proj` | 896 | 896 | `16 × (896+896) = 28672` |
| `k_proj` | 896 | **128** | `16 × (896+128) = 16384` |
| `v_proj` | 896 | **128** | `16 × (896+128) = 16384` |
| `o_proj` | 896 | 896 | `16 × (896+896) = 28672` |
| **每层合计** | | | **90112** |

```text
90112 × 24 层 = 2,162,688
```

**与实测完全一致**（`train_sft.py` 打印的 `trainable parameters: 2,162,688`）。

**全模型参数量**：496,195,456。所以：

```text
2,162,688 / 496,195,456 = 0.436%
```

#### ⚠️ 这里最容易算错的地方：GQA

**`k_proj` / `v_proj` 的 `out_features` 是 128，不是 896。**

为什么？因为 **GQA（Grouped Query Attention）**：

```text
传统 MHA：每个 attention head 都有自己的 K、V
          → K/V 的维度 = num_heads × head_dim = 14 × 64 = 896

GQA：    多个 query head 共享一组 K/V
          → K/V 的维度 = num_key_value_heads × head_dim = 2 × 64 = 128
```

所以 `k_proj` 的形状是 `[128, 896]` 而不是 `[896, 896]`。

**如果按 `hidden_size` 算（896），会得到：**

```text
错误：16 × (896+896) × 4 个模块 × 24 层 = 2,752,512   ← 多了 27%
正确：2,162,688
```

**记住这个：算 LoRA/模型参数量时，K 和 V 的维度看 `num_key_value_heads × head_dim`。**

**为什么 GQA 能省显存？** 推理时的 **KV cache** 大小正比于 `num_key_value_heads`。
MHA 是 14 组，GQA 是 2 组 → **KV cache 缩小 7 倍**。长上下文推理时这是决定性的。

---

## 3. `vram.py` —— 显存实测

```python
def cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def reset_peak_stats() -> bool:
    """Reset CUDA peak counters. Returns True when a GPU was actually reset."""
    if not cuda_available():
        return False
    import torch

    torch.cuda.reset_peak_memory_stats()
    return True


def peak_vram_gb() -> dict[str, float]:
    """Peak allocated / reserved VRAM in GB. Empty dict when CUDA is absent."""
    if not cuda_available():
        return {}
    import torch

    gib = 1024.0**3
    return {
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / gib, 2),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / gib, 2),
    }


def format_peak_vram() -> str:
    """One-line summary suitable for logs and experiment notes."""
    stats = peak_vram_gb()
    if not stats:
        return "n/a (no CUDA)"
    return (
        f"peak_allocated={stats['peak_allocated_gb']} GB, "
        f"peak_reserved={stats['peak_reserved_gb']} GB"
    )
```

### 3.1 三个函数的分工

```text
reset_peak_stats()    →  训练开始前清零计数器
   ...训练...              （否则峰值会包含模型加载阶段）
peak_vram_gb()        →  训练结束后读峰值
format_peak_vram()    →  格式化成一行，写进日志和 notes
```

**为什么要 `reset`？** 因为 `max_memory_allocated()` 记录的是**进程生命周期内**的峰值。
如果不重置，它会把"模型加载"那一刻的占用也算进去，而这个数字通常低于训练峰值
（所以不会出错，但不够精确）。重置后读到的是**纯训练阶段的峰值**。

本项目 SFT 脚本里的调用位置：

```python
model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
model.config.use_cache = False
if train_cfg.get("gradient_checkpointing", True):
    model.gradient_checkpointing_enable()

reset_peak_stats()          # ← 在模型加载之后
                            #   这样"模型加载"的占用会被计入当前值（不丢失）
                            #   但峰值计数器从这里重新开始
```

**这个位置是刻意的**：重置在模型加载**之后**，所以
`peak = max(模型加载占用, 训练峰值)`。两种占用都不会被漏掉。

### 3.2 ⚠️ `allocated` vs `reserved`：本项目最重要的显存知识点

```python
"peak_allocated_gb": round(torch.cuda.max_memory_allocated() / gib, 2),
"peak_reserved_gb":  round(torch.cuda.max_memory_reserved() / gib, 2),
```

**这两个数字的含义完全不同：**

| 指标 | 含义 |
|------|------|
| `allocated` | **真正被张量占用的字节**。有活着的 tensor 就有它 |
| `reserved` | **进程已向 CUDA 驱动申请的字节**，包含缓存和碎片 |

**PyTorch 的 caching allocator 的工作方式：**

```text
需要显存 → 先看 reserved 池里有没有合适的空闲块
         ├─ 有 → 直接复用（不向驱动申请，快）
         └─ 没有 → 向驱动申请一大块（cudaMalloc，慢），放进池子
释放张量 → 内存块**归还给池子**，不还给驱动
```

**所以 `reserved` 只增不减**（除非显式 `torch.cuda.empty_cache()`）。
这带来两个后果：

#### 后果 1：判断"会不会 OOM"必须看 `reserved`

```text
SFT 实测：allocated 1.87 GB，reserved 4.89 GB
         ↑ 只看这个会觉得"才用了 1.87/8 = 23%，很安全"
                              ↑ 实际占用 4.89/8 = 61%
```

**`nvidia-smi` 显示的是 `reserved` 级别。** 如果只看 `allocated` 会严重低估风险。

#### 后果 2：`reserved` 会随运行时长漂移，**不能跨实验直接比较**

这是本项目一个真实的分析陷阱：

| 实验 | allocated | reserved | 步数 |
|------|-----------|----------|------|
| SFT | 1.87 GB | **4.89 GB** | 250 |
| DPO | 2.76 GB | **3.56 GB** | 125 |

**DPO 的 `allocated` 比 SFT 高 48%（符合预期：要 forward chosen+rejected 两条序列
外加一次参考前向），但它的 `reserved` 反而更低。**

原因：**`reserved` 随运行步数增长**（缓存块越攒越多），而 SFT 跑了 250 步、
DPO 只有 125 步。所以这个对比是**被步数差异污染的**。

**结论：跨实验比显存必须固定步数，否则结论是假的。**

这条已写入 `docs/8gb-memory-guide.md` 和 DPO 的 `notes.md`。

### 3.3 `gib = 1024.0**3` 而不是 `1e9`

```python
gib = 1024.0**3
```

**这是 GiB 而不是 GB**：

```text
1 GB  = 1000³ = 1,000,000,000 字节
1 GiB = 1024³ = 1,073,741,824 字节
```

**为什么用 1024³？** 因为 `nvidia-smi` 和 PyTorch 内部都按 1024 进制报告，
用同一个基准才能对得上。变量名 `gib` 也很诚实（Gibibyte）。

**注意 `1024.0**3` 里的 `.0`** —— 强制浮点运算。写 `1024**3` 在 Python 里也是 int，
除法结果在 Python 3 里是 float，所以这里不影响正确性，但显式浮点更清楚。

### 3.4 `round(..., 2)` 和返回类型

```python
return {
    "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / gib, 2),
    ...
}
```

**为什么保留 2 位小数？** 显存这个量级上，0.01 GB = 10 MB 已经是分辨率极限，
再多位数是虚假精度。

**为什么返回 dict 而不是 tuple？** 因为：

1. **自解释**：`stats["peak_allocated_gb"]` 比 `stats[0]` 清楚
2. **可扩展**：以后加 `peak_fragmentation_gb` 不会破坏调用方
3. **`{}` 是天然的"无 CUDA"信号**：`if not stats:` 就够判断了，不需要额外的哨兵值

**`format_peak_vram()` 里的 `if not stats: return "n/a (no CUDA)"`** ——
这就是为什么返回空 dict 是个好设计：调用方不用区分"失败"和"没有数据"。

---

## 本篇小结：工具层的设计原则

1. **库不依赖硬件。** `my_lab/` 里 torch 是惰性导入的，所以 19 个单测能在纯 CPU 上跑。
2. **估算和实测分开，且都保留。** `memory.py`（公式，面试口述）和
   `vram.py`（实测，写进 notes）。**两者都不能替代对方** ——
   估算给不出 `reserved`，实测需要 GPU。
3. **用实测校准估算。** `memory.py` 的 `0.05` 系数与精算的 24.8 MB 吻合，
   这种"交叉验证"让粗估有了可信度。
4. **reward 的缺陷要记录，不要隐藏。** `extract_final_number` 的
   "取最后一个数字"假设在模型不守规矩时会读错，这是 reward misspecification
   的实例，比假装完美更有价值。
5. **`allocated` 和 `reserved` 是两个不同的东西。** 前者是"在用"，后者是"已申请"。
   判断 OOM 看后者，比较实验要固定步数。
6. **算参数量时 GQA 是关键。** `k_proj`/`v_proj` 的维度是
   `num_key_value_heads × head_dim`，不是 `hidden_size`。算错会多算 27%。
