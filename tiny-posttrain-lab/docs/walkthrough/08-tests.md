# 08 · 测试

这一篇讲 `tests/` 下四个文件，共 **18 个测试**。它们的共同特点是
**全部不需要 GPU，全部在 2~10 秒内跑完**。

---

## 0. 为什么一个"跑模型"的项目需要测试

### 常见的误解

```text
"训练是 GPU 上跑几个小时的，测不了"
"等跑完看 loss 曲线不就知道对不对了"
```

**问题在于**：

| 场景 | 没有测试的后果 |
|------|--------------|
| `extract_final_number` 解析错了 | GRPO 跑 40 分钟，奖励全 0，**你以为模型不行** |
| 数据生成器算错答案 | 训练集标准答案是错的 → 整个实验无意义 |
| train/eval 数据重叠 | 评测结果被高估 → 得出错误结论 |
| `vram.py` 在无 GPU 环境导入崩溃 | CI 上跑不了，或者换台机器就挂 |

**本项目踩过的坑正好印证**：`extract_final_number` 把 `9*9` 的输出读成 `1.0`
（见 [03 篇](./03-helpers.md)）—— **如果当时有测试覆盖这个 case，能提前发现。**

### 测试策略：**只测能纯 CPU 测的部分**

```text
可测（本项目的选择）：
  ✅ 数据生成逻辑（答案对不对、切分重不重叠）
  ✅ reward 函数（解析、比对）
  ✅ 文件结构契约（配置/文档是否存在）
  ✅ 显存模块的"无 GPU 安全性"

不测：
  ❌ 训练能不能收敛（需要 GPU + 几十分钟）
  ❌ 生成的文本质量（需要人判断）
  ❌ TRL 的行为（那是上游的职责）
```

**这是一个刻意的取舍**：不做端到端测试（太慢），
而是把**容易被搞错的纯逻辑**抽到 `my_lab/` 里单独测。

**这个取舍直接影响了架构**：`my_lab/` 里的模块**不在模块级 import torch**
（见 [03 篇](./03-helpers.md)），就是为了让它们能被纯 CPU 测试。

> **"可测试性"是一个设计约束，不是一个事后补充。**
> 如果 `vram.py` 在文件顶部写了 `import torch`，
> 这个项目就永远不会有 `test_vram.py`。

---

## 1. `tests/test_project_files.py` —— 结构契约

```python
"""Tiny offline checks for package import paths (no GPU)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
```

### 引导段

**和 `scripts/*.py` 里的一样**（见 [04 篇](./04-train-sft.md)）——
把 `src/` 加入 `sys.path`，让 `my_lab` 可导入。

**`ROOT = ...parents[1]`**：`tests/test_project_files.py` → `parents[0]` 是 `tests/`，
`parents[1]` 是项目根。

**为什么每个测试文件都重复这段？** 因为没有 `conftest.py`。
pytest 支持用 `conftest.py` 统一做路径设置，那样更 DRY。
**当前的写法更直白**（每个文件自包含，不依赖 pytest 的特定机制），
代价是重复。

### `test_import_my_lab`

```python
def test_import_my_lab():
    import my_lab
    from my_lab import accuracy_reward, estimate_training_vram_gb

    assert callable(accuracy_reward)
    assert callable(estimate_training_vram_gb)
```

**测什么？** 包的 `__init__.py` 的导出是否正常工作。

**为什么 import 写在函数内部？** 因为如果写在文件顶部，
**导入失败会导致整个文件无法收集**（pytest 报 collection error），
你看到的是"文件加载失败"而不是"这个测试失败"。

**放在函数里，导入失败就变成这个测试的失败**，定位更清楚。

**`assert callable(...)`** —— 验证导出的是函数而不是 `None`
（比如 `__init__.py` 写错了名字，`from my_lab import xxx` 会 `ImportError`，
但如果是 `xxx = None` 就不会）。**这个断言让"导出存在但为空"也能被抓住。**

### `test_configs_exist`

```python
def test_configs_exist():
    cfg_dir = ROOT / "configs"
    assert (cfg_dir / "sft_qwen05b_8g.yaml").exists()
    assert (cfg_dir / "dpo_qwen05b_8g.yaml").exists()
    assert (cfg_dir / "grpo_qwen05b_8g.yaml").exists()
```

**测什么？** 三个配置文件都存在。

**价值**：如果有人重命名或移动了配置，测试会立刻失败，
而不是等到跑训练时报"配置不存在"。

**注意这里没有检查文件内容** —— 只检查存在性。
因为配置的正确性由"能不能跑通训练"来验证，不适合单测。

### `test_math_sample_jsonl`

```python
def test_math_sample_jsonl():
    path = ROOT / "data" / "math_sample.jsonl"
    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 5
```

**测什么？** 最小样例数据集存在且非空（至少 5 行）。

**为什么是 `>= 5` 而不是精确数字？** 因为这是"最小可读示例"，
内容可以增删。**只保证"够用来演示"这个下界。**

**`.strip()` 后再 `splitlines()`** —— 去掉文件末尾的换行，
避免多算一个空行。

### `test_docs_referenced_by_readme_exist`

```python
def test_docs_referenced_by_readme_exist():
    """The READMEs link these; a rename would silently produce dead links."""
    docs = ROOT / "docs"
    for name in ("8gb-memory-guide.md", "project-layout.md", "trl-reading-notes.md"):
        assert (docs / name).exists(), f"missing doc referenced from the READMEs: {name}"
```

**测什么？** README 里链接的文档都存在。

**为什么需要？** 因为 **markdown 的链接不会被任何工具检查**。
重命名一个文档，README 里的链接就变成 404，但**没有任何报错** ——
只有读者点进去才发现。

**这个测试把"文档链接"变成了"可验证的契约"。**

**`assert ..., f"..."` 的用法**：`assert` 的第二个参数是**失败时显示的消息**。
比默认的 `AssertionError` 有用得多：

```text
# 没有消息：
AssertionError

# 有消息：
AssertionError: missing doc referenced from the READMEs: trl-reading-notes.md
```

### `test_every_experiment_has_notes`

```python
def test_every_experiment_has_notes():
    """An experiment folder without notes.md is unreviewable, so treat it as a bug."""
    experiments = ROOT / "experiments"
    folders = [p for p in experiments.iterdir() if p.is_dir()]
    assert folders, "no experiments found"
    for folder in folders:
        assert (folder / "notes.md").exists(), f"{folder.name} has no notes.md"
```

**测什么？** 每个实验文件夹都有 `notes.md`。

**这条测试编码了一个"项目规范"**：

```text
一次实验 = 一个文件夹
文件夹里必须有 notes.md
```

**为什么把它变成测试而不是写在文档里？**

因为**文档里的规范会被忘记，测试里的规范会被强制执行**。

**实测效果**：写完这条测试后，我新建 `2026-09-grpo-math` 文件夹时，
第一件事就是建 `notes.md`（否则测试失败）。

**`assert folders, "no experiments found"`** —— 额外的健全性检查：
如果 `experiments/` 是空的，后面的循环不会执行，
测试会**静默通过**。这个断言防止"空目录假装通过"。

> **这是一个容易被忽略的测试陷阱**：遍历式的断言在集合为空时
> 会无条件通过。**需要显式检查集合非空。**

---

## 2. `tests/test_rewards_and_memory.py` —— 核心逻辑

```python
from my_lab.memory import estimate_training_vram_gb
from my_lab.rewards import accuracy_reward, extract_final_number
```

**注意：这两个模块都是纯 Python**（不在模块级 import torch），
所以这个测试**不需要 torch 就能跑**。

### `test_extract_final_number`

```python
def test_extract_final_number():
    assert extract_final_number("answer is 42") == 42.0
    assert extract_final_number("step1=3 step2=7.5") == 7.5
    assert extract_final_number("no numbers here") is None
```

**三个 case 分别测：**

| 输入 | 期望 | 测什么 |
|------|------|--------|
| `"answer is 42"` | `42.0` | **基本功能**：能从文本里抽出数字 |
| `"step1=3 step2=7.5"` | `7.5` | **"取最后一个"语义**：`3` 和 `7.5` 都在，返回后面的 |
| `"no numbers here"` | `None` | **无数字时返回 `None`**，不是抛异常 |

**第 2 个 case 是核心**：它验证了"取最后一个数字"这个关键假设。
如果实现改成取第一个，这个测试会失败。

**`== 42.0` 而不是 `== 42`** —— 虽然 Python 里 `42.0 == 42` 为 `True`，
但写 `42.0` 表达了"返回的是 float"这个意图。

**`is None` 而不是 `== None`** —— PEP 8 明确要求用 `is` 比较 `None`。

### ⚠️ 这个测试的覆盖缺口

**它没有覆盖 reward 读错的情况**：

```python
# 缺失的 case：模型输出里有推理过程，最后一个数字不是答案
extract_final_number("To solve \\(9 \\times 9\\), multiply the two nu")
# → 实际返回 1.0（从被截断的文本里读到的）
```

**这正是 [03 篇](./03-helpers.md) 里讨论的 reward misspecification。**
**测试覆盖了"功能正常"的路径，没覆盖"输入不符合假设"的路径。**

**如果要补**：

```python
def test_extract_final_number_picks_the_last_number_not_the_answer():
    """Documented weakness: the function has no idea which number is the answer."""
    text = "Compute 9 * 9. We multiply 9 by 9 and the answer is 81."
    assert extract_final_number(text) == 81.0     # 这个 case 能过

    truncated = "Compute 9 * 9. We multiply the two nu"   # 被截断，没写完
    # 这里没有正确答案可抽，函数会返回 9（文本里的最后一个数字）
    assert extract_final_number(truncated) == 9.0
```

**这种测试叫"特征化测试"（characterization test）** ——
它不测"行为是否正确"，而是**把当前的（可能不理想的）行为固定下来**，
这样将来有人改动时能立刻知道。

> **这是一个可以改进的点，也是一个好的面试话题**：
> "我的 reward 函数有一个已知缺陷，我写了一个特征化测试把它记录下来，
> 而不是假装它是对的。"

### `test_accuracy_reward`

```python
def test_accuracy_reward():
    r = accuracy_reward(["The answer is 12"], ["12"])
    assert r == [1.0]
    r = accuracy_reward(["The answer is 12"], ["13"])
    assert r == [0.0]
    r = accuracy_reward(["no number"], ["12"])
    assert r == [0.0]
```

**三个 case：**

| completions | answer | 期望 | 测什么 |
|-------------|--------|------|--------|
| `["The answer is 12"]` | `["12"]` | `[1.0]` | **答对** |
| `["The answer is 12"]` | `["13"]` | `[0.0]` | **答错** |
| `["no number"]` | `["12"]` | `[0.0]` | **无法解析 → 0 分** |

**注意 `assert r == [1.0]` 而不是 `assert r == 1.0`** ——
**函数返回的是列表**（批量接口），不是单个值。

**这个"返回列表"的设计是 TRL 要求的**（见 [03 篇](./03-helpers.md)）：
reward 函数一次处理一个 batch，返回等长的奖励列表。

**第 3 个 case 很重要**：它固定了"解析失败给 0 分"的决策。
另一种合理设计是给 `None`（让 TRL 跳过），**测试把这个选择记录下来了。**

### `test_memory_estimate_increases_with_size`

```python
def test_memory_estimate_increases_with_size():
    small = estimate_training_vram_gb(0.5, lora=True, gradient_checkpointing=True)
    large = estimate_training_vram_gb(7.0, lora=True, gradient_checkpointing=True)
    assert large > small
    full = estimate_training_vram_gb(0.5, lora=False, gradient_checkpointing=False)
    assert full > small
```

**注意这个测试的断言风格：测的是"单调性"而不是"具体数值"。**

```python
assert large > small          # 7B 应该比 0.5B 用更多显存
assert full > small           # 全参应该比 LoRA 用更多显存
```

**为什么不断言具体数值（比如 `assert small == 1.68`）？**

因为 `memory.py` 的 docstring 明确说了它是**粗估**：

```python
"""Rough VRAM heuristics for laptop planning (NOT exact)."""
```

**如果断言精确值，任何系数微调都会让测试失败** ——
而那些微调是**预期内的**（估算本来就会随经验修正）。

**测"关系"而不是"数值"，是给启发式函数写测试的正确方式。**

**三个调用覆盖的组合：**

| 调用 | 参数 | 验证 |
|------|------|------|
| `small` | 0.5B, LoRA, ckpt | 基准 |
| `large` | 7B, LoRA, ckpt | **模型变大 → 显存变大** |
| `full` | 0.5B, 全参, 无 ckpt | **全参 + 无 ckpt → 显存变大** |

**第 3 个同时改了两个变量**（LoRA→全参、ckpt→无 ckpt），
两个都指向"更大"，所以断言成立。**如果只改一个也能过，但改了俩更保险。**

---

## 3. `tests/test_vram.py` —— CPU 安全性

```python
"""VRAM helpers must stay importable and safe without CUDA."""

from my_lab.vram import (
    cuda_available,
    format_peak_vram,
    peak_vram_gb,
    reset_peak_stats,
)
```

**注意：这个测试直接 import `my_lab.vram`，而 `vram.py` 内部会 import torch。**
能成功导入本身就验证了"惰性导入"的设计有效。

### `test_vram_helpers_are_cpu_safe`

```python
def test_vram_helpers_are_cpu_safe():
    assert isinstance(cuda_available(), bool)
    assert isinstance(reset_peak_stats(), bool)
    assert isinstance(peak_vram_gb(), dict)
    assert isinstance(format_peak_vram(), str)
```

**测什么？** 四个函数在**任何环境**下都能调用且返回正确类型。

**注意断言的是"类型"而不是"值"** —— 因为值取决于有没有 GPU：

```text
有 GPU：peak_vram_gb() → {"peak_allocated_gb": 1.87, "peak_reserved_gb": 4.89}
无 GPU：peak_vram_gb() → {}
```

**两者都是"正确的"**，所以只能断言"返回的是 dict"。

**这个测试的核心价值是"不崩"** —— 在无 CUDA 的机器上（CI、同事的 Mac）
调用这些函数不应该抛异常。

### `test_peak_vram_reports_gb_keys_when_cuda_present`

```python
def test_peak_vram_reports_gb_keys_when_cuda_present():
    stats = peak_vram_gb()
    if not stats:
        # CPU-only machine: nothing to assert beyond "no crash".
        return
    assert set(stats) == {"peak_allocated_gb", "peak_reserved_gb"}
    assert all(isinstance(v, float) and v >= 0 for v in stats.values())
    assert stats["peak_reserved_gb"] >= stats["peak_allocated_gb"]
```

#### `if not stats: return` 的模式

**这是"条件测试"**：有 GPU 时深入验证，没有时跳过。

**为什么不用 `pytest.mark.skipif`？**

```python
# 另一种写法
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_xxx():
    ...
```

**区别**：

| | `if not stats: return` | `skipif` |
|---|---|---|
| 需要 import torch | ❌ 不需要 | ✅ 需要（`torch.cuda.is_available()`）|
| 报告显示 | passed | skipped |
| 依赖 | 只用被测模块 | 引入额外依赖 |

**用 `return` 更好**，因为它不引入 torch 依赖 —— 这个测试文件本身
就能在无 torch 环境跑（虽然 `vram.py` 会尝试 import torch，但失败被捕获）。

#### 三条断言

**1. `set(stats) == {"peak_allocated_gb", "peak_reserved_gb"}`**

**用集合相等而不是逐 key 检查** —— 这样既验证"key 都存在"，
也验证"**没有多余的 key**"。

如果实现里多加了一个 `peak_fragmentation_gb`，这个测试会失败 ——
**这会强制你意识到"改动影响了对外契约"。**

**2. `all(isinstance(v, float) and v >= 0 for v in stats.values())`**

**两个检查合一**：

- `isinstance(v, float)` → 类型正确（不是 `int` 或 `str`）
- `v >= 0` → 显存不可能是负数

**`all(...)` 遍历所有值** —— 比逐个 `assert stats["a"] >= 0` 更简洁。

**3. `stats["peak_reserved_gb"] >= stats["peak_allocated_gb"]`**

**这是一条"物理约束"的断言。**

```text
reserved = 向驱动申请的显存（含缓存）
allocated = 真正被张量占用的显存
        ↓
reserved >= allocated 恒成立
```

**为什么这个断言有价值？** 因为如果实现里把两个函数写反了
（`max_memory_allocated` 和 `max_memory_reserved` 调换），
**这个断言会失败**。

**而"写反了"是一个很容易犯、又很难发现的错误** ——
两个函数名很像，返回类型也一样，在正常运行时不会报错，
只是数字含义错了。

> **这是"用不变量测代码"的好例子**：不断言具体数值，
> 而是断言两个量之间的**必然关系**。

### `test_format_peak_vram_mentions_allocated_when_cuda_present`

```python
def test_format_peak_vram_mentions_allocated_when_cuda_present():
    text = format_peak_vram()
    if not cuda_available():
        assert "n/a" in text
        return
    assert "peak_allocated" in text and "peak_reserved" in text
```

**两个分支都测：**

| 环境 | 期望 |
|------|------|
| 无 CUDA | 返回的字符串包含 `"n/a"` |
| 有 CUDA | 包含 `"peak_allocated"` 和 `"peak_reserved"` |

**为什么测字符串内容？** 因为这个函数的输出会**直接写进日志和 notes**：

```text
[info] peak VRAM: peak_allocated=1.87 GB, peak_reserved=4.89 GB
```

**如果 key 名写错**（比如 `allocated` 而不是 `peak_allocated`），
下游解析日志的脚本（比如我用 `Select-String` 提取数据的那些命令）会失效。

**这个测试固定了"输出格式契约"。**

**注意 `cuda_available()` 和 `peak_vram_gb()` 可能不一致** ——
理论上第一个返回 `True` 而第二个返回 `{}`（如果 CUDA 在这两次调用之间挂了）。
**实际上不会发生**，但这个测试用的是 `cuda_available()` 判断，
而上面那个用的是 `peak_vram_gb()` 判断。**轻微的不一致，无害。**

---

## 4. `tests/test_math_dataset.py` —— 数据正确性

**这是最有价值的一个测试文件。** 因为数据集的正确性
**直接决定了整个 GRPO 实验的有效性**。

```python
"""CPU-only checks for the math dataset generator.

The generator is the ground truth for the verifiable reward, so its two promises
have to hold: answers are arithmetically right, and the train/eval splits never
overlap (an earlier version used two independent seeds and leaked 16 prompts).
"""
```

**docstring 里直接记录了历史 bug**（"an earlier version ... leaked 16 prompts"）
—— **测试文件的 docstring 是解释"为什么需要这个测试"的好地方。**

### 引导段：加入 `scripts/`

```python
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_math_dataset import GENERATORS, build, split  # noqa: E402
```

**注意这里加的是 `scripts/` 而不是 `src/`** —— 因为要测的是脚本里的函数。

**这带来一个设计约束**：`make_math_dataset.py` 必须是**可导入的**
（不能在导入时执行 `main()`）。它用了 `if __name__ == "__main__": main()`，
所以导入安全。

> **这是"脚本可测试性"的一个要求**：把逻辑放在函数里，
> `main()` 只做参数解析和调用。**如果所有逻辑都写在 `main()` 里，就没法单测。**

### `solve()` —— 独立重新推导答案

```python
NUM = r"(\d+)"


def solve(prompt: str) -> int | None:
    """Re-derive the answer from the question text, independently of the generator."""
    patterns = [
        (rf"What is {NUM} \+ {NUM}\?", lambda m: int(m[0]) + int(m[1])),
        (rf"What is {NUM} - {NUM}\?", lambda m: int(m[0]) - int(m[1])),
        (rf"Compute {NUM} \* {NUM}\.", lambda m: int(m[0]) * int(m[1])),
        (rf"Compute {NUM} / {NUM}\.", lambda m: int(m[0]) // int(m[1])),
        (
            rf"A book has {NUM} pages\. You read {NUM}\. How many pages are left\?",
            lambda m: int(m[0]) - int(m[1]),
        ),
        # ... 其余四个
    ]
    for pattern, fn in patterns:
        match = re.search(pattern, prompt)
        if match:
            return fn([int(g) for g in match.groups()])
    return None
```

#### 这是"独立实现"而不是"复用生成器的答案"

**关键：`solve()` 从 prompt 文本反推答案，完全不看生成器算出的 `answer` 字段。**

```text
生成器： question, answer = _mul2x2(rng)      # 直接算出 a*b
测试：   solve(prompt) → 用正则从 prompt 里抠出 a 和 b，重新算 a*b
        ↓
比较两者是否一致
```

**如果测试直接复用生成器的 `answer`，那就什么也没测**（自己和自己比）。

**这是"独立验证"的核心**：用**不同的方法**得出同一个结论。

#### 正则里的转义

```python
rf"What is {NUM} \+ {NUM}\?"
```

**`\+` 和 `\?` 必须转义**：

- `+` 在正则里是"前一个字符重复 1 次以上"的量词
- `?` 是"前一个字符出现 0 或 1 次"的量词

**不转义的后果**：

```python
re.search(r"What is (\d+) + (\d+)?", "What is 12 + 30?")
#                          ↑ 匹配"一个或多个空格"
#                                     ↑ 把 "30" 变成可选
# → 可能匹配出错误的分组
```

**注意 `\*` 和 `\.` 也转义了**（乘号和点号在正则里都是元字符）。

#### 函数列表 + 循环匹配的设计

```python
    for pattern, fn in patterns:
        match = re.search(pattern, prompt)
        if match:
            return fn([int(g) for g in match.groups()])
    return None
```

**用"模式 + 处理函数"的列表，而不是 `if/elif` 链。**

**好处**：

- 加一个新题型只加一行元组
- 顺序清晰（从上到下尝试）
- `return None` 明确表示"不认识这个格式"

**`fn([int(g) for g in match.groups()])`** —— 把捕获组转成 int 列表传给函数。
`m[0]`、`m[1]` 对应第一个、第二个捕获组。

### `test_generator_is_deterministic`

```python
def test_generator_is_deterministic():
    assert build(20, seed=7) == build(20, seed=7)
    assert build(20, seed=7) != build(20, seed=8)
```

**两个断言测两个方向：**

1. **同 seed → 同结果**（可复现性）
2. **不同 seed → 不同结果**（随机性确实起作用）

**第 2 个断言为什么重要？**

如果实现里忘了传 `rng`（比如生成器内部用了全局 `random`），
那么**所有 seed 都会产生相同结果** —— 第 2 个断言会失败。

**这是"测两个方向"的典型例子**：只测第 1 个会漏掉"seed 无效"的 bug；
只测第 2 个会漏掉"不确定"的 bug。

**`== ` 直接比较列表** —— 因为 `build` 返回 `list[dict[str, str]]`，
dict 的相等是**逐键值比较**，列表的相等是**逐元素比较**。**可以直接用 `==`。**

### `test_prompts_are_unique`

```python
def test_prompts_are_unique():
    rows = build(150, seed=42)
    prompts = [r["prompt"] for r in rows]
    assert len(set(prompts)) == len(prompts)
```

**`len(set(x)) == len(x)`** 是"无重复"的经典写法。

**为什么需要这个测试？** 因为 `build()` 里有去重逻辑
（`seen` 集合 + `continue`）。**如果那段逻辑被改坏，这个测试会抓住。**

**训练集有重复题会怎样？**

```text
同一个 prompt 出现 3 次
→ 它的梯度被放大 3 倍
→ 模型偏向这个题
→ 评测时如果考到它，可能"看起来学会了"
```

### `test_every_answer_is_arithmetically_correct` —— 最重要的一个

```python
def test_every_answer_is_arithmetically_correct():
    rows = build(200, seed=42)
    unchecked = 0
    for row in rows:
        expected = solve(row["prompt"])
        if expected is None:
            unchecked += 1
            continue
        assert row["answer"] == str(expected), f"{row['prompt']} -> {row['answer']} != {expected}"
    # Every prompt shape should be covered by `solve`; otherwise this test is vacuous.
    assert unchecked == 0, f"{unchecked} prompts were not re-derivable by the test"
```

#### 逐行

```python
        assert row["answer"] == str(expected), f"{row['prompt']} -> {row['answer']} != {expected}"
```

**`str(expected)`** —— 因为生成器存的 `answer` 是字符串（见 [07 篇](./07-infer-eval.md)）。
**比较时两边类型要一致。**

**报错信息包含 prompt、实际值、期望值** —— 一眼能看出哪道题错了：

```text
AssertionError: Compute 28 * 17. End your response with the number only. -> 476 != 486
```

#### ⚠️ `assert unchecked == 0` 是防止"空测试"

```python
    # Every prompt shape should be covered by `solve`; otherwise this test is vacuous.
    assert unchecked == 0, f"{unchecked} prompts were not re-derivable by the test"
```

**如果没有这一行会发生什么？**

```text
假设有人改了生成器的措辞（比如 "What is 12 + 30?" → "Calculate 12 + 30."）
        ↓
solve() 的正则不再匹配新格式 → 返回 None
        ↓
循环里所有样本都走 continue 分支
        ↓
【没有任何断言被执行】 → 测试"通过"
        ↓
实际上答案对不对完全没被验证
```

**这叫"空测试"（vacuous test）** —— 测试通过但什么也没测。

**`assert unchecked == 0` 把这个漏洞堵上了**：只要有一道题不被 `solve` 认识，
测试就失败。

> **这是一条通用的测试设计原则**：
> **任何"遍历 + 条件跳过"的测试，都必须断言"跳过的数量为 0"**
> （或者至少断言"实际检查的数量达到预期"）。
> **否则测试可能在你以为它在工作时悄悄失效。**

### `test_splits_do_not_overlap` —— 修复的回归测试

```python
def test_splits_do_not_overlap():
    pool = build(300, seed=42)
    eval_rows, train_rows = split(pool, 100, seed=42)

    assert len(eval_rows) == 100
    assert len(train_rows) == 200
    eval_prompts = {r["prompt"] for r in eval_rows}
    train_prompts = {r["prompt"] for r in train_rows}
    assert not (eval_prompts & train_prompts), "train/eval overlap leaks the eval set"
```

#### 断言设计

```python
    assert not (eval_prompts & train_prompts), "train/eval overlap leaks the eval set"
```

**`set & set`** 是集合交集。`not` 表示"交集为空"。

**报错信息用 "leaks the eval set"** —— 点出了问题的严重性：
**这是数据泄漏，会让评测结果虚高。**

#### 同时验证长度

```python
    assert len(eval_rows) == 100
    assert len(train_rows) == 200
```

**为什么也要检查长度？**

因为**"交集为空"在极端情况下会假通过**：

```text
如果 split() 有 bug，返回 ([], pool)：
  eval_rows = []           → len 0
  train_rows = pool        → len 300
  交集 = 空                 → 不重叠断言通过 ✅
  但这不是我们想要的！
```

**加上长度断言就能抓住这种情况。**

> **这是"断言要能区分正确和错误实现"的体现**：
> 只测"不重叠"不够，还要测"切分比例正确"。

### `test_split_is_deterministic_and_covers_the_pool`

```python
def test_split_is_deterministic_and_covers_the_pool():
    pool = build(60, seed=1)
    a_eval, a_train = split(pool, 20, seed=1)
    b_eval, b_train = split(pool, 20, seed=1)
    assert a_eval == b_eval and a_train == b_train
    assert {r["prompt"] for r in a_eval + a_train} == {r["prompt"] for r in pool}
```

**两条断言：**

1. **确定性**：同参数两次调用结果相同
2. **完备性**：两个 split 合起来等于原始池子（没丢数据）

**第 2 条是关键** —— 它验证"切分"不是"丢弃"：

```text
如果 split() 有 bug 只返回了前 60 条中的 50 条：
  交集为空 ✅（仍然通过）
  长度对 ✅（20 + 40 = 60 ... 等等，这也能对上）
  但内容不全 → 第 2 条断言会失败
```

**用集合比较而不是列表比较**（`{...} == {...}`）——
因为**顺序可能因为洗牌而不同**，但内容必须一致。

### `test_no_eval_split_returns_the_whole_pool`

```python
def test_no_eval_split_returns_the_whole_pool():
    pool = build(30, seed=3)
    eval_rows, train_rows = split(pool, 0, seed=3)
    assert eval_rows == []
    assert len(train_rows) == 30
```

**测边界情况：`eval_n=0`。**

**这是"边界条件测试"** —— 对应 `split()` 里的：

```python
    if not eval_n:
        return [], shuffled
```

**如果不写这个分支会怎样？**

```python
return shuffled[:0], shuffled[0:]     # → ([], 全部)
```

**其实也对。** 所以这个分支是**冗余的**（`shuffled[:0]` 就是 `[]`）。

**但显式写出来更清楚**（表达意图："不要 eval 集就全给 train"），
而且这个测试固定了这个行为。

### `test_solve_covers_every_generator` —— 守卫测试自身的覆盖率

```python
def test_solve_covers_every_generator():
    """Guards the test's own coverage: if a generator's wording drifts, `solve`
    stops recognising it and `test_every_answer_is_arithmetically_correct` would
    silently skip that whole task family."""
    for fn in GENERATORS:
        question, answer = fn(random.Random(0))
        derived = solve(f"{question} End your response with the number only.")
        assert derived is not None, f"test parser does not recognise: {question!r}"
        assert derived == answer, f"{question!r}: parser got {derived}, generator says {answer}"
```

#### 这个测试的独特之处：**它在测"另一个测试"**

```text
test_every_answer_is_arithmetically_correct
    ↓ 依赖
solve() 能识别所有题型
    ↓ 如果某题型识别不了
unchecked += 1（然后被 assert unchecked == 0 抓住）
```

**这个测试提前把"识别不了"变成显式的失败**，而不是依赖那个 `unchecked` 计数器。

**为什么要单独测？** 因为**逐题验证比逐生成器验证慢**：
`test_every_answer_is_arithmetically_correct` 要跑 200 条，
而这个只跑 9 个生成器（每个一次）。

**更快地定位问题**：如果这个失败，你立刻知道是"某个题型的措辞和 `solve()` 不匹配"；
如果只有那个慢测试失败，你要从 200 条里找是哪一条。

#### `fn(random.Random(0))`

**给每个生成器一个全新的 `Random(0)`** —— 保证确定性，
且**每个生成器用独立的随机源**（互不干扰）。

**用固定 seed `0`** —— 这个测试不关心具体数值，
只要"生成 → 解析 → 比较"这条链能走通。

#### `!r` 格式化

```python
f"test parser does not recognise: {question!r}"
```

**`!r` 用 `repr()` 格式化** —— 会显示引号和转义字符：

```text
不用 !r： test parser does not recognise: Compute 28 * 17.
用 !r：   test parser does not recognise: 'Compute 28 * 17.'
```

**对于字符串，`!r` 更好** —— 能看出首尾空格、特殊字符等肉眼不可见的差异。

---

## 5. 运行测试

```powershell
cd E:\My_DL_Projects\tiny-posttrain-lab
$env:USE_TF="0"; $env:USE_FLAX="0"
.\.venv\Scripts\python.exe -m pytest tests -q
```

```text
..................                                                       [100%]
18 passed in 3.66s
```

### 为什么需要 `USE_TF=0`

因为 `test_project_files.py` 会 `import my_lab`，
而 `my_lab/__init__.py` 导入 `rewards` / `memory` / `vram` ——
**这些模块本身不需要 torch，但 `vram.py` 的函数内 import torch 会触发
transformers 的惰性导入链**（见 [01 篇](./01-environment.md)）。

**实际上**：`vram.py` 的 `cuda_available()` 里 `import torch` 会成功
（torch 装了），然后 `torch.cuda.is_available()` 正常返回。
**所以 `USE_TF=0` 在这个测试里其实不是必须的** ——
但设上更安全（万一某个导入链碰到 transformers）。

> **实测：不设 `USE_TF=0` 时，`test_vram.py` 能过**（因为 `vram.py` 只 import torch，
> 不碰 transformers）。**但设上无副作用，养成习惯更好。**

### `-q` 的含义

`-q` = quiet，只显示进度点和总结，不显示每个测试的名字。

**看详细信息**用 `-v`：

```text
tests/test_math_dataset.py::test_generator_is_deterministic PASSED
tests/test_math_dataset.py::test_prompts_are_unique PASSED
...
```

### 测试耗时分布

```text
18 passed in 3.66s
```

**大部分时间花在 `test_every_answer_is_arithmetically_correct`**
（生成 200 条 + 逐条正则匹配）。

**这个耗时是可接受的** —— 开发者可以频繁运行。

---

## 本篇小结：测试的设计要点

1. **只测能纯 CPU 测的部分。** 端到端训练测试太慢，价值密度低。
2. **"可测试性"是架构约束。** `my_lab/` 不在模块级 import torch，
   正是为了让测试能在任何机器上跑。
3. **把"项目规范"变成测试。** `test_every_experiment_has_notes` 强制每个实验都有记录，
   比写在文档里有效。
4. **遍历式断言必须检查集合非空**，否则空目录会"静默通过"。
5. **独立实现验证。** `solve()` 从 prompt 重新推导答案，不复用生成器的结果 ——
   否则就是自己和自己比。
6. **`assert unchecked == 0` 防止空测试。** 任何"遍历 + 跳过"的测试
   都要断言"实际检查的数量"。
7. **测两个方向。** "同 seed 同结果" + "不同 seed 不同结果"，
   只测一个会漏掉一半的 bug。
8. **边界情况要显式测。** `eval_n=0` 的分支即使冗余也值得固定行为。
9. **测不变量而不是数值。** `reserved >= allocated` 能抓住"两个函数写反"这种
   极难发现的错误。
10. **给启发式函数测"单调性"而不是"精确值"。** 否则系数微调就会让测试失败。
11. **报错信息要能定位问题。** `f"{prompt} -> {actual} != {expected}"` 比
    裸 `assert` 有用得多。
12. **docstring 里记录历史 bug。** 解释"为什么需要这个测试"。
13. **测试文件本身也要防漏。** `test_solve_covers_every_generator`
    守卫另一个测试的覆盖率。
14. **已知缺陷用"特征化测试"记录**，而不是假装不存在
    （`extract_final_number` 的截断问题是当前未覆盖的缺口）。
