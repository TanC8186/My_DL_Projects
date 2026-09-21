# 06 · GRPO 训练脚本 `scripts/train_grpo.py`

这是三个训练脚本里最复杂的一个：**要自己生成样本、要调 reward 函数、要在组内比较**。
也是踩坑最多的一个 —— 两个 bug 都让训练完全失效，但日志"看起来正常"。

---

## 0. GRPO 是什么（先建立直觉）

### 三种方法的定位

| | SFT | DPO | GRPO |
|---|---|---|---|
| 需要什么数据 | `(prompt, 好答案)` | `(prompt, 好答案, 坏答案)` | **只需要 `(prompt, 可验证的答案)`** |
| 需要什么额外模型 | 无 | 参考模型 | 无（`beta=0` 时）/ 参考模型（`beta>0`） |
| 训练信号来源 | 模仿标准答案 | 对比好坏 | **自己采样多个答案，按 reward 排序** |
| 适合什么任务 | 通用指令跟随 | 偏好对齐 | **有精确判据的任务**（数学、代码、格式） |

**GRPO 的核心优势：不需要人类标注的偏好对。**
只要有**可自动验证的答案**（比如 `12+30=42` 能用程序检查），
就能让模型自己探索、自己比较。

### 名字的含义：Group Relative Policy Optimization

拆开看：

```text
Group            → 对同一个 prompt 采样一组（比如 4 个）答案
Relative         → 用【组内的均值】当 baseline，而不是训练一个价值网络
Policy           → 优化的是策略（语言模型本身）
Optimization     → 策略梯度方法
```

**advantage 的计算**（`grpo_trainer.py:1683-1692`）：

```python
mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
std_grouped_rewards  = rewards.view(-1, self.num_generations).std(dim=1)
advantages = rewards - mean_grouped_rewards          # 减去组内均值
if self.scale_rewards:
    advantages = advantages / (std_grouped_rewards + 1e-4)   # 再除以组内标准差
```

即：

```text
A_i = (r_i − mean(r_group)) / (std(r_group) + 1e-4)
```

**相比 PPO 的关键简化**：

```text
PPO：  需要一个 critic（价值网络）来估计 baseline
       → 要训练两个模型，显存和时间翻倍

GRPO： 用同一 prompt 的多次采样的均值当 baseline
       → 不需要 critic
       → 代价是每个 prompt 要生成多次（num_generations 倍生成成本）
```

**⚠️ 关键推论：同一组内奖励全相同 → advantage 全 0 → 该步无梯度。**

这条推论是本项目 GRPO 实验结论的基石（见第 7 节）。

---

## 1. 脚本结构

```text
train_grpo.py
├── 引导段
├── build_zero_signal_guard(base_callback, patience)   ← 【新增】防呆回调工厂
└── main()
    ├── 参数（--config / --adapter）
    ├── 【新增】读本地 jsonl（不走 datasets 库）
    ├── 模型段（+ adapter 加载 + 守卫）
    ├── 【关键】构造对话式数据集
    ├── peft_config 二选一
    ├── 【新增】reward 函数的选择与校验
    ├── 训练段（GRPOConfig，多了 num_generations 等）
    └── 落盘段
```

---

## 2. `build_zero_signal_guard` —— 防呆回调

```python
def build_zero_signal_guard(base_callback, patience: int = 20):
    """Abort when GRPO has no learning signal.

    `frac_reward_zero_std == 1.0` means every completion in every group got the
    same reward, so all advantages are 0. loss and grad_norm are then exactly 0.0,
    which reads as "training is stable" rather than "training is dead".

    The usual cause is a prompt/format mismatch: if completions are truncated
    (clipped_ratio ~ 1.0) the model never emits an answer and every reward is 0.
    Aborting early matters because each step costs a full generation pass.
    """

    class ZeroSignalGuard(base_callback):
        def __init__(self) -> None:
            self.patience = patience
            self.streak = 0

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "frac_reward_zero_std" not in logs:
                return
            self.streak = self.streak + 1 if logs["frac_reward_zero_std"] >= 1.0 else 0

            if self.streak >= self.patience:
                print(
                    f"\n[ABORT] reward std has been 0 for {self.streak} logged steps, so every "
                    f"advantage is 0 and no learning can happen.\n"
                    f"         clipped_ratio={logs.get('completions/clipped_ratio')} "
                    f"mean_length={logs.get('completions/mean_length')} "
                    f"reward={logs.get('reward')}\n"
                    f"         Check that prompts carry the chat template and that "
                    f"max_completion_length leaves room for the answer.\n",
                    flush=True,
                )
                control.should_training_stop = True

    return ZeroSignalGuard()
```

### 2.1 为什么是"工厂函数"而不是"类"

**正常写法**（直接定义类）：

```python
class ZeroSignalGuard(TrainerCallback):    # ← 需要模块级 import TrainerCallback
    def __init__(self, patience=20):
        ...
```

**为什么这里要包一层函数？**

因为脚本的设计约定是**把重依赖的 import 放在 `main()` 里**
（让 `--help` 秒退）。但类定义在模块级，`class ZeroSignalGuard(TrainerCallback)`
需要 `TrainerCallback` **在模块加载时**就存在。

**矛盾**：

```text
模块级类定义 → 需要模块级 import TrainerCallback
但脚本约定 → 重型 import 放 main() 内
```

**第一次尝试**把类定义在模块级 + `main()` 里 import → 直接崩：

```text
File "scripts/train_grpo.py", line 20, in <module>
    class ZeroSignalGuard(TrainerCallback):
NameError: name 'TrainerCallback' is not defined
```

**解法：工厂函数。**

```python
def build_zero_signal_guard(base_callback, patience: int = 20):
    class ZeroSignalGuard(base_callback):   # ← base_callback 是运行时参数
        ...
    return ZeroSignalGuard()
```

- 模块级只定义**函数**，不引用 `TrainerCallback`
- `base_callback` 作为**参数**传进来（运行时才需要）
- 调用方在 `main()` 里：`build_zero_signal_guard(TrainerCallback)`

**这是"延迟绑定"的标准手法。** 代价是多了一层缩进，收益是保持了脚本的 import 约定。

> **注意一个细节**：这里返回的是**实例** `ZeroSignalGuard()`，不是类。
> 因为 `callbacks=[...]` 需要实例列表。

### 2.2 闭包捕获 `patience`

```python
def build_zero_signal_guard(base_callback, patience: int = 20):
    class ZeroSignalGuard(base_callback):
        def __init__(self) -> None:
            self.patience = patience      # ← 从外层函数捕获
```

`patience` 是外层函数的参数，内层类的 `__init__` 可以直接读它 —— 这是**闭包**。
比写死 20 更灵活（虽然本项目只用默认值）。

### 2.3 `on_log` 的逻辑

```python
def on_log(self, args, state, control, logs=None, **kwargs):
    if not logs or "frac_reward_zero_std" not in logs:
        return
    self.streak = self.streak + 1 if logs["frac_reward_zero_std"] >= 1.0 else 0
    ...
```

#### `TrainerCallback` 的钩子机制

HuggingFace 的 `Trainer` 在训练循环的关键节点调用注册的回调：

```text
on_train_begin / on_epoch_begin / on_step_begin / on_log / on_step_end / ...
```

`on_log` 在**每次写日志时**触发（受 `logging_steps` 控制）。

**参数含义：**

| 参数 | 内容 |
|------|------|
| `args` | `TrainingArguments` |
| `state` | `TrainerState`（含 `global_step`、`epoch` 等） |
| `control` | `TrainerControl` —— **可写它来干预训练流程** |
| `logs` | 本次要记录的指标字典 |

#### `if not logs or "frac_reward_zero_std" not in logs: return`

**双重保护**：

1. `not logs` → `logs` 可能是 `None` 或空 dict
2. `"frac_reward_zero_std" not in logs` → 这个 key 只在 GRPO 里存在，
   而且只在**有 reward 的日志步**才出现

**如果不在就返回**，避免 `KeyError`。

#### `self.streak` 的累加/重置

```python
self.streak = self.streak + 1 if logs["frac_reward_zero_std"] >= 1.0 else 0
```

**这是一个三元表达式，等价于：**

```python
if logs["frac_reward_zero_std"] >= 1.0:
    self.streak = self.streak + 1
else:
    self.streak = 0
```

**逻辑**：连续多少次日志步的 `frac_reward_zero_std == 1.0`。
一旦出现非零方差的步，计数归零（说明训练还活着）。

#### 为什么用 `patience`（容忍度）而不是立即中止

```python
if self.streak >= self.patience:
```

**因为 GRPO 天然会有零方差步** —— 即使训练正常，
某些 prompt 的 4 个采样也可能恰好全对或全错。

实测数据：**修复后的正常运行里，81.5% 的步都是零方差**。
如果"一出现零方差就中止"，正常训练会被立刻杀掉。

**`patience=20` 的含义**：连续 20 个日志步（`logging_steps=1` 时就是 20 个 micro-step）
**全部**零方差才判定为"死了"。

**这个阈值的依据**：正常训练里零方差的连续长度很少超过 10 步
（因为总会遇到一个有对有错的 prompt）。20 是一个安全的边界。

> **这是一个"经验阈值"，不是理论值。** 注释里没有解释为什么是 20 ——
> 这是可以改进的地方（应该说明"实测正常训练的最长连续零方差步数"）。

#### 中止机制

```python
control.should_training_stop = True
```

**`TrainerControl` 是 HF 提供的"外部干预"通道。**
设置 `should_training_stop = True` 后，`Trainer` 会在当前步结束后优雅停止
（保存 checkpoint、跑完回调），不是暴力 kill。

**对比 `sys.exit()` 或 `raise`**：那些会跳过清理逻辑，可能留下损坏的 checkpoint。

#### 报错信息里带诊断数据

```python
f"         clipped_ratio={logs.get('completions/clipped_ratio')} "
f"mean_length={logs.get('completions/mean_length')} "
f"reward={logs.get('reward')}\n"
f"         Check that prompts carry the chat template and that "
f"max_completion_length leaves room for the answer.\n"
```

**打印三个关键诊断量 + 两条排查方向。**

- `clipped_ratio ≈ 1.0` → 所有生成都被截断 → prompt 格式或长度问题
- `mean_length` 接近 `max_completion_length` → 同上
- `reward = 0` → 完全没有对的样本

**为什么这个设计重要？** 因为这个回调的**触发场景就是"出 bug 了"**。
在那一刻把诊断信息打出来，比事后从日志里翻要高效得多。

**实测：这个回调没被触发**（因为 bug 在 smoke test 阶段就被发现了），
但它存在的意义是**下次再遇到同类问题能在 20 步内止损，而不是跑完 400 步才发现。**

---

## 3. ⚠️ Bug 1：prompt 没套 chat template

### 3.1 现象

第一次跑 GRPO，指标全零：

```text
'completions/mean_length': 192.0            ← 全部撞上 max_completion_length
'completions/clipped_ratio': 1.0            ← 100% 被截断
'completions/mean_terminated_length': 0.0   ← 没有任何一条自然结束
'rewards/accuracy_reward/mean': 0.0         ← 奖励恒为 0
'frac_reward_zero_std': 1.0                 ← 组内无方差
'loss': 0.0, 'grad_norm': 0.0               ← 完全没有学习信号
```

### 3.2 根因：Instruct 模型需要 chat template

**`train_grpo.py` 原本的写法：**

```python
dataset = Dataset.from_list(
    [{"prompt": r["prompt"], "answer": r["answer"]} for r in rows]
)
```

`prompt` 是**裸字符串**：`"Compute 28 * 17. End your response with the number only."`

**TRL 的处理**（`grpo_trainer.py:1379`）：

```python
prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
```

**`maybe_apply_chat_template` 内部先判断 `is_conversational()`**（`data_utils.py:31-69`）：

```python
def is_conversational(example: dict[str, Any]) -> bool:
    supported_keys = ["prompt", "chosen", "rejected", "completion", "messages"]
    example_keys = {key for key in example.keys() if key in supported_keys}
    if example_keys:
        key = example_keys.pop()
        maybe_messages = example[key]
        if isinstance(maybe_messages, list):            # ← 必须是列表
            maybe_message = maybe_messages[0]
            if isinstance(maybe_message, dict) and "role" in maybe_message and "content" in maybe_message:
                return True
    return False
```

**裸字符串不是 list → `is_conversational` 返回 `False` → 不套模板。**

### 3.3 实测对比：同一句话，两种格式

```python
# 实测脚本：同一个 prompt，只改格式
gen("Compute 28 * 17. End your response with the number only.", "RAW-STRING")
gen(tok.apply_chat_template([{"role":"user","content":p}], tokenize=False,
                            add_generation_prompt=True), "CHAT-TEMPLATE")
```

```text
[RAW-STRING] len=192 hit_eos=False
    " Let's break this down step by step:\n\n\\( 28 \\times 17 = (20 + 8) \\times 17 \\)\n\n
     Now distribute the multiplication:\n\n\\( = 20 \\times 17 + 8 \\times 17 \\)\n\n
     \\( = 340 + 136 \\)\n\nFinally, add these two numbers:\n\n\\( = 476 \\)\n\nSo, "

[CHAT-TEMPLATE] len=4 hit_eos=True
    '476'
```

**对比极其鲜明：**

| 格式 | 长度 | 命中 EOS | 输出 |
|------|------|---------|------|
| 裸字符串 | 192（截断） | ❌ | 一路写 LaTeX 推理，**在 192 token 处被切断，永远没吐答案** |
| chat template | **4** | ✅ | `'476'` |

### 3.4 为什么会这样

**Instruct 模型是在"对话格式"上训练出来的。** 它的训练数据长这样：

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Compute 28 * 17. End your response with the number only.<|im_end|>
<|im_start|>assistant
476<|im_end|>
```

**给裸字符串，模型看到的输入是"半截文本"，它只能按"文本续写"的方式工作** ——
而它在预训练时见过的"数学题"后面往往跟着详细推导，所以就写 LaTeX 推理。

**给正确的 chat 格式，模型识别出"这是 user 在问，我该作为 assistant 回答"**，
就按指令要求只输出数字。

**这不是模型"笨"，是我们没给它熟悉的输入格式。**

### 3.5 附带暴露的问题：训练/评测不一致

**`eval_math.py` 一直套了 chat template：**

```python
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
)
```

**而 `train_grpo.py` 没有。** 所以：

```text
训练时：模型看到裸字符串 → 学会的行为是在裸字符串上"续写"
评测时：模型看到 chat 格式 → 但训练从未在这个格式上发生过
        ↓
训练和评测对同一句话的输入【根本不同】
```

**这种不一致不会报错，只会让你得到无法解释的结果。**
在本次的具体表现里，因为奖励全 0，问题被提前暴露了；
如果模型碰巧在裸字符串下也能答对，就会得到"训练有效但评测无提升"的诡异现象。

> **这是一条通用教训**：**训练和推理必须走同一条输入构造路径。**
> 最稳妥的做法是把"构造输入"抽成一个共享函数，两边都调它。

### 3.6 修复

```python
    # TRL applies the chat template only when the prompt is conversational (a list
    # of messages) — see `maybe_apply_chat_template` in grpo_trainer.py. Passing a
    # raw string makes an Instruct model continue plain text instead of answering:
    # every completion then runs to max_completion_length, scores 0, and GRPO gets
    # zero advantage (loss 0.0, grad_norm 0.0) while looking perfectly healthy.
    dataset = Dataset.from_list(
        [
            {"prompt": [{"role": "user", "content": r["prompt"]}], "answer": r["answer"]}
            for r in rows
        ]
    )
```

**改动只有一处**：`r["prompt"]` → `[{"role": "user", "content": r["prompt"]}]`。

**注意 `answer` 列保持不变** —— 它仍然作为额外列存在，
TRL 会把它作为 kwarg 传给 reward 函数（见第 5 节）。

---

## 4. ⚠️ Bug 2：gradient checkpointing 破坏生成

**这个 bug 更隐蔽，因为它在修好 Bug 1 之后仍然存在。**

### 4.1 现象

修好 chat template 后重跑，**指标仍然全零**：

```text
[info] trainable parameters: 494,032,768
{'loss': 0.0, 'grad_norm': 0.0, 'completions/mean_length': 192.0, 'completions/clipped_ratio': 1.0, ...}
```

**注意 `completions/mean_length` 仍然是 192** —— 说明 prompt 格式已经对了
（否则不会是这个数字吗？不，这个数字说明生成仍然被截断）。

### 4.2 排查过程：逐步排除

| 假设 | 实测方法 | 结果 |
|------|---------|------|
| prompt 没套 chat template | 拦截 `generate`，打印实际输入 | 输入是**正确的** chat 格式（46 token）→ **排除** |
| 生成参数（top_k / min_p / repetition_penalty）| 用 `GRPOConfig` 的确切默认值手动生成 | 4 token + EOS，正确 → **排除** |
| 模型状态 | 见下 | **命中** |

#### 拦截 `generate` 看实际输入

```python
_orig = GenerationMixin.generate
def spy(self, inputs=None, **kw):
    if "input_ids" in kw and "CAPTURED" not in _captured:
        _captured["ids"] = kw["input_ids"][0].tolist()
    return _orig(self, inputs=inputs, **kw)
GenerationMixin.generate = spy
```

输出：

```text
[SPY] n_input_ids: 46
[SPY] prompt text: '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.
                    <|im_end|>\n<|im_start|>user\nCompute 28 * 17. End your response with the number only.
                    <|im_end|>\n<|im_start|>assistant\n'
```

**格式完全正确。** 所以问题不在 prompt。

#### 用"只改一个变量"的方式二分模型状态

```python
for tag, kw in [("base/eval", {}),
                ("lora/eval", {"peft": True}),
                ("lora/train", {"peft": True, "train": True}),
                ("lora/train/ckpt", {"peft": True, "train": True, "ckpt": True}),
                ("base/train/ckpt", {"train": True, "ckpt": True})]:
    m = build(**kw); run(tag, m)
```

**结果：**

```text
[base/eval]        len=4   eos=True   '476'
[lora/eval]        len=4   eos=True   '476'
[lora/train]       len=4   eos=True   '476'
[lora/train/ckpt]  len=192 eos=False  '40% percentage form syntax)...'   ← 坏掉了！
```

**前三个都正确，只有第四个坏。** 唯一的差异是 `ckpt=True`。

而 `ckpt=True` 做的事是：

```python
def build(peft=False, train=False, ckpt=False):
    ...
    if ckpt:
        m.gradient_checkpointing_enable()
        m.enable_input_require_grads()      # ← 关键
```

### 4.3 根因

**TRL 的 `GRPOTrainer` 在初始化时启用 gradient checkpointing**（`grpo_trainer.py:983, 986`）：

```python
model.base_model.gradient_checkpointing_enable()
...
model.gradient_checkpointing_enable()
```

并通过 `_prepare_gradient_checkpointing` 调用 `enable_input_require_grads()`
（`grpo_trainer.py:596-608`）：

```python
def _prepare_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig):
    """Prepare the gradienting checkpointing for the model."""
    # For models that use gradient_checkpointing, we need to attach a hook that enables input
    # to explicitly have `requires_grad=True`, otherwise training will either silently
    # fail or completely fail.
    if args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    return model
```

**`enable_input_require_grads()` 注册了一个 forward hook**，
在输入 embedding 的输出上设置 `requires_grad_(True)`。

**为什么训练需要这个 hook？** 因为 gradient checkpointing 要求计算图连通。
如果整个模型的参数都冻结（LoRA 场景），输入 embedding 的输出不 require grad，
checkpoint 段就没有梯度可回传，会报 `None of the inputs have requires_grad=True`。

**所以 hook 本身是对的 —— 对训练是对的。**

**问题是：TRL 从不在生成前关掉它。**

```text
搜索整个 trl 0.21.0 源码：gradient_checkpointing_disable()
→ 只在 modeling_base.py:95-96 出现，而且是【赋值透传】，不是【调用】
→ 全仓库没有任何一处调用它
```

**于是生成阶段也带着这个 hook**，每个 token 的 forward 都受影响，
输出变成乱码，永远不吐 EOS。

### 4.4 实测验证

| 模型状态 | 生成长度 | 命中 EOS | 输出 |
|----------|---------|---------|------|
| base / eval | 4 | ✅ | `'476'` |
| + LoRA / eval | 4 | ✅ | `'476'` |
| + LoRA / train | 4 | ✅ | `'476'` |
| **+ LoRA / train / gradient_checkpointing** | **192** | ❌ | `'40% percentage form syntax)...'` |

**只有最后一个坏。** 而且注意输出内容 —— `'40% percentage form syntax)'`
完全是无关的乱码，不是"答错了"，是"生成过程被破坏了"。

### 4.5 修复：关掉 gradient checkpointing

```yaml
  # MUST stay false. TRL's GRPOTrainer calls gradient_checkpointing_enable() and
  # enable_input_require_grads() but never disables them before generating. The
  # embedding hook then corrupts the generation pass: measured on this machine,
  # the same prompt yields a clean 4-token answer ('476') without checkpointing
  # and 192 tokens of garbage with it, so every reward is 0 and GRPO gets no
  # signal at all. VRAM is not a constraint here (0.5B + LoRA peaks near 1.5 GB),
  # so simply leaving it off is the right trade.
  gradient_checkpointing: false
```

**为什么关掉是"正确的 trade"而不是"妥协"？**

因为 **8GB 上根本不需要它**：

```text
实测（关掉后）：GRPO 训练 peak_allocated = 1.22 GB / 8 GB
        ↓
用了 15% 的显存，完全不需要用计算换显存
```

**gradient checkpointing 是"显存紧张时用时间换显存"的手段。**
显存不紧张时用它，纯亏（多 20~30% 训练时间 + 引入这个 bug）。

### 4.6 意外收获：生成速度提升 17 倍

```text
修复前：8 条样本，64.15 s
修复后：8 条样本，  3.74 s
```

**为什么差这么多？** 因为 **gradient checkpointing 会强制 `use_cache=False`**：

```text
transformers 的警告：
  "Caching is incompatible with gradient checkpointing in QwenNDecoderLayer.
   Setting `past_key_value=None`."
```

**生成时没有 KV cache 意味着**：

```text
有 KV cache：生成第 n 个 token 只需算 1 次 attention（复用前 n-1 个的 K/V）
             → 总复杂度 O(n)

无 KV cache：生成第 n 个 token 要重算前 n-1 个 token 的 K/V
             → 总复杂度 O(n²)
```

**这就是 17 倍差距的来源。** 这个 bug 不只是"让训练失效"，
还让每次生成慢了 17 倍 —— **如果它没导致奖励全零（比如某些模型状态恰好能工作），
你会以为是"GRPO 本来就这么慢"。**

### 4.7 修复后验证

```text
修复前                         修复后
completions/mean_length: 192  →  4
clipped_ratio: 1.0            →  0.0
rewards/accuracy_reward/mean: 0.0 → 1.0
entropy: 5.38                 →  0.23
peak VRAM: 1.41 GB            →  1.05 GB
```

**`entropy` 从 5.38 降到 0.23** 是个好信号 —— 说明模型变得**自信**了
（正常回答"476"时分布很尖）。乱码输出对应的分布是平的（高熵）。

---

## 5. reward 函数的接线

### 5.1 选择与校验

```python
    # The config only ships one reward today; keep the lookup explicit so adding a
    # second one is a real change rather than a silent fallback.
    reward_name = grpo_cfg.get("reward_function", "accuracy")
    if reward_name != "accuracy":
        raise ValueError(f"Unknown reward_function: {reward_name!r} (only 'accuracy')")
    reward_fn = accuracy_reward
```

**这段代码替代了原来的：**

```python
# 原来的写法（有 bug）
reward_fn = accuracy_reward if grpo_cfg.get("reward_function", "accuracy") == "accuracy" else accuracy_reward
```

**原写法的问题**：三元表达式的**两个分支返回同一个东西**。这是个 no-op ——
无论配置写什么，都用 `accuracy_reward`。**配置里的 `reward_function` 字段完全无效。**

**这种代码很危险**：它看起来像是在做选择，实际什么都没做。
读代码的人会以为"支持多个 reward"，实际上不支持。

**修复后的逻辑**：

```text
读配置 → 不等于 "accuracy" → 抛异常
       → 等于 "accuracy" → 用 accuracy_reward
```

**关键改进：未知值报错，而不是静默回退。**

对比两种风格：

| 风格 | 行为 | 风险 |
|------|------|------|
| 静默回退（`else accuracy_reward`） | 配置写错 → 用默认值 → **不报错** | 你以为在用 A，实际在用 B |
| **显式报错（现在的写法）** | 配置写错 → 立刻失败 | 安全 |

**注释解释了设计意图**：

```python
# keep the lookup explicit so adding a second one is a real change rather than a silent fallback.
```

**这是"让扩展点显式化"** —— 将来加第二个 reward 时，
你**必须**改这段代码（而不是以为配置改了就行）。

### 5.2 TRL 怎么调用这个函数

**这是理解 reward 设计的关键。** 看 `grpo_trainer.py:1293-1322`：

```python
def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
    device = self.accelerator.device
    rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

    # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
    keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

    # This allows for dynamic reward shaping based on training progress.
    reward_kwargs["trainer_state"] = self.state

    for i, (reward_func, ...) in enumerate(...):
        ...
        output_reward_func = reward_func(
            prompts=prompts, completions=completions,
            completion_ids=completion_ids_list, **reward_kwargs
        )
        output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
        rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
```

**关键行：**

```python
keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
```

**数据集里除 `prompt` / `completion` / `completion_ids` 外的所有列，都会被收集成 kwargs。**

我们的数据集列是 `['prompt', 'answer']` → `keys = ['answer']` → 传 `answer=[...]`。

**这解释了 `accuracy_reward(completions, answer, **kwargs)` 的签名为什么能拿到 `answer`。**

#### 一个隐藏的重要细节：`trainer_state`

```python
reward_kwargs["trainer_state"] = self.state
```

**TRL 会传一个 `trainer_state`，里面有 `global_step`、`epoch` 等。**

**这是"动态 reward 塑形"的入口** —— 比如：

```python
def curriculum_reward(completions, answer, trainer_state=None, **kwargs):
    step = trainer_state.global_step if trainer_state else 0
    # 前 100 步只要求格式对，之后才要求答案对
    if step < 100:
        return [1.0 if looks_like_number(c) else 0.0 for c in completions]
    ...
```

**本项目没用这个能力**，但知道它存在很重要。

#### `None` 会被转成 `NaN`

```python
output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
```

**如果你的 reward 函数返回 `None`，TRL 会把它当 NaN 处理**
（后面有逻辑检测"整行都是 NaN"并报警告）。

**这意味着**：如果某个样本无法判定，返回 `None` 比返回 `0.0` 更诚实 ——
`0.0` 是"判定为错"，`None` 是"无法判定"。**但本项目选择返回 `0.0`**
（见 `rewards.py`），因为对错是二元的，无法解析就是错。

### 5.3 数据集列名必须是 `answer`

```python
dataset = Dataset.from_list(
    [{"prompt": [...], "answer": r["answer"]} for r in rows]
)
```

**`answer` 这个列名是**和 reward 函数的参数名**绑定的：

```python
def accuracy_reward(completions: Sequence[str], answer: Sequence[str], **kwargs) -> list[float]:
                              ↑ 名字必须匹配
```

**如果数据集里叫 `gold` 而不是 `answer`，`answer` 就会缺失 → 报错。**

**这是一个隐式契约**：数据集的列名 ←→ reward 函数的参数名。
**改任何一边都要改另一边**，而 Python 不会在导入时检查这一点
（只有运行时调用才会 `TypeError: missing required argument`）。

> **更健壮的做法**是给 `answer` 一个默认值或显式校验。但当前设计
> 依赖"列名和参数名一致"这个约定，并在 `data/README.md` 里写明了格式。

---

## 6. 数据段：不走 datasets 库

```python
    data_path = Path(data_cfg["dataset_path"])
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing {data_path}. Create math_sample.jsonl with fields prompt + answer."
        )

    rows = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    max_samples = data_cfg.get("max_samples")
    if max_samples:
        rows = rows[: int(max_samples)]

    print(f"[info] prompts={len(rows)} -> {out_dir}")
    print("[warn] keep gradient_checkpointing=false; see the note in the YAML.")
```

### 为什么不用 `datasets.load_dataset`

**SFT 用了 `load_dataset`，GRPO 却手写 jsonl 解析。** 为什么？

因为 GRPO 需要**在构建 Dataset 之前对 rows 做处理**（截断到 `max_samples`），
而且数据格式简单（只有 `prompt` / `answer` 两个字段）。
手写 5 行比 `load_dataset` + `select` 更直接。

**但这是一个不一致** —— 同一个项目里两种数据加载风格。
**好处**：GRPO 的加载逻辑一目了然（jsonl 一行一条）。
**代价**：如果将来要支持 parquet 或 HF 数据集，要改这里。

### `line.strip()` + `if line:` 的组合

```python
line = line.strip()
if line:
    rows.append(json.loads(line))
```

- `strip()` 去掉首尾空白和换行符
- `if line:` 跳过**空行**（jsonl 文件末尾常有空行）

**没有这个检查**，`json.loads("")` 会抛 `JSONDecodeError`。
**这是处理 jsonl 的标准防御。**

### 启动警告

```python
print("[warn] keep gradient_checkpointing=false; see the note in the YAML.")
```

**把最重要的约束在启动时再提醒一次。** 因为 YAML 的注释只有主动去看才会看到，
而这行会出现在日志里。

**实测：这个警告在修复后的训练日志里仍然打印**，作为持续提醒。

---

## 7. `GRPOConfig` 的差异参数

```python
    config = GRPOConfig(
        output_dir=str(out_dir),
        learning_rate=float(grpo_cfg.get("learning_rate", 1e-6)),
        per_device_train_batch_size=int(grpo_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(grpo_cfg.get("gradient_accumulation_steps", 4)),
        num_train_epochs=float(grpo_cfg.get("num_train_epochs", 1)),
        num_generations=int(grpo_cfg.get("num_generations", 2)),
        max_prompt_length=int(data_cfg.get("max_prompt_length", 128)),
        max_completion_length=int(data_cfg.get("max_completion_length", 128)),
        temperature=float(grpo_cfg.get("temperature", 1.0)),
        beta=float(grpo_cfg.get("beta", 0.0)),
        bf16=bool(grpo_cfg.get("bf16", True)),
        gradient_checkpointing=bool(grpo_cfg.get("gradient_checkpointing", True)),
        logging_steps=int(grpo_cfg.get("logging_steps", 1)),
        save_steps=int(grpo_cfg.get("save_steps", 100)),
        report_to=list(grpo_cfg.get("report_to") or []),
        seed=int(grpo_cfg.get("seed", 42)),
    )
```

### 7.1 `num_generations` —— GRPO 独有的核心参数

```python
num_generations=int(grpo_cfg.get("num_generations", 2)),
```

**同一个 prompt 采样几次。** 这个数字直接决定：

| 影响 | 说明 |
|------|------|
| **组内方差的可能性** | 采样越多，越可能出现"有对有错"的组 → 才有 advantage |
| 生成成本 | 线性增长（4 次采样 = 4 倍生成时间） |
| advantage 估计质量 | 样本越多，均值和标准差估计越准 |

**实测数据（这是本项目最有价值的分析之一）：**

```text
num_generations = 4 时：
  frac_reward_zero_std = 1.0 的步数占 81.5%
```

**即：81.5% 的步里，4 个采样拿到了完全相同的奖励 → advantage 全 0 → 该步白跑。**

**为什么？** 因为基座模型在这些题上已经 87% 正确，
且分布很尖（`entropy ≈ 0.23`）—— 4 次采样经常给出同一个答案。

**这就是 GRPO 的适用条件**：**任务必须对模型足够难**，
难到多次采样会产生分歧。如果模型已经全对，GRPO 拿不到任何信号。

**配置里的注释记录了这个推理**：

```yaml
  # 4 generations per prompt. With a sharp 0.5B distribution, 2 samples are often
  # byte-identical, which makes the group's advantage 0 and wastes the step. More
  # samples raise the chance of a mixed group, which is where GRPO actually learns.
  # Affordable here because generation is fast without gradient checkpointing.
```

### 7.2 `temperature` 被强制为 `do_sample=True`

```python
temperature=float(grpo_cfg.get("temperature", 1.0)),
```

**TRL 的 GRPO 生成配置是硬编码 `do_sample=True` 的**（`grpo_trainer.py:835-838`）：

```python
generation_kwargs = {
    "max_new_tokens": self.max_completion_length,
    "do_sample": True,                        # ← 强制采样
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
    "temperature": self.temperature,
    "top_p": self.top_p,
    "top_k": self.top_k,
    "min_p": self.min_p,
    "repetition_penalty": self.repetition_penalty,
    ...
}
```

**为什么必须采样？** 因为**贪心解码对同一个 prompt 永远给同一个答案**：

```text
贪心：prompt → 唯一确定的输出
      → 组内 4 个样本完全相同
      → 组内标准差 = 0
      → advantage = 0
      → 永远没有梯度
```

**采样才能产生组内多样性，组内多样性才是 GRPO 的学习信号来源。**

### 7.3 `beta=0.0` 的取舍

```python
beta=float(grpo_cfg.get("beta", 0.0)),
```

**GRPO 里 `beta` 是 KL 惩罚系数**（不是 DPO 的温度）：

```python
# grpo_trainer.py:1885-1886
if self.beta != 0.0:
    per_token_loss = per_token_loss + self.beta * per_token_kl
```

**`beta=0.0` 的效果**：

| | `beta = 0` | `beta > 0` |
|---|---|---|
| 参考模型 | **不需要**（省一次前向 + 省显存） | 需要（`ref_model` 或 adapter 禁用） |
| 约束 | 无。策略可以自由漂移 | 用 KL 拉回参考模型 |
| 风险 | **可能遗忘、跑偏、reward hacking** | 更稳，但探索受限 |

**本项目选 `beta=0`** 的理由：8GB 上省显存和时间的收益明显。

**代价**：没有 KL 约束，模型可能为了拿奖励而走偏。
**这可能是"修好 3 题又弄坏 2 题"（churn）的原因之一。**

> **可讲的点**：`beta` 是 GRPO 里少见的"可以设为 0 且有实际意义"的参数。
> 它对应的是"纯策略梯度 + 组内 baseline"，理论上仍然有效（类似 REINFORCE with baseline），
> 只是没有对预训练知识的保护。

### 7.4 `max_completion_length` 与截断的双刃剑

```python
max_completion_length=int(data_cfg.get("max_completion_length", 128)),
```

**生成超过这个长度会被硬截断。** 对"取最后一个数字"的 reward 来说，
截断 = 奖励 0。

**配置里的注释说得很清楚**：

```yaml
  # Completions longer than this are truncated, so the model never emits its
  # final number and scores 0. That punishes rambling, which is the intent,
  # but it also adds noise. Raise it if the reward curve looks erratic.
```

**双刃剑：**

- **好处**：惩罚啰嗦。模型学会"简短作答"能拿分
- **坏处**：如果模型真的需要更多 token 才能算对，它被惩罚了"思考"这个正确行为

**实测：修复后平均生成长度只有 4.62 token**（因为模型直接输出答案），
所以 192 完全够用。**这个参数在修复后其实不再是瓶颈。**

### 7.5 `accum` 必须 ≥ `num_generations`

**这个约束在 `GRPOConfig.__post_init__` 里检查**（`grpo_config.py:613`）：

```python
if self.generation_batch_size % self.num_generations != 0:
    raise ValueError(
        f"generation_batch_size ({self.generation_batch_size}) must be divisible "
        f"by num_generations ({self.num_generations})."
    )
```

其中：

```python
generation_batch_size = per_device_train_batch_size * gradient_accumulation_steps
```

**实测报错**（配 `batch=1, accum=2, num_generations=4`）：

```text
ValueError: generation_batch_size (2) must be divisible by num_generations (4).
```

**为什么有这个约束？** TRL 一次生成 `generation_batch_size` 条样本，
然后按 `num_generations` 分组算 advantage。**如果总量不能被组大小整除，
就会出现"半个组"，无法计算组内均值和标准差。**

**修复**：把 `accum` 从 2 提到 4。

```yaml
  # TRL requires generation_batch_size (= batch x accum) to be divisible by
  # num_generations, so accum must be >= 4 here. With 2 epochs that is
  # 200 prompts x 2 / 4 = 100 optimizer steps.
  gradient_accumulation_steps: 4
```

**这个坑的教训**：调 `num_generations` 时必须同时检查 `accum`。
**配置注释里直接写了这条约束**，避免下次再撞。

### 7.6 优化步数的算术（写在注释里）

```yaml
  num_train_epochs: 2
```

```text
200 条 prompt × 2 epoch = 400 micro-step
400 / accum(4) = 100 次优化器更新
```

**为什么要把这个算式写进配置注释？**

因为**"100 次更新"是判断实验是否可能有效果的关键数字**。
GRPO 的结论是"100 次更新里只有约 18 次有梯度信号"（81.5% 零方差）——
如果连总步数都不知道，这个分析根本做不出来。

**这是一种"把关键计算留在配置里"的习惯**：读配置的人立刻知道这次实验的规模。

---

## 8. 实测结果与解读

### 8.1 训练日志分析

```text
logged steps      : 400          ← micro-step 数
mean train reward : 0.8638       ← 训练集平均奖励
last 20 mean     : 0.8125       ← 末段反而略低
zero_std steps    : 81.5%        ← 组内无方差的比例
grad_norm != 0    : 74 steps (18.5%)
max grad_norm     : 67.67
```

**训练奖励没有提升**（0.8638 全程均值，末 20 步 0.8125）。

### 8.2 held-out 评测结果

```text
基座：    87/100 = 0.8700
GRPO 后： 88/100 = 0.8800
```

**净 +1 题。**

### 8.3 失败集合对比（比总分更有信息量）

```text
base failures (13): 280, 513, 65, 369, 340, 121, 73, 260, 129, 624, 27, 69, 229
grpo failures (12): 280, 513, 308, 340, 48, 121, 260, 129, 624, 27, 69, 229

fixed by GRPO : 65, 369, 73          ← 修好 3 题
broken by GRPO: 308, 48              ← 弄坏 2 题
still failing : 280, 513, 340, 121, 260, 129, 624, 27, 69, 229   ← 10 题纹丝不动
```

**这是 churn（此消彼长），不是系统性提升。**

**原 13 个失败里 10 个完全没变** —— 说明模型的核心弱点（两位数乘法）没被触动。

### 8.4 统计显著性

```text
n = 100, p ≈ 0.87
标准误 = sqrt(p(1-p)/n) = sqrt(0.87 × 0.13 / 100) = sqrt(0.001131) = 0.0336 ≈ 3.4%

观测到的变化：+1%（0.87 → 0.88）
```

**+1% 远小于 ±3.4% 的标准误。** 所以：

> **不能宣称"GRPO 提升了准确率"。**

**如果真要下结论**，需要：

1. 多个 seed 重复评测，量化噪声
2. 更大的评测集（n=1000 时标准误降到 1.1%）
3. 或者选择更敏感的指标（比如"平均生成长度"、"格式合规率"）

### 8.5 为什么没有提升（两个原因）

#### 原因 1：有效更新次数太少（主因）

```text
100 次优化器更新 × (1 - 81.5%) = 约 18.5 次有效更新
```

**18 次更新不足以改变一个 0.5B 模型的行为。** 证据是改动幅度：
`lora_B max|w| = 6.31e-04`（对比 DPO 的 `2.04e-04`，量级类似）。

#### 原因 2：任务对模型太简单

```text
基座 held-out 准确率 0.87
        ↓
组内 4 个采样大概率全对或全错
        ↓
81.5% 的步没有 advantage
```

**GRPO 的适用条件是"任务有难度、采样有分歧"。** 这里两个条件都不满足。

> **这是一个可操作的结论**：下一步应该**加大任务难度**（3×3 乘法、多步应用题），
> 把基座准确率压到 40~60%，那时组内方差才会充足。
> **这比"调大 lr"或"多跑几轮"更对症。**

### 8.6 ⚠️ 第三个发现：GRPO 的 `loss` 不可信

```text
{'loss': 0.0, 'grad_norm': 19.407, ...}    ← 同一行自相矛盾
train_loss: 3.166496753692627e-10          ← 几乎精确为零
```

**`loss` 恒为 0（或 1e-10），但 `grad_norm` 在 18.5% 的步上非零，最高 67.67。**

#### 怎么验证训练真的发生了

**从零初始化的 LoRA，`lora_B` 初始恒为 0。** 所以检查它是否还是 0：

```python
sd = load_file("experiments/2026-09-grpo-math/output/final/adapter_model.safetensors")
b_keys = [k for k in sd if "lora_B" in k]
b_allzero = all(bool((sd[k].float() == 0).all()) for k in b_keys)
b_absmax = max(sd[k].float().abs().max().item() for k in b_keys)
```

```text
total tensors=192  lora_A=96  lora_B=96
lora_B all-zero? False   lora_B max|w| = 6.313614e-04
VERDICT: TRAINED (B moved away from 0)
```

**训练确实发生了。**

#### 为什么 `loss` 会是 0

**看 GRPO 的 loss 计算**（`grpo_trainer.py:1888-1889`）：

```python
if self.loss_type == "grpo":
    loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
```

而 `per_token_loss` 来自：

```python
per_token_loss1 = coef_1 * advantages.unsqueeze(1)
per_token_loss2 = coef_2 * advantages.unsqueeze(1)
per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
```

**对于 advantage = 0 的组，`per_token_loss` 恒为 0。**

**81.5% 的步是零方差 → 那些步的 loss 精确为 0。**
剩下 18.5% 的步应该有非零 loss，但**被 400 步平均后拉到了 1e-10 量级**。

**所以 `loss` 不是"错的"，而是"被大量零值稀释到看不出信号"。**

#### 判断 GRPO 是否真的在训练，看什么

| 指标 | 可靠性 |
|------|--------|
| `loss` | ❌ 被零方差步稀释，不可信 |
| `grad_norm` | ⚠️ 能说明"有梯度"，但**不能说明"有参数被更新"**（DPO 的教训） |
| `frac_reward_zero_std` | ✅ 直接反映学习信号的比例 |
| **最终权重变化** | ✅✅ **唯一可靠的证据** |

**两个实验合起来得出的通用原则：**

```text
DPO 教训：grad_norm 非零 ≠ 有参数被更新（优化器可能是空的）
GRPO 教训：loss 为 0 ≠ 训练失败（被零值稀释）

→ 结论：指标只是线索，产物才是证据。
```

---

## 9. 显存分析

| 实验 | peak_allocated | peak_reserved |
|------|---------------|---------------|
| SFT | 1.87 GB | 4.89 GB |
| DPO | 2.76 GB | 3.56 GB |
| **GRPO** | **1.22 GB** | **1.49 GB** |
| 推理（base 或 +LoRA） | 0.94~0.95 GB | 1.00 GB |

**GRPO 的显存最低。** 为什么？三个原因：

1. **关掉了 gradient checkpointing** → 但这一项应该**增加**激活值显存才对
2. **LoRA `r=8`** 而不是 `r=16` → adapter 参数减半（但仍然很小）
3. **序列短**：`max_prompt_length=128` + `max_completion_length=192`，
   而实际生成长度平均只有 **4.62 token** → 实际序列很短

**第 3 点是关键。** SFT 的 `max_seq_length=512` 且实际用满，
GRPO 的实际序列长度只有 100 多 token（prompt 46 + completion 4.6）。

**这解释了为什么"关掉 gradient checkpointing 显存还更低"** ——
不是关掉它省了显存，而是**序列短得多**，激活值本来就小。

> **这是一个容易被误读的对比。** 如果只看到"GRPO 显存最低"，
> 可能得出"GRPO 比 SFT 省显存"的错误结论。
> **正确的解读是"这些实验的序列长度和模型配置不同，不能直接比"。**
>
> 这也再次印证了 [05 篇](./05-train-dpo.md) 的结论：
> **跨实验比显存必须控制变量**（固定步数、固定序列长度、固定模型）。

---

## 本篇小结：GRPO 脚本的设计要点

1. **GRPO 不需要人类偏好数据，只需要可验证的答案。** 这让它适合数学/代码/格式类任务。
2. **advantage 是组内相对的** `(r_i − mean)/std`，所以**组内奖励全同 → 无梯度**。
3. **`do_sample=True` 是强制的** —— 贪心解码会让组内样本完全相同，GRPO 无法学习。
4. **prompt 必须是对话格式**（消息列表），否则 Instruct 模型退化成文本续写器。
   实测差异：裸字符串 → 192 token 乱写；chat 格式 → 4 token 正确输出。
5. **训练和评测必须用同一条输入构造路径。** 否则不报错但结果无法解释。
6. **`gradient_checkpointing` 必须关掉。** TRL 开了 `enable_input_require_grads()` 的 hook
   却不在生成前关，会污染生成 → 奖励全 0 → 无信号。
7. **关掉它还有个意外收获：生成快 17 倍**（原本因 checkpointing 强制 `use_cache=False`，
   生成是 O(n²)）。
8. **`accum × batch` 必须能被 `num_generations` 整除**，否则 `GRPOConfig` 报错。
9. **`beta=0` 省一次参考前向**，代价是没有 KL 约束（可能是 churn 的原因之一）。
10. **reward 函数通过数据集列名接收额外信息**（我们的 `answer` 列 →
    reward 函数的 `answer` 参数）。改列名要同步改参数名。
11. **`GRPOConfig` 会传 `trainer_state`**，这是动态 reward 塑形的入口。
12. **`loss` 不可信**（被 81.5% 的零方差步稀释到 1e-10）。
    判断训练是否发生要看**最终权重变化**。
13. **GRPO 的适用条件是"任务足够难"**。基座 87% 正确 + 尖分布 →
    81.5% 的步没有信号。**下一步应该加大难度，而不是调 lr。**
14. **防呆回调在 20 步内止损**，避免白烧几十分钟的生成时间。
