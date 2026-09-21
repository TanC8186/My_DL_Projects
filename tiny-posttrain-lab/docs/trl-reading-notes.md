# TRL 源码精读笔记

对照阅读对象：

| | 路径 | 版本 |
|---|---|---|
| 本地安装（**实际运行**） | `.venv/Lib/site-packages/trl/` | **0.21.0** |
| 只读参考（`references/`） | `references/trl/` | **1.14.0.dev0**（main 分支） |

> ⚠️ **两个版本差了一个大版本，代码并不一致。**
> 本笔记的所有结论都基于 **0.21.0（实际跑的版本）**，引用会标注行号。
> 如果照 `references/trl` 去对照，`SFTTrainer` 的 loss 路径、`GRPOConfig` 的字段名
> 都可能对不上。读源码前先确认版本，这本身就是一条经验。

检索地图：

| 问题 | 去哪看 |
|------|--------|
| SFT loss 算在哪些 token | `sft_trainer.py` → `DataCollatorForLanguageModeling` |
| LoRA 怎么挂、参数量 | `peft`（不在 TRL 里）；TRL 只负责传 `peft_config` |
| DPO loss 公式 | `dpo_trainer.py` → `dpo_loss()` |
| DPO 参考模型是谁 | `dpo_trainer.py` → `null_ref_context()` |
| GRPO advantage | `grpo_trainer.py` → `_generate_and_score_completions()` |
| GRPO loss / 裁剪 | `grpo_trainer.py` → `compute_loss()` |
| reward 函数怎么被调用 | `grpo_trainer.py` → `_calculate_rewards()` |

---

## 1. SFT：loss 到底算在哪些 token 上

这是面试高频问题，也是最容易被想当然的地方。

### 0.21.0 的实际路径

TRL 用 `DataCollatorForLanguageModeling` 构造 `labels`（`sft_trainer.py:103`）：

```python
# sft_trainer.py:200-205
if "labels" in examples[0]:
    labels = [torch.tensor(example["labels"]) for example in examples]
else:
    labels = [torch.tensor(example["input_ids"]) for example in examples]   # ← 默认等于 input_ids
...
# sft_trainer.py:240-250
output["labels"] = pad(labels, padding_value=-100, ...)        # 只 mask padding
if self.completion_only_loss and "completion_mask" in examples[0]:
    output["labels"][completion_mask == 0] = -100              # mask 掉 prompt
if "assistant_masks" in examples[0]:
    output["labels"][assistant_masks == 0] = -100              # mask 掉非 assistant
```

**关键结论：默认 `labels = input_ids`，只把 padding 置为 -100。
也就是说 loss 是在全部 token 上算的，包括 prompt（指令）部分。**

### 一个容易误读的细节

```python
if self.completion_only_loss and "completion_mask" in examples[0]:
```

注意这里是 **`and`**。`completion_only_loss=True` 本身**不会**产生掩码，
它只是一个**开关**，真正的掩码来自数据集里的 `completion_mask` / `assistant_masks` 列。

所以"我设了 `completion_only_loss=True`，所以只学回答"是错的 ——
数据里没有对应列的话，它什么都没 mask。

### 那 `completion_only_loss` 默认是什么

```python
# sft_config.py:218
completion_only_loss: Optional[bool] = field(default=None, ...)

# sft_trainer.py:469-472
if args.completion_only_loss is None:
    self.completion_only_loss = "prompt" in dataset_sample    # ← 由数据集列决定
else:
    self.completion_only_loss = args.completion_only_loss
```

### 本项目的实际情况（实测）

`configs/sft_qwen05b_8g.yaml` 走的是 `dataset_text_field="text"` 路径，
数据集列是 `['instruction', 'input', 'output', 'text']`：

- 没有 `prompt` 列 → `completion_only_loss = False`
- 没有 `completion_mask` / `assistant_masks` 列 → 没有任何额外掩码

**⇒ 我们的 SFT loss 算在全部 token 上，模型在学习预测 `### Instruction:` 这些指令文本本身。**

这不是 bug，是默认行为；但它意味着 loss 里混入了"预测指令"的成分，
比较不同实验的 loss 时要知道这一点。

> 想只学回答，两条路：
> 1. 数据改成 `prompt` / `completion` 两列（对话式）；
> 2. 数据带上 `assistant_masks` 列（由 chat template 生成，推荐用于对话数据）。

### 顺带看到的

- `_chunked_cross_entropy_loss`（`sft_trainer.py:121`）：分块算 CE，避免一次性
  物化巨大的 logits（`[B, T, vocab]`）—— 8G 场景下有用。
- `dft_loss`（`sft_trainer.py:791`）：DFT 论文的实现，是另一种 SFT 目标。
- 新版 `references/trl` 里是 `labels = input_ids.clone(); labels[attention_mask == 0] = -100`，
  写法不同但语义等价。

---

## 2. LoRA：为什么省显存

LoRA 在 `peft` 里，TRL 只把 `peft_config` 传下去。所以"TRL 怎么实现 LoRA"这个问题问错了，
应该是"PEFT 怎么实现"。

### 参数量的精确计算

本项目实测：`r=16, alpha=32, target_modules=[q_proj, k_proj, v_proj, o_proj]`
在 Qwen2.5-0.5B 上是 **192 个张量 / 2,162,688 个参数 = 全模型的 0.436%**。

手算验证（`config.json`：`hidden_size=896`，`num_hidden_layers=24`，
`num_attention_heads=14`，`num_key_value_heads=2`）：

```text
head_dim = 896 / 14 = 64
kv_dim   = num_key_value_heads * head_dim = 2 * 64 = 128     ← GQA：KV 比 Q 窄

每个 LoRA 矩阵是 [r, in] + [out, r]，参数量 = r * (in + out)

q_proj: 16 * (896 + 896) = 28672
k_proj: 16 * (896 + 128) = 16384        ← 因为 GQA，k/v 的输出维度只有 128
v_proj: 16 * (896 + 128) = 16384
o_proj: 16 * (896 + 896) = 28672
------------------------------------
每层合计 90112
90112 * 24 层 = 2,162,688   ✓ 与实测完全一致
```

**注意 `k_proj` / `v_proj` 的 `out` 是 128 而不是 896** —— GQA 让 KV 投影的输出维度
等于 `num_key_value_heads * head_dim`。算 LoRA 参数量时这里最容易算错。

### 省显存省在哪

| 项 | 全参微调 | LoRA |
|---|---|---|
| 权重 | 0.5B × 2 B = 1.0 GB | 1.0 GB（冻结，仍需驻留）|
| 梯度 | 1.0 GB | ~4 MB（仅 adapter）|
| AdamW 状态 (m, v) | 2 × 4 B × 0.5B = **4.0 GB** | ~8 MB |
| 激活值 | 需存 | 需存（同样要 gradient checkpointing）|

**省的主要是梯度 + 优化器状态（约 5 GB → 约 12 MB）**，权重本身并不省。
这也解释了为什么 8G 上 LoRA 是刚需：全参的优化器状态就 4 GB 起。

实测印证：SFT（LoRA r=16）训练峰值 `allocated 1.87 GB`，
其中底座权重约 1.0 GB、LoRA 参数+梯度+优化器约 10 MB，剩下是激活值。

### 推理侧几乎免费

base 推理 0.94 GB → 挂上 r=16 adapter 0.95 GB，**只多约 10 MB**。
adapter 文件 8.47 MB（192 个张量，bf16 存储）。

---

## 3. DPO：loss 公式与参考模型

### loss（`dpo_trainer.py:1042-1046`）

```python
logratios = chosen_logps - rejected_logps
ref_logratios = ref_chosen_logps - ref_rejected_logps
logits = logratios - ref_logratios
```

对应论文里的

```text
logits = β [ log π_θ(y_w|x)/π_ref(y_w|x) − log π_θ(y_l|x)/π_ref(y_l|x) ]
```

注意 **`beta` 不在 `logits` 里**，而是在算 reward 时乘上（`dpo_trainer.py:1111-1112`）：

```python
chosen_rewards = (chosen_logps - ref_chosen_logps) * self.beta
rejected_rewards = (rejected_logps - ref_rejected_logps) * self.beta
```

所以日志里的 `rewards/chosen` 就是论文中的隐式奖励 `β log(π_θ/π_ref)`。

### `logps` 是求和，不是平均

```python
# dpo_trainer.py:1571
all_logps = per_token_logps[:, 1:].sum(-1)
```

**这是长度有偏的。** 实测同一批数据里：

```text
logps/chosen: -425.5   logps/rejected: -321.0    ← rejected 更短，所以更不负
logps/chosen: -237.4   logps/rejected: -238.0
```

直接把 `logps/chosen` 和 `logps/rejected` 比大小是错的，它主要反映长度差。

那 DPO 为什么还能工作？因为 `logits` 里做了**两次相减**：
先 `chosen − rejected`，再减去参考模型的同一个差值。参考模型在同两条序列上算出
同样量级的长度偏差，于是**模型无关的那部分长度效应被抵消掉了**。

但这不彻底 —— 策略自身的长度偏好仍留在里面，这就是文献里
"DPO 偏好更长输出"的来源。

> TRL 0.21.0 的 `DPOConfig` **没有** `average_log_prob` 这个长度归一化开关
> （它只作为 liger kernel 的内部参数被硬编码为 `False`，见 `dpo_trainer.py:362`）。
> 想归一化长度得自己改 loss。

### 参考模型是谁（本项目踩过的坑）

```python
# dpo_trainer.py:908-911
with torch.no_grad(), compte_ref_context_manager:
    if self.ref_model is None:
        with self.null_ref_context():
            ref_model_output = self.concatenated_forward(self.model, batch, is_ref_model=True)

# dpo_trainer.py:890-894
def null_ref_context(self):
    with (self.accelerator.unwrap_model(self.model).disable_adapter()
          if self.is_peft_model and not self.ref_adapter_name
          else nullcontext()):
        ...
```

`disable_adapter()` 禁用**所有** LoRA。所以当 `ref_model=None` + 单个 adapter 时：

**参考模型 = 原始底座，不是 SFT 之后的模型。**

标准 `SFT → DPO` 流水线要求参考 = SFT 模型，否则 KL 项会把模型往原始底座拉，
**部分抵消掉 SFT 的适配**。要满足有两条路：

1. 先把 SFT adapter **merge** 进底座，再基于 merge 后的模型另起一个新 LoRA 做 DPO；
2. 用多 adapter：`DPOConfig` 里确实有 `model_adapter_name` 和 `ref_adapter_name`
   （`dpo_config.py:229, 233`），可以把 SFT adapter 挂成 reference。

我们两条都没用 → 这是 `experiments/2026-09-dpo-from-sft` 里记录的已知方法论缺陷。

### 顺便：`is_trainable` 陷阱

见第 5 节。`PeftModel.from_pretrained` 的 `is_trainable` 默认 `False`，
会让 `Trainer.create_optimizer()` 收集到空参数列表，训练静默失效。

---

## 4. GRPO：advantage 与 loss

### advantage：组内相对（`grpo_trainer.py:1683-1692`）

```python
mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
std_grouped_rewards  = rewards.view(-1, self.num_generations).std(dim=1)
is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
std_grouped_rewards  = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
advantages = rewards - mean_grouped_rewards
if self.scale_rewards:
    advantages = advantages / (std_grouped_rewards + 1e-4)
```

即

```text
A_i = (r_i − mean(r_group)) / (std(r_group) + 1e-4)
```

**这就是 GRPO 相对 PPO 的核心简化：不需要 critic（价值网络），
用同一 prompt 的多次采样的均值当 baseline。**

关键推论：**同一组内奖励全相同 → advantage 全 0 → 该步无梯度。**

注意 `is_std_zero` 只用于日志（`grpo_trainer.py:1732` 记成 `frac_reward_zero_std`），
**并不参与清零** —— advantage 归零是 `rewards - mean = 0` 的自然结果，不是被强制抹掉的。

这正是本项目实测的 `frac_reward_zero_std`：

```text
zero_std 的步数占比 : 81.5%
```

因为 0.5B 在训练题上已经很准（平均 reward 0.8638）、分布很尖（`entropy ≈ 0.23`），
4 次采样经常给出同一个答案。**100 次优化器更新里只有约 18 次真正有效。**

### loss 与 PPO 式裁剪（`grpo_trainer.py:1852-1894`）

```python
log_ratio = per_token_logps - old_per_token_logps
log_importance_weights = log_ratio                     # importance_sampling_level="token"
coef_1 = torch.exp(log_importance_weights)             # 概率比 π_θ/π_old
coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

per_token_loss1 = coef_1 * advantages.unsqueeze(1)
per_token_loss2 = coef_2 * advantages.unsqueeze(1)
per_token_loss = -torch.min(per_token_loss1, per_token_loss2)   # 双侧裁剪
if self.beta != 0.0:
    per_token_loss = per_token_loss + self.beta * per_token_kl   # KL 项可选
```

裁剪形状和 PPO 一致：`-min(ρ·A, clip(ρ)·A)`。

**`beta=0` 时连参考模型前向都省掉了** —— 这是本项目配置的选择（省显存/时间），
代价是没有 KL 约束，策略可以漂移得比较远。

三种 `loss_type` 的区别只在最后怎么归约：

```python
if self.loss_type == "grpo":
    loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
elif self.loss_type == "bnpo":
    loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
elif self.loss_type == "dr_grpo":
    loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
```

- `grpo`：先按序列取平均，再按 batch 平均（默认，对长度不敏感）
- `bnpo`：全局按 token 平均（长序列权重更大）
- `dr_grpo`：除以固定长度而非实际 token 数（Dr. GRPO 论文针对长度偏置的修正）

### reward 函数怎么被调用（`grpo_trainer.py:1293-1322`）

```python
keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
reward_kwargs["trainer_state"] = self.state
...
output_reward_func = reward_func(prompts=prompts, completions=completions,
                                 completion_ids=completion_ids_list, **reward_kwargs)
```

**数据集里除 `prompt`/`completion`/`completion_ids` 外的所有列，都会作为 kwargs
传给 reward 函数。** 这就是 `accuracy_reward(completions, answer, **kwargs)` 能拿到
`answer` 的原因。设计自定义 reward 时利用这一点传元数据。

### prompt 格式：只有对话式才套 chat template（`grpo_trainer.py:1379`）

```python
prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
```

`maybe_apply_chat_template` 内部先判 `is_conversational(example)`
（`data_utils.py:31-69`，要求值是 `[{"role":..., "content":...}]` 形态）。
**传裸字符串不会套模板。**

实测对比（同一句 `Compute 28 * 17.`）：

| prompt 格式 | 生成长度 | 命中 EOS | 输出 |
|---|---|---|---|
| 裸字符串 | 192（截断）| ❌ | 一路写 LaTeX 推理，永不吐答案 |
| chat template | **4** | ✅ | `'476'` |

Instruct 模型看到裸字符串会退化成**文本续写器**。

### 生成配置从哪来（`grpo_trainer.py:833-853`）

```python
generation_kwargs = {
    "max_new_tokens": self.max_completion_length,
    "do_sample": True,
    "temperature": self.temperature,
    "top_p": self.top_p, "top_k": self.top_k, "min_p": self.min_p,
    "repetition_penalty": self.repetition_penalty,
    "cache_implementation": args.cache_implementation,
}
self.generation_config = GenerationConfig(**generation_kwargs)
```

**GRPO 强制 `do_sample=True`** —— 不采样就没有组内多样性，也就没有 advantage。
这些值来自 `GRPOConfig` 而非模型自带的 `generation_config.json`。

### 一个约束

```text
ValueError: generation_batch_size (2) must be divisible by num_generations (4).
```

`generation_batch_size = per_device_train_batch_size × gradient_accumulation_steps`
必须能被 `num_generations` 整除。本项目 `bs=1` 时，`accum` 必须 ≥ `num_generations`。

---

## 5. 本项目踩过的坑（速查）

每条都在对应实验的 `notes.md` 里有完整证据链。

| # | 现象 | 根因 | 出处 |
|---|------|------|------|
| 1 | 训练跑完但权重一字节未变（SHA256 相同） | `PeftModel.from_pretrained` 的 `is_trainable` 默认 `False` → `create_optimizer()` 收到空参数列表；之后参考前向的 `disable_adapter()`/`enable_adapter_layers()` 副作用又把 `requires_grad` 打开，于是**梯度非零但优化器空转** | `2026-09-dpo-from-sft` |
| 2 | GRPO 奖励恒为 0、`loss`/`grad_norm` 都是 0 | TRL 调用 `gradient_checkpointing_enable()` + `enable_input_require_grads()` 后**从不在生成前关掉**。`gradient_checkpointing_disable()` 在 0.21.0 里**从未被调用过**（`modeling_base.py:95-96` 只是把该方法透传给包装类），所以 embedding hook 一直生效并污染生成 | `2026-09-grpo-math` |
| 3 | Instruct 模型输出 LaTeX 推理、永不结束 | prompt 传裸字符串，未套 chat template（`maybe_apply_chat_template` 只处理对话式）；且 `eval_math.py` 套了、`train_grpo.py` 没套 → **训练/评测不一致** | `2026-09-grpo-math` |
| 4 | `import peft` 崩溃 | venv 用 `--system-site-packages` 暴露 conda 的 TensorFlow（NumPy 1.x 编译 vs 当前 2.2.6），`peft → transformers → image_transforms` 的**惰性导入**把它拉起来 | `2026-09-sft-baseline` |
| 5 | 显存指标报 `0.0 GB` | `from_pretrained` 不带 `device_map` 时加载到 CPU；`Trainer` 会自己 `.to(device)` 所以训练脚本没暴露，`generate.py` 暴露了 | `2026-09-sft-baseline` |
| 6 | train/eval 数据重叠 16 条 | 用两个独立 seed 各生成一份，去重后仍有交集；改为**单池洗牌后切分** | `2026-09-grpo-math` |

### 三条方法论

1. **指标只是线索，产物才是证据。**
   坑 1 里 `loss` 在降、`grad_norm` 非零；坑 2 里 `loss`/`grad_norm` 都是 0 却"看起来稳定"。
   只有把产物和输入做**字节级比对**（SHA256 / 逐张量 `max_abs_diff`）才能定论。

2. **`grad_norm` 非零 ≠ 有参数被更新；`loss` 为 0 ≠ 训练失败。**
   GRPO 的 `loss` 实测均值 `3.2e-10` 而 `grad_norm` 到 67.7，adapter 确实变了。
   两个实验合起来说明这两个指标都不能单独当健康判据。

3. **训练和评测必须用同一条输入路径。**
   坑 3 的 `eval` 套了 chat template、`train` 没套，等于两个脚本对同一句话
   的输入根本不同 —— 这种不一致不会报错，只会让你得到无法解释的结果。
