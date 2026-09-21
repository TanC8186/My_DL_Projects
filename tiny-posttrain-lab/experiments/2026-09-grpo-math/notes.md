# Experiment: 2026-09-grpo-math

## 目标

在 RTX 4060 Laptop 8GB 上跑通 **GRPO + 可验证 reward**（算术题），
用 held-out 集做前后对比，验证「有精确 checker 的强化学习」在小模型单卡上能否产生真实提升。

reward 是 `src/my_lab/rewards.py:accuracy_reward`（比对生成文本里最后一个数字与标准答案），
所以指标不是代理量，是真正的对错判定。

## 状态

- [x] 环境安装完成
- [x] 数据准备（生成器 + 严格不重叠的 train/eval 切分）
- [x] 基座 held-out 基线
- [x] 训练跑通
- [x] 记录显存峰值
- [x] 验证 adapter 权重确实被更新
- [x] 训练后 held-out 评测
- [x] 写出结论与局限

## 环境

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB)
- torch 2.5.1+cu124 / transformers 4.55.2 / trl 0.21.0 / peft 0.17.1
- 需 `USE_TF=0`、`USE_FLAX=0`、`HF_HUB_OFFLINE=1`（见 `scripts/env.ps1`）

## 数据

`scripts/make_math_dataset.py` 生成，答案由 Python 计算而非手写：

```powershell
python scripts/make_math_dataset.py --n 300 --eval-n 100 --seed 42 `
    --out data/math_train.jsonl --eval-out data/math_eval.jsonl
```

- 题型：两位数加减、1×2 / 2×2 乘法、整除、四类应用题
- **train 200 / eval 100，从同一池子确定性切分，重叠数为 0**
  （早期版本用两个独立 seed 生成，出现 16 条重叠 —— 那等于在训练集上评测）

## 配置摘要

- config: `configs/grpo_qwen05b_8g.yaml`
- 底座: `Qwen2.5-0.5B-Instruct`（**从原始 instruct 起步，不接 SFT/DPO adapter**，
  以隔离 GRPO 自身的效果；否则无法区分"GRPO 有用"和"起点本来就好"）
- LoRA r=8, alpha=16
- num_generations=4, temperature=1.0, beta=0.0（无 KL / 无参考模型）
- bs=1, accum=4, lr=1e-5, 2 epochs
- 400 micro-step = **100 次优化器更新**

## 显存

- 训练 peak_allocated / peak_reserved: **1.22 / 1.49 GB**
- 评测（推理）peak_allocated / peak_reserved: **0.94 / 1.00 GB**
- 是否 OOM: 否

**注意：训练显存这么低是因为关掉了 gradient checkpointing —— 而那是必须关的，见下。**

## 结果

| | 基座 | GRPO 后 |
|---|---|---|
| held-out 准确率（greedy, n=100） | **87/100 = 0.8700** | **88/100 = 0.8800** |
| 训练耗时 | — | 285.86 s（400 micro-step）|
| 训练集平均 reward | — | 0.8638（末 20 步 0.8125）|

### 失败集合对比（比总分更有信息量）

| | 数量 | 题目（gold 值） |
|---|---|---|
| GRPO 修好 | 3 | 65, 369, 73 |
| GRPO 弄坏 | 2 | 308, 48 |
| **仍然失败** | **10** | 280, 513, 340, 121, 260, 129, 624, 27, 69, 229 |

**净 +1 题是 churn（此消彼长），不是系统性提升。** 原 13 个失败里 10 个完全没变，
说明模型的核心弱点（两位数乘法）未被触动。

统计上也不显著：n=100、p≈0.87 时标准误 ≈ 3.4%，+1% 远在噪声内。
**不能宣称"GRPO 提升了准确率"。**

---

## 重大发现：gradient checkpointing 会破坏 GRPO 的生成

第一次运行看起来完全正常：400 步跑完、没有报错。但指标是：

```text
'completions/mean_length': 192.0        ← 全部撞上 max_completion_length
'completions/clipped_ratio': 1.0        ← 100% 被截断
'completions/mean_terminated_length': 0.0  ← 没有任何一条自然结束
'rewards/accuracy_reward/mean': 0.0     ← 奖励恒为 0
'frac_reward_zero_std': 1.0             ← 组内无方差
'loss': 0.0, 'grad_norm': 0.0           ← 完全没有学习信号
```

### 排查过程（逐步排除，最后二分定位）

| 假设 | 实测 | 结论 |
|------|------|------|
| prompt 没套 chat template | 拦截 `generate` 打印实际输入：`<\|im_start\|>system...<\|im_start\|>assistant\n`，46 token | 模板**正确**，排除 |
| 生成参数（top_k / min_p / repetition_penalty）| 用 GRPOConfig 的确切默认值手动生成 → 4 token + EOS | 参数**没问题**，排除 |
| 模型状态 | 见下表 | **命中** |

状态二分测试（同一 prompt，只改模型状态）：

| 模型状态 | 生成长度 | 命中 EOS | 输出 |
|----------|---------|---------|------|
| base / eval | 4 | ✅ | `'476'` |
| + LoRA / eval | 4 | ✅ | `'476'` |
| + LoRA / train | 4 | ✅ | `'476'` |
| **+ LoRA / train / gradient_checkpointing** | **192** | ❌ | `'40% percentage form syntax)...'` |

### 根因

TRL 的 `GRPOTrainer` 在 983/986 行调用 `gradient_checkpointing_enable()`，
并通过 `_prepare_gradient_checkpointing()` 调用 `enable_input_require_grads()`
（在输入 embedding 上注册一个 `output.requires_grad_(True)` 的 forward hook），
**但从不在这之后、生成之前把它们关掉**（全文件没有一处 `gradient_checkpointing_disable`）。

结果：生成阶段每个 token 都走被 hook 污染的 forward，输出变成乱码，
永远不吐 EOS，奖励恒为 0，GRPO 拿到 0 信号。

### 修复

`gradient_checkpointing: false`。这里完全不需要它 —— 0.5B + LoRA 训练峰值只有 1.22 GB。

副作用是**生成速度提升约 17 倍**：同样的 8 条样本，修复前 64 s，修复后 3.7 s
（原本因为没有 KV cache，生成是 O(n²)）。

修复后同一批数据：

| 指标 | 修复前 | 修复后 |
|---|---|---|
| `completions/mean_length` | 192 | **4** |
| `clipped_ratio` | 1.0 | **0.0** |
| `rewards/accuracy_reward/mean` | 0.0 | **1.0** |
| `entropy` | 5.38 | **0.23** |

---

## 第二个发现：prompt 格式决定 Instruct 模型的行为

修复 gradient checkpointing 之前，还先修了一个更基础的问题。
`train_grpo.py` 原本把 prompt 当**裸字符串**传给 TRL，而 TRL 只在
`is_conversational()` 为真时才套 chat template（`maybe_apply_chat_template`）。

同一句 `Compute 28 * 17.` 的对比：

| prompt 格式 | 生成长度 | 命中 EOS | 输出 |
|---|---|---|---|
| 裸字符串 | 192（截断） | ❌ | `" Let's break this down step by step:\n\n\\( 28 \\times 17 = ..."` 一路写 LaTeX，永不吐答案 |
| chat template | **4** | ✅ | `'476'` |

Instruct 模型看到裸字符串会退化成**文本续写器**，而不是回答者。

同时暴露了一个**训练/评测不一致**：`eval_math.py` 一直套了 chat template，
`train_grpo.py` 没有 —— 两个脚本对同一句话的输入完全不同。

修复后数据集改为对话格式：

```python
{"prompt": [{"role": "user", "content": r["prompt"]}], "answer": r["answer"]}
```

## 第三个发现：GRPO 日志里的 `loss` 不可信

```text
{'loss': 0.0, 'grad_norm': 19.407, ...}     ← 同一行自相矛盾
train_loss: 3.166496753692627e-10           ← 几乎精确为零
```

`loss` 恒为 0（或 1e-10），但 `grad_norm` 在 18.5% 的步上非零，最高 67.7。

判断 GRPO 是否真的在训练，要看：

1. `grad_norm` 是否非零
2. `frac_reward_zero_std` 是否 < 1.0
3. **最终权重是否变化**（最可靠）

本次用第三项验证：GRPO 从零初始化 LoRA 起步，`lora_B` 初始恒为 0，
训练后 `max|w| = 6.31e-04` → 确实更新了。

（对照 DPO 实验：那里的教训是"`grad_norm` 非零 ≠ 有参数被更新"。
两个实验合起来说明 —— **指标只是线索，产物才是证据**。）

## 关于「组内方差」：GRPO 的信号来源

- `frac_reward_zero_std = 1.0` 的步数占 **81.5%**
- 即：4 条生成拿到完全相同的奖励 → advantage 全为 0 → 该步无梯度

原因：0.5B 在训练题上已经很准（平均 reward 0.8638），且分布很尖
（`entropy ≈ 0.23`），4 次采样常常给出同一个答案。
**GRPO 只能从"组内有对有错"的样本里学习**，本次只有约 18.5% 的步满足条件。

这解释了为什么 100 次优化器更新里实际有效的不到 20 次。

## 结论与局限

- **GRPO + 可验证 reward 的链路在 8GB 单卡上跑通了**，显存 1.22 GB，400 micro-step 约 286 s。
- **本次没有产生可测量的提升**：held-out 88% vs 基座 87%，净 +1 题；
  失败集合显示 10/13 原错误未变，另有 2 题被弄坏。这是 churn，不是学习。
- 主要原因有二：
  1. 任务对模型太简单（基座已 87%），81.5% 的步没有组内方差 → 有效更新不到 20 次；
  2. lr=1e-5 下 100 步、`lora_B` 只移动到 6.3e-4，改动幅度本身很小。
- 局限：仅 0.5B、仅 200 条训练题、单次运行、单一 seed。
  不能据此说"GRPO 无效"，只能说**这个规模与任务难度不足以产生效果**。

## 下一步（若要继续）

1. **换更难的任务**：加大乘法位数（3×3）、多步应用题，把基座准确率压到 40–60%，
   这样组内方差才会充足。
2. **扩大 num_generations**（4 → 8）以提高混合组比例。
3. **加 `beta > 0`** 引入 KL 约束，观察是否抑制了"修好 3 题又弄坏 2 题"的漂移。
4. 用 `--seed` 多次评测，把 ±3.4% 的噪声量化出来，再谈差异是否真实。
