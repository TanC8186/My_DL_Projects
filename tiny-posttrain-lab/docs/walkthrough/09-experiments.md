# 09 · 实验记录与日志阅读

这一篇是收尾，讲**怎么把一个实验变成可复现、可审查、可面试讲述的记录**。

前面八篇讲"代码怎么写"，这一篇讲"**怎么知道代码真的在工作**"。

---

## 0. 实验文件夹的结构

```text
experiments/2026-09-sft-baseline/
├── notes.md                  ← 人写的：目标 / 配置 / 显存 / 结果 / 结论 / 局限
├── train_log.txt             ← 程序输出的：逐步日志（含 loss / grad_norm / lr）
├── generation_compare.md     ← 程序输出的：base vs LoRA 的生成对比
└── output/                   ← 产物（gitignore，只保留 train_metrics.txt）
    ├── final/                ← adapter 权重 + tokenizer
    └── train_metrics.txt     ← 最终指标 + 显存（唯一被提交的产物文件）
```

### 三类文件的分工

| 类型 | 谁写的 | 可信度 | 进版本库？ |
|------|--------|--------|-----------|
| **原始日志**（`train_log.txt`） | 程序 | 高（没经过人手） | ✅ |
| **程序生成的报告**（`generation_compare.md`） | 程序 | 高 | ✅ |
| **指标文件**（`train_metrics.txt`） | 程序 | 高 | ✅ |
| **人写的记录**（`notes.md`） | 人 | 中（可能抄错、可能选择性汇报） | ✅ |
| **模型权重** | 程序 | 高 | ❌（可再生产） |

**关键设计：原始数据和解读分开存。**

```text
train_metrics.txt（原始）  →  {'train_loss': 1.3924129104614258, ...}
                             peak_allocated=1.87 GB, peak_reserved=4.89 GB

notes.md（解读）           →  "train_loss 1.39，可复现性验证：两次运行差 0.03%"
```

**如果两者矛盾，以原始文件为准。** 人可能抄错，程序不会。

> **这就是为什么 0.2 KB 的 `train_metrics.txt` 值得从 gitignore 里豁免出来**
> （见 [01 篇](./01-environment.md)）。

### 命名约定

```text
2026-09-sft-baseline        ← 年月 + 方法 + 定位
2026-09-dpo-from-sft        ← 年月 + 方法 + 起点（明确是续训）
2026-09-grpo-math           ← 年月 + 方法 + 任务类型
```

**日期在前** —— 按文件名排序就是时间顺序。

**后缀说明"这次实验和默认有什么不同"** ——
`dpo-from-sft` 一眼看出是从 SFT adapter 续训的；
`grpo-math` 一眼看出是数学任务。

**避免的名字**：`test`、`exp1`、`final`、`new`、`v2`。
半年后你完全想不起来 `exp3` 是什么。

---

## 1. 怎么读训练日志

### 1.1 日志的两类行

**`train_sft.py` 输出的日志包含两类内容：**

```text
[info] experiment: 2026-09-sft-baseline            ← 脚本自己 print 的（带 [info] 前缀）
[info] model: E:/My_DL_Projects/qwen/...
[info] dataset rows: 2000
[info] columns: ['instruction', 'input', 'output']
  0%|          | 0/250 [00:00<?, ?it/s]           ← tqdm 进度条
  2%|��         | 5/250 [00:07<06:04,  1.49s/it]
{'loss': 1.6729, 'grad_norm': 1.26, ...}           ← HuggingFace Trainer 的指标字典
[info] peak VRAM: peak_allocated=1.87 GB, ...      ← 脚本自己 print 的
[done] metrics -> .../train_metrics.txt
```

**`[info]` / `[warn]` / `[done]` / `[hint]` / `[miss]` 前缀是自己加的**，
方便 `Select-String` 过滤：

```powershell
Get-Content train_log.txt | Select-String -Pattern "^\["
```

### 1.2 SFT 的指标字段

```python
{'loss': 1.6729, 'grad_norm': 1.2608592510223389, 'learning_rate': 0.0001,
 'num_tokens': 6044.0, 'mean_token_accuracy': 0.5884182788431644, 'epoch': 0.02}
```

| 字段 | 含义 | 怎么用 |
|------|------|--------|
| `loss` | 本窗口的平均损失 | **看趋势**，不看绝对值 |
| `grad_norm` | 梯度联合范数（裁剪前） | 非零 = 有梯度；**持续等于 `max_grad_norm` = 一直在被裁** |
| `learning_rate` | 当前 lr | 验证 warmup 和 scheduler 生效 |
| `num_tokens` | 累计处理的 token 数 | 用于算吞吐 |
| `mean_token_accuracy` | **token 级准确率** | SFT 特有。0.59 表示约 59% 的 token 预测正确 |
| `epoch` | 当前 epoch 进度 | 判断有没有跑完 |

#### `mean_token_accuracy` 是 SFT 独有的好指标

**注意它和 `loss` 的关系**：

```text
loss = 1.6729  →  exp(-1.6729) ≈ 0.188  ← 平均每 token 的正确概率
mean_token_accuracy = 0.588             ← 实际 argmax 命中的比例
```

**两者不矛盾**：`loss` 是概率的负对数（软指标），
`mean_token_accuracy` 是 argmax 是否命中（硬指标）。

**从 0.588 涨到 0.66 说明模型确实在学**（实测数据），
这比只看 loss 更直观。

#### `grad_norm` 的读法

实测值：`1.26, 1.32, 0.88, 1.14, 1.08, 0.93, 0.74, 0.66, 0.77, 1.16, 0.75, 0.73`

**大部分在 1.0 以下，偶尔到 1.3。** 而 `max_grad_norm = 1.0`。

**怎么解读？**

```text
grad_norm < max_grad_norm  →  没有触发裁剪，梯度是"自然"的
grad_norm ≈ max_grad_norm  →  触发了裁剪，被截到 1.0
grad_norm >> max_grad_norm →  频繁被裁，可能 lr 太大或数据有异常样本
```

**本项目的情况**：实测最大 1.32，说明**偶尔轻微裁剪**，属于正常。

> ⚠️ **`grad_norm` 非零不等于训练有效** —— DPO 实验里
> `grad_norm` 达到 16.9，但权重一个字节没变（优化器是空的）。
> 详见 [05 篇](./05-train-dpo.md)。

### 1.3 DPO 的指标字段

```python
{'loss': 0.826, 'grad_norm': 12.684, 'learning_rate': 4.84e-06,
 'rewards/chosen': 0.817, 'rewards/rejected': 0.949,
 'rewards/accuracies': 0.575, 'rewards/margins': -0.132,
 'logps/chosen': -425.7, 'logps/rejected': -321.1,
 'logits/chosen': -1.358, 'logits/rejected': -1.285, 'epoch': 0.04}
```

| 字段 | 含义 | 怎么用 |
|------|------|--------|
| `rewards/chosen` | `β·(logps_chosen − ref_logps_chosen)` | **隐式奖励**。看它是否上升 |
| `rewards/rejected` | 同上，rejected 的 | 看两者差距 |
| **`rewards/accuracies`** | **`chosen_rewards > rejected_rewards` 的比例** | **最重要的指标**，随机线 0.5 |
| `rewards/margins` | `chosen_rewards − rejected_rewards` | 应该为正；为负说明模型偏好反了 |
| `logps/chosen` | chosen 的 log 概率**之和** | ⚠️ **长度有偏，不能直接比大小** |
| `logits/chosen` | **平均** token logit | 比 `logps` 更可比（已经是均值） |

#### ⚠️ `logps` 的长度陷阱（实测）

```text
logps/chosen: -425.7   logps/rejected: -321.1    ← rejected 看起来"概率更高"？
logps/chosen: -237.4   logps/rejected: -238.0    ← 这里才对得上
```

**`logps` 是 `per_token_logps[:, 1:].sum(-1)` —— 按 token 求和。**

**序列越长，和越负。** 所以第一行的 `-425.7 vs -321.1` 主要反映
**chosen 比 rejected 长**，不是"rejected 更好"。

**要看偏好方向，应该看：**

1. `rewards/accuracies`（二元的，无长度问题）
2. `rewards/margins`（做了两次相减，长度效应大部分抵消）
3. `logits/chosen` vs `logits/rejected`（已经是**平均** logit）

#### 实测的 DPO 指标解读

```text
accuracy  overall mean : 0.4550      ← 低于随机线 0.5
accuracy  first half   : 0.4687
accuracy  last half    : 0.4375      ← 越训越差
margins   first half   : -0.0910     ← 负的
margins   last half    : -0.0771
```

**三条信息合起来说明"没有学到偏好"：**

1. accuracy < 0.5（比瞎猜还差）
2. 后半程比前半程更差（不是在收敛，是在漂移）
3. margins 为负（chosen 的奖励确实低于 rejected）

**如果只看 `loss: 0.826`，你完全看不出问题** —— 这就是为什么
**要看任务相关的指标，而不是只看 loss。**

### 1.4 GRPO 的指标字段

```python
{'loss': 0.0, 'grad_norm': 0.0, 'learning_rate': 1e-05,
 'num_tokens': 236.0,
 'completions/mean_length': 5.0, 'completions/min_length': 5.0,
 'completions/max_length': 5.0, 'completions/clipped_ratio': 0.0,
 'completions/mean_terminated_length': 5.0,
 'rewards/accuracy_reward/mean': 0.0, 'rewards/accuracy_reward/std': 0.0,
 'reward': 0.0, 'reward_std': 0.0,
 'frac_reward_zero_std': 1.0,
 'entropy': 0.23096361756324768,
 'clip_ratio/low_mean': 0.0, 'clip_ratio/high_mean': 0.0, 'clip_ratio/region_mean': 0.0,
 'epoch': 0.88}
```

**字段最多，也最容易看错。逐个解释：**

#### 生成质量类

| 字段 | 含义 | 健康值 |
|------|------|--------|
| `completions/mean_length` | 平均生成长度 | **应该远小于 `max_completion_length`** |
| `completions/clipped_ratio` | **被截断的比例** | **应该接近 0** |
| `completions/mean_terminated_length` | 自然结束（命中 EOS）的平均长度 | **应该 > 0** |

**⚠️ 这三个字段是判断"生成是否正常"的第一道关。**

**实测对比（同一份配置，修复前 vs 修复后）：**

| 字段 | 修复前（bug） | 修复后 |
|------|--------------|--------|
| `mean_length` | **192.0**（= max_completion_length） | **4.0** |
| `clipped_ratio` | **1.0**（100% 被截断） | **0.0** |
| `mean_terminated_length` | **0.0**（没有一条自然结束） | **4.0** |

**读法：**

```text
如果 mean_length ≈ max_completion_length 且 clipped_ratio ≈ 1.0
  → 模型在"乱写"，从来没说到答案
  → 检查 prompt 格式（chat template？）和 max_completion_length

如果 mean_terminated_length == 0
  → 没有任何一条生成命中 EOS
  → 这是强烈的不正常信号
```

#### 奖励类

| 字段 | 含义 |
|------|------|
| `rewards/<name>/mean` | 该 reward 函数的平均分 |
| `rewards/<name>/std` | 该 reward 函数的标准差 |
| `reward` | 所有 reward 函数的加权总分 |
| `reward_std` | **组内**标准差（不是全局的） |
| **`frac_reward_zero_std`** | **组内标准差为 0 的样本比例** ← **最关键的字段** |

**`frac_reward_zero_std` 为什么最重要？**

因为 GRPO 的 advantage 是组内相对的：

```text
A_i = (r_i − mean(r_group)) / (std(r_group) + 1e-4)
```

**组内标准差为 0 → 所有 `A_i = 0` → 该步完全没有梯度。**

**实测：`frac_reward_zero_std = 1.0` 的步数占 81.5%。**

```text
含义：81.5% 的步是白跑的
原因：模型在训练题上已经 87% 正确，分布很尖（entropy ≈ 0.23），
      4 次采样经常给出同一个答案
```

**这是一个"训练有效效率"的直接度量。**

#### 其他

| 字段 | 含义 | 读法 |
|------|------|------|
| `entropy` | 生成分布的熵 | **低 = 模型自信**。实测 0.23（健康）vs 5.38（乱码时） |
| `clip_ratio/*` | PPO 式裁剪的触发比例 | 全 0 表示没触发裁剪 |
| `num_tokens` | 累计 token 数 | 算吞吐用 |

**`entropy` 是一个被低估的诊断指标。**

```text
entropy ≈ 0.23  →  分布很尖，模型确信答案（健康）
entropy ≈ 5.38  →  分布很平，模型在乱猜（对应那个 bug）
```

**在 GRPO 的 bug 排查中，`entropy` 从 5.38 降到 0.23 是"修复生效"的明确证据。**

---

## 2. ⚠️ 核心：怎么判断训练是不是真的在工作

**这是整个项目最有价值的一节。** 三个实验里有两个遇到了"看起来正常但完全没训练"的情况。

### 2.1 五个可靠性等级

| 等级 | 证据 | 可靠性 | 说明 |
|------|------|--------|------|
| 1 | `loss` 在下降 | ⚠️⚠️ **很低** | DPO 那次 loss 一直在动，权重没变 |
| 2 | `grad_norm` 非零 | ⚠️ **低** | DPO 那次 grad_norm 到 16.9，但优化器是空的 |
| 3 | 框架无报错 | ⚠️ **很低** | `AdamW([])` 和空 `step()` 都是合法的 |
| 4 | 训练配置相关的指标 | ✅ **中** | `frac_reward_zero_std`、`clipped_ratio`、`entropy` |
| 5 | **产物和输入做字节比对** | ✅✅✅ **高** | **唯一不会骗人的证据** |

### 2.2 等级 5 的三种具体做法

#### 做法 A：SHA256 比对（适用于"续训"场景）

```powershell
$sft = (Get-FileHash .../sft/output/final/adapter_model.safetensors).Hash
$dpo = (Get-FileHash .../dpo/output/final/adapter_model.safetensors).Hash
if ($sft -eq $dpo) { "没变！" } else { "变了" }
```

**适用**：从一个已有 adapter 继续训练（DPO / adapter 续训的 GRPO）。

**实测抓到了 DPO 的 bug**：

```text
SFT: 31D619050DC286DF77D463709D900843EA8DF372CF6DEFBC4FB7E93FEF79D257
DPO: 31D619050DC286DF77D463709D900843EA8DF372CF6DEFBC4FB7E93FEF79D257
→ IDENTICAL
```

#### 做法 B：逐张量差异（比 A 更有信息量）

```python
from safetensors.torch import load_file

a = load_file(".../sft/output/final/adapter_model.safetensors")
b = load_file(".../dpo/output/final/adapter_model.safetensors")
diffs = {k: (a[k].float() - b[k].float()).abs().max().item() for k in a}
print(f"tensors_changed = {sum(1 for v in diffs.values() if v > 0)}/{len(a)}")
print(f"max_abs_diff    = {max(diffs.values()):.6e}")
print(f"mean_abs_diff   = {sum(diffs.values())/len(diffs):.6e}")
```

**实测输出：**

```text
# 修复前
tensors_changed = 0/192
max_abs_diff    = 0.000000e+00

# 修复后
tensors_changed = 192/192
max_abs_diff    = 2.040372e-04
mean_abs_diff   = 1.585718e-04
```

**优势**：不只回答"变没变"，还回答"**变了多少**"。

**`max_abs_diff = 2.04e-04` 这个数字后来成了关键证据** ——
它说明"训练量太小"，而不是"方法无效"（见 [05 篇](./05-train-dpo.md)）。

#### 做法 C：检查零初始化参数（适用于"从零起 adapter"）

**LoRA 的 `lora_B` 初始化为全 0**（这是 LoRA 的设计：初始时 `ΔW = B@A = 0`，
即 adapter 是恒等变换）。

**所以：如果训练后 `lora_B` 还是全 0，说明什么都没学。**

```python
sd = load_file(".../grpo/output/final/adapter_model.safetensors")
b_keys = [k for k in sd if "lora_B" in k]
b_allzero = all(bool((sd[k].float() == 0).all()) for k in b_keys)
b_absmax = max(sd[k].float().abs().max().item() for k in b_keys)
print(f"lora_B all-zero? {b_allzero}   lora_B max|w| = {b_absmax:.6e}")
```

**实测输出：**

```text
total tensors=192  lora_A=96  lora_B=96
lora_B all-zero? False   lora_B max|w| = 6.313614e-04
VERDICT: TRAINED (B moved away from 0)
```

**这个方法适用于 GRPO**（从原始权重起步，没有"输入 adapter"可比对）。

### 2.3 一张检查清单

**每次训练跑完后，按顺序检查：**

```text
□ 1. 有没有报错？（最低标准，但不够）
     ↓
□ 2. loss 趋势是否合理？（不是"是否下降"，而是"是否合理"）
     - SFT：应从 ~1.67 降到 ~1.39
     - DPO：会在 0.7~1.0 波动，不一定下降
     - GRPO：可能恒为 0（被零方差步稀释），这不是错误
     ↓
□ 3. 任务相关指标是否合理？
     - DPO：rewards/accuracies 是否 > 0.5？
     - GRPO：frac_reward_zero_std 是否 < 1.0？
     - GRPO：clipped_ratio 是否接近 0？
     ↓
□ 4. 显存是否正常？（异常低也是信号！）
     - 0.0 GB → 模型没上 GPU（实测遇到）
     - 接近满 → 可能 OOM 边缘
     ↓
□ 5. 【最关键】产物是否真的变了？
     - 续训场景 → SHA256 / 逐张量比对
     - 从零起 adapter → 检查 lora_B 是否非零
     ↓
□ 6. held-out 评测是否有提升？（用 eval 脚本，不是看训练指标）
```

**第 5 步是"承重墙"** —— 前面四步都可能骗你，只有它不会。

### 2.4 三个实验的检查结果对照

| 检查项 | SFT | DPO（第一次） | DPO（修复后） | GRPO（修复前） | GRPO（修复后） |
|--------|-----|--------------|--------------|---------------|---------------|
| 无报错 | ✅ | ✅ | ✅ | ✅ | ✅ |
| loss 合理 | ✅ 1.67→1.39 | ⚠️ 在动 | ⚠️ 在动 | ⚠️ 恒为 0 | ⚠️ 恒为 0 |
| grad_norm | ✅ 0.66~1.32 | ⚠️ 12~17 | ✅ 9~20 | ❌ **恒为 0** | ✅ 0~67 |
| 任务指标 | — | ❌ acc 0.455 | ❌ acc 0.455 | ❌ **reward 全 0** | ⚠️ 0.86 |
| 显存 | ✅ 1.87 GB | ✅ 2.76 GB | ✅ 2.76 GB | ⚠️ 1.41 GB | ✅ 1.22 GB |
| **产物变化** | — | ❌ **0/192** | ✅ **192/192** | — | ✅ **B 非零** |
| held-out | — | — | 无提升 | — | +1 题（噪声） |

**注意 DPO 第一次和修复后的对比**：

```text
             第一次        修复后
loss         在动     →    在动        ← 一样！
grad_norm    12~17    →    9~20       ← 一样！
显存         2.76 GB  →    2.76 GB    ← 一样！
产物变化     0/192    →    192/192    ← 【唯一的区别】
```

**只有第 5 步能区分这两次运行。** 这就是为什么它最关键。

---

## 3. 怎么写 `notes.md`

### 3.1 模板

```markdown
# Experiment: <experiment_id>

## 目标
<这次实验要回答什么问题？一句话>

## 状态
- [x] 环境安装完成
- [x] 数据确认
- [x] 训练跑通
- [x] 记录显存峰值
- [x] 验证产物确实被更新        ← 【关键】这一条不能省
- [x] 写出结论与局限

## 环境
- GPU / torch / transformers / trl / peft 版本
- 需要的环境变量

## 配置摘要
- config 路径、模型、数据、seq、batch/accum、lora、关键超参

## 显存
- peak_allocated / peak_reserved
- 是否 OOM
- 【如果有对比】和别的实验比，并说明可比性

## 结果
- 表格：关键指标 + 值
- 【关键】不只看 loss，看任务相关指标

## 问题记录
<遇到的坑，按编号列出>

## 结论与局限
- 这次实验证明了什么
- **不能证明什么**（局限）
- 下一步

## 下一步（若要继续）
<具体的、可执行的建议>
```

### 3.2 三个写作原则

#### 原则 1：**只报事实，不报期望**

```markdown
# ❌ 差
"DPO 训练成功，模型学会了偏好。"

# ✅ 好
"rewards/accuracies 全程均值 0.455（随机线 0.5），
 后半程（0.4375）比前半程（0.4687）更差。
 本次 DPO 没有产生可测量的偏好提升。"
```

**"成功"是一个判断，"0.455"是一个事实。** 事实可以被质疑和检验，判断不能。

#### 原则 2：**必须写"局限"**

三个实验的 `notes.md` 都有专门的"局限"一节：

```markdown
# SFT
- 局限：仅 0.5B、仅 2000 条 alpaca、单次采样、无 held-out 指标，
  结论不能外推到 1.5B/3B 或工业规模训练。

# DPO
- 局限：仅 0.5B、仅 1000 条 UltraFeedback、单次运行、未做 held-out 偏好评测。
  不能据此断言"DPO 无效"，只能说【这个配置规模不足以产生效果】。

# GRPO
- 局限：仅 0.5B、仅 200 条训练题、单次运行、单一 seed。
  不能据此说"GRPO 无效"，只能说【这个规模与任务难度不足以产生效果】。
```

**注意最后两句的措辞**：

```text
❌ "DPO 无效"
✅ "这个配置规模不足以产生效果"
```

**区分"方法不行"和"我没跑够"** —— 这是一个诚实的实验者必须做的区分。

**为什么局限这么重要？**

因为**没有局限的结论是危险的**。如果只写"DPO 准确率 0.455"，
读者可能得出"DPO 这个方法不好"的错误结论。
写上"参考模型错位 + 只有 15 次优化器更新"，
读者才知道这个数字不能和论文对比。

#### 原则 3：**记录"负面发现"，它们是资产**

**本项目最有价值的内容全是"负面"的：**

| 发现 | 价值 |
|------|------|
| SFT 让事实问答退化 | 说明 loss 下降 ≠ 效果变好 |
| DPO 静默失效（优化器为空） | 一个能骗过所有常规指标的 bug |
| GRPO 的 gradient checkpointing 破坏生成 | TRL 的真实缺陷 + 生成慢 17 倍 |
| `extract_final_number` 会读错答案 | reward misspecification 的实例 |
| train/eval 数据重叠 16 条 | 数据泄漏 |
| 跨实验比显存没固定步数 | 会得出相反结论 |

**这些比"我跑了 SFT 效果不错"有价值得多**，因为：

1. **可验证** —— 每条都有实测证据
2. **可迁移** —— 别人也会踩同样的坑
3. **展示能力** —— 能发现问题比能跑通脚本难得多

### 3.3 实测：`notes.md` 的"问题记录"一节

**SFT 的 `notes.md` 里有 6 条问题记录，摘录两条：**

```markdown
5. `scripts/generate.py` 初版把模型留在了 CPU：`from_pretrained` 不带 `device_map`
   时默认加载到 CPU；训练脚本没暴露这个问题，因为 `Trainer` 内部会 `model.to(device)`。
   修复：显式 `model.to("cuda")`。
   - 症状是 `torch.cuda.max_memory_allocated()` 报 **0.0 GB** ——
     显存指标本身成了 bug 探测器。

6. `.venv` 用 `--system-site-packages` 会暴露 conda `dl_env` 的 TensorFlow
   （用 NumPy 1.x 编译，当前 NumPy 2.2.6），而 `peft → transformers → image_transforms`
   的**惰性导入链**会把 TF 拉起来并崩溃。需 `USE_TF=0`、`USE_FLAX=0`。
```

**每条的结构：现象 → 根因 → 修复 → （可选）附带洞察。**

**第 5 条最后那句"显存指标本身成了 bug 探测器"是额外的收获** ——
它把一个具体 bug 提升成了一条通用经验。

---

## 4. 怎么读评测报告

### 4.1 `generation_compare.md`（定性）

**结构：**

```markdown
# Generation report

- model: `...`
- max_new_tokens: 128
- decoding: sample(t=0.7, top_p=0.9)
- seed: 42
- adapter: `...`
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

**读的时候看三件事：**

| 看什么 | 本次观察 |
|--------|---------|
| **风格有没有变** | ✅ base 说 "Certainly!"，SFT 后直接上编号列表 |
| **有没有变差** | ✅ Prompt 4 太阳系：base 8 个全对，SFT 后 7 项且 `Earth (Mars)` 错位 |
| **显存差异** | 0.94 → 0.95 GB（只多 10 MB） |

**关键：报告里同时保留了 base 和 adapter 的输出。**
如果只存 adapter 的输出，你无法判断"这个回答是变好了还是本来就长这样"。

### 4.2 `eval_*.md`（定量）

**结构：**

```markdown
# Math evaluation

- tag: `base`
- model: `...`
- adapter: none
- data: `data/math_eval.jsonl` (100 rows)
- decoding: greedy, max_new_tokens=256, seed=42
- VRAM: peak_allocated=0.94 GB, peak_reserved=0.99 GB

**Accuracy: 87/100 = 0.8700**

## Failures (13)

### gold=280 pred=270.0

> Compute 14 * 20. End your response with the number only.

```text
270
```
...
```

**读的时候看四件事：**

| 看什么 | 为什么 |
|--------|--------|
| **header 里的配置** | 确认 decoding 是 greedy（可复现）、数据是 held-out |
| **准确率** | 和基线对比 |
| **失败案例的 `gold` vs `pred`** | **区分"模型答错"和"reward 读错"** |
| **失败案例的 prompt 类型** | 定位模型的弱点在哪一类题 |

**第 3 点最关键**：

```text
gold=63 pred=83.0  →  模型答错了（真的算成 83）
gold=81 pred=1.0   →  reward 读错了（输出是 LaTeX 推理，被截断）
```

**只看"87% 正确"你无法区分这两种情况。**

### 4.3 失败集合比对（最有信息量的分析）

**两个评测报告的准确率对比信息量很小。** 真正有用的是**失败集合的 diff**：

```powershell
function Get-Fails($p) {
    (Get-Content $p | Select-String -Pattern "^### gold=(\d+) pred=(\S+)") |
        ForEach-Object { $_.Matches[0].Groups[1].Value }
}
$b = Get-Fails experiments/2026-09-grpo-math/eval_base.md
$a = Get-Fails experiments/2026-09-grpo-math/eval_after.md
"fixed by GRPO : $(($b | Where-Object { $a -notcontains $_ }) -join ', ')"
"broken by GRPO: $(($a | Where-Object { $b -notcontains $_ }) -join ', ')"
"still failing : $(($a | Where-Object { $b -contains $_ }) -join ', ')"
```

**输出：**

```text
fixed by GRPO : 65, 369, 73
broken by GRPO: 308, 48
still failing : 280, 513, 340, 121, 260, 129, 624, 27, 69, 229
```

**这个分析揭示了"净 +1 是 churn"**：

```text
如果只是"修好 1 题，没弄坏任何题"  →  真提升
实际是"修好 3 题，弄坏 2 题，10 题没动"  →  换了几个答案而已
```

**准确率 +1% 可以对应完全不同的两种实质。**

---

## 5. 三个实验的完整对照

| | SFT | DPO | GRPO |
|---|---|---|---|
| **目标** | 跑通 SFT 基线 | 续 SFT adapter 做偏好对齐 | 可验证 reward 的 RL |
| **数据** | alpaca-cleaned 2000 条 | UltraFeedback 1000 条 | 生成的算术题 200 条 |
| **可训练参数** | 2,162,688 (r=16) | 2,162,688 (r=16) | 494,032 → LoRA r=8 |
| **优化器更新次数** | 250 | **约 15** | **100** |
| **耗时** | 345 s | 336 s | 286 s |
| **显存 (allocated)** | 1.87 GB | 2.76 GB | 1.22 GB |
| **显存 (reserved)** | 4.89 GB | 3.56 GB | 1.49 GB |
| **核心指标** | train_loss 1.39 | accuracy 0.455 | held-out 88/100 |
| **结论** | 风格改了，能力退化 | **无提升** | **+1 题，噪声** |
| **副产品** | 发现设备放置 bug | **发现静默失效 bug** | **发现 2 个 bug** |

### 三条共同的教训

**教训 1：训练量比方法更重要**

```text
DPO:  15 次优化器更新  →  权重只动了 2.04e-04  →  不可能有效果
GRPO: 100 次更新，但 81.5% 无梯度  →  有效约 18 次  →  不可能有效果
```

**两个"失败"的实验，主要原因都是规模不足，不是方法不对。**

**教训 2：指标会骗人，产物不会**

```text
DPO:  loss 在降 + grad_norm 16.9  →  但优化器是空的
GRPO: loss = 0 + grad_norm = 0     →  但训练确实发生了（lora_B 非零）
```

**两个方向的误判都出现了。**

**教训 3：诚实记录负面结果的价值**

三个实验都没有"效果变好"，但**项目因此更有说服力**：

```text
如果写"我跑了 SFT/DPO/GRPO，效果都不错"
  → 面试官追问"具体提升多少？怎么测的？"
  → 答不上来，或者答案经不起追问

实际写"GRPO 净 +1 题但标准误 3.4%，失败集合显示 10/13 原错误未变，
        说明任务太简单（81.5% 的步无组内方差），下一步应该加大难度"
  → 面试官知道你真的理解 GRPO 的适用条件
```

---

## 本篇小结：实验记录的设计原则

1. **原始数据和解读分开存。** 程序写的（`train_log.txt` / `train_metrics.txt`）
   可信度高于人写的（`notes.md`）。矛盾时以原始文件为准。
2. **命名要有信息量。** `2026-09-dpo-from-sft` 比 `exp2` 好一百倍。
3. **判断训练是否工作，看五个等级。** 从 `loss`（不可靠）到
   **产物字节比对（唯一可靠）**。三个实验证明了前四个都可能骗人。
4. **三种产物验证方法**：SHA256（续训）、逐张量差异（更有信息量）、
   检查零初始化参数（从零起 adapter）。
5. **`notes.md` 必须写"局限"。** 区分"方法无效"和"配置规模不足"。
6. **只报事实，不报期望。** "accuracy 0.455" 而不是"训练成功"。
7. **记录负面发现。** 它们是资产，不是失败。
8. **评测要保留 base 的对照输出。** 否则无法判断"变好还是本来就那样"。
9. **失败案例要同时记 `gold` 和 `pred`。** 区分"模型答错"和"reward 读错"。
10. **比对失败集合比看准确率更有信息量。** +1% 可能是 churn。
11. **显存异常低也是信号。** `0.0 GB` 暴露了"模型没上 GPU"。
12. **跨实验对比要控制变量。** 显存受步数影响、序列长度影响，不能直接比。
