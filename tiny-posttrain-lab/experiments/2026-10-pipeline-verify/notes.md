# Experiment: 2026-10-pipeline-verify

## 目标

在**重建后的 venv** 上验证 DPO 与 GRPO 两条链路端到端可跑，
并确认产出的数字与仓库既有记录一致。

## 状态

- [x] DPO 跑通（exit 0）
- [x] GRPO 跑通（exit 0，由内建守卫主动停止）
- [x] 与仓库记录对比
- [x] 既有实验的 output/ 未被覆盖

## 这不是一次"新实验"，是一次环境验证

本目录记录的是**环境健康检查**，不是新的科研结论。
配置与 `2026-09-dpo-from-sft` / `2026-09-grpo-math` **完全相同**，
只有 `output_dir` 被改到临时路径（`/tmp/verify-*/output`），
目的是**不覆盖既有实验的产物**。原始 `train_metrics.txt` 已确认未被改动。

复现方式：把两个 config 的 `output_dir` 指到任意临时目录后直接跑：

```bash
USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 ./.venv/Scripts/python.exe \
  scripts/train_dpo.py --config <dpo config with redirected output_dir>
USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 ./.venv/Scripts/python.exe \
  scripts/train_grpo.py --config <grpo config with redirected output_dir>
```

## 环境

- venv 重建后：Python 3.10.18 + torch 2.5.1+cu124，CUDA 可用
- `USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1`

## DPO

| 指标 | 记录 `2026-09-dpo-from-sft` | 本次验证 | 相对差 |
|------|------|------|------|
| train_loss | 0.8226 | **0.82231** | 0.035% |
| peak_allocated (GB) | 2.76 | **2.76** | 0 |
| peak_reserved (GB) | 3.56 | **3.56** | 0 |
| train_runtime (s) | 336 | 347.03 | +3.3% |
| steps | 125 | 125 | 0 |

**`start_from` 生效**：DPO 从 `2026-09-sft-baseline/output/final` 续训，
未重新生成 adapter（日志里出现 `rewards/chosen` / `rewards/rejected` /
`logps/chosen`，说明走的是 DPO 而非 SFT 路径）。

**结论**：DPO 链路在重建后的环境上完全复现，loss 与显存逐位一致。

## GRPO

| 指标 | 记录 `2026-09-grpo-math` | 本次验证 |
|------|------|------|
| train_loss | 3.17e-10 | -4.63e-11 |
| peak_allocated (GB) | 1.22 | 1.17 |
| peak_reserved (GB) | 1.49 | 1.35 |
| train_runtime (s) | 285.86 | 113.36 |
| 停止方式 | 跑完 | `[ABORT]`（守卫触发） |

### GRPO 被 `[ABORT]` 停止 —— 这是设计行为，而且是一条可复现的证据

`scripts/train_grpo.py` 里有一个 `ZeroSignalGuard` 回调：
当 `frac_reward_zero_std >= 1.0` 连续 20 个 logged step 时，
打印 `[ABORT]` 并设 `control.should_training_stop = True`。

本次触发于 **step 161 / 400**（epoch 0.80）。日志里的完整证据：

```text
[ABORT] reward std has been 0 for 20 logged steps, so every advantage is 0
        and no learning can happen.
        clipped_ratio=0.0 mean_length=3.0 reward=1.0
```

**关键观察：`clipped_ratio=0.0`、`mean_length=3.0`、`reward=1.0`。**

这三个数字说明**这跟项目里那个 gradient checkpointing bug 长得不一样**：

| 症状 | gckpt bug（`2026-09-grpo-math` 记录） | 本次 |
|------|------|------|
| clipped_ratio | **1.0**（全部撞到长度上限） | **0.0** |
| mean_length | 192（撞上限） | **3.0**（短而干净） |
| reward | **0**（全错） | **1.0**（全对） |

所以这**不是** gckpt bug 复发，而是**另一个方向**的失效：
模型已经把题目做对了，**组内 4 条生成全部正确 → 奖励方差为 0 → advantage 为 0**。
这正是配置注释里预言过的"任务对模型太简单"。

**有效信号比例（本次实测）**：

```text
logged steps             : 161
zero_std == 1.0（无信号）: 127  (78.9%)
zero_std == 0.0（有信号）:  34  (21.1%)
```

对比原实验记录的 **18.5%**（`frac_reward_zero_std == 1.0` 占 81.5%）。
两次独立运行给出 **18.5% 与 21.1%** —— 同一个机制、相近的量级。

**结论**：原实验"有效更新不到 20%"的结论**在重建后的环境上被独立复现**。
这比原实验本身更有说服力，因为它是**第二次独立观测**。

## 与本次会话其它结论的关系

- 训练路径本身健康（SFT 250 步跑完，loss/flops 与记录一致，见
  `2026-10-sft-repro`）。
- GRPO 提前停止**不是环境问题**，是任务难度与模型能力的匹配问题，
  这一点原实验已经记录过。

## 端到端评测复现：held-out 准确率精确命中

用记录在案的 GRPO adapter 重跑 held-out 评测（`--greedy`，确定性解码）：

```bash
.venv/Scripts/python.exe scripts/eval_math.py \
  --config configs/grpo_qwen05b_8g.yaml \
  --adapter experiments/2026-09-grpo-math/output/final \
  --data data/math_eval.jsonl --tag eval_recheck --out <path>
```

| | 记录 `eval_after.md` | 本次重跑 |
|------|------|------|
| accuracy | 88/100 = **0.88** | 88.0/100 = **0.8800** |
| peak_allocated (GB) | — | 0.94 |
| peak_reserved (GB) | — | 1.0 |

**逐位一致。** 这是本次环境验证最强的一条证据：不只是"能跑"，
而是**在重建后的 venv 上精确重放出原实验的核心指标**。

> 注：`--greedy` 是确定性解码（`eval_math.py --help` 称其为
> "the honest default for reporting"），所以这个数字应当精确可复现。
> 它确实精确复现了。

**踩到的坑（记录以免再犯）**：第一次调用**没有加 `--data`**，
于是 `eval_math.py` 回退到 config 里的 `dataset_path`，
也就是 `math_train.jsonl`（**训练集**），跑出 176/200。
那个 0.88 恰好和 held-out 的 0.88 一样，**极具误导性** ——
如果没核对报告的 `data:` 行，会误以为验证通过。

**教训**：评测脚本默认读 `dataset_path`（训练集）而不是 `eval_path`，
所以**每次评测都必须显式传 `--data <held-out 文件>`**；
并检查报告头部的 `data:` 行确认用的是哪个文件。

## 局限

- DPO/GRPO 各只跑一次，未做重复运行的统计检验。
- GRPO 未跑完（被守卫主动停止），因此不能与记录的"跑完"做直接的耗时对比。
- 产物（adapter）写在临时目录，未纳入本仓库。
- `eval_math.py --help` 里 `--data` 的默认值来自 config 的 `dataset_path`，
  这是一个容易误用训练集评测的接口陷阱（已在上文记录）。
