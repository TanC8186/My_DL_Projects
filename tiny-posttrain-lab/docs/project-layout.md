# 项目结构说明

```text
E:\My_DL_Projects\                     # 工作区根（已 git init）
├── .gitignore
├── qwen\                              # 你本地已有的模型权重（勿提交）
│   ├── Qwen2___5-0___5B-Instruct\
│   ├── Qwen2___5-1___5B-Instruct\
│   └── Qwen2___5-3B-Instruct\
├── references\                        # 第三方源码（默认 gitignore）
│   └── trl\                           # huggingface/trl（浅克隆，只读参考）
└── tiny-posttrain-lab\                # 方向一：你的开源主项目
    ├── README.md
    ├── README_zh.md
    ├── .gitignore
    ├── pyproject.toml
    ├── requirements.txt
    ├── configs\
    │   ├── sft_qwen05b_8g.yaml        # 先从这个开始
    │   ├── dpo_qwen05b_8g.yaml
    │   └── grpo_qwen05b_8g.yaml
    ├── scripts\
    │   ├── train_sft.py
    │   ├── train_dpo.py
    │   └── train_grpo.py
    ├── src\my_lab\
    │   ├── __init__.py
    │   ├── config.py
    │   ├── memory.py
    │   └── rewards.py
    ├── tests\
    │   └── test_rewards_and_memory.py
    ├── docs\
    │   ├── 8gb-memory-guide.md
    │   ├── project-layout.md
    │   └── trl-reading-notes.md        # TRL 源码精读（SFT loss / LoRA / DPO / GRPO）
    ├── data\
    │   ├── README.md
    │   └── math_sample.jsonl          # GRPO 用小样例
    └── experiments\
        └── 2026-09-sft-baseline\
            └── notes.md
```

## 职责划分

| 位置 | 是否你的代码 | 说明 |
|------|----------------|------|
| `tiny-posttrain-lab/` | 是 | 写实验、提 PR、面试展示 |
| `references/trl/` | 否 | 只读：对照 TRL 源码 |
| `qwen/` | 否（权重） | 本地模型，`.gitignore` 已排除 |

## 推荐学习路径（动手顺序）

1. 读 `configs/sft_qwen05b_8g.yaml`，确认模型路径指向你的 `qwen\...`
2. 建 venv，安装 `requirements.txt`
3. 运行 `python scripts\train_sft.py --config configs\sft_qwen05b_8g.yaml`
4. 把显存 / loss / 样例写入 `experiments/2026-09-sft-baseline/notes.md`
5. 打开 `references/trl/trl` 搜 `SFTTrainer`、`GRPOTrainer` 对照实现
6. 再开 DPO / GRPO

## 与 TRL 源码怎么对照

> ⚠️ `references/trl` 是 **1.14.0.dev0**（main 分支），而实际安装运行的是 **0.21.0**。
> 两者差一个大版本，`SFTTrainer` 的 loss 路径、`GRPOConfig` 字段都可能对不上。
> **读源码前先确认版本。**

已经整理好的结论在 [`trl-reading-notes.md`](./trl-reading-notes.md)，含行号引用。
下面是可以自己动手复现的检索入口：

| 你要搞清的问题 | 建议在 references/trl 里搜 |
|----------------|----------------------------|
| SFT loss 在哪些 token 上算 | `SFTTrainer`, `DataCollatorForLanguageModeling`, `completion_only_loss` |
| DPO loss | `DPOTrainer`, `dpo_loss`, `logratios` |
| DPO 参考模型是谁 | `null_ref_context`, `disable_adapter` |
| GRPO advantage | `GRPOTrainer`, `mean_grouped_rewards`, `num_generations` |
| GRPO 裁剪 | `coef_1`, `coef_2`, `epsilon_low` |
| reward 函数怎么被调用 | `_calculate_rewards` |
| CLI 入口 | `examples/` |

笔记写到 `docs/` 或 `experiments/*/notes.md`，**不要改 references 里的第三方代码**（要改就 fork 到你自己的实验分支，并写明目的）。
