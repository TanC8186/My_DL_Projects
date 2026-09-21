# 00 · 项目全景与阅读指南

> 这是 `tiny-posttrain-lab` 的**逐文件、逐段代码**梳理笔记，共 10 篇。
> 每篇都按「这段代码在干什么 → 为什么这么写 → 背后的原理 → 踩过的坑」展开。

## 这个项目到底是什么

一句话：**在一张 8GB 笔记本显卡上，把 LLM 后训练的三种主要方法（SFT / DPO / GRPO）各跑通一次，
并留下可复现的配置、实测数字和诚实的结论。**

它不是一个框架，而是一个**实验库**：配置是 YAML，入口是脚本，产物是「模型 adapter + 实验记录」。

## 目录树与每个文件的职责

```text
tiny-posttrain-lab/
├── README.md / README_zh.md        # 项目说明、结果表、路线图（英文/中文）
├── pyproject.toml                  # 包元数据 + 依赖声明 + ruff 配置
├── requirements.txt                # 实测可用的精确版本锁定（含环境说明注释）
├── .gitignore                      # 忽略规则（关键：保留 train_metrics.txt）
│
├── configs/                        # 【配置层】三种方法的 YAML
│   ├── sft_qwen05b_8g.yaml
│   ├── dpo_qwen05b_8g.yaml
│   └── grpo_qwen05b_8g.yaml
│
├── scripts/                        # 【执行层】命令行入口
│   ├── env.ps1                     #   环境变量（USE_TF 等），dot-source 使用
│   ├── run_sft.cmd                 #   一键跑 SFT 的 cmd 包装
│   ├── train_sft.py                #   SFT 训练入口
│   ├── train_dpo.py                #   DPO 训练入口
│   ├── train_grpo.py               #   GRPO 训练入口
│   ├── generate.py                 #   生成 / base-vs-adapter 对比
│   ├── eval_math.py                #   held-out 数学评测
│   └── make_math_dataset.py        #   可复现的算术数据集生成器
│
├── src/my_lab/                     # 【工具层】可复用、可单测的纯逻辑
│   ├── __init__.py                 #   包导出
│   ├── config.py                   #   YAML 读取 + 建目录
│   ├── memory.py                   #   显存粗估（面试口述用）
│   ├── vram.py                     #   CUDA 峰值显存实测（惰性导入 torch）
│   └── rewards.py                  #   可验证 reward：抽取数字 + 比对
│
├── tests/                          # 【验证层】纯 CPU 单测，不需要 GPU
│   ├── test_project_files.py       #   文件/目录结构契约
│   ├── test_rewards_and_memory.py  #   reward 与显存估算
│   ├── test_vram.py                #   vram 模块的 CPU 安全性
│   └── test_math_dataset.py        #   数据集生成器的正确性与切分不重叠
│
├── data/                           # 【数据层】小样例，不提交大文件
│   ├── README.md                   #   数据格式约定 + 生成命令
│   ├── math_sample.jsonl           #   10 行手写最小样例
│   ├── math_train.jsonl            #   200 行生成训练集
│   └── math_eval.jsonl             #   100 行 held-out 评测集
│
├── docs/                           # 【文档层】
│   ├── project-layout.md           #   项目结构与学习路径
│   ├── 8gb-memory-guide.md         #   显存实测数据 + OOM 排查顺序
│   ├── trl-reading-notes.md        #   TRL 源码精读（机制层）
│   └── walkthrough/                #   ← 本系列（代码层）
│
└── experiments/                    # 【记录层】一次实验一个文件夹
    ├── 2026-09-sft-baseline/
    │   ├── notes.md                #   实验记录（目标/配置/显存/结果/结论/局限）
    │   ├── train_log.txt           #   训练原始日志
    │   ├── generation_compare.md   #   base vs LoRA 生成对比
    │   └── output/                 #   产物（gitignore，只保留 train_metrics.txt）
    ├── 2026-09-dpo-from-sft/
    └── 2026-09-grpo-math/
```

## 数据流：一次训练到底发生了什么

以 SFT 为例，把整条链路串起来：

```text
configs/sft_qwen05b_8g.yaml          ← 你手写的配置
        │
        │  scripts/train_sft.py 读入
        ▼
   cfg = load_config(...)             ← src/my_lab/config.py
        │
        ├─→ cfg["model"]   → AutoTokenizer / AutoModelForCausalLM   （加载本地权重）
        ├─→ cfg["lora"]    → LoraConfig                             （peft，可训练参数 0.436%）
        ├─→ cfg["data"]    → load_dataset + format_example          （变成 text 字段）
        └─→ cfg["training"]→ SFTConfig                              （传给 TRL）
                                    │
                                    ▼
                            SFTTrainer(...).train()
                                    │  内部：DataCollator 造 labels → forward → backward
                                    ▼
                    experiments/<id>/output/final/            ← adapter 权重
                    experiments/<id>/output/train_metrics.txt ← 指标 + 显存
        │
        ▼
   scripts/generate.py --compare      ← 用产物做 base vs LoRA 生成对比
   scripts/eval_math.py               ← 用产物做 held-out 评测
        │
        ▼
   experiments/<id>/notes.md          ← 人写的结论（含局限）
```

**关键设计：训练脚本只产出「事实」（权重、指标、显存），「解读」留给 `notes.md` 和评测脚本。**
这样指标不会和结论混在一起。

## 模块依赖关系

```text
scripts/*.py  ──────┐
                    ├──→ src/my_lab/config.py     (YAML 读取)
                    ├──→ src/my_lab/vram.py       (显存实测)
                    ├──→ src/my_lab/rewards.py    (只有 GRPO/eval 用)
                    └──→ src/my_lab/memory.py     (只有测试用，面试口述)

tests/*.py ─────────┴──→ src/my_lab/*  （纯 CPU，不在模块级 import torch）

src/my_lab/__init__.py  ← 聚合导出，让 `from my_lab import xxx` 可用
```

**`src/my_lab/` 里没有 torch 的模块级导入** —— 这是刻意设计，见 [03 篇](./03-helpers.md)。

## 三个实验的关系

```text
         Qwen2.5-0.5B-Instruct（本地原始权重）
                    │
      ┌─────────────┼──────────────────┐
      │             │                  │
      ▼             ▼                  ▼
   SFT 基线      DPO 实验          GRPO 实验
   (alpaca)    (续 SFT adapter)   (从原始权重起步)
      │             │                  │
      └─ adapter ───┘                  └─ 为什么从原始起步？
         SFT→DPO 是真实流水线             为了隔离 GRPO 自身的效果，
         但参考模型错位（见 05 篇）        否则分不清"GRPO 有用"和"起点本来就好"
```

三个实验的结论**都不是"效果变好了"**：

| 实验 | 结果 |
|------|------|
| SFT | 风格被成功改写（去掉客套），但**出现事实退化**（太阳系漏项错位） |
| DPO | 偏好准确率 0.455（随机 0.5），**无提升**，且暴露一个静默失效 bug |
| GRPO | held-out 88 vs 基座 87，**净 +1 属噪声**（churn），但修好两个 bug |

**这三个"没变好"的结论本身就是这个项目最有价值的部分** —— 它们有完整证据链，
比一个含糊的"我跑了 SFT 效果不错"可信得多。

## 建议阅读顺序

| 顺序 | 篇目 | 为什么先读它 |
|------|------|-------------|
| 1 | [01 · 环境与工程配置](./01-environment.md) | 不解决 `USE_TF` 问题，后面所有代码都跑不起来 |
| 2 | [02 · 配置层](./02-config-layer.md) | 三个 YAML 是全部行为的输入，先建立"参数从哪来"的认知 |
| 3 | [03 · 工具层](./03-helpers.md) | `my_lab/` 是唯一被单测覆盖的纯逻辑，也是面试最爱问的细节 |
| 4 | [04 · SFT 训练脚本](./04-train-sft.md) | 最简单的一条链路，把「配置→训练→产物」走通 |
| 5 | [05 · DPO 训练脚本](./05-train-dpo.md) | 引入参考模型与 adapter 续训，含最隐蔽的 bug |
| 6 | [06 · GRPO 训练脚本](./06-train-grpo.md) | 最复杂：要生成、要 reward、要组内比较 |
| 7 | [07 · 推理与评测](./07-infer-eval.md) | 怎么把"训练完了"变成"有没有变好" |
| 8 | [08 · 测试](./08-tests.md) | 怎么在没有 GPU 的情况下保证代码正确 |
| 9 | [09 · 实验记录与日志阅读](./09-experiments.md) | 怎么读日志、怎么写 notes、怎么判断训练是否真的在工作 |

## 读完这一系列你应该能回答

- 为什么 `my_lab` 里不在模块级 import torch？（→ 03 篇）
- 为什么 `config.py` 只有 20 行？（→ 02 篇：校验交给下游，不重复造轮子）
- 为什么 SFT 脚本要 `padding_side = "right"` 而推理脚本是 `"left"`？（→ 04、07 篇）
- 为什么 `PeftModel.from_pretrained` 必须加 `is_trainable=True`？（→ 05 篇，最贵的教训）
- 为什么 GRPO 的 `gradient_checkpointing` 必须是 `false`？（→ 06 篇）
- 为什么 `train_metrics.txt` 要从 gitignore 里豁免出来？（→ 01 篇）
- 为什么评测集要和训练集从同一个池子里切？（→ 07 篇）
