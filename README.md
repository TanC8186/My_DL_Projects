# My_DL_Projects

大模型工程练习工作区（预训练 / 后训练 / 推理 Serving — 实习向）。

## 当前优先级（必读）

| 顺序 | 项目 | 状态 | 说明 |
|------|------|------|------|
| **P0 现在** | [`tiny-posttrain-lab/`](./tiny-posttrain-lab/) | 可开工 | **8G 上含金量最高的主项目**：TRL 后训练（SFT→DPO→GRPO） |
| **P1 之后** | [`llm-serving-bench/`](./llm-serving-bench/) | 仅骨架 | 推理对比；后训练入门完成后再开 |
| 只读 | `references/` | 已 clone | 源码精读，勿当自己的代码改 |

**结论：先只做 `tiny-posttrain-lab`。**  
在 `tiny-posttrain-lab/experiments/2026-09-sft-baseline/notes.md` 写完第一份 SFT 实验记录之前，不要开 serving 压测。

## 为什么主项目选 tiny-posttrain-lab（而不是 vLLM）

1. **你的目标**是预训练/后训练实习，JD 更常考 SFT / RLHF / 数据与训练流程。  
2. **8G 能完整闭环**：Qwen2.5-0.5B 本地权重 + TRL LoRA SFT，结果可写进简历。  
3. vLLM/SGLang 在 8G 上难以做「有说服力的生产级压测」，适合 **精读 + 之后小实验**。  
4. 后训练里 rollout 会用到 vLLM——学完训练再读 serving，叙事更顺。

## 目录

```text
E:\My_DL_Projects\
├── README.md                 ← 本文件
├── .gitignore
├── qwen\                     ← 本地 Qwen2.5 权重（0.5B/1.5B/3B，已忽略提交）
├── tiny-posttrain-lab\       ← P0 主项目
├── llm-serving-bench\        ← P1 骨架（暂不深入）
└── references\               ← 第三方源码（只读）
    ├── trl\                  ← Hugging Face TRL
    ├── llama.cpp\            ← GGUF / 端侧推理
    ├── vllm\                 ← PagedAttention 等
    └── sglang\               ← RadixAttention 等
```

## 现在开始（P0）

```powershell
cd E:\My_DL_Projects\tiny-posttrain-lab
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\train_sft.py --config configs\sft_qwen05b_8g.yaml
```

说明文档：

- [tiny-posttrain-lab/README_zh.md](./tiny-posttrain-lab/README_zh.md)
- [tiny-posttrain-lab/docs/project-layout.md](./tiny-posttrain-lab/docs/project-layout.md)
- [llm-serving-bench/docs/serving-reading-guide.md](./llm-serving-bench/docs/serving-reading-guide.md)（Part 2 浏览即可）

## Git

`master` 分支。`qwen/`、`references/`、模型大文件默认 ignore。
