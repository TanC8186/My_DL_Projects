# llm-serving-bench

面向实习作品集的 **LLM 推理 / Serving** 实验仓库（MLSys、大模型 infra、部署向）。

**当前状态：仅骨架。** 请先学完并跑通 **`tiny-posttrain-lab`（后训练）**，再开这一 part。

English: [README.md](./README.md)

## 目标

在 **单卡 8GB**（RTX 4060 Laptop）上做出 **一条诚实、可复现的对比**：

> 同一小模型 + 同一批 prompt → HuggingFace `generate` vs llama.cpp (GGUF) vs vLLM（显存允许时）  
> 指标：tokens/s、近似首 token 延迟、显存峰值、生成质量抽检

## 优先级（先看这里）

| 顺序 | 项目 | 原因 |
|------|------|------|
| **1（现在）** | `../tiny-posttrain-lab` | 后训练 SFT/DPO/GRPO —— **8G 上含金量最高、最该先做完** |
| **2（之后）** | 本仓库 | 推理/serving 叙事；8G 上 **llama.cpp 最稳** |
| **只读参考** | `../references/*` | vLLM / SGLang / llama.cpp / TRL 源码 |

在写完 SFT 的 `experiments/notes.md` 之前，不要开大规模多引擎压测。

## 参考源码（已 clone，只读）

| 路径 | 项目 | 8G 用途 |
|------|------|---------|
| `../references/trl` | TRL | 后训练精读 |
| `../references/llama.cpp` | llama.cpp | **本地 serving 首选**（GGUF） |
| `../references/vllm` | vLLM | PagedAttention / batching 精读 + 可选小模型服务 |
| `../references/sglang` | SGLang | RadixAttention / 结构化生成精读 |

## 何时启动（Part 2）

```powershell
cd E:\My_DL_Projects\llm-serving-bench
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\bench_llama_cpp.py --config configs\llama_cpp_qwen05b.yaml
```

## 目录

```text
llm-serving-bench/
├── configs/  scripts/  src/serving_bench/
├── experiments/  results/  docs/  tests/
```
