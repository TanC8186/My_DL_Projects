# llm-serving-bench

Inference / serving experiments for **internship portfolio** (MLSys, LLM infra, deployment).

**Status:** skeleton only — finish **`tiny-posttrain-lab` first** (post-training入门), then open this part.

中文说明见 [README_zh.md](./README_zh.md).

## Goal

On a **single 8GB laptop GPU** (RTX 4060), produce **one honest, reproducible comparison**:

> Same small model + same prompts → HF `generate` vs llama.cpp (GGUF) vs vLLM (if VRAM allows)  
> Metrics: tokens/s, approximate TTFT, peak VRAM, quality smoke-check.

## Priority (read this first)

| Order | Project | Why |
|------:|---------|-----|
| **1 (now)** | `../tiny-posttrain-lab` | Post-training SFT/DPO/GRPO — **highest value that runs on 8G** for training/alignment internships |
| **2 (later)** | this repo `llm-serving-bench` | Inference/serving story; llama.cpp path is the most reliable on 8G |
| **read-only** | `../references/*` | vLLM / SGLang / llama.cpp / TRL source for deep-dives |

Do **not** start multi-engine benchmarking before your first SFT notes are written.

## Reference clones (read-only)

| Path | Project | Use on 8G |
|------|---------|-----------|
| `../references/trl` | Hugging Face TRL | Post-train code reading |
| `../references/llama.cpp` | llama.cpp | **Primary** local serving path (GGUF) |
| `../references/vllm` | vLLM | PagedAttention / batching reading + optional small-model server |
| `../references/sglang` | SGLang | RadixAttention / structured generation reading |

## Quick start (when you open Part 2)

```powershell
cd E:\My_DL_Projects\llm-serving-bench
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Phase A (recommended first): llama.cpp GGUF smoke bench
# Download a small GGUF into models/ then:
python scripts\bench_llama_cpp.py --config configs\llama_cpp_qwen05b.yaml
```

vLLM on 8GB is **optional** — only after llama.cpp baseline works.

## Repo layout

```text
llm-serving-bench/
├── configs/          # per-engine YAML
├── scripts/          # bench entrypoints
├── src/serving_bench/
├── experiments/      # notes per run
├── results/          # csv/json (small)
├── docs/
└── tests/
```

## License

Lab code: Apache-2.0. `references/*` keep upstream licenses.
