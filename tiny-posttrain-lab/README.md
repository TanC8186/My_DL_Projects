# tiny-posttrain-lab

A reproducible **LLM post-training lab** for SFT / DPO / GRPO experiments on a **single 8GB GPU** (RTX 4060 Laptop class).

中文说明见 [README_zh.md](./README_zh.md)。

## Problem

Students and self-learners who want pre-training / post-training internships often have only a laptop GPU. Official frameworks are industrial-scale; most tutorials either skip real configs or cannot run under 8GB VRAM. This repo collects **measured, minimal configs** and a clean experiment layout so you can build a portfolio project — not just “run a tutorial once”.

## Who cares

- Internship prep for LLM pre/post-training (algorithms + small-scale engineering)
- Anyone who must reproduce alignment methods on constrained hardware
- Interview stories grounded in **real trade-offs** (OOM, reward design, seq length)

## Hardware measured

| Item | Value |
|------|--------|
| GPU | NVIDIA RTX 4060 Laptop |
| VRAM | 8 GB |
| Primary models | Qwen2.5-0.5B / 1.5B-Instruct (local weights under `E:\My_DL_Projects\qwen\`) |
| Framework | [Hugging Face TRL](https://github.com/huggingface/trl) (source in `../references/trl`) |

## Quick start

```powershell
# from repo root: E:\My_DL_Projects\tiny-posttrain-lab
# 1) create env (once)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt

# 2) run SFT baseline (0.5B, 8GB-friendly)
python scripts\train_sft.py --config configs\sft_qwen05b_8g.yaml
```

If `bitsandbytes` fails on Windows, try WSL2 or set `load_in_4bit: false` and use LoRA on bf16 0.5B.

## Experiment layout

| Path | Purpose |
|------|---------|
| `configs/` | YAML configs sized for 8GB |
| `scripts/` | Entrypoints: SFT / DPO / GRPO |
| `src/my_lab/` | Reusable code: data, rewards, memory estimate |
| `experiments/` | One folder per run: notes + results |
| `docs/` | Memory guide, dataflow notes |
| `tests/` | CPU unit tests (no GPU required) |
| `data/` | Tiny sample / download instructions (no large blobs) |

Local open-source references (not vendored as our code):

- `../references/trl` — Hugging Face TRL (read-only reference)

## Results

> Fill after your first successful run. Example table:

| Exp ID | Model | Method | VRAM peak | Reward / eval | Notes |
|--------|-------|--------|-----------|---------------|-------|
| 2026-09-sft-baseline | Qwen2.5-0.5B-Instruct | SFT+LoRA r=16 | 1.87 GB allocated / 4.89 GB reserved | train_loss 1.39, 345 s | first baseline + base-vs-LoRA generation comparison |
| 2026-09-dpo-from-sft | Qwen2.5-0.5B-Instruct | DPO (continues the SFT adapter) | 2.76 GB allocated / 3.56 GB reserved | preference accuracy 0.455 (chance = 0.5), 336 s | no measurable preference gain; surfaced a silent no-op training bug |

## Limitations

- Conclusions are for **small models** (≤1.5B–3B), not industrial 70B+ scale.
- Full distributed RL (verl / OpenRLHF Ray multi-role) is **out of scope** for local 8GB; those are read via `docs/`.
- Laptop GPUs throttle; wall-clock is not comparable to data-center cards.

## Roadmap

- [x] Repo skeleton + 8GB SFT config
- [x] SFT baseline on local Qwen weights
- [x] DPO small run + preference data slice
- [ ] GRPO + verifiable math reward ablation
- [x] Memory estimate utility + tests
- [x] Write `docs/8gb-memory-guide.md` from measured numbers

## License

Apache-2.0 (this lab’s own code). Cloned third-party projects keep their upstream licenses.
