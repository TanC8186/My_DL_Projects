"""Evaluate a model (optionally with a LoRA adapter) on a verifiable math set.

This is the missing held-out measurement: training loss does not tell you whether
the model got better at the task, and `accuracy_reward` is an exact checker.

    python scripts/eval_math.py --config configs/grpo_qwen05b_8g.yaml --limit 100
    python scripts/eval_math.py --config configs/grpo_qwen05b_8g.yaml \
        --adapter experiments/2026-09-grpo-math/output/final --tag after

Writes a markdown report and prints the accuracy plus failure examples.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from my_lab.config import ensure_dir, load_config  # noqa: E402
from my_lab.rewards import accuracy_reward, extract_final_number  # noqa: E402
from my_lab.vram import format_peak_vram, reset_peak_stats  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="tiny-posttrain-lab math evaluation")
    parser.add_argument(
        "--config",
        default=str(_ROOT / "configs" / "grpo_qwen05b_8g.yaml"),
        help="YAML config (for the model path / dtype)",
    )
    parser.add_argument("--model", default=None, help="override model path")
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory")
    parser.add_argument(
        "--data",
        default=None,
        help="jsonl with prompt/answer; defaults to the config's dataset_path",
    )
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N rows")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="greedy decoding (deterministic, the honest default for reporting)",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="eval", help="label used in the report")
    parser.add_argument("--out", default=None, help="markdown report path")
    return parser.parse_args()


def load_rows(path: Path, limit: int | None) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def load_model(model_name: str, cfg: dict, adapter: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = cfg.get("model") or {}
    dtype_name = str(model_cfg.get("torch_dtype", "bfloat16")).lower()
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype_name, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    kwargs: dict = {"trust_remote_code": True, "torch_dtype": torch_dtype}
    attn = model_cfg.get("attn_implementation")
    if attn:
        kwargs["attn_implementation"] = attn

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.config.use_cache = True
    # Qwen ships top_k=20 in generation_config.json; transformers warns about it on
    # every greedy call. Neutralise it here (sampling overrides it explicitly).
    model.generation_config.top_k = 0
    model.eval()

    if adapter:
        from peft import PeftModel

        # Inference only, so the default is_trainable=False is correct here.
        model = PeftModel.from_pretrained(model, adapter)
        model.eval()

    return model, tokenizer


def generate_all(model, tokenizer, prompts: list[str], args: argparse.Namespace) -> list[str]:
    import torch

    outputs = []
    for i, prompt in enumerate(prompts, 1):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        gen_kwargs: dict = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": not args.greedy,
            "temperature": args.temperature if not args.greedy else 1.0,
            "top_p": 1.0,
            "top_k": 0 if args.greedy else 50,
        }
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        outputs.append(
            tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()
        )
        if i % 25 == 0 or i == len(prompts):
            print(f"[info] generated {i}/{len(prompts)}")
    return outputs


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model_name = args.model or (cfg.get("model") or {}).get("model_name_or_path")
    if not model_name:
        raise ValueError("No model path: set --model or config model.model_name_or_path")

    data_cfg = cfg.get("data") or {}
    data_path = Path(args.data or data_cfg.get("dataset_path") or _ROOT / "data" / "math_train.jsonl")
    rows = load_rows(data_path, args.limit)

    import torch

    torch.manual_seed(args.seed)
    reset_peak_stats()
    print(f"[info] model={model_name}" + (f" + {args.adapter}" if args.adapter else ""))
    print(f"[info] data={data_path} rows={len(rows)}")
    model, tokenizer = load_model(model_name, cfg, args.adapter)

    outputs = generate_all(model, tokenizer, [r["prompt"] for r in rows], args)
    rewards = accuracy_reward(outputs, [str(r["answer"]) for r in rows])
    vram = format_peak_vram()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    correct = sum(rewards)
    total = len(rewards)
    accuracy = correct / total

    failures = [
        (rows[i]["prompt"], rows[i]["answer"], outputs[i], extract_final_number(outputs[i]))
        for i, r in enumerate(rewards)
        if r < 1.0
    ]

    decoding = "greedy" if args.greedy else f"sample(t={args.temperature})"
    header = [
        f"- tag: `{args.tag}`",
        f"- model: `{model_name}`",
        f"- adapter: `{args.adapter}`" if args.adapter else "- adapter: none",
        f"- data: `{data_path}` ({total} rows)",
        f"- decoding: {decoding}, max_new_tokens={args.max_new_tokens}, seed={args.seed}",
        f"- VRAM: {vram}",
        "",
        f"**Accuracy: {correct}/{total} = {accuracy:.4f}**",
        "",
    ]

    out_path = Path(args.out) if args.out else (
        _ROOT / "experiments" / f"eval_{args.tag}.md"
    )
    ensure_dir(out_path.parent)

    lines = ["# Math evaluation", "", *header, f"## Failures ({len(failures)})", ""]
    for prompt, gold, out, pred in failures:
        lines += [
            f"### gold={gold} pred={pred}",
            "",
            f"> {prompt}",
            "",
            "```text",
            out[:600],
            "```",
            "",
        ]
    out_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n=== accuracy: {correct}/{total} = {accuracy:.4f} ===")
    print(f"=== VRAM: {vram} ===")
    print(f"[done] report -> {out_path}")
    for prompt, gold, out, pred in failures[:5]:
        print(f"[miss] gold={gold} pred={pred} | {out[:90]!r}")


if __name__ == "__main__":
    main()
