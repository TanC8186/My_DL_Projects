"""Generation / before-after comparison for tiny-posttrain-lab.

Produces the "does the fine-tune actually change anything?" evidence that
`experiments/*/notes.md` asks for.

Examples:
  # base vs LoRA adapter, writes a markdown report
  python scripts/generate.py --config configs/sft_qwen05b_8g.yaml --compare

  # single run with an explicit adapter
  python scripts/generate.py --config configs/sft_qwen05b_8g.yaml \
      --adapter experiments/2026-09-sft-baseline/output/final

  # custom prompts, greedy decoding
  python scripts/generate.py --prompts "What is 12 + 30?" --greedy
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from my_lab.config import ensure_dir, load_config  # noqa: E402
from my_lab.vram import format_peak_vram, reset_peak_stats  # noqa: E402

DEFAULT_PROMPTS = [
    "Give three tips for staying healthy.",
    "Explain what a GPU is to a 10-year-old.",
    "用一句话解释什么是大语言模型的后训练。",
    "List the planets in the solar system in order.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="tiny-posttrain-lab generation")
    parser.add_argument(
        "--config",
        default=str(_ROOT / "configs" / "sft_qwen05b_8g.yaml"),
        help="YAML config (used for model path + dtype)",
    )
    parser.add_argument("--model", default=None, help="override model path/id")
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="run base model and adapter side by side (writes markdown)",
    )
    parser.add_argument("--prompts", nargs="*", default=None, help="inline prompts")
    parser.add_argument("--prompts-file", default=None, help="file with one prompt per line")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--greedy", action="store_true", help="deterministic greedy decoding")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="markdown output path")
    return parser.parse_args()


def resolve_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts:
        return list(args.prompts)
    if args.prompts_file:
        lines = Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
        prompts = [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
        if not prompts:
            raise ValueError(f"No usable prompts in {args.prompts_file}")
        return prompts
    return list(DEFAULT_PROMPTS)


def default_adapter_path(cfg: dict) -> Path:
    """`<training.output_dir>/final` — where train_*.py saves the adapter."""
    out_dir = (cfg.get("training") or {}).get("output_dir")
    if not out_dir:
        raise ValueError("config has no training.output_dir; pass --adapter explicitly")
    return Path(out_dir) / "final"


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

    model_kwargs: dict = {"trust_remote_code": True, "torch_dtype": torch_dtype}
    attn = model_cfg.get("attn_implementation")
    if attn:
        model_kwargs["attn_implementation"] = attn

    if model_cfg.get("load_in_4bit"):
        model_kwargs["load_in_4bit"] = True
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Without a Trainer, from_pretrained stays on CPU. Move it unless the
    # quantized path already placed it via device_map="auto".
    if "device_map" not in model_kwargs:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"[info] device: {device}")

    model.config.use_cache = True
    model.eval()

    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model.eval()

    return model, tokenizer


def generate_one(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    import torch

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    gen_kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
        # Pass explicit no-op values when greedy so the model's saved
        # generation_config (temperature/top_p/top_k) does not trigger warnings.
        "temperature": temperature if do_sample else 1.0,
        "top_p": top_p if do_sample else 1.0,
        "top_k": 0 if not do_sample else 50,
    }

    with torch.inference_mode():
        output = model.generate(**inputs, **gen_kwargs)

    new_tokens = output[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_model(
    label: str,
    model_name: str,
    cfg: dict,
    adapter: str | None,
    prompts: list[str],
    args: argparse.Namespace,
) -> dict:
    import torch

    torch.manual_seed(args.seed)
    reset_peak_stats()

    print(f"[info] loading {label}: {model_name}" + (f" + {adapter}" if adapter else ""))
    model, tokenizer = load_model(model_name, cfg, adapter)

    outputs = []
    for i, prompt in enumerate(prompts, 1):
        print(f"[info] [{label}] generating {i}/{len(prompts)} ...")
        outputs.append(
            generate_one(
                model,
                tokenizer,
                prompt,
                args.max_new_tokens,
                do_sample=not args.greedy,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        )

    vram = format_peak_vram()
    print(f"[info] [{label}] {vram}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {"label": label, "adapter": adapter, "outputs": outputs, "vram": vram}


def write_report(
    path: Path,
    header: list[str],
    prompts: list[str],
    runs: list[dict],
) -> None:
    lines = ["# Generation report", "", *header, ""]
    for i, prompt in enumerate(prompts, 1):
        lines += [f"## Prompt {i}", "", f"> {prompt}", ""]
        for run in runs:
            lines += [f"**{run['label']}**", "", run["outputs"][i - 1] or "_(empty)_", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] report -> {path}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model_name = args.model or (cfg.get("model") or {}).get("model_name_or_path")
    if not model_name:
        raise ValueError("No model path: set --model or config model.model_name_or_path")

    prompts = resolve_prompts(args)
    decoding = "greedy" if args.greedy else f"sample(t={args.temperature}, top_p={args.top_p})"
    header = [
        f"- model: `{model_name}`",
        f"- max_new_tokens: {args.max_new_tokens}",
        f"- decoding: {decoding}",
        f"- seed: {args.seed}",
    ]

    runs: list[dict] = []

    if args.compare:
        adapter_path = Path(args.adapter) if args.adapter else default_adapter_path(cfg)
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"Adapter not found: {adapter_path}\n"
                f"Train first, or pass --adapter / a different --config."
            )
        runs.append(run_model("Base (no adapter)", model_name, cfg, None, prompts, args))
        runs.append(run_model("SFT + LoRA", model_name, cfg, str(adapter_path), prompts, args))
        header.append(f"- adapter: `{adapter_path}`")
        header.append(f"- VRAM base: {runs[0]['vram']}")
        header.append(f"- VRAM adapter: {runs[1]['vram']}")
    else:
        runs.append(run_model("Model", model_name, cfg, args.adapter, prompts, args))
        if args.adapter:
            header.append(f"- adapter: `{args.adapter}`")
        header.append(f"- VRAM: {runs[0]['vram']}")

    out_path = Path(args.out) if args.out else _ROOT / "experiments" / "generation_report.md"
    write_report(ensure_dir(out_path.parent) / out_path.name, header, prompts, runs)

    for i, prompt in enumerate(prompts, 1):
        print(f"\n=== Prompt {i}: {prompt}")
        for run in runs:
            print(f"--- {run['label']} ---")
            print(run["outputs"][i - 1])


if __name__ == "__main__":
    main()
