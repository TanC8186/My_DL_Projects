"""Minimal SFT entrypoint for 8GB laptop configs (TRL SFTTrainer)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without install: add src/ to path
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from my_lab.config import ensure_dir, load_config  # noqa: E402
from my_lab.vram import format_peak_vram, peak_vram_gb, reset_peak_stats  # noqa: E402


def build_lora_config(cfg: dict):
    from peft import LoraConfig

    lora = cfg.get("lora") or {}
    if not lora.get("enabled", False):
        return None
    return LoraConfig(
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=list(lora.get("target_modules", ["q_proj", "v_proj"])),
        task_type="CAUSAL_LM",
    )


def load_dataset_from_cfg(data_cfg: dict):
    from datasets import load_dataset

    if data_cfg.get("dataset_path"):
        path = Path(data_cfg["dataset_path"])
        if not path.exists():
            raise FileNotFoundError(
                f"dataset_path not found: {path}\n"
                f"Create a small jsonl or switch to dataset_name in the YAML."
            )
        if path.suffix == ".jsonl":
            ds = load_dataset("json", data_files=str(path), split="train")
        else:
            ds = load_dataset(str(path))
            if isinstance(ds, dict):
                ds = ds[list(ds.keys())[0]]
    else:
        name = data_cfg["dataset_name"]
        split = data_cfg.get("dataset_split", "train")
        ds = load_dataset(name, split=split)

    max_samples = data_cfg.get("max_samples")
    if max_samples:
        ds = ds.select(range(min(int(max_samples), len(ds))))
    return ds


def main() -> None:
    parser = argparse.ArgumentParser(description="tiny-posttrain-lab SFT")
    parser.add_argument(
        "--config",
        default=str(_ROOT / "configs" / "sft_qwen05b_8g.yaml"),
        help="Path to YAML config",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    out_dir = ensure_dir(train_cfg["output_dir"])

    print(f"[info] experiment: {cfg.get('experiment_id', 'unnamed')}")
    print(f"[info] model: {model_cfg['model_name_or_path']}")
    print(f"[info] output: {out_dir}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTConfig, SFTTrainer

    model_name = model_cfg["model_name_or_path"]
    dtype_name = str(model_cfg.get("torch_dtype", "bfloat16")).lower()
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype_name, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }
    if model_cfg.get("load_in_4bit"):
        model_kwargs["load_in_4bit"] = True
        model_kwargs["device_map"] = "auto"
    attn = model_cfg.get("attn_implementation")
    if attn:
        model_kwargs["attn_implementation"] = attn

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False
    if train_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    reset_peak_stats()

    dataset = load_dataset_from_cfg(data_cfg)
    print(f"[info] dataset rows: {len(dataset)}")
    print(f"[info] columns: {dataset.column_names}")

    # Alpaca-style prompt — adjust if your data uses chat messages only
    def format_example(example: dict) -> dict:
        instr = example.get("instruction") or example.get("prompt") or ""
        inp = example.get("input") or ""
        out = example.get("output") or example.get("response") or example.get("completion") or ""
        if inp:
            text = (
                f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
            )
        else:
            text = f"### Instruction:\n{instr}\n\n### Response:\n{out}"
        return {"text": text}

    # If dataset already has "text", keep it; else try format
    if "text" not in dataset.column_names:
        dataset = dataset.map(format_example)

    sft_kwargs = dict(
        output_dir=str(out_dir),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "cosine")),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        logging_steps=int(train_cfg.get("logging_steps", 5)),
        save_steps=int(train_cfg.get("save_steps", 200)),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        bf16=bool(train_cfg.get("bf16", True)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        optim=str(train_cfg.get("optim", "adamw_torch")),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        report_to=list(train_cfg.get("report_to") or []),
        seed=int(train_cfg.get("seed", 42)),
        packing=bool(data_cfg.get("packing", False)),
        max_length=int(data_cfg.get("max_seq_length", 512)),
        dataset_text_field="text",
    )

    try:
        sft_config = SFTConfig(**sft_kwargs)
    except TypeError:
        # Older TRL: pass TrainingArguments + max_seq_length to trainer
        sft_config = TrainingArguments(**{k: v for k, v in sft_kwargs.items() if k != "max_length"})

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=build_lora_config(cfg),
        processing_class=tokenizer,
    )

    result = trainer.train(
        resume_from_checkpoint=train_cfg.get("resume_from_checkpoint") or None
    )
    vram = peak_vram_gb()
    print(f"[info] peak VRAM: {format_peak_vram()}")
    trainer.save_model(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))

    metrics_path = out_dir / "train_metrics.txt"
    metrics_text = str(result.metrics) + "\n" + format_peak_vram() + "\n"
    metrics_path.write_text(metrics_text, encoding="utf-8")
    print(f"[done] metrics -> {metrics_path}")
    print(f"[done] model   -> {out_dir / 'final'}")

    if vram:
        notes_hint = out_dir.parent / "notes.md"
        print(
            f"[hint] record into {notes_hint}:\n"
            f"       peak_allocated={vram['peak_allocated_gb']} GB, "
            f"peak_reserved={vram['peak_reserved_gb']} GB"
        )


if __name__ == "__main__":
    main()
