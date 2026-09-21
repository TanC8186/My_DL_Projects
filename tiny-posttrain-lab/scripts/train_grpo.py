"""GRPO entrypoint for verifiable math rewards (8GB-safe defaults in YAML)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from my_lab.config import ensure_dir, load_config  # noqa: E402
from my_lab.rewards import accuracy_reward  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="tiny-posttrain-lab GRPO")
    parser.add_argument(
        "--config",
        default=str(_ROOT / "configs" / "grpo_qwen05b_8g.yaml"),
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    grpo_cfg = cfg["grpo"]
    out_dir = ensure_dir(cfg.get("output_dir", str(_ROOT / "experiments" / "grpo" / "output")))

    data_path = Path(data_cfg["dataset_path"])
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing {data_path}. Create math_sample.jsonl with fields prompt + answer."
        )

    rows = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    max_samples = data_cfg.get("max_samples")
    if max_samples:
        rows = rows[: int(max_samples)]

    print(f"[info] prompts={len(rows)} -> {out_dir}")
    print("[warn] GRPO on 8GB: keep num_generations and completion length small.")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    model_name = model_cfg["model_name_or_path"]
    torch_dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True, "torch_dtype": torch_dtype}
    if model_cfg.get("load_in_4bit"):
        model_kwargs["load_in_4bit"] = True
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False

    # TRL GRPO expects a "prompt" column; keep "answer" for reward
    dataset = Dataset.from_list(
        [{"prompt": r["prompt"], "answer": r["answer"]} for r in rows]
    )

    lora = cfg.get("lora") or {}
    peft_config = None
    if lora.get("enabled"):
        peft_config = LoraConfig(
            r=int(lora.get("r", 8)),
            lora_alpha=int(lora.get("alpha", 16)),
            lora_dropout=float(lora.get("dropout", 0.0)),
            target_modules=list(lora.get("target_modules", ["q_proj", "v_proj"])),
            task_type="CAUSAL_LM",
        )

    reward_fn = accuracy_reward if grpo_cfg.get("reward_function", "accuracy") == "accuracy" else accuracy_reward

    config = GRPOConfig(
        output_dir=str(out_dir),
        learning_rate=float(grpo_cfg.get("learning_rate", 1e-6)),
        per_device_train_batch_size=int(grpo_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(grpo_cfg.get("gradient_accumulation_steps", 4)),
        num_train_epochs=float(grpo_cfg.get("num_train_epochs", 1)),
        num_generations=int(grpo_cfg.get("num_generations", 2)),
        max_prompt_length=int(data_cfg.get("max_prompt_length", 128)),
        max_completion_length=int(data_cfg.get("max_completion_length", 128)),
        temperature=float(grpo_cfg.get("temperature", 1.0)),
        beta=float(grpo_cfg.get("beta", 0.0)),
        bf16=bool(grpo_cfg.get("bf16", True)),
        gradient_checkpointing=bool(grpo_cfg.get("gradient_checkpointing", True)),
        logging_steps=int(grpo_cfg.get("logging_steps", 1)),
        save_steps=int(grpo_cfg.get("save_steps", 100)),
        report_to=list(grpo_cfg.get("report_to") or []),
        seed=int(grpo_cfg.get("seed", 42)),
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    result = trainer.train()
    trainer.save_model(str(out_dir / "final"))
    (out_dir / "train_metrics.txt").write_text(str(result.metrics), encoding="utf-8")
    print(f"[done] {out_dir / 'final'}")


if __name__ == "__main__":
    main()
