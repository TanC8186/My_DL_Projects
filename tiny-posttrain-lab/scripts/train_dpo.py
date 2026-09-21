"""DPO entrypoint (8GB configs). Run after you finish SFT baseline notes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from my_lab.config import ensure_dir, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="tiny-posttrain-lab DPO")
    parser.add_argument(
        "--config",
        default=str(_ROOT / "configs" / "dpo_qwen05b_8g.yaml"),
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    out_dir = ensure_dir(train_cfg["output_dir"])

    print(f"[info] experiment: {cfg.get('experiment_id')}")
    print("[warn] DPO needs prompt/chosen/rejected columns — verify your dataset.")

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

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

    model_kwargs = {"trust_remote_code": True, "torch_dtype": torch_dtype}
    if model_cfg.get("load_in_4bit"):
        model_kwargs["load_in_4bit"] = True
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False

    ds = load_dataset(data_cfg["dataset_name"], split=data_cfg.get("dataset_split", "train"))
    max_samples = data_cfg.get("max_samples")
    if max_samples:
        ds = ds.select(range(min(int(max_samples), len(ds))))
    print(f"[info] dataset rows={len(ds)} columns={ds.column_names}")

    peft_config = None
    lora = cfg.get("lora") or {}
    if lora.get("enabled"):
        peft_config = LoraConfig(
            r=int(lora.get("r", 16)),
            lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            target_modules=list(lora.get("target_modules", ["q_proj", "v_proj"])),
            task_type="CAUSAL_LM",
        )

    dpo_args = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(train_cfg.get("learning_rate", 5e-6)),
        beta=float(train_cfg.get("beta", 0.1)),
        loss_type=str(train_cfg.get("loss_type", "sigmoid")),
        max_prompt_length=int(data_cfg.get("max_prompt_length", 256)),
        max_length=int(data_cfg.get("max_length", 512)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        bf16=bool(train_cfg.get("bf16", True)),
        logging_steps=int(train_cfg.get("logging_steps", 5)),
        save_steps=int(train_cfg.get("save_steps", 200)),
        report_to=list(train_cfg.get("report_to") or []),
        seed=int(train_cfg.get("seed", 42)),
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT adapter: TRL can use disabled adapter as ref
        args=dpo_args,
        train_dataset=ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    result = trainer.train()
    trainer.save_model(str(out_dir / "final"))
    (out_dir / "train_metrics.txt").write_text(str(result.metrics), encoding="utf-8")
    print(f"[done] {out_dir / 'final'}")


if __name__ == "__main__":
    main()
