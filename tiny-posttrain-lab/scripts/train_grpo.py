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
from my_lab.vram import format_peak_vram, peak_vram_gb, reset_peak_stats  # noqa: E402


def build_zero_signal_guard(base_callback, patience: int = 20):
    """Abort when GRPO has no learning signal.

    `frac_reward_zero_std == 1.0` means every completion in every group got the
    same reward, so all advantages are 0. loss and grad_norm are then exactly 0.0,
    which reads as "training is stable" rather than "training is dead".

    The usual cause is a prompt/format mismatch: if completions are truncated
    (clipped_ratio ~ 1.0) the model never emits an answer and every reward is 0.
    Aborting early matters because each step costs a full generation pass.
    """

    class ZeroSignalGuard(base_callback):
        def __init__(self) -> None:
            self.patience = patience
            self.streak = 0

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "frac_reward_zero_std" not in logs:
                return
            self.streak = self.streak + 1 if logs["frac_reward_zero_std"] >= 1.0 else 0

            if self.streak >= self.patience:
                print(
                    f"\n[ABORT] reward std has been 0 for {self.streak} logged steps, so every "
                    f"advantage is 0 and no learning can happen.\n"
                    f"         clipped_ratio={logs.get('completions/clipped_ratio')} "
                    f"mean_length={logs.get('completions/mean_length')} "
                    f"reward={logs.get('reward')}\n"
                    f"         Check that prompts carry the chat template and that "
                    f"max_completion_length leaves room for the answer.\n",
                    flush=True,
                )
                control.should_training_stop = True

    return ZeroSignalGuard()


def main() -> None:
    parser = argparse.ArgumentParser(description="tiny-posttrain-lab GRPO")
    parser.add_argument(
        "--config",
        default=str(_ROOT / "configs" / "grpo_qwen05b_8g.yaml"),
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help=(
            "Existing LoRA adapter to continue training. Defaults to the config's "
            "`start_from`; omit both to start a fresh adapter from the LoRA settings."
        ),
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    adapter = args.adapter or cfg.get("start_from")

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
    print("[warn] keep gradient_checkpointing=false; see the note in the YAML.")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
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

    if adapter:
        from peft import PeftModel

        # is_trainable=True is required, see the note in scripts/train_dpo.py:
        # the default leaves the optimizer empty and training silently no-ops.
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
        print(f"[info] continuing from adapter: {adapter}")

        # Only meaningful on this path. When TRL attaches a fresh LoRA itself, the
        # count here would just be the (still unfrozen) base model.
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[info] trainable parameters: {trainable:,}")
        if trainable == 0:
            raise RuntimeError(
                "No trainable parameters. The optimizer would be empty and training "
                "would silently do nothing. Check the adapter / LoRA configuration."
            )

    reset_peak_stats()

    # TRL applies the chat template only when the prompt is conversational (a list
    # of messages) — see `maybe_apply_chat_template` in grpo_trainer.py. Passing a
    # raw string makes an Instruct model continue plain text instead of answering:
    # every completion then runs to max_completion_length, scores 0, and GRPO gets
    # zero advantage (loss 0.0, grad_norm 0.0) while looking perfectly healthy.
    dataset = Dataset.from_list(
        [
            {"prompt": [{"role": "user", "content": r["prompt"]}], "answer": r["answer"]}
            for r in rows
        ]
    )

    lora = cfg.get("lora") or {}
    peft_config = None
    if adapter:
        print("[info] adapter given -> peft_config=None (keep training the loaded adapter)")
    elif lora.get("enabled"):
        peft_config = LoraConfig(
            r=int(lora.get("r", 8)),
            lora_alpha=int(lora.get("alpha", 16)),
            lora_dropout=float(lora.get("dropout", 0.0)),
            target_modules=list(lora.get("target_modules", ["q_proj", "v_proj"])),
            task_type="CAUSAL_LM",
        )

    # The config only ships one reward today; keep the lookup explicit so adding a
    # second one is a real change rather than a silent fallback.
    reward_name = grpo_cfg.get("reward_function", "accuracy")
    if reward_name != "accuracy":
        raise ValueError(f"Unknown reward_function: {reward_name!r} (only 'accuracy')")
    reward_fn = accuracy_reward

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
        callbacks=[build_zero_signal_guard(TrainerCallback)],
    )
    result = trainer.train()
    vram = peak_vram_gb()
    print(f"[info] peak VRAM: {format_peak_vram()}")
    trainer.save_model(str(out_dir / "final"))
    metrics_path = out_dir / "train_metrics.txt"
    metrics_path.write_text(str(result.metrics) + "\n" + format_peak_vram() + "\n", encoding="utf-8")
    print(f"[done] model   -> {out_dir / 'final'}")
    print(f"[done] metrics -> {metrics_path}")
    if vram:
        print(
            f"[hint] record into {out_dir.parent / 'notes.md'}:\n"
            f"       peak_allocated={vram['peak_allocated_gb']} GB, "
            f"peak_reserved={vram['peak_reserved_gb']} GB"
        )


if __name__ == "__main__":
    main()
