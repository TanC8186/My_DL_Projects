"""Rough VRAM heuristics for laptop planning (NOT exact)."""

from __future__ import annotations


def estimate_training_vram_gb(
    model_params_b: float,
    dtype_bytes: int = 2,
    lora: bool = True,
    gradient_checkpointing: bool = True,
    batch_tokens: int = 512,
    overhead_factor: float = 1.6,
) -> float:
    """
    Ballpark forward+backward VRAM for full/LoRA fine-tuning.

    This is intentionally simple for interview notes and OOM triage.
    Real usage depends on attention, optimizer state, fragmentation, etc.
    """
    # Weights
    weight_gb = model_params_b * dtype_bytes
    if lora:
        # Base frozen; adapters + grads small vs base — keep base weights.
        optimizer_gb = 0.05 * model_params_b  # loose adapter/optimizer estimate
        activation_scale = 0.35 if gradient_checkpointing else 0.9
    else:
        # AdamW m,v ≈ 2x fp32 states often; keep a simplified 2x weights extra
        optimizer_gb = model_params_b * dtype_bytes * 2
        activation_scale = 0.5 if gradient_checkpointing else 1.2

    # Activations scale roughly with batch tokens and model size
    activation_gb = model_params_b * (batch_tokens / 512.0) * activation_scale * 0.15
    total = (weight_gb + optimizer_gb + activation_gb) * overhead_factor
    return round(total, 2)
