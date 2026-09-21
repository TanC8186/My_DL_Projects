"""CUDA peak-memory helpers.

torch is imported lazily inside the functions so this module stays importable
on CPU-only machines and inside the torch-free unit tests.
"""

from __future__ import annotations


def cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def reset_peak_stats() -> bool:
    """Reset CUDA peak counters. Returns True when a GPU was actually reset."""
    if not cuda_available():
        return False
    import torch

    torch.cuda.reset_peak_memory_stats()
    return True


def peak_vram_gb() -> dict[str, float]:
    """Peak allocated / reserved VRAM in GB. Empty dict when CUDA is absent."""
    if not cuda_available():
        return {}
    import torch

    gib = 1024.0**3
    return {
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / gib, 2),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / gib, 2),
    }


def format_peak_vram() -> str:
    """One-line summary suitable for logs and experiment notes."""
    stats = peak_vram_gb()
    if not stats:
        return "n/a (no CUDA)"
    return (
        f"peak_allocated={stats['peak_allocated_gb']} GB, "
        f"peak_reserved={stats['peak_reserved_gb']} GB"
    )
