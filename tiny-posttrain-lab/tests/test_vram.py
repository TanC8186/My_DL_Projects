"""VRAM helpers must stay importable and safe without CUDA."""

from my_lab.vram import (
    cuda_available,
    format_peak_vram,
    peak_vram_gb,
    reset_peak_stats,
)


def test_vram_helpers_are_cpu_safe():
    assert isinstance(cuda_available(), bool)
    assert isinstance(reset_peak_stats(), bool)
    assert isinstance(peak_vram_gb(), dict)
    assert isinstance(format_peak_vram(), str)


def test_peak_vram_reports_gb_keys_when_cuda_present():
    stats = peak_vram_gb()
    if not stats:
        # CPU-only machine: nothing to assert beyond "no crash".
        return
    assert set(stats) == {"peak_allocated_gb", "peak_reserved_gb"}
    assert all(isinstance(v, float) and v >= 0 for v in stats.values())
    assert stats["peak_reserved_gb"] >= stats["peak_allocated_gb"]


def test_format_peak_vram_mentions_allocated_when_cuda_present():
    text = format_peak_vram()
    if not cuda_available():
        assert "n/a" in text
        return
    assert "peak_allocated" in text and "peak_reserved" in text
