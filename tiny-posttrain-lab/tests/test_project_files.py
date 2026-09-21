"""Tiny offline checks for package import paths (no GPU)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_import_my_lab():
    import my_lab
    from my_lab import accuracy_reward, estimate_training_vram_gb

    assert callable(accuracy_reward)
    assert callable(estimate_training_vram_gb)


def test_configs_exist():
    cfg_dir = ROOT / "configs"
    assert (cfg_dir / "sft_qwen05b_8g.yaml").exists()
    assert (cfg_dir / "dpo_qwen05b_8g.yaml").exists()
    assert (cfg_dir / "grpo_qwen05b_8g.yaml").exists()


def test_math_sample_jsonl():
    path = ROOT / "data" / "math_sample.jsonl"
    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 5
