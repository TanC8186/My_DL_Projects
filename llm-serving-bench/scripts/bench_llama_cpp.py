"""Skeleton entrypoint: llama.cpp GGUF bench (implement when you open Part 2)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompts(path: str | Path) -> list[str]:
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompts.append(row["prompt"])
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="llama.cpp bench skeleton")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "llama_cpp_qwen05b.yaml"),
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    gguf = Path(cfg["model"]["gguf_path"])
    prompts_file = Path(cfg["bench"]["prompts_file"])
    print(f"[skeleton] experiment={cfg.get('experiment_id')}")
    print(f"[skeleton] gguf exists={gguf.exists()} path={gguf}")
    print(f"[skeleton] prompts exists={prompts_file.exists()} n={len(load_prompos_safe(prompts_file))}")
    print("[todo] Implement generation + timing in Part 2.")
    print("[todo] Suggested metrics: tokens/s, latency, peak VRAM (nvidia-smi).")
    print("[todo] See references/llama.cpp/examples and docs for llama-server / CLI.")
    print(f"[info] start time {time.strftime('%Y-%m-%d %H:%M:%S')}")


def load_prompos_safe(path: Path) -> list[str]:
    if not path.exists():
        return []
    return load_prompts(path)


if __name__ == "__main__":
    main()
