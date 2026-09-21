"""Reward functions for verifiable tasks (math-style accuracy)."""

from __future__ import annotations

import re
from typing import Sequence


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_final_number(text: str) -> float | None:
    """Extract the last number in text; None if not found."""
    if not text:
        return None
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def accuracy_reward(
    completions: Sequence[str],
    answer: Sequence[str],
    **kwargs,
) -> list[float]:
    """
    TRL-style reward: 1.0 if final number matches, else 0.0.

    Args:
        completions: model outputs (strings or OpenAI-style message lists)
        answer: ground-truth answers (string or number-like)
    """
    rewards: list[float] = []
    for completion, gold in zip(completions, answer):
        if isinstance(completion, list):
            # OpenAI chat message list — take last assistant content
            text = ""
            for msg in completion:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    text = msg.get("content") or ""
                elif hasattr(msg, "get") and msg.get("role") == "assistant":
                    text = msg.get("content") or ""
            completion_text = text
        else:
            completion_text = str(completion)

        pred = extract_final_number(completion_text)
        gold_num = extract_final_number(str(gold))
        if pred is None or gold_num is None:
            rewards.append(0.0)
            continue
        rewards.append(1.0 if abs(pred - gold_num) < 1e-4 else 0.0)
    return rewards
