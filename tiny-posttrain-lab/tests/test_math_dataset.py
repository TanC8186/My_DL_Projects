"""CPU-only checks for the math dataset generator.

The generator is the ground truth for the verifiable reward, so its two promises
have to hold: answers are arithmetically right, and the train/eval splits never
overlap (an earlier version used two independent seeds and leaked 16 prompts).
"""

import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_math_dataset import GENERATORS, build, split  # noqa: E402

NUM = r"(\d+)"


def solve(prompt: str) -> int | None:
    """Re-derive the answer from the question text, independently of the generator."""
    patterns = [
        (rf"What is {NUM} \+ {NUM}\?", lambda m: int(m[0]) + int(m[1])),
        (rf"What is {NUM} - {NUM}\?", lambda m: int(m[0]) - int(m[1])),
        (rf"Compute {NUM} \* {NUM}\.", lambda m: int(m[0]) * int(m[1])),
        (rf"Compute {NUM} / {NUM}\.", lambda m: int(m[0]) // int(m[1])),
        (
            rf"A book has {NUM} pages\. You read {NUM}\. How many pages are left\?",
            lambda m: int(m[0]) - int(m[1]),
        ),
        (
            rf"There are {NUM} bags with {NUM} apples each\. How many apples in total\?",
            lambda m: int(m[0]) * int(m[1]),
        ),
        (
            rf"A train travels {NUM} km/h for {NUM} hours\. What distance does it cover\?",
            lambda m: int(m[0]) * int(m[1]),
        ),
        (
            rf"A pen costs {NUM} yuan\. How much do {NUM} pens cost\?",
            lambda m: int(m[0]) * int(m[1]),
        ),
    ]
    for pattern, fn in patterns:
        match = re.search(pattern, prompt)
        if match:
            return fn([int(g) for g in match.groups()])
    return None


def test_generator_is_deterministic():
    assert build(20, seed=7) == build(20, seed=7)
    assert build(20, seed=7) != build(20, seed=8)


def test_prompts_are_unique():
    rows = build(150, seed=42)
    prompts = [r["prompt"] for r in rows]
    assert len(set(prompts)) == len(prompts)


def test_every_answer_is_arithmetically_correct():
    rows = build(200, seed=42)
    unchecked = 0
    for row in rows:
        expected = solve(row["prompt"])
        if expected is None:
            unchecked += 1
            continue
        assert row["answer"] == str(expected), f"{row['prompt']} -> {row['answer']} != {expected}"
    # Every prompt shape should be covered by `solve`; otherwise this test is vacuous.
    assert unchecked == 0, f"{unchecked} prompts were not re-derivable by the test"


def test_splits_do_not_overlap():
    pool = build(300, seed=42)
    eval_rows, train_rows = split(pool, 100, seed=42)

    assert len(eval_rows) == 100
    assert len(train_rows) == 200
    eval_prompts = {r["prompt"] for r in eval_rows}
    train_prompts = {r["prompt"] for r in train_rows}
    assert not (eval_prompts & train_prompts), "train/eval overlap leaks the eval set"


def test_split_is_deterministic_and_covers_the_pool():
    pool = build(60, seed=1)
    a_eval, a_train = split(pool, 20, seed=1)
    b_eval, b_train = split(pool, 20, seed=1)
    assert a_eval == b_eval and a_train == b_train
    assert {r["prompt"] for r in a_eval + a_train} == {r["prompt"] for r in pool}


def test_no_eval_split_returns_the_whole_pool():
    pool = build(30, seed=3)
    eval_rows, train_rows = split(pool, 0, seed=3)
    assert eval_rows == []
    assert len(train_rows) == 30


def test_solve_covers_every_generator():
    """Guards the test's own coverage: if a generator's wording drifts, `solve`
    stops recognising it and `test_every_answer_is_arithmetically_correct` would
    silently skip that whole task family."""
    for fn in GENERATORS:
        question, answer = fn(random.Random(0))
        derived = solve(f"{question} End your response with the number only.")
        assert derived is not None, f"test parser does not recognise: {question!r}"
        assert derived == answer, f"{question!r}: parser got {derived}, generator says {answer}"
