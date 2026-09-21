"""Generate a reproducible arithmetic dataset for GRPO / verifiable-reward work.

The point is a reward that can be checked exactly, so answers are computed in
Python rather than written by hand.

    python scripts/make_math_dataset.py --n 300 --eval-n 100 \
        --out data/math_train.jsonl --eval-out data/math_eval.jsonl

Both splits are drawn from one shuffled pool, so they are guaranteed disjoint.
`--seed` fixes the shuffle, so the files are reproducible from the command line
recorded in `data/README.md`.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

SUFFIX = "End your response with the number only."


def _add2(rng: random.Random) -> tuple[str, int]:
    a, b = rng.randint(11, 99), rng.randint(11, 99)
    return f"What is {a} + {b}?", a + b


def _sub2(rng: random.Random) -> tuple[str, int]:
    a, b = rng.randint(11, 99), rng.randint(11, 99)
    if a < b:
        a, b = b, a
    return f"What is {a} - {b}?", a - b


def _mul1x2(rng: random.Random) -> tuple[str, int]:
    a, b = rng.randint(3, 9), rng.randint(11, 49)
    return f"Compute {a} * {b}.", a * b


def _mul2x2(rng: random.Random) -> tuple[str, int]:
    a, b = rng.randint(11, 29), rng.randint(11, 29)
    return f"Compute {a} * {b}.", a * b


def _div_exact(rng: random.Random) -> tuple[str, int]:
    b = rng.randint(3, 12)
    q = rng.randint(3, 30)
    return f"Compute {b * q} / {b}.", q


def _word_pages(rng: random.Random) -> tuple[str, int]:
    total, read = rng.randint(80, 400), rng.randint(10, 70)
    return (
        f"A book has {total} pages. You read {read}. How many pages are left?",
        total - read,
    )


def _word_bags(rng: random.Random) -> tuple[str, int]:
    bags, each = rng.randint(3, 12), rng.randint(4, 15)
    return (
        f"There are {bags} bags with {each} apples each. How many apples in total?",
        bags * each,
    )


def _word_speed(rng: random.Random) -> tuple[str, int]:
    speed, hours = rng.choice([40, 50, 60, 70, 80]), rng.randint(2, 6)
    return (
        f"A train travels {speed} km/h for {hours} hours. What distance does it cover?",
        speed * hours,
    )


def _word_price(rng: random.Random) -> tuple[str, int]:
    price, count = rng.randint(3, 25), rng.randint(4, 20)
    return (
        f"A pen costs {price} yuan. How much do {count} pens cost?",
        price * count,
    )


GENERATORS = [
    _add2,
    _sub2,
    _mul1x2,
    _mul2x2,
    _div_exact,
    _word_pages,
    _word_bags,
    _word_speed,
    _word_price,
]


def build(n: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    guard = 0
    while len(rows) < n and guard < n * 100:
        guard += 1
        question, answer = rng.choice(GENERATORS)(rng)
        if question in seen:
            continue
        seen.add(question)
        rows.append({"prompt": f"{question} {SUFFIX}", "answer": str(answer)})
    if len(rows) < n:
        raise RuntimeError(f"Only generated {len(rows)} unique rows out of {n}")
    return rows


def split(pool: list[dict[str, str]], eval_n: int, seed: int) -> tuple[list, list]:
    """Deterministically split a pool into (eval_rows, train_rows).

    Shuffling one pool before splitting is what guarantees the two files cannot
    overlap; generating them with two independent seeds does not.
    """
    shuffled = list(pool)
    random.Random(seed + 1).shuffle(shuffled)
    if not eval_n:
        return [], shuffled
    return shuffled[:eval_n], shuffled[eval_n:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a verifiable math dataset")
    parser.add_argument("--n", type=int, default=200, help="total rows in the pool")
    parser.add_argument(
        "--eval-n",
        type=int,
        default=0,
        help="hold out this many rows as an eval split (requires --eval-out)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default=str(_ROOT / "data" / "math_train.jsonl"),
        help="output jsonl path for the train split",
    )
    parser.add_argument(
        "--eval-out",
        default=None,
        help="output jsonl path for the held-out eval split",
    )
    args = parser.parse_args()

    if args.eval_n and not args.eval_out:
        parser.error("--eval-n requires --eval-out")
    if args.eval_n >= args.n:
        parser.error("--eval-n must be smaller than --n")

    pool = build(args.n, args.seed)
    eval_rows, train_rows = split(pool, args.eval_n, args.seed)

    _write(Path(args.out), train_rows)
    print(f"[done] train: {len(train_rows)} rows -> {args.out}")

    if eval_rows:
        _write(Path(args.eval_out), eval_rows)
        print(f"[done] eval : {len(eval_rows)} rows -> {args.eval_out}")

    for row in train_rows[:3]:
        print(f"  {row['prompt']}  => {row['answer']}")


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
