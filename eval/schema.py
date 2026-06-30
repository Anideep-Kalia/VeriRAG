"""Benchmark row schema + loader/validator.

CLI:  python -m eval.schema [path]   -> validate all rows, print a difficulty breakdown.
"""
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

DEFAULT_PATH = Path(__file__).parent / "benchmark.jsonl"

Difficulty = Literal["single_hop", "multi_hop", "ambiguous", "out_of_scope"]


class BenchmarkRow(BaseModel):
    id: str
    question: str
    ground_truth: str
    ground_truth_contexts: list[str] = []
    difficulty: Difficulty


def load_benchmark(path=DEFAULT_PATH) -> list[BenchmarkRow]:
    path = Path(path)
    rows = [
        BenchmarkRow(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [r.id for r in rows]
    dupes = [i for i, c in Counter(ids).items() if c > 1]
    if dupes:
        raise ValueError(f"duplicate ids in benchmark: {dupes}")
    # out_of_scope rows should have no supporting context (they must abstain)
    for r in rows:
        if r.difficulty == "out_of_scope" and r.ground_truth_contexts:
            raise ValueError(f"{r.id}: out_of_scope row should have empty ground_truth_contexts")
    return rows


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    path = Path(args[0]) if args else DEFAULT_PATH
    rows = load_benchmark(path)
    breakdown = Counter(r.difficulty for r in rows)
    print(f"OK — {len(rows)} rows validated from {path}")
    for diff, n in sorted(breakdown.items()):
        print(f"  {diff:14} {n}")


if __name__ == "__main__":
    main()
