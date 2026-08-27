"""Derive counterfactual answer-shift evidence from immutable rollouts."""

from __future__ import annotations

import argparse

from src.generate_rollouts import Rollout
from src.resampling import estimate_answer_shifts
from src.tasks import read_jsonl, write_jsonl


def main() -> None:
    """Create per-question, per-hinted-option answer-shift evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rollouts = read_jsonl(args.input, model=Rollout)
    write_jsonl(args.output, estimate_answer_shifts(rollouts))


if __name__ == "__main__":
    main()
