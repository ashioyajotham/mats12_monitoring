"""Compute balanced discrimination and prevalence-adjusted precision."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.metrics import evaluate_scores
from src.tasks import read_jsonl_objects


def main() -> None:
    """Evaluate stored monitor scores and create a metrics artifact."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL with binary_label and score")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fixed-recall", type=float, default=0.5)
    args = parser.parse_args()
    rows = read_jsonl_objects(args.input)
    metrics = evaluate_scores(
        [int(row["binary_label"]) for row in rows],
        [float(row["score"]) for row in rows],
        fixed_recall=args.fixed_recall,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(metrics.model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
