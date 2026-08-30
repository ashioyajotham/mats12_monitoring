"""Fit/evaluate the surface baseline on an explicitly split labelled JSONL file."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.metrics import evaluate_scores
from src.monitors.surface import SurfaceFeatureMonitor, extract_surface_features
from src.tasks import assert_no_group_leakage, read_jsonl_objects


def read_rows(path: str) -> list[dict]:
    """Read non-empty dictionary records from a JSONL input."""
    return read_jsonl_objects(path)


def main() -> None:
    """Fit the surface baseline and create a held-out metrics artifact."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = read_rows(args.input)
    assert_no_group_leakage(rows)
    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    if not train or not test:
        raise SystemExit("input must contain non-empty train and test splits")

    monitor = SurfaceFeatureMonitor().fit(
        [extract_surface_features(row["response"], row.get("parsed_answer")) for row in train],
        [int(row["binary_label"]) for row in train],
    )
    scores = monitor.predict_score(
        [extract_surface_features(row["response"], row.get("parsed_answer")) for row in test]
    )
    metrics = evaluate_scores([int(row["binary_label"]) for row in test], scores)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(metrics.model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
