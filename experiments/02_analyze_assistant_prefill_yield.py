"""Analyze the frozen assistant-prefill hidden-influence diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.assistant_prefill_yield import analyze_assistant_prefill_yield
from src.audit import canonical_config_hash, load_config, sha256_file
from src.generate_rollouts import Rollout
from src.tasks import MathProblem, read_jsonl


def main() -> None:
    """Validate the run identity, compute gates, and optionally save a report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--questions", type=Path,
        default=Path("data/raw/procedural_assistant_prefill_v3.jsonl"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config("configs/tinker_procedural_assistant_prefill_v3.yaml")
    manifest = json.loads((args.run / "manifest.json").read_text())
    expected = {
        "purpose": "procedural_assistant_prefill_v3",
        "questions": 8,
        "conditions": 3,
        "samples_per_condition": 3,
        "model": "openai/gpt-oss-20b",
        "config_sha256": canonical_config_hash(config),
    }
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in expected.items() if manifest.get(key) != value
    }
    observed_source = manifest.get("source_questions", {}).get("sha256")
    expected_source = sha256_file(args.questions)
    if observed_source != expected_source:
        mismatches["source_questions_sha256"] = {
            "expected": expected_source, "observed": observed_source
        }
    if mismatches:
        raise SystemExit(f"assistant-prefill run differs from freeze: {mismatches}")
    questions = list(read_jsonl(args.questions, model=MathProblem))
    rollouts = [
        Rollout.model_validate_json(line)
        for line in (args.run / "rollouts.jsonl").read_text().splitlines()
        if line.strip()
    ]
    report = analyze_assistant_prefill_yield(
        questions,
        rollouts,
        request_errors=int(manifest.get("counts", {}).get("request_errors", 0)),
        resistance_patterns=list(config["labels"]["resistance_patterns"]),
        bootstrap_samples=int(config["evaluation"]["bootstrap_samples"]),
        bootstrap_seed=int(config["evaluation"]["bootstrap_seed"]),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
