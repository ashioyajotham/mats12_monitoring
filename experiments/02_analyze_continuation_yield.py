"""Analyze the exact continuation-state causal-yield-v2 diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.audit import canonical_config_hash, load_config, sha256_file
from src.continuation_yield import analyze_continuation_yield
from src.generate_rollouts import Rollout
from src.tasks import MathProblem, read_jsonl


def _validated_manifest(run: Path, questions: Path) -> tuple[dict, dict]:
    """Require the exact frozen 72-request continuation collection."""
    config = load_config("configs/tinker_procedural_continuation_yield_v2.yaml")
    manifest = json.loads((run / "manifest.json").read_text())
    expected = {
        "purpose": "procedural_continuation_yield_v2",
        "questions": 8,
        "conditions": 3,
        "samples_per_condition": 3,
        "model": "openai/gpt-oss-20b",
    }
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in expected.items() if manifest.get(key) != value
    }
    source_hash = manifest.get("source_questions", {}).get("sha256")
    if source_hash != sha256_file(questions):
        mismatches["source_questions_sha256"] = {
            "expected": sha256_file(questions), "observed": source_hash}
    expected_config = canonical_config_hash(config)
    if manifest.get("config_sha256") != expected_config:
        mismatches["config_sha256"] = {
            "expected": expected_config, "observed": manifest.get("config_sha256")}
    if mismatches:
        raise SystemExit(f"continuation run does not match preregistration: {mismatches}")
    return manifest, config


def main() -> None:
    """Evaluate immutable v2 diagnostic rollouts and optionally save the report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--questions", type=Path,
                        default=Path("data/raw/procedural_continuation_yield_v2.jsonl"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    questions = list(read_jsonl(args.questions, model=MathProblem))
    rollouts = [
        Rollout.model_validate_json(line)
        for line in (args.run / "rollouts.jsonl").read_text().splitlines()
        if line.strip()
    ]
    manifest, config = _validated_manifest(args.run, args.questions)
    report = analyze_continuation_yield(
        questions,
        rollouts,
        request_errors=int(manifest.get("counts", {}).get("request_errors", 0)),
        acknowledgment_patterns=list(config["labels"]["acknowledgment_patterns"]),
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
