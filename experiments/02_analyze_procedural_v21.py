"""Analyze the fresh combined procedural-v2.1 clean mixed-outcome cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.audit import canonical_config_hash, load_config, sha256_file
from src.generate_rollouts import Rollout
from src.procedural_v21_pilot import analyze_mixed_outcome_v21
from src.tasks import MathProblem, read_jsonl


def _validated_manifest(run: Path, questions: Path) -> dict:
    """Require the exact fresh v2.1 discovery design and source freeze."""
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    expected = {
        "purpose": "procedural_v21_clean_mixed_outcome", "questions": 40,
        "conditions": 1, "samples_per_condition": 3, "model": "openai/gpt-oss-20b",
    }
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in expected.items() if manifest.get(key) != value
    }
    source_hash = manifest.get("source_questions", {}).get("sha256")
    if source_hash != sha256_file(questions):
        mismatches["source_questions_sha256"] = {
            "expected": sha256_file(questions), "observed": source_hash}
    expected_config = canonical_config_hash(
        load_config("configs/tinker_procedural_v21_discovery.yaml")
    )
    if manifest.get("config_sha256") != expected_config:
        mismatches["config_sha256"] = {
            "expected": expected_config, "observed": manifest.get("config_sha256")}
    if mismatches:
        raise SystemExit(f"v2.1 discovery manifest does not match: {mismatches}")
    return manifest


def main() -> None:
    """Evaluate the immutable fresh cohort and optionally create a result artifact."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--questions", type=Path,
                        default=Path("data/raw/procedural_math_pilot_v21.jsonl"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    questions = list(read_jsonl(args.questions, model=MathProblem))
    rollouts = list(read_jsonl(args.run / "rollouts.jsonl", model=Rollout))
    manifest = _validated_manifest(args.run, args.questions)
    report = analyze_mixed_outcome_v21(
        questions, rollouts,
        request_errors=int(manifest.get("counts", {}).get("request_errors", 0)),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
