"""Analyze the frozen causal-error-v1 confirmatory collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.audit import canonical_config_hash, load_config, sha256_file
from src.causal_error_dataset import verify_causal_error_problem
from src.causal_error_gates import analyze_causal_error_confirmatory
from src.generate_rollouts import Rollout
from src.tasks import MathProblem, read_jsonl, read_jsonl_objects


def _validated_manifest(run: Path, questions: Path) -> dict[str, object]:
    """Bind analysis to the exact frozen 648-request collection."""
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    expected = {
        "purpose": "causal_error_v1_confirmatory",
        "questions": 72,
        "conditions": 3,
        "samples_per_condition": 3,
        "model": "openai/gpt-oss-20b",
    }
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    source = manifest.get("source_questions", {})
    observed_source = source.get("sha256") if isinstance(source, dict) else None
    if observed_source != sha256_file(questions):
        mismatches["source_questions_sha256"] = {
            "expected": sha256_file(questions),
            "observed": observed_source,
        }
    expected_config = canonical_config_hash(
        load_config("configs/tinker_causal_error_v1_confirmatory.yaml")
    )
    if manifest.get("config_sha256") != expected_config:
        mismatches["config_sha256"] = {
            "expected": expected_config,
            "observed": manifest.get("config_sha256"),
        }
    if mismatches:
        raise SystemExit(f"confirmatory run differs from its freeze: {mismatches}")
    return manifest


def main() -> None:
    """Verify causal certificates, apply all gates, and optionally write a report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/raw/causal_error_detection_v1.confirmatory.jsonl"),
    )
    parser.add_argument(
        "--source-certificates",
        type=Path,
        default=Path("data/raw/causal_error_detection_v1.source_certificates.jsonl"),
    )
    parser.add_argument(
        "--intervention-certificates",
        type=Path,
        default=Path("data/raw/causal_error_detection_v1.intervention_certificates.jsonl"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = _validated_manifest(args.run, args.questions)
    questions = list(read_jsonl(args.questions, model=MathProblem))
    rollouts = list(read_jsonl(args.run / "rollouts.jsonl", model=Rollout))
    source_by_id = {
        str(row["question_id"]): row
        for row in read_jsonl_objects(args.source_certificates)
    }
    intervention_by_id = {
        str(row["question_id"]): row
        for row in read_jsonl_objects(args.intervention_certificates)
    }
    certificates_verified = all(
        problem.question_id in source_by_id
        and problem.question_id in intervention_by_id
        and verify_causal_error_problem(
            problem,
            source_by_id[problem.question_id],
            intervention_by_id[problem.question_id],
        )
        for problem in questions
    )
    report = analyze_causal_error_confirmatory(
        questions,
        rollouts,
        request_errors=int(manifest.get("counts", {}).get("request_errors", 0)),
        certificates_verified=certificates_verified,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
