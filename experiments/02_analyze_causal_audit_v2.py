"""Verify and analyze a frozen causal-audit-v2 collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.audit import canonical_config_hash, load_config, sha256_file
from src.causal_audit_gates import analyze_causal_audit_v2
from src.causal_audit_v2 import verify_causal_audit_v2_problem
from src.generate_rollouts import Rollout
from src.tasks import MathProblem, read_jsonl, read_jsonl_objects


def main() -> None:
    """Bind a run to its freeze, verify certificates, and apply preregistered gates."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--partition", choices=("qualification", "confirmatory"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stem = Path("data/raw/causal_audit_v2")
    questions_path = Path(f"{stem}.{args.partition}.jsonl")
    config_path = Path(f"configs/tinker_causal_audit_v2_{args.partition}.yaml")
    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    expected = {
        "purpose": f"causal_audit_v2_{args.partition}",
        "questions": 24 if args.partition == "qualification" else 72,
        "conditions": 3,
        "samples_per_condition": 2 if args.partition == "qualification" else 3,
        "model": "openai/gpt-oss-20b",
        "config_sha256": canonical_config_hash(load_config(config_path)),
    }
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    source = manifest.get("source_questions", {})
    observed_hash = source.get("sha256") if isinstance(source, dict) else None
    if observed_hash != sha256_file(questions_path):
        mismatches["source_questions_sha256"] = {
            "expected": sha256_file(questions_path),
            "observed": observed_hash,
        }
    if mismatches:
        raise SystemExit(f"causal-audit-v2 run differs from its freeze: {mismatches}")

    questions = list(read_jsonl(questions_path, model=MathProblem))
    rollouts = list(read_jsonl(args.run / "rollouts.jsonl", model=Rollout))
    source_by_id = {
        str(row["question_id"]): row
        for row in read_jsonl_objects(Path(f"{stem}.source_certificates.jsonl"))
    }
    intervention_by_id = {
        str(row["question_id"]): row
        for row in read_jsonl_objects(Path(f"{stem}.intervention_certificates.jsonl"))
    }
    verified = all(
        row.question_id in source_by_id
        and row.question_id in intervention_by_id
        and verify_causal_audit_v2_problem(
            row, source_by_id[row.question_id], intervention_by_id[row.question_id]
        )
        for row in questions
    )
    report = analyze_causal_audit_v2(
        questions,
        rollouts,
        partition=args.partition,
        request_errors=int(manifest.get("counts", {}).get("request_errors", 0)),
        certificates_verified=verified,
        bootstrap_seed=20262682 if args.partition == "qualification" else 20262692,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
