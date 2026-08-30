"""Materialize causal-error monitor examples after a passed confirmatory gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.audit import git_revision, runtime_environment, sha256_file
from src.causal_error_dataset import verify_causal_error_problem
from src.causal_error_gates import analyze_causal_error_confirmatory
from src.generate_rollouts import Rollout, write_manifest
from src.monitor_dataset import materialize_monitor_examples
from src.tasks import MathProblem, read_jsonl, read_jsonl_objects, write_jsonl


def main() -> None:
    """Verify gate/run binding and create immutable primary and audit artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
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
    parser.add_argument("--output-dir", type=Path, default=Path("data/reviewed"))
    parser.add_argument("--secondary-cap", type=int, default=96)
    parser.add_argument("--secondary-seed", type=int, default=20262603)
    args = parser.parse_args()

    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rollouts_path = args.run / "rollouts.jsonl"
    if manifest.get("output_sha256") != sha256_file(rollouts_path):
        raise SystemExit("confirmatory rollout hash differs from its run manifest")
    if manifest.get("source_questions", {}).get("sha256") != sha256_file(args.questions):
        raise SystemExit("confirmatory question hash differs from its run manifest")
    if report.get("stored_rollouts") != manifest.get("counts", {}).get("completed"):
        raise SystemExit("confirmatory report count differs from its run manifest")
    if report.get("gate_checks", {}).get(
        "all_source_and_intervention_certificates_verified"
    ) is not True:
        raise SystemExit("confirmatory report lacks verified causal certificates")

    questions = list(read_jsonl(args.questions, model=MathProblem))
    rollouts = list(read_jsonl(rollouts_path, model=Rollout))
    source_by_id = {
        str(row["question_id"]): row
        for row in read_jsonl_objects(args.source_certificates)
    }
    intervention_by_id = {
        str(row["question_id"]): row
        for row in read_jsonl_objects(args.intervention_certificates)
    }
    certificates_verified = all(
        question.question_id in source_by_id
        and question.question_id in intervention_by_id
        and verify_causal_error_problem(
            question,
            source_by_id[question.question_id],
            intervention_by_id[question.question_id],
        )
        for question in questions
    )
    recomputed_report = analyze_causal_error_confirmatory(
        questions,
        rollouts,
        request_errors=int(manifest.get("counts", {}).get("request_errors", 0)),
        certificates_verified=certificates_verified,
    )
    if recomputed_report != report:
        raise SystemExit("provided confirmatory report does not match a fresh gate recomputation")
    primary, audit, summary = materialize_monitor_examples(
        questions,
        rollouts,
        report,
        secondary_cap=args.secondary_cap,
        secondary_seed=args.secondary_seed,
    )
    primary_path = args.output_dir / "causal_error_v1.primary.jsonl"
    audit_path = args.output_dir / "causal_error_v1.secondary_audit.jsonl"
    summary_path = args.output_dir / "causal_error_v1.monitor_dataset.json"
    manifest_path = args.output_dir / "causal_error_v1.monitor_dataset.manifest.json"
    write_jsonl(primary_path, primary)
    write_jsonl(audit_path, audit)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_manifest(
        manifest_path,
        {
            **summary,
            "purpose": "causal_error_v1_monitor_dataset",
            "confirmatory_run": str(args.run),
            "confirmatory_rollouts_sha256": sha256_file(rollouts_path),
            "confirmatory_manifest_sha256": sha256_file(args.run / "manifest.json"),
            "confirmatory_report_sha256": sha256_file(args.report),
            "questions_sha256": sha256_file(args.questions),
            "source_certificates_sha256": sha256_file(args.source_certificates),
            "intervention_certificates_sha256": sha256_file(
                args.intervention_certificates
            ),
            "primary_path": str(primary_path),
            "primary_sha256": sha256_file(primary_path),
            "secondary_audit_path": str(audit_path),
            "secondary_audit_sha256": sha256_file(audit_path),
            "summary_sha256": sha256_file(summary_path),
            "entrypoint_sha256": sha256_file(__file__),
            "code_revision": git_revision(),
            "runtime_environment": runtime_environment(),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
