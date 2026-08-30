"""Freeze a diagnostic-only cohort for reasoning-effort attribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import procedural_pilot
from src.audit import (
    canonical_config_hash,
    git_revision,
    load_config,
    runtime_environment,
    sha256_file,
)
from src.generate_rollouts import Rollout, write_manifest
from src.procedural_pilot import build_reasoning_effort_diagnostic
from src.tasks import MathProblem, read_jsonl, write_jsonl


def _validate_screening_manifest(run: Path, questions: Path) -> dict:
    """Require the exact failed medium-reasoning screening configuration."""
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    expected_hash = canonical_config_hash(load_config("configs/tinker_procedural_screen.yaml"))
    checks = {
        "purpose": manifest.get("purpose") == "procedural_screening",
        "questions": manifest.get("questions") == 120,
        "samples": manifest.get("samples_per_condition") == 1,
        "model": manifest.get("model") == "openai/gpt-oss-20b",
        "config": manifest.get("config_sha256") == expected_hash,
        "source": manifest.get("source_questions", {}).get("sha256")
        == sha256_file(questions),
        "complete": manifest.get("counts", {}).get("completed") == 120,
        "zero_request_errors": manifest.get("counts", {}).get("request_errors") == 0,
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise SystemExit(f"screening manifest failed diagnostic source checks: {failed}")
    return manifest


def main() -> None:
    """Select matched prior-truncation and clean-control questions immutably."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--screening-run", type=Path, required=True)
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/raw/procedural_math_candidates_v1.jsonl"),
    )
    parser.add_argument(
        "--certificates",
        type=Path,
        default=Path("data/raw/procedural_math_certificates_v1.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--selection-seed", type=int, default=20261101)
    args = parser.parse_args()

    manifest = _validate_screening_manifest(args.screening_run, args.questions)
    questions = list(read_jsonl(args.questions, model=MathProblem))
    rollouts = [
        Rollout.model_validate_json(line)
        for line in (args.screening_run / "rollouts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    certificates = [
        json.loads(line)
        for line in args.certificates.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report, selected, selected_certificates = build_reasoning_effort_diagnostic(
        questions,
        rollouts,
        certificates,
        selection_seed=args.selection_seed,
    )

    questions_path = args.output_dir / "procedural_low_reasoning_diagnostic_v1.jsonl"
    certificates_path = (
        args.output_dir / "procedural_low_reasoning_diagnostic_v1.certificates.jsonl"
    )
    manifest_path = args.output_dir / "procedural_low_reasoning_diagnostic_v1.manifest.json"
    write_jsonl(questions_path, selected)
    write_jsonl(certificates_path, selected_certificates)
    write_manifest(
        manifest_path,
        {
            **report,
            "purpose": "procedural_low_reasoning_diagnostic_freeze",
            "source_screening_run": str(args.screening_run),
            "source_screening_manifest_sha256": manifest["manifest_sha256"],
            "source_screening_rollouts_sha256": sha256_file(
                args.screening_run / "rollouts.jsonl"
            ),
            "source_questions_sha256": sha256_file(args.questions),
            "source_certificates_sha256": sha256_file(args.certificates),
            "questions_path": str(questions_path),
            "questions_sha256": sha256_file(questions_path),
            "certificates_path": str(certificates_path),
            "certificates_sha256": sha256_file(certificates_path),
            "code_revision": git_revision(),
            "selection_code_sha256": sha256_file(procedural_pilot.__file__),
            "entrypoint_sha256": sha256_file(__file__),
            "runtime_environment": runtime_environment(),
        },
    )
    print(
        json.dumps(
            {
                "questions": str(questions_path),
                "certificates": str(certificates_path),
                "manifest": str(manifest_path),
                "count": len(selected),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
