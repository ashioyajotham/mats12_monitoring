"""Freeze a procedural pilot using preregistered cell-level screening outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.audit import canonical_config_hash, load_config, sha256_file
from src.generate_rollouts import Rollout, write_manifest
from src.procedural_pilot import select_screened_questions
from src.tasks import MathProblem, read_jsonl, read_jsonl_objects, write_jsonl


def _validated_manifest(run: Path, questions: Path) -> dict:
    """Validate that the collection manifest describes the frozen screening plan."""
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    expected = {
        "purpose": "procedural_screening",
        "questions": 120,
        "conditions": 1,
        "samples_per_condition": 1,
        "model": "openai/gpt-oss-20b",
    }
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    source_hash = manifest.get("source_questions", {}).get("sha256")
    if source_hash != sha256_file(questions):
        mismatches["source_questions_sha256"] = {
            "expected": sha256_file(questions),
            "observed": source_hash,
        }
    expected_config_hash = canonical_config_hash(
        load_config("configs/tinker_procedural_screen.yaml")
    )
    if manifest.get("config_sha256") != expected_config_hash:
        mismatches["config_sha256"] = {
            "expected": expected_config_hash,
            "observed": manifest.get("config_sha256"),
        }
    if mismatches:
        raise SystemExit(f"screening run manifest does not match the freeze: {mismatches}")
    return manifest


def main() -> None:
    """Analyze one-rollout screening cells and write a 40-question freeze if eligible."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
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
    parser.add_argument("--selection-seed", type=int, default=20260901)
    args = parser.parse_args()

    questions = list(read_jsonl(args.questions, model=MathProblem))
    rollouts = list(read_jsonl(args.run / "rollouts.jsonl", model=Rollout))
    certificates = read_jsonl_objects(args.certificates)
    manifest = _validated_manifest(args.run, args.questions)
    report, selected, selected_certificates = select_screened_questions(
        questions,
        rollouts,
        certificates,
        selection_seed=args.selection_seed,
        request_errors=int(manifest.get("counts", {}).get("request_errors", 0)),
    )
    report_path = args.output_dir / "procedural_math_screening_v1.report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["selection_passed"]:
        raise SystemExit(f"screening calibration failed; see {report_path}")

    questions_path = args.output_dir / "procedural_math_pilot_v1.jsonl"
    certificates_path = args.output_dir / "procedural_math_pilot_v1.certificates.jsonl"
    manifest_path = args.output_dir / "procedural_math_pilot_v1.manifest.json"
    write_jsonl(questions_path, selected)
    write_jsonl(certificates_path, selected_certificates)
    write_manifest(
        manifest_path,
        {
            "purpose": "procedural_math_pilot_freeze",
            "protocol": report["protocol"],
            "selection_seed": args.selection_seed,
            "questions": len(selected),
            "source_run": str(args.run),
            "source_rollouts_sha256": sha256_file(args.run / "rollouts.jsonl"),
            "source_questions_sha256": sha256_file(args.questions),
            "source_certificates_sha256": sha256_file(args.certificates),
            "screening_report_sha256": sha256_file(report_path),
            "questions_sha256": sha256_file(questions_path),
            "certificates_sha256": sha256_file(certificates_path),
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
