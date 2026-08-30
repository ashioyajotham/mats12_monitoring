"""Freeze a combined v2.1 pilot after fresh subset replacement screening."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.audit import canonical_config_hash, load_config, sha256_file
from src.generate_rollouts import Rollout, write_manifest
from src.procedural_v21_pilot import select_combined_pilot_v21
from src.tasks import MathProblem, read_jsonl, read_jsonl_objects, write_jsonl

FROZEN_V2_RUN = Path("data/generated/tinker_procedural_v2_screening_20260830T091412Z")
FROZEN_V2_ROLLOUTS_SHA256 = "e56d237cb489bf0a82c86d59556c7cb7b71a5979c22c57f4ca618bfe6934d536"
FROZEN_V2_REPORT_SHA256 = "50374aece835d499952bd69d46e6d869a1c0b34d524a7109ecc747404888d56d"


def _rollouts(run: Path) -> list[Rollout]:
    """Read immutable rollout records from a run directory."""
    return list(read_jsonl(run / "rollouts.jsonl", model=Rollout))


def _certificates(path: Path) -> list[dict[str, object]]:
    """Read JSONL solver certificates."""
    return read_jsonl_objects(path)


def _validate_v2_evidence(run: Path, questions: Path, report: Path) -> dict:
    """Bind reuse to the exact failed v2 calibration evidence."""
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    expected = {
        "purpose": "procedural_v2_screening", "questions": 80, "conditions": 1,
        "samples_per_condition": 1, "model": "openai/gpt-oss-20b",
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise SystemExit("the supplied v2 run is not the frozen failed calibration")
    if sha256_file(run / "rollouts.jsonl") != FROZEN_V2_ROLLOUTS_SHA256:
        raise SystemExit("the v2 rollout evidence hash differs from the amendment")
    if sha256_file(report) != FROZEN_V2_REPORT_SHA256:
        raise SystemExit("the v2 screening report hash differs from the amendment")
    if manifest.get("source_questions", {}).get("sha256") != sha256_file(questions):
        raise SystemExit("the v2 source question hash differs from its run manifest")
    expected_config = canonical_config_hash(load_config("configs/tinker_procedural_v2_screen.yaml"))
    if manifest.get("config_sha256") != expected_config:
        raise SystemExit("the v2 config differs from its frozen screening plan")
    return manifest


def _validate_replacement_run(run: Path, questions: Path) -> dict:
    """Require the exact prospective 20-request replacement screen."""
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    expected = {
        "purpose": "procedural_v21_subset_screening", "questions": 20, "conditions": 1,
        "samples_per_condition": 1, "model": "openai/gpt-oss-20b",
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
        load_config("configs/tinker_procedural_v21_subset_screen.yaml")
    )
    if manifest.get("config_sha256") != expected_config:
        mismatches["config_sha256"] = {
            "expected": expected_config, "observed": manifest.get("config_sha256")}
    if mismatches:
        raise SystemExit(f"replacement run does not match the amendment: {mismatches}")
    return manifest


def main() -> None:
    """Apply both calibration gates and freeze 40 questions only on a pass."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--replacement-run", type=Path, required=True)
    parser.add_argument("--v2-run", type=Path, default=FROZEN_V2_RUN)
    parser.add_argument("--v2-questions", type=Path,
                        default=Path("data/raw/procedural_math_candidates_v2.jsonl"))
    parser.add_argument("--v2-certificates", type=Path,
                        default=Path("data/raw/procedural_math_certificates_v2.jsonl"))
    parser.add_argument("--v2-report", type=Path,
                        default=Path("data/raw/procedural_math_screening_v2.report.json"))
    parser.add_argument("--replacement-questions", type=Path,
                        default=Path("data/raw/procedural_math_subset_replacements_v21.jsonl"))
    parser.add_argument("--replacement-certificates", type=Path, default=Path(
        "data/raw/procedural_math_subset_replacements_v21.certificates.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    v2_manifest = _validate_v2_evidence(args.v2_run, args.v2_questions, args.v2_report)
    replacement_manifest = _validate_replacement_run(
        args.replacement_run, args.replacement_questions
    )
    v2_questions = list(read_jsonl(args.v2_questions, model=MathProblem))
    replacement_questions = list(read_jsonl(args.replacement_questions, model=MathProblem))
    report, selected, selected_certificates = select_combined_pilot_v21(
        v2_questions, _rollouts(args.v2_run), _certificates(args.v2_certificates),
        replacement_questions, _rollouts(args.replacement_run),
        _certificates(args.replacement_certificates),
        v2_request_errors=int(v2_manifest.get("counts", {}).get("request_errors", 0)),
        replacement_request_errors=int(
            replacement_manifest.get("counts", {}).get("request_errors", 0)
        ),
    )
    report_path = args.output_dir / "procedural_math_screening_v21.report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["selection_passed"]:
        raise SystemExit(f"v2.1 replacement screening failed; see {report_path}")
    questions_path = args.output_dir / "procedural_math_pilot_v21.jsonl"
    certificates_path = args.output_dir / "procedural_math_pilot_v21.certificates.jsonl"
    manifest_path = args.output_dir / "procedural_math_pilot_v21.manifest.json"
    write_jsonl(questions_path, selected)
    write_jsonl(certificates_path, selected_certificates)
    write_manifest(manifest_path, {
        "purpose": "procedural_math_v21_combined_pilot_freeze", "protocol": report["protocol"],
        "questions": len(selected), "v2_source_run": str(args.v2_run),
        "replacement_source_run": str(args.replacement_run),
        "v2_source_rollouts_sha256": sha256_file(args.v2_run / "rollouts.jsonl"),
        "replacement_source_rollouts_sha256": sha256_file(
            args.replacement_run / "rollouts.jsonl"),
        "selection_report_sha256": sha256_file(report_path),
        "questions_sha256": sha256_file(questions_path),
        "certificates_sha256": sha256_file(certificates_path),
        "individual_screening_outcomes_used_for_selection": False,
    })
    print(json.dumps({"questions": str(questions_path), "certificates": str(certificates_path),
                      "manifest": str(manifest_path), "count": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
