"""Freeze unused questions with exact continuation-state interventions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src import continuation_interventions
from src.audit import git_revision, runtime_environment, sha256_file
from src.continuation_interventions import (
    build_continuation_freeze,
    verify_continuation,
)
from src.generate_rollouts import write_manifest
from src.tasks import MathProblem, read_jsonl, write_jsonl


def _jsonl(path: Path) -> list[dict[str, object]]:
    """Read nonempty JSONL objects."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    """Create an immutable eight-question continuation-yield pilot."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path,
                        default=Path("data/raw/procedural_math_pilot_v21.jsonl"))
    parser.add_argument("--source-certificates", type=Path,
                        default=Path("data/raw/procedural_math_pilot_v21.certificates.jsonl"))
    parser.add_argument("--source-manifest", type=Path,
                        default=Path("data/raw/procedural_math_pilot_v21.manifest.json"))
    parser.add_argument("--prior-pilot", type=Path,
                        default=Path("data/raw/procedural_causal_yield_v1.jsonl"))
    parser.add_argument("--prior-manifest", type=Path,
                        default=Path("data/raw/procedural_causal_yield_v1.manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--selection-seed", type=int, default=20262101)
    parser.add_argument("--per-family", type=int, default=2)
    args = parser.parse_args()
    source_manifest = json.loads(args.source_manifest.read_text())
    prior_manifest = json.loads(args.prior_manifest.read_text())
    if source_manifest.get("questions_sha256") != sha256_file(args.questions):
        raise SystemExit("combined source question hash differs from its manifest")
    if source_manifest.get("certificates_sha256") != sha256_file(args.source_certificates):
        raise SystemExit("combined source certificate hash differs from its manifest")
    if prior_manifest.get("questions_sha256") != sha256_file(args.prior_pilot):
        raise SystemExit("prior diagnostic question hash differs from its manifest")
    questions = list(read_jsonl(args.questions, model=MathProblem))
    prior = list(read_jsonl(args.prior_pilot, model=MathProblem))
    source_certificates = _jsonl(args.source_certificates)
    source_certificate_by_id = {
        str(row["question_id"]): row for row in source_certificates
    }
    frozen, certificates, selection = build_continuation_freeze(
        questions,
        source_certificates,
        excluded_question_ids={row.question_id for row in prior},
        per_family=args.per_family,
        selection_seed=args.selection_seed,
    )
    if not all(
        verify_continuation(
            problem,
            certificate,
            source_certificate_by_id[problem.question_id],
        )
        for problem, certificate in zip(frozen, certificates, strict=True)
    ):
        raise SystemExit("a continuation certificate failed exact propagation verification")
    questions_path = args.output_dir / "procedural_continuation_yield_v2.jsonl"
    certificates_path = (
        args.output_dir / "procedural_continuation_yield_v2.certificates.jsonl"
    )
    selection_path = args.output_dir / "procedural_continuation_yield_v2.selection.json"
    manifest_path = args.output_dir / "procedural_continuation_yield_v2.manifest.json"
    write_jsonl(questions_path, frozen)
    write_jsonl(certificates_path, certificates)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    with selection_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    family_counts = Counter(
        str(problem.metadata["generator_family"]) for problem in frozen
    )
    write_manifest(manifest_path, {
        "purpose": "procedural_continuation_yield_v2_freeze",
        "protocol": selection["protocol"], "questions": len(frozen),
        "selection_seed": args.selection_seed, "per_family": args.per_family,
        "family_counts": dict(sorted(family_counts.items())),
        "selection_uses_clean_outcomes": False, "diagnostic_only": True,
        "excluded_from_monitor_data": True,
        "source_questions_sha256": sha256_file(args.questions),
        "source_certificates_sha256": sha256_file(args.source_certificates),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "prior_pilot_sha256": sha256_file(args.prior_pilot),
        "prior_manifest_sha256": sha256_file(args.prior_manifest),
        "questions_sha256": sha256_file(questions_path),
        "certificates_sha256": sha256_file(certificates_path),
        "selection_sha256": sha256_file(selection_path),
        "intervention_code_sha256": sha256_file(continuation_interventions.__file__),
        "entrypoint_sha256": sha256_file(__file__), "code_revision": git_revision(),
        "runtime_environment": runtime_environment(),
    })
    print(json.dumps({"questions": str(questions_path), "certificates": str(certificates_path),
                      "selection": str(selection_path), "manifest": str(manifest_path),
                      "count": len(frozen)}, indent=2))


if __name__ == "__main__":
    main()
