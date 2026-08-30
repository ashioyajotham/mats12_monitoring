"""Freeze a balanced diagnostic-only matched-partial-solution pilot."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src import partial_solution_interventions
from src.audit import git_revision, runtime_environment, sha256_file
from src.generate_rollouts import write_manifest
from src.partial_solution_interventions import (
    build_causal_yield_freeze,
    verify_intervention,
)
from src.tasks import MathProblem, read_jsonl, write_jsonl


def main() -> None:
    """Select 12 questions and bind matched correct/corrupted notes immutably."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path,
                        default=Path("data/raw/procedural_math_pilot_v21.jsonl"))
    parser.add_argument("--source-certificates", type=Path,
                        default=Path("data/raw/procedural_math_pilot_v21.certificates.jsonl"))
    parser.add_argument("--source-manifest", type=Path,
                        default=Path("data/raw/procedural_math_pilot_v21.manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--selection-seed", type=int, default=20261901)
    parser.add_argument("--per-family", type=int, default=3)
    args = parser.parse_args()
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if source_manifest.get("questions_sha256") != sha256_file(args.questions):
        raise SystemExit("source question hash does not match the combined-pilot manifest")
    if source_manifest.get("certificates_sha256") != sha256_file(args.source_certificates):
        raise SystemExit("source certificate hash does not match the combined-pilot manifest")
    questions = list(read_jsonl(args.questions, model=MathProblem))
    source_certificates = [
        json.loads(line)
        for line in args.source_certificates.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frozen, certificates, selection = build_causal_yield_freeze(
        questions, source_certificates,
        per_family=args.per_family, selection_seed=args.selection_seed,
    )
    if not all(
        verify_intervention(problem, certificate)
        for problem, certificate in zip(frozen, certificates, strict=True)
    ):
        raise SystemExit("a matched intervention certificate failed verification")
    questions_path = args.output_dir / "procedural_causal_yield_v1.jsonl"
    certificates_path = args.output_dir / "procedural_causal_yield_v1.certificates.jsonl"
    selection_path = args.output_dir / "procedural_causal_yield_v1.selection.json"
    manifest_path = args.output_dir / "procedural_causal_yield_v1.manifest.json"
    write_jsonl(questions_path, frozen)
    write_jsonl(certificates_path, certificates)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    with selection_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    family_counts = Counter(
        str(problem.metadata["generator_family"]) for problem in frozen
    )
    write_manifest(manifest_path, {
        "purpose": "procedural_causal_yield_v1_freeze",
        "protocol": selection["protocol"], "questions": len(frozen),
        "selection_seed": args.selection_seed, "per_family": args.per_family,
        "family_counts": dict(sorted(family_counts.items())),
        "selection_uses_clean_outcomes": False, "diagnostic_only": True,
        "excluded_from_monitor_data": True,
        "source_questions_sha256": sha256_file(args.questions),
        "source_certificates_sha256": sha256_file(args.source_certificates),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "questions_sha256": sha256_file(questions_path),
        "certificates_sha256": sha256_file(certificates_path),
        "selection_sha256": sha256_file(selection_path),
        "intervention_code_sha256": sha256_file(
            partial_solution_interventions.__file__
        ),
        "entrypoint_sha256": sha256_file(__file__), "code_revision": git_revision(),
        "runtime_environment": runtime_environment(),
    })
    print(json.dumps({"questions": str(questions_path), "certificates": str(certificates_path),
                      "selection": str(selection_path), "manifest": str(manifest_path),
                      "count": len(frozen)}, indent=2))


if __name__ == "__main__":
    main()
