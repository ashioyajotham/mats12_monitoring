"""Freeze the mechanism-held-out causal-audit-v2 question bank."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src import causal_audit_v2
from src.audit import git_revision, runtime_environment, sha256_file
from src.causal_audit_v2 import (
    CAUSAL_AUDIT_VERSION,
    build_causal_audit_v2,
    verify_causal_audit_v2_problem,
)
from src.generate_rollouts import write_manifest
from src.tasks import read_jsonl_objects, write_jsonl

DEFAULT_PRIOR_FILES = (
    "data/raw/procedural_math_candidates_v1.jsonl",
    "data/raw/procedural_math_candidates_v2.jsonl",
    "data/raw/procedural_math_subset_replacements_v21.jsonl",
    "data/raw/procedural_math_pilot_v21.jsonl",
    "data/raw/procedural_low_reasoning_diagnostic_v1.jsonl",
    "data/raw/procedural_causal_yield_v1.jsonl",
    "data/raw/procedural_continuation_yield_v2.jsonl",
    "data/raw/procedural_assistant_prefill_v3.jsonl",
    "data/raw/causal_error_detection_v1.jsonl",
)


def _prior_ids(paths: list[Path]) -> tuple[set[str], dict[str, str]]:
    """Return all excluded IDs and content hashes for their source files."""
    ids: set[str] = set()
    hashes: dict[str, str] = {}
    for path in paths:
        ids.update(str(row["question_id"]) for row in read_jsonl_objects(path))
        hashes[str(path)] = sha256_file(path)
    return ids, hashes


def main() -> None:
    """Write immutable qualification/external-test banks and exact certificates."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--root-seed", type=int, default=20262680)
    parser.add_argument("--prior-question-file", action="append", type=Path, dest="prior_files")
    args = parser.parse_args()
    prior_files = args.prior_files or [Path(value) for value in DEFAULT_PRIOR_FILES]
    prior_ids, prior_hashes = _prior_ids(prior_files)
    problems, sources, interventions, selection = build_causal_audit_v2(
        root_seed=args.root_seed, excluded_question_ids=prior_ids
    )
    source_by_id = {str(row["question_id"]): row for row in sources}
    intervention_by_id = {str(row["question_id"]): row for row in interventions}
    if not all(
        verify_causal_audit_v2_problem(
            problem, source_by_id[problem.question_id], intervention_by_id[problem.question_id]
        )
        for problem in problems
    ):
        raise SystemExit("a causal-audit-v2 certificate failed exact verification")

    qualification = [row for row in problems if row.metadata["study_partition"] == "qualification"]
    confirmatory = [row for row in problems if row.metadata["study_partition"] == "confirmatory"]
    stem = args.output_dir / "causal_audit_v2"
    paths = {
        "all_questions": Path(f"{stem}.jsonl"),
        "qualification_questions": Path(f"{stem}.qualification.jsonl"),
        "confirmatory_questions": Path(f"{stem}.confirmatory.jsonl"),
        "source_certificates": Path(f"{stem}.source_certificates.jsonl"),
        "intervention_certificates": Path(f"{stem}.intervention_certificates.jsonl"),
        "selection": Path(f"{stem}.selection.json"),
        "manifest": Path(f"{stem}.manifest.json"),
    }
    write_jsonl(paths["all_questions"], problems)
    write_jsonl(paths["qualification_questions"], qualification)
    write_jsonl(paths["confirmatory_questions"], confirmatory)
    write_jsonl(paths["source_certificates"], sources)
    write_jsonl(paths["intervention_certificates"], interventions)
    paths["selection"].parent.mkdir(parents=True, exist_ok=True)
    with paths["selection"].open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(selection, indent=2, sort_keys=True) + "\n")

    artifact_hashes = {key: sha256_file(path) for key, path in paths.items() if key != "manifest"}
    cell_counts = Counter(
        (
            str(row.metadata["study_partition"]),
            str(row.metadata["generator_family"]),
            str(row.metadata["corruption_mechanism"]),
        )
        for row in problems
    )
    write_manifest(
        paths["manifest"],
        {
            "purpose": "causal_audit_v2_freeze",
            "protocol": CAUSAL_AUDIT_VERSION,
            "license": "MIT",
            "root_seed": args.root_seed,
            "questions": 96,
            "qualification_questions": 24,
            "confirmatory_questions": 72,
            "planned_tinker_calls": 792,
            "cell_counts": {
                f"{partition}:{family}:{mechanism}": count
                for (partition, family, mechanism), count in sorted(cell_counts.items())
            },
            "selection_uses_model_outcomes": False,
            "all_partitions_frozen_before_collection": True,
            "confirmatory_is_external_test_only": True,
            "prior_question_files": prior_hashes,
            "prior_question_ids_excluded": len(prior_ids),
            "artifact_paths": {key: str(path) for key, path in paths.items()},
            "artifact_sha256": artifact_hashes,
            "all_certificates_verified": True,
            "preregistration_sha256": sha256_file("docs/PREREGISTRATION_CAUSAL_AUDIT_V2.md"),
            "qualification_config_sha256": sha256_file(
                "configs/tinker_causal_audit_v2_qualification.yaml"
            ),
            "confirmatory_config_sha256": sha256_file(
                "configs/tinker_causal_audit_v2_confirmatory.yaml"
            ),
            "generator_code_sha256": sha256_file(causal_audit_v2.__file__),
            "entrypoint_sha256": sha256_file(__file__),
            "code_revision": git_revision(),
            "runtime_environment": runtime_environment(),
        },
    )
    print(json.dumps({"questions": 96, "qualification": 24, "confirmatory": 72}, indent=2))


if __name__ == "__main__":
    main()
