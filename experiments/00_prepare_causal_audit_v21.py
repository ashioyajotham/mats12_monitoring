"""Freeze the qualification-informed causal-audit-v2.1 external bank."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src import causal_audit_v21
from src.audit import git_revision, runtime_environment, sha256_file
from src.causal_audit_v21 import (
    CAUSAL_AUDIT_V21_VERSION,
    build_causal_audit_v21,
    verify_causal_audit_v21_problem,
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
    "data/raw/causal_audit_v2.jsonl",
)


def main() -> None:
    """Write the immutable external-test bank, certificates, and freeze manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--root-seed", type=int, default=20262720)
    args = parser.parse_args()
    prior_ids: set[str] = set()
    prior_hashes: dict[str, str] = {}
    for value in DEFAULT_PRIOR_FILES:
        path = Path(value)
        prior_ids.update(str(row["question_id"]) for row in read_jsonl_objects(path))
        prior_hashes[value] = sha256_file(path)
    problems, sources, interventions, selection = build_causal_audit_v21(
        root_seed=args.root_seed, excluded_question_ids=prior_ids
    )
    source_by_id = {str(row["question_id"]): row for row in sources}
    intervention_by_id = {str(row["question_id"]): row for row in interventions}
    if not all(
        verify_causal_audit_v21_problem(
            row, source_by_id[row.question_id], intervention_by_id[row.question_id]
        )
        for row in problems
    ):
        raise SystemExit("a causal-audit-v2.1 certificate failed exact verification")

    stem = args.output_dir / "causal_audit_v21"
    paths = {
        "confirmatory_questions": Path(f"{stem}.confirmatory.jsonl"),
        "source_certificates": Path(f"{stem}.source_certificates.jsonl"),
        "intervention_certificates": Path(f"{stem}.intervention_certificates.jsonl"),
        "selection": Path(f"{stem}.selection.json"),
        "manifest": Path(f"{stem}.manifest.json"),
    }
    write_jsonl(paths["confirmatory_questions"], problems)
    write_jsonl(paths["source_certificates"], sources)
    write_jsonl(paths["intervention_certificates"], interventions)
    paths["selection"].parent.mkdir(parents=True, exist_ok=True)
    with paths["selection"].open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    artifact_hashes = {key: sha256_file(path) for key, path in paths.items() if key != "manifest"}
    cell_counts = Counter(
        (str(row.metadata["generator_family"]), str(row.metadata["corruption_mechanism"]))
        for row in problems
    )
    write_manifest(
        paths["manifest"],
        {
            "purpose": "causal_audit_v21_freeze",
            "protocol": CAUSAL_AUDIT_V21_VERSION,
            "license": "MIT",
            "root_seed": args.root_seed,
            "questions": 72,
            "planned_tinker_calls": 648,
            "cell_counts": {
                f"{family}:{mechanism}": count
                for (family, mechanism), count in sorted(cell_counts.items())
            },
            "qualification_informed_family_selection": True,
            "excluded_family": "subset_counting",
            "selection_uses_v21_model_outcomes": False,
            "all_questions_frozen_before_collection": True,
            "external_test_only": True,
            "prior_question_files": prior_hashes,
            "prior_question_ids_excluded": len(prior_ids),
            "artifact_paths": {key: str(path) for key, path in paths.items()},
            "artifact_sha256": artifact_hashes,
            "all_certificates_verified": True,
            "preregistration_sha256": sha256_file("docs/PREREGISTRATION_CAUSAL_AUDIT_V21.md"),
            "config_sha256": sha256_file("configs/tinker_causal_audit_v21_confirmatory.yaml"),
            "generator_code_sha256": sha256_file(causal_audit_v21.__file__),
            "entrypoint_sha256": sha256_file(__file__),
            "gate_code_sha256": sha256_file("src/causal_audit_gates.py"),
            "gate_entrypoint_sha256": sha256_file("experiments/02_analyze_causal_audit_v21.py"),
            "evaluation_entrypoint_sha256": sha256_file(
                "experiments/03_evaluate_causal_audit_v2.py"
            ),
            "code_revision": git_revision(),
            "runtime_environment": runtime_environment(),
        },
    )
    print(json.dumps({"questions": 72, "cells": 6, "planned_calls": 648}, indent=2))


if __name__ == "__main__":
    main()
