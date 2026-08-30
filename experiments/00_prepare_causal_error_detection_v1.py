"""Freeze the fresh causal-error-detection-v1 task and intervention bank."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src import causal_error_dataset
from src.audit import git_revision, runtime_environment, sha256_file
from src.causal_error_dataset import (
    CAUSAL_ERROR_VERSION,
    build_causal_error_dataset,
    verify_causal_error_problem,
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
)


def _prior_ids(paths: list[Path]) -> tuple[set[str], dict[str, str]]:
    """Read prior question IDs and bind every exclusion source by content hash."""
    question_ids: set[str] = set()
    hashes: dict[str, str] = {}
    for path in paths:
        rows = read_jsonl_objects(path)
        question_ids.update(str(row["question_id"]) for row in rows)
        hashes[str(path)] = sha256_file(path)
    return question_ids, hashes


def main() -> None:
    """Create immutable all/qualification/confirmatory freezes and their certificates."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--root-seed", type=int, default=20262501)
    parser.add_argument("--partition-seed", type=int, default=20262502)
    parser.add_argument("--split-seed", type=int, default=20262503)
    parser.add_argument("--per-family", type=int, default=24)
    parser.add_argument(
        "--prior-question-file",
        action="append",
        type=Path,
        dest="prior_files",
        help="Question JSONL whose IDs must be excluded; repeat to override defaults.",
    )
    args = parser.parse_args()
    prior_files = args.prior_files or [Path(path) for path in DEFAULT_PRIOR_FILES]
    prior_ids, prior_hashes = _prior_ids(prior_files)
    problems, source_certificates, intervention_certificates, selection = (
        build_causal_error_dataset(
            root_seed=args.root_seed,
            partition_seed=args.partition_seed,
            split_seed=args.split_seed,
            per_family=args.per_family,
            excluded_question_ids=prior_ids,
        )
    )
    source_by_id = {str(row["question_id"]): row for row in source_certificates}
    intervention_by_id = {
        str(row["question_id"]): row for row in intervention_certificates
    }
    if not all(
        verify_causal_error_problem(
            problem,
            source_by_id[problem.question_id],
            intervention_by_id[problem.question_id],
        )
        for problem in problems
    ):
        raise SystemExit("a causal-error certificate failed exact verification")

    qualification = [
        row for row in problems if row.metadata["study_partition"] == "qualification"
    ]
    confirmatory = [
        row for row in problems if row.metadata["study_partition"] == "confirmatory"
    ]
    stem = args.output_dir / "causal_error_detection_v1"
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
    write_jsonl(paths["source_certificates"], source_certificates)
    write_jsonl(paths["intervention_certificates"], intervention_certificates)
    paths["selection"].parent.mkdir(parents=True, exist_ok=True)
    with paths["selection"].open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(selection, indent=2, sort_keys=True) + "\n")

    cell_counts = Counter(
        (
            str(row.metadata["generator_family"]),
            str(row.metadata["difficulty_tier"]),
            int(row.metadata["renderer_id"]),
        )
        for row in problems
    )
    artifact_hashes = {
        key: sha256_file(path)
        for key, path in paths.items()
        if key != "manifest"
    }
    write_manifest(
        paths["manifest"],
        {
            "purpose": "causal_error_detection_v1_freeze",
            "protocol": CAUSAL_ERROR_VERSION,
            "license": "MIT",
            "generation_method": "fresh_original_dual_exact_verification",
            "root_seed": args.root_seed,
            "partition_seed": args.partition_seed,
            "monitor_split_seed": args.split_seed,
            "per_family": args.per_family,
            "questions": len(problems),
            "qualification_questions": len(qualification),
            "confirmatory_questions": len(confirmatory),
            "cell_counts": {
                f"{family}:{tier}:renderer-{renderer}": count
                for (family, tier, renderer), count in sorted(cell_counts.items())
            },
            "partition_counts": selection["partition_counts"],
            "monitor_split_counts": selection["monitor_split_counts"],
            "selection_uses_model_outcomes": False,
            "all_partitions_frozen_before_collection": True,
            "prior_question_files": prior_hashes,
            "prior_question_ids_excluded": len(prior_ids),
            "artifact_paths": {key: str(path) for key, path in paths.items()},
            "artifact_sha256": artifact_hashes,
            "all_certificates_verified": True,
            "generator_code_sha256": sha256_file(causal_error_dataset.__file__),
            "entrypoint_sha256": sha256_file(__file__),
            "code_revision": git_revision(),
            "runtime_environment": runtime_environment(),
        },
    )
    print(
        json.dumps(
            {
                "questions": len(problems),
                "qualification": len(qualification),
                "confirmatory": len(confirmatory),
                "manifest": str(paths["manifest"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
