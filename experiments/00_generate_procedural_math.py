"""Create the immutable solver-verified procedural mathematics candidate bank."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.audit import git_revision, runtime_environment, sha256_file
from src.datasets import procedural_math
from src.datasets.procedural_math import (
    FAMILIES,
    GENERATOR_VERSION,
    TIERS,
    generate_candidate_bank,
    verify_problem,
)
from src.generate_rollouts import write_manifest
from src.tasks import write_jsonl


def main() -> None:
    """Generate candidates, certificates, and a content-addressed manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--root-seed", type=int, default=20260830)
    parser.add_argument("--per-cell", type=int, default=10)
    args = parser.parse_args()

    problems, certificates = generate_candidate_bank(
        root_seed=args.root_seed, per_cell=args.per_cell
    )
    if not all(
        verify_problem(problem, certificate)
        for problem, certificate in zip(problems, certificates, strict=True)
    ):
        raise SystemExit("at least one generated certificate failed independent verification")

    questions_path = args.output_dir / "procedural_math_candidates_v1.jsonl"
    certificates_path = args.output_dir / "procedural_math_certificates_v1.jsonl"
    manifest_path = args.output_dir / "procedural_math_candidates_v1.manifest.json"
    write_jsonl(questions_path, problems)
    write_jsonl(certificates_path, certificates)
    counts = Counter(
        (problem.metadata["generator_family"], problem.metadata["difficulty_tier"])
        for problem in problems
    )
    write_manifest(
        manifest_path,
        {
            "purpose": "procedural_math_candidate_freeze",
            "generator_version": GENERATOR_VERSION,
            "license": "MIT",
            "generation_method": "deterministic_original_solver_verified",
            "root_seed": args.root_seed,
            "per_cell": args.per_cell,
            "families": list(FAMILIES),
            "tiers": list(TIERS),
            "questions": len(problems),
            "certificates": len(certificates),
            "cell_counts": {
                f"{family}:{tier}": count for (family, tier), count in sorted(counts.items())
            },
            "unique_question_ids": len({problem.question_id for problem in problems}),
            "unique_prompts": len({problem.prompt for problem in problems}),
            "all_certificates_verified": True,
            "questions_path": str(questions_path),
            "questions_sha256": sha256_file(questions_path),
            "certificates_path": str(certificates_path),
            "certificates_sha256": sha256_file(certificates_path),
            "code_revision": git_revision(),
            "generator_code_sha256": sha256_file(procedural_math.__file__),
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
                "count": len(problems),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
