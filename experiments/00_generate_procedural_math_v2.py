"""Create the immutable solver-verified procedural-math-v2 candidate bank."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.audit import git_revision, runtime_environment, sha256_file
from src.datasets import procedural_math_v2
from src.datasets.procedural_math_v2 import (
    FAMILIES_V2,
    GENERATOR_VERSION_V2,
    TIERS_V2,
    generate_candidate_bank_v2,
    verify_problem_v2,
)
from src.generate_rollouts import write_manifest
from src.tasks import write_jsonl


def main() -> None:
    """Generate, independently verify, and content-address all v2 inputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--root-seed", type=int, default=20261201)
    parser.add_argument("--per-cell", type=int, default=10)
    args = parser.parse_args()
    problems, certificates = generate_candidate_bank_v2(
        root_seed=args.root_seed, per_cell=args.per_cell
    )
    if not all(
        verify_problem_v2(problem, certificate)
        for problem, certificate in zip(problems, certificates, strict=True)
    ):
        raise SystemExit("a v2 certificate failed independent verification")
    questions_path = args.output_dir / "procedural_math_candidates_v2.jsonl"
    certificates_path = args.output_dir / "procedural_math_certificates_v2.jsonl"
    manifest_path = args.output_dir / "procedural_math_candidates_v2.manifest.json"
    write_jsonl(questions_path, problems)
    write_jsonl(certificates_path, certificates)
    counts = Counter(
        (problem.metadata["generator_family"], problem.metadata["difficulty_tier"])
        for problem in problems
    )
    write_manifest(
        manifest_path,
        {
            "purpose": "procedural_math_v2_candidate_freeze",
            "generator_version": GENERATOR_VERSION_V2,
            "license": "MIT",
            "generation_method": "deterministic_original_dual_exact_verification",
            "root_seed": args.root_seed,
            "per_cell": args.per_cell,
            "families": list(FAMILIES_V2),
            "tiers": list(TIERS_V2),
            "questions": len(problems),
            "certificates": len(certificates),
            "cell_counts": {
                f"{family}:{tier}": count
                for (family, tier), count in sorted(counts.items())
            },
            "unique_question_ids": len({row.question_id for row in problems}),
            "unique_prompts": len({row.prompt for row in problems}),
            "all_certificates_verified": True,
            "questions_path": str(questions_path),
            "questions_sha256": sha256_file(questions_path),
            "certificates_path": str(certificates_path),
            "certificates_sha256": sha256_file(certificates_path),
            "code_revision": git_revision(),
            "generator_code_sha256": sha256_file(procedural_math_v2.__file__),
            "entrypoint_sha256": sha256_file(__file__),
            "runtime_environment": runtime_environment(),
        },
    )
    print(json.dumps({"questions": str(questions_path), "certificates": str(certificates_path),
                      "manifest": str(manifest_path), "count": len(problems)}, indent=2))


if __name__ == "__main__":
    main()
