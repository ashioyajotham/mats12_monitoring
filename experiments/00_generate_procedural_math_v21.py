"""Freeze fresh subset replacements for the procedural v2.1 amendment."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.audit import git_revision, runtime_environment, sha256_file
from src.datasets import procedural_math_v21
from src.datasets.procedural_math_v21 import (
    FAMILY_V21,
    GENERATOR_VERSION_V21,
    TIERS_V21,
    generate_subset_replacement_bank,
    verify_subset_replacement,
)
from src.generate_rollouts import write_manifest
from src.tasks import write_jsonl


def main() -> None:
    """Generate, verify, and content-address the 20 replacement candidates."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--root-seed", type=int, default=20261501)
    parser.add_argument("--per-tier", type=int, default=10)
    args = parser.parse_args()
    problems, certificates = generate_subset_replacement_bank(
        root_seed=args.root_seed, per_tier=args.per_tier
    )
    if not all(
        verify_subset_replacement(problem, certificate)
        for problem, certificate in zip(problems, certificates, strict=True)
    ):
        raise SystemExit("a v2.1 certificate failed independent verification")
    questions_path = args.output_dir / "procedural_math_subset_replacements_v21.jsonl"
    certificates_path = (
        args.output_dir / "procedural_math_subset_replacements_v21.certificates.jsonl"
    )
    manifest_path = args.output_dir / "procedural_math_subset_replacements_v21.manifest.json"
    write_jsonl(questions_path, problems)
    write_jsonl(certificates_path, certificates)
    counts = Counter(str(row.difficulty) for row in problems)
    write_manifest(manifest_path, {
        "purpose": "procedural_math_v21_subset_replacement_freeze",
        "generator_version": GENERATOR_VERSION_V21,
        "license": "MIT", "generation_method": "deterministic_original_dual_exact_verification",
        "root_seed": args.root_seed, "per_tier": args.per_tier,
        "family": FAMILY_V21, "tiers": list(TIERS_V21), "tier_counts": dict(sorted(counts.items())),
        "questions": len(problems), "certificates": len(certificates),
        "unique_question_ids": len({row.question_id for row in problems}),
        "unique_prompts": len({row.prompt for row in problems}), "all_certificates_verified": True,
        "questions_path": str(questions_path), "questions_sha256": sha256_file(questions_path),
        "certificates_path": str(certificates_path),
        "certificates_sha256": sha256_file(certificates_path),
        "code_revision": git_revision(),
        "generator_code_sha256": sha256_file(procedural_math_v21.__file__),
        "entrypoint_sha256": sha256_file(__file__), "runtime_environment": runtime_environment(),
    })
    print(json.dumps({"questions": str(questions_path), "certificates": str(certificates_path),
                      "manifest": str(manifest_path), "count": len(problems)}, indent=2))


if __name__ == "__main__":
    main()
