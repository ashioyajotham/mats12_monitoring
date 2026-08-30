"""Generate and freeze fresh assistant-prefill diagnostic inputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from src import assistant_prefill_interventions
from src.assistant_prefill_interventions import (
    build_assistant_prefill_freeze,
    token_pair_audit,
    verify_assistant_prefill,
)
from src.audit import git_revision, runtime_environment, sha256_file
from src.datasets.procedural_math_v2 import generate_candidate_bank_v2
from src.datasets.procedural_math_v21 import generate_subset_replacement_bank
from src.generate_rollouts import write_manifest
from src.tasks import write_jsonl


def main() -> None:
    """Create a balanced immutable eight-question prefill-v3 diagnostic."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--v2-root-seed", type=int, default=20262301)
    parser.add_argument("--v21-root-seed", type=int, default=20262302)
    parser.add_argument("--selection-seed", type=int, default=20262303)
    parser.add_argument("--per-cell", type=int, default=12)
    parser.add_argument("--per-subset-tier", type=int, default=12)
    parser.add_argument("--per-family", type=int, default=2)
    parser.add_argument(
        "--tokenizer",
        default="openai/gpt-oss-20b",
        help="Pinned tokenizer name or local snapshot used for the token-pair audit.",
    )
    args = parser.parse_args()

    v2_questions, v2_certificates = generate_candidate_bank_v2(
        root_seed=args.v2_root_seed, per_cell=args.per_cell
    )
    subset_questions, subset_certificates = generate_subset_replacement_bank(
        root_seed=args.v21_root_seed, per_tier=args.per_subset_tier
    )
    retained = [
        (question, certificate)
        for question, certificate in zip(v2_questions, v2_certificates, strict=True)
        if question.metadata["generator_family"] != "subset_counting"
    ]
    retained.extend(zip(subset_questions, subset_certificates, strict=True))
    questions = [row[0] for row in retained]
    source_certificates = [row[1] for row in retained]
    frozen, certificates, selection = build_assistant_prefill_freeze(
        questions,
        source_certificates,
        per_family=args.per_family,
        selection_seed=args.selection_seed,
    )
    source_certificate_by_id = {
        str(row["question_id"]): row for row in source_certificates
    }
    if not all(
        verify_assistant_prefill(
            problem,
            certificate,
            source_certificate_by_id[problem.question_id],
        )
        for problem, certificate in zip(frozen, certificates, strict=True)
    ):
        raise SystemExit("an assistant-prefill certificate failed exact verification")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    audits = [
        {
            "question_id": problem.question_id,
            **token_pair_audit(
                str(certificate["correct_prefill"]),
                str(certificate["corrupted_prefill"]),
                tokenizer,
            ),
        }
        for problem, certificate in zip(frozen, certificates, strict=True)
    ]
    questions_path = args.output_dir / "procedural_assistant_prefill_v3.jsonl"
    certificates_path = args.output_dir / "procedural_assistant_prefill_v3.certificates.jsonl"
    selection_path = args.output_dir / "procedural_assistant_prefill_v3.selection.json"
    token_audit_path = args.output_dir / "procedural_assistant_prefill_v3.token_audit.jsonl"
    manifest_path = args.output_dir / "procedural_assistant_prefill_v3.manifest.json"
    write_jsonl(questions_path, frozen)
    write_jsonl(certificates_path, certificates)
    write_jsonl(token_audit_path, audits)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    with selection_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    family_counts = Counter(
        str(problem.metadata["generator_family"]) for problem in frozen
    )
    write_manifest(manifest_path, {
        "purpose": "procedural_assistant_prefill_v3_freeze",
        "protocol": selection["protocol"],
        "questions": len(frozen),
        "family_counts": dict(sorted(family_counts.items())),
        "v2_root_seed": args.v2_root_seed,
        "v21_root_seed": args.v21_root_seed,
        "selection_seed": args.selection_seed,
        "selection_uses_model_outcomes": False,
        "diagnostic_only": True,
        "excluded_from_monitor_data": True,
        "tokenizer": args.tokenizer,
        "all_certificates_verified": True,
        "all_token_pairs_equal_length": True,
        "questions_sha256": sha256_file(questions_path),
        "certificates_sha256": sha256_file(certificates_path),
        "selection_sha256": sha256_file(selection_path),
        "token_audit_sha256": sha256_file(token_audit_path),
        "intervention_code_sha256": sha256_file(
            assistant_prefill_interventions.__file__
        ),
        "entrypoint_sha256": sha256_file(__file__),
        "code_revision": git_revision(),
        "runtime_environment": runtime_environment(),
    })
    print(json.dumps({
        "questions": str(questions_path),
        "certificates": str(certificates_path),
        "selection": str(selection_path),
        "token_audit": str(token_audit_path),
        "manifest": str(manifest_path),
        "count": len(frozen),
    }, indent=2))


if __name__ == "__main__":
    main()
