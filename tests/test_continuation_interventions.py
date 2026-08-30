import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from src.audit import sha256_file
from src.continuation_interventions import (
    build_continuation_freeze,
    verify_continuation,
)
from src.tasks import MathProblem, read_jsonl


def _source() -> tuple[list[MathProblem], list[dict[str, object]], set[str]]:
    questions = list(read_jsonl("data/raw/procedural_math_pilot_v21.jsonl", model=MathProblem))
    certificates = [
        json.loads(line)
        for line in Path(
            "data/raw/procedural_math_pilot_v21.certificates.jsonl"
        ).read_text().splitlines()
        if line.strip()
    ]
    prior = list(read_jsonl("data/raw/procedural_causal_yield_v1.jsonl", model=MathProblem))
    return questions, certificates, {row.question_id for row in prior}


def test_continuation_freeze_is_unused_balanced_deterministic_and_verified() -> None:
    questions, source_certificates, excluded = _source()
    first, first_certificates, first_report = build_continuation_freeze(
        questions, source_certificates, excluded_question_ids=excluded
    )
    second, second_certificates, second_report = build_continuation_freeze(
        questions, source_certificates, excluded_question_ids=excluded
    )
    assert first == second
    assert first_certificates == second_certificates
    assert first_report == second_report
    assert len(first) == 8
    assert not ({row.question_id for row in first} & excluded)
    assert Counter(row.metadata["generator_family"] for row in first) == Counter({
        "affine_modular": 2,
        "conditional_dag": 2,
        "finite_state": 2,
        "subset_counting": 2,
    })
    source_by_id = {str(row["question_id"]): row for row in source_certificates}
    assert all(
        verify_continuation(problem, certificate, source_by_id[problem.question_id])
        for problem, certificate in zip(first, first_certificates, strict=True)
    )
    assert all(
        certificate["target_answer"] != problem.gold_answer
        and certificate["checkpoint"]["corrupted_value"]
        != certificate["checkpoint"]["correct_value"]
        for problem, certificate in zip(first, first_certificates, strict=True)
    )


def test_continuation_certificate_mutation_is_rejected() -> None:
    questions, source_certificates, excluded = _source()
    frozen, certificates, _ = build_continuation_freeze(
        questions, source_certificates, excluded_question_ids=excluded
    )
    source_by_id = {str(row["question_id"]): row for row in source_certificates}
    changed = copy.deepcopy(certificates[0])
    changed["target_answer"] = frozen[0].gold_answer
    assert not verify_continuation(
        frozen[0], changed, source_by_id[frozen[0].question_id]
    )


def test_committed_continuation_freeze_matches_generator_and_manifest() -> None:
    questions_path = Path("data/raw/procedural_continuation_yield_v2.jsonl")
    certificates_path = Path(
        "data/raw/procedural_continuation_yield_v2.certificates.jsonl"
    )
    manifest_path = Path("data/raw/procedural_continuation_yield_v2.manifest.json")
    committed = list(read_jsonl(questions_path, model=MathProblem))
    committed_certificates = [
        json.loads(line) for line in certificates_path.read_text().splitlines() if line.strip()
    ]
    source_questions, source_certificates, excluded = _source()
    regenerated, regenerated_certificates, _ = build_continuation_freeze(
        source_questions, source_certificates, excluded_question_ids=excluded
    )
    assert [row.model_dump(mode="json") for row in committed] == [
        row.model_dump(mode="json") for row in regenerated
    ]
    assert committed_certificates == regenerated_certificates
    manifest = json.loads(manifest_path.read_text())
    recorded_hash = manifest.pop("manifest_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    assert recorded_hash == hashlib.sha256(canonical.encode()).hexdigest()
    assert manifest["questions_sha256"] == sha256_file(questions_path)
    assert manifest["certificates_sha256"] == sha256_file(certificates_path)
    assert manifest["intervention_code_sha256"] == sha256_file(
        "src/continuation_interventions.py"
    )
    assert manifest["entrypoint_sha256"] == sha256_file(
        "experiments/00_prepare_continuation_yield_pilot.py"
    )
