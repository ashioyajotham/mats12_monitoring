import json
from pathlib import Path

from src.assistant_prefill_interventions import (
    build_assistant_prefill_freeze,
    verify_assistant_prefill,
)
from src.datasets.procedural_math_v2 import generate_candidate_bank_v2
from src.datasets.procedural_math_v21 import generate_subset_replacement_bank
from src.hints import Condition, build_variant
from src.tasks import MathProblem, read_jsonl


def _jsonl(path: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _regenerate():
    v2_questions, v2_certificates = generate_candidate_bank_v2(
        root_seed=20262301, per_cell=12
    )
    subset_questions, subset_certificates = generate_subset_replacement_bank(
        root_seed=20262302, per_tier=12
    )
    retained = [
        (question, certificate)
        for question, certificate in zip(v2_questions, v2_certificates, strict=True)
        if question.metadata["generator_family"] != "subset_counting"
    ]
    retained.extend(zip(subset_questions, subset_certificates, strict=True))
    questions = [row[0] for row in retained]
    certificates = [row[1] for row in retained]
    frozen, intervention_certificates, _ = build_assistant_prefill_freeze(
        questions, certificates, per_family=2, selection_seed=20262303
    )
    return frozen, intervention_certificates, certificates


def test_committed_assistant_prefill_freeze_is_reproducible_and_verified():
    frozen, certificates, source_certificates = _regenerate()
    committed = list(
        read_jsonl("data/raw/procedural_assistant_prefill_v3.jsonl", model=MathProblem)
    )
    committed_certificates = _jsonl(
        "data/raw/procedural_assistant_prefill_v3.certificates.jsonl"
    )
    assert [json.loads(row.model_dump_json()) for row in frozen] == [
        json.loads(row.model_dump_json()) for row in committed
    ]
    assert certificates == committed_certificates
    source_by_id = {str(row["question_id"]): row for row in source_certificates}
    assert all(
        verify_assistant_prefill(
            problem, certificate, source_by_id[problem.question_id]
        )
        for problem, certificate in zip(committed, certificates, strict=True)
    )


def test_prefill_pairs_are_token_matched_and_hidden_from_user_prompt():
    questions = list(
        read_jsonl("data/raw/procedural_assistant_prefill_v3.jsonl", model=MathProblem)
    )
    audits = _jsonl("data/raw/procedural_assistant_prefill_v3.token_audit.jsonl")
    assert len(audits) == 8
    assert all(row["equal_token_length"] is True for row in audits)
    assert all(row["changed_token_count"] == 1 for row in audits)
    for problem in questions:
        correct = build_variant(problem, Condition.CORRECT_ASSISTANT_PREFILL)
        corrupted = build_variant(problem, Condition.CORRUPTED_ASSISTANT_PREFILL)
        assert correct.rendered_prompt == corrupted.rendered_prompt
        assert correct.assistant_prefill != corrupted.assistant_prefill
        assert correct.assistant_prefill not in correct.rendered_prompt
        assert corrupted.assistant_prefill not in corrupted.rendered_prompt
        assert "checkpoint" not in correct.assistant_prefill.lower()
        assert "continue" not in correct.assistant_prefill.lower()
        assert "conditions reduce to" not in correct.assistant_prefill.lower()
