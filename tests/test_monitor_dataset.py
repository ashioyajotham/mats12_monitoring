import pytest
from pydantic import ValidationError

from src.generate_rollouts import Rollout, RolloutStatus
from src.hints import Condition
from src.monitor_dataset import (
    CausalErrorExampleKind,
    TranscriptOnlyEvidence,
    materialize_monitor_examples,
    qualification_smoke_examples,
)
from src.tasks import MathProblem


def _fixture():
    questions: list[MathProblem] = []
    rollouts: list[Rollout] = []
    for question_index in range(9):
        split = ("train", "validation", "test")[question_index % 3]
        question_id = f"q-{question_index}"
        question = MathProblem(
            question_id=question_id,
            prompt=f"Question {question_index}",
            gold_answer="1",
            metadata={
                "study_partition": "confirmatory",
                "excluded_from_monitor_data": False,
                "monitor_split": split,
                "generator_family": f"family-{question_index % 3}",
                "difficulty_tier": "tier",
                "renderer_id": question_index % 2,
                "intervention_target_answer": "2",
            },
        )
        questions.append(question)
        specifications = (
            (Condition.CLEAN, "3"),
            (Condition.CORRUPTED_CONTINUATION, "2"),
            (Condition.CORRECT_CONTINUATION, "4"),
            (Condition.CORRUPTED_CONTINUATION, "5"),
            (Condition.CLEAN, "1"),
        )
        for rollout_index, (condition, answer) in enumerate(specifications):
            rollout_id = f"r-{question_index}-{rollout_index}"
            rollouts.append(
                Rollout(
                    rollout_id=rollout_id,
                    question_id=question_id,
                    task_family="causal_error_math",
                    condition=condition,
                    hinted_option="2" if condition is Condition.CORRUPTED_CONTINUATION else None,
                    hint_template=None,
                    prompt=f"Rendered prompt {condition}",
                    response=f"reasoning {answer}",
                    reasoning=f"reasoning {answer}",
                    final_response=f"\\boxed{{{answer}}}",
                    parsed_answer=answer,
                    gold_answer="1",
                    seed=question_index * 100 + rollout_index,
                    model="model",
                    generation={"temperature": 1.0},
                    finish_reason="stop",
                    created_at="2026-08-30T00:00:00+00:00",
                    status=RolloutStatus.CLEAN_STOP,
                )
            )
    return questions, rollouts


def test_transcript_evidence_structurally_rejects_forbidden_fields() -> None:
    with pytest.raises(ValidationError, match="condition"):
        TranscriptOnlyEvidence(
            question="q",
            reasoning="r",
            final_answer="a",
            condition="corrupted_continuation",
        )


def test_materializer_builds_primary_and_bounded_secondary_views() -> None:
    questions, rollouts = _fixture()
    primary, secondary, summary = materialize_monitor_examples(
        questions,
        rollouts,
        {
            "confirmatory_causal_gate_passed": True,
            "monitor_training_authorized": True,
        },
        secondary_cap=5,
    )
    assert len(primary) == 18
    assert {row.kind for row in primary} == {
        CausalErrorExampleKind.ORDINARY_ERROR,
        CausalErrorExampleKind.CAUSALLY_INDUCED_ERROR,
    }
    assert len(secondary) == 5
    assert all(row.binary_label == 0 for row in secondary)
    assert summary["primary_examples"] == 18
    forbidden = {
        "condition",
        "target_answer",
        "certificate",
        "binary_label",
        "split",
        "family",
    }
    assert not forbidden & set(primary[0].transcript.model_dump())
    assert "rendered_prompt" not in primary[0].transcript.model_dump()
    assert "rendered_prompt" in primary[0].context.model_dump()


def test_materializer_hard_blocks_before_confirmatory_gate() -> None:
    questions, rollouts = _fixture()
    with pytest.raises(ValueError, match="has not passed"):
        materialize_monitor_examples(
            questions,
            rollouts,
            {
                "confirmatory_causal_gate_passed": False,
                "monitor_training_authorized": False,
            },
        )


def test_judge_smoke_uses_only_excluded_qualification_errors() -> None:
    questions, rollouts = _fixture()
    qualification = [
        question.model_copy(
            update={
                "metadata": {
                    **question.metadata,
                    "study_partition": "qualification",
                    "excluded_from_monitor_data": True,
                }
            }
        )
        for question in questions
    ]
    selected = qualification_smoke_examples(qualification, rollouts, count=2, seed=9)
    assert len(selected) == 2
    assert all(row.binary_label == 0 for row in selected)
    assert all(row.kind is CausalErrorExampleKind.ORDINARY_ERROR for row in selected)
    assert "condition" not in selected[0].transcript.model_dump()
    assert "binary_label" not in selected[0].context.model_dump()
