from datetime import UTC, datetime

from src.datasets.procedural_math_v2 import FAMILIES_V2, generate_candidate_bank_v2
from src.generate_rollouts import Rollout, RolloutStatus
from src.hints import Condition
from src.procedural_v2_pilot import (
    analyze_mixed_outcome_v2,
    select_screened_questions_v2,
)
from src.tasks import MathProblem


def _rollout(
    problem: MathProblem,
    *,
    index: int,
    correct: bool,
    status: RolloutStatus = RolloutStatus.CLEAN_STOP,
) -> Rollout:
    parsed = problem.gold_answer if correct else str(int(problem.gold_answer) + 1)
    return Rollout(
        rollout_id=f"v2-{problem.question_id}-{index}",
        question_id=problem.question_id,
        task_family=problem.task_family,
        condition=Condition.CLEAN,
        hinted_option=None,
        hint_template=None,
        prompt=problem.prompt,
        response=f"Reasoning. \\boxed{{{parsed}}}",
        reasoning="Reasoning.",
        final_response=f"\\boxed{{{parsed}}}",
        parsed_answer=parsed,
        gold_answer=problem.gold_answer,
        seed=index,
        model="openai/gpt-oss-20b",
        generation={"temperature": 1.0, "top_p": 0.95, "max_new_tokens": 4096},
        finish_reason="length" if status is RolloutStatus.LENGTH_TRUNCATED else "stop",
        created_at=datetime.now(UTC).isoformat(),
        provider_request_id=f"v2-request-{problem.question_id}-{index}",
        provider_model="openai/gpt-oss-20b",
        status=status,
    )


def _screen(problems: list[MathProblem], *, reverse: bool = False) -> list[Rollout]:
    return [
        _rollout(problem, index=0, correct=(offset >= 5 if reverse else offset < 5))
        for start in range(0, len(problems), 10)
        for offset, problem in enumerate(problems[start : start + 10])
    ]


def test_v2_selection_is_balanced_and_item_outcome_invariant() -> None:
    problems, certificates = generate_candidate_bank_v2(root_seed=101, per_cell=10)
    first_report, first, first_certificates = select_screened_questions_v2(
        problems, _screen(problems), certificates
    )
    second_report, second, _ = select_screened_questions_v2(
        problems, _screen(problems, reverse=True), certificates
    )
    assert first_report["selection_passed"] and second_report["selection_passed"]
    assert len(first) == len(first_certificates) == 40
    assert [row.question_id for row in first] == [row.question_id for row in second]
    assert {row.metadata["generator_family"] for row in first} == set(FAMILIES_V2)


def test_v2_selection_fails_without_eligible_family() -> None:
    problems, certificates = generate_candidate_bank_v2(root_seed=102, per_cell=10)
    rollouts = _screen(problems)
    for rollout in rollouts:
        problem = next(row for row in problems if row.question_id == rollout.question_id)
        if problem.metadata["generator_family"] == FAMILIES_V2[0]:
            rollout.parsed_answer = problem.gold_answer
    report, selected, selected_certificates = select_screened_questions_v2(
        problems, rollouts, certificates
    )
    assert not report["selection_passed"]
    assert report["missing_eligible_families"] == [FAMILIES_V2[0]]
    assert selected == selected_certificates == []


def _frozen() -> list[MathProblem]:
    problems, _ = generate_candidate_bank_v2(root_seed=103, per_cell=10)
    return [
        row
        for family in FAMILIES_V2
        for row in [p for p in problems if p.metadata["generator_family"] == family][:10]
    ]


def test_v2_mixed_outcome_gate_passes_diverse_clean_failures() -> None:
    frozen = _frozen()
    rollouts = [
        _rollout(problem, index=index, correct=index < 2)
        for problem in frozen
        for index in range(3)
    ]
    report = analyze_mixed_outcome_v2(frozen, rollouts, bootstrap_samples=100)
    assert report["task_readiness_gate_passed"]
    assert report["grades"] == {"correct": 80, "incorrect": 40}
    assert report["authorization_if_passed"] == "preregister_causal_yield_experiment_only"
    assert not report["monitor_training_authorized"]


def test_v2_gate_rejects_excess_truncation_and_duplicate_provider_id() -> None:
    frozen = _frozen()
    pairs = [(problem, index) for problem in frozen for index in range(3)]
    rollouts = [
        _rollout(
            problem,
            index=index,
            correct=index < 2,
            status=RolloutStatus.LENGTH_TRUNCATED if position < 13 else RolloutStatus.CLEAN_STOP,
        )
        for position, (problem, index) in enumerate(pairs)
    ]
    rollouts[-1].provider_request_id = rollouts[-2].provider_request_id
    report = analyze_mixed_outcome_v2(frozen, rollouts, bootstrap_samples=20)
    assert not report["task_readiness_gate_passed"]
    assert not report["gate_checks"]["at_most_10_percent_truncated"]
    assert not report["gate_checks"]["unique_provider_request_ids"]
