from datetime import UTC, datetime

from src.datasets.procedural_math import generate_candidate_bank
from src.generate_rollouts import Rollout, RolloutStatus
from src.hints import Condition
from src.procedural_pilot import analyze_frozen_discovery, select_screened_questions
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
        rollout_id=f"rollout-{problem.question_id}-{index}",
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
        provider_request_id=f"request-{problem.question_id}-{index}",
        provider_model="openai/gpt-oss-20b",
        status=status,
    )


def _screening_rollouts(problems: list[MathProblem], *, reverse: bool = False) -> list[Rollout]:
    rows: list[Rollout] = []
    for cell_start in range(0, len(problems), 10):
        cell = problems[cell_start : cell_start + 10]
        for offset, problem in enumerate(cell):
            correct = offset >= 5 if reverse else offset < 5
            rows.append(_rollout(problem, index=0, correct=correct))
    return rows


def test_screening_selection_is_balanced_and_item_outcome_invariant():
    problems, certificates = generate_candidate_bank(root_seed=17, per_cell=10)
    first_report, first_selected, first_certificates = select_screened_questions(
        problems, _screening_rollouts(problems), certificates
    )
    second_report, second_selected, _ = select_screened_questions(
        problems, _screening_rollouts(problems, reverse=True), certificates
    )

    assert first_report["selection_passed"]
    assert second_report["selection_passed"]
    assert len(first_selected) == 40
    assert len(first_certificates) == 40
    assert [row.question_id for row in first_selected] == [
        row.question_id for row in second_selected
    ]
    assert {row.metadata["generator_family"] for row in first_selected} == {
        "crt",
        "linear_system",
        "dag_counting",
        "recurrence",
    }


def test_screening_fails_when_one_family_has_no_eligible_cell():
    problems, certificates = generate_candidate_bank(root_seed=18, per_cell=10)
    rollouts = _screening_rollouts(problems)
    recurrence_ids = {
        row.question_id
        for row in problems
        if row.metadata["generator_family"] == "recurrence"
    }
    for rollout in rollouts:
        if rollout.question_id in recurrence_ids:
            rollout.parsed_answer = rollout.gold_answer
    report, selected, selected_certificates = select_screened_questions(
        problems, rollouts, certificates
    )
    assert not report["selection_passed"]
    assert report["missing_eligible_families"] == ["recurrence"]
    assert selected == []
    assert selected_certificates == []


def test_screening_fails_for_missing_rollout_or_request_error():
    problems, certificates = generate_candidate_bank(root_seed=19, per_cell=10)
    rollouts = _screening_rollouts(problems)
    report, _, _ = select_screened_questions(
        problems, rollouts[:-1], certificates, request_errors=1
    )
    assert not report["selection_passed"]
    assert report["request_errors"] == 1
    assert report["duplicate_or_missing_rollout_ids"] == [problems[-1].question_id]


def test_frozen_discovery_passes_with_diverse_clean_errors():
    problems, _ = generate_candidate_bank(root_seed=20, per_cell=10)
    frozen = [
        problem
        for family in ("crt", "linear_system", "dag_counting", "recurrence")
        for problem in [
            row for row in problems if row.metadata["generator_family"] == family
        ][:10]
    ]
    rollouts = [
        _rollout(problem, index=index, correct=index < 2)
        for problem in frozen
        for index in range(3)
    ]
    report = analyze_frozen_discovery(frozen, rollouts, bootstrap_samples=100)

    assert report["task_readiness_gate_passed"]
    assert report["grades"] == {"correct": 80, "incorrect": 40}
    assert report["unique_provider_request_ids"] == 120
    interval = report["clustered_accuracy_interval_95"]
    assert interval["low"] == interval["high"] == 2 / 3


def test_frozen_discovery_rejects_excess_truncation_and_duplicate_provider_ids():
    problems, _ = generate_candidate_bank(root_seed=21, per_cell=10)
    frozen = problems[:40]
    rollouts = [
        _rollout(
            problem,
            index=index,
            correct=index < 2,
            status=(
                RolloutStatus.LENGTH_TRUNCATED
                if position < 13
                else RolloutStatus.CLEAN_STOP
            ),
        )
        for position, (problem, index) in enumerate(
            [(problem, index) for problem in frozen for index in range(3)]
        )
    ]
    rollouts[-1].provider_request_id = rollouts[-2].provider_request_id
    report = analyze_frozen_discovery(frozen, rollouts, bootstrap_samples=20)

    assert not report["task_readiness_gate_passed"]
    assert not report["gate_checks"]["at_most_10_percent_truncated"]
    assert not report["gate_checks"]["unique_provider_request_ids"]
