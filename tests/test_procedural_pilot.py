from datetime import UTC, datetime

from src.datasets.procedural_math import generate_candidate_bank
from src.generate_rollouts import Rollout, RolloutStatus
from src.hints import Condition
from src.procedural_pilot import (
    analyze_frozen_discovery,
    analyze_reasoning_effort_diagnostic,
    build_reasoning_effort_diagnostic,
    select_screened_questions,
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


def _diagnostic_source():
    problems, certificates = generate_candidate_bank(root_seed=22, per_cell=10)
    rollouts = [_rollout(problem, index=0, correct=True) for problem in problems]
    by_id = {rollout.question_id: rollout for rollout in rollouts}
    truncated: list[MathProblem] = []
    truncated.extend(
        [
            problem
            for problem in problems
            if problem.metadata["generator_family"] == "crt" and problem.difficulty == "hard"
        ][:2]
    )
    truncated.extend(
        [
            problem
            for problem in problems
            if problem.metadata["generator_family"] == "linear_system"
            and problem.difficulty in {"medium", "hard"}
        ][:6]
    )
    truncated.extend(
        [
            problem
            for problem in problems
            if problem.metadata["generator_family"] == "recurrence"
            and problem.metadata["renderer_id"] != 1
        ][:4]
    )
    for problem in truncated:
        by_id[problem.question_id].status = RolloutStatus.LENGTH_TRUNCATED
        by_id[problem.question_id].finish_reason = "length"
    return problems, certificates, rollouts


def test_low_reasoning_diagnostic_freeze_is_matched_and_excludes_ambiguous_renderer():
    problems, certificates, screening = _diagnostic_source()
    first_report, first, first_certificates = build_reasoning_effort_diagnostic(
        problems, screening, certificates
    )
    second_report, second, _ = build_reasoning_effort_diagnostic(
        problems, screening, certificates
    )

    assert first_report == second_report
    assert [problem.question_id for problem in first] == [
        problem.question_id for problem in second
    ]
    assert len(first) == len(first_certificates) == 24
    assert sum(
        problem.metadata["diagnostic_stratum"] == "previously_truncated"
        for problem in first
    ) == 12
    assert sum(
        problem.metadata["diagnostic_stratum"] == "matched_clean_control"
        for problem in first
    ) == 12
    assert all(problem.metadata["excluded_from_monitor_data"] for problem in first)
    assert all(
        not (
            problem.metadata["generator_family"] == "recurrence"
            and problem.metadata["renderer_id"] == 1
        )
        for problem in first
    )
    for pair in first_report["pairs"]:
        assert pair["family"] in pair["previously_truncated_question_id"]
        assert pair["family"] in pair["matched_control_question_id"]


def test_low_reasoning_attribution_gate_requires_completed_diverse_errors():
    problems, certificates, screening = _diagnostic_source()
    _, diagnostic, _ = build_reasoning_effort_diagnostic(
        problems, screening, certificates
    )
    families = sorted({problem.metadata["generator_family"] for problem in diagnostic})[:2]
    wrong_ids = {
        problem.question_id
        for family in families
        for problem in [
            row for row in diagnostic if row.metadata["generator_family"] == family
        ][:2]
    }
    rollouts = [
        _rollout(problem, index=1, correct=problem.question_id not in wrong_ids)
        for problem in diagnostic
    ]
    report = analyze_reasoning_effort_diagnostic(diagnostic, rollouts)
    assert report["diagnostic_gate_passed"]
    assert report["grades"] == {"correct": 20, "incorrect": 4}

    for rollout in rollouts[:3]:
        rollout.status = RolloutStatus.LENGTH_TRUNCATED
        rollout.finish_reason = "length"
    failed = analyze_reasoning_effort_diagnostic(diagnostic, rollouts)
    assert not failed["diagnostic_gate_passed"]
    assert not failed["gate_checks"]["at_most_two_truncated"]
