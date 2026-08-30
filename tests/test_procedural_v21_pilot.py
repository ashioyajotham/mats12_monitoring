from datetime import UTC, datetime

from src.datasets.procedural_math_v2 import FAMILIES_V2, generate_candidate_bank_v2
from src.datasets.procedural_math_v21 import generate_subset_replacement_bank
from src.generate_rollouts import Rollout, RolloutStatus
from src.hints import Condition
from src.procedural_v21_pilot import (
    analyze_mixed_outcome_v21,
    select_combined_pilot_v21,
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
        rollout_id=f"v21-{problem.question_id}-{index}",
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
        generation={"temperature": 1.0, "top_p": 0.95, "max_new_tokens": 8192},
        finish_reason="length" if status is RolloutStatus.LENGTH_TRUNCATED else "stop",
        created_at=datetime.now(UTC).isoformat(),
        provider_request_id=f"v21-request-{problem.question_id}-{index}",
        provider_model="openai/gpt-oss-20b",
        status=status,
    )


def _v2_screen(problems: list[MathProblem], *, reverse: bool = False) -> list[Rollout]:
    rows: list[Rollout] = []
    for start in range(0, len(problems), 10):
        cell = problems[start : start + 10]
        family = str(cell[0].metadata["generator_family"])
        for offset, problem in enumerate(cell):
            if family == "subset_counting":
                correct = cell[0].difficulty == "boundary_low"
            else:
                correct = offset >= 5 if reverse else offset < 5
            rows.append(_rollout(problem, index=0, correct=correct))
    return rows


def _replacement_screen(
    problems: list[MathProblem], *, reverse: bool = False
) -> list[Rollout]:
    return [
        _rollout(problem, index=0, correct=(offset >= 5 if reverse else offset < 5))
        for start in range(0, len(problems), 10)
        for offset, problem in enumerate(problems[start : start + 10])
    ]


def test_combined_selection_passes_and_is_item_outcome_invariant() -> None:
    v2, v2_certificates = generate_candidate_bank_v2(root_seed=301, per_cell=10)
    replacements, replacement_certificates = generate_subset_replacement_bank(
        root_seed=302, per_tier=10
    )
    first_report, first, first_certificates = select_combined_pilot_v21(
        v2, _v2_screen(v2), v2_certificates,
        replacements, _replacement_screen(replacements), replacement_certificates,
    )
    second_report, second, _ = select_combined_pilot_v21(
        v2, _v2_screen(v2, reverse=True), v2_certificates,
        replacements, _replacement_screen(replacements, reverse=True),
        replacement_certificates,
    )
    assert first_report["selection_passed"] and second_report["selection_passed"]
    assert not first_report["individual_screening_outcomes_used_for_selection"]
    assert len(first) == len(first_certificates) == 40
    assert [row.question_id for row in first] == [row.question_id for row in second]
    assert first_report["selected_family_counts"] == {family: 10 for family in FAMILIES_V2}


def test_combined_selection_stops_when_replacement_has_no_eligible_tier() -> None:
    v2, v2_certificates = generate_candidate_bank_v2(root_seed=303, per_cell=10)
    replacements, replacement_certificates = generate_subset_replacement_bank(
        root_seed=304, per_tier=10
    )
    replacement_rollouts = [
        _rollout(problem, index=0, correct=True) for problem in replacements
    ]
    report, selected, selected_certificates = select_combined_pilot_v21(
        v2, _v2_screen(v2), v2_certificates,
        replacements, replacement_rollouts, replacement_certificates,
    )
    assert not report["selection_passed"]
    assert selected == selected_certificates == []


def test_v21_fresh_mixed_outcome_wrapper_preserves_monitor_boundary() -> None:
    v2, _ = generate_candidate_bank_v2(root_seed=305, per_cell=10)
    replacements, _ = generate_subset_replacement_bank(root_seed=306, per_tier=10)
    frozen = [
        row
        for family in [name for name in FAMILIES_V2 if name != "subset_counting"]
        for row in [p for p in v2 if p.metadata["generator_family"] == family][:10]
    ] + replacements[:10]
    rollouts = [
        _rollout(problem, index=index, correct=index < 2)
        for problem in frozen for index in range(3)
    ]
    report = analyze_mixed_outcome_v21(frozen, rollouts)
    assert report["task_readiness_gate_passed"]
    assert report["protocol"] == "procedural-clean-mixed-outcome-v2.1"
    assert not report["monitor_training_authorized"]
