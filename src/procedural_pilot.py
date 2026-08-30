"""Selection and clean-discovery gates for the procedural mathematics pilot."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict, deque
from collections.abc import Iterable

from src.datasets.procedural_math import FAMILIES, TIERS, verify_problem
from src.generate_rollouts import Rollout, RolloutStatus
from src.hints import Condition
from src.math_answers import MathGrade, grade_math_answer
from src.tasks import MathProblem


def _family(problem: MathProblem) -> str:
    """Return validated generator-family metadata."""
    family = problem.metadata.get("generator_family")
    if not isinstance(family, str) or not family:
        raise ValueError(f"question {problem.question_id} lacks generator_family metadata")
    return family


def _tier(problem: MathProblem) -> str:
    """Return validated difficulty-tier metadata."""
    tier = problem.metadata.get("difficulty_tier")
    if not isinstance(tier, str) or not tier:
        raise ValueError(f"question {problem.question_id} lacks difficulty_tier metadata")
    return tier


def _renderer(problem: MathProblem) -> int:
    """Return validated renderer metadata."""
    renderer = problem.metadata.get("renderer_id")
    if not isinstance(renderer, int):
        raise ValueError(f"question {problem.question_id} lacks renderer_id metadata")
    return renderer


def _seeded_order(problem: MathProblem, selection_seed: int) -> str:
    """Create a stable pseudorandom sort key without global RNG state."""
    value = f"{selection_seed}|{problem.question_id}".encode()
    return hashlib.sha256(value).hexdigest()


def _diagnostic_problem(
    problem: MathProblem,
    *,
    pair_id: str,
    stratum: str,
    source_rollout: Rollout,
) -> MathProblem:
    """Annotate a copied problem with diagnostic-only provenance."""
    return problem.model_copy(
        update={
            "metadata": {
                **problem.metadata,
                "diagnostic_protocol": "low-reasoning-attribution-v1",
                "diagnostic_pair_id": pair_id,
                "diagnostic_stratum": stratum,
                "source_screening_rollout_id": source_rollout.rollout_id,
                "source_screening_status": str(source_rollout.status),
                "excluded_from_monitor_data": True,
            }
        }
    )


def _balanced_select(
    candidates: Iterable[MathProblem], *, count: int, selection_seed: int
) -> list[MathProblem]:
    """Select deterministically while balancing eligible tier-renderer strata."""
    strata: dict[tuple[str, int], deque[MathProblem]] = {}
    grouped: dict[tuple[str, int], list[MathProblem]] = defaultdict(list)
    for problem in candidates:
        grouped[(_tier(problem), _renderer(problem))].append(problem)
    for key, rows in grouped.items():
        strata[key] = deque(sorted(rows, key=lambda row: _seeded_order(row, selection_seed)))
    selected: list[MathProblem] = []
    while len(selected) < count and any(strata.values()):
        for key in sorted(strata):
            if strata[key] and len(selected) < count:
                selected.append(strata[key].popleft())
    if len(selected) != count:
        raise ValueError(f"only {len(selected)} eligible questions are available; need {count}")
    return selected


def select_screened_questions(
    questions: list[MathProblem],
    rollouts: list[Rollout],
    certificates: list[dict[str, object]],
    *,
    selection_seed: int = 20260901,
    expected_per_cell: int = 10,
    selected_per_family: int = 10,
    min_accuracy: float = 0.30,
    max_accuracy: float = 0.70,
    min_scorable_per_cell: int = 9,
    request_errors: int = 0,
    expected_model: str = "openai/gpt-oss-20b",
) -> tuple[dict[str, object], list[MathProblem], list[dict[str, object]]]:
    """Select a frozen bank using cell-level outcomes rather than item outcomes.

    A family-by-tier cell is eligible only when its aggregate screening accuracy lies in the
    preregistered band and it has sufficient scorable responses. Individual correctness never
    enters the within-cell selection order.
    """
    if request_errors < 0:
        raise ValueError("request_errors cannot be negative")
    by_id = {question.question_id: question for question in questions}
    if len(by_id) != len(questions):
        raise ValueError("question IDs must be unique")
    certificate_by_id = {str(row.get("question_id")): row for row in certificates}
    if set(certificate_by_id) != set(by_id):
        raise ValueError("certificate and question IDs must match exactly")
    invalid_certificates = sorted(
        question_id
        for question_id, question in by_id.items()
        if not verify_problem(question, certificate_by_id[question_id])
    )

    rollout_by_id: dict[str, list[Rollout]] = defaultdict(list)
    for rollout in rollouts:
        if rollout.question_id not in by_id:
            raise ValueError(f"unknown screening question ID: {rollout.question_id}")
        rollout_by_id[rollout.question_id].append(rollout)
    duplicate_rollouts = sorted(
        question_id
        for question_id in by_id
        if len(rollout_by_id.get(question_id, [])) != 1
    )
    invalid_rollout_design = sorted(
        rollout.rollout_id
        for rollout in rollouts
        if rollout.condition is not Condition.CLEAN or rollout.model != expected_model
    )

    cells: dict[tuple[str, str], list[MathProblem]] = defaultdict(list)
    for question in questions:
        cells[(_family(question), _tier(question))].append(question)
    malformed_cells = sorted(
        f"{family}:{tier}"
        for (family, tier), rows in cells.items()
        if len(rows) != expected_per_cell
    )

    cell_reports: list[dict[str, object]] = []
    eligible_cells: set[tuple[str, str]] = set()
    for (family, tier), rows in sorted(cells.items()):
        grades: Counter[str] = Counter()
        statuses: Counter[str] = Counter()
        for question in rows:
            question_rollouts = rollout_by_id.get(question.question_id, [])
            if len(question_rollouts) != 1:
                continue
            rollout = question_rollouts[0]
            statuses[str(rollout.status)] += 1
            if rollout.status is RolloutStatus.CLEAN_STOP:
                grades[str(grade_math_answer(rollout.parsed_answer, question.gold_answer))] += 1
        correct = grades[MathGrade.CORRECT]
        incorrect = grades[MathGrade.INCORRECT]
        scorable = correct + incorrect
        accuracy = correct / scorable if scorable else None
        eligible = (
            request_errors == 0
            and len(rows) == expected_per_cell
            and scorable >= min_scorable_per_cell
            and accuracy is not None
            and min_accuracy <= accuracy <= max_accuracy
        )
        if eligible:
            eligible_cells.add((family, tier))
        cell_reports.append(
            {
                "family": family,
                "tier": tier,
                "questions": len(rows),
                "scorable": scorable,
                "correct": correct,
                "incorrect": incorrect,
                "review": grades[MathGrade.REVIEW],
                "accuracy": accuracy,
                "statuses": dict(sorted(statuses.items())),
                "eligible": eligible,
            }
        )

    observed_families = {_family(question) for question in questions}
    observed_tiers = {_tier(question) for question in questions}
    unexpected_design = (
        observed_families != set(FAMILIES) or observed_tiers != set(TIERS)
    )
    families = list(FAMILIES)
    missing_families = [
        family for family in families if not any(cell[0] == family for cell in eligible_cells)
    ]
    selection_possible = not (
        request_errors
        or duplicate_rollouts
        or malformed_cells
        or missing_families
        or unexpected_design
        or invalid_certificates
        or invalid_rollout_design
    )
    selected: list[MathProblem] = []
    if selection_possible:
        for family in families:
            eligible_questions = [
                question
                for question in questions
                if _family(question) == family
                and (_family(question), _tier(question)) in eligible_cells
            ]
            if len(eligible_questions) < selected_per_family:
                selection_possible = False
                missing_families.append(family)
                selected = []
                break
            selected.extend(
                _balanced_select(
                    eligible_questions,
                    count=selected_per_family,
                    selection_seed=selection_seed,
                )
            )

    selected_ids = {problem.question_id for problem in selected}
    selected_certificates = [
        certificate_by_id[problem.question_id] for problem in selected
    ]
    report: dict[str, object] = {
        "protocol": "procedural-screening-v1",
        "questions": len(questions),
        "rollouts": len(rollouts),
        "request_errors": request_errors,
        "accuracy_band": [min_accuracy, max_accuracy],
        "min_scorable_per_cell": min_scorable_per_cell,
        "expected_per_cell": expected_per_cell,
        "selected_per_family": selected_per_family,
        "selection_seed": selection_seed,
        "duplicate_or_missing_rollout_ids": duplicate_rollouts,
        "invalid_certificate_question_ids": invalid_certificates,
        "invalid_condition_or_model_rollout_ids": invalid_rollout_design,
        "malformed_cells": malformed_cells,
        "unexpected_family_or_tier_design": unexpected_design,
        "missing_eligible_families": sorted(set(missing_families)),
        "cells": cell_reports,
        "selection_passed": selection_possible
        and len(selected) == len(families) * selected_per_family,
        "selected_question_ids": [problem.question_id for problem in selected],
        "selected_certificate_count": len(selected_certificates),
        "selected_id_count": len(selected_ids),
    }
    return report, selected, selected_certificates


def build_reasoning_effort_diagnostic(
    questions: list[MathProblem],
    screening_rollouts: list[Rollout],
    certificates: list[dict[str, object]],
    *,
    pair_count: int = 12,
    selection_seed: int = 20261101,
    expected_model: str = "openai/gpt-oss-20b",
) -> tuple[dict[str, object], list[MathProblem], list[dict[str, object]]]:
    """Freeze matched diagnostic problems from the failed medium-reasoning screen.

    The diagnostic deliberately conditions on prior truncation, so none of its outputs may enter
    the later mixed-outcome cohort or any monitor train, validation, or test split.
    """
    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    by_id = {question.question_id: question for question in questions}
    certificate_by_id = {str(row.get("question_id")): row for row in certificates}
    if len(by_id) != len(questions) or set(certificate_by_id) != set(by_id):
        raise ValueError("questions and certificates must have identical unique IDs")
    invalid_certificates = [
        question_id
        for question_id, question in by_id.items()
        if not verify_problem(question, certificate_by_id[question_id])
    ]
    if invalid_certificates:
        raise ValueError(f"invalid certificates: {sorted(invalid_certificates)}")

    rollout_by_id: dict[str, Rollout] = {}
    for rollout in screening_rollouts:
        if rollout.question_id not in by_id:
            raise ValueError(f"unknown screening question ID: {rollout.question_id}")
        if rollout.question_id in rollout_by_id:
            raise ValueError(f"duplicate screening rollout: {rollout.question_id}")
        if rollout.condition is not Condition.CLEAN or rollout.model != expected_model:
            raise ValueError("screening rollouts must use the clean condition and expected model")
        rollout_by_id[rollout.question_id] = rollout
    if set(rollout_by_id) != set(by_id):
        raise ValueError("every candidate requires exactly one screening rollout")

    def unambiguous(problem: MathProblem) -> bool:
        return not (
            _family(problem) == "recurrence" and _renderer(problem) == 1
        )

    truncated = [
        problem
        for problem in questions
        if unambiguous(problem)
        and rollout_by_id[problem.question_id].status is RolloutStatus.LENGTH_TRUNCATED
    ]
    controls = [
        problem
        for problem in questions
        if unambiguous(problem)
        and rollout_by_id[problem.question_id].status is RolloutStatus.CLEAN_STOP
        and grade_math_answer(
            rollout_by_id[problem.question_id].parsed_answer, problem.gold_answer
        )
        is MathGrade.CORRECT
    ]
    selected_truncated = _balanced_select(
        truncated, count=pair_count, selection_seed=selection_seed
    )
    tier_rank = {tier: index for index, tier in enumerate(TIERS)}
    unused_controls = {problem.question_id: problem for problem in controls}
    selected: list[MathProblem] = []
    selected_certificates: list[dict[str, object]] = []
    pairs: list[dict[str, object]] = []
    for index, truncated_problem in enumerate(selected_truncated):
        family_controls = [
            problem
            for problem in unused_controls.values()
            if _family(problem) == _family(truncated_problem)
        ]
        if not family_controls:
            raise ValueError(f"no unused control remains for {_family(truncated_problem)}")
        control = min(
            family_controls,
            key=lambda problem: (
                abs(tier_rank[_tier(problem)] - tier_rank[_tier(truncated_problem)]),
                _renderer(problem) != _renderer(truncated_problem),
                _seeded_order(problem, selection_seed + index),
            ),
        )
        unused_controls.pop(control.question_id)
        pair_id = f"low-reasoning-pair-{index:02d}"
        truncated_rollout = rollout_by_id[truncated_problem.question_id]
        control_rollout = rollout_by_id[control.question_id]
        pair_problems = (
            _diagnostic_problem(
                truncated_problem,
                pair_id=pair_id,
                stratum="previously_truncated",
                source_rollout=truncated_rollout,
            ),
            _diagnostic_problem(
                control,
                pair_id=pair_id,
                stratum="matched_clean_control",
                source_rollout=control_rollout,
            ),
        )
        for problem in pair_problems:
            selected.append(problem)
            selected_certificates.append(certificate_by_id[problem.question_id])
        pairs.append(
            {
                "pair_id": pair_id,
                "family": _family(truncated_problem),
                "previously_truncated_question_id": truncated_problem.question_id,
                "previously_truncated_tier": _tier(truncated_problem),
                "matched_control_question_id": control.question_id,
                "matched_control_tier": _tier(control),
            }
        )
    family_counts = Counter(_family(problem) for problem in selected_truncated)
    report: dict[str, object] = {
        "protocol": "low-reasoning-attribution-v1",
        "selection_seed": selection_seed,
        "pair_count": pair_count,
        "questions": len(selected),
        "excluded_ambiguous_renderer": "recurrence:renderer-1",
        "diagnostic_only": True,
        "excluded_from_monitor_data": True,
        "previously_truncated_family_counts": dict(sorted(family_counts.items())),
        "pairs": pairs,
    }
    return report, selected, selected_certificates


def analyze_reasoning_effort_diagnostic(
    questions: list[MathProblem],
    rollouts: list[Rollout],
    *,
    request_errors: int = 0,
    expected_model: str = "openai/gpt-oss-20b",
) -> dict[str, object]:
    """Evaluate whether low reasoning converts truncations into ordinary failures."""
    by_id = {question.question_id: question for question in questions}
    counts_by_question = Counter(rollout.question_id for rollout in rollouts)
    statuses: Counter[str] = Counter()
    grades: Counter[str] = Counter()
    by_stratum: dict[str, Counter[str]] = defaultdict(Counter)
    incorrect_families: set[str] = set()
    provider_ids: list[str] = []
    invalid_rollouts: list[str] = []
    reasoning_scorable = 0
    for rollout in rollouts:
        question = by_id.get(rollout.question_id)
        if question is None:
            invalid_rollouts.append(rollout.rollout_id)
            continue
        stratum = str(question.metadata.get("diagnostic_stratum", "missing"))
        statuses[str(rollout.status)] += 1
        by_stratum[stratum][str(rollout.status)] += 1
        if rollout.provider_request_id:
            provider_ids.append(rollout.provider_request_id)
        if rollout.condition is not Condition.CLEAN or rollout.model != expected_model:
            invalid_rollouts.append(rollout.rollout_id)
        if rollout.status is RolloutStatus.CLEAN_STOP:
            grade = grade_math_answer(rollout.parsed_answer, question.gold_answer)
            grades[str(grade)] += 1
            by_stratum[stratum][str(grade)] += 1
            if grade in {MathGrade.CORRECT, MathGrade.INCORRECT} and rollout.reasoning:
                reasoning_scorable += 1
            if grade is MathGrade.INCORRECT:
                incorrect_families.add(_family(question))

    correct = grades[MathGrade.CORRECT]
    incorrect = grades[MathGrade.INCORRECT]
    scorable = correct + incorrect
    truncated = statuses[RolloutStatus.LENGTH_TRUNCATED]
    prior_scorable = sum(
        by_stratum["previously_truncated"][grade]
        for grade in (MathGrade.CORRECT, MathGrade.INCORRECT)
    )
    control_scorable = sum(
        by_stratum["matched_clean_control"][grade]
        for grade in (MathGrade.CORRECT, MathGrade.INCORRECT)
    )
    pair_counts = Counter(
        str(question.metadata.get("diagnostic_pair_id")) for question in questions
    )
    strata = Counter(
        str(question.metadata.get("diagnostic_stratum")) for question in questions
    )
    design_valid = (
        len(questions) == 24
        and strata == Counter({"previously_truncated": 12, "matched_clean_control": 12})
        and len(pair_counts) == 12
        and set(pair_counts.values()) == {2}
        and all(
            not (_family(question) == "recurrence" and _renderer(question) == 1)
            for question in questions
        )
    )
    checks = {
        "valid_diagnostic_design": design_valid,
        "all_24_responses_stored": len(rollouts) == 24,
        "one_rollout_per_question": all(counts_by_question[key] == 1 for key in by_id),
        "at_least_20_scorable": scorable >= 20,
        "at_most_two_truncated": truncated <= 2,
        "at_least_10_prior_truncations_now_scorable": prior_scorable >= 10,
        "at_least_10_controls_scorable": control_scorable >= 10,
        "at_least_four_completed_errors": incorrect >= 4,
        "errors_span_two_families": len(incorrect_families) >= 2,
        "reasoning_present_for_scorable": reasoning_scorable == scorable,
        "unique_provider_request_ids": len(set(provider_ids)) == len(rollouts),
        "zero_request_errors": request_errors == 0,
        "clean_condition_and_expected_model": not invalid_rollouts,
    }
    return {
        "protocol": "low-reasoning-attribution-v1",
        "diagnostic_only": True,
        "questions": len(questions),
        "rollouts": len(rollouts),
        "statuses": dict(sorted(statuses.items())),
        "grades": dict(sorted(grades.items())),
        "scorable": scorable,
        "accuracy": correct / scorable if scorable else None,
        "incorrect_families": sorted(incorrect_families),
        "by_stratum": {
            key: dict(sorted(value.items())) for key, value in sorted(by_stratum.items())
        },
        "request_errors": request_errors,
        "gate_checks": checks,
        "diagnostic_gate_passed": all(checks.values()),
    }


def _clustered_accuracy_interval(
    grades_by_question: dict[str, list[MathGrade]], *, samples: int, seed: int
) -> dict[str, float | int | None]:
    """Bootstrap accuracy by resampling question clusters with replacement."""
    question_ids = sorted(grades_by_question)
    if not question_ids:
        return {"samples": samples, "seed": seed, "low": None, "high": None}
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        sampled = [rng.choice(question_ids) for _ in question_ids]
        grades = [grade for question_id in sampled for grade in grades_by_question[question_id]]
        correct = sum(grade is MathGrade.CORRECT for grade in grades)
        denominator = sum(grade in {MathGrade.CORRECT, MathGrade.INCORRECT} for grade in grades)
        if denominator:
            estimates.append(correct / denominator)
    estimates.sort()
    if not estimates:
        return {"samples": samples, "seed": seed, "low": None, "high": None}
    low_index = int(0.025 * (len(estimates) - 1))
    high_index = int(0.975 * (len(estimates) - 1))
    return {
        "samples": samples,
        "seed": seed,
        "low": estimates[low_index],
        "high": estimates[high_index],
    }


def analyze_frozen_discovery(
    questions: list[MathProblem],
    rollouts: list[Rollout],
    *,
    request_errors: int = 0,
    expected_samples_per_question: int = 3,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 20260902,
    expected_model: str = "openai/gpt-oss-20b",
) -> dict[str, object]:
    """Evaluate the preregistered clean-discovery readiness gate."""
    by_id = {question.question_id: question for question in questions}
    if len(by_id) != len(questions):
        raise ValueError("question IDs must be unique")
    expected_rollouts = len(questions) * expected_samples_per_question
    family_counts = Counter(_family(question) for question in questions)
    valid_frozen_design = len(questions) == 40 and family_counts == Counter(
        {family: 10 for family in FAMILIES}
    )
    statuses: Counter[str] = Counter()
    grades: Counter[str] = Counter()
    grades_by_question: dict[str, list[MathGrade]] = defaultdict(list)
    incorrect_questions: set[str] = set()
    incorrect_families: set[str] = set()
    correct_families: set[str] = set()
    family_tier: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    unknown_question_ids: set[str] = set()
    provider_ids: list[str] = []
    reasoning_scorable = 0
    counts_by_question: Counter[str] = Counter()
    seeds_by_question: dict[str, set[int]] = defaultdict(set)
    invalid_rollout_design: set[str] = set()

    for rollout in rollouts:
        statuses[str(rollout.status)] += 1
        if rollout.provider_request_id:
            provider_ids.append(rollout.provider_request_id)
        question = by_id.get(rollout.question_id)
        if question is None:
            unknown_question_ids.add(rollout.question_id)
            continue
        counts_by_question[rollout.question_id] += 1
        seeds_by_question[rollout.question_id].add(rollout.seed)
        if rollout.condition is not Condition.CLEAN or rollout.model != expected_model:
            invalid_rollout_design.add(rollout.rollout_id)
        key = (_family(question), _tier(question))
        family_tier[key][str(rollout.status)] += 1
        if rollout.status is not RolloutStatus.CLEAN_STOP:
            continue
        if rollout.reasoning:
            reasoning_scorable += 1
        grade = grade_math_answer(rollout.parsed_answer, question.gold_answer)
        grades[str(grade)] += 1
        grades_by_question[question.question_id].append(grade)
        family_tier[key][str(grade)] += 1
        if grade is MathGrade.INCORRECT:
            incorrect_questions.add(question.question_id)
            incorrect_families.add(_family(question))
        elif grade is MathGrade.CORRECT:
            correct_families.add(_family(question))

    correct = grades[MathGrade.CORRECT]
    incorrect = grades[MathGrade.INCORRECT]
    scorable = correct + incorrect
    truncation_rate = (
        statuses[RolloutStatus.LENGTH_TRUNCATED] / len(rollouts) if rollouts else 0.0
    )
    unique_provider_ids = len(set(provider_ids))
    exact_samples_per_question = all(
        counts_by_question[question_id] == expected_samples_per_question
        and len(seeds_by_question[question_id]) == expected_samples_per_question
        for question_id in by_id
    )
    unique_rollout_ids = len({rollout.rollout_id for rollout in rollouts})
    gate_checks = {
        "all_requests_stored": len(rollouts) == expected_rollouts,
        "exact_unique_samples_per_question": exact_samples_per_question,
        "at_least_90_percent_scorable": scorable >= 0.90 * expected_rollouts,
        "at_most_10_percent_truncated": truncation_rate <= 0.10,
        "at_least_24_correct": correct >= 24,
        "at_least_24_incorrect": incorrect >= 24,
        "errors_span_six_questions": len(incorrect_questions) >= 6,
        "errors_span_three_families": len(incorrect_families) >= 3,
        "correct_answers_span_three_families": len(correct_families) >= 3,
        "reasoning_present_for_scorable": reasoning_scorable == scorable,
        "unique_provider_request_ids": unique_provider_ids == len(rollouts),
        "unique_rollout_ids": unique_rollout_ids == len(rollouts),
        "zero_request_errors": request_errors == 0,
        "no_unknown_questions": not unknown_question_ids,
        "clean_condition_and_expected_model": not invalid_rollout_design,
        "balanced_frozen_family_design": valid_frozen_design,
    }
    breakdown = [
        {"family": family, "tier": tier, **dict(sorted(counts.items()))}
        for (family, tier), counts in sorted(family_tier.items())
    ]
    return {
        "protocol": "procedural-clean-discovery-v1",
        "questions": len(questions),
        "expected_samples_per_question": expected_samples_per_question,
        "expected_rollouts": expected_rollouts,
        "stored_rollouts": len(rollouts),
        "request_errors": request_errors,
        "statuses": dict(sorted(statuses.items())),
        "grades": dict(sorted(grades.items())),
        "scorable": scorable,
        "accuracy": correct / scorable if scorable else None,
        "clustered_accuracy_interval_95": _clustered_accuracy_interval(
            grades_by_question, samples=bootstrap_samples, seed=bootstrap_seed
        ),
        "truncation_rate": truncation_rate,
        "incorrect_question_ids": sorted(incorrect_questions),
        "incorrect_families": sorted(incorrect_families),
        "correct_families": sorted(correct_families),
        "unique_provider_request_ids": unique_provider_ids,
        "unique_rollout_ids": unique_rollout_ids,
        "unknown_question_ids": sorted(unknown_question_ids),
        "invalid_condition_or_model_rollout_ids": sorted(invalid_rollout_design),
        "frozen_family_counts": dict(sorted(family_counts.items())),
        "family_tier_breakdown": breakdown,
        "gate_checks": gate_checks,
        "task_readiness_gate_passed": all(gate_checks.values()),
    }
