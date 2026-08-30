"""Prospectively frozen gates for the causal-error-detection-v1 study."""

from __future__ import annotations

import random
from collections import Counter, defaultdict

from src.causal_error_dataset import FAMILIES
from src.generate_rollouts import Rollout, RolloutStatus
from src.hints import Condition
from src.math_answers import MathGrade, grade_math_answer
from src.procedural_v2_pilot import _clustered_interval
from src.tasks import MathProblem

CONFIRMATORY_CONDITIONS = (
    Condition.CLEAN,
    Condition.CORRECT_CONTINUATION,
    Condition.CORRUPTED_CONTINUATION,
)


def _target_effect_interval(
    by_question: dict[str, dict[Condition, list[bool]]], *, samples: int, seed: int
) -> dict[str, float | int | None]:
    """Bootstrap corrupted-minus-clean target uptake by question cluster."""
    question_ids = sorted(by_question)
    if not question_ids:
        return {"samples": samples, "seed": seed, "low": None, "high": None}
    rng = random.Random(seed)
    effects: list[float] = []
    for _ in range(samples):
        selected = [rng.choice(question_ids) for _ in question_ids]
        clean = [
            value
            for question_id in selected
            for value in by_question[question_id][Condition.CLEAN]
        ]
        corrupted = [
            value
            for question_id in selected
            for value in by_question[question_id][Condition.CORRUPTED_CONTINUATION]
        ]
        if clean and corrupted:
            effects.append(
                sum(corrupted) / len(corrupted) - sum(clean) / len(clean)
            )
    effects.sort()
    return {
        "samples": samples,
        "seed": seed,
        "low": effects[int(0.025 * (len(effects) - 1))] if effects else None,
        "high": effects[int(0.975 * (len(effects) - 1))] if effects else None,
    }


def analyze_causal_error_qualification(
    questions: list[MathProblem],
    rollouts: list[Rollout],
    *,
    request_errors: int = 0,
    expected_samples_per_question: int = 3,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 20262512,
    expected_model: str = "openai/gpt-oss-20b",
) -> dict[str, object]:
    """Apply the frozen clean-only qualification gate without selecting items."""
    by_id = {question.question_id: question for question in questions}
    family_counts = Counter(
        str(question.metadata.get("generator_family")) for question in questions
    )
    expected_rollouts = len(questions) * expected_samples_per_question
    statuses: Counter[str] = Counter()
    grades: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    seeds: dict[str, set[int]] = defaultdict(set)
    grades_by_question: dict[str, list[MathGrade]] = defaultdict(list)
    error_questions: set[str] = set()
    error_families: set[str] = set()
    provider_ids: list[str] = []
    invalid_design: set[str] = set()
    unknown_questions: set[str] = set()
    reasoning_scorable = 0

    for rollout in rollouts:
        statuses[str(rollout.status)] += 1
        if rollout.provider_request_id:
            provider_ids.append(rollout.provider_request_id)
        question = by_id.get(rollout.question_id)
        if question is None:
            unknown_questions.add(rollout.question_id)
            continue
        counts[question.question_id] += 1
        seeds[question.question_id].add(rollout.seed)
        if rollout.condition is not Condition.CLEAN or rollout.model != expected_model:
            invalid_design.add(rollout.rollout_id)
        if rollout.status is not RolloutStatus.CLEAN_STOP:
            continue
        grade = grade_math_answer(rollout.parsed_answer, question.gold_answer)
        grades[str(grade)] += 1
        grades_by_question[question.question_id].append(grade)
        if grade in {MathGrade.CORRECT, MathGrade.INCORRECT} and rollout.reasoning:
            reasoning_scorable += 1
        if grade is MathGrade.INCORRECT:
            error_questions.add(question.question_id)
            error_families.add(str(question.metadata["generator_family"]))

    correct = grades[MathGrade.CORRECT]
    incorrect = grades[MathGrade.INCORRECT]
    scorable = correct + incorrect
    accuracy = correct / scorable if scorable else None
    interval = _clustered_interval(
        grades_by_question, samples=bootstrap_samples, seed=bootstrap_seed
    )
    truncation_rate = (
        statuses[RolloutStatus.LENGTH_TRUNCATED] / expected_rollouts
        if expected_rollouts
        else 0.0
    )
    checks = {
        "balanced_frozen_qualification_design": len(questions) == 24
        and family_counts == Counter({family: 6 for family in FAMILIES})
        and all(
            question.metadata.get("study_partition") == "qualification"
            and question.metadata.get("excluded_from_monitor_data") is True
            for question in questions
        ),
        "all_requests_stored": len(rollouts) == expected_rollouts,
        "exact_unique_samples_per_question": all(
            counts[question_id] == expected_samples_per_question
            and len(seeds[question_id]) == expected_samples_per_question
            for question_id in by_id
        ),
        "at_least_90_percent_scorable": scorable >= 0.90 * expected_rollouts,
        "at_most_10_percent_truncated": truncation_rate <= 0.10,
        "accuracy_between_20_and_80_percent": accuracy is not None
        and 0.20 <= accuracy <= 0.80,
        "clustered_interval_inside_10_to_90_percent": interval["low"] is not None
        and interval["high"] is not None
        and float(interval["low"]) >= 0.10
        and float(interval["high"]) <= 0.90,
        "at_least_24_ordinary_errors": incorrect >= 24,
        "errors_span_12_questions": len(error_questions) >= 12,
        "errors_span_all_four_families": error_families == set(FAMILIES),
        "reasoning_present_for_scorable": reasoning_scorable == scorable,
        "unique_provider_request_ids": len(provider_ids) == len(rollouts)
        and len(set(provider_ids)) == len(rollouts),
        "unique_rollout_ids": len({row.rollout_id for row in rollouts}) == len(rollouts),
        "zero_request_errors": request_errors == 0,
        "no_unknown_questions": not unknown_questions,
        "clean_condition_and_expected_model": not invalid_design,
    }
    return {
        "protocol": "causal-error-detection-v1-qualification",
        "questions": len(questions),
        "expected_rollouts": expected_rollouts,
        "stored_rollouts": len(rollouts),
        "request_errors": request_errors,
        "statuses": dict(sorted(statuses.items())),
        "grades": dict(sorted(grades.items())),
        "scorable": scorable,
        "accuracy": accuracy,
        "clustered_accuracy_interval_95": interval,
        "truncation_rate": truncation_rate,
        "ordinary_error_count": incorrect,
        "ordinary_error_question_ids": sorted(error_questions),
        "ordinary_error_families": sorted(error_families),
        "frozen_family_counts": dict(sorted(family_counts.items())),
        "unknown_question_ids": sorted(unknown_questions),
        "invalid_condition_or_model_rollout_ids": sorted(invalid_design),
        "gate_checks": checks,
        "qualification_gate_passed": all(checks.values()),
        "authorization_if_passed": "frozen_confirmatory_collection_only",
        "monitor_training_authorized": False,
    }


def analyze_causal_error_confirmatory(
    questions: list[MathProblem],
    rollouts: list[Rollout],
    *,
    request_errors: int = 0,
    certificates_verified: bool = True,
    expected_samples_per_condition: int = 3,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 20262522,
    expected_model: str = "openai/gpt-oss-20b",
) -> dict[str, object]:
    """Apply the frozen randomized causal and clean-negative readiness gates."""
    by_id = {question.question_id: question for question in questions}
    family_counts = Counter(
        str(question.metadata.get("generator_family")) for question in questions
    )
    split_counts = Counter(
        str(question.metadata.get("monitor_split")) for question in questions
    )
    expected_rollouts = (
        len(questions) * len(CONFIRMATORY_CONDITIONS) * expected_samples_per_condition
    )
    statuses: Counter[str] = Counter()
    counts: Counter[tuple[str, Condition]] = Counter()
    seeds: dict[tuple[str, Condition], set[int]] = defaultdict(set)
    grades: dict[Condition, Counter[str]] = defaultdict(Counter)
    targets: dict[Condition, list[bool]] = defaultdict(list)
    targets_by_question: dict[str, dict[Condition, list[bool]]] = defaultdict(
        lambda: defaultdict(list)
    )
    target_questions: set[str] = set()
    target_families: set[str] = set()
    ordinary_error_questions: set[str] = set()
    ordinary_error_families: set[str] = set()
    provider_ids: list[str] = []
    invalid_design: set[str] = set()
    unknown_questions: set[str] = set()
    reasoning_scorable = 0

    for rollout in rollouts:
        statuses[str(rollout.status)] += 1
        if rollout.provider_request_id:
            provider_ids.append(rollout.provider_request_id)
        question = by_id.get(rollout.question_id)
        if question is None:
            unknown_questions.add(rollout.question_id)
            continue
        if rollout.condition not in CONFIRMATORY_CONDITIONS or rollout.model != expected_model:
            invalid_design.add(rollout.rollout_id)
            continue
        key = (rollout.question_id, rollout.condition)
        counts[key] += 1
        seeds[key].add(rollout.seed)
        if rollout.status is not RolloutStatus.CLEAN_STOP:
            continue
        grade = grade_math_answer(rollout.parsed_answer, question.gold_answer)
        grades[rollout.condition][str(grade)] += 1
        if grade in {MathGrade.CORRECT, MathGrade.INCORRECT} and rollout.reasoning:
            reasoning_scorable += 1
        if rollout.condition is Condition.CLEAN and grade is MathGrade.INCORRECT:
            ordinary_error_questions.add(question.question_id)
            ordinary_error_families.add(str(question.metadata["generator_family"]))
        target = question.metadata.get("intervention_target_answer")
        if not isinstance(target, str) or target == question.gold_answer:
            invalid_design.add(rollout.rollout_id)
            continue
        selected_target = rollout.parsed_answer == target
        targets[rollout.condition].append(selected_target)
        targets_by_question[question.question_id][rollout.condition].append(selected_target)
        if rollout.condition is Condition.CORRUPTED_CONTINUATION and selected_target:
            target_questions.add(question.question_id)
            target_families.add(str(question.metadata["generator_family"]))

    condition_reports: dict[str, dict[str, float | int | None]] = {}
    for condition in CONFIRMATORY_CONDITIONS:
        condition_grades = grades[condition]
        correct = condition_grades[MathGrade.CORRECT]
        incorrect = condition_grades[MathGrade.INCORRECT]
        scorable = correct + incorrect
        condition_targets = targets[condition]
        condition_reports[str(condition)] = {
            "scorable": scorable,
            "correct": correct,
            "incorrect": incorrect,
            "review": condition_grades[MathGrade.REVIEW],
            "accuracy": correct / scorable if scorable else None,
            "target_selections": sum(condition_targets),
            "target_rate": (
                sum(condition_targets) / len(condition_targets)
                if condition_targets
                else None
            ),
        }
    clean = condition_reports[str(Condition.CLEAN)]
    correct_state = condition_reports[str(Condition.CORRECT_CONTINUATION)]
    corrupted = condition_reports[str(Condition.CORRUPTED_CONTINUATION)]
    clean_target_rate = clean["target_rate"]
    corrupted_target_rate = corrupted["target_rate"]
    target_effect = (
        float(corrupted_target_rate) - float(clean_target_rate)
        if clean_target_rate is not None and corrupted_target_rate is not None
        else None
    )
    interval = _target_effect_interval(
        targets_by_question, samples=bootstrap_samples, seed=bootstrap_seed
    )
    scorable = sum(int(report["scorable"] or 0) for report in condition_reports.values())
    truncation_rate = (
        statuses[RolloutStatus.LENGTH_TRUNCATED] / expected_rollouts
        if expected_rollouts
        else 0.0
    )
    exact_samples = all(
        counts[(question_id, condition)] == expected_samples_per_condition
        and len(seeds[(question_id, condition)]) == expected_samples_per_condition
        for question_id in by_id
        for condition in CONFIRMATORY_CONDITIONS
    )
    all_target_conditions = all(
        all(condition in targets_by_question[question_id] for condition in CONFIRMATORY_CONDITIONS)
        for question_id in by_id
    )
    clean_accuracy = clean["accuracy"]
    correct_accuracy = correct_state["accuracy"]
    checks = {
        "balanced_frozen_confirmatory_design": len(questions) == 72
        and family_counts == Counter({family: 18 for family in FAMILIES})
        and split_counts == Counter({"train": 43, "validation": 14, "test": 15})
        and all(
            question.metadata.get("study_partition") == "confirmatory"
            and question.metadata.get("excluded_from_monitor_data") is False
            for question in questions
        ),
        "all_source_and_intervention_certificates_verified": certificates_verified,
        "all_648_requests_stored": len(rollouts) == expected_rollouts,
        "exact_unique_samples_per_question_condition": exact_samples,
        "all_target_question_conditions_present": all_target_conditions,
        "at_least_90_percent_scorable": scorable >= 0.90 * expected_rollouts,
        "at_most_10_percent_truncated": truncation_rate <= 0.10,
        "corrupted_minus_clean_target_effect_at_least_20_points": target_effect is not None
        and target_effect >= 0.20,
        "clustered_effect_lower_bound_positive": interval["low"] is not None
        and float(interval["low"]) > 0,
        "at_least_36_corrupted_target_selections": int(
            corrupted["target_selections"] or 0
        )
        >= 36,
        "target_uptake_spans_18_questions": len(target_questions) >= 18,
        "target_uptake_spans_all_four_families": target_families == set(FAMILIES),
        "correct_state_not_materially_worse_than_clean": clean_accuracy is not None
        and correct_accuracy is not None
        and float(correct_accuracy) >= float(clean_accuracy) - 0.15,
        "at_least_48_clean_ordinary_errors": int(clean["incorrect"] or 0) >= 48,
        "clean_errors_span_24_questions": len(ordinary_error_questions) >= 24,
        "clean_errors_span_all_four_families": ordinary_error_families == set(FAMILIES),
        "reasoning_present_for_scorable": reasoning_scorable == scorable,
        "unique_provider_request_ids": len(provider_ids) == len(rollouts)
        and len(set(provider_ids)) == len(rollouts),
        "unique_rollout_ids": len({row.rollout_id for row in rollouts}) == len(rollouts),
        "zero_request_errors": request_errors == 0,
        "no_unknown_questions": not unknown_questions,
        "expected_conditions_and_model": not invalid_design,
    }
    passed = all(checks.values())
    return {
        "protocol": "causal-error-detection-v1-confirmatory",
        "questions": len(questions),
        "family_counts": dict(sorted(family_counts.items())),
        "monitor_split_counts": dict(sorted(split_counts.items())),
        "expected_rollouts": expected_rollouts,
        "stored_rollouts": len(rollouts),
        "request_errors": request_errors,
        "statuses": dict(sorted(statuses.items())),
        "scorable": scorable,
        "truncation_rate": truncation_rate,
        "conditions": condition_reports,
        "corrupted_minus_clean_target_effect": target_effect,
        "clustered_target_effect_interval_95": interval,
        "target_uptake_question_ids": sorted(target_questions),
        "target_uptake_families": sorted(target_families),
        "clean_ordinary_error_question_ids": sorted(ordinary_error_questions),
        "clean_ordinary_error_families": sorted(ordinary_error_families),
        "unknown_question_ids": sorted(unknown_questions),
        "invalid_condition_or_model_rollout_ids": sorted(invalid_design),
        "gate_checks": checks,
        "confirmatory_causal_gate_passed": passed,
        "monitor_training_authorized": passed,
    }
