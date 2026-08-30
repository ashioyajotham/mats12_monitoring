"""Prospectively frozen gates for the causal-error-detection-v1 study."""

from __future__ import annotations

from collections import Counter, defaultdict

from src.causal_error_dataset import FAMILIES
from src.generate_rollouts import Rollout, RolloutStatus
from src.hints import Condition
from src.math_answers import MathGrade, grade_math_answer
from src.procedural_v2_pilot import _clustered_interval
from src.tasks import MathProblem


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
