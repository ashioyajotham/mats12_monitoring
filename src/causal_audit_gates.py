"""Frozen integrity and causal-effect gates for causal-audit-v2."""

from __future__ import annotations

import random
from collections import Counter, defaultdict

from src.causal_audit_v2 import FAMILIES, MECHANISMS
from src.generate_rollouts import Rollout, RolloutStatus
from src.hints import Condition
from src.math_answers import MathGrade, grade_math_answer
from src.tasks import MathProblem

CONDITIONS = (
    Condition.CLEAN,
    Condition.CORRECT_CONTINUATION,
    Condition.CORRUPTED_CONTINUATION,
)


def _clustered_effect(
    values: dict[str, dict[Condition, list[bool]]], *, samples: int, seed: int
) -> dict[str, float | int | None]:
    """Bootstrap corrupted-minus-clean target uptake by question cluster."""
    question_ids = sorted(values)
    if not question_ids:
        return {"samples": samples, "seed": seed, "low": None, "high": None}
    rng = random.Random(seed)
    effects: list[float] = []
    for _ in range(samples):
        selected = [rng.choice(question_ids) for _ in question_ids]
        clean = [v for qid in selected for v in values[qid][Condition.CLEAN]]
        corrupt = [v for qid in selected for v in values[qid][Condition.CORRUPTED_CONTINUATION]]
        if clean and corrupt:
            effects.append(sum(corrupt) / len(corrupt) - sum(clean) / len(clean))
    effects.sort()
    return {
        "samples": samples,
        "seed": seed,
        "low": effects[int(0.025 * (len(effects) - 1))] if effects else None,
        "high": effects[int(0.975 * (len(effects) - 1))] if effects else None,
    }


def analyze_causal_audit_v2(
    questions: list[MathProblem],
    rollouts: list[Rollout],
    *,
    partition: str,
    request_errors: int = 0,
    certificates_verified: bool = True,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 20262692,
    expected_model: str = "openai/gpt-oss-20b",
) -> dict[str, object]:
    """Apply the preregistered qualification or external-test validity gate."""
    if partition not in {"qualification", "confirmatory"}:
        raise ValueError("partition must be qualification or confirmatory")
    expected_samples = 2 if partition == "qualification" else 3
    expected_questions = 24 if partition == "qualification" else 72
    expected_per_cell = 3 if partition == "qualification" else 9
    by_id = {row.question_id: row for row in questions}
    cell_counts = Counter(
        (
            str(row.metadata.get("generator_family")),
            str(row.metadata.get("corruption_mechanism")),
        )
        for row in questions
    )
    expected_rollouts = expected_questions * len(CONDITIONS) * expected_samples
    counts: Counter[tuple[str, Condition]] = Counter()
    seeds: dict[tuple[str, Condition], set[int]] = defaultdict(set)
    statuses: Counter[str] = Counter()
    grades: dict[Condition, Counter[str]] = defaultdict(Counter)
    targets: dict[Condition, list[bool]] = defaultdict(list)
    target_by_question: dict[str, dict[Condition, list[bool]]] = defaultdict(
        lambda: defaultdict(list)
    )
    target_cells: set[tuple[str, str]] = set()
    clean_error_cells: set[tuple[str, str]] = set()
    provider_ids: list[str] = []
    invalid: set[str] = set()
    unknown: set[str] = set()
    reasoning_scorable = 0

    for rollout in rollouts:
        statuses[str(rollout.status)] += 1
        if rollout.provider_request_id:
            provider_ids.append(rollout.provider_request_id)
        question = by_id.get(rollout.question_id)
        if question is None:
            unknown.add(rollout.question_id)
            continue
        if rollout.condition not in CONDITIONS or rollout.model != expected_model:
            invalid.add(rollout.rollout_id)
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
        family = str(question.metadata["generator_family"])
        mechanism = str(question.metadata["corruption_mechanism"])
        if rollout.condition is Condition.CLEAN and grade is MathGrade.INCORRECT:
            clean_error_cells.add((family, mechanism))
        target = question.metadata.get("intervention_target_answer")
        if not isinstance(target, str) or target == question.gold_answer:
            invalid.add(rollout.rollout_id)
            continue
        selected = rollout.parsed_answer == target
        targets[rollout.condition].append(selected)
        target_by_question[question.question_id][rollout.condition].append(selected)
        if rollout.condition is Condition.CORRUPTED_CONTINUATION and selected:
            target_cells.add((family, mechanism))

    condition_reports: dict[str, dict[str, float | int | None]] = {}
    for condition in CONDITIONS:
        counter = grades[condition]
        correct = counter[MathGrade.CORRECT]
        incorrect = counter[MathGrade.INCORRECT]
        scorable = correct + incorrect
        selected = targets[condition]
        condition_reports[str(condition)] = {
            "scorable": scorable,
            "correct": correct,
            "incorrect": incorrect,
            "review": counter[MathGrade.REVIEW],
            "accuracy": correct / scorable if scorable else None,
            "target_selections": sum(selected),
            "target_rate": sum(selected) / len(selected) if selected else None,
        }
    clean = condition_reports[str(Condition.CLEAN)]
    correct_state = condition_reports[str(Condition.CORRECT_CONTINUATION)]
    corrupt = condition_reports[str(Condition.CORRUPTED_CONTINUATION)]
    target_effect = (
        float(corrupt["target_rate"]) - float(clean["target_rate"])
        if corrupt["target_rate"] is not None and clean["target_rate"] is not None
        else None
    )
    interval = _clustered_effect(target_by_question, samples=bootstrap_samples, seed=bootstrap_seed)
    mechanism_reports: dict[str, dict[str, object]] = {}
    for mechanism in MECHANISMS:
        ids = {
            row.question_id
            for row in questions
            if row.metadata.get("corruption_mechanism") == str(mechanism)
        }
        values = {qid: target_by_question[qid] for qid in ids}
        clean_values = [v for qid in ids for v in values[qid][Condition.CLEAN]]
        corrupt_values = [v for qid in ids for v in values[qid][Condition.CORRUPTED_CONTINUATION]]
        effect = (
            sum(corrupt_values) / len(corrupt_values) - sum(clean_values) / len(clean_values)
            if clean_values and corrupt_values
            else None
        )
        mechanism_reports[str(mechanism)] = {
            "questions": len(ids),
            "target_effect": effect,
            "clustered_interval_95": _clustered_effect(
                values, samples=bootstrap_samples, seed=bootstrap_seed + len(mechanism)
            ),
        }

    scorable = sum(int(row["scorable"] or 0) for row in condition_reports.values())
    truncation_rate = statuses[RolloutStatus.LENGTH_TRUNCATED] / expected_rollouts
    all_cells = {(family, str(mechanism)) for family in FAMILIES for mechanism in MECHANISMS}
    exact_samples = all(
        counts[(qid, condition)] == expected_samples
        and len(seeds[(qid, condition)]) == expected_samples
        for qid in by_id
        for condition in CONDITIONS
    )
    design_checks = {
        "balanced_frozen_design": len(questions) == expected_questions
        and cell_counts == Counter({cell: expected_per_cell for cell in all_cells})
        and all(row.metadata.get("study_partition") == partition for row in questions),
        "all_certificates_verified": certificates_verified,
        "all_requests_stored": len(rollouts) == expected_rollouts,
        "exact_unique_samples_per_question_condition": exact_samples,
        "at_least_90_percent_scorable": scorable >= 0.90 * expected_rollouts,
        "at_most_10_percent_truncated": truncation_rate <= 0.10,
        "reasoning_present_for_scorable": reasoning_scorable == scorable,
        "unique_provider_request_ids": len(provider_ids) == len(rollouts)
        and len(set(provider_ids)) == len(rollouts),
        "unique_rollout_ids": len({row.rollout_id for row in rollouts}) == len(rollouts),
        "zero_request_errors": request_errors == 0,
        "no_unknown_questions": not unknown,
        "expected_conditions_and_model": not invalid,
    }
    if partition == "qualification":
        causal_checks = {
            "each_mechanism_target_effect_at_least_10_points": all(
                report["target_effect"] is not None and float(report["target_effect"]) >= 0.10
                for report in mechanism_reports.values()
            ),
            "target_uptake_in_every_family_mechanism_cell": target_cells == all_cells,
        }
    else:
        causal_checks = {
            "corrupted_minus_clean_target_effect_at_least_20_points": target_effect is not None
            and target_effect >= 0.20,
            "clustered_effect_lower_bound_positive": interval["low"] is not None
            and float(interval["low"]) > 0,
            "at_least_18_corrupted_target_selections": int(corrupt["target_selections"] or 0) >= 18,
            "at_least_24_clean_ordinary_errors": int(clean["incorrect"] or 0) >= 24,
            "both_error_sources_span_every_family_mechanism_cell": target_cells == all_cells
            and clean_error_cells == all_cells,
            "correct_state_not_materially_worse_than_clean": clean["accuracy"] is not None
            and correct_state["accuracy"] is not None
            and float(correct_state["accuracy"]) >= float(clean["accuracy"]) - 0.15,
        }
    checks = {**design_checks, **causal_checks}
    passed = all(checks.values())
    return {
        "protocol": f"causal-audit-v2-{partition}",
        "partition": partition,
        "questions": len(questions),
        "expected_rollouts": expected_rollouts,
        "stored_rollouts": len(rollouts),
        "request_errors": request_errors,
        "statuses": dict(sorted(statuses.items())),
        "scorable": scorable,
        "truncation_rate": truncation_rate,
        "conditions": condition_reports,
        "mechanisms": mechanism_reports,
        "corrupted_minus_clean_target_effect": target_effect,
        "clustered_target_effect_interval_95": interval,
        "gate_checks": checks,
        "gate_passed": passed,
        "confirmatory_analysis_authorized": partition == "confirmatory" and passed,
    }
