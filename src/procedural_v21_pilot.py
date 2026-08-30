"""Adaptive subset replacement and combined clean gate for procedural v2.1."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict, deque

from src.datasets.procedural_math_v2 import FAMILIES_V2
from src.datasets.procedural_math_v21 import (
    FAMILY_V21,
    TIERS_V21,
    verify_subset_replacement,
)
from src.generate_rollouts import Rollout, RolloutStatus
from src.hints import Condition
from src.math_answers import MathGrade, grade_math_answer
from src.procedural_v2_pilot import analyze_mixed_outcome_v2, select_screened_questions_v2
from src.tasks import MathProblem


def _family(problem: MathProblem) -> str:
    """Return validated family metadata."""
    value = problem.metadata.get("generator_family")
    if not isinstance(value, str):
        raise ValueError(f"{problem.question_id} lacks generator_family")
    return value


def _tier(problem: MathProblem) -> str:
    """Return validated tier metadata."""
    value = problem.metadata.get("difficulty_tier")
    if not isinstance(value, str):
        raise ValueError(f"{problem.question_id} lacks difficulty_tier")
    return value


def _renderer(problem: MathProblem) -> int:
    """Return validated renderer metadata."""
    value = problem.metadata.get("renderer_id")
    if not isinstance(value, int):
        raise ValueError(f"{problem.question_id} lacks renderer_id")
    return value


def _order(problem: MathProblem, seed: int) -> str:
    """Produce an outcome-blind stable question order."""
    return hashlib.sha256(f"{seed}|{problem.question_id}".encode()).hexdigest()


def _balanced_select(
    candidates: list[MathProblem], *, count: int, selection_seed: int
) -> list[MathProblem]:
    """Select round-robin across tier-renderer strata without item outcomes."""
    grouped: dict[tuple[str, int], list[MathProblem]] = defaultdict(list)
    for problem in candidates:
        grouped[(_tier(problem), _renderer(problem))].append(problem)
    queues = {
        key: deque(sorted(rows, key=lambda row: _order(row, selection_seed)))
        for key, rows in grouped.items()
    }
    selected: list[MathProblem] = []
    while len(selected) < count and any(queues.values()):
        for key in sorted(queues):
            if queues[key] and len(selected) < count:
                selected.append(queues[key].popleft())
    if len(selected) != count:
        raise ValueError(f"only {len(selected)} eligible questions; need {count}")
    return selected


def _replacement_screen(
    questions: list[MathProblem],
    rollouts: list[Rollout],
    certificates: list[dict[str, object]],
    *,
    request_errors: int,
    expected_model: str,
) -> tuple[dict[str, object], set[str]]:
    """Apply unchanged cell gates to the fresh subset replacements."""
    by_id = {row.question_id: row for row in questions}
    certificate_by_id = {str(row.get("question_id")): row for row in certificates}
    if len(by_id) != len(questions) or set(certificate_by_id) != set(by_id):
        raise ValueError("replacement questions and certificates require identical unique IDs")
    invalid_certificates = sorted(
        question_id
        for question_id, question in by_id.items()
        if not verify_subset_replacement(question, certificate_by_id[question_id])
    )
    rollout_by_id: dict[str, list[Rollout]] = defaultdict(list)
    unknown: list[str] = []
    for rollout in rollouts:
        if rollout.question_id not in by_id:
            unknown.append(rollout.rollout_id)
        else:
            rollout_by_id[rollout.question_id].append(rollout)
    duplicate_or_missing = sorted(
        question_id for question_id in by_id if len(rollout_by_id[question_id]) != 1
    )
    invalid_design = sorted(
        rollout.rollout_id
        for rollout in rollouts
        if rollout.condition is not Condition.CLEAN or rollout.model != expected_model
    )
    cells: dict[str, list[MathProblem]] = defaultdict(list)
    for question in questions:
        if _family(question) != FAMILY_V21:
            raise ValueError("replacement bank contains a non-subset family")
        cells[_tier(question)].append(question)
    malformed = sorted(tier for tier in TIERS_V21 if len(cells[tier]) != 10)
    unexpected_tiers = set(cells) != set(TIERS_V21)
    eligible_tiers: set[str] = set()
    cell_reports: list[dict[str, object]] = []
    for tier in TIERS_V21:
        grades: Counter[str] = Counter()
        statuses: Counter[str] = Counter()
        for question in cells[tier]:
            rows = rollout_by_id[question.question_id]
            if len(rows) != 1:
                continue
            rollout = rows[0]
            statuses[str(rollout.status)] += 1
            if rollout.status is RolloutStatus.CLEAN_STOP:
                grades[str(grade_math_answer(rollout.parsed_answer, question.gold_answer))] += 1
        correct, incorrect = grades[MathGrade.CORRECT], grades[MathGrade.INCORRECT]
        scorable = correct + incorrect
        accuracy = correct / scorable if scorable else None
        eligible = bool(
            request_errors == 0
            and len(cells[tier]) == 10
            and scorable >= 9
            and accuracy is not None
            and 0.30 <= accuracy <= 0.70
        )
        if eligible:
            eligible_tiers.add(tier)
        cell_reports.append({
            "family": FAMILY_V21, "tier": tier, "questions": len(cells[tier]),
            "scorable": scorable, "correct": correct, "incorrect": incorrect,
            "review": grades[MathGrade.REVIEW], "accuracy": accuracy,
            "statuses": dict(sorted(statuses.items())), "eligible": eligible,
        })
    checks = {
        "exact_20_question_design": len(questions) == 20 and not malformed and not unexpected_tiers,
        "one_rollout_per_question": not duplicate_or_missing and len(rollouts) == 20,
        "valid_certificates": not invalid_certificates,
        "clean_condition_and_expected_model": not invalid_design,
        "no_unknown_rollouts": not unknown,
        "zero_request_errors": request_errors == 0,
        "at_least_one_eligible_tier": bool(eligible_tiers),
    }
    return ({
        "protocol": "procedural-v2.1-subset-screening",
        "questions": len(questions), "rollouts": len(rollouts),
        "accuracy_band": [0.30, 0.70], "min_scorable_per_cell": 9,
        "request_errors": request_errors, "cells": cell_reports,
        "eligible_tiers": sorted(eligible_tiers),
        "invalid_certificate_question_ids": invalid_certificates,
        "duplicate_or_missing_rollout_ids": duplicate_or_missing,
        "invalid_condition_or_model_rollout_ids": invalid_design,
        "unknown_rollout_ids": unknown, "gate_checks": checks,
        "replacement_screen_passed": all(checks.values()),
    }, eligible_tiers)


def select_combined_pilot_v21(
    v2_questions: list[MathProblem],
    v2_rollouts: list[Rollout],
    v2_certificates: list[dict[str, object]],
    replacement_questions: list[MathProblem],
    replacement_rollouts: list[Rollout],
    replacement_certificates: list[dict[str, object]],
    *,
    v2_request_errors: int = 0,
    replacement_request_errors: int = 0,
    v2_selection_seed: int = 20261301,
    replacement_selection_seed: int = 20261701,
    expected_model: str = "openai/gpt-oss-20b",
) -> tuple[dict[str, object], list[MathProblem], list[dict[str, object]]]:
    """Combine three eligible v2 families with a fresh eligible subset tier."""
    v2_report, _, _ = select_screened_questions_v2(
        v2_questions, v2_rollouts, v2_certificates,
        selection_seed=v2_selection_seed, request_errors=v2_request_errors,
        expected_model=expected_model,
    )
    old_families = tuple(family for family in FAMILIES_V2 if family != FAMILY_V21)
    eligible_v2_cells = {
        (str(cell["family"]), str(cell["tier"]))
        for cell in v2_report["cells"] if cell["eligible"]
    }
    valid_v2_calibration = (
        not v2_report["selection_passed"]
        and v2_report["missing_eligible_families"] == [FAMILY_V21]
        and all(any(cell[0] == family for cell in eligible_v2_cells) for family in old_families)
        and not v2_report["invalid_certificate_question_ids"]
        and not v2_report["duplicate_or_missing_rollout_ids"]
        and not v2_report["invalid_condition_or_model_rollout_ids"]
        and v2_report["request_errors"] == 0
    )
    replacement_report, eligible_replacement_tiers = _replacement_screen(
        replacement_questions, replacement_rollouts, replacement_certificates,
        request_errors=replacement_request_errors, expected_model=expected_model,
    )
    passed = valid_v2_calibration and replacement_report["replacement_screen_passed"]
    selected: list[MathProblem] = []
    certificate_by_id = {
        str(row["question_id"]): row for row in [*v2_certificates, *replacement_certificates]
    }
    if passed:
        for family in old_families:
            candidates = [
                question for question in v2_questions
                if _family(question) == family
                and (family, _tier(question)) in eligible_v2_cells
            ]
            selected.extend(_balanced_select(
                candidates, count=10, selection_seed=v2_selection_seed
            ))
        replacement_candidates = [
            question for question in replacement_questions
            if _tier(question) in eligible_replacement_tiers
        ]
        selected.extend(_balanced_select(
            replacement_candidates, count=10, selection_seed=replacement_selection_seed
        ))
    selected_certificates = [certificate_by_id[row.question_id] for row in selected]
    report: dict[str, object] = {
        "protocol": "procedural-screening-v2.1-adaptive-subset-replacement",
        "adaptation_scope": "replace_failed_subset_counting_family_only",
        "v2_calibration_valid": valid_v2_calibration,
        "v2_eligible_cells_reused": [f"{family}:{tier}" for family, tier in sorted(
            cell for cell in eligible_v2_cells if cell[0] in old_families
        )],
        "replacement_screen": replacement_report,
        "v2_selection_seed": v2_selection_seed,
        "replacement_selection_seed": replacement_selection_seed,
        "selected_question_ids": [row.question_id for row in selected],
        "selected_family_counts": dict(sorted(Counter(_family(row) for row in selected).items())),
        "selection_passed": passed and len(selected) == 40 and len(selected_certificates) == 40,
        "individual_screening_outcomes_used_for_selection": False,
    }
    return report, selected, selected_certificates


def analyze_mixed_outcome_v21(
    questions: list[MathProblem], rollouts: list[Rollout], *, request_errors: int = 0
) -> dict[str, object]:
    """Apply the unchanged fresh mixed-outcome gate to the combined v2.1 cohort."""
    report = analyze_mixed_outcome_v2(
        questions, rollouts, request_errors=request_errors, bootstrap_seed=20261802
    )
    report["protocol"] = "procedural-clean-mixed-outcome-v2.1"
    report["adaptive_calibration_then_fresh_validation"] = True
    return report
