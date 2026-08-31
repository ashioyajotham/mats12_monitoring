"""Qualification-informed three-family external replication for causal-audit-v2.1."""

from __future__ import annotations

import random
from collections import Counter

from src.causal_audit_v2 import (
    MECHANISMS,
    CorruptionMechanism,
    _digest,
    _generate_eligible_problem,
    _render_transformed_checkpoint,
)
from src.causal_error_dataset import _source_certificate_for_verifier
from src.datasets.procedural_math_v2 import GENERATOR_VERSION_V2, TIERS_V2, verify_problem_v2
from src.tasks import MathProblem

CAUSAL_AUDIT_V21_VERSION = "causal-audit-v2.1"
FAMILIES_V21 = ("affine_modular", "conditional_dag", "finite_state")
QUESTIONS_PER_CELL = 12


def _rebind_source(
    problem: MathProblem, source: dict[str, object]
) -> tuple[MathProblem, dict[str, object]]:
    """Rebind a generated source to the v2.1 namespace without changing its content."""
    question_id = problem.question_id.replace("cav2-", "cav21-", 1)
    core = {
        key: value
        for key, value in source.items()
        if key not in {"certificate_sha256", "study_generator_version"}
    }
    core["question_id"] = question_id
    rebound_source = {
        **core,
        "certificate_sha256": _digest(core),
        "study_generator_version": CAUSAL_AUDIT_V21_VERSION,
    }
    rebound_problem = problem.model_copy(
        update={
            "question_id": question_id,
            "task_family": "causal_audit_v21_math",
            "source": f"original:{CAUSAL_AUDIT_V21_VERSION}",
            "metadata": {
                **problem.metadata,
                "study_generator_version": CAUSAL_AUDIT_V21_VERSION,
                "lineage_id": question_id,
                "certificate_sha256": rebound_source["certificate_sha256"],
            },
        }
    )
    return rebound_problem, rebound_source


def build_causal_audit_v21(
    *, root_seed: int = 20262720, excluded_question_ids: set[str] | None = None
) -> tuple[list[MathProblem], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Build 72 fresh external questions balanced across six supported cells."""
    rng = random.Random(root_seed)
    excluded = excluded_question_ids or set()
    problems: list[MathProblem] = []
    sources: list[dict[str, object]] = []
    interventions: list[dict[str, object]] = []
    for family in FAMILIES_V21:
        for mechanism in MECHANISMS:
            for index in range(QUESTIONS_PER_CELL):
                cell = index % 4
                problem, source = _generate_eligible_problem(
                    rng=rng,
                    family=family,
                    tier=TIERS_V2[cell // 2],
                    renderer=cell % 2,
                    mechanism=mechanism,
                    index=index,
                )
                problem, source = _rebind_source(problem, source)
                if problem.question_id in excluded:
                    raise ValueError(
                        f"fresh bank overlaps excluded question: {problem.question_id}"
                    )
                correct, corrupted, target, checkpoint = _render_transformed_checkpoint(
                    problem, dict(source["parameters"]), mechanism
                )
                core: dict[str, object] = {
                    "question_id": problem.question_id,
                    "intervention_version": CAUSAL_AUDIT_V21_VERSION,
                    "family": family,
                    "mechanism": str(mechanism),
                    "source_generator_version": GENERATOR_VERSION_V2,
                    "source_certificate_sha256": source["certificate_sha256"],
                    "gold_answer": problem.gold_answer,
                    "target_answer": target,
                    "correct_prefix": correct,
                    "corrupted_prefix": corrupted,
                    "checkpoint": checkpoint,
                    "single_changed_field": "checkpoint_state_value",
                    "study_partition": "external_confirmatory",
                    "selection_basis": "causal-audit-v2 qualification-supported family",
                }
                intervention = {
                    **core,
                    "intervention_certificate_sha256": _digest(core),
                }
                problems.append(
                    problem.model_copy(
                        update={
                            "metadata": {
                                **problem.metadata,
                                "causal_yield_protocol": CAUSAL_AUDIT_V21_VERSION,
                                "corruption_mechanism": str(mechanism),
                                "continuation_correct_prefix": correct,
                                "continuation_corrupted_prefix": corrupted,
                                "intervention_target_answer": target,
                                "intervention_certificate_sha256": intervention[
                                    "intervention_certificate_sha256"
                                ],
                                "study_partition": "external_confirmatory",
                                "monitor_split": "external_test",
                                "excluded_from_monitor_data": False,
                                "eligible_for_external_audit": True,
                                "qualification_informed_family_selection": True,
                            }
                        }
                    )
                )
                sources.append(source)
                interventions.append(intervention)

    if len(problems) != 72 or len({row.question_id for row in problems}) != 72:
        raise AssertionError("causal-audit-v2.1 requires 72 unique questions")
    if len({row.prompt for row in problems}) != 72:
        raise AssertionError("causal-audit-v2.1 prompts must be unique")
    cell_counts = Counter(
        (str(row.metadata["generator_family"]), str(row.metadata["corruption_mechanism"]))
        for row in problems
    )
    selection = {
        "protocol": CAUSAL_AUDIT_V21_VERSION,
        "root_seed": root_seed,
        "questions": 72,
        "families": list(FAMILIES_V21),
        "excluded_family": "subset_counting",
        "exclusion_basis": "failed both causal-audit-v2 qualification mechanism cells",
        "mechanisms": [str(value) for value in MECHANISMS],
        "family_mechanism_counts": {
            f"{family}:{mechanism}": count
            for (family, mechanism), count in sorted(cell_counts.items())
        },
        "selection_uses_prior_qualification_outcomes": True,
        "selection_uses_v21_model_outcomes": False,
        "all_questions_frozen_before_v21_collection": True,
        "external_test_only": True,
    }
    return problems, sources, interventions, selection


def verify_causal_audit_v21_problem(
    problem: MathProblem,
    source: dict[str, object],
    intervention: dict[str, object],
) -> bool:
    """Recompute and bind a v2.1 source, transformation, target, and provenance."""
    core = {
        key: value
        for key, value in intervention.items()
        if key != "intervention_certificate_sha256"
    }
    if intervention.get("intervention_certificate_sha256") != _digest(core):
        return False
    if intervention.get("question_id") != problem.question_id:
        return False
    if not verify_problem_v2(problem, _source_certificate_for_verifier(source)):
        return False
    try:
        mechanism = CorruptionMechanism(str(intervention["mechanism"]))
        correct, corrupted, target, checkpoint = _render_transformed_checkpoint(
            problem, dict(source["parameters"]), mechanism
        )
    except (KeyError, TypeError, ValueError):
        return False
    return (
        intervention.get("intervention_version") == CAUSAL_AUDIT_V21_VERSION
        and intervention.get("source_certificate_sha256") == source.get("certificate_sha256")
        and intervention.get("correct_prefix") == correct
        and intervention.get("corrupted_prefix") == corrupted
        and intervention.get("target_answer") == target
        and intervention.get("checkpoint") == checkpoint
        and problem.metadata.get("study_generator_version") == CAUSAL_AUDIT_V21_VERSION
        and problem.metadata.get("corruption_mechanism") == str(mechanism)
        and problem.metadata.get("continuation_correct_prefix") == correct
        and problem.metadata.get("continuation_corrupted_prefix") == corrupted
        and problem.metadata.get("intervention_target_answer") == target
        and problem.metadata.get("study_partition") == "external_confirmatory"
        and problem.metadata.get("eligible_for_external_audit") is True
        and problem.metadata.get("qualification_informed_family_selection") is True
    )
