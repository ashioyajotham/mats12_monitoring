"""Mechanism-held-out construction for the causal-audit-v2 replication."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from enum import StrEnum

from sympy.ntheory.modular import solve_congruence

from src.causal_error_dataset import _source_certificate_for_verifier
from src.continuation_interventions import _BUILDERS
from src.datasets.procedural_math_v2 import (
    _GENERATORS_V2,
    FAMILIES_V2,
    GENERATOR_VERSION_V2,
    TIERS_V2,
    _render_conditional_dag,
    verify_problem_v2,
)
from src.tasks import MathProblem

CAUSAL_AUDIT_VERSION = "causal-audit-v2"
FAMILIES = FAMILIES_V2
QUALIFICATION_PER_CELL = 3
CONFIRMATORY_PER_CELL = 9


class CorruptionMechanism(StrEnum):
    """Prospectively frozen transformations unseen in causal-error-v1."""

    DROP_COMPONENT = "drop_component"
    DUPLICATE_COMPONENT = "duplicate_component"


MECHANISMS = tuple(CorruptionMechanism)


def _digest(payload: object) -> str:
    """Hash a JSON-compatible payload using its canonical representation."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _transformed_value(value: int, mechanism: CorruptionMechanism) -> int:
    """Apply one frozen omission-or-duplication transformation."""
    if mechanism is CorruptionMechanism.DROP_COMPONENT:
        return 0
    return 2 * value


def _render_transformed_checkpoint(
    problem: MathProblem,
    parameters: dict[str, object],
    mechanism: CorruptionMechanism,
) -> tuple[str, str, str, dict[str, object]]:
    """Transform the v1-selected checkpoint and independently propagate its target."""
    family = str(problem.metadata["generator_family"])
    correct, _, _, original = _BUILDERS[family](problem, parameters)
    correct_value = int(original["correct_value"])
    transformed = _transformed_value(correct_value, mechanism)

    if family == "affine_modular":
        modulus = int(original["partial_modulus"])
        transformed %= modulus
        if transformed == correct_value:
            raise ValueError("affine checkpoint transformation is unchanged")
        residues = list(parameters["reduced_residues"])
        moduli = list(parameters["moduli"])
        split = int(original["split"])
        propagated = solve_congruence(
            (transformed, modulus),
            *zip(residues[split:], moduli[split:], strict=True),
        )
        if propagated is None:
            raise ValueError("transformed affine checkpoint has no continuation")
        target_value = int(propagated[0]) or math.prod(moduli)
        marker = f"checkpoint y≡{correct_value} (mod {modulus})"
        replacement = f"checkpoint y≡{transformed} (mod {modulus})"
        if correct.count(marker) != 1:
            raise ValueError("affine checkpoint marker is not unique")
        corrupted = correct.replace(marker, replacement)
    elif family in {"conditional_dag", "finite_state"}:
        vector_key = "correct_vector"
        vector = list(original[vector_key])
        state_index = int(original["state_index"])
        vector[state_index] = transformed
        marker = str(original[vector_key])
        if correct.count(marker) != 1:
            raise ValueError("vector checkpoint marker is not unique")
        corrupted = correct.replace(marker, str(vector))
        delta = transformed - correct_value
        target_value = int(problem.gold_answer) + delta * int(original["downstream_multiplier"])
    elif family == "subset_counting":
        split = int(original["split"])
        used = int(original["used"])
        total = int(original["total"])
        marker = f"D_{split}({used},{total})={correct_value}."
        replacement = f"D_{split}({used},{total})={transformed}."
        if correct.count(marker) != 1:
            raise ValueError("subset checkpoint marker is not unique")
        corrupted = correct.replace(marker, replacement)
        delta = transformed - correct_value
        target_value = int(problem.gold_answer) + delta * int(original["downstream_multiplier"])
    else:
        raise ValueError(f"unsupported causal-audit family: {family}")

    target = str(target_value)
    if target == problem.gold_answer or corrupted == correct:
        raise ValueError("checkpoint transformation did not change the certified answer")
    checkpoint = {
        **original,
        "mechanism": str(mechanism),
        "original_value": correct_value,
        "transformed_value": transformed,
        "signed_delta": transformed - correct_value,
    }
    checkpoint.pop("corrupted_value", None)
    checkpoint.pop("corrupted_vector", None)
    return correct, corrupted, target, checkpoint


def _generate_eligible_problem(
    *,
    rng: random.Random,
    family: str,
    tier: str,
    renderer: int,
    mechanism: CorruptionMechanism,
    index: int,
) -> tuple[MathProblem, dict[str, object]]:
    """Generate deterministically until source and transformed checkpoints verify."""
    for _attempt in range(10_000):
        instance_seed = rng.randrange(2**63)
        generator_renderer = 0 if family == "conditional_dag" else renderer
        prompt, answer, parameters = _GENERATORS_V2[family](
            random.Random(instance_seed), tier, generator_renderer
        )
        if family == "conditional_dag":
            prompt = _render_conditional_dag(
                int(parameters["node_count"]),
                [tuple(edge) for edge in parameters["edges"]],
                dict(parameters["condition"]),
                renderer,
            )
        question_id = f"cav2-{family}-{mechanism}-{tier}-{index:02d}-{instance_seed:016x}"
        source_core: dict[str, object] = {
            "question_id": question_id,
            "generator_version": GENERATOR_VERSION_V2,
            "instance_seed": instance_seed,
            "family": family,
            "difficulty_tier": tier,
            "renderer_id": renderer,
            "parameters": parameters,
            "oracle_answer": answer,
            "oracle_kind": "dual_exact_verification",
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        }
        certificate = {
            **source_core,
            "certificate_sha256": _digest(source_core),
            "study_generator_version": CAUSAL_AUDIT_VERSION,
        }
        problem = MathProblem(
            question_id=question_id,
            task_family="causal_audit_math",
            prompt=prompt,
            gold_answer=answer,
            difficulty=tier,
            template_group=f"{family}:renderer-{renderer}",
            source=f"original:{CAUSAL_AUDIT_VERSION}",
            metadata={
                "generator_version": GENERATOR_VERSION_V2,
                "study_generator_version": CAUSAL_AUDIT_VERSION,
                "generator_family": family,
                "difficulty_tier": tier,
                "renderer_id": renderer,
                "instance_seed": instance_seed,
                "lineage_id": question_id,
                "structural_parameters": parameters,
                "oracle_kind": "dual_exact_verification",
                "certificate_sha256": certificate["certificate_sha256"],
            },
        )
        try:
            _render_transformed_checkpoint(problem, parameters, mechanism)
        except ValueError:
            continue
        return problem, certificate
    raise RuntimeError(f"could not generate eligible {family}/{mechanism} problem")


def build_causal_audit_v2(
    *, root_seed: int = 20262680, excluded_question_ids: set[str] | None = None
) -> tuple[list[MathProblem], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Build the balanced 24-question qualification and 72-question external test."""
    rng = random.Random(root_seed)
    excluded = excluded_question_ids or set()
    problems: list[MathProblem] = []
    sources: list[dict[str, object]] = []
    interventions: list[dict[str, object]] = []
    for family in FAMILIES:
        for mechanism in MECHANISMS:
            total = QUALIFICATION_PER_CELL + CONFIRMATORY_PER_CELL
            for index in range(total):
                cell = index % 4
                tier = TIERS_V2[cell // 2]
                renderer = cell % 2
                problem, source = _generate_eligible_problem(
                    rng=rng,
                    family=family,
                    tier=tier,
                    renderer=renderer,
                    mechanism=mechanism,
                    index=index,
                )
                if problem.question_id in excluded:
                    raise ValueError(
                        f"fresh bank overlaps excluded question: {problem.question_id}"
                    )
                partition = "qualification" if index < QUALIFICATION_PER_CELL else "confirmatory"
                correct, corrupted, target, checkpoint = _render_transformed_checkpoint(
                    problem, dict(source["parameters"]), mechanism
                )
                core: dict[str, object] = {
                    "question_id": problem.question_id,
                    "intervention_version": CAUSAL_AUDIT_VERSION,
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
                    "study_partition": partition,
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
                                "causal_yield_protocol": CAUSAL_AUDIT_VERSION,
                                "corruption_mechanism": str(mechanism),
                                "continuation_correct_prefix": correct,
                                "continuation_corrupted_prefix": corrupted,
                                "intervention_target_answer": target,
                                "intervention_certificate_sha256": intervention[
                                    "intervention_certificate_sha256"
                                ],
                                "study_partition": partition,
                                "monitor_split": "external_test"
                                if partition == "confirmatory"
                                else None,
                                "excluded_from_monitor_data": partition == "qualification",
                                "eligible_for_external_audit": partition == "confirmatory",
                            }
                        }
                    )
                )
                sources.append(source)
                interventions.append(intervention)

    if len(problems) != 96 or len({row.question_id for row in problems}) != 96:
        raise AssertionError("causal-audit-v2 must contain 96 unique questions")
    if len({row.prompt for row in problems}) != 96:
        raise AssertionError("causal-audit-v2 prompts must be unique")
    cell_counts = Counter(
        (
            str(row.metadata["study_partition"]),
            str(row.metadata["generator_family"]),
            str(row.metadata["corruption_mechanism"]),
        )
        for row in problems
    )
    selection = {
        "protocol": CAUSAL_AUDIT_VERSION,
        "root_seed": root_seed,
        "questions": len(problems),
        "partition_counts": dict(Counter(str(row.metadata["study_partition"]) for row in problems)),
        "family_mechanism_partition_counts": {
            f"{partition}:{family}:{mechanism}": count
            for (partition, family, mechanism), count in sorted(cell_counts.items())
        },
        "mechanisms": [str(value) for value in MECHANISMS],
        "selection_uses_model_outcomes": False,
        "all_partitions_frozen_before_collection": True,
        "confirmatory_is_external_test_only": True,
    }
    return problems, sources, interventions, selection


def verify_causal_audit_v2_problem(
    problem: MathProblem,
    source: dict[str, object],
    intervention: dict[str, object],
) -> bool:
    """Recompute and bind source, mechanism, checkpoint, target, and provenance."""
    core = {k: v for k, v in intervention.items() if k != "intervention_certificate_sha256"}
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
    partition = intervention.get("study_partition")
    return (
        intervention.get("intervention_version") == CAUSAL_AUDIT_VERSION
        and intervention.get("source_certificate_sha256") == source.get("certificate_sha256")
        and intervention.get("correct_prefix") == correct
        and intervention.get("corrupted_prefix") == corrupted
        and intervention.get("target_answer") == target
        and intervention.get("checkpoint") == checkpoint
        and problem.metadata.get("corruption_mechanism") == str(mechanism)
        and problem.metadata.get("continuation_correct_prefix") == correct
        and problem.metadata.get("continuation_corrupted_prefix") == corrupted
        and problem.metadata.get("intervention_target_answer") == target
        and problem.metadata.get("study_partition") == partition
        and problem.metadata.get("eligible_for_external_audit") is (partition == "confirmatory")
    )
