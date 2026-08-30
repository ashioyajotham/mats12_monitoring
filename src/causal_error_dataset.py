"""Prospective construction for the causal-error-detection-v1 study."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter

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

CAUSAL_ERROR_VERSION = "causal-error-detection-v1"
FAMILIES = FAMILIES_V2
PARTITION_COUNTS_PER_FAMILY = {"qualification": 6, "confirmatory": 18}
MONITOR_SPLIT_COUNTS = {"train": 43, "validation": 14, "test": 15}


def _digest(payload: object) -> str:
    """Hash a JSON-compatible object using its canonical encoding."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _stable_key(namespace: str, seed: int, question_id: str) -> str:
    """Return a deterministic, outcome-independent ordering key."""
    return hashlib.sha256(f"{namespace}|{seed}|{question_id}".encode()).hexdigest()


def _generate_source_bank(
    *, root_seed: int, per_family: int
) -> tuple[list[MathProblem], list[dict[str, object]]]:
    """Generate fresh v2-compatible problems with exact-continuation eligibility."""
    if per_family <= 0 or per_family % 4:
        raise ValueError("per_family must be a positive multiple of four")
    root_rng = random.Random(root_seed)
    problems: list[MathProblem] = []
    certificates: list[dict[str, object]] = []
    for family in FAMILIES:
        for family_index in range(per_family):
            cell_index = family_index % 4
            tier = TIERS_V2[cell_index // 2]
            renderer = cell_index % 2
            instance_seed = root_rng.randrange(2**63)
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
            question_id = (
                f"ced-v1-{family}-{tier}-{family_index:02d}-{instance_seed:016x}"
            )
            core: dict[str, object] = {
                "question_id": question_id,
                "generator_version": GENERATOR_VERSION_V2,
                "study_generator_version": CAUSAL_ERROR_VERSION,
                "instance_seed": instance_seed,
                "family": family,
                "difficulty_tier": tier,
                "renderer_id": renderer,
                "parameters": parameters,
                "oracle_answer": answer,
                "oracle_kind": "dual_exact_verification",
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
            source_core = {
                key: value
                for key, value in core.items()
                if key != "study_generator_version"
            }
            certificate = {
                **source_core,
                "certificate_sha256": _digest(source_core),
                "study_generator_version": CAUSAL_ERROR_VERSION,
            }
            problem = MathProblem(
                question_id=question_id,
                task_family="causal_error_math",
                prompt=prompt,
                gold_answer=answer,
                difficulty=tier,
                template_group=f"{family}:renderer-{renderer}",
                source=f"original:{CAUSAL_ERROR_VERSION}",
                metadata={
                    "generator_version": GENERATOR_VERSION_V2,
                    "study_generator_version": CAUSAL_ERROR_VERSION,
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
            problems.append(problem)
            certificates.append(certificate)
    if len({row.question_id for row in problems}) != len(problems):
        raise AssertionError("generated duplicate causal-error question IDs")
    if len({row.prompt for row in problems}) != len(problems):
        raise AssertionError("generated duplicate causal-error prompts")
    return problems, certificates


def _source_certificate_for_verifier(certificate: dict[str, object]) -> dict[str, object]:
    """Remove study-only provenance before invoking the frozen v2 verifier."""
    return {
        key: value
        for key, value in certificate.items()
        if key != "study_generator_version"
    }


def _partition_assignments(
    problems: list[MathProblem], *, partition_seed: int, split_seed: int
) -> tuple[dict[str, str], dict[str, str | None]]:
    """Assign qualification/cohort and grouped monitor splits before model calls."""
    partitions: dict[str, str] = {}
    monitor_splits: dict[str, str | None] = {}
    confirmatory_by_family: dict[str, list[MathProblem]] = {}
    for family in FAMILIES:
        rows = [row for row in problems if row.metadata["generator_family"] == family]
        rows.sort(key=lambda row: _stable_key("partition", partition_seed, row.question_id))
        qualification_count = PARTITION_COUNTS_PER_FAMILY["qualification"]
        for row in rows[:qualification_count]:
            partitions[row.question_id] = "qualification"
            monitor_splits[row.question_id] = None
        confirmatory = rows[qualification_count:]
        confirmatory_by_family[family] = confirmatory
        for row in confirmatory:
            partitions[row.question_id] = "confirmatory"

    # The nearest integer 60/20/20 allocation for 72 groups is 43/14/15. Rotating
    # per-family allocations keeps every family represented in every split.
    family_allocations = (
        {"train": 11, "validation": 3, "test": 4},
        {"train": 11, "validation": 4, "test": 3},
        {"train": 11, "validation": 3, "test": 4},
        {"train": 10, "validation": 4, "test": 4},
    )
    for family, allocation in zip(FAMILIES, family_allocations, strict=True):
        rows = sorted(
            confirmatory_by_family[family],
            key=lambda row: _stable_key("monitor-split", split_seed, row.question_id),
        )
        cursor = 0
        for split in ("train", "validation", "test"):
            next_cursor = cursor + allocation[split]
            for row in rows[cursor:next_cursor]:
                monitor_splits[row.question_id] = split
            cursor = next_cursor
        if cursor != len(rows):
            raise AssertionError("monitor split allocation did not consume a family")
    return partitions, monitor_splits


def build_causal_error_dataset(
    *,
    root_seed: int = 20262501,
    partition_seed: int = 20262502,
    split_seed: int = 20262503,
    per_family: int = 24,
    excluded_question_ids: set[str] | None = None,
) -> tuple[
    list[MathProblem],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    """Build the fresh certified 96-question study and freeze all partitions."""
    if per_family != 24:
        raise ValueError("causal-error-detection-v1 requires exactly 24 questions per family")
    problems, source_certificates = _generate_source_bank(
        root_seed=root_seed, per_family=per_family
    )
    excluded = excluded_question_ids or set()
    overlap = sorted({row.question_id for row in problems} & excluded)
    if overlap:
        raise ValueError(f"fresh bank overlaps excluded questions: {overlap}")
    source_by_id = {
        str(row["question_id"]): row for row in source_certificates
    }
    if not all(
        verify_problem_v2(
            problem,
            _source_certificate_for_verifier(source_by_id[problem.question_id]),
        )
        for problem in problems
    ):
        raise ValueError("a fresh source certificate failed independent verification")

    partitions, monitor_splits = _partition_assignments(
        problems, partition_seed=partition_seed, split_seed=split_seed
    )
    frozen: list[MathProblem] = []
    intervention_certificates: list[dict[str, object]] = []
    for problem in problems:
        source_certificate = source_by_id[problem.question_id]
        family = str(problem.metadata["generator_family"])
        correct, corrupted, target, checkpoint = _BUILDERS[family](
            problem, dict(source_certificate["parameters"])
        )
        if target == problem.gold_answer:
            raise AssertionError("corrupted-state target must differ from the gold answer")
        partition = partitions[problem.question_id]
        monitor_split = monitor_splits[problem.question_id]
        core: dict[str, object] = {
            "question_id": problem.question_id,
            "intervention_version": CAUSAL_ERROR_VERSION,
            "family": family,
            "source_generator_version": GENERATOR_VERSION_V2,
            "source_certificate_sha256": source_certificate["certificate_sha256"],
            "gold_answer": problem.gold_answer,
            "target_answer": target,
            "correct_prefix": correct,
            "corrupted_prefix": corrupted,
            "checkpoint": checkpoint,
            "single_changed_field": "checkpoint_state_value",
            "study_partition": partition,
            "monitor_split": monitor_split,
        }
        intervention_certificate = {
            **core,
            "intervention_certificate_sha256": _digest(core),
        }
        excluded_from_monitor_data = partition == "qualification"
        frozen.append(
            problem.model_copy(
                update={
                    "metadata": {
                        **problem.metadata,
                        "causal_yield_protocol": CAUSAL_ERROR_VERSION,
                        "continuation_correct_prefix": correct,
                        "continuation_corrupted_prefix": corrupted,
                        "intervention_target_answer": target,
                        "intervention_certificate_sha256": intervention_certificate[
                            "intervention_certificate_sha256"
                        ],
                        "study_partition": partition,
                        "monitor_split": monitor_split,
                        "excluded_from_monitor_data": excluded_from_monitor_data,
                        "eligible_for_monitor_pipeline": not excluded_from_monitor_data,
                    }
                }
            )
        )
        intervention_certificates.append(intervention_certificate)

    partition_counts = Counter(partitions.values())
    family_partition_counts = Counter(
        (str(row.metadata["generator_family"]), str(row.metadata["study_partition"]))
        for row in frozen
    )
    split_counts = Counter(
        str(row.metadata["monitor_split"])
        for row in frozen
        if row.metadata["study_partition"] == "confirmatory"
    )
    selection: dict[str, object] = {
        "protocol": CAUSAL_ERROR_VERSION,
        "root_seed": root_seed,
        "partition_seed": partition_seed,
        "monitor_split_seed": split_seed,
        "questions": len(frozen),
        "per_family": per_family,
        "partition_counts": dict(sorted(partition_counts.items())),
        "family_partition_counts": {
            f"{family}:{partition}": count
            for (family, partition), count in sorted(family_partition_counts.items())
        },
        "monitor_split_counts": dict(sorted(split_counts.items())),
        "qualification_question_ids": [
            row.question_id for row in frozen
            if row.metadata["study_partition"] == "qualification"
        ],
        "confirmatory_question_ids": [
            row.question_id for row in frozen
            if row.metadata["study_partition"] == "confirmatory"
        ],
        "selection_uses_model_outcomes": False,
        "all_partitions_frozen_before_collection": True,
    }
    if dict(split_counts) != MONITOR_SPLIT_COUNTS:
        raise AssertionError("unexpected confirmatory monitor split counts")
    return frozen, source_certificates, intervention_certificates, selection


def verify_causal_error_problem(
    problem: MathProblem,
    source_certificate: dict[str, object],
    intervention_certificate: dict[str, object],
) -> bool:
    """Recompute the source oracle, state perturbation, target, and partition binding."""
    core = {
        key: value
        for key, value in intervention_certificate.items()
        if key != "intervention_certificate_sha256"
    }
    if intervention_certificate.get("intervention_certificate_sha256") != _digest(core):
        return False
    if intervention_certificate.get("question_id") != problem.question_id:
        return False
    if not verify_problem_v2(problem, _source_certificate_for_verifier(source_certificate)):
        return False
    if intervention_certificate.get("source_certificate_sha256") != source_certificate.get(
        "certificate_sha256"
    ):
        return False
    family = str(problem.metadata.get("generator_family"))
    try:
        correct, corrupted, target, checkpoint = _BUILDERS[family](
            problem, dict(source_certificate["parameters"])
        )
    except (KeyError, TypeError, ValueError):
        return False
    partition = intervention_certificate.get("study_partition")
    monitor_split = intervention_certificate.get("monitor_split")
    expected_exclusion = partition == "qualification"
    return (
        intervention_certificate.get("intervention_version") == CAUSAL_ERROR_VERSION
        and intervention_certificate.get("correct_prefix") == correct
        and intervention_certificate.get("corrupted_prefix") == corrupted
        and intervention_certificate.get("target_answer") == target
        and intervention_certificate.get("checkpoint") == checkpoint
        and target != problem.gold_answer
        and problem.metadata.get("continuation_correct_prefix") == correct
        and problem.metadata.get("continuation_corrupted_prefix") == corrupted
        and problem.metadata.get("intervention_target_answer") == target
        and problem.metadata.get("study_partition") == partition
        and problem.metadata.get("monitor_split") == monitor_split
        and problem.metadata.get("excluded_from_monitor_data") is expected_exclusion
        and problem.metadata.get("eligible_for_monitor_pipeline") is not expected_exclusion
    )
