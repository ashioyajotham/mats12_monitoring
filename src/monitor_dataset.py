"""Gate-bound monitor examples for the causal-error-detection-v1 study."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict, deque
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from src.generate_rollouts import Rollout, RolloutStatus
from src.hints import Condition
from src.math_answers import MathGrade, grade_math_answer
from src.tasks import MathProblem


class CausalErrorExampleKind(StrEnum):
    """Frozen primary and secondary causal-error example taxonomy."""

    ORDINARY_ERROR = "ordinary_error"
    CAUSALLY_INDUCED_ERROR = "causally_induced_error"
    CORRECT_STATE_ERROR = "correct_state_error"
    CORRUPT_OTHER_ERROR = "corrupt_other_error"


class TranscriptOnlyEvidence(BaseModel):
    """Evidence allowed to transcript-only and surface monitors."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str
    reasoning: str
    final_answer: str


class ContextAwareEvidence(BaseModel):
    """Evidence allowed to a judge that sees the rendered generation prompt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str
    reasoning: str
    final_answer: str
    rendered_prompt: str


class MonitorExample(BaseModel):
    """Labelled example whose monitor-visible views are explicit nested types."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    example_id: str
    rollout_id: str
    question_id: str
    split: str
    binary_label: int
    kind: CausalErrorExampleKind
    family: str
    tier: str
    renderer_id: int
    transcript: TranscriptOnlyEvidence
    context: ContextAwareEvidence


def _example(
    question: MathProblem,
    rollout: Rollout,
    *,
    kind: CausalErrorExampleKind,
    binary_label: int,
    split_override: str | None = None,
) -> MonitorExample:
    """Construct one example while keeping audit fields outside evidence views."""
    reasoning = rollout.reasoning
    if not reasoning or rollout.parsed_answer is None:
        raise ValueError(f"rollout {rollout.rollout_id} lacks monitor-visible reasoning or answer")
    split = split_override or question.metadata.get("monitor_split")
    family = question.metadata.get("generator_family")
    tier = question.metadata.get("difficulty_tier")
    renderer = question.metadata.get("renderer_id")
    allowed_splits = {"train", "validation", "test"}
    if split_override is not None:
        allowed_splits.add("excluded_qualification")
    if split not in allowed_splits:
        raise ValueError(f"question {question.question_id} lacks a frozen monitor split")
    if not isinstance(family, str) or not isinstance(tier, str) or not isinstance(renderer, int):
        raise ValueError(f"question {question.question_id} lacks frozen nuisance metadata")
    transcript = TranscriptOnlyEvidence(
        question=question.prompt,
        reasoning=reasoning,
        final_answer=rollout.parsed_answer,
    )
    context = ContextAwareEvidence(
        **transcript.model_dump(),
        rendered_prompt=rollout.prompt,
    )
    return MonitorExample(
        example_id=rollout.rollout_id,
        rollout_id=rollout.rollout_id,
        question_id=rollout.question_id,
        split=split,
        binary_label=binary_label,
        kind=kind,
        family=family,
        tier=tier,
        renderer_id=renderer,
        transcript=transcript,
        context=context,
    )


def _stable_key(example: MonitorExample, seed: int) -> str:
    """Return an outcome-independent order within an already defined audit stratum."""
    return hashlib.sha256(f"{seed}|{example.example_id}".encode()).hexdigest()


def qualification_smoke_examples(
    questions: list[MathProblem],
    rollouts: list[Rollout],
    *,
    count: int = 2,
    seed: int = 20262609,
) -> list[MonitorExample]:
    """Select known-wrong examples from the permanently excluded qualification cohort.

    Selection is stable and does not consume any confirmatory example. Labels are used only to
    identify eligible known-wrong smoke inputs; the typed evidence passed to each judge excludes
    the label, condition, certificates, provenance, and sibling outcomes.
    """
    if count <= 0:
        raise ValueError("qualification smoke count must be positive")
    by_id = {question.question_id: question for question in questions}
    if len(by_id) != len(questions):
        raise ValueError("qualification question IDs must be unique")
    if any(
        question.metadata.get("study_partition") != "qualification"
        or question.metadata.get("excluded_from_monitor_data") is not True
        for question in questions
    ):
        raise ValueError("smoke questions must be the excluded qualification partition")
    eligible: list[MonitorExample] = []
    for rollout in rollouts:
        question = by_id.get(rollout.question_id)
        if question is None:
            raise ValueError(f"unknown qualification rollout question: {rollout.question_id}")
        if (
            rollout.condition is not Condition.CLEAN
            or rollout.status is not RolloutStatus.CLEAN_STOP
        ):
            continue
        if (
            grade_math_answer(rollout.parsed_answer, question.gold_answer)
            is not MathGrade.INCORRECT
        ):
            continue
        eligible.append(
            _example(
                question,
                rollout,
                kind=CausalErrorExampleKind.ORDINARY_ERROR,
                binary_label=0,
                split_override="excluded_qualification",
            )
        )
    selected = sorted(
        eligible,
        key=lambda row: hashlib.sha256(f"{seed}|{row.example_id}".encode()).hexdigest(),
    )[:count]
    if len(selected) != count:
        raise ValueError(f"qualification cohort has only {len(selected)} eligible smoke examples")
    return selected


def select_secondary_audit(
    examples: list[MonitorExample], *, cap: int = 96, seed: int = 20262603
) -> list[MonitorExample]:
    """Select a stable round-robin audit across kind, family, and frozen split."""
    if cap <= 0:
        raise ValueError("secondary audit cap must be positive")
    grouped: dict[tuple[str, str, str], deque[MonitorExample]] = defaultdict(deque)
    for example in examples:
        if example.binary_label != 0 or example.kind not in {
            CausalErrorExampleKind.CORRECT_STATE_ERROR,
            CausalErrorExampleKind.CORRUPT_OTHER_ERROR,
        }:
            raise ValueError("secondary audit input contains a non-secondary example")
        key = (str(example.kind), example.family, example.split)
        grouped[key].append(example)
    for key, rows in grouped.items():
        grouped[key] = deque(sorted(rows, key=lambda row: _stable_key(row, seed)))
    selected: list[MonitorExample] = []
    while len(selected) < cap and any(grouped.values()):
        for key in sorted(grouped):
            if grouped[key] and len(selected) < cap:
                selected.append(grouped[key].popleft())
    return selected


def materialize_monitor_examples(
    questions: list[MathProblem],
    rollouts: list[Rollout],
    gate_report: dict[str, object],
    *,
    secondary_cap: int = 96,
    secondary_seed: int = 20262603,
) -> tuple[list[MonitorExample], list[MonitorExample], dict[str, object]]:
    """Create primary and secondary examples only after every confirmatory gate passes."""
    if gate_report.get("confirmatory_causal_gate_passed") is not True:
        raise ValueError("confirmatory causal gate has not passed")
    if gate_report.get("monitor_training_authorized") is not True:
        raise ValueError("monitor training is not authorized")
    by_id = {question.question_id: question for question in questions}
    if len(by_id) != len(questions):
        raise ValueError("question IDs must be unique")
    if any(
        question.metadata.get("study_partition") != "confirmatory"
        or question.metadata.get("excluded_from_monitor_data") is not False
        for question in questions
    ):
        raise ValueError("monitor questions must be the untouched confirmatory partition")
    if len({rollout.rollout_id for rollout in rollouts}) != len(rollouts):
        raise ValueError("rollout IDs must be unique")

    primary: list[MonitorExample] = []
    secondary: list[MonitorExample] = []
    for rollout in rollouts:
        question = by_id.get(rollout.question_id)
        if question is None:
            raise ValueError(f"unknown rollout question: {rollout.question_id}")
        if rollout.status is not RolloutStatus.CLEAN_STOP:
            continue
        grade = grade_math_answer(rollout.parsed_answer, question.gold_answer)
        if grade is not MathGrade.INCORRECT:
            continue
        target = question.metadata.get("intervention_target_answer")
        if not isinstance(target, str) or target == question.gold_answer:
            raise ValueError(f"question {question.question_id} lacks a certified wrong target")
        if rollout.condition is Condition.CLEAN:
            primary.append(
                _example(
                    question,
                    rollout,
                    kind=CausalErrorExampleKind.ORDINARY_ERROR,
                    binary_label=0,
                )
            )
        elif (
            rollout.condition is Condition.CORRUPTED_CONTINUATION
            and rollout.parsed_answer == target
        ):
            primary.append(
                _example(
                    question,
                    rollout,
                    kind=CausalErrorExampleKind.CAUSALLY_INDUCED_ERROR,
                    binary_label=1,
                )
            )
        elif rollout.condition is Condition.CORRECT_CONTINUATION:
            secondary.append(
                _example(
                    question,
                    rollout,
                    kind=CausalErrorExampleKind.CORRECT_STATE_ERROR,
                    binary_label=0,
                )
            )
        elif rollout.condition is Condition.CORRUPTED_CONTINUATION:
            secondary.append(
                _example(
                    question,
                    rollout,
                    kind=CausalErrorExampleKind.CORRUPT_OTHER_ERROR,
                    binary_label=0,
                )
            )

    for split in ("train", "validation", "test"):
        labels = {
            example.binary_label for example in primary if example.split == split
        }
        if labels != {0, 1}:
            raise ValueError(f"primary {split} split does not contain both classes")
    audit = select_secondary_audit(
        secondary, cap=secondary_cap, seed=secondary_seed
    )
    summary: dict[str, object] = {
        "protocol": "causal-error-monitor-dataset-v1",
        "primary_examples": len(primary),
        "secondary_candidates": len(secondary),
        "secondary_audit_examples": len(audit),
        "secondary_audit_cap": secondary_cap,
        "secondary_audit_seed": secondary_seed,
        "primary_kind_counts": dict(sorted(Counter(str(row.kind) for row in primary).items())),
        "primary_split_label_counts": {
            f"{split}:label-{label}": count
            for (split, label), count in sorted(
                Counter((row.split, row.binary_label) for row in primary).items()
            )
        },
        "secondary_kind_counts": dict(
            sorted(Counter(str(row.kind) for row in audit).items())
        ),
        "question_group_splits_preserved": True,
    }
    return primary, audit, summary
