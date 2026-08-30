"""Certified assistant-reasoning prefills for the final hidden-influence diagnostic."""

from __future__ import annotations

import hashlib
import json

from src.continuation_interventions import (
    _BUILDERS,
    _family,
    _source_valid,
    build_continuation_freeze,
)
from src.tasks import MathProblem

PREFILL_VERSION = "matched-assistant-prefill-v3"


def _digest(payload: object) -> str:
    """Return a stable digest for a JSON-compatible payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def naturalize_prefix(prefix: str) -> str:
    """Turn a solver checkpoint instruction into unfinished first-person reasoning."""
    cut_markers = (
        ". Continue to ",
        ". Apply the stated transitions",
        ". Continue through all weights",
        ". Continue CRT with the rest",
    )
    rendered = prefix
    for marker in cut_markers:
        if marker in rendered:
            rendered = rendered.split(marker, 1)[0] + "."
            break
    rendered = rendered.replace("Use D_", "I'll use D_", 1)
    rendered = rendered.replace("Use ", "I'll use ", 1)
    rendered = rendered.replace("The affine conditions", "I reduce the affine conditions", 1)
    rendered = rendered.replace("The table is complete", "My table is complete", 1)
    rendered = rendered.replace("the checkpoint ", "", 1)
    rendered = rendered.replace("its checkpoint ", "", 1)
    forbidden = ("provided", "supplied", "checkpoint", "continue")
    if any(word in rendered.lower() for word in forbidden):
        raise ValueError("assistant prefill exposes intervention provenance")
    return rendered.strip() + " "


def build_assistant_prefill_freeze(
    questions: list[MathProblem],
    source_certificates: list[dict[str, object]],
    *,
    per_family: int = 2,
    selection_seed: int = 20262303,
) -> tuple[list[MathProblem], list[dict[str, object]], dict[str, object]]:
    """Freeze a balanced fresh set with matched correct and corrupted prefills."""
    source_by_id = {row.question_id: row for row in questions}
    continuation_rows, continuation_certificates, selection = build_continuation_freeze(
        questions,
        source_certificates,
        excluded_question_ids=set(),
        per_family=per_family,
        selection_seed=selection_seed,
    )
    frozen: list[MathProblem] = []
    certificates: list[dict[str, object]] = []
    for row, continuation_certificate in zip(
        continuation_rows, continuation_certificates, strict=True
    ):
        source = source_by_id[row.question_id]
        correct = naturalize_prefix(str(continuation_certificate["correct_prefix"]))
        corrupted = naturalize_prefix(str(continuation_certificate["corrupted_prefix"]))
        core: dict[str, object] = {
            "question_id": source.question_id,
            "intervention_version": PREFILL_VERSION,
            "family": _family(source),
            "source_generator_version": source.metadata["generator_version"],
            "source_certificate_sha256": continuation_certificate[
                "source_certificate_sha256"
            ],
            "gold_answer": source.gold_answer,
            "target_answer": continuation_certificate["target_answer"],
            "correct_prefill": correct,
            "corrupted_prefill": corrupted,
            "checkpoint": continuation_certificate["checkpoint"],
            "single_changed_field": "checkpoint_state_value",
            "prefill_role": "assistant",
            "prefill_channel": "analysis",
        }
        certificate = {**core, "intervention_certificate_sha256": _digest(core)}
        frozen.append(source.model_copy(update={
            "source": f"diagnostic-only:{PREFILL_VERSION}",
            "metadata": {
                **source.metadata,
                "causal_yield_protocol": PREFILL_VERSION,
                "assistant_prefill_correct": correct,
                "assistant_prefill_corrupted": corrupted,
                "intervention_target_answer": continuation_certificate["target_answer"],
                "intervention_certificate_sha256": certificate[
                    "intervention_certificate_sha256"
                ],
                "excluded_from_monitor_data": True,
            },
        }))
        certificates.append(certificate)
    return frozen, certificates, {
        "protocol": PREFILL_VERSION,
        "selection_seed": selection_seed,
        "per_family": per_family,
        "questions": len(frozen),
        "selected_question_ids": [row.question_id for row in frozen],
        "selection_uses_model_outcomes": False,
        "source_selection_protocol": selection["protocol"],
        "diagnostic_only": True,
        "excluded_from_monitor_data": True,
    }


def verify_assistant_prefill(
    problem: MathProblem,
    intervention_certificate: dict[str, object],
    source_certificate: dict[str, object],
) -> bool:
    """Recompute the exact prefix pair and propagated target from source data."""
    core = {
        key: value for key, value in intervention_certificate.items()
        if key != "intervention_certificate_sha256"
    }
    if intervention_certificate.get("intervention_certificate_sha256") != _digest(core):
        return False
    if intervention_certificate.get("question_id") != problem.question_id:
        return False
    if not _source_valid(problem, source_certificate):
        return False
    try:
        correct, corrupted, target, checkpoint = _BUILDERS[_family(problem)](
            problem, dict(source_certificate["parameters"])
        )
    except (KeyError, TypeError, ValueError):
        return False
    return (
        intervention_certificate.get("correct_prefill") == naturalize_prefix(correct)
        and intervention_certificate.get("corrupted_prefill") == naturalize_prefix(corrupted)
        and intervention_certificate.get("target_answer") == target
        and intervention_certificate.get("checkpoint") == checkpoint
        and target != problem.gold_answer
        and problem.metadata.get("assistant_prefill_correct")
        == intervention_certificate.get("correct_prefill")
        and problem.metadata.get("assistant_prefill_corrupted")
        == intervention_certificate.get("corrupted_prefill")
        and problem.metadata.get("excluded_from_monitor_data") is True
    )


def token_pair_audit(correct: str, corrupted: str, tokenizer: object) -> dict[str, object]:
    """Require equal-length matched prefills with one contiguous changed token span."""
    encode = tokenizer.encode  # type: ignore[attr-defined]
    correct_tokens = list(encode(correct, add_special_tokens=False))
    corrupted_tokens = list(encode(corrupted, add_special_tokens=False))
    if len(correct_tokens) != len(corrupted_tokens):
        raise ValueError("matched prefills have unequal token lengths")
    changed = [
        index for index, pair in enumerate(zip(correct_tokens, corrupted_tokens, strict=True))
        if pair[0] != pair[1]
    ]
    if not changed or changed != list(range(changed[0], changed[-1] + 1)):
        raise ValueError("prefill token changes must form one contiguous span")
    return {
        "correct_token_count": len(correct_tokens),
        "corrupted_token_count": len(corrupted_tokens),
        "changed_token_start": changed[0],
        "changed_token_count": len(changed),
        "equal_token_length": True,
        "single_contiguous_changed_span": True,
    }
