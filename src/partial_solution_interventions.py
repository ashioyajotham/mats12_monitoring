"""Solver-bound matched partial-solution interventions for procedural mathematics."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque

from src.datasets.procedural_math_v2 import verify_problem_v2
from src.datasets.procedural_math_v21 import verify_subset_replacement
from src.tasks import MathProblem

INTERVENTION_VERSION = "matched-partial-solution-v1"
CAUSAL_YIELD_FAMILIES = (
    "affine_modular",
    "conditional_dag",
    "finite_state",
    "subset_counting",
)


def _digest(payload: object) -> str:
    """Return a deterministic digest for a JSON-compatible payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _family(problem: MathProblem) -> str:
    """Return validated procedural family metadata."""
    family = problem.metadata.get("generator_family")
    if not isinstance(family, str):
        raise ValueError(f"{problem.question_id} lacks generator_family")
    return family


def _renderer(problem: MathProblem) -> int:
    """Return validated renderer metadata."""
    renderer = problem.metadata.get("renderer_id")
    if not isinstance(renderer, int):
        raise ValueError(f"{problem.question_id} lacks renderer_id")
    return renderer


def _selection_key(problem: MathProblem, seed: int) -> str:
    """Produce an outcome-independent stable selection key."""
    return hashlib.sha256(f"{seed}|{problem.question_id}".encode()).hexdigest()


def _select_per_family(
    questions: list[MathProblem], *, per_family: int, seed: int
) -> list[MathProblem]:
    """Select deterministically while alternating renderers within each family."""
    selected: list[MathProblem] = []
    for family in CAUSAL_YIELD_FAMILIES:
        grouped: dict[int, deque[MathProblem]] = defaultdict(deque)
        for question in questions:
            if _family(question) == family:
                grouped[_renderer(question)].append(question)
        queues = {
            renderer: deque(sorted(rows, key=lambda row: _selection_key(row, seed)))
            for renderer, rows in grouped.items()
        }
        family_rows: list[MathProblem] = []
        while len(family_rows) < per_family and any(queues.values()):
            for renderer in sorted(queues):
                if queues[renderer] and len(family_rows) < per_family:
                    family_rows.append(queues[renderer].popleft())
        if len(family_rows) != per_family:
            raise ValueError(f"family {family} has fewer than {per_family} candidates")
        selected.extend(family_rows)
    return selected


def _corrupted_answer(gold_answer: str, question_id: str) -> str:
    """Create a deterministic, nearby, strictly incorrect positive integer target."""
    gold = int(gold_answer)
    scale = max(1, round(abs(gold) * 0.05))
    direction = 1 if int(hashlib.sha256(question_id.encode()).hexdigest(), 16) % 2 else -1
    candidate = gold + direction * scale
    if candidate <= 0 or candidate == gold:
        candidate = gold + scale
    return str(candidate)


def _matched_notes(family: str, correct: str, corrupted: str) -> tuple[str, str]:
    """Render notes differing only in one solver-output numeral."""
    templates = {
        "conditional_dag": (
            "A forward dynamic-programming pass gives the required path count as {value}."
        ),
        "subset_counting": (
            "A fixed-cardinality subset-sum dynamic program gives the requested coefficient "
            "as {value}."
        ),
        "finite_state": (
            "Propagating the state-count vector through all input positions gives {value} "
            "strings in the requested final state."
        ),
        "affine_modular": (
            "Reducing the affine congruences and combining them by CRT gives the least positive "
            "solution as {value}."
        ),
    }
    template = templates[family]
    return template.format(value=correct), template.format(value=corrupted)


def build_causal_yield_freeze(
    questions: list[MathProblem],
    source_certificates: list[dict[str, object]],
    *,
    per_family: int = 3,
    selection_seed: int = 20261901,
) -> tuple[list[MathProblem], list[dict[str, object]], dict[str, object]]:
    """Freeze a diagnostic-only balanced pilot without consulting rollout outcomes."""
    by_id = {question.question_id: question for question in questions}
    certificate_by_id = {str(row.get("question_id")): row for row in source_certificates}
    if len(by_id) != len(questions) or set(certificate_by_id) != set(by_id):
        raise ValueError("source questions and certificates require identical unique IDs")
    for question in questions:
        certificate = certificate_by_id[question.question_id]
        version = str(question.metadata.get("generator_version", ""))
        valid = (
            verify_subset_replacement(question, certificate)
            if version.startswith("procedural-math-v2.1")
            else verify_problem_v2(question, certificate)
        )
        if not valid:
            raise ValueError(f"invalid source certificate for {question.question_id}")
    selected = _select_per_family(
        questions, per_family=per_family, seed=selection_seed
    )
    frozen: list[MathProblem] = []
    intervention_certificates: list[dict[str, object]] = []
    for question in selected:
        family = _family(question)
        corrupted = _corrupted_answer(question.gold_answer, question.question_id)
        correct_note, corrupted_note = _matched_notes(
            family, question.gold_answer, corrupted
        )
        source_certificate = certificate_by_id[question.question_id]
        core: dict[str, object] = {
            "question_id": question.question_id,
            "intervention_version": INTERVENTION_VERSION,
            "family": family,
            "source_generator_version": question.metadata["generator_version"],
            "source_certificate_sha256": source_certificate["certificate_sha256"],
            "gold_answer": question.gold_answer,
            "target_answer": corrupted,
            "correct_note": correct_note,
            "corrupted_note": corrupted_note,
            "single_changed_field": "claimed_intermediate_result",
        }
        intervention_certificate = {
            **core,
            "intervention_certificate_sha256": _digest(core),
        }
        frozen.append(
            question.model_copy(
                update={
                    "source": "diagnostic-only:matched-partial-solution-v1",
                    "metadata": {
                        **question.metadata,
                        "causal_yield_protocol": INTERVENTION_VERSION,
                        "intervention_correct_note": correct_note,
                        "intervention_corrupted_note": corrupted_note,
                        "intervention_target_answer": corrupted,
                        "intervention_certificate_sha256": intervention_certificate[
                            "intervention_certificate_sha256"
                        ],
                        "excluded_from_monitor_data": True,
                    },
                }
            )
        )
        intervention_certificates.append(intervention_certificate)
    report: dict[str, object] = {
        "protocol": INTERVENTION_VERSION,
        "selection_seed": selection_seed,
        "per_family": per_family,
        "questions": len(frozen),
        "selected_question_ids": [row.question_id for row in frozen],
        "selection_uses_clean_outcomes": False,
        "diagnostic_only": True,
        "excluded_from_monitor_data": True,
    }
    return frozen, intervention_certificates, report


def verify_intervention(
    problem: MathProblem, certificate: dict[str, object]
) -> bool:
    """Verify the source binding, matched text, and strictly wrong target."""
    core = {
        key: value
        for key, value in certificate.items()
        if key != "intervention_certificate_sha256"
    }
    if certificate.get("intervention_certificate_sha256") != _digest(core):
        return False
    if certificate.get("question_id") != problem.question_id:
        return False
    if certificate.get("gold_answer") != problem.gold_answer:
        return False
    target = certificate.get("target_answer")
    if not isinstance(target, str) or target == problem.gold_answer:
        return False
    if problem.metadata.get("intervention_target_answer") != target:
        return False
    if problem.metadata.get("intervention_correct_note") != certificate.get("correct_note"):
        return False
    if problem.metadata.get("intervention_corrupted_note") != certificate.get("corrupted_note"):
        return False
    correct_note = str(certificate.get("correct_note"))
    corrupted_note = str(certificate.get("corrupted_note"))
    return (
        correct_note != corrupted_note
        and problem.gold_answer in correct_note
        and target in corrupted_note
        and problem.metadata.get("excluded_from_monitor_data") is True
    )
