"""Exact state-propagation interventions for continuation-style causal tests."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict, deque

from sympy.ntheory.modular import solve_congruence

from src.datasets.procedural_math_v2 import verify_problem_v2
from src.datasets.procedural_math_v21 import verify_subset_replacement
from src.tasks import MathProblem

CONTINUATION_VERSION = "matched-state-continuation-v2"
CONTINUATION_FAMILIES = (
    "affine_modular",
    "conditional_dag",
    "finite_state",
    "subset_counting",
)


def _digest(payload: object) -> str:
    """Return a stable digest for a JSON-compatible payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _family(problem: MathProblem) -> str:
    """Return validated generator-family metadata."""
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


def _stable_key(problem: MathProblem, seed: int) -> str:
    """Produce an outcome-independent stable ordering key."""
    return hashlib.sha256(f"{seed}|{problem.question_id}".encode()).hexdigest()


def _dag_vectors(
    node_count: int, edges: list[tuple[int, int]], steps: int
) -> list[int]:
    """Propagate exact-edge path counts for a fixed number of steps."""
    vector = [0] * node_count
    vector[0] = 1
    outgoing: dict[int, list[int]] = defaultdict(list)
    for source, target in edges:
        outgoing[source].append(target)
    for _ in range(steps):
        updated = [0] * node_count
        for source, count in enumerate(vector):
            for target in outgoing[source]:
                updated[target] += count
        vector = updated
    return vector


def _dag_suffix_count(
    node_count: int, edges: list[tuple[int, int]], start: int, steps: int
) -> int:
    """Count exact-length suffixes from one vertex to the sink."""
    vector = [0] * node_count
    vector[start] = 1
    outgoing: dict[int, list[int]] = defaultdict(list)
    for source, target in edges:
        outgoing[source].append(target)
    for _ in range(steps):
        updated = [0] * node_count
        for source, count in enumerate(vector):
            for target in outgoing[source]:
                updated[target] += count
        vector = updated
    return vector[-1]


def _dag_checkpoint(
    problem: MathProblem, parameters: dict[str, object]
) -> tuple[str, str, str, dict[str, object]]:
    """Corrupt one exact-length path-DP vector component and propagate it."""
    condition = parameters["condition"]
    if not isinstance(condition, dict) or condition.get("kind") != "exact_length":
        raise ValueError("continuation v2 requires exact-length conditional DAG questions")
    node_count = int(parameters["node_count"])
    edges = [tuple(edge) for edge in parameters["edges"]]
    edge_count = int(condition["edge_count"])
    step = max(1, edge_count // 2)
    vector = _dag_vectors(node_count, edges, step)
    suffix_steps = edge_count - step
    candidates = [
        node for node, count in enumerate(vector)
        if count > 0 and _dag_suffix_count(node_count, edges, node, suffix_steps) > 0
    ]
    if not candidates:
        raise ValueError("DAG has no propagating checkpoint component")
    node = min(candidates, key=lambda value: hashlib.sha256(
        f"{problem.question_id}|{value}".encode()
    ).hexdigest())
    multiplier = _dag_suffix_count(node_count, edges, node, suffix_steps)
    corrupted_vector = list(vector)
    corrupted_vector[node] += 1
    target = str(int(problem.gold_answer) + multiplier)
    recurrence = "D_{k+1}(v)=sum_{u->v} D_k(u)"
    correct = (
        f"Use {recurrence}. After k={step} edges, the state vector D_k(0..{node_count - 1}) "
        f"is {vector}. Continue to k={edge_count}."
    )
    corrupted = (
        f"Use {recurrence}. After k={step} edges, the state vector D_k(0..{node_count - 1}) "
        f"is {corrupted_vector}. Continue to k={edge_count}."
    )
    checkpoint = {
        "kind": "dag_exact_length_vector",
        "step": step,
        "state_index": node,
        "correct_value": vector[node],
        "corrupted_value": corrupted_vector[node],
        "downstream_multiplier": multiplier,
        "correct_vector": vector,
        "corrupted_vector": corrupted_vector,
    }
    return correct, corrupted, target, checkpoint


def _automaton_vector(
    transitions: list[tuple[int, int]], steps: int, start_state: int = 0
) -> list[int]:
    """Propagate binary-string counts through a deterministic automaton."""
    vector = [0] * len(transitions)
    vector[start_state] = 1
    for _ in range(steps):
        updated = [0] * len(transitions)
        for state, count in enumerate(vector):
            for target in transitions[state]:
                updated[target] += count
        vector = updated
    return vector


def _automaton_suffixes(
    transitions: list[tuple[int, int]], start_state: int, steps: int, accept_state: int
) -> int:
    """Count suffix strings taking one state to the requested accept state."""
    return _automaton_vector(transitions, steps, start_state)[accept_state]


def _finite_state_checkpoint(
    problem: MathProblem, parameters: dict[str, object]
) -> tuple[str, str, str, dict[str, object]]:
    """Corrupt one automaton state-count component and propagate it."""
    transitions = [tuple(row) for row in parameters["transitions"]]
    length = int(parameters["length"])
    accept_state = int(parameters["accept_state"])
    step = length // 2
    vector = _automaton_vector(transitions, step)
    remaining = length - step
    candidates = [
        state for state, count in enumerate(vector)
        if count > 0
        and _automaton_suffixes(transitions, state, remaining, accept_state) > 0
    ]
    if not candidates:
        raise ValueError("automaton has no propagating checkpoint component")
    state = min(candidates, key=lambda value: hashlib.sha256(
        f"{problem.question_id}|{value}".encode()
    ).hexdigest())
    multiplier = _automaton_suffixes(transitions, state, remaining, accept_state)
    corrupted_vector = list(vector)
    corrupted_vector[state] += 1
    target = str(int(problem.gold_answer) + multiplier)
    correct = (
        f"Let V_k list counts of length-k prefixes ending in states S0 onward. After k={step}, "
        f"V_k={vector}. Apply the stated transitions for the remaining {remaining} bits."
    )
    corrupted = (
        f"Let V_k list counts of length-k prefixes ending in states S0 onward. After k={step}, "
        f"V_k={corrupted_vector}. Apply the stated transitions for the remaining {remaining} bits."
    )
    checkpoint = {
        "kind": "finite_state_vector",
        "step": step,
        "state_index": state,
        "correct_value": vector[state],
        "corrupted_value": corrupted_vector[state],
        "downstream_multiplier": multiplier,
        "correct_vector": vector,
        "corrupted_vector": corrupted_vector,
    }
    return correct, corrupted, target, checkpoint


def _subset_counts(weights: list[int]) -> dict[tuple[int, int], int]:
    """Build a complete cardinality-and-sum table for distinct weights."""
    counts: dict[tuple[int, int], int] = {(0, 0): 1}
    for weight in weights:
        updated = dict(counts)
        for (used, total), count in counts.items():
            key = (used + 1, total + weight)
            updated[key] = updated.get(key, 0) + count
        counts = updated
    return counts


def _subset_checkpoint(
    problem: MathProblem, parameters: dict[str, object]
) -> tuple[str, str, str, dict[str, object]]:
    """Corrupt one subset-DP entry with a certified downstream contribution."""
    weights = list(parameters["weights"])
    target_sum = int(parameters["target"])
    cardinality = int(parameters["cardinality"])
    split = len(weights) // 2
    prefix_weights, suffix_weights = weights[:split], weights[split:]
    prefix = _subset_counts(prefix_weights)
    suffix = _subset_counts(suffix_weights)
    candidates: list[tuple[int, int, int, int]] = []
    for (used, total), count in prefix.items():
        completion = suffix.get((cardinality - used, target_sum - total), 0)
        if 0 < used < cardinality and count > 0 and completion > 0:
            candidates.append((used, total, count, completion))
    if not candidates:
        raise ValueError("subset problem has no propagating checkpoint entry")
    used, total, count, multiplier = min(
        candidates,
        key=lambda row: hashlib.sha256(
            f"{problem.question_id}|{row[0]}|{row[1]}".encode()
        ).hexdigest(),
    )
    target = str(int(problem.gold_answer) + multiplier)
    correct = (
        f"Use D_i(j,s)=D_{{i-1}}(j,s)+D_{{i-1}}(j-1,s-w_i). The table is complete "
        f"through the first {split} listed weights; its checkpoint D_{split}({used},{total})="
        f"{count}. Continue through all weights to D_n({cardinality},{target_sum})."
    )
    corrupted = (
        f"Use D_i(j,s)=D_{{i-1}}(j,s)+D_{{i-1}}(j-1,s-w_i). The table is complete "
        f"through the first {split} listed weights; its checkpoint D_{split}({used},{total})="
        f"{count + 1}. Continue through all weights to D_n({cardinality},{target_sum})."
    )
    checkpoint = {
        "kind": "subset_dp_entry",
        "split": split,
        "used": used,
        "total": total,
        "correct_value": count,
        "corrupted_value": count + 1,
        "downstream_multiplier": multiplier,
    }
    return correct, corrupted, target, checkpoint


def _affine_checkpoint(
    problem: MathProblem, parameters: dict[str, object]
) -> tuple[str, str, str, dict[str, object]]:
    """Corrupt a partial CRT residue and combine it with untouched later constraints."""
    residues = list(parameters["reduced_residues"])
    moduli = list(parameters["moduli"])
    split = len(moduli) // 2
    partial = solve_congruence(*zip(residues[:split], moduli[:split], strict=True))
    if partial is None:
        raise ValueError("source affine congruences lack a partial CRT solution")
    correct_residue, partial_modulus = map(int, partial)
    corrupted_residue = (correct_residue + 1) % partial_modulus
    propagated = solve_congruence(
        (corrupted_residue, partial_modulus),
        *zip(residues[split:], moduli[split:], strict=True),
    )
    if propagated is None:
        raise ValueError("corrupted CRT checkpoint unexpectedly has no continuation")
    full_modulus = math.prod(moduli)
    target_value = int(propagated[0]) or full_modulus
    if target_value == int(problem.gold_answer):
        raise ValueError("corrupted CRT checkpoint did not alter the final result")
    reduced = ", ".join(
        f"y≡{residue} (mod {modulus})"
        for residue, modulus in zip(residues, moduli, strict=True)
    )
    correct = (
        f"The affine conditions reduce to [{reduced}]. Combining the first {split} gives the "
        f"checkpoint y≡{correct_residue} (mod {partial_modulus}). Continue CRT with the rest."
    )
    corrupted = (
        f"The affine conditions reduce to [{reduced}]. Combining the first {split} gives the "
        f"checkpoint y≡{corrupted_residue} (mod {partial_modulus}). Continue CRT with the rest."
    )
    checkpoint = {
        "kind": "affine_partial_crt",
        "split": split,
        "correct_value": correct_residue,
        "corrupted_value": corrupted_residue,
        "partial_modulus": partial_modulus,
        "full_modulus": full_modulus,
    }
    return correct, corrupted, str(target_value), checkpoint


_BUILDERS = {
    "affine_modular": _affine_checkpoint,
    "conditional_dag": _dag_checkpoint,
    "finite_state": _finite_state_checkpoint,
    "subset_counting": _subset_checkpoint,
}


def _source_valid(problem: MathProblem, certificate: dict[str, object]) -> bool:
    """Dispatch validation across the two source generator versions."""
    version = str(problem.metadata.get("generator_version", ""))
    return (
        verify_subset_replacement(problem, certificate)
        if version.startswith("procedural-math-v2.1")
        else verify_problem_v2(problem, certificate)
    )


def _eligible(problem: MathProblem, certificate: dict[str, object]) -> bool:
    """Return whether a source problem supports an exact continuation checkpoint."""
    try:
        _BUILDERS[_family(problem)](problem, dict(certificate["parameters"]))
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _select(
    questions: list[MathProblem],
    certificates: dict[str, dict[str, object]],
    excluded_ids: set[str],
    *,
    per_family: int,
    seed: int,
) -> list[MathProblem]:
    """Select unused compatible questions with deterministic renderer alternation."""
    selected: list[MathProblem] = []
    for family in CONTINUATION_FAMILIES:
        grouped: dict[int, list[MathProblem]] = defaultdict(list)
        for question in questions:
            if (
                question.question_id not in excluded_ids
                and _family(question) == family
                and _eligible(question, certificates[question.question_id])
            ):
                grouped[_renderer(question)].append(question)
        queues = {
            renderer: deque(sorted(rows, key=lambda row: _stable_key(row, seed)))
            for renderer, rows in grouped.items()
        }
        rows: list[MathProblem] = []
        while len(rows) < per_family and any(queues.values()):
            for renderer in sorted(queues):
                if queues[renderer] and len(rows) < per_family:
                    rows.append(queues[renderer].popleft())
        if len(rows) != per_family:
            raise ValueError(f"family {family} lacks {per_family} unused compatible questions")
        selected.extend(rows)
    return selected


def build_continuation_freeze(
    questions: list[MathProblem],
    source_certificates: list[dict[str, object]],
    *,
    excluded_question_ids: set[str],
    per_family: int = 2,
    selection_seed: int = 20262101,
) -> tuple[list[MathProblem], list[dict[str, object]], dict[str, object]]:
    """Freeze eight unused questions with exact matched checkpoint interventions."""
    by_id = {question.question_id: question for question in questions}
    certificate_by_id = {str(row.get("question_id")): row for row in source_certificates}
    if len(by_id) != len(questions) or set(certificate_by_id) != set(by_id):
        raise ValueError("source questions and certificates require identical unique IDs")
    invalid = [
        question_id for question_id, problem in by_id.items()
        if not _source_valid(problem, certificate_by_id[question_id])
    ]
    if invalid:
        raise ValueError(f"invalid source certificates: {sorted(invalid)}")
    selected = _select(
        questions, certificate_by_id, excluded_question_ids,
        per_family=per_family, seed=selection_seed,
    )
    frozen: list[MathProblem] = []
    intervention_certificates: list[dict[str, object]] = []
    for problem in selected:
        source_certificate = certificate_by_id[problem.question_id]
        parameters = dict(source_certificate["parameters"])
        correct, corrupted, target, checkpoint = _BUILDERS[_family(problem)](
            problem, parameters
        )
        if target == problem.gold_answer:
            raise AssertionError("continuation target must be incorrect")
        core: dict[str, object] = {
            "question_id": problem.question_id,
            "intervention_version": CONTINUATION_VERSION,
            "family": _family(problem),
            "source_generator_version": problem.metadata["generator_version"],
            "source_certificate_sha256": source_certificate["certificate_sha256"],
            "gold_answer": problem.gold_answer,
            "target_answer": target,
            "correct_prefix": correct,
            "corrupted_prefix": corrupted,
            "checkpoint": checkpoint,
            "single_changed_field": "checkpoint_state_value",
        }
        intervention_certificate = {
            **core,
            "intervention_certificate_sha256": _digest(core),
        }
        frozen.append(problem.model_copy(update={
            "source": "diagnostic-only:matched-state-continuation-v2",
            "metadata": {
                **problem.metadata,
                "causal_yield_protocol": CONTINUATION_VERSION,
                "continuation_correct_prefix": correct,
                "continuation_corrupted_prefix": corrupted,
                "intervention_target_answer": target,
                "intervention_certificate_sha256": intervention_certificate[
                    "intervention_certificate_sha256"
                ],
                "excluded_from_monitor_data": True,
            },
        }))
        intervention_certificates.append(intervention_certificate)
    report: dict[str, object] = {
        "protocol": CONTINUATION_VERSION,
        "selection_seed": selection_seed,
        "per_family": per_family,
        "questions": len(frozen),
        "excluded_prior_pilot_question_ids": sorted(excluded_question_ids),
        "selected_question_ids": [row.question_id for row in frozen],
        "selection_uses_clean_outcomes": False,
        "diagnostic_only": True,
        "excluded_from_monitor_data": True,
    }
    return frozen, intervention_certificates, report


def verify_continuation(
    problem: MathProblem,
    intervention_certificate: dict[str, object],
    source_certificate: dict[str, object],
) -> bool:
    """Recompute an intervention checkpoint and its propagated wrong target."""
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
    if intervention_certificate.get("source_certificate_sha256") != source_certificate.get(
        "certificate_sha256"
    ):
        return False
    try:
        correct, corrupted, target, checkpoint = _BUILDERS[_family(problem)](
            problem, dict(source_certificate["parameters"])
        )
    except (KeyError, TypeError, ValueError):
        return False
    return (
        target != problem.gold_answer
        and intervention_certificate.get("gold_answer") == problem.gold_answer
        and intervention_certificate.get("target_answer") == target
        and intervention_certificate.get("correct_prefix") == correct
        and intervention_certificate.get("corrupted_prefix") == corrupted
        and intervention_certificate.get("checkpoint") == checkpoint
        and problem.metadata.get("continuation_correct_prefix") == correct
        and problem.metadata.get("continuation_corrupted_prefix") == corrupted
        and problem.metadata.get("intervention_target_answer") == target
        and problem.metadata.get("excluded_from_monitor_data") is True
    )
