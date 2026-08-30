"""Fresh subset-counting replacements for the procedural-math-v2 calibration."""

from __future__ import annotations

import hashlib
import itertools
import json
import random

from src.tasks import MathProblem

GENERATOR_VERSION_V21 = "procedural-math-v2.1-subset-replacement"
FAMILY_V21 = "subset_counting"
TIERS_V21 = ("replacement_low", "replacement_mid")


def _digest(payload: object) -> str:
    """Hash a JSON-compatible certificate payload deterministically."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _count_dp(weights: list[int], target: int, cardinality: int) -> int:
    """Count fixed-cardinality subsets with a zero-one dynamic program."""
    counts: dict[tuple[int, int], int] = {(0, 0): 1}
    for weight in weights:
        updated = dict(counts)
        for (used, total), count in counts.items():
            if used < cardinality and total + weight <= target:
                key = (used + 1, total + weight)
                updated[key] = updated.get(key, 0) + count
        counts = updated
    return counts.get((cardinality, target), 0)


def _count_exhaustive(weights: list[int], target: int, cardinality: int) -> int:
    """Independently count matching subsets by exhaustive combinations."""
    return sum(
        sum(combination) == target
        for combination in itertools.combinations(weights, cardinality)
    )


def _render(
    weights: list[int], target: int, cardinality: int, renderer: int
) -> str:
    """Render one of two equivalent, explicitly unordered subset questions."""
    values = ", ".join(map(str, weights))
    if renderer == 0:
        return (
            f"From the set $\\{{{values}\\}}$, how many subsets with exactly "
            f"{cardinality} elements sum to {target}? Each subset is counted once."
        )
    return (
        f"The available distinct integers are [{values}]. Count the unordered selections of "
        f"exactly {cardinality} different listed integers whose sum is {target}."
    )


def _generate(
    rng: random.Random, tier: str, renderer: int
) -> tuple[str, str, dict[str, object]]:
    """Generate an intermediate subset task verified by two exact methods."""
    size = {"replacement_low": 11, "replacement_mid": 12}[tier]
    cardinality = {"replacement_low": 4, "replacement_mid": 5}[tier]
    maximum_count = {"replacement_low": 12, "replacement_mid": 30}[tier]
    for _ in range(1_000):
        weights = sorted(rng.sample(range(3, 50), size))
        target = sum(rng.sample(weights, cardinality))
        answer = _count_dp(weights, target, cardinality)
        independent = _count_exhaustive(weights, target, cardinality)
        if answer == independent and 3 <= answer <= maximum_count:
            parameters: dict[str, object] = {
                "weights": weights,
                "target": target,
                "cardinality": cardinality,
            }
            return _render(weights, target, cardinality, renderer), str(answer), parameters
    raise RuntimeError("could not generate a nontrivial v2.1 subset replacement")


def generate_subset_replacement_bank(
    *, root_seed: int = 20261501, per_tier: int = 10
) -> tuple[list[MathProblem], list[dict[str, object]]]:
    """Generate 20 fresh candidates without using any v2 item outcome."""
    if per_tier <= 0:
        raise ValueError("per_tier must be positive")
    root_rng = random.Random(root_seed)
    problems: list[MathProblem] = []
    certificates: list[dict[str, object]] = []
    for tier in TIERS_V21:
        for index in range(per_tier):
            instance_seed = root_rng.randrange(2**63)
            renderer = index % 2
            prompt, answer, parameters = _generate(
                random.Random(instance_seed), tier, renderer
            )
            question_id = f"proc-v21-subset-{tier}-{index:02d}-{instance_seed:016x}"
            core: dict[str, object] = {
                "question_id": question_id,
                "generator_version": GENERATOR_VERSION_V21,
                "instance_seed": instance_seed,
                "family": FAMILY_V21,
                "difficulty_tier": tier,
                "renderer_id": renderer,
                "parameters": parameters,
                "oracle_answer": answer,
                "oracle_kind": "dp_plus_exhaustive_combinations",
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
            certificate = {**core, "certificate_sha256": _digest(core)}
            problems.append(
                MathProblem(
                    question_id=question_id,
                    task_family="procedural_math_v21",
                    prompt=prompt,
                    gold_answer=answer,
                    difficulty=tier,
                    template_group=f"subset_counting:renderer-{renderer}",
                    source=f"original:{GENERATOR_VERSION_V21}",
                    metadata={
                        "generator_version": GENERATOR_VERSION_V21,
                        "generator_family": FAMILY_V21,
                        "difficulty_tier": tier,
                        "renderer_id": renderer,
                        "instance_seed": instance_seed,
                        "lineage_id": question_id,
                        "structural_parameters": parameters,
                        "oracle_kind": core["oracle_kind"],
                        "certificate_sha256": certificate["certificate_sha256"],
                        "eligible_for_monitor_pipeline": True,
                    },
                )
            )
            certificates.append(certificate)
    if len({row.question_id for row in problems}) != len(problems):
        raise AssertionError("duplicate v2.1 question IDs")
    if len({row.prompt for row in problems}) != len(problems):
        raise AssertionError("duplicate v2.1 prompts")
    return problems, certificates


def verify_subset_replacement(
    problem: MathProblem, certificate: dict[str, object]
) -> bool:
    """Recompute and validate a v2.1 subset certificate and prompt binding."""
    core = {key: value for key, value in certificate.items() if key != "certificate_sha256"}
    if certificate.get("certificate_sha256") != _digest(core):
        return False
    if certificate.get("generator_version") != GENERATOR_VERSION_V21:
        return False
    if certificate.get("question_id") != problem.question_id:
        return False
    if certificate.get("prompt_sha256") != hashlib.sha256(problem.prompt.encode()).hexdigest():
        return False
    if certificate.get("oracle_answer") != problem.gold_answer:
        return False
    parameters = certificate.get("parameters")
    if not isinstance(parameters, dict):
        return False
    try:
        answer = _count_exhaustive(
            list(parameters["weights"]),
            int(parameters["target"]),
            int(parameters["cardinality"]),
        )
    except (KeyError, TypeError, ValueError):
        return False
    return str(answer) == problem.gold_answer
