"""Deterministic, solver-verified procedural mathematics problems."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable
from fractions import Fraction

import sympy

from src.tasks import MathProblem

GENERATOR_VERSION = "procedural-math-v1"
FAMILIES = ("crt", "linear_system", "dag_counting", "recurrence")
TIERS = ("easy", "medium", "hard")
RENDERERS_PER_FAMILY = 3


def _canonical_digest(payload: object) -> str:
    """Hash a JSON-compatible payload using canonical serialization."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _crt_incremental(residues: list[int], moduli: list[int]) -> int:
    """Solve pairwise-coprime congruences with an incremental exact search."""
    value = residues[0]
    step = moduli[0]
    for residue, modulus in zip(residues[1:], moduli[1:], strict=True):
        while value % modulus != residue:
            value += step
        step *= modulus
    return value % step


def _render_crt(residues: list[int], moduli: list[int], renderer: int) -> str:
    statements = [f"x \\equiv {r} \\pmod{{{m}}}" for r, m in zip(residues, moduli, strict=True)]
    if renderer == 0:
        body = ", ".join(statements)
        return f"Find the least positive integer $x$ satisfying {body}."
    if renderer == 1:
        body = "; ".join(statements)
        return f"Determine the smallest positive solution of the simultaneous congruences {body}."
    lines = "\n".join(f"- ${statement}$" for statement in statements)
    return (
        f"An integer $x>0$ obeys all of the following conditions:\n{lines}\n"
        "What is its least possible value?"
    )


def _generate_crt(rng: random.Random, tier: str, renderer: int) -> tuple[str, str, dict]:
    """Construct and independently solve one Chinese-remainder problem."""
    count = {"easy": 3, "medium": 4, "hard": 5}[tier]
    primes = [5, 7, 11, 13, 17, 19, 23, 29, 31]
    moduli = sorted(rng.sample(primes, count))
    modulus_product = math.prod(moduli)
    latent = rng.randint(max(moduli) + 1, modulus_product - 1)
    residues = [latent % modulus for modulus in moduli]
    answer = _crt_incremental(residues, moduli)
    if answer == 0:
        answer = modulus_product
    if answer != latent:
        raise AssertionError("CRT verifier disagrees with the constructed least solution")
    parameters = {
        "moduli": moduli,
        "residues": residues,
        "latent_solution": latent,
        "modulus_product": modulus_product,
    }
    return _render_crt(residues, moduli, renderer), str(answer), parameters


def _fraction_solve(matrix: list[list[int]], values: list[int]) -> list[Fraction]:
    """Solve a square linear system using exact Gauss-Jordan elimination."""
    size = len(matrix)
    augmented = [
        list(map(Fraction, row)) + [Fraction(value)]
        for row, value in zip(matrix, values, strict=True)
    ]
    for column in range(size):
        pivot = next((row for row in range(column, size) if augmented[row][column]), None)
        if pivot is None:
            raise ValueError("singular system")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    left - factor * right
                    for left, right in zip(augmented[row], augmented[column], strict=True)
                ]
    return [augmented[index][-1] for index in range(size)]


def _render_linear_system(
    matrix: list[list[int]], values: list[int], target: list[int], renderer: int
) -> str:
    names = [chr(ord("a") + index) for index in range(len(matrix))]
    equations: list[str] = []
    for row, value in zip(matrix, values, strict=True):
        terms = [f"{coefficient}{name}" for coefficient, name in zip(row, names, strict=True)]
        equations.append(" + ".join(terms).replace("+ -", "- ") + f" = {value}")
    target_text = " + ".join(
        f"{coefficient}{name}"
        for coefficient, name in zip(target, names, strict=True)
        if coefficient
    ).replace("+ -", "- ")
    body = "; ".join(equations)
    if renderer == 0:
        return f"The integers {', '.join(names)} satisfy {body}. Find ${target_text}$."
    if renderer == 1:
        return (
            "Solve the following integer system and report the value of "
            f"${target_text}$: {body}."
        )
    lines = "\n".join(f"- ${equation}$" for equation in equations)
    return (
        f"There is a unique integer tuple $({', '.join(names)})$ satisfying:\n{lines}\n"
        f"Compute ${target_text}$."
    )


def _generate_linear_system(
    rng: random.Random, tier: str, renderer: int
) -> tuple[str, str, dict]:
    """Construct a nonsingular integer system and verify it with exact elimination."""
    size = {"easy": 3, "medium": 4, "hard": 5}[tier]
    while True:
        matrix = [[rng.randint(-7, 8) or 1 for _ in range(size)] for _ in range(size)]
        if int(sympy.Matrix(matrix).det()) != 0:
            break
    latent = [rng.randint(-9, 12) for _ in range(size)]
    values = [sum(a * x for a, x in zip(row, latent, strict=True)) for row in matrix]
    solved = _fraction_solve(matrix, values)
    if solved != list(map(Fraction, latent)):
        raise AssertionError("linear-system verifier disagrees with construction")
    target = [rng.choice([-3, -2, -1, 0, 1, 2, 3]) for _ in range(size)]
    if not any(target):
        target[0] = 1
    answer = sum(coefficient * value for coefficient, value in zip(target, latent, strict=True))
    parameters = {
        "matrix": matrix,
        "values": values,
        "latent_solution": latent,
        "target_coefficients": target,
    }
    return _render_linear_system(matrix, values, target, renderer), str(answer), parameters


def _count_dag_paths_dp(
    node_count: int, edges: list[tuple[int, int]], blocked: set[int]
) -> int:
    """Count source-to-sink paths in a DAG using forward dynamic programming."""
    outgoing: dict[int, list[int]] = {node: [] for node in range(node_count)}
    for source, target in edges:
        outgoing[source].append(target)
    counts = [0] * node_count
    counts[0] = 1
    for source in range(node_count):
        if source in blocked:
            counts[source] = 0
            continue
        for target in outgoing[source]:
            if target not in blocked:
                counts[target] += counts[source]
    return counts[-1]


def _count_dag_paths_recursive(
    node_count: int, edges: list[tuple[int, int]], blocked: set[int]
) -> int:
    """Independently count DAG paths with recursive enumeration."""
    outgoing: dict[int, list[int]] = {node: [] for node in range(node_count)}
    for source, target in edges:
        outgoing[source].append(target)

    def visit(node: int) -> int:
        if node in blocked:
            return 0
        if node == node_count - 1:
            return 1
        return sum(visit(target) for target in outgoing[node])

    return visit(0)


def _render_dag(
    node_count: int, edges: list[tuple[int, int]], blocked: set[int], renderer: int
) -> str:
    edge_text = ", ".join(f"{source}->{target}" for source, target in edges)
    blocked_text = ", ".join(map(str, sorted(blocked)))
    restriction = f" without visiting node(s) {blocked_text}" if blocked else ""
    if renderer == 0:
        return (
            f"A directed acyclic graph has vertices $0,1,\\ldots,{node_count - 1}$ and edges "
            f"{edge_text}. How many directed paths lead from 0 to {node_count - 1}{restriction}?"
        )
    if renderer == 1:
        return (
            f"Using only the directed links {edge_text}, count the distinct routes from vertex 0 "
            f"to vertex {node_count - 1}{restriction}."
        )
    return (
        f"Consider the DAG on nodes $0$ through ${node_count - 1}$. Its edge list is "
        f"[{edge_text}]. Determine the number of $0$-to-${node_count - 1}$ paths{restriction}."
    )


def _generate_dag(rng: random.Random, tier: str, renderer: int) -> tuple[str, str, dict]:
    """Generate a nontrivial DAG path-counting problem with two exact oracles."""
    node_count = {"easy": 8, "medium": 11, "hard": 14}[tier]
    probability = {"easy": 0.34, "medium": 0.29, "hard": 0.25}[tier]
    for _ in range(200):
        edges = [(node, node + 1) for node in range(node_count - 1)]
        edges.extend(
            (source, target)
            for source in range(node_count)
            for target in range(source + 2, node_count)
            if rng.random() < probability
        )
        edges = sorted(set(edges))
        blocked = set()
        if renderer in {1, 2}:
            candidates = list(range(2, node_count - 2))
            blocked = set(rng.sample(candidates, 1 if tier != "hard" else 2))
        answer = _count_dag_paths_dp(node_count, edges, blocked)
        verified = _count_dag_paths_recursive(node_count, edges, blocked)
        if answer == verified and 2 <= answer <= 100_000:
            break
    else:
        raise RuntimeError("could not construct a nontrivial DAG instance")
    parameters = {
        "node_count": node_count,
        "edges": edges,
        "blocked_nodes": sorted(blocked),
    }
    return _render_dag(node_count, edges, blocked, renderer), str(answer), parameters


def _matmul(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    """Multiply small integer matrices exactly."""
    return [
        [
            sum(a * b for a, b in zip(row, column, strict=True))
            for column in zip(*right, strict=True)
        ]
        for row in left
    ]


def _matrix_power(matrix: list[list[int]], exponent: int) -> list[list[int]]:
    """Raise a square integer matrix with binary exponentiation."""
    size = len(matrix)
    result = [[int(row == column) for column in range(size)] for row in range(size)]
    base = matrix
    while exponent:
        if exponent & 1:
            result = _matmul(result, base)
        base = _matmul(base, base)
        exponent //= 2
    return result


def _recurrence_direct(a0: int, a1: int, p: int, q: int, r: int, s: int, n: int) -> int:
    """Evaluate the affine second-order recurrence by direct iteration."""
    previous, current = a0, a1
    for index in range(2, n + 1):
        previous, current = current, p * current + q * previous + r * index + s
    return current


def _recurrence_matrix(a0: int, a1: int, p: int, q: int, r: int, s: int, n: int) -> int:
    """Independently evaluate the recurrence using a state-transition matrix."""
    transition = [[p, q, r, r + s], [1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]
    powered = _matrix_power(transition, n - 1)
    state = [[a1], [a0], [1], [1]]
    return _matmul(powered, state)[0][0]


def _render_recurrence(
    a0: int, a1: int, p: int, q: int, r: int, s: int, n: int, renderer: int
) -> str:
    formula = f"a_n={p}a_{{n-1}}+{q}a_{{n-2}}+{r}n+{s}".replace("+-", "-")
    if renderer == 0:
        return f"Let $a_0={a0}$, $a_1={a1}$, and ${formula}$ for $n\\ge2$. Find $a_{{{n}}}$."
    if renderer == 1:
        return (
            f"A sequence begins with ${a0}, {a1}$. For every $n\\ge2$ it obeys ${formula}$. "
            f"Compute its term with index ${n}$."
        )
    return (
        f"Define $(a_n)$ by $a_0={a0}$ and $a_1={a1}$. The update rule is ${formula}$ "
        f"for $n\\ge2$. What is the exact value of $a_{{{n}}}$?"
    )


def _generate_recurrence(
    rng: random.Random, tier: str, renderer: int
) -> tuple[str, str, dict]:
    """Generate a recurrence problem verified by matrix exponentiation."""
    n = {"easy": 8, "medium": 13, "hard": 20}[tier]
    while True:
        a0, a1 = rng.randint(-5, 8), rng.randint(-5, 8)
        p = rng.choice([-2, -1, 1, 2])
        q = rng.choice([-2, -1, 1, 2])
        r, s = rng.randint(-3, 3), rng.randint(-5, 5)
        direct = _recurrence_direct(a0, a1, p, q, r, s, n)
        verified = _recurrence_matrix(a0, a1, p, q, r, s, n)
        if direct == verified and 10 <= abs(direct) <= 10**15:
            break
    parameters = {"a0": a0, "a1": a1, "p": p, "q": q, "r": r, "s": s, "n": n}
    return _render_recurrence(a0, a1, p, q, r, s, n, renderer), str(direct), parameters


_GENERATORS: dict[str, Callable[[random.Random, str, int], tuple[str, str, dict]]] = {
    "crt": _generate_crt,
    "linear_system": _generate_linear_system,
    "dag_counting": _generate_dag,
    "recurrence": _generate_recurrence,
}


def generate_candidate_bank(
    *, root_seed: int = 20260830, per_cell: int = 10
) -> tuple[list[MathProblem], list[dict[str, object]]]:
    """Generate a balanced problem bank and its separate verification certificates.

    Args:
        root_seed: Seed controlling all instance seeds and renderer assignments.
        per_cell: Number of questions for each family-by-tier cell.

    Returns:
        A pair containing normalized problems and matching certificate dictionaries.

    Raises:
        ValueError: If ``per_cell`` is not positive.
    """
    if per_cell <= 0:
        raise ValueError("per_cell must be positive")
    root_rng = random.Random(root_seed)
    problems: list[MathProblem] = []
    certificates: list[dict[str, object]] = []
    for family in FAMILIES:
        for tier in TIERS:
            for cell_index in range(per_cell):
                instance_seed = root_rng.randrange(2**63)
                renderer = cell_index % RENDERERS_PER_FAMILY
                prompt, answer, parameters = _GENERATORS[family](
                    random.Random(instance_seed), tier, renderer
                )
                question_id = f"proc-v1-{family}-{tier}-{cell_index:02d}-{instance_seed:016x}"
                certificate_core: dict[str, object] = {
                    "question_id": question_id,
                    "generator_version": GENERATOR_VERSION,
                    "instance_seed": instance_seed,
                    "family": family,
                    "difficulty_tier": tier,
                    "renderer_id": renderer,
                    "parameters": parameters,
                    "oracle_answer": answer,
                    "oracle_kind": "construct_then_independent_exact_verifier",
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                }
                certificate = {
                    **certificate_core,
                    "certificate_sha256": _canonical_digest(certificate_core),
                }
                problem = MathProblem(
                    question_id=question_id,
                    task_family="procedural_math",
                    prompt=prompt,
                    gold_answer=answer,
                    difficulty=tier,
                    template_group=f"{family}:renderer-{renderer}",
                    source=f"original:{GENERATOR_VERSION}",
                    metadata={
                        "generator_version": GENERATOR_VERSION,
                        "generator_family": family,
                        "difficulty_tier": tier,
                        "renderer_id": renderer,
                        "instance_seed": instance_seed,
                        "lineage_id": question_id,
                        "structural_parameters": parameters,
                        "oracle_kind": certificate_core["oracle_kind"],
                        "certificate_sha256": certificate["certificate_sha256"],
                    },
                )
                problems.append(problem)
                certificates.append(certificate)
    if len({problem.question_id for problem in problems}) != len(problems):
        raise AssertionError("generated duplicate question IDs")
    if len({problem.prompt for problem in problems}) != len(problems):
        raise AssertionError("generated duplicate prompts")
    return problems, certificates


def verify_problem(problem: MathProblem, certificate: dict[str, object]) -> bool:
    """Recompute a generated problem's oracle and validate its certificate binding."""
    core = {key: value for key, value in certificate.items() if key != "certificate_sha256"}
    if certificate.get("certificate_sha256") != _canonical_digest(core):
        return False
    if certificate.get("question_id") != problem.question_id:
        return False
    if certificate.get("prompt_sha256") != hashlib.sha256(problem.prompt.encode()).hexdigest():
        return False
    if certificate.get("oracle_answer") != problem.gold_answer:
        return False
    family = str(certificate.get("family"))
    params = certificate.get("parameters")
    if not isinstance(params, dict):
        return False
    try:
        if family == "crt":
            answer = _crt_incremental(list(params["residues"]), list(params["moduli"]))
            if answer == 0:
                answer = int(params["modulus_product"])
        elif family == "linear_system":
            solution = _fraction_solve(list(params["matrix"]), list(params["values"]))
            target = list(params["target_coefficients"])
            result = sum(
                coefficient * value
                for coefficient, value in zip(target, solution, strict=True)
            )
            if result.denominator != 1:
                return False
            answer = result.numerator
        elif family == "dag_counting":
            answer = _count_dag_paths_recursive(
                int(params["node_count"]),
                [tuple(edge) for edge in params["edges"]],
                set(params["blocked_nodes"]),
            )
        elif family == "recurrence":
            answer = _recurrence_matrix(
                *(
                    int(params[name])
                    for name in ("a0", "a1", "p", "q", "r", "s", "n")
                )
            )
        else:
            return False
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    return str(answer) == problem.gold_answer
