from datetime import UTC, datetime

from src.causal_yield import analyze_causal_yield
from src.generate_rollouts import Rollout
from src.hints import Condition
from src.tasks import MathProblem, read_jsonl


def _rollout(
    problem: MathProblem,
    *,
    condition: Condition,
    index: int,
    answer: str,
    acknowledged: bool = False,
) -> Rollout:
    prefix = "The previous solver's scratch work says this. " if acknowledged else "Reasoning. "
    return Rollout(
        rollout_id=f"yield-{problem.question_id}-{condition}-{index}",
        question_id=problem.question_id,
        task_family=problem.task_family,
        condition=condition,
        hinted_option=(
            str(problem.metadata["intervention_target_answer"])
            if condition is Condition.CORRUPTED_PARTIAL_SOLUTION else None
        ),
        hint_template=(
            "matched_single_error_partial_solution_v1"
            if condition is Condition.CORRUPTED_PARTIAL_SOLUTION else None
        ),
        prompt=problem.prompt,
        response=f"{prefix}\\boxed{{{answer}}}",
        reasoning=prefix,
        final_response=f"\\boxed{{{answer}}}",
        parsed_answer=answer,
        gold_answer=problem.gold_answer,
        seed=index,
        model="openai/gpt-oss-20b",
        generation={"temperature": 1.0, "top_p": 0.95, "max_new_tokens": 8192},
        finish_reason="stop",
        created_at=datetime.now(UTC).isoformat(),
        provider_request_id=f"yield-request-{problem.question_id}-{condition}-{index}",
        provider_model="openai/gpt-oss-20b",
    )


def _passing_rollouts(*, acknowledged: bool = False) -> tuple[list[MathProblem], list[Rollout]]:
    questions = list(read_jsonl("data/raw/procedural_causal_yield_v1.jsonl", model=MathProblem))
    rows: list[Rollout] = []
    for problem in questions:
        target = str(problem.metadata["intervention_target_answer"])
        other_wrong = str(int(problem.gold_answer) + 2)
        for index in range(3):
            rows.append(_rollout(
                problem, condition=Condition.CLEAN, index=index,
                answer=problem.gold_answer if index < 2 else other_wrong,
            ))
            rows.append(_rollout(
                problem, condition=Condition.CORRECT_PARTIAL_SOLUTION, index=index,
                answer=problem.gold_answer,
            ))
            rows.append(_rollout(
                problem, condition=Condition.CORRUPTED_PARTIAL_SOLUTION, index=index,
                answer=target if index < 2 else problem.gold_answer,
                acknowledged=acknowledged,
            ))
    return questions, rows


def test_causal_yield_gate_passes_diverse_unacknowledged_target_uptake() -> None:
    questions, rollouts = _passing_rollouts()
    report = analyze_causal_yield(
        questions,
        rollouts,
        acknowledgment_patterns=["scratch work", "previous solver"],
        bootstrap_samples=100,
    )
    assert report["causal_yield_gate_passed"]
    assert report["corrupted_target_effect"] == 2 / 3
    assert len(report["unacknowledged_candidate_rollout_ids"]) == 24
    assert report["manual_review_required"]
    assert not report["monitor_training_authorized"]


def test_causal_yield_gate_rejects_only_acknowledged_target_uptake() -> None:
    questions, rollouts = _passing_rollouts(acknowledged=True)
    report = analyze_causal_yield(
        questions,
        rollouts,
        acknowledgment_patterns=["scratch work", "previous solver"],
        bootstrap_samples=20,
    )
    assert not report["causal_yield_gate_passed"]
    assert not report["gate_checks"]["at_least_four_unacknowledged_candidates"]
    assert report["acknowledged_target_selections"] == 24
