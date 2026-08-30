from datetime import UTC, datetime

from src.assistant_prefill_yield import analyze_assistant_prefill_yield
from src.generate_rollouts import Rollout, RolloutStatus
from src.hints import Condition
from src.tasks import MathProblem, read_jsonl


def _rollout(
    problem: MathProblem, condition: Condition, sample: int, answer: str
) -> Rollout:
    prefill = None
    if condition is Condition.CORRECT_ASSISTANT_PREFILL:
        prefill = str(problem.metadata["assistant_prefill_correct"])
    elif condition is Condition.CORRUPTED_ASSISTANT_PREFILL:
        prefill = str(problem.metadata["assistant_prefill_corrupted"])
    generated = "I propagate the state to the requested result." if prefill else "I solve it."
    reasoning = f"{prefill}{generated}" if prefill else generated
    return Rollout(
        rollout_id=f"{problem.question_id}-{condition}-{sample}",
        question_id=problem.question_id,
        task_family=problem.task_family,
        condition=condition,
        hinted_option=(
            str(problem.metadata["intervention_target_answer"])
            if condition is Condition.CORRUPTED_ASSISTANT_PREFILL else None
        ),
        hint_template=None,
        prompt=problem.prompt,
        response=f"{reasoning}\n\n\\boxed{{{answer}}}",
        reasoning=reasoning,
        final_response=f"\\boxed{{{answer}}}",
        parsed_answer=answer,
        gold_answer=problem.gold_answer,
        seed=20262410 + sample,
        model="openai/gpt-oss-20b",
        generation={"temperature": 1.0, "top_p": 0.95, "max_new_tokens": 8192},
        finish_reason="stop",
        created_at=datetime.now(UTC).isoformat(),
        provider_request_id=f"provider-{problem.question_id}-{condition}-{sample}",
        status=RolloutStatus.CLEAN_STOP,
        assistant_prefill=prefill,
        generated_reasoning=generated,
        prefill_tokens=20 if prefill else 0,
    )


def test_assistant_prefill_gate_passes_diverse_exact_propagation():
    questions = list(
        read_jsonl("data/raw/procedural_assistant_prefill_v3.jsonl", model=MathProblem)
    )
    rollouts = []
    for problem in questions:
        target = str(problem.metadata["intervention_target_answer"])
        for sample in range(3):
            rollouts.append(_rollout(problem, Condition.CLEAN, sample, problem.gold_answer))
            rollouts.append(
                _rollout(
                    problem,
                    Condition.CORRECT_ASSISTANT_PREFILL,
                    sample,
                    problem.gold_answer,
                )
            )
            rollouts.append(
                _rollout(
                    problem,
                    Condition.CORRUPTED_ASSISTANT_PREFILL,
                    sample,
                    target,
                )
            )
    report = analyze_assistant_prefill_yield(
        questions, rollouts, resistance_patterns=["recheck", "mistake"]
    )
    assert report["automated_causal_gate_passed"] is True
    assert report["manual_validity_gate_passed"] is False
    assert report["corrupted_minus_correct_target_effect"] == 1.0
    assert report["prefill_reconstruction_failure_ids"] == []
