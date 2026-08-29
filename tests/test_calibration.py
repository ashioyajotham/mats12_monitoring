"""Tests for calibration question selection."""

from src.calibration import rank_by_clean_deliberation
from src.generate_rollouts import Rollout


def rollout(question_id: str, tokens: int, answer: str = "A") -> Rollout:
    """Build a minimal clean rollout with token telemetry."""
    return Rollout(
        rollout_id=f"{question_id}-{tokens}",
        question_id=question_id,
        task_family="test",
        condition="clean",
        hinted_option=None,
        hint_template=None,
        model="test",
        seed=tokens,
        prompt="prompt",
        reasoning="reasoning",
        response=f"Final answer: {answer}",
        final_response=f"Final answer: {answer}",
        parsed_answer=answer,
        gold_answer="A",
        generation={"temperature": 1.0, "top_p": 0.95, "max_new_tokens": 100},
        finish_reason="stop",
        created_at="2026-08-29T00:00:00+00:00",
        usage={"completion_tokens": tokens},
    )


def test_rank_by_clean_deliberation_uses_mean_and_stable_ties():
    rows = [rollout("q2", 30), rollout("q1", 20), rollout("q1", 40)]

    assert rank_by_clean_deliberation(rows) == [("q1", 30.0, 2), ("q2", 30.0, 1)]


def test_rank_excludes_questions_with_observed_clean_error():
    rows = [rollout("correct", 10), rollout("error", 100, answer="B")]

    assert rank_by_clean_deliberation(rows) == [("correct", 10.0, 1)]
