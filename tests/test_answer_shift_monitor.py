import pytest

from src.generate_rollouts import Rollout, RolloutStatus
from src.hints import Condition
from src.monitors.answer_shift import score_counterfactual_answer_shifts


def _row(index: int, condition: Condition, answer: str) -> Rollout:
    return Rollout(
        rollout_id=f"r-{index}",
        question_id="q1",
        task_family="test",
        condition=condition,
        hinted_option="B" if condition is Condition.CORRUPTED_CONTINUATION else None,
        hint_template=None,
        prompt="p",
        response="r",
        reasoning="r",
        final_response=f"\\boxed{{{answer}}}",
        parsed_answer=answer,
        gold_answer="A",
        seed=index,
        model="model",
        generation={"temperature": 1.0},
        finish_reason="stop",
        created_at="2026-08-30T00:00:00+00:00",
        status=RolloutStatus.CLEAN_STOP,
    )


def test_answer_shift_is_leave_one_out_and_uses_conservative_control() -> None:
    rows = [
        _row(0, Condition.CLEAN, "C"),
        _row(1, Condition.CLEAN, "C"),
        _row(2, Condition.CLEAN, "A"),
        _row(3, Condition.CORRECT_CONTINUATION, "A"),
        _row(4, Condition.CORRECT_CONTINUATION, "A"),
        _row(5, Condition.CORRECT_CONTINUATION, "A"),
        _row(6, Condition.CORRUPTED_CONTINUATION, "B"),
        _row(7, Condition.CORRUPTED_CONTINUATION, "B"),
        _row(8, Condition.CORRUPTED_CONTINUATION, "B"),
    ]
    corrupted, clean = score_counterfactual_answer_shifts(["r-6", "r-0"], rows)
    assert corrupted.raw_shift == 1.0
    assert corrupted.score == 1.0
    assert clean.clean_frequency == pytest.approx(0.5)
    assert clean.raw_shift == pytest.approx(-0.5)
    assert clean.score == pytest.approx(0.25)


def test_answer_shift_rejects_missing_sibling_cells() -> None:
    rows = [_row(0, Condition.CLEAN, "C")]
    with pytest.raises(ValueError, match="cell is empty"):
        score_counterfactual_answer_shifts(["r-0"], rows)
