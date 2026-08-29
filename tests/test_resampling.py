import pytest

from src.generate_rollouts import Rollout
from src.hints import HINT_CONDITIONS, Condition
from src.resampling import estimate_answer_shifts


def make(
    condition: Condition,
    answer: str | None,
    seed: int,
    *,
    hinted_option: str = "B",
) -> Rollout:
    return Rollout(
        rollout_id=f"r{seed}",
        question_id="q1",
        task_family="test",
        condition=condition,
        hinted_option=hinted_option if condition in HINT_CONDITIONS else None,
        hint_template=None,
        prompt="p",
        response=f"Final answer: {answer}" if answer else "Unparseable",
        parsed_answer=answer,
        gold_answer="A",
        seed=seed,
        model="mock",
        generation={"temperature": 1.0},
        finish_reason="stop",
        created_at="2026-08-27T00:00:00+00:00",
    )


def test_answer_shift_uses_distributions_not_single_flip():
    rows = [make(Condition.CLEAN, answer, i) for i, answer in enumerate("AAAAB")]
    rows += [
        make(Condition.INCORRECT_ANSWER_HINT, answer, i + 5) for i, answer in enumerate("ABBBB")
    ]
    result = estimate_answer_shifts(rows)[0]
    assert result.p_clean_hint == 0.2
    assert result.p_hinted_hint == 0.8
    assert result.hint_effect == pytest.approx(0.6)


def test_invalid_answers_are_excluded_from_denominators():
    rows = [make(Condition.CLEAN, "A", 1), make(Condition.CLEAN, None, 2)]
    rows += [
        make(Condition.INCORRECT_ANSWER_HINT, "B", 3),
        make(Condition.INCORRECT_ANSWER_HINT, None, 4),
    ]
    result = estimate_answer_shifts(rows)[0]
    assert result.n_clean == 1
    assert result.n_hinted == 1
    assert result.hint_effect == 1.0


def test_multiple_hinted_options_produce_separate_estimates():
    rows = [make(Condition.CLEAN, "A", 1), make(Condition.CLEAN, "B", 2)]
    rows += [make(Condition.INCORRECT_ANSWER_HINT, "B", 3, hinted_option="B")]
    rows += [make(Condition.INCORRECT_ANSWER_HINT, "C", 4, hinted_option="C")]
    results = estimate_answer_shifts(rows)
    assert [result.hinted_option for result in results] == ["B", "C"]
    assert [result.n_hinted for result in results] == [1, 1]


def test_trusted_hint_condition_contributes_to_shift():
    rows = [make(Condition.CLEAN, "A", 1)]
    rows += [make(Condition.TRUSTED_INCORRECT_ANSWER_HINT, "B", 2)]

    result = estimate_answer_shifts(rows)[0]

    assert result.hint_effect == 1.0
