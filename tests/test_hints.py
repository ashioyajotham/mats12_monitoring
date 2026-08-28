import pytest

from src.hints import Condition, build_variant
from src.tasks import Question


def question() -> Question:
    return Question(
        question_id="q1",
        task_family="test",
        prompt="Choose.",
        options={"A": "one", "B": "two", "C": "three"},
        gold_answer="A",
    )


def test_clean_has_no_hint():
    variant = build_variant(question(), Condition.CLEAN)
    assert variant.hinted_option is None
    assert "earlier system" not in variant.rendered_prompt
    assert "carefully but concisely" in variant.rendered_prompt
    assert variant.rendered_prompt.endswith("`Final answer: <OPTION>`.")


def test_incorrect_hint_is_not_gold():
    variant = build_variant(question(), Condition.INCORRECT_ANSWER_HINT)
    assert variant.hinted_option != question().gold_answer
    assert variant.hinted_option in variant.rendered_prompt


def test_explicit_gold_hint_is_rejected():
    with pytest.raises(ValueError, match="must be incorrect"):
        build_variant(question(), Condition.INCORRECT_ANSWER_HINT, hinted_option="A")
