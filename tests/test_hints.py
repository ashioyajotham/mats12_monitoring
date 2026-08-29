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


def test_trusted_hint_is_stronger_and_still_records_incorrect_option():
    variant = build_variant(
        question(), Condition.TRUSTED_INCORRECT_ANSWER_HINT, hinted_option="B"
    )

    assert variant.hinted_option == "B"
    assert variant.hint_template == "trusted_answer_key_metadata_v1"
    assert "trusted answer-key retrieval system" in variant.rendered_prompt
    assert "passed its validation checks" in variant.rendered_prompt


def test_intermediate_hint_uses_uncertain_automated_annotation():
    variant = build_variant(
        question(), Condition.INTERMEDIATE_INCORRECT_ANSWER_HINT, hinted_option="B"
    )

    assert variant.hinted_option == "B"
    assert variant.hint_template == "automated_preprocessing_annotation_v1"
    assert "likely correct" in variant.rendered_prompt
    assert "trusted" not in variant.rendered_prompt
    assert "passed its validation checks" not in variant.rendered_prompt


def test_peer_scratch_note_is_non_authoritative_and_records_option():
    variant = build_variant(question(), Condition.PEER_SCRATCH_NOTE, hinted_option="B")

    assert variant.hinted_option == "B"
    assert variant.hint_template == "peer_scratch_note_v1"
    assert "scratch note" in variant.rendered_prompt
    assert "trusted" not in variant.rendered_prompt
    assert "correct" not in variant.rendered_prompt


@pytest.mark.parametrize(
    ("condition", "authority", "directive", "template"),
    [
        (
            Condition.AUTHORITY_INFORMATIONAL_HINT,
            True,
            False,
            "factorial_authority_1_directive_0_v1",
        ),
        (
            Condition.ORDINARY_DIRECTIVE_HINT,
            False,
            True,
            "factorial_authority_0_directive_1_v1",
        ),
        (
            Condition.AUTHORITY_DIRECTIVE_HINT,
            True,
            True,
            "factorial_authority_1_directive_1_v1",
        ),
    ],
)
def test_factorial_hints_change_only_authority_and_directive(
    condition: Condition, authority: bool, directive: bool, template: str
):
    variant = build_variant(question(), condition, hinted_option="B")

    assert variant.hint_template == template
    assert ("trusted" in variant.rendered_prompt) is authority
    assert ("Use this annotation" in variant.rendered_prompt) is directive
    assert "likely correct" in variant.rendered_prompt
