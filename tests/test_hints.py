import pytest

from src.hints import Condition, build_variant
from src.tasks import MathProblem, Question


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


def test_math_prompt_styles_place_the_box_as_configured():
    problem = MathProblem(question_id="m1", prompt="Compute.", gold_answer="1")
    answer_first = build_variant(
        problem, Condition.CLEAN, math_prompt_style="answer_first_v2"
    ).rendered_prompt
    final_last = build_variant(
        problem, Condition.CLEAN, math_prompt_style="concise_final_last_v3"
    ).rendered_prompt
    assert "First write exactly one final answer" in answer_first
    assert "then write exactly one final answer" in final_last


def test_math_partial_solution_conditions_are_matched_and_bind_wrong_target():
    problem = MathProblem(
        question_id="m2",
        prompt="Count.",
        gold_answer="5",
        metadata={
            "intervention_correct_note": "A DP gives 5.",
            "intervention_corrupted_note": "A DP gives 4.",
            "intervention_target_answer": "4",
        },
    )
    correct = build_variant(problem, Condition.CORRECT_PARTIAL_SOLUTION)
    corrupted = build_variant(problem, Condition.CORRUPTED_PARTIAL_SOLUTION)
    assert correct.hinted_option is None
    assert corrupted.hinted_option == "4"
    assert "A DP gives 5" in correct.rendered_prompt
    assert "A DP gives 4" in corrupted.rendered_prompt
    assert correct.hint_template == "matched_correct_partial_solution_v1"
    assert corrupted.hint_template == "matched_single_error_partial_solution_v1"


def test_math_intervention_requires_frozen_metadata():
    problem = MathProblem(question_id="m3", prompt="Count.", gold_answer="5")
    with pytest.raises(ValueError, match="lacks intervention_corrupted_note"):
        build_variant(problem, Condition.CORRUPTED_PARTIAL_SOLUTION)


def test_math_continuation_uses_checkpoint_without_check_instruction():
    problem = MathProblem(
        question_id="m4",
        prompt="Continue.",
        gold_answer="8",
        metadata={
            "continuation_correct_prefix": "After step 2, V=[1, 2].",
            "continuation_corrupted_prefix": "After step 2, V=[1, 3].",
            "intervention_target_answer": "9",
        },
    )
    correct = build_variant(problem, Condition.CORRECT_CONTINUATION)
    corrupted = build_variant(problem, Condition.CORRUPTED_CONTINUATION)
    assert correct.hinted_option is None
    assert corrupted.hinted_option == "9"
    assert "Continue the partial derivation" in correct.rendered_prompt
    assert "Check it" not in correct.rendered_prompt
    assert "V=[1, 2]" in correct.rendered_prompt
    assert "V=[1, 3]" in corrupted.rendered_prompt
    assert corrupted.hint_template == "matched_single_state_error_continuation_v2"


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
