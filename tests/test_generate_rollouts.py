from src.generate_rollouts import (
    MockBackend,
    collect_rollout,
    parse_answer,
    rollout_id,
)
from src.hints import Condition, build_variant
from src.tasks import Question


def question() -> Question:
    return Question(
        question_id="q1",
        task_family="test",
        prompt="Choose.",
        options={"A": "one", "B": "two"},
        gold_answer="A",
    )


def test_parse_answer_is_case_insensitive_and_option_checked():
    assert parse_answer("Reasoning. final answer: b", ["A", "B"]) == "B"
    assert parse_answer("Final answer: C", ["A", "B"]) is None
    assert parse_answer("No final answer", ["A", "B"]) is None


def test_rollout_id_is_stable_and_sensitive_to_identity_fields():
    first = rollout_id("q1", "clean", 1, "model")
    assert first == rollout_id("q1", "clean", 1, "model")
    assert first != rollout_id("q1", "clean", 2, "model")


def test_collect_rollout_records_prompt_and_generation_settings():
    item = question()
    variant = build_variant(item, Condition.CLEAN)
    rollout = collect_rollout(
        item,
        variant,
        MockBackend(["A", "B"], "A", None),
        model="mock",
        seed=9,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=64,
    )
    assert rollout.question_id == item.question_id
    assert rollout.prompt == variant.rendered_prompt
    assert rollout.parsed_answer in item.options
    assert rollout.generation == {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 64}
