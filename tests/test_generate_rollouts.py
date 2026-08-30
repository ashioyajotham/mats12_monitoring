from src.generate_rollouts import (
    GenerationRequest,
    GenerationResult,
    MockBackend,
    RolloutStatus,
    collect_rollout,
    parse_answer,
    rollout_id,
    summarize_rollouts,
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


def test_collect_rollout_preserves_reasoning_and_provider_metadata():
    class Backend:
        def generate(self, request: GenerationRequest) -> GenerationResult:
            return GenerationResult(
                text="Final answer: B",
                reasoning="First compare the options.",
                usage={"total_tokens": 12},
                provider_request_id="request-1",
                provider_model="glm-4.7-flash",
            )

    item = question()
    rollout = collect_rollout(
        item,
        build_variant(item, Condition.CLEAN),
        Backend(),
        model="glm-4.7-flash",
        seed=9,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=1024,
    )
    assert rollout.response == "First compare the options.\n\nFinal answer: B"
    assert rollout.reasoning == "First compare the options."
    assert rollout.final_response == "Final answer: B"
    assert rollout.parsed_answer == "B"
    assert rollout.provider_request_id == "request-1"
    assert rollout.usage == {"total_tokens": 12}

    summary = summarize_rollouts([rollout])
    assert summary["completed"] == 1
    assert summary["invalid"] == 0
    assert summary["reasoning_present"] == 1
    assert summary["unique_provider_request_ids"] == 1
    assert summary["provider_models"] == {"glm-4.7-flash": 1}
    assert summary["usage"] == {"total_tokens": 12}


def test_collect_rollout_classifies_truncated_provider_response():
    class Backend:
        def generate(self, request: GenerationRequest) -> GenerationResult:
            return GenerationResult(
                text="",
                reasoning="An unfinished derivation",
                finish_reason="length",
                termination_clean=False,
                parse_termination="malformed",
            )

    rollout = collect_rollout(
        question(),
        build_variant(question(), Condition.CLEAN),
        Backend(),
        model="model",
        seed=1,
        temperature=1.0,
        top_p=0.95,
        max_new_tokens=32,
    )
    assert rollout.status is RolloutStatus.LENGTH_TRUNCATED
    assert rollout.parsed_answer is None
    assert summarize_rollouts([rollout])["truncated"] == 1
