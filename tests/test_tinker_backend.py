from types import SimpleNamespace

import pytest

from src.backends.tinker import TinkerBackend, TinkerBackendError
from src.generate_rollouts import GenerationRequest


class FakePrompt:
    def to_ints(self):
        return [1, 2, 3]


class FakeFuture:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class FakeRenderer:
    def build_generation_prompt(self, messages):
        assert messages == [{"role": "user", "content": "Question"}]
        return FakePrompt()

    def get_stop_sequences(self):
        return [99]

    def parse_response(self, tokens):
        assert tokens == [4, 5]
        return {"role": "assistant", "content": []}, SimpleNamespace(is_clean=True)

    def to_openai_message(self, message):
        return {
            "role": "assistant",
            "reasoning_content": "Compare options.",
            "content": "Final answer: B",
        }


class FakeTypes:
    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs


def request(model="Qwen/Qwen3.6-35B-A3B"):
    return GenerationRequest(
        model=model,
        prompt="Question",
        seed=1700,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=1024,
    )


def test_tinker_backend_sends_true_seed_and_preserves_reasoning():
    captured = {}

    class SamplingClient:
        def sample(self, *, prompt, sampling_params, num_samples):
            captured["prompt"] = prompt
            captured["params"] = sampling_params.kwargs
            captured["num_samples"] = num_samples
            sequence = SimpleNamespace(
                tokens=[4, 5], stop_reason="stop", sequence_id="sequence-1"
            )
            return FakeFuture(
                SimpleNamespace(sequences=[sequence], prompt_cache_hit_tokens=2)
            )

    backend = TinkerBackend(
        "Qwen/Qwen3.6-35B-A3B",
        sampling_client=SamplingClient(),
        renderer=FakeRenderer(),
        types_module=FakeTypes,
    )
    result = backend.generate(request())
    assert captured["params"]["seed"] == 1700
    assert captured["params"]["stop"] == [99]
    assert captured["num_samples"] == 1
    assert result.reasoning == "Compare options."
    assert result.text == "Final answer: B"
    assert result.provider_request_id == "sequence-1"
    assert result.usage == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
        "prompt_cache_hit_tokens": 2,
    }


def test_tinker_backend_rejects_model_mismatch():
    backend = TinkerBackend(
        "Qwen/Qwen3.6-35B-A3B",
        sampling_client=object(),
        renderer=FakeRenderer(),
        types_module=FakeTypes,
    )
    with pytest.raises(TinkerBackendError, match="does not match"):
        backend.generate(request(model="different"))


def test_tinker_backend_requires_environment_key(monkeypatch):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    with pytest.raises(TinkerBackendError, match="not set"):
        TinkerBackend("Qwen/Qwen3.6-35B-A3B")
