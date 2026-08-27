import json
from io import BytesIO
from urllib.error import HTTPError, URLError

import pytest

from src.backends.zai import ZAIBackend, ZAIBackendError
from src.generate_rollouts import GenerationRequest


def generation_request() -> GenerationRequest:
    return GenerationRequest(
        model="glm-4.7-flash",
        prompt="Question",
        seed=1700,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=1024,
    )


def response_body() -> bytes:
    return json.dumps(
        {
            "id": "completion-id",
            "request_id": "request-id",
            "model": "glm-4.7-flash",
            "choices": [
                {
                    "message": {
                        "reasoning_content": "Compare the options.",
                        "content": "Final answer: B",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
    ).encode()


def test_zai_backend_sends_sampling_but_not_unsupported_seed():
    captured = {}

    def transport(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return response_body()

    result = ZAIBackend("secret", transport=transport).generate(generation_request())
    assert captured["url"].endswith("/chat/completions")
    assert captured["payload"]["thinking"] == {"type": "enabled", "clear_thinking": True}
    assert captured["payload"]["temperature"] == 0.8
    assert "seed" not in captured["payload"]
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert result.reasoning == "Compare the options."
    assert result.text == "Final answer: B"
    assert result.provider_request_id == "request-id"
    assert result.usage["total_tokens"] == 30
    assert result.latency_seconds is not None


def test_zai_backend_retries_transient_transport_failures():
    attempts = 0
    sleeps = []

    def transport(request, timeout):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise URLError("temporary")
        return response_body()

    result = ZAIBackend(
        "secret", transport=transport, max_retries=3, sleep=sleeps.append
    ).generate(generation_request())
    assert result.text == "Final answer: B"
    assert attempts == 3
    assert sleeps == [1, 2]


def test_zai_backend_does_not_retry_authentication_failure():
    attempts = 0

    def transport(request, timeout):
        nonlocal attempts
        attempts += 1
        raise HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    with pytest.raises(ZAIBackendError, match="HTTP 401"):
        ZAIBackend("secret", transport=transport).generate(generation_request())
    assert attempts == 1


def test_zai_backend_reports_bounded_provider_error_detail():
    def transport(request, timeout):
        body = BytesIO(json.dumps({"error": {"code": "1305", "message": "Rate limit"}}).encode())
        raise HTTPError(request.full_url, 429, "Too Many Requests", {}, body)

    with pytest.raises(ZAIBackendError, match=r"HTTP 429 \(1305: Rate limit\)"):
        ZAIBackend("secret", transport=transport, max_retries=0).generate(generation_request())


@pytest.mark.parametrize(
    "body",
    [b"not-json", b"{}", b'{"choices": []}', b'{"choices":[{"message":{"content":""}}]}'],
)
def test_zai_backend_rejects_malformed_or_empty_responses(body):
    backend = ZAIBackend("secret", transport=lambda request, timeout: body)
    with pytest.raises(ZAIBackendError, match="malformed|empty"):
        backend.generate(generation_request())


def test_zai_backend_retries_socket_timeouts():
    attempts = 0

    def transport(request, timeout):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("read timed out")
        return response_body()

    result = ZAIBackend(
        "secret", transport=transport, max_retries=1, sleep=lambda _: None
    ).generate(generation_request())
    assert result.text == "Final answer: B"
    assert attempts == 2


def test_zai_backend_requires_environment_key(monkeypatch):
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    with pytest.raises(ZAIBackendError, match="not set"):
        ZAIBackend.from_env()
