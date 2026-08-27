"""Z.AI chat-completion adapter for GLM reasoning models."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.generate_rollouts import GenerationRequest, GenerationResult

DEFAULT_BASE_URL = "https://api.z.ai/api/paas/v4"
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
Transport = Callable[[Request, float], bytes]


class ZAIBackendError(RuntimeError):
    """Raised when Z.AI transport or response validation fails."""


def _default_transport(request: Request, timeout: float) -> bytes:
    """Execute one HTTPS request and return its response body."""
    with urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


class ZAIBackend:
    """Synchronous Z.AI adapter preserving reasoning and request-level provenance."""

    seed_supported = False

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 180.0,
        max_retries: int = 3,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        """Configure authentication, transport, and bounded retry behavior."""
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.transport = transport or _default_transport
        self.sleep = sleep

    @classmethod
    def from_env(cls, **kwargs: Any) -> ZAIBackend:
        """Create an adapter from ``ZAI_API_KEY`` without persisting the secret."""
        api_key = os.environ.get("ZAI_API_KEY")
        if not api_key:
            raise ZAIBackendError("ZAI_API_KEY is not set")
        return cls(api_key, **kwargs)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate one GLM response with thinking enabled.

        The Z.AI API does not document deterministic seeds, so ``request.seed`` is deliberately
        not sent. Temperature and nucleus sampling remain active to obtain repeated samples.
        """
        payload = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "do_sample": True,
            "stream": False,
            "thinking": {"type": "enabled", "clear_thinking": True},
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_new_tokens,
        }
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        http_request = Request(
            f"{self.base_url}/chat/completions",
            data=encoded,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept-Language": "en-US,en",
                "User-Agent": "mats12-monitoring/0.1.0",
            },
            method="POST",
        )
        started = time.monotonic()
        body = self._send_with_retries(http_request)
        result = self._parse_response(body)
        return result.model_copy(update={"latency_seconds": time.monotonic() - started})

    def _send_with_retries(self, request: Request) -> bytes:
        """Retry rate limits and transient transport/server failures with bounded backoff."""
        for attempt in range(self.max_retries + 1):
            try:
                return self.transport(request, self.timeout_seconds)
            except HTTPError as exc:
                cause: Exception = exc
                retryable = exc.code in RETRYABLE_STATUS_CODES
                message = f"Z.AI request failed with HTTP {exc.code}"
            except URLError as exc:
                cause = exc
                retryable = True
                message = f"Z.AI transport failed: {exc.reason}"
            if not retryable or attempt == self.max_retries:
                raise ZAIBackendError(message) from cause
            self.sleep(2**attempt)
        raise AssertionError("retry loop exhausted unexpectedly")

    @staticmethod
    def _parse_response(body: bytes) -> GenerationResult:
        """Validate the provider response and normalize fields used by the rollout schema."""
        try:
            payload = json.loads(body)
            choice = payload["choices"][0]
            message = choice["message"]
            content = message["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise ZAIBackendError("Z.AI returned a malformed chat-completion response") from exc
        if not isinstance(content, str) or not content.strip():
            raise ZAIBackendError("Z.AI returned empty final response content")
        reasoning = message.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ZAIBackendError("Z.AI returned non-text reasoning content")
        raw_usage = payload.get("usage") or {}
        usage = {
            key: value
            for key, value in raw_usage.items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
        request_id = payload.get("request_id") or payload.get("id")
        provider_model = payload.get("model")
        return GenerationResult(
            text=content,
            reasoning=reasoning.strip() if reasoning and reasoning.strip() else None,
            finish_reason=str(choice.get("finish_reason") or "unknown"),
            usage=usage,
            provider_request_id=str(request_id) if request_id else None,
            provider_model=str(provider_model) if provider_model else None,
        )
