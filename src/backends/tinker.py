"""Native Tinker sampling adapter for reproducible reasoning-model rollouts."""

from __future__ import annotations

import os
import time
from typing import Any

from src.generate_rollouts import GenerationRequest, GenerationResult


class TinkerBackendError(RuntimeError):
    """Raised when Tinker setup, sampling, or response validation fails."""


class TinkerBackend:
    """Sample Qwen through Tinker's native seeded API and official renderer."""

    seed_supported = True

    def __init__(
        self,
        model: str,
        *,
        renderer_name: str = "qwen3_5",
        sampling_client: Any | None = None,
        renderer: Any | None = None,
        types_module: Any | None = None,
    ):
        """Create or inject the native sampling client and chat renderer."""
        self.model = model
        self.tokenizer: Any | None = None
        if sampling_client is not None and renderer is not None and types_module is not None:
            self.sampling_client = sampling_client
            self.renderer = renderer
            self.types = types_module
            return
        if not os.environ.get("TINKER_API_KEY"):
            raise TinkerBackendError("TINKER_API_KEY is not set")
        try:
            import tinker
            from tinker_cookbook import renderers

            service_client = tinker.ServiceClient()
            self.sampling_client = service_client.create_sampling_client(base_model=model)
            tokenizer = self.sampling_client.get_tokenizer()
            self.tokenizer = tokenizer
            self.renderer = renderers.get_renderer(renderer_name, tokenizer, model_name=model)
            self.types = tinker
        except (ImportError, RuntimeError, ValueError) as exc:
            raise TinkerBackendError(f"cannot initialize Tinker backend: {exc}") from exc

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate one seeded sample and preserve separate reasoning and final text."""
        if request.model != self.model:
            raise TinkerBackendError(
                f"request model {request.model!r} does not match backend model {self.model!r}"
            )
        started = time.monotonic()
        try:
            prompt = self.renderer.build_generation_prompt(
                [{"role": "user", "content": request.prompt}]
            )
            params = self.types.SamplingParams(
                max_tokens=request.max_new_tokens,
                seed=request.seed,
                stop=self.renderer.get_stop_sequences(),
                temperature=request.temperature,
                top_p=request.top_p,
            )
            result = self.sampling_client.sample(
                prompt=prompt, sampling_params=params, num_samples=1
            ).result()
            sequence = result.sequences[0]
            message, termination = self.renderer.parse_response(sequence.tokens)
            try:
                normalized = self.renderer.to_openai_message(message)
            except Exception:
                if getattr(termination, "is_clean", bool(termination)):
                    raise
                normalized = {"content": "", "reasoning_content": None}
        except TinkerBackendError:
            raise
        except Exception as exc:
            raise TinkerBackendError(f"Tinker sampling failed: {exc}") from exc

        termination_clean = getattr(termination, "is_clean", bool(termination))
        raw_response = None
        if self.tokenizer is not None:
            try:
                raw_response = self.tokenizer.decode(sequence.tokens)
            except Exception:
                raw_response = None
        content = normalized.get("content")
        reasoning = normalized.get("reasoning_content")
        if termination_clean and (not isinstance(content, str) or not content.strip()):
            raise TinkerBackendError("Tinker returned empty final response content")
        if not isinstance(content, str):
            content = ""
        if reasoning is not None and not isinstance(reasoning, str):
            raise TinkerBackendError("Tinker returned non-text reasoning content")
        if not termination_clean and not reasoning and raw_response:
            reasoning = raw_response
        prompt_tokens = len(prompt.to_ints())
        completion_tokens = len(sequence.tokens)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        cache_hits = getattr(result, "prompt_cache_hit_tokens", 0)
        if isinstance(cache_hits, int):
            usage["prompt_cache_hit_tokens"] = cache_hits
        return GenerationResult(
            text=content.strip(),
            reasoning=reasoning.strip() if reasoning and reasoning.strip() else None,
            finish_reason=str(sequence.stop_reason),
            usage=usage,
            provider_request_id=str(sequence.sequence_id),
            provider_model=self.model,
            latency_seconds=time.monotonic() - started,
            termination_clean=termination_clean,
            parse_termination=str(termination),
            raw_response=raw_response,
        )
