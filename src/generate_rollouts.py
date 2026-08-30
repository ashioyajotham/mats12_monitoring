"""Backend-neutral, auditable rollout collection."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from src.hints import Condition, PromptVariant
from src.math_answers import extract_math_answer
from src.tasks import MathProblem, Question

ANSWER_RE = re.compile(r"final answer\s*:\s*([A-Za-z0-9]+)", re.IGNORECASE)


class GenerationRequest(BaseModel):
    """Backend-neutral request for one stochastic model generation.

    ``seed`` is always a stable sample identifier. A backend may additionally use it as an
    inference seed only when the provider explicitly supports seeded generation.
    """
    model: str
    prompt: str
    assistant_prefill: str | None = None
    seed: int
    temperature: float
    top_p: float
    max_new_tokens: int


class GenerationResult(BaseModel):
    """Text and provider metadata returned by a generation backend."""
    text: str
    reasoning: str | None = None
    finish_reason: str = "stop"
    usage: dict[str, int] = Field(default_factory=dict)
    provider_request_id: str | None = None
    provider_model: str | None = None
    latency_seconds: float | None = Field(default=None, ge=0.0)
    termination_clean: bool = True
    parse_termination: str | None = None
    raw_response: str | None = None
    generated_reasoning: str | None = None
    assistant_prefill: str | None = None
    prefill_tokens: int = 0


class RolloutStatus(StrEnum):
    """Whether a provider response is suitable for experimental scoring."""

    CLEAN_STOP = "clean_stop"
    PARSE_INVALID = "parse_invalid"
    LENGTH_TRUNCATED = "length_truncated"
    MALFORMED = "malformed"


class GenerationBackend(Protocol):
    """Interface implemented by model-provider adapters."""

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate one result for a normalized request."""
        ...


class Rollout(BaseModel):
    """Immutable record of one prompt, response, and decoding configuration."""
    rollout_id: str
    question_id: str
    task_family: str
    condition: Condition
    hinted_option: str | None
    hint_template: str | None
    prompt: str
    response: str
    reasoning: str | None = None
    final_response: str | None = None
    parsed_answer: str | None
    gold_answer: str
    seed: int
    model: str
    generation: dict[str, float | int]
    finish_reason: str
    created_at: str
    usage: dict[str, int] = Field(default_factory=dict)
    provider_request_id: str | None = None
    provider_model: str | None = None
    latency_seconds: float | None = Field(default=None, ge=0.0)
    status: RolloutStatus = RolloutStatus.CLEAN_STOP
    parse_termination: str | None = None
    raw_response: str | None = None
    assistant_prefill: str | None = None
    generated_reasoning: str | None = None
    prefill_tokens: int = 0


class MockBackend:
    """Deterministic plumbing test. Its outputs are not research evidence."""

    def __init__(self, options: list[str], gold_answer: str, hinted_option: str | None):
        """Initialize deterministic plumbing behavior for one question variant."""
        self.options = options
        self.gold_answer = gold_answer
        self.hinted_option = hinted_option

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate a deterministic synthetic answer from the request seed."""
        rng = random.Random(request.seed)
        answer = self.gold_answer
        if self.hinted_option and rng.random() < 0.45:
            answer = self.hinted_option
        elif rng.random() < 0.15:
            answer = rng.choice(self.options)
        return GenerationResult(text=f"I compared the available options. Final answer: {answer}")


def parse_answer(text: str, valid_options: Iterable[str]) -> str | None:
    """Extract a final answer and require membership in the valid option set."""
    match = ANSWER_RE.search(text)
    if not match:
        return None
    answer = match.group(1).upper()
    return answer if answer in set(valid_options) else None


def rollout_id(question_id: str, condition: str, seed: int, model: str) -> str:
    """Create a stable identifier for a rollout's experimental identity."""
    value = f"{question_id}|{condition}|{seed}|{model}".encode()
    return hashlib.sha256(value).hexdigest()[:20]


def collect_rollout(
    question: Question | MathProblem,
    variant: PromptVariant,
    backend: GenerationBackend,
    *,
    model: str,
    seed: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> Rollout:
    """Collect and normalize one generation from a backend."""
    request = GenerationRequest(
        model=model,
        prompt=variant.rendered_prompt,
        assistant_prefill=variant.assistant_prefill,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    result = backend.generate(request)
    transcript = result.text
    if result.reasoning:
        transcript = f"{result.reasoning.rstrip()}\n\n{result.text.lstrip()}"
    parsed_answer = (
        parse_answer(result.text, question.options)
        if isinstance(question, Question)
        else extract_math_answer(result.text)
    )
    if not result.termination_clean:
        status = (
            RolloutStatus.LENGTH_TRUNCATED
            if result.finish_reason == "length"
            else RolloutStatus.MALFORMED
        )
    elif parsed_answer is None:
        status = RolloutStatus.PARSE_INVALID
    else:
        status = RolloutStatus.CLEAN_STOP
    return Rollout(
        rollout_id=rollout_id(question.question_id, variant.condition, seed, model),
        question_id=question.question_id,
        task_family=question.task_family,
        condition=variant.condition,
        hinted_option=variant.hinted_option,
        hint_template=variant.hint_template,
        prompt=variant.rendered_prompt,
        response=transcript,
        reasoning=result.reasoning,
        final_response=result.text,
        parsed_answer=parsed_answer,
        gold_answer=question.gold_answer,
        seed=seed,
        model=model,
        generation={
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
        },
        finish_reason=result.finish_reason,
        created_at=datetime.now(UTC).isoformat(),
        usage=result.usage,
        provider_request_id=result.provider_request_id,
        provider_model=result.provider_model,
        latency_seconds=result.latency_seconds,
        status=status,
        parse_termination=result.parse_termination,
        raw_response=result.raw_response,
        assistant_prefill=result.assistant_prefill or variant.assistant_prefill,
        generated_reasoning=result.generated_reasoning,
        prefill_tokens=result.prefill_tokens,
    )


def write_manifest(path: str | Path, payload: dict) -> None:
    """Create a content-addressed run manifest without overwriting prior evidence.

    The manifest digest covers the supplied payload and therefore excludes the digest field
    itself.

    Raises:
        FileExistsError: If ``path`` already exists.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    manifest = {**payload, "manifest_sha256": hashlib.sha256(canonical.encode()).hexdigest()}
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def summarize_rollouts(rollouts: Iterable[Rollout]) -> dict:
    """Summarize response integrity, provider identity, latency, and usage for a run."""
    rows = list(rollouts)
    latencies = [item.latency_seconds for item in rows if item.latency_seconds is not None]
    request_ids = [item.provider_request_id for item in rows if item.provider_request_id]
    usage: Counter[str] = Counter()
    for item in rows:
        usage.update(item.usage)
    return {
        "completed": len(rows),
        "invalid": sum(item.parsed_answer is None for item in rows),
        "scorable": sum(item.status is RolloutStatus.CLEAN_STOP for item in rows),
        "truncated": sum(item.status is RolloutStatus.LENGTH_TRUNCATED for item in rows),
        "malformed": sum(item.status is RolloutStatus.MALFORMED for item in rows),
        "parse_invalid": sum(item.status is RolloutStatus.PARSE_INVALID for item in rows),
        "reasoning_present": sum(bool(item.reasoning) for item in rows),
        "unique_provider_request_ids": len(set(request_ids)),
        "finish_reasons": dict(sorted(Counter(item.finish_reason for item in rows).items())),
        "provider_models": dict(
            sorted(Counter(item.provider_model or "unknown" for item in rows).items())
        ),
        "latency_seconds": {
            "mean": sum(latencies) / len(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "usage": dict(sorted(usage.items())),
    }
