"""Backend-neutral, auditable rollout collection."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from src.hints import Condition, PromptVariant
from src.tasks import Question

ANSWER_RE = re.compile(r"final answer\s*:\s*([A-Za-z0-9]+)", re.IGNORECASE)


class GenerationRequest(BaseModel):
    """Backend-neutral request for one stochastic model generation."""
    model: str
    prompt: str
    seed: int
    temperature: float
    top_p: float
    max_new_tokens: int


class GenerationResult(BaseModel):
    """Text and provider metadata returned by a generation backend."""
    text: str
    finish_reason: str = "stop"
    usage: dict[str, int] = Field(default_factory=dict)


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
    parsed_answer: str | None
    gold_answer: str
    seed: int
    model: str
    generation: dict[str, float | int]
    finish_reason: str
    created_at: str


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
    question: Question,
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
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    result = backend.generate(request)
    return Rollout(
        rollout_id=rollout_id(question.question_id, variant.condition, seed, model),
        question_id=question.question_id,
        task_family=question.task_family,
        condition=variant.condition,
        hinted_option=variant.hinted_option,
        hint_template=variant.hint_template,
        prompt=variant.rendered_prompt,
        response=result.text,
        parsed_answer=parse_answer(result.text, question.options),
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
