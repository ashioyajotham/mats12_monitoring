"""Resumable, append-only execution helpers for typed LLM judges."""

from __future__ import annotations

import hashlib
from collections import Counter

from pydantic import BaseModel, Field

from src.monitor_dataset import MonitorExample
from src.monitors.llm_judge import JudgeScore, TinkerQwenJudge


class JudgeScoreRecord(BaseModel):
    """One score bound to an example, view, judge, prompt, and logical seed."""

    score_id: str
    example_id: str
    question_id: str
    split: str
    view: str
    score: float = Field(ge=0.0, le=1.0)
    rationale: str
    model: str
    prompt_version: str
    seed: int
    attempts: int
    provider_request_id: str
    finish_reason: str
    usage: dict[str, int]


def judge_score_id(
    example_id: str, view: str, model: str, prompt_version: str, seed: int
) -> str:
    """Create a stable identity for resumable judge scoring."""
    payload = f"{example_id}|{view}|{model}|{prompt_version}|{seed}".encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def judge_plan(
    examples: list[MonitorExample], *, model: str, prompt_version: str, base_seed: int
) -> list[tuple[MonitorExample, str, int, str]]:
    """Create the stable transcript/context scoring plan."""
    plan: list[tuple[MonitorExample, str, int, str]] = []
    for example_index, example in enumerate(examples):
        for view_index, view in enumerate(("transcript_only", "context_aware")):
            seed = base_seed + example_index * 10 + view_index
            score_id = judge_score_id(
                example.example_id, view, model, prompt_version, seed
            )
            plan.append((example, view, seed, score_id))
    return plan


def score_judge_plan(
    plan: list[tuple[MonitorExample, str, int, str]],
    judge: TinkerQwenJudge,
    *,
    existing: list[JudgeScoreRecord] | None = None,
) -> tuple[list[JudgeScoreRecord], list[dict[str, str]]]:
    """Score missing plan identities and return records plus bounded failures."""
    records = list(existing or [])
    existing_ids = {row.score_id for row in records}
    if len(existing_ids) != len(records):
        raise ValueError("existing judge score IDs must be unique")
    planned_ids = {item[3] for item in plan}
    if not existing_ids <= planned_ids:
        raise ValueError("existing judge scores contain identities outside the frozen plan")
    failures: list[dict[str, str]] = []
    for example, view, seed, score_id in plan:
        if score_id in existing_ids:
            continue
        evidence = example.transcript if view == "transcript_only" else example.context
        try:
            result: JudgeScore = judge.score(example.example_id, evidence, seed=seed)
        except Exception as exc:
            failures.append(
                {
                    "score_id": score_id,
                    "example_id": example.example_id,
                    "view": view,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        record = JudgeScoreRecord(
            score_id=score_id,
            example_id=example.example_id,
            question_id=example.question_id,
            split=example.split,
            view=result.view,
            score=result.causal_error_probability,
            rationale=result.rationale,
            model=result.model,
            prompt_version=result.prompt_version,
            seed=seed,
            attempts=result.attempts,
            provider_request_id=result.provider_request_id,
            finish_reason=result.finish_reason,
            usage=result.usage,
        )
        records.append(record)
        existing_ids.add(score_id)
    return records, failures


def summarize_judge_scores(
    plan: list[tuple[MonitorExample, str, int, str]],
    records: list[JudgeScoreRecord],
    failures: list[dict[str, str]],
) -> dict[str, object]:
    """Summarize integrity without accessing example labels."""
    provider_ids = [row.provider_request_id for row in records]
    usage: Counter[str] = Counter()
    for row in records:
        usage.update(row.usage)
    return {
        "requested": len(plan),
        "completed": len(records),
        "failures": len(failures),
        "view_counts": dict(sorted(Counter(row.view for row in records).items())),
        "unique_score_ids": len({row.score_id for row in records}),
        "unique_provider_request_ids": len(set(provider_ids)),
        "usage": dict(sorted(usage.items())),
    }
