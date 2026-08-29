"""Difficulty-proxy selection for bounded intervention calibration."""

from __future__ import annotations

from collections import defaultdict

from src.generate_rollouts import Rollout
from src.hints import Condition


def rank_by_clean_deliberation(
    rollouts: list[Rollout], *, require_correct: bool = True
) -> list[tuple[str, float, int]]:
    """Rank questions by mean clean completion tokens, with stable tie-breaking.

    Only valid, stopped clean rollouts with completion-token telemetry are used. When
    ``require_correct`` is true, every included clean rollout must match its gold answer.
    The returned tuples contain question ID, mean tokens, and contributing sample count.
    """
    grouped: dict[str, list[Rollout]] = defaultdict(list)
    for rollout in rollouts:
        completion_tokens = rollout.usage.get("completion_tokens")
        if (
            rollout.condition == Condition.CLEAN
            and rollout.finish_reason == "stop"
            and rollout.parsed_answer is not None
            and isinstance(completion_tokens, int)
            and completion_tokens > 0
        ):
            grouped[rollout.question_id].append(rollout)

    ranked: list[tuple[str, float, int]] = []
    for question_id, rows in grouped.items():
        if require_correct and any(row.parsed_answer != row.gold_answer for row in rows):
            continue
        mean_tokens = sum(row.usage["completion_tokens"] for row in rows) / len(rows)
        ranked.append((question_id, mean_tokens, len(rows)))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))
