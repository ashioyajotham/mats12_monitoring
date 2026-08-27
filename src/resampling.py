"""Question-level counterfactual answer-shift evidence."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable

from pydantic import BaseModel

from src.generate_rollouts import Rollout
from src.hints import Condition


class AnswerShift(BaseModel):
    """Difference in hinted-option frequency between clean and hinted samples."""
    question_id: str
    hinted_option: str
    n_clean: int
    n_hinted: int
    p_clean_hint: float
    p_hinted_hint: float
    hint_effect: float
    standard_error: float
    lower_bound_95: float
    upper_bound_95: float


def _frequency(answers: list[str | None], option: str) -> float:
    """Calculate the empirical frequency of an option in parsed answers."""
    if not answers:
        raise ValueError("cannot estimate frequency from zero samples")
    return Counter(answers)[option] / len(answers)


def estimate_answer_shifts(rollouts: Iterable[Rollout]) -> list[AnswerShift]:
    """Estimate answer shifts for every question and hinted-option combination.

    Invalid generations with no parsed answer remain in raw data but are excluded from the
    probability denominators, matching the preregistered exclusion rule. Different hinted
    options for one question produce separate estimates rather than invalidating the group.
    """
    grouped: dict[str, list[Rollout]] = defaultdict(list)
    for rollout in rollouts:
        grouped[rollout.question_id].append(rollout)

    estimates: list[AnswerShift] = []
    for question_id, rows in grouped.items():
        hinted_rows = [
            r
            for r in rows
            if r.condition is Condition.INCORRECT_ANSWER_HINT and r.parsed_answer is not None
        ]
        clean_rows = [
            r for r in rows if r.condition is Condition.CLEAN and r.parsed_answer is not None
        ]
        hinted_options = {r.hinted_option for r in hinted_rows if r.hinted_option}
        if not clean_rows or not hinted_rows:
            continue
        for option in sorted(hinted_options):
            option_rows = [r for r in hinted_rows if r.hinted_option == option]
            p_clean = _frequency([r.parsed_answer for r in clean_rows], option)
            p_hinted = _frequency([r.parsed_answer for r in option_rows], option)
            variance = p_clean * (1 - p_clean) / len(clean_rows)
            variance += p_hinted * (1 - p_hinted) / len(option_rows)
            se = math.sqrt(variance)
            effect = p_hinted - p_clean
            estimates.append(
                AnswerShift(
                    question_id=question_id,
                    hinted_option=option,
                    n_clean=len(clean_rows),
                    n_hinted=len(option_rows),
                    p_clean_hint=p_clean,
                    p_hinted_hint=p_hinted,
                    hint_effect=effect,
                    standard_error=se,
                    lower_bound_95=max(-1.0, effect - 1.96 * se),
                    upper_bound_95=min(1.0, effect + 1.96 * se),
                )
            )
    return estimates
