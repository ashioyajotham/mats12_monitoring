"""Run preregistered, zero-credit robustness analyses on causal-error-v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.generate_rollouts import Rollout
from src.hints import Condition
from src.metrics import evaluate_scores
from src.monitor_dataset import MonitorExample
from src.monitors.answer_shift import score_counterfactual_answer_shifts
from src.tasks import read_jsonl


def _metric(examples: list[MonitorExample], scores: dict[str, float]) -> dict[str, object]:
    """Evaluate aligned scores on the frozen v1 test partition."""
    return evaluate_scores(
        [row.binary_label for row in examples],
        [scores[row.rollout_id] for row in examples],
    ).model_dump()


def main() -> None:
    """Measure placebo behavior, sibling cost, and score-component dependence."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--primary",
        type=Path,
        default=Path("data/reviewed/causal_error_v1.primary.jsonl"),
    )
    parser.add_argument(
        "--rollouts",
        type=Path,
        default=Path(
            "data/generated/tinker_causal_error_v1_confirmatory_20260830T194441Z/rollouts.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/causal_error_v1.answer_shift_robustness.json"),
    )
    args = parser.parse_args()
    primary = list(read_jsonl(args.primary, model=MonitorExample))
    test = [row for row in primary if row.split == "test"]
    rollouts = list(read_jsonl(args.rollouts, model=Rollout))
    focal_ids = [row.rollout_id for row in test]

    variants = {
        "all_available": score_counterfactual_answer_shifts(focal_ids, rollouts),
        "one_sibling_per_condition": score_counterfactual_answer_shifts(
            focal_ids, rollouts, siblings_per_condition=1
        ),
        "two_siblings_per_condition": score_counterfactual_answer_shifts(
            focal_ids, rollouts, siblings_per_condition=2
        ),
    }
    metrics = {
        name: _metric(test, {row.rollout_id: row.score for row in rows})
        for name, rows in variants.items()
    }
    full = {row.rollout_id: row for row in variants["all_available"]}
    decomposition = {
        "corrupted_recurrence_only": _metric(
            test, {key: row.corrupted_state_frequency for key, row in full.items()}
        ),
        "control_suppression_only": _metric(
            test,
            {
                key: 1.0 - max(row.clean_frequency, row.correct_state_frequency)
                for key, row in full.items()
            },
        ),
    }

    swapped = [
        row.model_copy(
            update={
                "condition": (
                    Condition.CORRECT_CONTINUATION
                    if row.condition is Condition.CORRUPTED_CONTINUATION
                    else Condition.CORRUPTED_CONTINUATION
                    if row.condition is Condition.CORRECT_CONTINUATION
                    else row.condition
                )
            }
        )
        for row in rollouts
    ]
    placebo_rows = score_counterfactual_answer_shifts(focal_ids, swapped)
    placebo = _metric(test, {row.rollout_id: row.score for row in placebo_rows})
    full_auroc = float(metrics["all_available"]["auroc"])
    candidate_assessments = {}
    for name in ("one_sibling_per_condition", "two_siblings_per_condition"):
        reduced_auroc = float(metrics[name]["auroc"])
        candidate_assessments[name] = {
            "retains_at_least_90_percent_full_auroc": reduced_auroc >= 0.90 * full_auroc,
            "absolute_auroc_loss_at_most_0_05": full_auroc - reduced_auroc <= 0.05,
            "qualifies": reduced_auroc >= 0.90 * full_auroc and full_auroc - reduced_auroc <= 0.05,
        }
    report = {
        "protocol": "causal-error-v1-answer-shift-robustness",
        "partition": "frozen_test_only",
        "n": len(test),
        "sibling_sensitivity": metrics,
        "score_decomposition": decomposition,
        "condition_identity_swap_placebo": placebo,
        "cheaper_sibling_candidates": {
            **candidate_assessments,
            "cheapest_qualifying": (
                "one_sibling_per_condition"
                if candidate_assessments["one_sibling_per_condition"]["qualifies"]
                else "two_siblings_per_condition"
                if candidate_assessments["two_siblings_per_condition"]["qualifies"]
                else None
            ),
            "status": "exploratory_not_confirmatory",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
