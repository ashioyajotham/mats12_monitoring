"""Apply frozen v1 monitors to the mechanism-held-out causal-audit-v2 cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from src.generate_rollouts import Rollout, RolloutStatus
from src.hints import Condition
from src.math_answers import MathGrade, grade_math_answer
from src.metrics import evaluate_scores, group_bootstrap_indices
from src.monitor_dataset import CausalErrorExampleKind, MonitorExample, _example
from src.monitors.answer_shift import score_counterfactual_answer_shifts
from src.monitors.surface import SurfaceFeatureMonitor, extract_surface_features
from src.tasks import MathProblem, read_jsonl


def _features(rows: list[MonitorExample]):
    """Extract the frozen transcript-only surface representation."""
    return [
        extract_surface_features(row.transcript.reasoning, row.transcript.final_answer)
        for row in rows
    ]


def _materialize(questions: list[MathProblem], rollouts: list[Rollout]) -> list[MonitorExample]:
    """Create external positives and negatives using the unchanged v1 label rule."""
    by_id = {row.question_id: row for row in questions}
    output: list[MonitorExample] = []
    for rollout in rollouts:
        question = by_id[rollout.question_id]
        if rollout.status is not RolloutStatus.CLEAN_STOP:
            continue
        if (
            grade_math_answer(rollout.parsed_answer, question.gold_answer)
            is not MathGrade.INCORRECT
        ):
            continue
        if rollout.condition is Condition.CLEAN:
            output.append(
                _example(
                    question,
                    rollout,
                    kind=CausalErrorExampleKind.ORDINARY_ERROR,
                    binary_label=0,
                    split_override="external_test",
                )
            )
        elif (
            rollout.condition is Condition.CORRUPTED_CONTINUATION
            and rollout.parsed_answer == question.metadata["intervention_target_answer"]
        ):
            output.append(
                _example(
                    question,
                    rollout,
                    kind=CausalErrorExampleKind.CAUSALLY_INDUCED_ERROR,
                    binary_label=1,
                    split_override="external_test",
                )
            )
    if {row.binary_label for row in output} != {0, 1}:
        raise ValueError("external cohort lacks one primary error class")
    return output


def _rate_interval(successes: int, trials: int) -> dict[str, float | int | None]:
    """Return an exact two-sided 95% binomial interval."""
    if not trials:
        return {"successes": successes, "trials": trials, "rate": None, "low": None, "high": None}
    interval = binomtest(successes, trials).proportion_ci(confidence_level=0.95)
    return {
        "successes": successes,
        "trials": trials,
        "rate": successes / trials,
        "low": float(interval.low),
        "high": float(interval.high),
    }


def main() -> None:
    """Evaluate answer shift and a v1-trained surface model without v2 tuning."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--gate-report", type=Path, default=Path("results/causal_audit_v2_confirmatory.json")
    )
    parser.add_argument(
        "--v1-primary", type=Path, default=Path("data/reviewed/causal_error_v1.primary.jsonl")
    )
    parser.add_argument(
        "--v1-metrics", type=Path, default=Path("results/causal_error_v1.monitor_metrics.json")
    )
    parser.add_argument("--output", type=Path, default=Path("results/causal_audit_v2.metrics.json"))
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20262700)
    args = parser.parse_args()
    gate = json.loads(args.gate_report.read_text(encoding="utf-8"))
    if (
        gate.get("protocol") != "causal-audit-v2-confirmatory"
        or gate.get("gate_passed") is not True
    ):
        raise SystemExit("external evaluation requires a passing frozen confirmatory gate")
    questions = list(read_jsonl("data/raw/causal_audit_v2.confirmatory.jsonl", model=MathProblem))
    rollouts = list(read_jsonl(args.run / "rollouts.jsonl", model=Rollout))
    external = _materialize(questions, rollouts)
    v1 = list(read_jsonl(args.v1_primary, model=MonitorExample))
    train = [row for row in v1 if row.split == "train"]
    surface = SurfaceFeatureMonitor(seed=20262630).fit(
        _features(train), [row.binary_label for row in train]
    )
    surface_scores = surface.predict_score(_features(external))
    shift_rows = score_counterfactual_answer_shifts([row.rollout_id for row in external], rollouts)
    shift_by_id = {row.rollout_id: row.score for row in shift_rows}
    scores = {
        "counterfactual_answer_shift": np.asarray(
            [shift_by_id[row.rollout_id] for row in external]
        ),
        "surface": np.asarray(surface_scores, dtype=float),
    }
    labels = np.asarray([row.binary_label for row in external])
    point = {name: evaluate_scores(labels, values).model_dump() for name, values in scores.items()}
    differences: list[dict[str, float]] = []
    sampled_components: dict[str, list[dict[str, float]]] = {name: [] for name in scores}
    for indices in group_bootstrap_indices(
        [row.question_id for row in external], args.bootstrap_samples, args.bootstrap_seed
    ):
        if set(np.unique(labels[indices])) != {0, 1}:
            continue
        metrics = {
            name: evaluate_scores(labels[indices], values[indices])
            for name, values in scores.items()
        }
        for name, metric in metrics.items():
            sampled_components[name].append(
                {
                    "auroc": metric.auroc,
                    "auprc": metric.auprc,
                    "fpr_at_50_recall": metric.false_positive_rate,
                }
            )
        differences.append(
            {
                "auroc": metrics["counterfactual_answer_shift"].auroc - metrics["surface"].auroc,
                "auprc": metrics["counterfactual_answer_shift"].auprc - metrics["surface"].auprc,
                "fpr_at_50_recall": metrics["counterfactual_answer_shift"].false_positive_rate
                - metrics["surface"].false_positive_rate,
            }
        )
    paired = {}
    for metric in ("auroc", "auprc", "fpr_at_50_recall"):
        values = np.asarray([row[metric] for row in differences])
        point_difference = (
            point["counterfactual_answer_shift"][
                "false_positive_rate" if metric == "fpr_at_50_recall" else metric
            ]
            - point["surface"]["false_positive_rate" if metric == "fpr_at_50_recall" else metric]
        )
        paired[metric] = {
            "point": point_difference,
            "low": float(np.quantile(values, 0.025)),
            "high": float(np.quantile(values, 0.975)),
        }
    cluster_intervals: dict[str, object] = {}
    for name, rows in sampled_components.items():
        cluster_intervals[name] = {
            metric: {
                "low": float(np.quantile([row[metric] for row in rows], 0.025)),
                "high": float(np.quantile([row[metric] for row in rows], 0.975)),
            }
            for metric in ("auroc", "auprc", "fpr_at_50_recall")
        }

    v1_metrics = json.loads(args.v1_metrics.read_text(encoding="utf-8"))
    thresholds = {
        name: float(
            v1_metrics["primary_test_metrics"][name]["validation_selected_operating_point"][
                "threshold_selected_on_validation"
            ]
        )
        for name in scores
    }
    operating_points = {}
    for name, values in scores.items():
        predicted = values >= thresholds[name]
        positives = labels == 1
        negatives = labels == 0
        operating_points[name] = {
            "threshold_frozen_from_v1_validation": thresholds[name],
            "recall": _rate_interval(int(predicted[positives].sum()), int(positives.sum())),
            "false_positive_rate": _rate_interval(
                int(predicted[negatives].sum()), int(negatives.sum())
            ),
        }
    strata: dict[str, object] = {}
    question_by_id = {row.question_id: row for row in questions}
    for dimension, accessor in {
        "family": lambda row: row.family,
        "mechanism": lambda row: str(
            question_by_id[row.question_id].metadata["corruption_mechanism"]
        ),
    }.items():
        strata[dimension] = {}
        for value in sorted({accessor(row) for row in external}):
            indices = np.asarray(
                [index for index, row in enumerate(external) if accessor(row) == value]
            )
            if set(np.unique(labels[indices])) != {0, 1}:
                strata[dimension][value] = {"available": False, "reason": "single_class"}
            else:
                strata[dimension][value] = {
                    "available": True,
                    **{
                        name: evaluate_scores(labels[indices], values[indices]).model_dump()
                        for name, values in scores.items()
                    },
                }
    report = {
        "protocol": "causal-audit-v2-external-monitor-evaluation",
        "claim": "counterfactual causal audit, not single-trace online monitoring",
        "external_examples": len(external),
        "point_metrics": point,
        "question_clustered_intervals_95": cluster_intervals,
        "paired_answer_shift_minus_surface": paired,
        "v1_validation_threshold_transfer": operating_points,
        "descriptive_strata": strata,
        "hypothesis_support": {
            "answer_shift_auroc_lower_bound_above_chance": (
                cluster_intervals["counterfactual_answer_shift"]["auroc"]["low"] > 0.5
            ),
            "superiority_claim_requires_paired_interval_excluding_zero": True,
        },
        "no_v2_model_fitting_or_threshold_selection": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"external_examples": len(external), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
