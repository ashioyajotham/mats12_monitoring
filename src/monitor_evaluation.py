"""Grouped component scoring, hybrid fitting, and monitor evaluation."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from pydantic import BaseModel, Field
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src.generate_rollouts import Rollout
from src.judge_runner import JudgeScoreRecord
from src.metrics import evaluate_scores, group_bootstrap_indices, threshold_for_recall
from src.monitor_dataset import MonitorExample
from src.monitors.answer_shift import score_counterfactual_answer_shifts
from src.monitors.controls import ReasoningLengthMonitor, TemplateIdentityMonitor
from src.monitors.hybrid import HybridMonitor
from src.monitors.surface import SurfaceFeatureMonitor, extract_surface_features


class ComponentScore(BaseModel):
    """One monitor score bound to one immutable example."""

    example_id: str
    question_id: str
    split: str
    component: str
    score: float = Field(ge=0.0, le=1.0)
    score_origin: str


def _surface_features(examples: list[MonitorExample]):
    """Extract surface inputs from the transcript-only evidence view."""
    return [
        extract_surface_features(row.transcript.reasoning, row.transcript.final_answer)
        for row in examples
    ]


def _identities(examples: list[MonitorExample]) -> list[tuple[str, str, str]]:
    """Return nuisance-only categorical identities."""
    return [(row.family, row.tier, str(row.renderer_id)) for row in examples]


def _validate_primary_splits(examples: list[MonitorExample]) -> None:
    """Require both classes and disjoint question groups in every frozen split."""
    question_splits: dict[str, set[str]] = defaultdict(set)
    for row in examples:
        question_splits[row.question_id].add(row.split)
    if any(len(splits) != 1 for splits in question_splits.values()):
        raise ValueError("question siblings cross frozen monitor splits")
    for split in ("train", "validation", "test"):
        if {row.binary_label for row in examples if row.split == split} != {0, 1}:
            raise ValueError(f"primary {split} split must contain both classes")


def _surface_oof_scores(
    train: list[MonitorExample], *, folds: int, seed: int
) -> dict[str, float]:
    """Create genuine question-group out-of-fold surface scores."""
    if folds != 5:
        raise ValueError("causal-error-v1 freezes five cross-fitting folds")
    labels = np.asarray([row.binary_label for row in train])
    groups = np.asarray([row.question_id for row in train])
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = np.full(len(train), np.nan)
    features = _surface_features(train)
    for fit_indices, heldout_indices in splitter.split(np.zeros(len(train)), labels, groups):
        fit_labels = labels[fit_indices]
        if set(np.unique(fit_labels)) != {0, 1}:
            raise ValueError("a surface cross-fitting fold lacks one class")
        monitor = SurfaceFeatureMonitor(seed=seed).fit(
            [features[index] for index in fit_indices], fit_labels.tolist()
        )
        scores[heldout_indices] = monitor.predict_score(
            [features[index] for index in heldout_indices]
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("surface cross-fitting did not score every train example")
    return {row.example_id: float(scores[index]) for index, row in enumerate(train)}


def build_local_component_scores(
    primary: list[MonitorExample],
    secondary_audit: list[MonitorExample],
    rollouts: list[Rollout],
    *,
    seed: int = 20262630,
) -> list[ComponentScore]:
    """Fit local controls on train and score primary plus secondary audit examples."""
    _validate_primary_splits(primary)
    train = [row for row in primary if row.split == "train"]
    all_examples = [*primary, *secondary_audit]
    labels = [row.binary_label for row in train]
    surface_oof = _surface_oof_scores(train, folds=5, seed=seed)

    surface = SurfaceFeatureMonitor(seed=seed).fit(_surface_features(train), labels)
    shuffled_labels = np.random.default_rng(seed + 1).permutation(labels).tolist()
    shuffled_surface = SurfaceFeatureMonitor(seed=seed + 1).fit(
        _surface_features(train), shuffled_labels
    )
    length = ReasoningLengthMonitor(seed=seed).fit(
        [len(row.transcript.reasoning) for row in train], labels
    )
    template = TemplateIdentityMonitor(seed=seed).fit(_identities(train), labels)
    surface_all = surface.predict_score(_surface_features(all_examples))
    shuffled_surface_all = shuffled_surface.predict_score(_surface_features(all_examples))
    length_all = length.predict_score(
        [len(row.transcript.reasoning) for row in all_examples]
    )
    template_all = template.predict_score(_identities(all_examples))
    answer_shift = {
        row.rollout_id: row.score
        for row in score_counterfactual_answer_shifts(
            [example.rollout_id for example in all_examples], rollouts
        )
    }

    output: list[ComponentScore] = []
    for index, example in enumerate(all_examples):
        fixed = (
            ("correctness_only", 1.0),
            ("reasoning_length", float(length_all[index])),
            ("template_identity", float(template_all[index])),
            ("shuffled_label_surface", float(shuffled_surface_all[index])),
            ("counterfactual_answer_shift", answer_shift[example.rollout_id]),
        )
        for component, score in fixed:
            output.append(
                ComponentScore(
                    example_id=example.example_id,
                    question_id=example.question_id,
                    split=example.split,
                    component=component,
                    score=score,
                    score_origin=(
                        "constant_known_incorrect_control"
                        if component == "correctness_only"
                        else "full_train_model"
                        if component
                        in {
                            "reasoning_length",
                            "template_identity",
                            "shuffled_label_surface",
                        }
                        else "fixed_label_free"
                    ),
                )
            )
        surface_score = (
            surface_oof[example.example_id]
            if example.example_id in surface_oof
            else float(surface_all[index])
        )
        output.append(
            ComponentScore(
                example_id=example.example_id,
                question_id=example.question_id,
                split=example.split,
                component="surface",
                score=surface_score,
                score_origin=(
                    "five_fold_question_oof"
                    if example.example_id in surface_oof
                    else "full_train_model"
                ),
            )
        )
    return output


def build_hybrid_scores(
    primary: list[MonitorExample],
    local_scores: list[ComponentScore],
    judge_scores: list[JudgeScoreRecord],
    *,
    seed: int = 20262631,
) -> list[ComponentScore]:
    """Fit the frozen hybrid on OOF train components and score validation/test."""
    _validate_primary_splits(primary)
    local = {(row.example_id, row.component): row for row in local_scores}
    judge = {(row.example_id, row.view): row for row in judge_scores}
    components = (
        "surface",
        "transcript_only",
        "context_aware",
        "counterfactual_answer_shift",
    )

    def vector(example: MonitorExample) -> list[float]:
        surface = local.get((example.example_id, "surface"))
        shift = local.get((example.example_id, "counterfactual_answer_shift"))
        transcript = judge.get((example.example_id, "transcript_only"))
        context = judge.get((example.example_id, "context_aware"))
        if not all((surface, shift, transcript, context)):
            raise ValueError(f"example {example.example_id} lacks a hybrid component score")
        return [surface.score, transcript.score, context.score, shift.score]  # type: ignore[union-attr]

    train = [row for row in primary if row.split == "train"]
    for row in train:
        surface = local[(row.example_id, "surface")]
        if surface.score_origin != "five_fold_question_oof":
            raise ValueError("hybrid train surface scores must be genuinely out-of-fold")
    model = HybridMonitor(seed=seed).fit(
        np.asarray([vector(row) for row in train]),
        np.asarray([row.binary_label for row in train]),
    )
    output: list[ComponentScore] = []
    for split in ("validation", "test"):
        rows = [row for row in primary if row.split == split]
        values = model.predict_score(np.asarray([vector(row) for row in rows]))
        output.extend(
            ComponentScore(
                example_id=row.example_id,
                question_id=row.question_id,
                split=row.split,
                component="hybrid",
                score=float(values[index]),
                score_origin="five_fold_oof_component_fusion",
            )
            for index, row in enumerate(rows)
        )
    if components != (
        "surface",
        "transcript_only",
        "context_aware",
        "counterfactual_answer_shift",
    ):
        raise AssertionError("hybrid component order changed")
    return output


def evaluate_component(
    examples: list[MonitorExample],
    scores: list[ComponentScore],
    *,
    component: str,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 20262632,
) -> dict[str, object]:
    """Evaluate one component on frozen test groups with clustered intervals."""
    test = [row for row in examples if row.split == "test"]
    score_by_id = {
        row.example_id: row.score for row in scores if row.component == component
    }
    if set(score_by_id) & {row.example_id for row in test} != {
        row.example_id for row in test
    }:
        raise ValueError(f"component {component} lacks test scores")
    labels = [row.binary_label for row in test]
    values = [score_by_id[row.example_id] for row in test]
    groups = [row.question_id for row in test]
    point = evaluate_scores(labels, values)
    bootstraps: list[dict[str, float]] = []
    for indices in group_bootstrap_indices(
        groups, n_samples=bootstrap_samples, seed=bootstrap_seed
    ):
        sampled_labels = np.asarray(labels)[indices]
        if set(np.unique(sampled_labels)) != {0, 1}:
            continue
        sampled_scores = np.asarray(values)[indices]
        metric = evaluate_scores(sampled_labels, sampled_scores)
        bootstraps.append(
            {
                "auroc": metric.auroc,
                "auprc": metric.auprc,
                "false_positive_rate": metric.false_positive_rate,
            }
        )

    def interval(name: str) -> dict[str, float | None]:
        values_for_metric = sorted(row[name] for row in bootstraps)
        if not values_for_metric:
            return {"low": None, "high": None}
        return {
            "low": values_for_metric[int(0.025 * (len(values_for_metric) - 1))],
            "high": values_for_metric[int(0.975 * (len(values_for_metric) - 1))],
        }

    family_results: dict[str, object] = {}
    for family in sorted({row.family for row in test}):
        rows = [row for row in test if row.family == family]
        if {row.binary_label for row in rows} != {0, 1}:
            family_results[family] = {"available": False, "reason": "single_class"}
            continue
        metric = evaluate_scores(
            [row.binary_label for row in rows],
            [score_by_id[row.example_id] for row in rows],
        )
        family_results[family] = {"available": True, **metric.model_dump()}
    return {
        "component": component,
        "point": point.model_dump(),
        "cluster_bootstrap": {
            "requested_samples": bootstrap_samples,
            "successful_samples": len(bootstraps),
            "seed": bootstrap_seed,
            "auroc": interval("auroc"),
            "auprc": interval("auprc"),
            "false_positive_rate_at_50_recall": interval("false_positive_rate"),
        },
        "by_family": family_results,
    }


def validation_operating_point(
    examples: list[MonitorExample],
    scores: list[ComponentScore],
    *,
    component: str,
    target_recall: float = 0.5,
) -> dict[str, float]:
    """Choose a threshold on validation data and apply it once to the frozen test set."""
    score_by_id = {
        row.example_id: row.score for row in scores if row.component == component
    }
    partitions = {
        split: [row for row in examples if row.split == split]
        for split in ("validation", "test")
    }
    for split, rows in partitions.items():
        if {row.binary_label for row in rows} != {0, 1}:
            raise ValueError(f"primary {split} split must contain both classes")
        if any(row.example_id not in score_by_id for row in rows):
            raise ValueError(f"component {component} lacks {split} scores")
    validation = partitions["validation"]
    threshold, validation_recall, validation_fpr = threshold_for_recall(
        np.asarray([row.binary_label for row in validation]),
        np.asarray([score_by_id[row.example_id] for row in validation]),
        target_recall,
    )
    test = partitions["test"]
    test_labels = np.asarray([row.binary_label for row in test])
    test_predictions = np.asarray(
        [score_by_id[row.example_id] >= threshold for row in test]
    )
    positives = test_labels == 1
    negatives = test_labels == 0
    return {
        "target_recall": target_recall,
        "threshold_selected_on_validation": threshold,
        "validation_recall": validation_recall,
        "validation_false_positive_rate": validation_fpr,
        "test_recall": float(test_predictions[positives].mean()),
        "test_false_positive_rate": float(test_predictions[negatives].mean()),
    }


def summarize_secondary_audit(
    examples: list[MonitorExample], scores: list[ComponentScore], *, component: str
) -> dict[str, object]:
    """Summarize hard-negative audit scores descriptively, never as primary metrics."""
    score_by_id = {
        row.example_id: row.score for row in scores if row.component == component
    }
    if any(row.example_id not in score_by_id for row in examples):
        raise ValueError(f"component {component} lacks secondary-audit scores")
    output: dict[str, object] = {}
    strata = {
        "overall": examples,
        **{
            f"kind:{kind}": [row for row in examples if str(row.kind) == kind]
            for kind in sorted({str(row.kind) for row in examples})
        },
        **{
            f"family:{family}": [row for row in examples if row.family == family]
            for family in sorted({row.family for row in examples})
        },
    }
    for name, rows in strata.items():
        values = np.asarray([score_by_id[row.example_id] for row in rows])
        output[name] = {
            "n": len(rows),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "q10": float(np.quantile(values, 0.1)),
            "q90": float(np.quantile(values, 0.9)),
        }
    return output


def paired_component_comparison(
    examples: list[MonitorExample],
    scores: list[ComponentScore],
    *,
    first: str,
    second: str,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 20262633,
) -> dict[str, object]:
    """Estimate paired test-metric differences with question-cluster resampling."""
    test = [row for row in examples if row.split == "test"]
    score_maps = {
        component: {
            row.example_id: row.score for row in scores if row.component == component
        }
        for component in (first, second)
    }
    if any(
        row.example_id not in score_maps[component]
        for component in (first, second)
        for row in test
    ):
        raise ValueError("paired comparison lacks test scores")
    labels = np.asarray([row.binary_label for row in test])
    values = {
        component: np.asarray([score_maps[component][row.example_id] for row in test])
        for component in (first, second)
    }

    def differences(indices: np.ndarray) -> dict[str, float]:
        metrics = {
            component: evaluate_scores(labels[indices], values[component][indices])
            for component in (first, second)
        }
        return {
            "auroc": metrics[first].auroc - metrics[second].auroc,
            "auprc": metrics[first].auprc - metrics[second].auprc,
            "false_positive_rate_at_50_recall": (
                metrics[first].false_positive_rate - metrics[second].false_positive_rate
            ),
        }

    point = differences(np.arange(len(test)))
    bootstrap: list[dict[str, float]] = []
    for indices in group_bootstrap_indices(
        [row.question_id for row in test],
        n_samples=bootstrap_samples,
        seed=bootstrap_seed,
    ):
        if set(np.unique(labels[indices])) == {0, 1}:
            bootstrap.append(differences(indices))

    intervals: dict[str, object] = {}
    for metric in point:
        sampled = np.asarray([row[metric] for row in bootstrap])
        intervals[metric] = {
            "point": point[metric],
            "low": float(np.quantile(sampled, 0.025)) if len(sampled) else None,
            "high": float(np.quantile(sampled, 0.975)) if len(sampled) else None,
        }
    return {
        "contrast": f"{first}_minus_{second}",
        "requested_samples": bootstrap_samples,
        "successful_samples": len(bootstrap),
        "seed": bootstrap_seed,
        "differences": intervals,
    }


def invalid_rollout_split_diagnostic(
    examples: list[MonitorExample], *, seed: int = 20262634
) -> dict[str, object]:
    """Quantify the deliberately invalid result from splitting sibling rollouts directly."""
    indices = np.arange(len(examples))
    labels = np.asarray([row.binary_label for row in examples])
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    features = _surface_features(examples)
    model = SurfaceFeatureMonitor(seed=seed).fit(
        [features[index] for index in train_indices], labels[train_indices].tolist()
    )
    scores = model.predict_score([features[index] for index in test_indices])
    train_groups = {examples[index].question_id for index in train_indices}
    test_groups = {examples[index].question_id for index in test_indices}
    return {
        "status": "invalid_leakage_diagnostic_not_research_evidence",
        "seed": seed,
        "train_examples": len(train_indices),
        "test_examples": len(test_indices),
        "question_groups_crossing_split": len(train_groups & test_groups),
        "metrics": evaluate_scores(labels[test_indices], scores).model_dump(),
    }
