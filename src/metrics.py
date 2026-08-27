"""Metrics appropriate for rare positive events."""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


class PrevalenceMetric(BaseModel):
    """Precision implied by monitor rates at one assumed deployment prevalence."""
    prevalence: float
    precision: float


class MonitorMetrics(BaseModel):
    """Balanced discrimination and low-base-rate operating metrics."""
    n: int
    positives: int
    auroc: float
    auprc: float
    threshold_at_fixed_recall: float
    achieved_recall: float
    false_positive_rate: float
    precision_by_prevalence: list[PrevalenceMetric]


def precision_from_rates(tpr: float, fpr: float, prevalence: float) -> float:
    """Calculate positive predictive value from rates and class prevalence."""
    if not math.isfinite(tpr) or not 0 <= tpr <= 1:
        raise ValueError("tpr must be finite and in [0, 1]")
    if not math.isfinite(fpr) or not 0 <= fpr <= 1:
        raise ValueError("fpr must be finite and in [0, 1]")
    if not math.isfinite(prevalence) or not 0 <= prevalence <= 1:
        raise ValueError("prevalence must be finite and in [0, 1]")
    numerator = tpr * prevalence
    denominator = numerator + fpr * (1 - prevalence)
    if denominator == 0:
        return math.nan
    return numerator / denominator


def threshold_for_recall(
    labels: np.ndarray, scores: np.ndarray, target_recall: float
) -> tuple[float, float, float]:
    """Choose the highest threshold with minimum FPR that reaches target recall."""
    if not math.isfinite(target_recall) or not 0 <= target_recall <= 1:
        raise ValueError("target_recall must be finite and in [0, 1]")
    fpr, tpr, thresholds = roc_curve(labels, scores)
    candidates = np.flatnonzero(tpr >= target_recall)
    if not len(candidates):
        raise ValueError("target recall is unattainable")
    # Among thresholds reaching recall, choose the lowest FPR; break ties by higher threshold.
    best_fpr = np.min(fpr[candidates])
    tied = candidates[fpr[candidates] == best_fpr]
    index = tied[np.argmax(thresholds[tied])]
    return float(thresholds[index]), float(tpr[index]), float(fpr[index])


def evaluate_scores(
    labels: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
    *,
    fixed_recall: float = 0.5,
    prevalences: tuple[float, ...] = (0.5, 0.1, 0.05, 0.01),
) -> MonitorMetrics:
    """Evaluate binary monitor scores at balanced and assumed deployment prevalences."""
    y_true = np.asarray(labels, dtype=int)
    y_score = np.asarray(scores, dtype=float)
    if y_true.shape != y_score.shape or y_true.ndim != 1:
        raise ValueError("labels and scores must be equally sized one-dimensional arrays")
    if not np.all(np.isfinite(y_score)):
        raise ValueError("scores must contain only finite values")
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("labels must contain both binary classes")
    threshold, recall, fpr = threshold_for_recall(y_true, y_score, fixed_recall)
    return MonitorMetrics(
        n=len(y_true),
        positives=int(y_true.sum()),
        auroc=float(roc_auc_score(y_true, y_score)),
        auprc=float(average_precision_score(y_true, y_score)),
        threshold_at_fixed_recall=threshold,
        achieved_recall=recall,
        false_positive_rate=fpr,
        precision_by_prevalence=[
            PrevalenceMetric(prevalence=p, precision=precision_from_rates(recall, fpr, p))
            for p in prevalences
        ],
    )


def group_bootstrap_indices(groups: list[str], n_samples: int, seed: int = 17) -> list[np.ndarray]:
    """Sample question groups, retaining every row within selected groups."""
    if not groups:
        raise ValueError("groups must not be empty")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    unique = np.asarray(sorted(set(groups)))
    group_array = np.asarray(groups)
    rng = np.random.default_rng(seed)
    outputs: list[np.ndarray] = []
    for _ in range(n_samples):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        parts = [np.flatnonzero(group_array == group) for group in sampled]
        outputs.append(np.concatenate(parts))
    return outputs
