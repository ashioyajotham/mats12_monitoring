import math

import numpy as np
import pytest

from src.metrics import evaluate_scores, group_bootstrap_indices, precision_from_rates


def test_low_prevalence_precision_exposes_false_positive_cost():
    value = precision_from_rates(tpr=0.8, fpr=0.05, prevalence=0.01)
    assert value == pytest.approx(0.1391, abs=1e-3)


def test_zero_denominator_returns_nan():
    assert math.isnan(precision_from_rates(0.0, 0.0, 0.5))


def test_evaluate_scores_perfect_ranking():
    result = evaluate_scores([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert result.auroc == 1.0
    assert result.false_positive_rate == 0.0


def test_group_bootstrap_keeps_whole_groups():
    groups = ["q1", "q1", "q2", "q2", "q3"]
    samples = group_bootstrap_indices(groups, n_samples=3)
    for indices in samples:
        selected = np.asarray(groups)[indices]
        assert np.count_nonzero(selected == "q1") % 2 == 0
        assert np.count_nonzero(selected == "q2") % 2 == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tpr": -0.1, "fpr": 0.1, "prevalence": 0.1}, "tpr"),
        ({"tpr": 0.5, "fpr": 1.1, "prevalence": 0.1}, "fpr"),
        ({"tpr": 0.5, "fpr": 0.1, "prevalence": float("nan")}, "prevalence"),
    ],
)
def test_precision_rejects_invalid_rates(kwargs, message):
    with pytest.raises(ValueError, match=message):
        precision_from_rates(**kwargs)


def test_evaluate_scores_rejects_nonfinite_scores_and_recall():
    with pytest.raises(ValueError, match="finite"):
        evaluate_scores([0, 1], [0.1, float("nan")])
    with pytest.raises(ValueError, match="target_recall"):
        evaluate_scores([0, 1], [0.1, 0.9], fixed_recall=1.1)


def test_group_bootstrap_rejects_empty_or_zero_samples():
    with pytest.raises(ValueError, match="must not be empty"):
        group_bootstrap_indices([], n_samples=1)
    with pytest.raises(ValueError, match="must be positive"):
        group_bootstrap_indices(["q1"], n_samples=0)
