"""Frozen scalar and nuisance controls for causal-error monitoring."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class ReasoningLengthMonitor:
    """Class-balanced logistic control using only reasoning character count."""

    def __init__(self, seed: int = 17):
        """Initialize the reproducible scalar control."""
        self.model = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed
        )

    def fit(self, lengths: list[int], labels: list[int]) -> ReasoningLengthMonitor:
        """Fit on one non-negative length per example."""
        if any(length < 0 for length in lengths):
            raise ValueError("reasoning lengths must be non-negative")
        self.model.fit(np.asarray(lengths).reshape(-1, 1), np.asarray(labels))
        return self

    def predict_score(self, lengths: list[int]) -> np.ndarray:
        """Return positive-class probabilities."""
        return self.model.predict_proba(np.asarray(lengths).reshape(-1, 1))[:, 1]


class TemplateIdentityMonitor:
    """Nuisance control over family, tier, and renderer identity only."""

    def __init__(self, seed: int = 17):
        """Initialize one-hot encoding and class-balanced logistic regression."""
        categorical = OneHotEncoder(handle_unknown="ignore")
        self.model = Pipeline(
            [
                (
                    "encode",
                    ColumnTransformer(
                        [("categorical", categorical, [0, 1, 2])],
                        remainder="drop",
                    ),
                ),
                (
                    "classify",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )

    def fit(
        self, identities: list[tuple[str, str, str]], labels: list[int]
    ) -> TemplateIdentityMonitor:
        """Fit the nuisance-only control."""
        self.model.fit(np.asarray(identities, dtype=object), np.asarray(labels))
        return self

    def predict_score(self, identities: list[tuple[str, str, str]]) -> np.ndarray:
        """Return positive-class probabilities."""
        return self.model.predict_proba(np.asarray(identities, dtype=object))[:, 1]
