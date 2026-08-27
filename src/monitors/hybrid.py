"""Simple fusion model used only with out-of-fold component scores."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class HybridMonitor:
    """Logistic fusion of out-of-fold component monitor scores."""

    def __init__(self, seed: int = 17):
        """Initialize a class-balanced fusion model with a reproducible seed."""
        self.model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)

    def fit(self, component_scores: np.ndarray, labels: np.ndarray) -> HybridMonitor:
        """Fit on a matrix containing one out-of-fold score per component."""
        if component_scores.ndim != 2:
            raise ValueError("component_scores must have shape [examples, monitors]")
        self.model.fit(component_scores, labels)
        return self

    def predict_score(self, component_scores: np.ndarray) -> np.ndarray:
        """Return positive-class probabilities for component-score rows."""
        return self.model.predict_proba(component_scores)[:, 1]
