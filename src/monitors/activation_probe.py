"""Optional activation probe contract; deliberately downstream of behavioural gates."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class ActivationProbe:
    """Small linear probe with no architecture-specific circuit assumptions."""

    def __init__(self, seed: int = 17):
        """Initialize a class-balanced logistic probe with a reproducible seed."""
        self.model = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)

    def fit(self, activations: np.ndarray, labels: np.ndarray) -> ActivationProbe:
        """Fit the probe on a two-dimensional activation matrix."""
        if activations.ndim != 2:
            raise ValueError("activations must have shape [examples, hidden_size]")
        self.model.fit(activations, labels)
        return self

    def predict_score(self, activations: np.ndarray) -> np.ndarray:
        """Return positive-class probabilities for activation rows."""
        return self.model.predict_proba(activations)[:, 1]


def shuffled_control(labels: np.ndarray, seed: int = 17) -> np.ndarray:
    """Permute labels reproducibly for a selectivity control."""
    return np.random.default_rng(seed).permutation(labels)
