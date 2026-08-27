"""Cheap textual baseline that cannot access causal label fields."""

from __future__ import annotations

import re

import numpy as np
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression


class SurfaceFeatures(BaseModel):
    """Label-blind textual features available to the surface baseline."""
    character_count: int
    token_proxy_count: int
    uncertainty_count: int
    backtrack_count: int
    contradiction_count: int
    answer_mentions: int

    def vector(self) -> list[float]:
        """Return feature values in stable model-field order."""
        return [float(value) for value in self.model_dump().values()]


def extract_surface_features(text: str, final_answer: str | None) -> SurfaceFeatures:
    """Extract inexpensive lexical features without causal-label information."""
    lowered = text.lower()
    uncertainty = ("maybe", "perhaps", "uncertain", "likely", "not sure")
    backtrack = ("however", "actually", "reconsider", "on second thought", "wait")
    contradiction = ("but", "although", "yet", "contradict")
    answer_pattern = rf"\b{re.escape(final_answer)}\b" if final_answer else None
    answer_mentions = len(re.findall(answer_pattern, text)) if answer_pattern else 0
    return SurfaceFeatures(
        character_count=len(text),
        token_proxy_count=len(text.split()),
        uncertainty_count=sum(lowered.count(term) for term in uncertainty),
        backtrack_count=sum(lowered.count(term) for term in backtrack),
        contradiction_count=sum(lowered.count(term) for term in contradiction),
        answer_mentions=answer_mentions,
    )


class SurfaceFeatureMonitor:
    """Class-balanced logistic regression over surface features."""

    def __init__(self, seed: int = 17):
        """Initialize the baseline with a reproducible seed."""
        self.model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)

    def fit(self, features: list[SurfaceFeatures], labels: list[int]) -> SurfaceFeatureMonitor:
        """Fit the baseline to extracted features and binary labels."""
        self.model.fit(np.asarray([item.vector() for item in features]), np.asarray(labels))
        return self

    def predict_score(self, features: list[SurfaceFeatures]) -> np.ndarray:
        """Return positive-class probabilities for extracted features."""
        return self.model.predict_proba(np.asarray([item.vector() for item in features]))[:, 1]
