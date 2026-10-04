"""Routing policies and their baselines.

A policy scores each example by how much it is worth sending to the large model.
Given a budget (the fraction of examples we may escalate), the top-scoring
examples are escalated. Sweeping the budget traces the cost-accuracy curve.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .features import FEATURE_NAMES


class RoutingPolicy(ABC):
    """Scores examples by expected benefit of escalating to the large model."""

    name: str

    @abstractmethod
    def scores(self, features: np.ndarray) -> np.ndarray:
        """Return one score per row; higher means escalate sooner."""


class ConfidenceThreshold(RoutingPolicy):
    """Standard cascade baseline: escalate where the small model is unsure."""

    name = "confidence"

    def __init__(self) -> None:
        self._column = FEATURE_NAMES.index("small_confidence")

    def scores(self, features: np.ndarray) -> np.ndarray:
        return -features[:, self._column]


class RandomEscalation(RoutingPolicy):
    """Control: escalate at random, so any gain must come from the signal."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def scores(self, features: np.ndarray) -> np.ndarray:
        return self._rng.random(features.shape[0])


class LearnedRouter(RoutingPolicy):
    """Logistic regression predicting "the large model fixes this example".

    Trained on the SummaC validation cut, evaluated on the test cut.
    """

    name = "learned"

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._model = None

    def fit(self, features: np.ndarray, improves: np.ndarray) -> "LearnedRouter":
        """Fit on a binary target: does escalation change a wrong answer to right?"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if len(np.unique(improves)) < 2:
            raise ValueError(
                "Training target has one class: the large model never helps (or "
                "always does) on this split, so routing cannot be learned."
            )
        self._model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=self._seed),
        ).fit(features, improves)
        return self

    def scores(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before scores()")
        return self._model.predict_proba(features)[:, 1]


def escalation_mask(scores: np.ndarray, budget: float) -> np.ndarray:
    """Boolean mask escalating the highest-scoring ``budget`` share of examples.

    Ties are broken by position, so the mask is deterministic.
    """
    if not 0.0 <= budget <= 1.0:
        raise ValueError(f"budget must be in [0, 1], got {budget}")
    n = len(scores)
    k = int(round(budget * n))
    mask = np.zeros(n, dtype=bool)
    if k:
        order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
        mask[order[:k]] = True
    return mask
