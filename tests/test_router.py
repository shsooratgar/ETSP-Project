"""Routing policies and the cost-accuracy curve."""

from __future__ import annotations

import numpy as np
import pytest

from frugal_faith.evaluate import cost_accuracy_curve
from frugal_faith.features import FEATURE_NAMES
from frugal_faith.router import (
    ConfidenceThreshold,
    LearnedRouter,
    RandomEscalation,
    escalation_mask,
)


def _features_with_confidence(confidences: list[float]) -> np.ndarray:
    matrix = np.zeros((len(confidences), len(FEATURE_NAMES)))
    matrix[:, FEATURE_NAMES.index("small_confidence")] = confidences
    return matrix


def test_escalation_mask_respects_budget():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    mask = escalation_mask(scores, 0.5)
    assert mask.sum() == 2
    assert mask.tolist() == [False, True, False, True]  # the two highest scores


def test_escalation_mask_endpoints():
    scores = np.array([0.1, 0.9])
    assert escalation_mask(scores, 0.0).sum() == 0
    assert escalation_mask(scores, 1.0).all()


def test_escalation_mask_rejects_bad_budget():
    with pytest.raises(ValueError):
        escalation_mask(np.array([0.5]), 1.5)


def test_confidence_policy_escalates_the_least_confident_first():
    features = _features_with_confidence([0.99, 0.51, 0.80])
    mask = escalation_mask(ConfidenceThreshold().scores(features), 1 / 3)
    assert mask.tolist() == [False, True, False]


def test_random_policy_is_reproducible():
    features = _features_with_confidence([0.5] * 5)
    assert RandomEscalation(seed=1).scores(features).tolist() == pytest.approx(
        RandomEscalation(seed=1).scores(features).tolist()
    )


def test_learned_router_needs_fitting_first():
    with pytest.raises(RuntimeError):
        LearnedRouter().scores(_features_with_confidence([0.5]))


def test_learned_router_rejects_single_class_target():
    pytest.importorskip("sklearn")
    features = _features_with_confidence([0.5, 0.6, 0.7])
    with pytest.raises(ValueError, match="one class"):
        LearnedRouter().fit(features, np.zeros(3, dtype=int))


def test_learned_router_beats_random_when_signal_exists():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    n = 200
    confidences = rng.uniform(0.5, 1.0, size=n)
    features = _features_with_confidence(list(confidences))
    # The large model helps exactly where the small model was unsure.
    improves = (confidences < 0.7).astype(int)
    router = LearnedRouter(seed=0).fit(features, improves)
    top = escalation_mask(router.scores(features), 0.25)
    assert improves[top].mean() > improves.mean()


def test_cost_accuracy_curve_is_monotone_in_cost():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    small = np.array([1, 1, 1, 1, 1, 1])  # always says "consistent"
    large = y_true.copy()  # perfect
    features = _features_with_confidence([0.9, 0.8, 0.7, 0.6, 0.55, 0.51])
    curve = cost_accuracy_curve(
        ConfidenceThreshold(),
        features,
        small,
        large,
        y_true,
        budgets=(0.0, 0.5, 1.0),
        cost_small=1.0,
        cost_large=10.0,
    )
    assert curve["cost_per_example"].is_monotonic_increasing
    assert curve["balanced_accuracy"].iloc[-1] == pytest.approx(1.0)
