"""Metrics and prediction combination."""

from __future__ import annotations

import numpy as np
import pytest

from frugal_faith.evaluate import balanced_accuracy, combine, reference_points


def test_balanced_accuracy_is_not_fooled_by_class_imbalance():
    # 9 consistent, 1 inconsistent; always predicting "consistent" is 90% accurate
    # but only 50% balanced.
    y_true = np.array([1] * 9 + [0])
    y_pred = np.ones(10, dtype=int)
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(0.5)


def test_balanced_accuracy_perfect_and_inverted():
    y_true = np.array([0, 1, 0, 1])
    assert balanced_accuracy(y_true, y_true) == pytest.approx(1.0)
    assert balanced_accuracy(y_true, 1 - y_true) == pytest.approx(0.0)


def test_balanced_accuracy_requires_both_classes():
    with pytest.raises(ValueError):
        balanced_accuracy(np.array([]), np.array([]))


def test_combine_takes_large_only_where_escalated():
    small = np.array([0, 0, 0])
    large = np.array([1, 1, 1])
    escalate = np.array([True, False, True])
    assert combine(small, large, escalate).tolist() == [1, 0, 1]


def test_reference_points_bound_the_curve():
    y_true = np.array([0, 1, 0, 1])
    frame = reference_points(
        small_pred=np.array([1, 1, 1, 1]),
        large_pred=y_true,
        y_true=y_true,
        cost_small=1.0,
        cost_large=10.0,
    )
    assert frame.set_index("policy").loc["always_large", "balanced_accuracy"] == 1.0
    assert frame.set_index("policy").loc["always_small", "cost_per_example"] == 1.0
