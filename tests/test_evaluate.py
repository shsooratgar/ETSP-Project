"""Metrics and prediction combination."""

from __future__ import annotations

import numpy as np
import pytest

from frugal_faith.evaluate import (
    balanced_accuracy,
    bootstrap_ci,
    combine,
    paired_bootstrap_test,
    reference_points,
)


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


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 300)
    y_pred = np.where(rng.random(300) < 0.85, y_true, 1 - y_true)
    point = balanced_accuracy(y_true, y_pred)
    ci_low, ci_high = bootstrap_ci(y_true, y_pred, n_resamples=300, seed=0)
    assert ci_low < point < ci_high
    assert 0.0 <= ci_low <= ci_high <= 1.0


def test_bootstrap_ci_is_reproducible():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, 100)
    y_pred = rng.integers(0, 2, 100)
    first = bootstrap_ci(y_true, y_pred, n_resamples=200, seed=7)
    assert first == bootstrap_ci(y_true, y_pred, n_resamples=200, seed=7)


def test_bootstrap_ci_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        bootstrap_ci(np.array([0, 1]), np.array([0]))


def test_paired_test_detects_a_real_difference():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 400)
    strong = np.where(rng.random(400) < 0.9, y_true, 1 - y_true)
    weak = np.where(rng.random(400) < 0.6, y_true, 1 - y_true)
    result = paired_bootstrap_test(y_true, strong, weak, n_resamples=300, seed=0)
    assert result["difference"] > 0
    assert result["ci_low"] > 0  # interval excludes zero
    assert result["p_value"] < 0.05


def test_paired_test_sees_no_difference_between_identical_systems():
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 2, 200)
    pred = np.where(rng.random(200) < 0.8, y_true, 1 - y_true)
    result = paired_bootstrap_test(y_true, pred, pred, n_resamples=200, seed=0)
    assert result["difference"] == pytest.approx(0.0)
    assert result["ci_low"] == pytest.approx(0.0)
    assert result["ci_high"] == pytest.approx(0.0)
