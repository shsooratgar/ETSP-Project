"""Metrics and the cost-accuracy curve that the project's claim rests on."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .router import RoutingPolicy, escalation_mask


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean of per-class recall, the metric SummaC reports.

    Plain accuracy flatters a classifier on these datasets because most summaries
    are labelled consistent.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    recalls = []
    for label in (0, 1):
        in_class = y_true == label
        if in_class.any():
            recalls.append(float(np.mean(y_pred[in_class] == label)))
    if not recalls:
        raise ValueError("y_true contains neither class")
    return float(np.mean(recalls))


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for balanced accuracy.

    Resamples examples with replacement. Resamples that lose a class entirely are
    skipped rather than counted, which matters on the smaller SummaC datasets
    where inconsistent summaries are rare.

    Returns:
        The lower and upper bounds of the interval.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(balanced_accuracy(y_true[idx], y_pred[idx]))
    if not scores:
        raise ValueError("No resample contained both classes; sample is too small")
    tail = (1.0 - confidence) / 2.0
    return (
        float(np.percentile(scores, 100 * tail)),
        float(np.percentile(scores, 100 * (1.0 - tail))),
    )


def paired_bootstrap_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Is model A better than model B, or is the difference noise?

    Resamples the *same* examples for both systems, so the comparison is paired
    and the shared difficulty of the examples cancels out. This is what we report
    when claiming a larger model beats a smaller one.

    Returns:
        The observed difference (A minus B), the interval around it, and the
        share of resamples in which A did not beat B (a one-sided p-value).
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    observed = balanced_accuracy(y_true, pred_a) - balanced_accuracy(y_true, pred_b)

    diffs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        diffs.append(
            balanced_accuracy(y_true[idx], pred_a[idx])
            - balanced_accuracy(y_true[idx], pred_b[idx])
        )
    if not diffs:
        raise ValueError("No resample contained both classes; sample is too small")
    diffs_arr = np.asarray(diffs)
    return {
        "difference": float(observed),
        "ci_low": float(np.percentile(diffs_arr, 2.5)),
        "ci_high": float(np.percentile(diffs_arr, 97.5)),
        "p_value": float(np.mean(diffs_arr <= 0.0)),
    }


def combine(
    small_pred: np.ndarray, large_pred: np.ndarray, escalate: np.ndarray
) -> np.ndarray:
    """Take the large model's answer where escalated, the small one's elsewhere."""
    return np.where(escalate, large_pred, small_pred)


def cost_accuracy_curve(
    policy: RoutingPolicy,
    features: np.ndarray,
    small_pred: np.ndarray,
    large_pred: np.ndarray,
    y_true: np.ndarray,
    budgets: tuple[float, ...],
    cost_small: float,
    cost_large: float,
) -> pd.DataFrame:
    """Balanced accuracy and compute cost at each escalation budget.

    Cost is per example and includes the small model everywhere, because the
    cascade always pays for the first stage.
    """
    scores = policy.scores(features)
    rows = []
    for budget in budgets:
        escalate = escalation_mask(scores, budget)
        combined = combine(small_pred, large_pred, escalate)
        rows.append(
            {
                "policy": policy.name,
                "budget": budget,
                "escalated": float(np.mean(escalate)),
                "balanced_accuracy": balanced_accuracy(y_true, combined),
                "cost_per_example": cost_small + float(np.mean(escalate)) * cost_large,
            }
        )
    return pd.DataFrame(rows)


def reference_points(
    small_pred: np.ndarray,
    large_pred: np.ndarray,
    y_true: np.ndarray,
    cost_small: float,
    cost_large: float,
) -> pd.DataFrame:
    """The two bounds: always-small (cheapest) and always-large (most accurate)."""
    return pd.DataFrame(
        [
            {
                "policy": "always_small",
                "balanced_accuracy": balanced_accuracy(y_true, small_pred),
                "cost_per_example": cost_small,
            },
            {
                "policy": "always_large",
                "balanced_accuracy": balanced_accuracy(y_true, large_pred),
                "cost_per_example": cost_large,
            },
        ]
    )
