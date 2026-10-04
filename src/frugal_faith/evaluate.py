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
