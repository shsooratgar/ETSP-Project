"""The analyses behind each research question, as pure functions.

``scripts/analyze.py`` does the file handling and plotting; everything that
computes a number lives here so it can be tested.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .evaluate import balanced_accuracy
from .scoring import Prediction


def tune_threshold(prob_consistent: np.ndarray, y_true: np.ndarray) -> float:
    """Decision threshold on P(consistent) that maximises balanced accuracy.

    Small chat models put most of their mass on "Yes" whatever the input, so the
    default 0.5 cut-off makes them predict "consistent" almost always. Following
    SummaC, the threshold is tuned on the validation cut and applied unchanged
    to the test cut. Ties go to the threshold closest to 0.5.
    """
    probs = np.asarray(prob_consistent, dtype=np.float64)
    y_true = np.asarray(y_true)
    candidates = np.unique(np.concatenate([probs, [0.5]]))
    preds = probs[None, :] >= candidates[:, None]  # (n_candidates, n_examples)
    scores = np.array([balanced_accuracy(y_true, row.astype(int)) for row in preds])
    best = np.flatnonzero(scores == scores.max())
    return float(candidates[best[np.argmin(np.abs(candidates[best] - 0.5))]])


def calibrate(prediction: Prediction, threshold: float) -> Prediction:
    """Re-express a prediction relative to its tuned threshold.

    The label becomes ``P(consistent) >= threshold`` and the confidence the
    distance from the threshold, rescaled to [0, 1], so a router reading
    "confidence" sees how far the model is from changing its answer.
    """
    p = prediction.prob_consistent
    span = max(threshold, 1.0 - threshold)
    return replace(
        prediction,
        label=int(p >= threshold),
        confidence=abs(p - threshold) / span if span > 0 else 0.0,
    )


def macro_balanced_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray
) -> float:
    """Balanced accuracy averaged over datasets, as SummaC reports it.

    Pooling would let the largest dataset dominate and mix very different
    base rates of inconsistent summaries.
    """
    y_true, y_pred, groups = map(np.asarray, (y_true, y_pred, groups))
    return float(
        np.mean(
            [balanced_accuracy(y_true[groups == g], y_pred[groups == g]) for g in np.unique(groups)]
        )
    )


def outcome_breakdown(
    y_true: np.ndarray, small_pred: np.ndarray, large_pred: np.ndarray
) -> dict[str, float]:
    """Where two models agree and disagree, as shares of all examples.

    A net accuracy gap hides how much the large model wins *and* loses; this
    shows both. ``large_only`` is what escalation can gain, ``small_only`` what
    it can cost.
    """
    y_true, small_pred, large_pred = map(np.asarray, (y_true, small_pred, large_pred))
    small_ok = small_pred == y_true
    large_ok = large_pred == y_true
    return {
        "both_right": float(np.mean(small_ok & large_ok)),
        "large_only": float(np.mean(~small_ok & large_ok)),
        "small_only": float(np.mean(small_ok & ~large_ok)),
        "both_wrong": float(np.mean(~small_ok & ~large_ok)),
    }


def detection_by_category(
    y_true: np.ndarray, y_pred: np.ndarray, categories: np.ndarray
) -> pd.DataFrame:
    """Share of inconsistent summaries flagged, per error category.

    Only inconsistent summaries carry an error category, so this is recall on
    the "inconsistent" class, split by the kind of error.
    """
    y_true, y_pred, categories = map(np.asarray, (y_true, y_pred, categories))
    inconsistent = y_true == 0
    rows = []
    for category in sorted(set(categories[inconsistent])):
        mask = inconsistent & (categories == category)
        rows.append(
            {"category": category, "n": int(mask.sum()), "detected": float(np.mean(y_pred[mask] == 0))}
        )
    return pd.DataFrame(rows)


def within_group_terciles(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Label each value short / medium / long relative to its own dataset.

    Documents differ in length across datasets (XSum articles are much shorter
    than CNN/DailyMail ones), so pooled bins would just re-encode the dataset.
    """
    values, groups = np.asarray(values, dtype=np.float64), np.asarray(groups)
    labels = np.empty(len(values), dtype=object)
    for g in np.unique(groups):
        mask = groups == g
        low, high = np.quantile(values[mask], [1 / 3, 2 / 3])
        labels[mask] = np.where(
            values[mask] <= low, "short", np.where(values[mask] <= high, "medium", "long")
        )
    return labels


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve, threshold-free.

    Used for prompt sensitivity, where no validation run exists to tune a
    threshold for every wording.
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        raise ValueError("roc_auc needs both classes")
    return float(roc_auc_score(y_true, np.asarray(scores, dtype=np.float64)))


def _stratified_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample with replacement *within* each dataset, keeping dataset sizes."""
    return np.concatenate(
        [rng.choice(np.flatnonzero(groups == g), size=int(np.sum(groups == g))) for g in np.unique(groups)]
    )


def macro_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    groups: np.ndarray,
    pred_b: np.ndarray | None = None,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap the macro-averaged balanced accuracy, or a paired difference.

    With one system, returns its score and a 95% interval. With ``pred_b``,
    returns the paired difference A minus B, its interval, and the share of
    resamples in which A did not beat B (a one-sided p-value). Resampling is
    stratified by dataset so every resample keeps all datasets.
    """
    y_true, pred_a, groups = map(np.asarray, (y_true, pred_a, groups))
    rng = np.random.default_rng(seed)

    def statistic(idx: np.ndarray) -> float:
        a = macro_balanced_accuracy(y_true[idx], pred_a[idx], groups[idx])
        if pred_b is None:
            return a
        return a - macro_balanced_accuracy(y_true[idx], np.asarray(pred_b)[idx], groups[idx])

    observed = statistic(np.arange(len(y_true)))
    samples = []
    for _ in range(n_resamples):
        idx = _stratified_indices(groups, rng)
        if any(len(np.unique(y_true[idx][groups[idx] == g])) < 2 for g in np.unique(groups)):
            continue
        samples.append(statistic(idx))
    samples_arr = np.asarray(samples)
    result = {
        "value": float(observed),
        "ci_low": float(np.percentile(samples_arr, 2.5)),
        "ci_high": float(np.percentile(samples_arr, 97.5)),
    }
    if pred_b is not None:
        result["p_value"] = float(np.mean(samples_arr <= 0.0))
    return result


def cross_fitted_thresholds(
    prob_consistent: np.ndarray, y_true: np.ndarray, n_folds: int = 5, seed: int = 0
) -> np.ndarray:
    """One threshold per example, each tuned on the *other* folds.

    The router is trained on the validation cut. If the checker's thresholds
    were tuned on that same cut, the router would learn from labels that are
    optimistically accurate in-sample and look nothing like test-time
    behaviour. Cross-fitting gives every validation example a label from a
    threshold that never saw it.
    """
    probs = np.asarray(prob_consistent, dtype=np.float64)
    y_true = np.asarray(y_true)
    folds = np.random.default_rng(seed).permutation(len(probs)) % n_folds
    thresholds = np.empty(len(probs))
    for k in range(n_folds):
        held_out = folds == k
        thresholds[held_out] = tune_threshold(probs[~held_out], y_true[~held_out])
    return thresholds


def paired_auc_bootstrap(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """AUC of A minus AUC of B on the same examples, with a 95% interval.

    Used to compare prompt wordings scored on one shared subsample.
    """
    y_true = np.asarray(y_true)
    scores_a, scores_b = np.asarray(scores_a), np.asarray(scores_b)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        diffs.append(roc_auc(y_true[idx], scores_a[idx]) - roc_auc(y_true[idx], scores_b[idx]))
    diffs_arr = np.asarray(diffs)
    return {
        "difference": roc_auc(y_true, scores_a) - roc_auc(y_true, scores_b),
        "ci_low": float(np.percentile(diffs_arr, 2.5)),
        "ci_high": float(np.percentile(diffs_arr, 97.5)),
    }
