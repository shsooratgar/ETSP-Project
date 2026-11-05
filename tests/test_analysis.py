"""Analysis functions behind RQ1-RQ3."""

from __future__ import annotations

import numpy as np
import pytest

from frugal_faith.analysis import (
    calibrate,
    detection_by_category,
    macro_balanced_accuracy,
    outcome_breakdown,
    roc_auc,
    tune_threshold,
    within_group_terciles,
)
from frugal_faith.router import OracleRouter, escalation_mask
from frugal_faith.scoring import to_prediction


def test_tune_threshold_rescues_a_yes_biased_model():
    # Every probability is above 0.5, so the default cut-off says "consistent"
    # for everything, but the ranking separates the classes perfectly.
    probs = np.array([0.70, 0.72, 0.74, 0.90, 0.92, 0.94])
    y_true = np.array([0, 0, 0, 1, 1, 1])
    threshold = tune_threshold(probs, y_true)
    assert 0.74 < threshold <= 0.90
    assert ((probs >= threshold).astype(int) == y_true).all()


def test_tune_threshold_prefers_half_on_ties():
    # Any threshold in (0.2, 0.8] separates perfectly; 0.5 is the natural pick.
    assert tune_threshold(np.array([0.2, 0.8]), np.array([0, 1])) == 0.5


def test_calibrate_relabels_and_measures_distance_from_threshold():
    prediction = to_prediction("ex", "m", np.array([0.2, 0.8]), input_tokens=10)
    calibrated = calibrate(prediction, threshold=0.9)
    assert calibrated.label == 0  # 0.8 is below the tuned threshold
    assert calibrated.confidence == pytest.approx(0.1 / 0.9)
    assert calibrated.prob_consistent == prediction.prob_consistent


def test_macro_average_weights_datasets_equally():
    y_true = np.array([0, 1] + [0, 1] * 50)
    y_pred = np.array([1, 0] + [0, 1] * 50)  # wrong on the tiny dataset only
    groups = np.array(["tiny"] * 2 + ["big"] * 100)
    assert macro_balanced_accuracy(y_true, y_pred, groups) == pytest.approx(0.5)


def test_outcome_breakdown_sums_to_one_and_separates_wins_from_losses():
    y_true = np.array([1, 1, 1, 1])
    small = np.array([1, 0, 1, 0])
    large = np.array([1, 1, 0, 0])
    outcome = outcome_breakdown(y_true, small, large)
    assert sum(outcome.values()) == pytest.approx(1.0)
    assert outcome == {
        "both_right": 0.25,
        "large_only": 0.25,
        "small_only": 0.25,
        "both_wrong": 0.25,
    }


def test_detection_by_category_only_counts_inconsistent_summaries():
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    categories = np.array(["EntE", "EntE", "OutE", "NoE"])
    table = detection_by_category(y_true, y_pred, categories).set_index("category")
    assert set(table.index) == {"EntE", "OutE"}
    assert table.loc["EntE", "detected"] == pytest.approx(0.5)
    assert table.loc["OutE", "n"] == 1


def test_terciles_are_relative_to_each_dataset():
    values = np.array([1, 2, 3, 100, 200, 300])
    groups = np.array(["a", "a", "a", "b", "b", "b"])
    labels = within_group_terciles(values, groups)
    assert labels.tolist() == ["short", "medium", "long"] * 2


def test_roc_auc_known_values():
    y_true = np.array([0, 0, 1, 1])
    assert roc_auc(y_true, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert roc_auc(y_true, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    assert roc_auc(y_true, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)


def test_oracle_escalates_exactly_the_fixable_examples():
    improves = np.array([0, 1, 0, 1])
    mask = escalation_mask(OracleRouter(improves).scores(np.zeros((4, 1))), 0.5)
    assert mask.tolist() == [False, True, False, True]


def test_macro_bootstrap_brackets_the_score_and_tests_differences():
    from frugal_faith.analysis import macro_bootstrap

    rng = np.random.default_rng(0)
    groups = np.array(["a"] * 200 + ["b"] * 200)
    y_true = rng.integers(0, 2, 400)
    strong = np.where(rng.random(400) < 0.9, y_true, 1 - y_true)
    weak = np.where(rng.random(400) < 0.6, y_true, 1 - y_true)

    single = macro_bootstrap(y_true, strong, groups, n_resamples=200)
    assert single["ci_low"] < single["value"] < single["ci_high"]
    assert "p_value" not in single

    paired = macro_bootstrap(y_true, strong, groups, pred_b=weak, n_resamples=200)
    assert paired["value"] > 0 and paired["ci_low"] > 0
    assert paired["p_value"] < 0.05
