"""Scoring maths: no model, no GPU, no network."""

from __future__ import annotations

import numpy as np
import pytest

from frugal_faith.scoring import entropy, label_probabilities, softmax, to_prediction


def test_softmax_sums_to_one():
    assert softmax(np.array([1.0, 2.0, 3.0])).sum() == pytest.approx(1.0)


def test_softmax_is_shift_invariant():
    logits = np.array([0.5, -1.0, 2.0])
    assert softmax(logits) == pytest.approx(softmax(logits + 100.0))


def test_label_probabilities_only_uses_label_tokens():
    # A huge logit on an unrelated token must not affect the two label words.
    logits = np.zeros(10)
    logits[7] = 50.0  # some unrelated continuation
    logits[2], logits[5] = 1.0, 1.0  # the label words, equally likely
    probs = label_probabilities(logits, (2, 5))
    assert probs == pytest.approx([0.5, 0.5])


def test_label_probabilities_rejects_batched_input():
    with pytest.raises(ValueError):
        label_probabilities(np.zeros((2, 10)), (0, 1))


def test_entropy_bounds():
    assert entropy(np.array([0.5, 0.5])) == pytest.approx(1.0)
    assert entropy(np.array([1.0, 0.0])) == pytest.approx(0.0)


def test_to_prediction_picks_the_larger_probability():
    prediction = to_prediction("ex-1", "tiny", np.array([0.2, 0.8]), input_tokens=42)
    assert prediction.label == 1
    assert prediction.prob_consistent == pytest.approx(0.8)
    assert prediction.confidence == pytest.approx(0.8)
    assert prediction.input_tokens == 42
