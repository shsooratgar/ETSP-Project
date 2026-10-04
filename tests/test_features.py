"""Router features."""

from __future__ import annotations

import numpy as np
import pytest

from frugal_faith.data import Example
from frugal_faith.features import FEATURE_NAMES, extract_features, novel_word_ratio, to_matrix
from frugal_faith.scoring import to_prediction


@pytest.fixture
def example() -> Example:
    return Example(
        example_id="ex-1",
        dataset="frank",
        document="The cat sat on the mat in 2019. It was not raining.",
        summary="The dog sat on the mat.",
        label=0,
    )


def test_novel_word_ratio_counts_unseen_words():
    # "dog" is the only summary word absent from the document.
    assert novel_word_ratio("the cat sat", "the dog sat") == pytest.approx(1 / 3)


def test_novel_word_ratio_handles_empty_summary():
    assert novel_word_ratio("anything", "") == 0.0


def test_extract_features_matches_declared_names(example):
    prediction = to_prediction("ex-1", "tiny", np.array([0.3, 0.7]), input_tokens=20)
    features = extract_features(example, prediction)
    assert set(features) == set(FEATURE_NAMES)
    assert features["negation_count"] == 0.0  # the negation is in the document
    assert features["small_confidence"] == pytest.approx(0.7)


def test_to_matrix_preserves_column_order(example):
    prediction = to_prediction("ex-1", "tiny", np.array([0.3, 0.7]), input_tokens=20)
    row = extract_features(example, prediction)
    matrix = to_matrix([row, row])
    assert matrix.shape == (2, len(FEATURE_NAMES))
    assert matrix[0, FEATURE_NAMES.index("small_confidence")] == pytest.approx(0.7)


def test_to_matrix_handles_no_rows():
    assert to_matrix([]).shape == (0, len(FEATURE_NAMES))
