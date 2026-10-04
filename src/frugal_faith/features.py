"""Cheap features for the router.

Every feature must be computable *without* running the large model, otherwise the
router cannot save anything. The small model's own output is free: we already
paid for it in the first stage of the cascade.
"""

from __future__ import annotations

import re

import numpy as np

from .data import Example
from .scoring import Prediction

#: Stable column order. Tests pin this so a reordering cannot silently change a model.
FEATURE_NAMES: tuple[str, ...] = (
    "small_confidence",
    "small_entropy",
    "small_prob_consistent",
    "doc_tokens",
    "summary_tokens",
    "length_ratio",
    "novel_word_ratio",
    "negation_count",
    "number_count",
)

_NEGATIONS = frozenset(
    {"not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "without"}
)
_WORD = re.compile(r"[a-z0-9']+")
_NUMBER = re.compile(r"\b\d[\d,.]*\b")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def novel_word_ratio(document: str, summary: str) -> float:
    """Share of summary words absent from the document.

    A summary made entirely of words copied from the source is unlikely to be
    unfaithful, so this is a cheap proxy for how much the model must actually
    reason about. Returns 0.0 for an empty summary.
    """
    doc_words = set(_words(document))
    sum_words = _words(summary)
    if not sum_words:
        return 0.0
    return sum(word not in doc_words for word in sum_words) / len(sum_words)


def extract_features(example: Example, small: Prediction) -> dict[str, float]:
    """Feature dictionary for one example, given the small model's prediction."""
    doc_words = _words(example.document)
    sum_words = _words(example.summary)
    features = {
        "small_confidence": small.confidence,
        "small_entropy": small.entropy,
        "small_prob_consistent": small.prob_consistent,
        "doc_tokens": float(len(doc_words)),
        "summary_tokens": float(len(sum_words)),
        "length_ratio": len(sum_words) / max(len(doc_words), 1),
        "novel_word_ratio": novel_word_ratio(example.document, example.summary),
        "negation_count": float(sum(w in _NEGATIONS for w in sum_words)),
        "number_count": float(len(_NUMBER.findall(example.summary))),
    }
    assert set(features) == set(FEATURE_NAMES), "feature set drifted from FEATURE_NAMES"
    return features


def to_matrix(rows: list[dict[str, float]]) -> np.ndarray:
    """Stack feature dicts into a ``(n_examples, n_features)`` array."""
    if not rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    return np.asarray(
        [[row[name] for name in FEATURE_NAMES] for row in rows], dtype=np.float64
    )
