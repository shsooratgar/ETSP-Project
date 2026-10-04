"""Turning model logits into a label, a probability and a confidence.

Kept free of torch so it can be unit-tested without a GPU or a model download.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Prediction:
    """One model's verdict on one (document, summary) pair."""

    example_id: str
    model: str
    label: int  # 1 = consistent, 0 = inconsistent
    prob_consistent: float
    confidence: float  # max of the two renormalised label probabilities
    entropy: float
    input_tokens: int


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def label_probabilities(
    next_token_logits: np.ndarray, label_token_ids: tuple[int, int]
) -> np.ndarray:
    """Probabilities of the two label words, renormalised to sum to 1.

    Only the label tokens compete, so the model cannot spend probability mass on
    unrelated continuations. Index 0 is "inconsistent", index 1 "consistent".

    Args:
        next_token_logits: logits for the first generated position, shape ``(vocab,)``.
        label_token_ids: token ids of the negative and positive label words.
    """
    if next_token_logits.ndim != 1:
        raise ValueError(f"expected 1-D logits, got shape {next_token_logits.shape}")
    picked = np.asarray(
        [next_token_logits[i] for i in label_token_ids], dtype=np.float64
    )
    return softmax(picked)


def entropy(probs: np.ndarray) -> float:
    """Shannon entropy in bits; 0 means certain, 1 means a coin flip.

    Zero probabilities contribute nothing (``0 log 0 = 0``) and are dropped rather
    than clipped, so a certain prediction scores exactly 0.0.
    """
    p = np.asarray(probs, dtype=np.float64)
    p = p[p > 0.0]
    return float(-np.sum(p * np.log2(p)))


def to_prediction(
    example_id: str,
    model: str,
    probs: np.ndarray,
    input_tokens: int,
) -> Prediction:
    """Package renormalised label probabilities into a :class:`Prediction`."""
    probs = np.asarray(probs, dtype=np.float64)
    return Prediction(
        example_id=example_id,
        model=model,
        label=int(np.argmax(probs)),
        prob_consistent=float(probs[1]),
        confidence=float(np.max(probs)),
        entropy=entropy(probs),
        input_tokens=int(input_tokens),
    )
