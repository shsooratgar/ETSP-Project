"""Running a Qwen model over the benchmark and measuring what it costs.

torch and transformers are imported lazily so that the rest of the package (and
the whole test suite) works on a laptop without them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import Example
from .prompts import LABEL_WORDS, build
from .scoring import Prediction, label_probabilities, to_prediction


@dataclass(frozen=True)
class CostRecord:
    """What one model cost on one run, for the cost axis of the curve."""

    model: str
    seconds_per_example: float
    peak_memory_gb: float
    flops_per_example: float


def estimate_flops(n_parameters: int, n_tokens: int) -> float:
    """Rough forward-pass FLOPs: ``2 * parameters * tokens``.

    The standard approximation (two FLOPs per parameter per token). Reported
    alongside measured latency because latency depends on the GPU we happen to get.
    """
    return 2.0 * n_parameters * n_tokens


class FaithfulnessScorer:
    """Scores (document, summary) pairs with one model via label-word probabilities."""

    def __init__(
        self,
        model_name: str,
        max_document_tokens: int = 1024,
        load_in_8bit: bool = False,
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.max_document_tokens = max_document_tokens
        self.load_in_8bit = load_in_8bit
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        kwargs: dict = {"torch_dtype": torch.float16, "device_map": self.device}
        if self.load_in_8bit:
            kwargs = {"load_in_8bit": True, "device_map": "auto"}
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self._model.eval()

    def _label_token_ids(self) -> tuple[int, int]:
        """Token ids of the label words as they appear after the prompt.

        Qwen's tokenizer splits on leading whitespace, so the ids are taken from
        the first token of each label word encoded without a prefix space.
        """
        ids = []
        for word in LABEL_WORDS:
            encoded = self._tokenizer.encode(word, add_special_tokens=False)
            if not encoded:
                raise ValueError(f"label word {word!r} encodes to nothing")
            ids.append(encoded[0])
        return ids[0], ids[1]

    def _truncate(self, document: str) -> str:
        ids = self._tokenizer.encode(document, add_special_tokens=False)
        if len(ids) <= self.max_document_tokens:
            return document
        return self._tokenizer.decode(ids[: self.max_document_tokens])

    def score(self, examples: list[Example], prompt_id: str) -> tuple[list[Prediction], CostRecord]:
        """Score every example, returning predictions and the measured cost."""
        import torch

        self._load()
        label_ids = self._label_token_ids()
        predictions: list[Prediction] = []
        total_tokens = 0
        start = time.perf_counter()

        for example in examples:
            messages = build(prompt_id, self._truncate(example.document), example.summary)
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                logits = self._model(**inputs).logits[0, -1, :]
            probs = label_probabilities(logits.float().cpu().numpy(), label_ids)
            n_tokens = int(inputs["input_ids"].shape[1])
            total_tokens += n_tokens
            predictions.append(
                to_prediction(example.example_id, self.model_name, probs, n_tokens)
            )

        elapsed = time.perf_counter() - start
        n = max(len(examples), 1)
        peak_gb = (
            torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        )
        n_params = sum(p.numel() for p in self._model.parameters())
        cost = CostRecord(
            model=self.model_name,
            seconds_per_example=elapsed / n,
            peak_memory_gb=peak_gb,
            flops_per_example=estimate_flops(n_params, total_tokens // n),
        )
        return predictions, cost


def predictions_to_frame(predictions: list[Prediction]):
    """Flatten predictions for caching to disk (parquet/csv)."""
    import pandas as pd

    return pd.DataFrame([p.__dict__ for p in predictions])


def cache_path(results_dir: Path, model_name: str, prompt_id: str, cut: str) -> Path:
    """Where one model's outputs are cached, so analysis never re-runs inference."""
    slug = model_name.replace("/", "__")
    return results_dir / "predictions" / f"{slug}__{prompt_id}__{cut}.csv"
