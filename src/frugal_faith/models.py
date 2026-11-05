"""Running a Qwen model over the benchmark and measuring what it costs.

torch and transformers are imported lazily so that the rest of the package (and
the whole test suite) works without them.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .data import Example
from .prompts import LABEL_WORDS, build
from .scoring import Prediction, label_probabilities, to_prediction


@dataclass(frozen=True)
class CostRecord:
    """What one model cost on one run, for the cost axis of the curve."""

    model: str
    device: str
    n_examples: int
    seconds_per_example: float
    peak_memory_gb: float
    tokens_per_example: float
    flops_per_example: float


def estimate_flops(n_parameters: int, n_tokens: float) -> float:
    """Rough forward-pass FLOPs: ``2 * parameters * tokens``.

    The standard approximation (two FLOPs per parameter per token). Reported
    alongside measured latency because latency depends on the hardware.
    """
    return 2.0 * n_parameters * n_tokens


def pick_device(requested: str = "auto") -> str:
    """CUDA if present, else Apple's MPS, else CPU."""
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class FaithfulnessScorer:
    """Scores (document, summary) pairs with one model via label-word probabilities."""

    def __init__(
        self,
        model_name: str,
        max_document_tokens: int = 1024,
        batch_size: int = 4,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.max_document_tokens = max_document_tokens
        self.batch_size = batch_size
        self.device = pick_device(device)
        self._model = None
        self._tokenizer = None
        self.n_parameters = 0

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # bf16 halves memory and is numerically safer than fp16 for Qwen;
        # CPUs fall back to fp32.
        dtype = torch.bfloat16 if self.device in ("cuda", "mps") else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._tokenizer.padding_side = "left"  # last position is the next token
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype)
        self._model.to(self.device)
        self._model.eval()
        self.n_parameters = sum(p.numel() for p in self._model.parameters())

    def _label_token_ids(self) -> tuple[int, int]:
        """Token ids of the label words as the first token of the answer.

        The chat template ends the prompt with a newline, so the answer starts
        without a leading space and the bare word's first token is the right id.
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

    def _prompt(self, example: Example, prompt_id: str) -> str:
        messages = build(prompt_id, self._truncate(example.document), example.summary)
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _peak_memory_gb(self) -> float:
        import torch

        if self.device == "cuda":
            return torch.cuda.max_memory_allocated() / 1e9
        if self.device == "mps":
            return torch.mps.driver_allocated_memory() / 1e9
        return 0.0

    def score_iter(
        self, examples: list[Example], prompt_id: str
    ) -> Iterator[tuple[list[Prediction], float]]:
        """Yield ``(predictions, forward_seconds)`` batch by batch.

        Callers can checkpoint after every batch. Only the forward pass is timed,
        so prompt preparation is not billed to the model. Batches group prompts of
        similar length to keep padding, and wasted compute, small.
        """
        import torch

        self._load()
        label_ids = self._label_token_ids()
        prompts = [(ex, self._prompt(ex, prompt_id)) for ex in examples]
        lengths = [len(self._tokenizer.encode(p)) for _, p in prompts]
        order = sorted(range(len(prompts)), key=lambda i: lengths[i])

        for start in range(0, len(order), self.batch_size):
            batch = [prompts[i] for i in order[start : start + self.batch_size]]
            inputs = self._tokenizer(
                [p for _, p in batch], return_tensors="pt", padding=True
            ).to(self.device)
            start_time = time.perf_counter()
            with torch.no_grad():
                logits = self._model(**inputs).logits[:, -1, :].float().cpu().numpy()
            seconds = time.perf_counter() - start_time  # .cpu() forces a device sync
            real_tokens = inputs["attention_mask"].sum(dim=1).tolist()
            yield [
                to_prediction(
                    ex.example_id,
                    self.model_name,
                    label_probabilities(row, label_ids),
                    int(n_tokens),
                )
                for (ex, _), row, n_tokens in zip(batch, logits, real_tokens)
            ], seconds

    def cost(self, n_examples: int, seconds: float, total_tokens: int) -> CostRecord:
        """Summarise one run's cost."""
        n = max(n_examples, 1)
        return CostRecord(
            model=self.model_name,
            device=self.device,
            n_examples=n_examples,
            seconds_per_example=seconds / n,
            peak_memory_gb=self._peak_memory_gb(),
            tokens_per_example=total_tokens / n,
            flops_per_example=estimate_flops(self.n_parameters, total_tokens / n),
        )


def cache_path(results_dir: Path, model_name: str, prompt_id: str, cut: str) -> Path:
    """Where one model's outputs are cached, so analysis never re-runs inference."""
    slug = model_name.replace("/", "__")
    return results_dir / "predictions" / f"{slug}__{prompt_id}__{cut}.csv"

