"""Central configuration. Everything tunable lives here, not in the scripts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Qwen2.5-Instruct sizes, smallest first. One family keeps size the main variable.
MODEL_LADDER: tuple[str, ...] = (
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
)

#: The six SummaC datasets. Drop any that will not download (see README).
SUMMAC_DATASETS: tuple[str, ...] = (
    "cogensumm",
    "xsumfaith",
    "polytope",
    "factcc",
    "summeval",
    "frank",
)


@dataclass(frozen=True)
class Config:
    """Run configuration. Immutable so it can be logged alongside results."""

    data_dir: Path = REPO_ROOT / "data"
    results_dir: Path = REPO_ROOT / "results"
    models: tuple[str, ...] = MODEL_LADDER
    datasets: tuple[str, ...] = SUMMAC_DATASETS
    prompt_id: str = "direct"
    #: Source documents are long; truncate to keep 7B inference affordable.
    max_document_tokens: int = 1024
    batch_size: int = 8
    seed: int = 20260101
    #: Budgets (fraction of examples escalated to the large model) for the curve.
    budgets: tuple[float, ...] = field(
        default_factory=lambda: tuple(i / 20 for i in range(21))
    )

    @property
    def small_model(self) -> str:
        return self.models[0]

    @property
    def large_model(self) -> str:
        return self.models[-1]
