"""Loading the SummaC benchmark, plus FRANK's error typology.

The SummaC loaders download on first use. Two of them are fragile (see README):
XSumFaith needs ``datasets<3.0`` and SummEval is fetched from Google Drive. Both
are caught here and reported per dataset rather than failing the whole run.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

#: FRANK's annotations, used for the per-error-type analysis (RQ2).
FRANK_URL = "https://raw.githubusercontent.com/artidoro/frank/main/data/human_annotations.json"


@dataclass(frozen=True)
class Example:
    """One (document, summary) pair with its consistency label."""

    example_id: str
    dataset: str
    document: str
    summary: str
    label: int  # 1 = consistent, 0 = inconsistent
    #: FRANK error categories, empty for the other five datasets.
    error_types: tuple[str, ...] = field(default=())


def load_summac(
    cut: str,
    data_dir: Path,
    datasets: tuple[str, ...],
) -> tuple[list[Example], dict[str, str]]:
    """Load the SummaC benchmark.

    Args:
        cut: ``"val"`` or ``"test"``.
        data_dir: where the benchmark caches its downloads.
        datasets: which of the six to load.

    Returns:
        The examples, and a mapping of dataset name to error message for any
        dataset that could not be loaded. A partial benchmark is usable; the
        README says which datasets are safe to drop.
    """
    try:
        from summac.benchmark import SummaCBenchmark
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "The 'summac' package is required: pip install -r requirements.txt"
        ) from exc

    data_dir.mkdir(parents=True, exist_ok=True)
    examples: list[Example] = []
    failures: dict[str, str] = {}

    for name in datasets:
        try:
            benchmark = SummaCBenchmark(
                benchmark_folder=str(data_dir / "summac"),
                cut=cut,
                dataset_names=[name],
            )
        except Exception as exc:  # noqa: BLE001 - we report, not crash
            failures[name] = f"{type(exc).__name__}: {exc}"
            continue
        for split in benchmark.datasets:
            for i, row in enumerate(split["dataset"]):
                examples.append(
                    Example(
                        example_id=f"{name}-{cut}-{i}",
                        dataset=name,
                        document=row["document"],
                        summary=row["claim"],
                        label=int(row["label"]),
                    )
                )
    return examples, failures


def download_frank_annotations(data_dir: Path) -> Path:
    """Fetch FRANK's human annotations, which carry the error typology."""
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "frank_human_annotations.json"
    if not target.exists():
        urllib.request.urlretrieve(FRANK_URL, target)  # noqa: S310 - fixed https URL
    return target


def frank_error_types(annotations_path: Path) -> dict[str, tuple[str, ...]]:
    """Map FRANK's ``hash``/``model_name`` key to its annotated error categories.

    FRANK labels each summary sentence with a category from its typology
    (semantic frame, discourse, content verifiability). We keep the set of
    categories present anywhere in the summary.
    """
    with open(annotations_path, encoding="utf-8") as handle:
        records = json.load(handle)

    out: dict[str, tuple[str, ...]] = {}
    for record in records:
        key = f"{record.get('hash')}-{record.get('model_name')}"
        categories: set[str] = set()
        for annotation in record.get("summary_sentences_annotations", []):
            for votes in annotation.values():
                categories.update(v for v in votes if v and v != "NoE")
        out[key] = tuple(sorted(categories))
    return out


def attach_error_types(
    examples: list[Example], error_types: dict[str, tuple[str, ...]]
) -> list[Example]:
    """Return FRANK examples enriched with their error categories.

    Note:
        The join key depends on fields SummaC's loader keeps for FRANK. Verify
        the match rate before trusting the RQ2 numbers; ``scripts/download_data.py``
        reports it.
    """
    enriched = []
    for example in examples:
        key = example.example_id
        enriched.append(
            Example(
                example_id=example.example_id,
                dataset=example.dataset,
                document=example.document,
                summary=example.summary,
                label=example.label,
                error_types=error_types.get(key, ()),
            )
        )
    return enriched
