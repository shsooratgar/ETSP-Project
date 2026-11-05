"""Loading the SummaC benchmark.

We use the loaders shipped in the ``summac`` package (0.0.4). They download each
dataset on first use. ``summac`` itself pins an old ``transformers`` that cannot
load Qwen2.5, so it is installed with ``--no-deps`` (see the Makefile); only its
data loaders are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: FRANK's error typology (Pagnoni et al., 2021), grouping its fine-grained labels.
FRANK_ERROR_GROUPS: dict[str, str] = {
    "RelE": "semantic frame",
    "EntE": "semantic frame",
    "CircE": "semantic frame",
    "CorefE": "discourse",
    "LinkE": "discourse",
    "OutE": "content verifiability",
    "GramE": "content verifiability",
    "OtherE": "other",
}

#: Readable names for FRANK's fine-grained labels, used in tables and figures.
FRANK_ERROR_NAMES: dict[str, str] = {
    "RelE": "predicate",
    "EntE": "entity",
    "CircE": "circumstance",
    "CorefE": "coreference",
    "LinkE": "discourse link",
    "OutE": "out of article",
    "GramE": "grammatical",
    "OtherE": "other",
}


@dataclass(frozen=True)
class Example:
    """One (document, summary) pair with its consistency label."""

    example_id: str
    dataset: str
    document: str
    summary: str
    label: int  # 1 = consistent, 0 = inconsistent
    #: Majority error category for inconsistent summaries where the dataset
    #: provides one (FRANK, SummEval); "NoE" for consistent ones, "" otherwise.
    error_type: str = ""


def _use_parquet_xsum() -> None:
    """Point SummaC's XSum loader at the Hub's Parquet export.

    The loader calls ``load_dataset("xsum")``, a script-based dataset whose
    original download URL is no longer reliable. The Parquet conversion of the
    same test split carries identical ids and documents.
    """
    from datasets import load_dataset
    from summac.benchmark import SummaCBenchmark

    def get_xsum_document(self, aid):  # noqa: ANN001 - matches summac's signature
        if self.xsum is None:
            self.xsum = load_dataset(
                "EdinburghNLP/xsum", revision="refs/convert/parquet", split="test"
            )
            self.xsumid2article = {d["id"]: d["document"] for d in self.xsum}
        return self.xsumid2article[aid]

    SummaCBenchmark.get_xsum_document = get_xsum_document


def load_summac(
    cut: str,
    data_dir: Path,
    datasets: tuple[str, ...],
) -> tuple[list[Example], dict[str, str]]:
    """Load the SummaC benchmark.

    Args:
        cut: ``"val"`` or ``"test"``.
        data_dir: where the benchmark caches its downloads.
        datasets: which of the six to load, using summac's names
            (``cogensum``, ``xsumfaith``, ``polytope``, ``factcc``, ``summeval``, ``frank``).

    Returns:
        The examples, and a mapping of dataset name to error message for any
        dataset that could not be loaded. A partial benchmark is usable; the
        README says which datasets are safe to drop.
    """
    try:
        from summac.benchmark import SummaCBenchmark
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "The 'summac' package is required: run `make install`"
        ) from exc

    _use_parquet_xsum()
    data_dir.mkdir(parents=True, exist_ok=True)
    examples: list[Example] = []
    failures: dict[str, str] = {}

    for name in datasets:
        try:
            benchmark = SummaCBenchmark(
                benchmark_folder=str(data_dir / "summac") + "/",
                dataset_names=[name],
                cut=cut,
            )
        except Exception as exc:  # noqa: BLE001 - we report, not crash
            failures[name] = f"{type(exc).__name__}: {exc}"
            continue
        for split in benchmark.datasets:
            for i, row in enumerate(split["dataset"]):
                examples.append(
                    Example(
                        example_id=f"{split['name']}-{cut}-{i}",
                        dataset=split["name"],
                        document=row["document"],
                        summary=row["claim"],
                        label=int(row["label"]),
                        error_type=str(row.get("error_type", "")),
                    )
                )
    return examples, failures
