#!/usr/bin/env python3
"""Week-one job: fetch and cache every dataset, and report what failed.

Run this before any modelling. It is the step the proposal flags as a risk:
XSumFaith needs ``datasets<3.0`` and SummEval comes from Google Drive.

Usage:
    python scripts/download_data.py --cut val
"""

from __future__ import annotations

import argparse
import sys

from frugal_faith.config import Config
from frugal_faith.data import download_frank_annotations, frank_error_types, load_summac


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cut", default="val", choices=["val", "test"])
    args = parser.parse_args()

    config = Config()
    examples, failures = load_summac(args.cut, config.data_dir, config.datasets)

    print(f"Loaded {len(examples)} examples from the '{args.cut}' cut.")
    by_dataset: dict[str, int] = {}
    for example in examples:
        by_dataset[example.dataset] = by_dataset.get(example.dataset, 0) + 1
    for name in config.datasets:
        status = by_dataset.get(name)
        print(f"  {name:<12} {status if status is not None else 'FAILED':>8}")

    if failures:
        print("\nFailures (drop these datasets if they stay broken):")
        for name, message in failures.items():
            print(f"  {name}: {message}")

    try:
        path = download_frank_annotations(config.data_dir)
        types = frank_error_types(path)
        print(f"\nFRANK error typology: {len(types)} annotated summaries.")
        frank = [e for e in examples if e.dataset == "frank"]
        matched = sum(1 for e in frank if e.example_id in types)
        print(f"Join check: {matched}/{len(frank)} FRANK examples matched a typology entry.")
        if frank and matched == 0:
            print("  -> Join key is wrong. Fix data.attach_error_types before RQ2.")
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        print(f"\nFRANK annotations unavailable: {exc}")

    return 0 if examples else 1


if __name__ == "__main__":
    sys.exit(main())
