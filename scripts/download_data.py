#!/usr/bin/env python3
"""Week-one job: fetch and cache every dataset, and report what failed.

Run this before any modelling. Source documents come from CNN/DailyMail and
XSum on the Hugging Face Hub; the annotations from each dataset's own release.

Usage:
    python scripts/download_data.py --cut val
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from frugal_faith.config import Config
from frugal_faith.data import load_summac


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cut", default="val", choices=["val", "test"])
    args = parser.parse_args()

    config = Config()
    examples, failures = load_summac(args.cut, config.data_dir, config.datasets)

    print(f"Loaded {len(examples)} examples from the '{args.cut}' cut.")
    counts = Counter(e.dataset for e in examples)
    consistent = Counter(e.dataset for e in examples if e.label == 1)
    for name, n in sorted(counts.items()):
        print(f"  {name:<12} {n:>6}  ({consistent[name] / n:.0%} consistent)")

    if failures:
        print("\nFailures (drop these datasets if they stay broken):")
        for name, message in failures.items():
            print(f"  {name}: {message}")

    frank_errors = Counter(
        e.error_type for e in examples if e.dataset == "frank" and e.label == 0
    )
    if frank_errors:
        print(f"\nFRANK error types among inconsistent summaries: {dict(frank_errors)}")

    return 0 if examples else 1


if __name__ == "__main__":
    sys.exit(main())
