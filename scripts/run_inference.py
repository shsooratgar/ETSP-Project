#!/usr/bin/env python3
"""Score the benchmark with one model and cache the predictions.

One model per invocation, so a cluster job can run each size separately and a
crash never costs more than one model's work.

Usage:
    python scripts/run_inference.py --model Qwen/Qwen2.5-0.5B-Instruct --cut val
"""

from __future__ import annotations

import argparse
import json
import sys

from frugal_faith.config import Config
from frugal_faith.data import load_summac
from frugal_faith.models import FaithfulnessScorer, cache_path, predictions_to_frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cut", default="val", choices=["val", "test"])
    parser.add_argument("--prompt", default="direct")
    parser.add_argument("--limit", type=int, default=None, help="subsample for a pilot")
    parser.add_argument("--load-in-8bit", action="store_true")
    args = parser.parse_args()

    config = Config()
    examples, failures = load_summac(args.cut, config.data_dir, config.datasets)
    if failures:
        print(f"Warning: skipped {sorted(failures)}", file=sys.stderr)
    if args.limit:
        examples = examples[: args.limit]
    if not examples:
        print("No examples loaded; run scripts/download_data.py first", file=sys.stderr)
        return 1

    scorer = FaithfulnessScorer(
        args.model,
        max_document_tokens=config.max_document_tokens,
        load_in_8bit=args.load_in_8bit,
    )
    predictions, cost = scorer.score(examples, args.prompt)

    out = cache_path(config.results_dir, args.model, args.prompt, args.cut)
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions_to_frame(predictions).to_csv(out, index=False)
    with open(out.with_suffix(".cost.json"), "w", encoding="utf-8") as handle:
        json.dump(cost.__dict__, handle, indent=2)

    print(f"Wrote {len(predictions)} predictions to {out}")
    print(f"{cost.seconds_per_example:.3f}s/example, peak {cost.peak_memory_gb:.1f}GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
