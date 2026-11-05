#!/usr/bin/env python3
"""Score the benchmark with one model and cache the predictions.

One model per invocation, so each size can run separately. Predictions are
appended after every batch, and a rerun skips examples already in the cache, so
an interrupted run resumes instead of starting over.

Usage:
    python scripts/run_inference.py --model Qwen/Qwen2.5-0.5B-Instruct --cut val
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys

import pandas as pd

from frugal_faith.config import Config
from frugal_faith.data import load_summac
from frugal_faith.models import FaithfulnessScorer, cache_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cut", default="val", choices=["val", "test"])
    parser.add_argument("--prompt", default="direct")
    parser.add_argument("--limit", type=int, default=None, help="first N examples, for a pilot")
    parser.add_argument(
        "--sample", type=int, default=None, help="fixed random N examples across datasets"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tag", default="", help="suffix for the cache file, e.g. 'pilot'")
    args = parser.parse_args()

    config = Config()
    examples, failures = load_summac(args.cut, config.data_dir, config.datasets)
    if failures:
        print(f"Warning: skipped {sorted(failures)}", file=sys.stderr)
    if args.sample:
        rng = random.Random(config.seed)
        examples = rng.sample(examples, min(args.sample, len(examples)))
    if args.limit:
        examples = examples[: args.limit]
    if not examples:
        print("No examples loaded; run scripts/download_data.py first", file=sys.stderr)
        return 1

    out = cache_path(config.results_dir, args.model, args.prompt, args.cut)
    if args.tag:
        out = out.with_name(f"{out.stem}__{args.tag}{out.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if out.exists():
        done = set(pd.read_csv(out, usecols=["example_id"])["example_id"])
        print(f"Resuming: {len(done)} of {len(examples)} already cached")
    todo = [e for e in examples if e.example_id not in done]

    scorer = FaithfulnessScorer(
        args.model,
        max_document_tokens=config.max_document_tokens,
        batch_size=args.batch_size,
        device=args.device,
    )
    seconds = 0.0
    tokens = 0
    for i, (batch, batch_seconds) in enumerate(scorer.score_iter(todo, args.prompt)):
        seconds += batch_seconds
        tokens += sum(p.input_tokens for p in batch)
        frame = pd.DataFrame([dataclasses.asdict(p) for p in batch])
        frame.to_csv(out, mode="a", header=not out.exists(), index=False)
        if i % 25 == 0:
            n = (i + 1) * args.batch_size
            print(f"  {min(n, len(todo))}/{len(todo)}  {seconds / max(n, 1):.3f}s/example", flush=True)

    if todo:
        cost = scorer.cost(len(todo), seconds, tokens)
        with open(out.with_suffix(".cost.json"), "w", encoding="utf-8") as handle:
            json.dump(dataclasses.asdict(cost), handle, indent=2)
        print(f"Done: {cost.seconds_per_example:.3f}s/example on {cost.device}, "
              f"peak {cost.peak_memory_gb:.1f} GB -> {out}")
    else:
        print(f"Nothing to do; {out} is complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
