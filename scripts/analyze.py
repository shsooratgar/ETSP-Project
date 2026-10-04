#!/usr/bin/env python3
"""Build the cost-accuracy curves from cached predictions. No GPU needed.

Usage:
    python scripts/analyze.py --cut test
"""

from __future__ import annotations

import argparse
import sys

from frugal_faith.config import Config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cut", default="test", choices=["val", "test"])
    parser.add_argument("--prompt", default="direct")
    args = parser.parse_args()

    _ = Config()
    print(
        "TODO: load cached predictions for config.small_model and config.large_model\n"
        "via models.cache_path(), align them on example_id, then:\n"
        "  features = to_matrix([extract_features(ex, small[ex.example_id]) ...])\n"
        "  router   = LearnedRouter(config.seed).fit(val_features, val_improves)\n"
        "  curves   = [cost_accuracy_curve(p, ...) for p in policies]\n"
        "Left explicit so the first real run forces a decision about alignment.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
