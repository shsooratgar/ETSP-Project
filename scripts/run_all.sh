#!/usr/bin/env bash
# The full experiment. Resumable: rerun after an interruption and every model
# picks up where it stopped.
#
# The smallest and largest models run first because the router needs both; the
# middle sizes complete the size curve for RQ1.
set -euo pipefail
PY="${PYTHON:-python}"

for size in 0.5B 7B 1.5B 3B; do
  for cut in val test; do
    echo "=== Qwen2.5-${size} / ${cut} ==="
    "$PY" scripts/run_inference.py --model "Qwen/Qwen2.5-${size}-Instruct" --cut "$cut"
  done
done

# Prompt sensitivity: two alternative wordings on a fixed random subsample of the
# test cut. The "direct" wording's scores on the same examples come from the
# full runs above.
for size in 0.5B 1.5B 3B; do
  for prompt in entailment fact_check; do
    echo "=== Qwen2.5-${size} / prompt ${prompt} ==="
    "$PY" scripts/run_inference.py --model "Qwen/Qwen2.5-${size}-Instruct" \
      --cut test --prompt "$prompt" --sample 400 --tag sample400
  done
done
echo "ALL DONE"
