# When Is a Bigger Model Worth It?

Size, cost and accuracy in checking whether a summary is faithful to its source.

Course project for *Essentials in Text and Speech Processing* (UZH, HS 2026).

## The question

Systems that generate summaries increasingly verify each output against its source,
framed as natural language inference: does the document entail the summary? That check
runs on **every** output, so its cost adds up — and an LLM is usually the default choice.

We ask three things:

1. How does faithfulness-checking accuracy change with model size within one model family?
2. Is the size gap concentrated in identifiable error types, or spread evenly?
3. Can cheap signals decide, per input, when the large model is actually needed?

The claim is about the **cost–accuracy curve**, not a single score. Always-large is the
accuracy reference and always-small the cost reference; they bound the curve. The baselines
the learned router has to beat are a plain confidence threshold (a standard cascade) and
random escalation at an equal compute budget.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
make install
make test
```

Then fetch the data (do this first — see *Data risks* below):

```bash
make data
```

## Reproducing a run

```bash
# 1. Cache predictions, one model at a time (GPU).
python scripts/run_inference.py --model Qwen/Qwen2.5-0.5B-Instruct --cut val
python scripts/run_inference.py --model Qwen/Qwen2.5-7B-Instruct  --cut val --load-in-8bit
# ... repeat for test, and for the two middle sizes

# 2. Build the curves from the cache (no GPU).
python scripts/analyze.py --cut test
```

Inference and analysis are separate on purpose: predictions are cached to
`results/predictions/`, so every analysis re-run is free, and a crash costs at most one
model's work. The cached predictions are what we release with the report, so others can
test new routing policies without re-running a model.

## Layout

| Path | What lives there |
| --- | --- |
| `src/frugal_faith/config.py` | Every tunable: model ladder, datasets, budgets, seed |
| `src/frugal_faith/data.py` | SummaC loading, FRANK error typology |
| `src/frugal_faith/prompts.py` | Prompt templates and the two label words |
| `src/frugal_faith/scoring.py` | Logits → label, probability, confidence (pure numpy) |
| `src/frugal_faith/models.py` | Qwen inference and cost measurement (lazy torch import) |
| `src/frugal_faith/features.py` | Cheap router features |
| `src/frugal_faith/router.py` | Routing policies and their baselines |
| `src/frugal_faith/evaluate.py` | Balanced accuracy, bootstrap CIs, paired tests, cost–accuracy curves |
| `scripts/` | The three entry points: download, infer, analyse |

`scoring`, `features`, `router` and `evaluate` have no torch dependency, so the whole test
suite runs on a laptop in about a second.

## Data

We use the SummaC loaders from the `summac` package. As of November 2026, three of
the six SummaC datasets still download:

| Dataset | Validation | Test | Consistent | Status |
| --- | ---: | ---: | ---: | --- |
| XSumFaith | 1,250 | 1,250 | ~10% | works (XSum documents via the Hub's Parquet export) |
| Polytope | 634 | 634 | ~7% | works |
| FRANK | 671 | 1,575 | ~33% | works; carries the error typology used for RQ2 |
| FactCC | | | | source bucket returns 403 |
| SummEval | | | | Google Drive file returns 404 |
| CoGenSumm | | | | host now sits behind a bot check; download by hand |

`scripts/download_data.py` reports per-dataset failures instead of crashing. To add a
dataset you downloaded by hand, put it where the summac loader expects it under
`data/summac/` and add its name to `SUMMAC_DATASETS` in `config.py`.

## Hardware

Everything runs on an Apple Silicon laptop (M5, 32 GB) through PyTorch's MPS backend in
bf16; the 7B model fits in unified memory. On CUDA machines the same code uses the GPU
automatically. Runs are resumable: predictions are appended after every batch.

## Status

Scaffold. Implemented and tested: scoring, features, routing policies, metrics,
bootstrap confidence intervals and paired significance tests, cost-accuracy curves.
Stubbed: `scripts/analyze.py` needs the prediction-alignment step once the first real
predictions exist. `data.attach_error_types` has an unverified join key — the download
script reports the match rate, and RQ2 depends on it.

## Team

Shayan Sooratgar, Batuhan Iliev
