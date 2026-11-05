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

Then fetch the data (see *Data* below):

```bash
make data
```

## Reproducing a run

```bash
# 1. Every model on both cuts, plus prompt variants (~3.5 h on an M5 laptop; resumable).
make experiments

# 2. Every table, figure and number in the report, from the cache (no GPU, ~20 s).
make analyze
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

## Results

Test cut, three SummaC datasets, thresholds tuned on the validation cut. Balanced
accuracy (BA) is averaged over datasets, with a stratified bootstrap 95% CI. Cost is
measured on an Apple M5 laptop (MPS, bf16).

| Model | s / 1,000 checks | Macro BA [95% CI] | BA at default 0.5 cut-off | AUC |
| --- | ---: | --- | ---: | ---: |
| Qwen2.5-0.5B | 80 | 0.594 [0.561, 0.623] | 0.512 | 0.634 |
| Qwen2.5-1.5B | 205 | 0.663 [0.634, 0.692] | 0.599 | 0.717 |
| Qwen2.5-3B | 399 | 0.680 [0.653, 0.711] | 0.578 | 0.755 |
| Qwen2.5-7B | 1,097 | 0.674 [0.646, 0.704] | 0.637 | 0.747 |

- **Accuracy stops improving after 1.5B–3B.** 7B vs 3B: −0.5 points (paired
  p = 0.61) at 2.75× the compute; 7B vs 1.5B: +1.1 (p = 0.25) at 5.4×.
- **Different sizes make different mistakes.** The 7B fixes 27.8% of the 0.5B's test
  answers and breaks 10.7%. On FRANK it gains on entity, predicate and circumstance
  errors (+16–17 points) and loses on coreference errors (0.51 → 0.35).
- **Routing:** an oracle escalating the right 25% of inputs from 0.5B to 7B reaches 0.81.
  A plain confidence cascade gets within one point of the 7B at 60% escalation (33%
  compute saved); the learned router does not beat it.
- **Prompt wording matters as much as size.** On a fixed 400-example subsample, two
  alternative wordings beat the main prompt at every size (AUC +0.05 to +0.15, all 95%
  CIs above zero), and the gain shrinks as models grow. Under the better wordings the
  size curve is flatter still: the 7B is again no better than the 3B. The main prompt
  was fixed in the proposal; choosing a wording on test data would contaminate it.
- **Threshold tuning** is worth +8.2 points for the 0.5B model, as much as moving to 7B.
- **Polytope labels:** 22% of its "inconsistent" test summaries carry only omission or
  duplication errors, no factual error.

Every table, figure and number comes from `make analyze`, which reads the cached
predictions in `results/predictions/`. Those files are released so new routing rules
can be tested without running a model.

| Output | What it holds |
| --- | --- |
| `results/summary.json` | Thresholds, paired tests, outcome breakdowns, cascade summaries, router weights |
| `results/tables/` | One CSV per research question, plus prompt sensitivity |
| `results/figures/` | Size vs. accuracy, cascade curves, FRANK error types |

## Team

Shayan Sooratgar, Batuhan Iliev
