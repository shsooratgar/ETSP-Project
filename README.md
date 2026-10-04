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
| `src/frugal_faith/evaluate.py` | Balanced accuracy, cost–accuracy curves |
| `scripts/` | The three entry points: download, infer, analyse |

`scoring`, `features`, `router` and `evaluate` have no torch dependency, so the whole test
suite runs on a laptop in about a second.

## Data risks

All six SummaC datasets download automatically, but two are fragile:

- **XSumFaith** — the loader calls `load_dataset("xsum", trust_remote_code=True)`, which
  `datasets>=3.0` removed. Hence the `datasets<3.0` pin in `requirements.txt`. If that pin
  becomes unworkable, switch to a Parquet mirror such as `EdinburghNLP/xsum`.
- **SummEval** — fetched from Google Drive, which hits quota errors unpredictably. Keep a
  manual copy once it downloads successfully.

`scripts/download_data.py` reports per-dataset failures instead of crashing. Four of the
six datasets are enough for the project; drop anything that stays broken.

## Status

Scaffold. Implemented and tested: scoring, features, routing policies, metrics, curves.
Stubbed: `scripts/analyze.py` needs the prediction-alignment step once the first real
predictions exist. `data.attach_error_types` has an unverified join key — the download
script reports the match rate, and RQ2 depends on it.

## Team

Shayan Sooratgar, Mustafa [Surname]
