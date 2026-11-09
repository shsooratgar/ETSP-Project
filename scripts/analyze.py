#!/usr/bin/env python3
"""Turn cached predictions into every table, figure and number in the report.

No GPU needed. Works with whatever models have finished: a model is included
once both its validation and test runs are complete.

Usage:
    python scripts/analyze.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from frugal_faith.analysis import (
    calibrate,
    cross_fitted_thresholds,
    detection_by_category,
    macro_balanced_accuracy,
    macro_bootstrap,
    outcome_breakdown,
    paired_auc_bootstrap,
    roc_auc,
    tune_threshold,
    within_group_terciles,
)
from frugal_faith.config import Config
from frugal_faith.data import FRANK_ERROR_GROUPS, FRANK_ERROR_NAMES, Example, load_summac
from frugal_faith.evaluate import (
    balanced_accuracy,
    bootstrap_ci,
    cost_accuracy_curve,
    reference_points,
)
from frugal_faith.features import extract_features, to_matrix
from frugal_faith.models import cache_path
from frugal_faith.router import (
    ConfidenceThreshold,
    LearnedRouter,
    OracleRouter,
    RandomEscalation,
)
from frugal_faith.scoring import Prediction

DATASET_NAMES = {"xsumfaith": "XSumFaith", "polytope": "Polytope", "frank": "FRANK"}
N_BOOT = 1000


def size_of(model: str) -> str:
    """'Qwen/Qwen2.5-0.5B-Instruct' -> '0.5B'."""
    return model.split("-")[1]


def load_predictions(path: Path) -> dict[str, Prediction]:
    frame = pd.read_csv(path)
    return {
        row.example_id: Prediction(
            example_id=row.example_id,
            model=row.model,
            label=int(row.label),
            prob_consistent=float(row.prob_consistent),
            confidence=float(row.confidence),
            entropy=float(row.entropy),
            input_tokens=int(row.input_tokens),
        )
        for row in frame.itertuples(index=False)
    }


class Run:
    """One model's calibrated predictions on both cuts, aligned to the examples."""

    def __init__(
        self,
        model: str,
        val_ex: list[Example],
        test_ex: list[Example],
        val_pred: dict[str, Prediction],
        test_pred: dict[str, Prediction],
        cost: dict[str, float],
    ) -> None:
        self.model = model
        self.size = size_of(model)
        self.cost = cost
        y_val = np.array([e.label for e in val_ex])
        g_val = np.array([e.dataset for e in val_ex])
        raw_val = [val_pred[e.example_id] for e in val_ex]
        raw_test = [test_pred[e.example_id] for e in test_ex]
        p_val = np.array([p.prob_consistent for p in raw_val])

        self.thresholds = {
            g: tune_threshold(p_val[g_val == g], y_val[g_val == g]) for g in np.unique(g_val)
        }
        self.val = [calibrate(p, self.thresholds[e.dataset]) for p, e in zip(raw_val, val_ex)]
        # For training the router: validation labels from thresholds tuned on
        # other folds, so they behave like test-time predictions.
        cross = np.empty(len(val_ex))
        for g in np.unique(g_val):
            mask = g_val == g
            cross[mask] = cross_fitted_thresholds(p_val[mask], y_val[mask], seed=0)
        self.val_cross = [calibrate(p, t) for p, t in zip(raw_val, cross)]
        self.val_cross_labels = np.array([p.label for p in self.val_cross])
        self.test = [calibrate(p, self.thresholds[e.dataset]) for p, e in zip(raw_test, test_ex)]
        self.val_labels = np.array([p.label for p in self.val])
        self.test_labels = np.array([p.label for p in self.test])
        self.test_raw_labels = np.array([p.label for p in raw_test])  # default 0.5 cut-off
        self.test_probs = np.array([p.prob_consistent for p in raw_test])

    @property
    def seconds_per_1k(self) -> float:
        return 1000.0 * self.cost["seconds_per_example"]

    @property
    def parameters(self) -> float:
        return self.cost["flops_per_example"] / (2.0 * self.cost["tokens_per_example"])


def load_runs(config: Config, val_ex: list[Example], test_ex: list[Example]) -> list[Run]:
    runs = []
    for model in config.models:
        paths = {cut: cache_path(config.results_dir, model, "direct", cut) for cut in ("val", "test")}
        if not all(p.exists() for p in paths.values()):
            print(f"  skip {size_of(model)}: not run on both cuts yet")
            continue
        val_pred, test_pred = load_predictions(paths["val"]), load_predictions(paths["test"])
        missing = sum(e.example_id not in val_pred for e in val_ex) + sum(
            e.example_id not in test_pred for e in test_ex
        )
        if missing:
            print(f"  skip {size_of(model)}: {missing} examples still to run")
            continue
        cost = json.loads(paths["test"].with_suffix(".cost.json").read_text())
        runs.append(Run(model, val_ex, test_ex, val_pred, test_pred, cost))
    return runs


def rq1(runs: list[Run], y: np.ndarray, groups: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Accuracy and cost per model size."""
    rows = []
    for run in runs:
        row = {
            "size": run.size,
            "parameters_b": run.parameters / 1e9,
            "seconds_per_1k": run.seconds_per_1k,
            "peak_memory_gb": run.cost["peak_memory_gb"],
        }
        macro = macro_bootstrap(y, run.test_labels, groups, n_resamples=N_BOOT)
        row.update(macro_ba=macro["value"], macro_low=macro["ci_low"], macro_high=macro["ci_high"])
        row["macro_ba_default_threshold"] = macro_balanced_accuracy(y, run.test_raw_labels, groups)
        row["macro_auc"] = float(
            np.mean([roc_auc(y[groups == g], run.test_probs[groups == g]) for g in np.unique(groups)])
        )
        for g in np.unique(groups):
            mask = groups == g
            low, high = bootstrap_ci(y[mask], run.test_labels[mask], n_resamples=N_BOOT)
            row[f"{g}_ba"] = balanced_accuracy(y[mask], run.test_labels[mask])
            row[f"{g}_low"], row[f"{g}_high"] = low, high
        rows.append(row)
    table = pd.DataFrame(rows)

    largest = runs[-1]
    comparisons = {}
    for run in runs[:-1]:
        test = macro_bootstrap(
            y, largest.test_labels, groups, pred_b=run.test_labels, n_resamples=N_BOOT
        )
        comparisons[f"{largest.size}_vs_{run.size}"] = {
            **test,
            **outcome_breakdown(y, run.test_labels, largest.test_labels),
        }
    return table, comparisons


def rq2(
    runs: list[Run], test_ex: list[Example], y: np.ndarray, groups: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Where the gap lives: FRANK error types and relative document length."""
    frank = groups == "frank"
    categories = np.array([e.error_type for e in test_ex])
    errors = []
    for run in runs:
        table = detection_by_category(y[frank], run.test_labels[frank], categories[frank])
        table["size"] = run.size
        errors.append(table)
        kept = float(np.mean(run.test_labels[frank & (y == 1)] == 1))
        errors.append(
            pd.DataFrame([{"category": "NoE", "n": int(np.sum(frank & (y == 1))), "detected": kept, "size": run.size}])
        )
    errors_df = pd.concat(errors, ignore_index=True)
    errors_df["name"] = errors_df["category"].map(FRANK_ERROR_NAMES).fillna("consistent (kept)")
    errors_df["group"] = errors_df["category"].map(FRANK_ERROR_GROUPS).fillna("-")

    lengths = np.array([len(e.document.split()) for e in test_ex])
    bins = within_group_terciles(lengths, groups)
    length_rows = []
    for run in runs:
        for b in ("short", "medium", "long"):
            mask = bins == b
            length_rows.append(
                {
                    "size": run.size,
                    "length": b,
                    "macro_ba": macro_balanced_accuracy(y[mask], run.test_labels[mask], groups[mask]),
                }
            )
    return errors_df, pd.DataFrame(length_rows)


def rq3(
    small: Run,
    large: Run,
    val_ex: list[Example],
    test_ex: list[Example],
    budgets: tuple[float, ...],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Cascade from ``small`` to ``large``: curves for every routing policy."""
    y_val = np.array([e.label for e in val_ex])
    y_test = np.array([e.label for e in test_ex])
    groups = np.array([e.dataset for e in test_ex])

    def metric(yt: np.ndarray, yp: np.ndarray) -> float:
        return macro_balanced_accuracy(yt, yp, groups)

    x_val = to_matrix([extract_features(e, p) for e, p in zip(val_ex, small.val_cross)])
    x_test = to_matrix([extract_features(e, p) for e, p in zip(test_ex, small.test)])
    improves_val = (small.val_cross_labels != y_val) & (large.val_cross_labels == y_val)
    improves_test = (small.test_labels != y_test) & (large.test_labels == y_test)

    learned = LearnedRouter(seed=seed).fit(x_val, improves_val.astype(int))
    args = dict(
        features=x_test,
        small_pred=small.test_labels,
        large_pred=large.test_labels,
        y_true=y_test,
        budgets=budgets,
        cost_small=small.seconds_per_1k,
        cost_large=large.seconds_per_1k,
        metric=metric,
    )
    curves = [
        cost_accuracy_curve(learned, **args),
        cost_accuracy_curve(ConfidenceThreshold(), **args),
        cost_accuracy_curve(OracleRouter(improves_test), **args),
    ]
    randoms = [cost_accuracy_curve(RandomEscalation(seed=s), **args) for s in range(20)]
    random_mean = randoms[0].copy()
    random_mean["balanced_accuracy"] = np.mean([r["balanced_accuracy"] for r in randoms], axis=0)
    curves.append(random_mean)
    frame = pd.concat(curves, ignore_index=True)
    frame["cascade"] = f"{small.size}->{large.size}"

    refs = reference_points(
        small.test_labels, large.test_labels, y_test, small.seconds_per_1k, large.seconds_per_1k, metric
    )
    large_ba = float(refs.set_index("policy").loc["always_large", "balanced_accuracy"])
    summary = {"always_large": large_ba, "always_small": float(refs.iloc[0]["balanced_accuracy"])}
    summary["fixable_share"] = float(improves_test.mean())
    summary["breakable_share"] = float(((small.test_labels == y_test) & (large.test_labels != y_test)).mean())
    for policy, curve in frame.groupby("policy"):
        match = curve[curve["balanced_accuracy"] >= large_ba - 0.01]
        first = match.iloc[0] if len(match) else None
        summary[policy] = {
            "ba_at_25pct": float(curve.loc[curve["budget"].sub(0.25).abs().idxmin(), "balanced_accuracy"]),
            "ba_at_50pct": float(curve.loc[curve["budget"].sub(0.5).abs().idxmin(), "balanced_accuracy"]),
            "best_ba": float(curve["balanced_accuracy"].max()),
            "budget_within_1pt": None if first is None else float(first["budget"]),
            "cost_within_1pt": None if first is None else float(first["cost_per_example"]),
            "saving_within_1pt": None if first is None else float(1 - first["cost_per_example"] / large.seconds_per_1k),
        }
    summary["router_coefficients"] = learned.coefficients()
    return frame, refs, summary


def prompt_sensitivity(config: Config, runs: list[Run], test_ex: list[Example]) -> pd.DataFrame:
    """Threshold-free AUC of each wording on the fixed 400-example subsample."""
    by_id = {e.example_id: e for e in test_ex}
    full = {run.model: dict(zip([e.example_id for e in test_ex], run.test_probs)) for run in runs}
    rows = []
    for model in config.models:
        for prompt in ("direct", "entailment", "fact_check"):
            path = cache_path(config.results_dir, model, prompt, "test")
            path = path.with_name(f"{path.stem}__sample400{path.suffix}")
            sample = pd.read_csv(path) if path.exists() else None
            if prompt == "direct":
                ref = cache_path(config.results_dir, model, "entailment", "test")
                ref = ref.with_name(f"{ref.stem}__sample400{ref.suffix}")
                if not ref.exists() or model not in full:
                    continue
                ids = pd.read_csv(ref)["example_id"]
                probs = np.array([full[model][i] for i in ids])
            elif sample is not None and len(sample) >= 400:
                ids, probs = sample["example_id"], sample["prob_consistent"].to_numpy()
            else:
                continue
            labels = np.array([by_id[i].label for i in ids])
            rows.append(
                {"size": size_of(model), "prompt": prompt, "auc": roc_auc(labels, probs),
                 "ids": list(ids), "probs": probs, "labels": labels}
            )
    # Compare each wording with the main one on the same examples.
    frame = pd.DataFrame(rows)
    for i, row in frame.iterrows():
        ref = frame[(frame["size"] == row["size"]) & (frame["prompt"] == "direct")]
        if row["prompt"] == "direct" or ref.empty:
            continue
        ref = ref.iloc[0]
        # Files list examples in batch order, which depends on prompt length,
        # so align this wording's scores to the reference order by id.
        score_of = dict(zip(row["ids"], row["probs"]))
        aligned = np.array([score_of[k] for k in ref["ids"]])
        test = paired_auc_bootstrap(ref["labels"], aligned, ref["probs"], n_resamples=N_BOOT)
        frame.loc[i, ["vs_direct", "vs_direct_low", "vs_direct_high"]] = (
            test["difference"], test["ci_low"], test["ci_high"]
        )
    return frame.drop(columns=["ids", "probs", "labels"])


def plot(
    results_dir: Path,
    rq1_table: pd.DataFrame,
    curves: pd.DataFrame,
    refs: pd.DataFrame,
    errors_df: pd.DataFrame,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = results_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.errorbar(
        rq1_table["seconds_per_1k"],
        rq1_table["macro_ba"],
        yerr=[rq1_table["macro_ba"] - rq1_table["macro_low"], rq1_table["macro_high"] - rq1_table["macro_ba"]],
        marker="o",
        capsize=3,
        color="black",
        label="macro average",
    )
    for g, name in DATASET_NAMES.items():
        if f"{g}_ba" in rq1_table:
            ax.plot(rq1_table["seconds_per_1k"], rq1_table[f"{g}_ba"], marker=".", alpha=0.6, label=name)
    for _, row in rq1_table.iterrows():
        ax.annotate(row["size"], (row["seconds_per_1k"], row["macro_ba"]), textcoords="offset points", xytext=(4, -10))
    ax.set_xscale("log")
    ax.set_xlabel("seconds per 1,000 checks (log)")
    ax.set_ylabel("balanced accuracy")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "size_vs_accuracy.png", dpi=200)
    plt.close(fig)

    main = curves[curves["cascade"] == curves["cascade"].iloc[0]]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for policy, curve in main.groupby("policy"):
        ax.plot(curve["cost_per_example"], curve["balanced_accuracy"], label=policy, linestyle="--" if policy == "oracle" else "-")
    for _, ref in refs.iterrows():
        ax.scatter(ref["cost_per_example"], ref["balanced_accuracy"], marker="s", zorder=3)
        ax.annotate(ref["policy"].replace("_", " "), (ref["cost_per_example"], ref["balanced_accuracy"]), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("seconds per 1,000 checks")
    ax.set_ylabel("balanced accuracy (macro)")
    ax.set_title(f"Cascade {main['cascade'].iloc[0]}", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "cascade.png", dpi=200)
    plt.close(fig)

    pivot = errors_df.pivot_table(index="name", columns="size", values="detected")
    sizes = [s for s in ("0.5B", "1.5B", "3B", "7B") if s in pivot.columns]
    ax = pivot[sizes].plot.bar(figsize=(6, 3.4), rot=30)
    ax.set_ylabel("share flagged / kept")
    ax.figure.tight_layout()
    ax.figure.savefig(out / "frank_error_types.png", dpi=200)
    plt.close(ax.figure)


def main() -> int:
    config = Config()
    print("Loading examples ...")
    val_ex, _ = load_summac("val", config.data_dir, config.datasets)
    test_ex, _ = load_summac("test", config.data_dir, config.datasets)
    y = np.array([e.label for e in test_ex])
    groups = np.array([e.dataset for e in test_ex])

    runs = load_runs(config, val_ex, test_ex)
    if len(runs) < 2:
        print("Need at least two complete models; run scripts/run_all.sh first.", file=sys.stderr)
        return 1
    print(f"Analysing {[r.size for r in runs]}")

    tables = config.results_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    summary: dict = {"models": [r.size for r in runs], "thresholds": {r.size: r.thresholds for r in runs}}

    rq1_table, comparisons = rq1(runs, y, groups)
    rq1_table.to_csv(tables / "rq1_size_accuracy_cost.csv", index=False)
    summary["rq1_comparisons"] = comparisons

    errors_df, length_df = rq2(runs, test_ex, y, groups)
    errors_df.to_csv(tables / "rq2_frank_error_types.csv", index=False)
    length_df.to_csv(tables / "rq2_document_length.csv", index=False)

    all_curves, all_refs = [], []
    summary["rq3"] = {}
    # Every cascade from a smaller into a larger model, so the practical choice
    # (route into the model where accuracy stops improving) is covered too.
    pairs = [(a, b) for i, a in enumerate(runs) for b in runs[i + 1 :]]
    for small, large in pairs:
        curves, refs, rq3_summary = rq3(small, large, val_ex, test_ex, config.budgets, config.seed)
        refs["cascade"] = curves["cascade"].iloc[0]
        all_curves.append(curves)
        all_refs.append(refs)
        summary["rq3"][curves["cascade"].iloc[0]] = rq3_summary
    curves_df = pd.concat(all_curves, ignore_index=True)
    curves_df.to_csv(tables / "rq3_cascade_curves.csv", index=False)
    pd.concat(all_refs).to_csv(tables / "rq3_reference_points.csv", index=False)

    prompts = prompt_sensitivity(config, runs, test_ex)
    prompts.to_csv(tables / "prompt_sensitivity.csv", index=False)

    with open(config.results_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    main_cascade = f"{runs[0].size}->{runs[-1].size}"
    main_refs = next(r for r in all_refs if r["cascade"].iloc[0] == main_cascade)
    plot(config.results_dir, rq1_table, curves_df[curves_df["cascade"] == main_cascade], main_refs, errors_df)

    pd.set_option("display.width", 160)
    print("\nRQ1\n", rq1_table[["size", "macro_ba", "macro_low", "macro_high", "macro_ba_default_threshold", "macro_auc", "seconds_per_1k"]].round(3).to_string(index=False))
    print(f"\nRQ3 ({main_cascade})\n", json.dumps(summary["rq3"][main_cascade], indent=1)[:1500])
    if len(prompts):
        print("\nPrompt sensitivity\n", prompts.pivot(index="size", columns="prompt", values="auc").round(3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
