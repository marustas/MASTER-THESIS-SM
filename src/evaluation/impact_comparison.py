"""Compare the impact of Step 27 (ESCO description embeddings) and Step 31
(programme-level IDF) on hybrid alignment results.

Runs four configurations:
  A) baseline       — no skill embeddings, no programme IDF (current pipeline)
  B) +desc_emb      — ESCO description embeddings for coherence boost
  C) +prog_idf      — programme-level IDF in symbolic refinement
  D) +both          — description embeddings + programme IDF

For each configuration, captures top-10 matches per programme and computes:
  - Top-1 / top-5 / top-10 diversity (unique jobs)
  - Score distributions (mean, median, max, CoV)
  - Top-1 match changes vs baseline
  - Generalist frequency (jobs appearing in top-K of many programmes)

Usage:
    python -m src.evaluation.impact_comparison
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.alignment.hybrid import align_hybrid
from src.scraping.config import DATA_DIR

DATASET_PATH = DATA_DIR / "dataset" / "dataset.parquet"
RESULTS_DIR = DATA_DIR.parent / "experiments" / "results"
OUTPUT_DIR = RESULTS_DIR / "impact_comparison"


# ── Metrics ───────────────────────────────────────────────────────────────────

def _compute_metrics(rankings: pd.DataFrame, top_k: int = 10) -> dict:
    """Compute summary metrics from hybrid rankings."""
    top = (
        rankings.sort_values(["programme_id", "hybrid_score"], ascending=[True, False])
        .groupby("programme_id")
        .head(top_k)
    )
    n_prog = top["programme_id"].nunique()

    top1 = top.groupby("programme_id").first().reset_index()
    top5 = top.groupby("programme_id").head(5)

    scores = top1["hybrid_score"]
    cov = float(scores.std() / scores.mean()) if scores.mean() > 0 else 0.0

    # Generalist frequency: jobs in top-K of > 3 programmes
    top_k_counts = top.groupby("job_id")["programme_id"].nunique()
    generalists_3 = int((top_k_counts > 3).sum())
    generalists_5 = int((top_k_counts > 5).sum())
    max_freq = int(top_k_counts.max()) if len(top_k_counts) > 0 else 0

    return {
        "n_programmes": int(n_prog),
        "top1_unique_jobs": int(top1["job_id"].nunique()),
        "top1_diversity": round(top1["job_id"].nunique() / n_prog, 3),
        "top5_unique_jobs": int(top5["job_id"].nunique()),
        "top5_diversity": round(top5["job_id"].nunique() / (n_prog * 5), 3),
        "top10_unique_jobs": int(top["job_id"].nunique()),
        "top10_diversity": round(top["job_id"].nunique() / (n_prog * top_k), 3),
        "top1_score_mean": round(float(scores.mean()), 4),
        "top1_score_median": round(float(scores.median()), 4),
        "top1_score_max": round(float(scores.max()), 4),
        "top1_score_cov": round(cov, 4),
        "generalists_gt3": generalists_3,
        "generalists_gt5": generalists_5,
        "max_top10_freq": max_freq,
    }


def _top1_changes(
    baseline: pd.DataFrame, variant: pd.DataFrame, label: str,
) -> dict:
    """Compare top-1 matches between baseline and variant."""
    def _get_top1(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.sort_values(["programme_id", "hybrid_score"], ascending=[True, False])
            .groupby("programme_id")
            .first()
            .reset_index()
        )

    b1 = _get_top1(baseline).set_index("programme_id")
    v1 = _get_top1(variant).set_index("programme_id")

    common = b1.index.intersection(v1.index)
    same = int((b1.loc[common, "job_id"] == v1.loc[common, "job_id"]).sum())
    changed = len(common) - same

    # Score delta for programmes that kept the same top-1 match
    same_mask = b1.loc[common, "job_id"] == v1.loc[common, "job_id"]
    if same_mask.any():
        same_ids = common[same_mask]
        score_deltas = v1.loc[same_ids, "hybrid_score"] - b1.loc[same_ids, "hybrid_score"]
        mean_delta = round(float(score_deltas.mean()), 4)
    else:
        mean_delta = 0.0

    # Score delta for ALL programmes (regardless of match change)
    all_deltas = v1.loc[common, "hybrid_score"] - b1.loc[common, "hybrid_score"]

    return {
        "variant": label,
        "n_programmes": len(common),
        "top1_same": same,
        "top1_changed": changed,
        "top1_change_pct": round(changed / len(common) * 100, 1),
        "mean_score_delta_same_match": mean_delta,
        "mean_score_delta_all": round(float(all_deltas.mean()), 4),
        "median_score_delta_all": round(float(all_deltas.median()), 4),
    }


def _top1_change_details(
    baseline: pd.DataFrame, variant: pd.DataFrame,
) -> pd.DataFrame:
    """Per-programme detail of top-1 changes."""
    def _get_top1(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.sort_values(["programme_id", "hybrid_score"], ascending=[True, False])
            .groupby("programme_id")
            .first()
            .reset_index()
        )

    b1 = _get_top1(baseline).set_index("programme_id")
    v1 = _get_top1(variant).set_index("programme_id")
    common = b1.index.intersection(v1.index)

    records = []
    for pid in common:
        changed = b1.at[pid, "job_id"] != v1.at[pid, "job_id"]
        records.append({
            "programme_id": pid,
            "programme_name": b1.at[pid, "programme_name"],
            "baseline_job": b1.at[pid, "job_title"],
            "baseline_score": round(float(b1.at[pid, "hybrid_score"]), 4),
            "variant_job": v1.at[pid, "job_title"],
            "variant_score": round(float(v1.at[pid, "hybrid_score"]), 4),
            "score_delta": round(
                float(v1.at[pid, "hybrid_score"] - b1.at[pid, "hybrid_score"]), 4
            ),
            "changed": changed,
        })
    return pd.DataFrame(records)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_comparison() -> dict:
    """Run all four configurations and return comparison results."""
    logger.info(f"Loading dataset from {DATASET_PATH}…")
    df = pd.read_parquet(DATASET_PATH)

    configs = {
        "A_baseline": {"use_programme_idf": False},
        "B_prog_idf": {"use_programme_idf": True},
    }

    results: dict[str, pd.DataFrame] = {}
    metrics: dict[str, dict] = {}

    for name, kwargs in configs.items():
        logger.info(f"\n{'='*60}\nRunning configuration: {name}\n{'='*60}")
        rankings = align_hybrid(df, **kwargs)
        results[name] = rankings
        metrics[name] = _compute_metrics(rankings)
        logger.info(f"  → {name}: top-1 diversity={metrics[name]['top1_diversity']}, "
                     f"score mean={metrics[name]['top1_score_mean']}")

    # ── Comparisons vs baseline ───────────────────────────────────────────
    baseline = results["A_baseline"]
    changes = {}
    details = {}
    for name in ["B_prog_idf"]:
        changes[name] = _top1_changes(baseline, results[name], name)
        details[name] = _top1_change_details(baseline, results[name])

    # ── Persist ───────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, rankings in results.items():
        rankings.to_parquet(OUTPUT_DIR / f"rankings_{name}.parquet", index=False)

    for name, df_detail in details.items():
        df_detail.to_csv(OUTPUT_DIR / f"top1_changes_{name}.csv", index=False)

    summary = {
        "metrics": metrics,
        "top1_changes": changes,
    }
    with open(OUTPUT_DIR / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print report ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("IMPACT COMPARISON — Programme IDF")
    print("=" * 80)

    header = f"{'Metric':<30} {'Baseline':>10} {'+ Prog IDF':>10}"
    print(f"\n{header}")
    print("-" * len(header))

    metric_keys = [
        ("top1_unique_jobs", "Top-1 unique jobs"),
        ("top1_diversity", "Top-1 diversity"),
        ("top5_unique_jobs", "Top-5 unique jobs"),
        ("top10_unique_jobs", "Top-10 unique jobs"),
        ("top1_score_mean", "Top-1 score mean"),
        ("top1_score_median", "Top-1 score median"),
        ("top1_score_max", "Top-1 score max"),
        ("top1_score_cov", "Top-1 score CoV"),
        ("generalists_gt3", "Generalists (>3 prog)"),
        ("generalists_gt5", "Generalists (>5 prog)"),
        ("max_top10_freq", "Max top-10 frequency"),
    ]

    config_keys = ["A_baseline", "B_prog_idf"]
    for key, label in metric_keys:
        vals = [str(metrics[c][key]) for c in config_keys]
        print(f"{label:<30} {vals[0]:>10} {vals[1]:>10}")

    print(f"\n{'Top-1 Changes vs Baseline':<30} {'':>10} {'+ Prog IDF':>10}")
    print("-" * 50)
    for key, label in [
        ("top1_changed", "Programmes changed"),
        ("top1_change_pct", "Change %"),
        ("mean_score_delta_all", "Mean score delta"),
    ]:
        vals = ["—"] + [str(changes[c][key]) for c in ["B_prog_idf"]]
        print(f"{label:<30} {vals[0]:>10} {vals[1]:>10}")

    print(f"\nResults saved to {OUTPUT_DIR}/")
    return summary


if __name__ == "__main__":
    run_comparison()
