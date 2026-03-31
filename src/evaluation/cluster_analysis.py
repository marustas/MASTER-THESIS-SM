"""
Step 18 — Cluster-Stratified Alignment Analysis.

Examines how alignment scores and skill gaps vary across programme and
job-ad clusters:

  1. Programme-cluster × job-cluster contingency table + chi-squared test
     to check whether top matches cluster non-randomly.
  2. Per-cluster alignment score distributions (all 3 strategies).
  3. Cluster-specific skill gaps — which specialisations have the largest
     market mismatch.
  4. Strategy performance by cluster — does symbolic/semantic/hybrid vary
     by programme specialisation.

Input:
  - dataset.parquet (for cluster labels)
  - Rankings from Steps 8–10

Output  (experiments/results/evaluation/):
  cluster_analysis.json     — aggregate results
  cluster_analysis.parquet  — per-programme cluster-level metrics

Usage:
    python -m src.evaluation.cluster_analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import chi2_contingency

from src.scraping.config import DATA_DIR

# ── Paths ──────────────────────────────────────────────────────────────────────

RESULTS_DIR = DATA_DIR.parent / "experiments" / "results"
DATASET_PATH = DATA_DIR / "dataset" / "dataset.parquet"
SYMBOLIC_RANKINGS = RESULTS_DIR / "exp1_symbolic" / "rankings.parquet"
SEMANTIC_RANKINGS = RESULTS_DIR / "exp2_semantic" / "rankings.parquet"
HYBRID_RANKINGS = RESULTS_DIR / "exp3_hybrid" / "rankings.parquet"
SKILL_GAPS = RESULTS_DIR / "exp1_symbolic" / "skill_gaps.parquet"
EVAL_DIR = RESULTS_DIR / "evaluation"

_SCORE_COL = {
    "symbolic": "weighted_jaccard",
    "semantic": "cosine_combined",
    "hybrid": "hybrid_score",
}


# ── Contingency analysis ─────────────────────────────────────────────────────

def contingency_test(
    dataset: pd.DataFrame,
    rankings: pd.DataFrame,
    score_col: str,
    top_n: int = 5,
) -> dict:
    """
    Build programme-cluster × job-cluster contingency table from top-N
    matches and run chi-squared test.

    Returns dict with contingency_table, chi2, p_value, dof, cramers_v.
    """
    progs = dataset[dataset["source_type"] == "programme"][["cluster_label"]].copy()
    progs["programme_id"] = range(len(progs))

    jobs = dataset[dataset["source_type"] == "job_ad"][["cluster_label"]].copy()
    jobs["job_id"] = range(len(jobs))

    # Take top-N matches per programme
    top_matches = (
        rankings.sort_values(["programme_id", score_col], ascending=[True, False])
        .groupby("programme_id", sort=False)
        .head(top_n)
    )

    merged = (
        top_matches.merge(
            progs.rename(columns={"cluster_label": "prog_cluster"}),
            on="programme_id",
        )
        .merge(
            jobs.rename(columns={"cluster_label": "job_cluster"}),
            on="job_id",
        )
    )

    if merged.empty:
        return {"contingency_table": {}, "chi2": None, "p_value": None, "dof": None, "cramers_v": None}

    ct = pd.crosstab(merged["prog_cluster"], merged["job_cluster"])

    try:
        chi2, p, dof, _ = chi2_contingency(ct)
        n_obs = ct.values.sum()
        k = min(ct.shape) - 1
        cramers_v = float(np.sqrt(chi2 / (n_obs * k))) if k > 0 and n_obs > 0 else 0.0
    except ValueError:
        chi2, p, dof, cramers_v = None, None, None, None

    return {
        "contingency_table": ct.to_dict(),
        "chi2": round(float(chi2), 4) if chi2 is not None else None,
        "p_value": round(float(p), 6) if p is not None else None,
        "dof": int(dof) if dof is not None else None,
        "cramers_v": round(float(cramers_v), 4) if cramers_v is not None else None,
    }


# ── Per-cluster score distributions ──────────────────────────────────────────

def per_cluster_scores(
    dataset: pd.DataFrame,
    rankings: dict[str, pd.DataFrame],
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Compute mean alignment score per programme cluster for each strategy.

    Returns DataFrame: programme_id, prog_cluster, and mean score columns.
    """
    progs = dataset[dataset["source_type"] == "programme"][["cluster_label"]].copy()
    progs["programme_id"] = range(len(progs))

    records = []
    for p_id in sorted(progs["programme_id"]):
        row = {
            "programme_id": p_id,
            "prog_cluster": int(progs.loc[progs["programme_id"] == p_id, "cluster_label"].iloc[0]),
        }
        for name, df in rankings.items():
            score_col = _SCORE_COL[name]
            p_df = df[df["programme_id"] == p_id]
            top = p_df.nlargest(top_n, score_col)[score_col]
            row[f"mean_{name}"] = round(float(top.mean()), 4) if len(top) > 0 else 0.0
        records.append(row)

    return pd.DataFrame(records)


def cluster_score_summary(per_cluster: pd.DataFrame) -> dict:
    """Aggregate per-cluster mean scores."""
    result = {}
    for cluster in sorted(per_cluster["prog_cluster"].unique()):
        subset = per_cluster[per_cluster["prog_cluster"] == cluster]
        entry = {"n_programmes": len(subset)}
        for col in [c for c in subset.columns if c.startswith("mean_")]:
            entry[col] = round(float(subset[col].mean()), 4)
        result[int(cluster)] = entry
    return result


# ── Cluster-specific skill gaps ───────────────────────────────────────────────

def cluster_skill_gaps(
    dataset: pd.DataFrame,
    skill_gaps: pd.DataFrame,
    top_n_gaps: int = 10,
) -> dict:
    """
    Aggregate skill gaps by programme cluster.

    Returns {cluster_id: [(gap_uri, count), ...]} — top N most frequent
    gap URIs per cluster.
    """
    from collections import Counter

    progs = dataset[dataset["source_type"] == "programme"][["cluster_label"]].copy()
    progs["programme_id"] = range(len(progs))

    merged = skill_gaps.merge(progs, on="programme_id")

    result = {}
    for cluster in sorted(merged["cluster_label"].unique()):
        sub = merged[merged["cluster_label"] == cluster]
        counts = Counter(sub["gap_uri"])
        result[int(cluster)] = counts.most_common(top_n_gaps)
    return result


# ── Best strategy per cluster ─────────────────────────────────────────────────

def best_strategy_per_cluster(per_cluster: pd.DataFrame) -> dict:
    """For each cluster, which strategy has the highest mean top-N score."""
    result = {}
    score_cols = [c for c in per_cluster.columns if c.startswith("mean_")]
    for cluster in sorted(per_cluster["prog_cluster"].unique()):
        subset = per_cluster[per_cluster["prog_cluster"] == cluster]
        means = {col: float(subset[col].mean()) for col in score_cols}
        best = max(means, key=means.get)
        result[int(cluster)] = {
            "best_strategy": best.replace("mean_", ""),
            "scores": {k.replace("mean_", ""): round(v, 4) for k, v in means.items()},
        }
    return result


# ── Core analysis ─────────────────────────────────────────────────────────────

def compute_cluster_analysis(
    dataset: pd.DataFrame,
    symbolic: pd.DataFrame,
    semantic: pd.DataFrame,
    hybrid: pd.DataFrame,
    skill_gaps: pd.DataFrame | None = None,
    top_n: int = 10,
) -> tuple[pd.DataFrame, dict]:
    """
    Run full cluster-stratified analysis.

    Returns (per_cluster_df, summary_dict).
    """
    rankings = {"symbolic": symbolic, "semantic": semantic, "hybrid": hybrid}

    logger.info("Computing per-cluster alignment scores…")
    per_cluster = per_cluster_scores(dataset, rankings, top_n=top_n)

    logger.info("Running contingency test (programme × job clusters)…")
    ct_result = contingency_test(dataset, hybrid, _SCORE_COL["hybrid"], top_n=top_n)

    logger.info("Computing cluster score summary…")
    score_summary = cluster_score_summary(per_cluster)

    logger.info("Identifying best strategy per cluster…")
    best_per_cluster = best_strategy_per_cluster(per_cluster)

    summary: dict = {
        "n_programme_clusters": int(per_cluster["prog_cluster"].nunique()),
        "top_n": top_n,
        "contingency_test": ct_result,
        "cluster_scores": score_summary,
        "best_strategy_per_cluster": best_per_cluster,
    }

    if skill_gaps is not None and not skill_gaps.empty:
        logger.info("Aggregating cluster-specific skill gaps…")
        gaps = cluster_skill_gaps(dataset, skill_gaps)
        summary["cluster_skill_gaps"] = gaps

    return per_cluster, summary


# ── Pipeline entry point ───────────────────────────────────────────────────────

def run_cluster_analysis(
    dataset_path: Path = DATASET_PATH,
    symbolic_path: Path = SYMBOLIC_RANKINGS,
    semantic_path: Path = SEMANTIC_RANKINGS,
    hybrid_path: Path = HYBRID_RANKINGS,
    skill_gaps_path: Path = SKILL_GAPS,
    output_dir: Path = EVAL_DIR,
    top_n: int = 10,
) -> None:
    """Load data, run cluster analysis, persist results."""
    logger.info("Loading data for cluster analysis…")
    dataset = pd.read_parquet(dataset_path)
    symbolic = pd.read_parquet(symbolic_path)
    semantic = pd.read_parquet(semantic_path)
    hybrid = pd.read_parquet(hybrid_path)

    skill_gaps = None
    if skill_gaps_path.exists():
        skill_gaps = pd.read_parquet(skill_gaps_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    per_cluster, summary = compute_cluster_analysis(
        dataset, symbolic, semantic, hybrid, skill_gaps, top_n=top_n,
    )

    per_cluster.to_parquet(output_dir / "cluster_analysis.parquet", index=False)
    with open(output_dir / "cluster_analysis.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info(f"Cluster analysis → {output_dir}")


if __name__ == "__main__":
    run_cluster_analysis()
