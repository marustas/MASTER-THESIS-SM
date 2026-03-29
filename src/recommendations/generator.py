"""
Step 12 — Curriculum Recommendations.

Synthesises results from all three alignment experiments and the cross-strategy
evaluation into actionable curriculum recommendations:

  1. Best strategy selection — picks the alignment approach that agrees most
     with the other two strategies (highest mean Spearman ρ with peers).
  2. Per-programme job matches — top-N job ads per programme using the best
     strategy's ranking, annotated with top skill gaps from symbolic alignment.
  3. Market trends — skill URI frequency across job ads vs programme coverage
     to highlight underserved competencies.

Input:
  experiments/results/exp1_symbolic/rankings.parquet
  experiments/results/exp1_symbolic/skill_gaps.parquet
  experiments/results/exp2_semantic/rankings.parquet
  experiments/results/exp3_hybrid/rankings.parquet
  experiments/results/evaluation/summary.json
  data/dataset/dataset.parquet

Output:
  experiments/results/recommendations/programme_recommendations.parquet
  experiments/results/recommendations/market_trends.parquet
  experiments/results/recommendations/summary.json

Usage:
    python -m src.recommendations.generator
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.scraping.config import DATA_DIR

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

RESULTS_DIR       = Path("experiments/results")
SYMBOLIC_RANKINGS = RESULTS_DIR / "exp1_symbolic" / "rankings.parquet"
SYMBOLIC_GAPS     = RESULTS_DIR / "exp1_symbolic" / "skill_gaps.parquet"
SEMANTIC_RANKINGS = RESULTS_DIR / "exp2_semantic"  / "rankings.parquet"
HYBRID_RANKINGS   = RESULTS_DIR / "exp3_hybrid"    / "rankings.parquet"
EVAL_SUMMARY      = RESULTS_DIR / "evaluation"     / "summary.json"
DATASET_PATH      = DATA_DIR / "dataset" / "dataset.parquet"
OUTPUT_DIR        = RESULTS_DIR / "recommendations"

# Score column used for ranking within each strategy
_SCORE_COL: dict[str, str] = {
    "symbolic": "weighted_jaccard",
    "semantic": "cosine_combined",
    "hybrid":   "hybrid_score",
}


# ── Internal helpers ───────────────────────────────────────────────────────────

def _safe_mean(values: list[float]) -> float:
    """Mean of finite values; returns NaN if none are finite."""
    finite = [v for v in values if v == v]  # filter NaN via reflexivity
    return float(np.mean(finite)) if finite else float("nan")


def _best_strategy(eval_summary: dict) -> str:
    """
    Select the most consistently-performing alignment strategy.

    Each strategy's "centrality" score = mean Spearman ρ with its two peers.
    The strategy that agrees most with the others is the most stable choice.
    Falls back to "hybrid" when evaluation data is missing.
    """
    spearman = eval_summary.get("spearman", {})
    sym_sem = spearman.get("sym_sem", {}).get("mean", float("nan"))
    sym_hyb = spearman.get("sym_hyb", {}).get("mean", float("nan"))
    sem_hyb = spearman.get("sem_hyb", {}).get("mean", float("nan"))

    centrality: dict[str, float] = {
        "symbolic": _safe_mean([sym_sem, sym_hyb]),
        "semantic": _safe_mean([sym_sem, sem_hyb]),
        "hybrid":   _safe_mean([sym_hyb, sem_hyb]),
    }

    # All NaN → evaluation not yet run; default to hybrid
    if all(v != v for v in centrality.values()):
        return "hybrid"

    return max(centrality, key=lambda k: centrality[k] if centrality[k] == centrality[k] else -1.0)


def _top_gap_uris_per_programme(
    skill_gaps: pd.DataFrame,
    top_pairs: pd.DataFrame,
    n: int = 5,
) -> pd.DataFrame:
    """
    For each programme, return the top-n most frequent gap URIs across its
    top-N matched jobs.

    Parameters
    ----------
    skill_gaps : DataFrame with columns programme_id, job_id, gap_uri, gap_weight
    top_pairs  : DataFrame with columns programme_id, job_id (selected top-N)
    n          : how many top gap URIs to keep per programme

    Returns
    -------
    DataFrame with columns: programme_id, top_gap_uris (list[str])
    """
    relevant = skill_gaps.merge(top_pairs[["programme_id", "job_id"]], on=["programme_id", "job_id"])
    if relevant.empty:
        return pd.DataFrame(columns=["programme_id", "top_gap_uris"])

    top_per_prog = (
        relevant.groupby("programme_id")["gap_uri"]
        .apply(lambda uris: uris.value_counts().head(n).index.tolist())
        .reset_index()
        .rename(columns={"gap_uri": "top_gap_uris"})
    )
    return top_per_prog


def _market_trends(dataset: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """
    Compute skill URI demand frequency across job ads vs programme coverage.

    Rows are sorted by gap_index descending — skills most demanded by the
    market but least represented in programmes appear first.

    Columns: skill_uri, job_ad_count, programme_count, frequency,
             programme_coverage, gap_index
    """
    n_jobs      = int((dataset["source_type"] == "job_ad").sum())
    n_programmes = int((dataset["source_type"] == "programme").sum())

    def _uri_counts(subset: pd.DataFrame) -> pd.Series:
        return subset["skill_uris"].explode().dropna().value_counts()

    job_counts  = _uri_counts(dataset[dataset["source_type"] == "job_ad"])
    prog_counts = _uri_counts(dataset[dataset["source_type"] == "programme"])

    all_uris = job_counts.index.union(prog_counts.index)
    trends = pd.DataFrame({"skill_uri": all_uris})
    trends["job_ad_count"]       = trends["skill_uri"].map(job_counts).fillna(0).astype(int)
    trends["programme_count"]    = trends["skill_uri"].map(prog_counts).fillna(0).astype(int)
    trends["frequency"]          = trends["job_ad_count"]    / max(n_jobs, 1)
    trends["programme_coverage"] = trends["programme_count"] / max(n_programmes, 1)
    trends["gap_index"]          = trends["frequency"] - trends["programme_coverage"]

    return (
        trends
        .sort_values("gap_index", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_recommendations(
    dataset: pd.DataFrame,
    symbolic_rankings: pd.DataFrame,
    skill_gaps: pd.DataFrame,
    semantic_rankings: pd.DataFrame,
    hybrid_rankings: pd.DataFrame,
    eval_summary: dict,
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Generate curriculum alignment recommendations from experiment results.

    Parameters
    ----------
    dataset           : Unified dataset (source_type, skill_uris required).
    symbolic_rankings : Output of align_symbolic (programme_id, job_id,
                        weighted_jaccard, programme_name, job_title).
    skill_gaps        : Output of align_symbolic skill gaps
                        (programme_id, job_id, gap_uri, gap_weight).
    semantic_rankings : Output of align_semantic (programme_id, job_id,
                        cosine_combined, programme_name, job_title).
    hybrid_rankings   : Output of align_hybrid (programme_id, job_id,
                        hybrid_score, programme_name, job_title).
    eval_summary      : Dict from evaluation/summary.json.
    top_n             : Number of top job matches to recommend per programme.

    Returns
    -------
    programme_recommendations : DataFrame — top-N jobs per programme with
                                alignment score and top skill gaps.
    market_trends             : DataFrame — skill demand vs programme coverage.
    summary                   : Dict — metadata and aggregate findings.
    """
    strategy  = _best_strategy(eval_summary)
    score_col = _SCORE_COL[strategy]
    rankings_map = {
        "symbolic": symbolic_rankings,
        "semantic": semantic_rankings,
        "hybrid":   hybrid_rankings,
    }
    rankings = rankings_map[strategy]

    logger.info(f"Best strategy: {strategy}  (score column: {score_col})")

    # ── Top-N jobs per programme ───────────────────────────────────────────────
    top_jobs = (
        rankings
        .sort_values(["programme_id", score_col], ascending=[True, False])
        .groupby("programme_id", sort=False)
        .head(top_n)
        .copy()
    )
    top_jobs["rank"] = (
        top_jobs.groupby("programme_id")[score_col]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    top_jobs["strategy"] = strategy
    top_jobs = top_jobs.rename(columns={score_col: "alignment_score"})

    keep_cols = [
        "programme_id", "programme_name", "strategy", "rank",
        "job_id", "job_title", "alignment_score",
    ]
    top_jobs = top_jobs[[c for c in keep_cols if c in top_jobs.columns]]

    # ── Skill gaps for selected pairs ─────────────────────────────────────────
    gap_df = _top_gap_uris_per_programme(
        skill_gaps, top_jobs[["programme_id", "job_id"]], n=5
    )
    top_jobs = top_jobs.merge(gap_df, on="programme_id", how="left")
    top_jobs["top_gap_uris"] = top_jobs["top_gap_uris"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # ── Market trends ─────────────────────────────────────────────────────────
    trends = _market_trends(dataset)

    # ── Summary dict ──────────────────────────────────────────────────────────
    spearman_means = {
        k: v.get("mean") for k, v in eval_summary.get("spearman", {}).items()
    }
    summary: dict = {
        "best_strategy":       strategy,
        "n_programmes":        int(top_jobs["programme_id"].nunique()),
        "n_job_ads":           int((dataset["source_type"] == "job_ad").sum()),
        "top_n":               top_n,
        "top1_agreement_rate": eval_summary.get("top1_agreement_rate"),
        "spearman_means":      spearman_means,
        "n_market_trend_skills": len(trends),
        "top_market_skills":   trends.head(10)["skill_uri"].tolist(),
    }

    return top_jobs, trends, summary


# ── Pipeline entry-point ───────────────────────────────────────────────────────

def run_recommendations(
    dataset_path: Path = DATASET_PATH,
    symbolic_rankings_path: Path = SYMBOLIC_RANKINGS,
    symbolic_gaps_path: Path = SYMBOLIC_GAPS,
    semantic_rankings_path: Path = SEMANTIC_RANKINGS,
    hybrid_rankings_path: Path = HYBRID_RANKINGS,
    eval_summary_path: Path = EVAL_SUMMARY,
    output_dir: Path = OUTPUT_DIR,
    top_n: int = 10,
) -> None:
    logger.info("Loading alignment experiment results…")
    dataset           = pd.read_parquet(dataset_path)
    symbolic_rankings = pd.read_parquet(symbolic_rankings_path)
    skill_gaps        = pd.read_parquet(symbolic_gaps_path)
    semantic_rankings = pd.read_parquet(semantic_rankings_path)
    hybrid_rankings   = pd.read_parquet(hybrid_rankings_path)
    with open(eval_summary_path, encoding="utf-8") as f:
        eval_summary = json.load(f)

    recs, trends, summary = generate_recommendations(
        dataset=dataset,
        symbolic_rankings=symbolic_rankings,
        skill_gaps=skill_gaps,
        semantic_rankings=semantic_rankings,
        hybrid_rankings=hybrid_rankings,
        eval_summary=eval_summary,
        top_n=top_n,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    recs.to_parquet(output_dir / "programme_recommendations.parquet", index=False)
    trends.to_parquet(output_dir / "market_trends.parquet", index=False)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Step 12 complete — {summary['n_programmes']} programmes, "
        f"strategy: {summary['best_strategy']}, "
        f"top-{top_n} jobs recommended per programme"
    )
    logger.info(f"Results → {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    run_recommendations()
