"""
Step 17 — Consensus-Based IR Metrics.

Without ground-truth relevance labels, uses cross-strategy agreement
as a proxy: a job is "relevant" for a programme if it appears in the
top-K of at least ``min_strategies`` strategies (default 2 of 3).

Metrics computed per strategy against the consensus relevance set:
  - Precision@K  — fraction of top-K results that are relevant
  - NDCG@K       — normalised discounted cumulative gain
  - MRR          — mean reciprocal rank of first relevant result
  - Coverage@K   — fraction of programmes with ≥1 relevant result in top-K

Output  (experiments/results/evaluation/):
  ir_metrics.json — per-strategy metrics and aggregate summary

Usage:
    python -m src.evaluation.ir_metrics
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.scraping.config import DATA_DIR

# ── Paths ──────────────────────────────────────────────────────────────────────

RESULTS_DIR = DATA_DIR.parent / "experiments" / "results"
SYMBOLIC_RANKINGS = RESULTS_DIR / "exp1_symbolic" / "rankings.parquet"
SEMANTIC_RANKINGS = RESULTS_DIR / "exp2_semantic" / "rankings.parquet"
HYBRID_RANKINGS = RESULTS_DIR / "exp3_hybrid" / "rankings.parquet"
EVAL_DIR = RESULTS_DIR / "evaluation"

_SCORE_COL = {
    "symbolic": "weighted_jaccard",
    "semantic": "cosine_combined",
    "hybrid": "hybrid_score",
}


# ── Consensus relevance set ──────────────────────────────────────────────────

def build_consensus(
    rankings: dict[str, pd.DataFrame],
    k: int,
    min_strategies: int = 2,
) -> dict[int, set[int]]:
    """
    Build pseudo-relevance set per programme.

    A job is relevant if it appears in the top-K of at least
    ``min_strategies`` strategies.

    Returns {programme_id: {relevant_job_ids}}.
    """
    from collections import Counter

    programme_ids = set()
    for df in rankings.values():
        programme_ids |= set(df["programme_id"].unique())

    relevance: dict[int, set[int]] = {}
    for p_id in sorted(programme_ids):
        counts: Counter = Counter()
        for name, df in rankings.items():
            score_col = _SCORE_COL[name]
            p_df = df[df["programme_id"] == p_id]
            top_jobs = set(p_df.nlargest(k, score_col)["job_id"])
            counts.update(top_jobs)
        relevance[p_id] = {
            job_id for job_id, cnt in counts.items() if cnt >= min_strategies
        }

    return relevance


# ── IR metric helpers ─────────────────────────────────────────────────────────

def precision_at_k(ranked_jobs: list[int], relevant: set[int], k: int) -> float:
    """Fraction of top-K results that are relevant."""
    top_k = ranked_jobs[:k]
    if not top_k:
        return 0.0
    return sum(1 for j in top_k if j in relevant) / len(top_k)


def dcg_at_k(ranked_jobs: list[int], relevant: set[int], k: int) -> float:
    """Discounted cumulative gain at K."""
    dcg = 0.0
    for i, job_id in enumerate(ranked_jobs[:k]):
        rel = 1.0 if job_id in relevant else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1)=0
    return dcg


def ndcg_at_k(ranked_jobs: list[int], relevant: set[int], k: int) -> float:
    """Normalised DCG at K."""
    dcg = dcg_at_k(ranked_jobs, relevant, k)
    # Ideal DCG: all relevant items at the top
    ideal_n = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def reciprocal_rank(ranked_jobs: list[int], relevant: set[int]) -> float:
    """Reciprocal rank of the first relevant result."""
    for i, job_id in enumerate(ranked_jobs):
        if job_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


# ── Per-strategy evaluation ───────────────────────────────────────────────────

def evaluate_strategy(
    rankings_df: pd.DataFrame,
    score_col: str,
    relevance: dict[int, set[int]],
    k: int,
) -> dict:
    """
    Compute IR metrics for one strategy against the consensus relevance set.

    Returns dict with per-programme and aggregate metrics.
    """
    precisions = []
    ndcgs = []
    rrs = []
    has_relevant = 0

    for p_id in sorted(relevance.keys()):
        p_df = rankings_df[rankings_df["programme_id"] == p_id]
        if p_df.empty:
            continue

        ranked_jobs = p_df.nlargest(k, score_col)["job_id"].tolist()
        rel = relevance[p_id]

        p_at_k = precision_at_k(ranked_jobs, rel, k)
        n_at_k = ndcg_at_k(ranked_jobs, rel, k)
        rr = reciprocal_rank(ranked_jobs, rel)

        precisions.append(p_at_k)
        ndcgs.append(n_at_k)
        rrs.append(rr)

        if any(j in rel for j in ranked_jobs):
            has_relevant += 1

    n = len(precisions)
    return {
        "precision_at_k": round(float(np.mean(precisions)), 4) if precisions else 0.0,
        "ndcg_at_k": round(float(np.mean(ndcgs)), 4) if ndcgs else 0.0,
        "mrr": round(float(np.mean(rrs)), 4) if rrs else 0.0,
        "coverage_at_k": round(has_relevant / n, 4) if n > 0 else 0.0,
        "n_programmes": n,
    }


# ── Core evaluation ───────────────────────────────────────────────────────────

def compute_ir_metrics(
    symbolic: pd.DataFrame,
    semantic: pd.DataFrame,
    hybrid: pd.DataFrame,
    k: int = 10,
    min_strategies: int = 2,
) -> dict:
    """
    Compute consensus-based IR metrics for all three strategies.

    Parameters
    ----------
    symbolic, semantic, hybrid : ranking DataFrames from Steps 8–10.
    k              : top-K cutoff for relevance and metrics.
    min_strategies : minimum strategies a job must appear in to be relevant.

    Returns dict with per-strategy metrics and consensus stats.
    """
    rankings = {"symbolic": symbolic, "semantic": semantic, "hybrid": hybrid}

    logger.info(f"Building consensus relevance set (k={k}, min_strategies={min_strategies})…")
    relevance = build_consensus(rankings, k=k, min_strategies=min_strategies)

    # Stats on the consensus set
    rel_sizes = [len(v) for v in relevance.values()]
    consensus_stats = {
        "mean_relevant_per_programme": round(float(np.mean(rel_sizes)), 2),
        "median_relevant_per_programme": float(np.median(rel_sizes)),
        "programmes_with_zero_relevant": sum(1 for s in rel_sizes if s == 0),
    }

    logger.info(
        f"  Consensus: {consensus_stats['mean_relevant_per_programme']:.1f} "
        f"relevant jobs/programme (avg)"
    )

    result: dict = {
        "parameters": {"k": k, "min_strategies": min_strategies},
        "consensus": consensus_stats,
        "strategies": {},
    }

    for name, df in rankings.items():
        score_col = _SCORE_COL[name]
        metrics = evaluate_strategy(df, score_col, relevance, k)
        result["strategies"][name] = metrics
        logger.info(
            f"  {name}: P@{k}={metrics['precision_at_k']:.3f}  "
            f"NDCG@{k}={metrics['ndcg_at_k']:.3f}  "
            f"MRR={metrics['mrr']:.3f}  "
            f"Cov@{k}={metrics['coverage_at_k']:.3f}"
        )

    return result


# ── Pipeline entry point ───────────────────────────────────────────────────────

def run_ir_metrics(
    symbolic_path: Path = SYMBOLIC_RANKINGS,
    semantic_path: Path = SEMANTIC_RANKINGS,
    hybrid_path: Path = HYBRID_RANKINGS,
    output_dir: Path = EVAL_DIR,
    k: int = 10,
    min_strategies: int = 2,
) -> None:
    """Load rankings, compute IR metrics, persist results."""
    logger.info("Loading rankings for IR metrics…")
    symbolic = pd.read_parquet(symbolic_path)
    semantic = pd.read_parquet(semantic_path)
    hybrid = pd.read_parquet(hybrid_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    result = compute_ir_metrics(symbolic, semantic, hybrid, k=k, min_strategies=min_strategies)

    output_path = output_dir / "ir_metrics.json"
    with open(output_path, "w") as fh:
        json.dump(result, fh, indent=2)
    logger.info(f"IR metrics → {output_path}")


if __name__ == "__main__":
    run_ir_metrics()
