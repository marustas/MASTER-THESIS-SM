"""
Step 19 — BM25 Baseline.

TF-IDF/BM25 text retrieval baseline over ``cleaned_text``.
Ranks job ads per programme by BM25 score, providing a simple
lexical baseline for comparison with the skill-based (symbolic),
embedding-based (semantic), and hybrid alignment strategies.

Uses the ``rank_bm25`` library (Okapi BM25) for scoring.

The programme's ``cleaned_text`` is treated as the query, the job ad's
``cleaned_text`` as the document corpus.

Output  (experiments/results/exp0_bm25/):
  rankings.parquet  — all (programme, job) pairs with bm25_score;
                      sorted by (programme_id asc, bm25_score desc)
  summary.json      — aggregate statistics

Usage:
    python -m src.alignment.bm25_baseline
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from rank_bm25 import BM25Okapi

from src.scraping.config import DATA_DIR

# ── Paths ──────────────────────────────────────────────────────────────────────

DATASET_PATH = DATA_DIR / "dataset" / "dataset.parquet"
RESULTS_DIR = DATA_DIR.parent / "experiments" / "results"


# ── Tokenisation ─────────────────────────────────────────────────────────────

def tokenise(text: str) -> list[str]:
    """Simple whitespace tokenisation on lowered text."""
    if not isinstance(text, str):
        return []
    return text.lower().split()


# ── Alignment ─────────────────────────────────────────────────────────────────

def align_bm25(
    df: pd.DataFrame,
    k1: float = 1.5,
    b: float = 0.75,
) -> pd.DataFrame:
    """
    Compute BM25 alignment scores for all programmes × job ads.

    Parameters
    ----------
    df : unified dataset with ``source_type`` and ``cleaned_text``.
    k1, b : BM25 hyperparameters.

    Returns
    -------
    rankings : pd.DataFrame
        Columns: programme_id, job_id, programme_name, job_title, bm25_score.
        Sorted by (programme_id asc, bm25_score desc).
    """
    programmes = df[df["source_type"] == "programme"].reset_index(drop=True)
    jobs = df[df["source_type"] == "job_ad"].reset_index(drop=True)

    n_prog = len(programmes)
    n_jobs = len(jobs)

    logger.info(
        f"BM25 baseline: {n_prog} programmes × {n_jobs} job ads "
        f"({n_prog * n_jobs:,} pairs, k1={k1}, b={b})"
    )

    # Build BM25 index over job corpus
    job_tokens = [tokenise(t) for t in jobs["cleaned_text"]]
    bm25 = BM25Okapi(job_tokens, k1=k1, b=b)

    # Query with each programme
    prog_tokens = [tokenise(t) for t in programmes["cleaned_text"]]

    has_prog_name = "name" in programmes.columns
    has_job_title = "job_title" in jobs.columns

    records = []
    for p_id in range(n_prog):
        p_name = programmes.at[p_id, "name"] if has_prog_name else str(p_id)
        scores = bm25.get_scores(prog_tokens[p_id])
        for j_id in range(n_jobs):
            j_title = jobs.at[j_id, "job_title"] if has_job_title else str(j_id)
            records.append({
                "programme_id": p_id,
                "job_id": j_id,
                "programme_name": p_name,
                "job_title": j_title,
                "bm25_score": round(float(scores[j_id]), 6),
            })

    rankings = (
        pd.DataFrame(records)
        .sort_values(["programme_id", "bm25_score"], ascending=[True, False])
        .reset_index(drop=True)
    )

    logger.info(f"  → {len(rankings):,} pairs scored")
    return rankings


# ── Summary ────────────────────────────────────────────────────────────────────

def _compute_summary(rankings: pd.DataFrame, top_n: int) -> dict:
    top = rankings.groupby("programme_id").head(top_n)
    return {
        "n_programmes": int(rankings["programme_id"].nunique()),
        "n_jobs": int(rankings["job_id"].nunique()),
        "n_pairs": int(len(rankings)),
        "top_n": top_n,
        "bm25_score_all": {
            "mean": float(rankings["bm25_score"].mean()),
            "median": float(rankings["bm25_score"].median()),
            "max": float(rankings["bm25_score"].max()),
        },
        "bm25_score_top_n": {
            "mean": float(top["bm25_score"].mean()),
            "median": float(top["bm25_score"].median()),
        },
    }


# ── Pipeline entry point ─────────────────────────────────────────────────────

def run_bm25_alignment(
    dataset_path: Path = DATASET_PATH,
    output_dir: Path = RESULTS_DIR / "exp0_bm25",
    top_n: int = 20,
    k1: float = 1.5,
    b: float = 0.75,
) -> None:
    """Load dataset, run BM25 alignment, persist results."""
    logger.info(f"Loading dataset from {dataset_path}…")
    df = pd.read_parquet(dataset_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    rankings = align_bm25(df, k1=k1, b=b)

    rankings_path = output_dir / "rankings.parquet"
    summary_path = output_dir / "summary.json"

    rankings.to_parquet(rankings_path, index=False)
    logger.info(f"Rankings → {rankings_path}  ({len(rankings):,} pairs)")

    summary = _compute_summary(rankings, top_n)
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"Summary → {summary_path}")


if __name__ == "__main__":
    run_bm25_alignment()
