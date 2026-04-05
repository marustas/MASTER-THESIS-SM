"""
Step 10 — Experiment 3: Hybrid Alignment.

Two-stage pipeline that balances recall with transparency:

  Stage 1 — Retrieval (semantic):
      For each programme, select the top ``semantic_top_n`` candidate job ads
      ranked by cosine similarity on the combined ``embedding`` column.
      Semantic search provides high recall at low cost.

  Stage 2 — Refinement (symbolic):
      Re-score each candidate with programme_recall over ESCO skill-URI
      sets — the fraction of job-demanded skill weight the programme covers.
      This is asymmetric by design: it answers "how much of what the job
      requires does the programme provide?"

  Hybrid score (per-programme min-max normalised):
      hybrid_score = α · norm(cosine) + (1 − α) · norm(programme_recall)

      Both components are min-max normalised within each programme's
      candidate set to ensure they contribute equally regardless of their
      raw scale.

      α (``alpha``) defaults to 0.5.  Varying α between 0 and 1 gives a
      continuum from fully symbolic to fully semantic.

Output  (experiments/results/exp3_hybrid/):
  rankings.parquet  — top-``semantic_top_n`` job ads per programme with
                      cosine_score, programme_recall, hybrid_score columns;
                      sorted by (programme_id asc, hybrid_score desc)
  summary.json      — aggregate statistics and parameter record

Usage:
    python -m src.alignment.hybrid
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from src.alignment.semantic import align_semantic
from src.alignment.symbolic import align_symbolic_weighted
from src.scraping.config import DATA_DIR

# ── Paths ──────────────────────────────────────────────────────────────────────

DATASET_PATH = DATA_DIR / "dataset" / "dataset.parquet"
RESULTS_DIR = DATA_DIR.parent / "experiments" / "results"


# ── Core ───────────────────────────────────────────────────────────────────────

def align_hybrid(
    df: pd.DataFrame,
    semantic_top_n: int = 50,
    alpha: float = 0.5,
) -> pd.DataFrame:
    """
    Two-stage hybrid alignment for all programmes × job ads.

    Parameters
    ----------
    df             : unified dataset with ``source_type``, embedding columns,
                     and ``skill_details``.
    semantic_top_n : number of candidates retrieved per programme in Stage 1.
    alpha          : weight of cosine_score in the hybrid formula
                     (1 − alpha applied to programme_recall).

    Returns
    -------
    rankings : pd.DataFrame
        Columns: programme_id, job_id, programme_name, job_title,
                 cosine_score, programme_recall, hybrid_score.
        One row per (programme, candidate) pair — at most
        ``semantic_top_n`` rows per programme.
        Sorted by (programme_id asc, hybrid_score desc).
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    n_prog = (df["source_type"] == "programme").sum()
    n_jobs = (df["source_type"] == "job_ad").sum()
    logger.info(
        f"Hybrid alignment: {n_prog} programmes × {n_jobs} job ads  "
        f"(semantic_top_n={semantic_top_n}, alpha={alpha})"
    )

    # ── Stage 1: semantic retrieval ────────────────────────────────────────────
    logger.info("Stage 1: semantic retrieval…")
    sem = align_semantic(df)[["programme_id", "job_id", "programme_name",
                               "job_title", "cosine_combined"]]
    sem = sem.rename(columns={"cosine_combined": "cosine_score"})

    candidates = (
        sem.sort_values(["programme_id", "cosine_score"], ascending=[True, False])
        .groupby("programme_id", sort=False)
        .head(semantic_top_n)
        .reset_index(drop=True)
    )

    # ── Stage 2: symbolic refinement (IDF-weighted programme recall) ────────
    logger.info("Stage 2: symbolic refinement (IDF-weighted programme_recall)…")
    sym, _ = align_symbolic_weighted(df, top_n=semantic_top_n)
    sym = sym[["programme_id", "job_id", "programme_recall"]]

    merged = candidates.merge(sym, on=["programme_id", "job_id"], how="left")
    merged["programme_recall"] = merged["programme_recall"].fillna(0.0)

    # ── Per-programme min-max normalisation ────────────────────────────────────
    def _minmax(series: pd.Series) -> pd.Series:
        lo, hi = series.min(), series.max()
        return (series - lo) / (hi - lo) if hi > lo else pd.Series(0.0, index=series.index)

    merged["cosine_norm"] = merged.groupby("programme_id")["cosine_score"].transform(_minmax)
    merged["recall_norm"] = merged.groupby("programme_id")["programme_recall"].transform(_minmax)

    # ── Hybrid score ───────────────────────────────────────────────────────────
    merged["hybrid_score"] = (
        alpha * merged["cosine_norm"] + (1.0 - alpha) * merged["recall_norm"]
    )

    rankings = (
        merged.drop(columns=["cosine_norm", "recall_norm"])
        .sort_values(["programme_id", "hybrid_score"], ascending=[True, False])
        .reset_index(drop=True)
    )

    logger.info(
        f"  → {len(rankings):,} candidate pairs "
        f"(avg {len(rankings) / max(n_prog, 1):.1f} per programme)"
    )
    return rankings


# ── Summary ────────────────────────────────────────────────────────────────────

def _compute_summary(rankings: pd.DataFrame, semantic_top_n: int, alpha: float) -> dict:
    return {
        "parameters": {"semantic_top_n": semantic_top_n, "alpha": alpha},
        "n_programmes": int(rankings["programme_id"].nunique()),
        "n_jobs_total": int(rankings["job_id"].nunique()),
        "n_candidate_pairs": int(len(rankings)),
        "hybrid_score": {
            "mean":   float(rankings["hybrid_score"].mean()),
            "median": float(rankings["hybrid_score"].median()),
            "max":    float(rankings["hybrid_score"].max()),
        },
        "cosine_score": {
            "mean":   float(rankings["cosine_score"].mean()),
            "median": float(rankings["cosine_score"].median()),
        },
        "programme_recall": {
            "mean":   float(rankings["programme_recall"].mean()),
            "median": float(rankings["programme_recall"].median()),
        },
    }


# ── Pipeline entry point ───────────────────────────────────────────────────────

def run_hybrid_alignment(
    dataset_path: Path = DATASET_PATH,
    output_dir: Path = RESULTS_DIR / "exp3_hybrid",
    semantic_top_n: int = 50,
    alpha: float = 0.5,
) -> None:
    """Load dataset, run hybrid alignment, persist results."""
    logger.info(f"Loading dataset from {dataset_path}…")
    df = pd.read_parquet(dataset_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    rankings = align_hybrid(df, semantic_top_n=semantic_top_n, alpha=alpha)

    rankings_path = output_dir / "rankings.parquet"
    summary_path = output_dir / "summary.json"

    rankings.to_parquet(rankings_path, index=False)
    logger.info(f"Rankings → {rankings_path}  ({len(rankings):,} pairs)")

    summary = _compute_summary(rankings, semantic_top_n, alpha)
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"Summary → {summary_path}")


if __name__ == "__main__":
    run_hybrid_alignment()
