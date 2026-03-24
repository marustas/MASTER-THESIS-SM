"""
Step 9 — Experiment 2: Semantic Text-Based Alignment.

Computes pairwise similarity between programme and job-ad embeddings using
two metrics:

  cosine_score  — cosine similarity (sklearn, re-normalises internally);
                  range [-1, 1]; magnitude-invariant.
  dot_score     — raw dot product on the stored vectors; for L2-normalised
                  embeddings this is numerically identical to cosine, which
                  is logged and confirms correct normalisation.

Alignment is run for three embedding variants:

  combined  — programme `embedding`  vs job `embedding`  (cleaned_text)
  brief     — programme `embedding_brief`  vs job `embedding`
  extended  — programme `embedding_extended` vs job `embedding`

All six scores (2 metrics × 3 variants) are stored as columns in a single
rankings.parquet so Step 11 can compare them directly.  Variants columns are
NaN for programmes that lack brief/extended descriptions.

Output  (experiments/results/exp2_semantic/):
  rankings.parquet  — all (programme, job) pairs with six score columns;
                      sorted by (programme_id asc, cosine_score desc)
  summary.json      — aggregate statistics per metric/variant

Usage:
    python -m src.alignment.semantic
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

from src.scraping.config import DATA_DIR

# ── Paths ──────────────────────────────────────────────────────────────────────

DATASET_PATH = DATA_DIR / "dataset" / "dataset.parquet"
RESULTS_DIR = DATA_DIR.parent / "experiments" / "results"

# ── Embedding extraction ───────────────────────────────────────────────────────

def _to_matrix(series: pd.Series) -> np.ndarray:
    """Stack a Series of embedding lists into a float32 matrix."""
    return np.array(series.tolist(), dtype=np.float32)


def _extract_embeddings(
    programmes: pd.DataFrame,
    jobs: pd.DataFrame,
) -> dict[str, np.ndarray | None]:
    """
    Return embedding matrices for all variants.

    Keys: prog_combined, prog_brief, prog_extended, job_combined.
    Values are None when the column is absent or entirely NaN.
    """
    def _safe(df: pd.DataFrame, col: str) -> np.ndarray | None:
        if col not in df.columns:
            return None
        valid = df[col].dropna()
        if valid.empty:
            return None
        # Build matrix aligned to df.index; fill missing rows with NaN
        mat = np.full((len(df), len(valid.iloc[0])), np.nan, dtype=np.float32)
        for i, (idx, val) in enumerate(valid.items()):
            pos = df.index.get_loc(idx)
            mat[pos] = val
        return mat

    return {
        "prog_combined": _safe(programmes, "embedding"),
        "prog_brief":    _safe(programmes, "embedding_brief"),
        "prog_extended": _safe(programmes, "embedding_extended"),
        "job_combined":  _safe(jobs, "embedding"),
    }


# ── Similarity computation ─────────────────────────────────────────────────────

def _pairwise_scores(
    prog_mat: np.ndarray,
    job_mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute cosine and dot-product similarity matrices.

    Parameters
    ----------
    prog_mat : (P, D) float32
    job_mat  : (J, D) float32

    Returns
    -------
    cosine_mat : (P, J) — sklearn cosine_similarity (re-normalises)
    dot_mat    : (P, J) — raw dot product
    """
    cosine_mat = cosine_similarity(prog_mat, job_mat)
    dot_mat = prog_mat @ job_mat.T
    return cosine_mat.astype(np.float32), dot_mat.astype(np.float32)


def _variant_scores(
    prog_mat: np.ndarray | None,
    job_mat: np.ndarray | None,
    n_prog: int,
    n_jobs: int,
    cosine_col: str,
    dot_col: str,
) -> dict[str, np.ndarray]:
    """
    Compute a score variant; returns flat arrays (length = n_prog * n_jobs).
    Returns NaN-filled arrays when either matrix is None.
    """
    n = n_prog * n_jobs
    nan = np.full(n, np.nan, dtype=np.float32)
    if prog_mat is None or job_mat is None:
        return {cosine_col: nan, dot_col: nan}

    # Filter to rows without NaN (partial coverage for brief/extended)
    prog_valid_mask = ~np.isnan(prog_mat).any(axis=1)
    job_valid_mask = ~np.isnan(job_mat).any(axis=1)

    cosine_flat = np.full(n, np.nan, dtype=np.float32)
    dot_flat = np.full(n, np.nan, dtype=np.float32)

    if prog_valid_mask.any() and job_valid_mask.any():
        p_idx = np.where(prog_valid_mask)[0]
        j_idx = np.where(job_valid_mask)[0]
        csim, dsim = _pairwise_scores(prog_mat[p_idx], job_mat[j_idx])
        for pi_local, pi_global in enumerate(p_idx):
            for ji_local, ji_global in enumerate(j_idx):
                flat_pos = pi_global * n_jobs + ji_global
                cosine_flat[flat_pos] = csim[pi_local, ji_local]
                dot_flat[flat_pos] = dsim[pi_local, ji_local]

    return {cosine_col: cosine_flat, dot_col: dot_flat}


# ── Main alignment ─────────────────────────────────────────────────────────────

def align_semantic(
    df: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Compute semantic alignment scores for all programmes × job ads.

    Parameters
    ----------
    df    : unified dataset with ``source_type`` and embedding columns.
    top_n : unused at this stage (kept for API symmetry with symbolic);
            Step 11 applies top-N filtering during evaluation.

    Returns
    -------
    rankings : pd.DataFrame
        Columns: programme_id, job_id, programme_name, job_title,
                 cosine_combined, dot_combined,
                 cosine_brief, dot_brief,
                 cosine_extended, dot_extended.
        Sorted by (programme_id asc, cosine_combined desc).
    """
    programmes = df[df["source_type"] == "programme"].reset_index(drop=False)
    jobs = df[df["source_type"] == "job_ad"].reset_index(drop=False)

    n_prog = len(programmes)
    n_jobs = len(jobs)

    logger.info(
        f"Semantic alignment: {n_prog} programmes × {n_jobs} job ads "
        f"({n_prog * n_jobs:,} pairs)"
    )

    mats = _extract_embeddings(
        programmes.set_index("index"),
        jobs.set_index("index"),
    )

    # ── Build flat index arrays ────────────────────────────────────────────────
    prog_ids = np.repeat(programmes["index"].values, n_jobs)
    job_ids = np.tile(jobs["index"].values, n_prog)

    has_prog_name = "name" in programmes.columns
    has_job_title = "job_title" in jobs.columns
    prog_names = np.repeat(
        programmes["name"].values if has_prog_name else programmes["index"].astype(str).values,
        n_jobs,
    )
    job_titles = np.tile(
        jobs["job_title"].values if has_job_title else jobs["index"].astype(str).values,
        n_prog,
    )

    # ── Score each variant ─────────────────────────────────────────────────────
    combined = _variant_scores(
        mats["prog_combined"], mats["job_combined"],
        n_prog, n_jobs, "cosine_combined", "dot_combined",
    )
    brief = _variant_scores(
        mats["prog_brief"], mats["job_combined"],
        n_prog, n_jobs, "cosine_brief", "dot_brief",
    )
    extended = _variant_scores(
        mats["prog_extended"], mats["job_combined"],
        n_prog, n_jobs, "cosine_extended", "dot_extended",
    )

    _log_cosine_dot_diff(combined["cosine_combined"], combined["dot_combined"])

    rankings = pd.DataFrame({
        "programme_id":   prog_ids,
        "job_id":         job_ids,
        "programme_name": prog_names,
        "job_title":      job_titles,
        **combined,
        **brief,
        **extended,
    })

    rankings.sort_values(
        ["programme_id", "cosine_combined"], ascending=[True, False], inplace=True,
    )
    rankings.reset_index(drop=True, inplace=True)

    logger.info(f"  → {len(rankings):,} pairs scored")
    return rankings


def _log_cosine_dot_diff(cosine: np.ndarray, dot: np.ndarray) -> None:
    """Warn if cosine and dot product diverge (indicates un-normalised embeddings)."""
    valid = ~(np.isnan(cosine) | np.isnan(dot))
    if not valid.any():
        return
    max_diff = float(np.abs(cosine[valid] - dot[valid]).max())
    if max_diff > 1e-4:
        logger.warning(
            f"cosine vs dot max divergence = {max_diff:.6f} — "
            "embeddings may not be L2-normalised"
        )
    else:
        logger.info(f"cosine ≈ dot (max diff {max_diff:.2e}) — embeddings confirmed L2-normalised")


# ── Summary ────────────────────────────────────────────────────────────────────

def _compute_summary(rankings: pd.DataFrame, top_n: int) -> dict:
    score_cols = [
        "cosine_combined", "dot_combined",
        "cosine_brief", "dot_brief",
        "cosine_extended", "dot_extended",
    ]
    top = rankings.groupby("programme_id").head(top_n)
    stats: dict = {
        "n_programmes": int(rankings["programme_id"].nunique()),
        "n_jobs": int(rankings["job_id"].nunique()),
        "n_pairs": int(len(rankings)),
        "top_n": top_n,
        "scores": {},
    }
    for col in score_cols:
        if col not in rankings.columns:
            continue
        valid = rankings[col].dropna()
        if valid.empty:
            stats["scores"][col] = None
            continue
        top_valid = top[col].dropna()
        stats["scores"][col] = {
            "all_mean":   float(valid.mean()),
            "all_median": float(valid.median()),
            "all_max":    float(valid.max()),
            "top_n_mean": float(top_valid.mean()) if not top_valid.empty else None,
        }
    return stats


# ── Pipeline entry point ───────────────────────────────────────────────────────

def run_semantic_alignment(
    dataset_path: Path = DATASET_PATH,
    output_dir: Path = RESULTS_DIR / "exp2_semantic",
    top_n: int = 20,
) -> None:
    """Load dataset, run semantic alignment, persist results."""
    logger.info(f"Loading dataset from {dataset_path}…")
    df = pd.read_parquet(dataset_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    rankings = align_semantic(df, top_n=top_n)

    rankings_path = output_dir / "rankings.parquet"
    summary_path = output_dir / "summary.json"

    rankings.to_parquet(rankings_path, index=False)
    logger.info(f"Rankings → {rankings_path}  ({len(rankings):,} pairs)")

    summary = _compute_summary(rankings, top_n)
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"Summary → {summary_path}")


if __name__ == "__main__":
    run_semantic_alignment()
