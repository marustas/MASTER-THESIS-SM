"""
Step 11 — Cross-Strategy Evaluation.

Compares the three alignment experiments on three dimensions:

  1. Ranking consistency (Spearman ρ)
     Per-programme pairwise Spearman rank correlation between strategies,
     computed on the intersection of job ads each pair has in common.
     Pairs: symbolic × semantic, symbolic × hybrid, semantic × hybrid.

  2. Top-K overlap (Jaccard)
     Per-programme pairwise Jaccard similarity of top-K job-ad sets.
     J(A, B) = |A ∩ B| / |A ∪ B|.  Measured at k=5 and k=10.

  3. Top-1 agreement
     Whether all three strategies agree on the single best job per programme.
     Reports per-programme flag and overall agreement rate.

Score columns used:
  symbolic  → weighted_jaccard
  semantic  → cosine_combined
  hybrid    → hybrid_score

Input DataFrames are produced by Steps 8–10.  The module can also load
pre-computed parquet files from the experiments/results/ tree.

Output  (experiments/results/evaluation/):
  per_programme.parquet  — one row per programme_id with all metrics
  summary.json           — aggregate statistics across programmes

Usage:
    python -m src.evaluation.cross_strategy
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

from src.scraping.config import DATA_DIR

# ── Paths ──────────────────────────────────────────────────────────────────────

RESULTS_DIR = DATA_DIR.parent / "experiments" / "results"
SYMBOLIC_RANKINGS = RESULTS_DIR / "exp1_symbolic" / "rankings.parquet"
SEMANTIC_RANKINGS = RESULTS_DIR / "exp2_semantic" / "rankings.parquet"
HYBRID_RANKINGS   = RESULTS_DIR / "exp3_hybrid"   / "rankings.parquet"
EVAL_DIR          = RESULTS_DIR / "evaluation"

_SCORE = {
    "symbolic": "weighted_jaccard",
    "semantic": "cosine_combined",
    "hybrid":   "hybrid_score",
}

_PAIRS = [
    ("symbolic", "semantic"),
    ("symbolic", "hybrid"),
    ("semantic", "hybrid"),
]


# ── Per-metric helpers ─────────────────────────────────────────────────────────

def _spearman_pair(
    scores_a: pd.Series,
    scores_b: pd.Series,
    job_ids_a: pd.Series,
    job_ids_b: pd.Series,
) -> float:
    """
    Spearman ρ on the intersection of job IDs between two strategy rankings.
    Returns NaN when fewer than 3 shared jobs exist.
    """
    merged = (
        pd.DataFrame({"job_id": job_ids_a, "score_a": scores_a.values})
        .merge(
            pd.DataFrame({"job_id": job_ids_b, "score_b": scores_b.values}),
            on="job_id",
        )
    )
    if len(merged) < 3:
        return float("nan")
    rho, _ = spearmanr(merged["score_a"], merged["score_b"])
    return float(rho)


def _top_k_jaccard(
    job_ids_a: pd.Series,
    job_ids_b: pd.Series,
    k: int,
) -> float:
    """Jaccard similarity of top-k job-ID sets from two strategy rankings."""
    set_a = set(job_ids_a.head(k))
    set_b = set(job_ids_b.head(k))
    union = set_a | set_b
    if not union:
        return float("nan")
    return len(set_a & set_b) / len(union)


def _top1_job(job_ids: pd.Series) -> int | None:
    """Return the top-ranked job_id or None if empty."""
    return int(job_ids.iloc[0]) if len(job_ids) > 0 else None


# ── Core evaluation ────────────────────────────────────────────────────────────

def evaluate(
    symbolic: pd.DataFrame,
    semantic: pd.DataFrame,
    hybrid: pd.DataFrame,
    top_k: int = 10,
) -> tuple[pd.DataFrame, dict]:
    """
    Compare the three strategy rankings across all programmes.

    Parameters
    ----------
    symbolic : rankings from align_symbolic (needs programme_id, job_id,
               weighted_jaccard); sorted desc by score within each programme.
    semantic : rankings from align_semantic (needs programme_id, job_id,
               cosine_combined).
    hybrid   : rankings from align_hybrid (needs programme_id, job_id,
               hybrid_score).
    top_k    : k used for Jaccard overlap (also evaluated at k//2).

    Returns
    -------
    per_programme : pd.DataFrame
        One row per programme with Spearman ρ, Jaccard@k/2, Jaccard@k,
        top-1 job per strategy, and top-1 agreement flag.
    summary : dict
        Aggregate statistics across programmes.
    """
    rankings = {
        "symbolic": symbolic.sort_values(
            ["programme_id", "weighted_jaccard"], ascending=[True, False]
        ),
        "semantic": semantic.sort_values(
            ["programme_id", "cosine_combined"], ascending=[True, False]
        ),
        "hybrid": hybrid.sort_values(
            ["programme_id", "hybrid_score"], ascending=[True, False]
        ),
    }

    programme_ids = sorted(
        set(symbolic["programme_id"]) & set(semantic["programme_id"])
    )
    logger.info(
        f"Evaluating {len(programme_ids)} programmes across 3 strategies "
        f"(top_k={top_k})"
    )

    k_half = max(1, top_k // 2)
    records = []

    for p_id in programme_ids:
        row: dict = {"programme_id": p_id}

        groups = {
            name: df[df["programme_id"] == p_id]
            for name, df in rankings.items()
        }
        scores = {
            name: groups[name][_SCORE[name]]
            for name in _SCORE
        }
        jobs = {
            name: groups[name]["job_id"].reset_index(drop=True)
            for name in _SCORE
        }

        # Spearman ρ
        for a, b in _PAIRS:
            key = f"spearman_{a[:3]}_{b[:3]}"
            row[key] = _spearman_pair(
                scores[a], scores[b], jobs[a], jobs[b]
            )

        # Top-K Jaccard
        for a, b in _PAIRS:
            prefix = f"jaccard_{a[:3]}_{b[:3]}"
            row[f"{prefix}_at_{k_half}"] = _top_k_jaccard(jobs[a], jobs[b], k_half)
            row[f"{prefix}_at_{top_k}"]  = _top_k_jaccard(jobs[a], jobs[b], top_k)

        # Top-1 per strategy
        for name in _SCORE:
            row[f"top1_{name}"] = _top1_job(jobs[name])

        # Top-1 agreement: all three agree on the best job
        top1_vals = [row[f"top1_{n}"] for n in _SCORE if row[f"top1_{n}"] is not None]
        row["top1_all_agree"] = len(set(top1_vals)) == 1 if top1_vals else False

        records.append(row)

    per_programme = pd.DataFrame(records)

    summary = _summarise(per_programme, top_k, k_half)
    logger.info(
        f"  top-1 agreement rate: "
        f"{summary['top1_agreement_rate']:.1%}  "
        f"({summary['top1_agreements']}/{len(programme_ids)} programmes)"
    )
    return per_programme, summary


def _summarise(per_programme: pd.DataFrame, top_k: int, k_half: int) -> dict:
    def _stats(col: str) -> dict | None:
        s = per_programme[col].dropna()
        if s.empty:
            return None
        return {
            "mean":   float(s.mean()),
            "median": float(s.median()),
            "std":    float(s.std()),
            "min":    float(s.min()),
            "max":    float(s.max()),
        }

    n = len(per_programme)
    n_agree = int(per_programme["top1_all_agree"].sum()) if "top1_all_agree" in per_programme.columns else 0

    summary: dict = {
        "n_programmes": n,
        "top_k": top_k,
        "top1_agreements": n_agree,
        "top1_agreement_rate": n_agree / n if n > 0 else 0.0,
        "spearman": {},
        "jaccard": {},
    }

    for a, b in _PAIRS:
        key = f"{a[:3]}_{b[:3]}"
        summary["spearman"][key] = _stats(f"spearman_{key}")
        summary["jaccard"][f"{key}_at_{k_half}"] = _stats(f"jaccard_{key}_at_{k_half}")
        summary["jaccard"][f"{key}_at_{top_k}"]  = _stats(f"jaccard_{key}_at_{top_k}")

    return summary


# ── Pipeline entry point ───────────────────────────────────────────────────────

def run_evaluation(
    symbolic_path: Path = SYMBOLIC_RANKINGS,
    semantic_path: Path = SEMANTIC_RANKINGS,
    hybrid_path:   Path = HYBRID_RANKINGS,
    output_dir:    Path = EVAL_DIR,
    top_k: int = 10,
) -> None:
    """Load experiment rankings, run cross-strategy evaluation, persist results."""
    logger.info("Loading experiment rankings…")
    symbolic = pd.read_parquet(symbolic_path)
    semantic = pd.read_parquet(semantic_path)
    hybrid   = pd.read_parquet(hybrid_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    per_programme, summary = evaluate(symbolic, semantic, hybrid, top_k=top_k)

    pp_path = output_dir / "per_programme.parquet"
    summary_path = output_dir / "summary.json"

    per_programme.to_parquet(pp_path, index=False)
    logger.info(f"Per-programme metrics → {pp_path}")

    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"Summary → {summary_path}")


if __name__ == "__main__":
    run_evaluation()
