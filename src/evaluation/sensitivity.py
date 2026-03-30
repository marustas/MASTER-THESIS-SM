"""
Step 15 — Hybrid Alpha Sensitivity Analysis.

Sweeps alpha ∈ [0.0, 0.1, 0.2, … 1.0] for the hybrid alignment formula:

    hybrid_score = α · cosine_score + (1 − α) · weighted_jaccard

For each alpha value, recompute hybrid scores from the pre-computed
candidate pool (cosine_score + weighted_jaccard already stored in the
hybrid rankings parquet), re-rank within each programme, and evaluate
against the symbolic and semantic baselines.

Metrics computed per alpha:
  - hybrid_score mean/median/max
  - Spearman ρ with symbolic and semantic rankings (mean over programmes)
  - Jaccard@10 with symbolic and semantic rankings (mean over programmes)
  - Top-1 agreement rate across all 3 strategies

Output  (experiments/results/sensitivity/):
  alpha_sweep.parquet      — one row per alpha with all metrics
  alpha_sweep_summary.json — full results + optimal alpha selection

Usage:
    python -m src.evaluation.sensitivity
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
HYBRID_RANKINGS = RESULTS_DIR / "exp3_hybrid" / "rankings.parquet"
SYMBOLIC_RANKINGS = RESULTS_DIR / "exp1_symbolic" / "rankings.parquet"
SEMANTIC_RANKINGS = RESULTS_DIR / "exp2_semantic" / "rankings.parquet"
SENSITIVITY_DIR = RESULTS_DIR / "sensitivity"


# ── Core helpers ───────────────────────────────────────────────────────────────

def _rerank_hybrid(
    candidates: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    """Recompute hybrid_score and re-sort within each programme."""
    df = candidates.copy()
    df["hybrid_score"] = (
        alpha * df["cosine_score"] + (1.0 - alpha) * df["weighted_jaccard"]
    )
    return (
        df.sort_values(["programme_id", "hybrid_score"], ascending=[True, False])
        .reset_index(drop=True)
    )


def _mean_spearman(
    hybrid: pd.DataFrame,
    baseline: pd.DataFrame,
    hybrid_col: str,
    baseline_col: str,
) -> float:
    """Mean per-programme Spearman ρ between hybrid and a baseline strategy."""
    rhos = []
    for p_id in hybrid["programme_id"].unique():
        h = hybrid[hybrid["programme_id"] == p_id][["job_id", hybrid_col]]
        b = baseline[baseline["programme_id"] == p_id][["job_id", baseline_col]]
        merged = h.merge(b, on="job_id")
        if len(merged) < 3:
            continue
        rho, _ = spearmanr(merged[hybrid_col], merged[baseline_col])
        rhos.append(float(rho))
    return float(np.nanmean(rhos)) if rhos else float("nan")


def _mean_jaccard_at_k(
    hybrid: pd.DataFrame,
    baseline: pd.DataFrame,
    hybrid_col: str,
    baseline_col: str,
    k: int,
) -> float:
    """Mean per-programme Jaccard@k between hybrid and baseline top-k sets."""
    jaccards = []
    for p_id in hybrid["programme_id"].unique():
        h_jobs = set(
            hybrid[hybrid["programme_id"] == p_id]
            .nlargest(k, hybrid_col)["job_id"]
        )
        b_jobs = set(
            baseline[baseline["programme_id"] == p_id]
            .nlargest(k, baseline_col)["job_id"]
        )
        union = h_jobs | b_jobs
        if not union:
            continue
        jaccards.append(len(h_jobs & b_jobs) / len(union))
    return float(np.mean(jaccards)) if jaccards else float("nan")


def _top1_agreement_rate(
    hybrid: pd.DataFrame,
    symbolic: pd.DataFrame,
    semantic: pd.DataFrame,
) -> float:
    """Fraction of programmes where all 3 strategies agree on top-1 job."""
    n_agree = 0
    programme_ids = sorted(hybrid["programme_id"].unique())
    for p_id in programme_ids:
        t1_hyb = hybrid[hybrid["programme_id"] == p_id].iloc[0]["job_id"]
        sym_p = symbolic[symbolic["programme_id"] == p_id]
        sem_p = semantic[semantic["programme_id"] == p_id]
        if sym_p.empty or sem_p.empty:
            continue
        t1_sym = sym_p.nlargest(1, "weighted_jaccard").iloc[0]["job_id"]
        t1_sem = sem_p.nlargest(1, "cosine_combined").iloc[0]["job_id"]
        if t1_hyb == t1_sym == t1_sem:
            n_agree += 1
    return n_agree / len(programme_ids) if programme_ids else 0.0


# ── Sweep ──────────────────────────────────────────────────────────────────────

def alpha_sweep(
    hybrid_candidates: pd.DataFrame,
    symbolic: pd.DataFrame,
    semantic: pd.DataFrame,
    alphas: list[float] | None = None,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Sweep alpha values and compute evaluation metrics for each.

    Parameters
    ----------
    hybrid_candidates : hybrid rankings with cosine_score + weighted_jaccard
    symbolic          : symbolic rankings (needs weighted_jaccard)
    semantic          : semantic rankings (needs cosine_combined)
    alphas            : alpha values to sweep (default: 0.0 to 1.0, step 0.1)
    top_k             : k for Jaccard overlap metric

    Returns
    -------
    pd.DataFrame with columns: alpha, hybrid_mean, hybrid_median, hybrid_max,
        spearman_sym, spearman_sem, jaccard_sym_at_k, jaccard_sem_at_k,
        top1_agreement_rate
    """
    if alphas is None:
        alphas = [round(a * 0.1, 1) for a in range(11)]

    records = []
    for alpha in alphas:
        logger.info(f"  alpha={alpha:.1f}")
        reranked = _rerank_hybrid(hybrid_candidates, alpha)

        record = {
            "alpha": alpha,
            "hybrid_mean": float(reranked["hybrid_score"].mean()),
            "hybrid_median": float(reranked["hybrid_score"].median()),
            "hybrid_max": float(reranked["hybrid_score"].max()),
            "spearman_sym": _mean_spearman(
                reranked, symbolic, "hybrid_score", "weighted_jaccard",
            ),
            "spearman_sem": _mean_spearman(
                reranked, semantic, "hybrid_score", "cosine_combined",
            ),
            f"jaccard_sym_at_{top_k}": _mean_jaccard_at_k(
                reranked, symbolic, "hybrid_score", "weighted_jaccard", top_k,
            ),
            f"jaccard_sem_at_{top_k}": _mean_jaccard_at_k(
                reranked, semantic, "hybrid_score", "cosine_combined", top_k,
            ),
            "top1_agreement_rate": _top1_agreement_rate(
                reranked, symbolic, semantic,
            ),
        }
        records.append(record)

    return pd.DataFrame(records)


# ── Pipeline entry point ───────────────────────────────────────────────────────

def run_sensitivity(
    hybrid_path: Path = HYBRID_RANKINGS,
    symbolic_path: Path = SYMBOLIC_RANKINGS,
    semantic_path: Path = SEMANTIC_RANKINGS,
    output_dir: Path = SENSITIVITY_DIR,
    top_k: int = 10,
) -> None:
    """Load pre-computed rankings, sweep alpha, persist results."""
    logger.info("Loading rankings for sensitivity analysis…")
    hybrid = pd.read_parquet(hybrid_path)
    symbolic = pd.read_parquet(symbolic_path)
    semantic = pd.read_parquet(semantic_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running alpha sweep [0.0 → 1.0]…")
    results = alpha_sweep(hybrid, symbolic, semantic, top_k=top_k)

    # Identify optimal alpha (highest mean of spearman_sym + spearman_sem)
    results["spearman_mean"] = (
        results["spearman_sym"] + results["spearman_sem"]
    ) / 2.0
    best_row = results.loc[results["spearman_mean"].idxmax()]
    best_alpha = float(best_row["alpha"])

    results_path = output_dir / "alpha_sweep.parquet"
    summary_path = output_dir / "alpha_sweep_summary.json"

    results.to_parquet(results_path, index=False)
    logger.info(f"Results → {results_path}  ({len(results)} alpha values)")

    summary = {
        "best_alpha": best_alpha,
        "best_spearman_mean": float(best_row["spearman_mean"]),
        "top_k": top_k,
        "sweep": results.drop(columns=["spearman_mean"]).to_dict(orient="records"),
    }
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"Summary → {summary_path}  (best alpha={best_alpha})")


if __name__ == "__main__":
    run_sensitivity()
