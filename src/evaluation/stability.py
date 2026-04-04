"""
Step 21 — Bootstrap Ranking Stability.

Resample 80% of job ads *n_resamples* times. For each resample, re-run all
alignment strategies and compare the resampled rankings to the full-data
rankings via Kendall tau per programme. Reports mean, std, and 95% CI of
rank stability across programmes and resamples.

Output  (experiments/results/evaluation/):
  stability.json — per-strategy stability metrics

Usage:
    python -m src.evaluation.stability
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import kendalltau

from src.alignment.bm25_baseline import align_bm25
from src.alignment.hybrid import align_hybrid
from src.alignment.semantic import align_semantic
from src.alignment.symbolic import align_symbolic
from src.scraping.config import DATA_DIR

# ── Paths ──────────────────────────────────────────────────────────────────────

DATASET_PATH = DATA_DIR / "dataset" / "dataset.parquet"
RESULTS_DIR = DATA_DIR.parent / "experiments" / "results" / "evaluation"

# ── Strategy runners ───────────────────────────────────────────────────────────

STRATEGIES: dict[str, dict] = {
    "symbolic": {
        "fn": lambda df: align_symbolic(df, top_n=20)[0],
        "score_col": "weighted_jaccard",
    },
    "semantic": {
        "fn": lambda df: align_semantic(df, top_n=20),
        "score_col": "cosine_combined",
    },
    "hybrid": {
        "fn": lambda df: align_hybrid(df),
        "score_col": "hybrid_score",
    },
    "bm25": {
        "fn": lambda df: align_bm25(df),
        "score_col": "bm25_score",
    },
}


# ── Core logic ─────────────────────────────────────────────────────────────────


def _rank_vector(rankings: pd.DataFrame, programme_id, score_col: str) -> pd.Series:
    """Return a Series mapping job_id → rank for one programme, sorted by score desc."""
    mask = rankings["programme_id"] == programme_id
    prog_ranks = (
        rankings.loc[mask]
        .sort_values(score_col, ascending=False)
        .reset_index(drop=True)
    )
    return pd.Series(
        data=prog_ranks.index.values,
        index=prog_ranks["job_id"].values,
        name="rank",
    )


def compute_kendall_tau(
    full_rankings: pd.DataFrame,
    resampled_rankings: pd.DataFrame,
    programme_ids: list,
    score_col: str,
) -> list[float]:
    """Compute Kendall tau between full and resampled rankings per programme.

    Only job ads present in both rankings are compared.
    Returns one tau value per programme.
    """
    taus: list[float] = []
    for pid in programme_ids:
        full_ranks = _rank_vector(full_rankings, pid, score_col)
        resamp_ranks = _rank_vector(resampled_rankings, pid, score_col)

        common_jobs = full_ranks.index.intersection(resamp_ranks.index)
        if len(common_jobs) < 3:
            continue

        tau, _ = kendalltau(
            full_ranks.loc[common_jobs].values,
            resamp_ranks.loc[common_jobs].values,
        )
        if not np.isnan(tau):
            taus.append(float(tau))

    return taus


def run_stability(
    df: pd.DataFrame,
    n_resamples: int = 100,
    sample_fraction: float = 0.80,
    strategies: dict[str, dict] | None = None,
    seed: int = 42,
) -> dict[str, dict]:
    """Run bootstrap ranking stability analysis.

    Parameters
    ----------
    df:
        Unified dataset.
    n_resamples:
        Number of bootstrap resamples.
    sample_fraction:
        Fraction of job ads to include in each resample.
    strategies:
        Strategy definitions. Defaults to all four strategies.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict mapping strategy name → stability metrics.
    """
    if strategies is None:
        strategies = STRATEGIES

    rng = np.random.default_rng(seed)

    programmes = df[df["source_type"] == "programme"]
    jobs = df[df["source_type"] == "job_ad"]
    programme_ids = programmes.index.tolist()
    job_indices = jobs.index.tolist()
    n_sample = max(1, int(len(job_indices) * sample_fraction))

    logger.info(
        f"Stability analysis: {len(programme_ids)} programmes, "
        f"{len(job_indices)} jobs, {n_resamples} resamples ({n_sample} jobs each)"
    )

    # Full-data rankings per strategy
    full_rankings: dict[str, pd.DataFrame] = {}
    for name, spec in strategies.items():
        logger.info(f"Computing full rankings for {name}…")
        full_rankings[name] = spec["fn"](df)

    # Bootstrap resamples
    results: dict[str, dict] = {}
    for name, spec in strategies.items():
        all_taus: list[float] = []

        for i in range(n_resamples):
            sampled_jobs = rng.choice(job_indices, size=n_sample, replace=False)
            df_resample = pd.concat([programmes, df.loc[sampled_jobs]])

            resamp_rankings = spec["fn"](df_resample)

            taus = compute_kendall_tau(
                full_rankings[name],
                resamp_rankings,
                programme_ids,
                spec["score_col"],
            )
            all_taus.extend(taus)

            if (i + 1) % 25 == 0:
                logger.info(f"  {name}: {i + 1}/{n_resamples} resamples done")

        tau_arr = np.array(all_taus)
        ci_lo, ci_hi = np.percentile(tau_arr, [2.5, 97.5]) if len(tau_arr) > 0 else (0.0, 0.0)

        results[name] = {
            "mean_tau": float(np.mean(tau_arr)) if len(tau_arr) > 0 else 0.0,
            "std_tau": float(np.std(tau_arr)) if len(tau_arr) > 0 else 0.0,
            "median_tau": float(np.median(tau_arr)) if len(tau_arr) > 0 else 0.0,
            "ci_95_lower": float(ci_lo),
            "ci_95_upper": float(ci_hi),
            "n_tau_values": len(tau_arr),
            "n_resamples": n_resamples,
            "sample_fraction": sample_fraction,
        }

        logger.info(
            f"  {name}: mean_tau={results[name]['mean_tau']:.4f} "
            f"[{ci_lo:.4f}, {ci_hi:.4f}]"
        )

    return results


# ── Pipeline entry point ───────────────────────────────────────────────────────


def run(
    dataset_path: Path = DATASET_PATH,
    output_dir: Path = RESULTS_DIR,
    n_resamples: int = 100,
) -> None:
    """Load dataset, run stability analysis, persist results."""
    logger.info(f"Loading dataset from {dataset_path}")
    df = pd.read_parquet(dataset_path)

    results = run_stability(df, n_resamples=n_resamples)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "stability.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info(f"Stability results → {out_path}")


if __name__ == "__main__":
    run()
