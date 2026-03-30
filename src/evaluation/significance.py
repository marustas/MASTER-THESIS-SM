"""
Step 16 — Statistical Significance Testing.

Provides statistical rigour for cross-strategy evaluation results:

  1. Bootstrap confidence intervals (1000 resamples over programmes)
     on per-programme Spearman ρ and Jaccard@10.
  2. Wilcoxon signed-rank test on paired per-programme scores to test
     whether strategy differences are statistically significant.
  3. Effect sizes (rank-biserial correlation r = Z / √N).

Input: per_programme.parquet from Step 11 (cross-strategy evaluation).

Output  (experiments/results/evaluation/):
  significance.json — CIs, p-values, effect sizes per metric pair

Usage:
    python -m src.evaluation.significance
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import wilcoxon

from src.scraping.config import DATA_DIR

# ── Paths ──────────────────────────────────────────────────────────────────────

RESULTS_DIR = DATA_DIR.parent / "experiments" / "results"
PER_PROGRAMME = RESULTS_DIR / "evaluation" / "per_programme.parquet"
EVAL_DIR = RESULTS_DIR / "evaluation"


# ── Bootstrap CI ───────────────────────────────────────────────────────────────

def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Non-parametric bootstrap confidence interval on the mean.

    Returns dict with mean, ci_lower, ci_upper, std.
    """
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"mean": None, "ci_lower": None, "ci_upper": None, "std": None}

    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])

    alpha = (1.0 - ci) / 2.0
    return {
        "mean": round(float(np.mean(values)), 4),
        "ci_lower": round(float(np.percentile(boot_means, 100 * alpha)), 4),
        "ci_upper": round(float(np.percentile(boot_means, 100 * (1 - alpha))), 4),
        "std": round(float(np.std(values, ddof=1)), 4),
    }


# ── Wilcoxon signed-rank test ─────────────────────────────────────────────────

def paired_wilcoxon(
    a: np.ndarray,
    b: np.ndarray,
) -> dict:
    """
    Wilcoxon signed-rank test on paired observations.

    Returns dict with statistic, p_value, effect_size (rank-biserial r).
    """
    mask = ~(np.isnan(a) | np.isnan(b))
    a_clean, b_clean = a[mask], b[mask]
    diff = a_clean - b_clean

    # Remove zero differences (Wilcoxon requires non-zero)
    nonzero = diff != 0
    if nonzero.sum() < 3:
        return {"statistic": None, "p_value": None, "effect_size": None, "n": int(mask.sum())}

    stat, p_val = wilcoxon(diff[nonzero])
    n_eff = int(nonzero.sum())
    # Rank-biserial effect size: r = 1 - (2T / (n(n+1)/2))
    t_plus = float(stat)
    r = 1.0 - (2.0 * t_plus) / (n_eff * (n_eff + 1) / 2.0)

    return {
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_val), 6),
        "effect_size": round(float(r), 4),
        "n": n_eff,
    }


# ── Core analysis ─────────────────────────────────────────────────────────────

_SPEARMAN_PAIRS = [
    ("spearman_sym_sem", "spearman_sym_hyb"),
    ("spearman_sym_sem", "spearman_sem_hyb"),
    ("spearman_sym_hyb", "spearman_sem_hyb"),
]

_SPEARMAN_COLS = ["spearman_sym_sem", "spearman_sym_hyb", "spearman_sem_hyb"]


def compute_significance(
    per_programme: pd.DataFrame,
    n_boot: int = 1000,
    ci: float = 0.95,
) -> dict:
    """
    Compute bootstrap CIs, Wilcoxon tests, and effect sizes.

    Parameters
    ----------
    per_programme : DataFrame from Step 11 with per-programme Spearman ρ columns.
    n_boot        : number of bootstrap resamples.
    ci            : confidence level (0.95 = 95%).

    Returns
    -------
    dict with keys: bootstrap_ci, wilcoxon_tests, n_programmes, parameters.
    """
    result: dict = {
        "parameters": {"n_boot": n_boot, "ci": ci},
        "n_programmes": len(per_programme),
        "bootstrap_ci": {},
        "wilcoxon_tests": {},
    }

    # Bootstrap CIs on each Spearman column
    for col in _SPEARMAN_COLS:
        if col not in per_programme.columns:
            continue
        values = per_programme[col].to_numpy(dtype=float)
        result["bootstrap_ci"][col] = bootstrap_ci(values, n_boot=n_boot, ci=ci)

    # Bootstrap CIs on Jaccard columns
    jaccard_cols = [c for c in per_programme.columns if c.startswith("jaccard_")]
    for col in jaccard_cols:
        values = per_programme[col].to_numpy(dtype=float)
        result["bootstrap_ci"][col] = bootstrap_ci(values, n_boot=n_boot, ci=ci)

    # Wilcoxon signed-rank tests on Spearman pairs
    for col_a, col_b in _SPEARMAN_PAIRS:
        if col_a not in per_programme.columns or col_b not in per_programme.columns:
            continue
        a = per_programme[col_a].to_numpy(dtype=float)
        b = per_programme[col_b].to_numpy(dtype=float)
        key = f"{col_a}_vs_{col_b}"
        result["wilcoxon_tests"][key] = paired_wilcoxon(a, b)

    return result


# ── Pipeline entry point ───────────────────────────────────────────────────────

def run_significance(
    per_programme_path: Path = PER_PROGRAMME,
    output_dir: Path = EVAL_DIR,
    n_boot: int = 1000,
) -> None:
    """Load per-programme metrics, compute significance, persist results."""
    logger.info(f"Loading per-programme metrics from {per_programme_path}…")
    per_programme = pd.read_parquet(per_programme_path)

    result = compute_significance(per_programme, n_boot=n_boot)

    output_path = output_dir / "significance.json"
    with open(output_path, "w") as fh:
        json.dump(result, fh, indent=2)
    logger.info(f"Significance results → {output_path}")

    # Log key findings
    for col, ci_data in result["bootstrap_ci"].items():
        if ci_data["mean"] is not None and col.startswith("spearman"):
            logger.info(
                f"  {col}: {ci_data['mean']:.3f} "
                f"[{ci_data['ci_lower']:.3f}, {ci_data['ci_upper']:.3f}]"
            )
    for key, wt in result["wilcoxon_tests"].items():
        if wt["p_value"] is not None:
            sig = "***" if wt["p_value"] < 0.001 else "**" if wt["p_value"] < 0.01 else "*" if wt["p_value"] < 0.05 else "ns"
            logger.info(f"  {key}: p={wt['p_value']:.4f} {sig}, r={wt['effect_size']:.3f}")


if __name__ == "__main__":
    run_significance()
