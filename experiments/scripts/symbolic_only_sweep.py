"""Symbolic-only sweep — rerun every symbolic refinement with metrics that
do not depend on the hybrid pipeline.

For each configuration, compute symbolic-only metrics:
  - Coefficient of variation (per-programme std / mean of the score, averaged)
  - Top-1 unique (distinct top-1 jobs across the 45 programmes)
  - Head-tied count (programmes whose top-1 vs top-2 relative gap < 10 %)
  - Top-5 generalists (jobs appearing in > 5 programmes' top-5)
  - Top-5 max repeat (the worst-repeat job's frequency in any top-5)
  - Top-1 score mean / max

Refinements covered, in order:
  1. Uniform baseline (paper E1=1.0, E3=0.5)
  2. Corpus IDF with cap selection (cap 3.0, cap 2.5, uncapped, with/without tier)
  3. Programme-restricted IDF on the programme side
  4. Confidence-aware implicit weighting (uniform / sqrt / linear)
  5. High-IDF recall blend  — λ sweep
  6. High-IDF F1 blend     — μ sweep

Outputs:
  experiments/results/exp1_symbolic_only_sweep/all_configs.csv
  experiments/results/exp1_symbolic_only_sweep/lambda_sweep.csv
  experiments/results/exp1_symbolic_only_sweep/mu_sweep.csv
  experiments/results/exp1_symbolic_only_sweep/SUMMARY.md

Usage:
  python -m experiments.scripts.symbolic_only_sweep
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import wilcoxon

from src.alignment.symbolic import align_symbolic, align_symbolic_weighted
from src.scraping.config import DATA_DIR

DATASET_PATH = DATA_DIR / "dataset" / "dataset.parquet"
OUTPUT_DIR = DATA_DIR.parent / "experiments" / "results" / "exp1_symbolic_only_sweep"

HEAD_TIED_REL_THRESHOLD = 0.10
TOP5_GENERALIST_THRESHOLD = 5


# ── Symbolic-only metrics ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class SymbolicMetrics:
    name: str
    n_programmes: int
    cov_mean: float
    cov_std: float
    top1_unique: int
    head_tied: int
    top5_generalists: int
    top5_max_repeat: int
    top1_score_mean: float
    top1_score_max: float
    top1_score_median: float
    score_mean_all: float
    score_median_all: float

    def as_row(self) -> dict:
        return {
            "config": self.name,
            "n_programmes": self.n_programmes,
            "cov_mean": round(self.cov_mean, 4),
            "cov_std": round(self.cov_std, 4),
            "top1_unique": self.top1_unique,
            "head_tied": self.head_tied,
            "top5_generalists": self.top5_generalists,
            "top5_max_repeat": self.top5_max_repeat,
            "top1_score_mean": round(self.top1_score_mean, 4),
            "top1_score_max": round(self.top1_score_max, 4),
            "top1_score_median": round(self.top1_score_median, 4),
            "score_mean_all": round(self.score_mean_all, 4),
            "score_median_all": round(self.score_median_all, 4),
        }


def evaluate(rankings: pd.DataFrame, score_col: str, name: str) -> SymbolicMetrics:
    """Compute the full symbolic-only metrics battery for one ranking + score."""
    grp = rankings.groupby("programme_id")[score_col]
    means = grp.mean()
    stds = grp.std()
    # Replace zero means with NaN so the resulting CoV is NaN (we drop NaNs).
    safe_means = means.replace(0.0, np.nan)
    covs = (stds / safe_means).dropna()

    sorted_df = rankings.sort_values(
        ["programme_id", score_col], ascending=[True, False]
    )
    top1 = sorted_df.groupby("programme_id").head(1)
    top5 = sorted_df.groupby("programme_id").head(5)

    # Head-tied: relative gap (top1 - top2) / top1 < 0.10
    head_tied = 0
    for _, sub in sorted_df.groupby("programme_id"):
        scores = sub[score_col].head(2).values
        if len(scores) >= 2 and scores[0] > 0:
            rel = (scores[0] - scores[1]) / scores[0]
            if rel < HEAD_TIED_REL_THRESHOLD:
                head_tied += 1

    job_top5_counts = top5["job_id"].value_counts()
    generalists = int((job_top5_counts > TOP5_GENERALIST_THRESHOLD).sum())
    max_repeat = int(job_top5_counts.max()) if len(job_top5_counts) else 0

    return SymbolicMetrics(
        name=name,
        n_programmes=int(rankings["programme_id"].nunique()),
        cov_mean=float(covs.mean()),
        cov_std=float(covs.std()),
        top1_unique=int(top1["job_id"].nunique()),
        head_tied=head_tied,
        top5_generalists=generalists,
        top5_max_repeat=max_repeat,
        top1_score_mean=float(top1[score_col].mean()),
        top1_score_max=float(top1[score_col].max()),
        top1_score_median=float(top1[score_col].median()),
        score_mean_all=float(rankings[score_col].mean()),
        score_median_all=float(rankings[score_col].median()),
    )


# ── Wilcoxon helper ──────────────────────────────────────────────────────────


def wilcoxon_top20_mean(
    rankings_a: pd.DataFrame, col_a: str,
    rankings_b: pd.DataFrame, col_b: str,
) -> tuple[float, float]:
    """Compare per-programme mean top-20 score between two configurations."""
    def top20_mean(df, col):
        return (
            df.sort_values(["programme_id", col], ascending=[True, False])
              .groupby("programme_id")
              .head(20)
              .groupby("programme_id")[col]
              .mean()
        )
    a = top20_mean(rankings_a, col_a)
    b = top20_mean(rankings_b, col_b)
    common = a.index.intersection(b.index)
    stat, p = wilcoxon(a.loc[common].values, b.loc[common].values)
    return float(stat), float(p)


# ── Blend helpers (symbolic-only post-processing) ────────────────────────────


def apply_recall_blend(df: pd.DataFrame, lam: float) -> pd.DataFrame:
    """Blend programme_recall with programme_recall_high_idf at λ."""
    out = df.copy()
    out["recall_blend"] = (
        (1.0 - lam) * out["programme_recall"]
        + lam * out["programme_recall_high_idf"]
    )
    return out


def apply_f1_blend(df: pd.DataFrame, lam: float, mu: float) -> pd.DataFrame:
    """Add an F1_hi column and the final blended symbolic signal."""
    out = apply_recall_blend(df, lam)
    p_hi = out["programme_precision_high_idf"].astype(float)
    r_hi = out["programme_recall_high_idf"].astype(float)
    denom = (p_hi + r_hi).replace(0.0, np.nan)
    f1_hi = (2.0 * p_hi * r_hi / denom).fillna(0.0)
    out["f1_high_idf"] = f1_hi
    out["symbolic_signal"] = (1.0 - mu) * out["recall_blend"] + mu * f1_hi
    return out


# ── Driver ───────────────────────────────────────────────────────────────────


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading dataset from {DATASET_PATH}…")
    df = pd.read_parquet(DATASET_PATH)

    metrics_rows: list[dict] = []

    # ── (1) Uniform baseline ────────────────────────────────────────────────
    logger.info("=== (1) Uniform baseline ===")
    rk_uniform, _ = align_symbolic(df, top_n=20)
    for sc in ("weighted_jaccard", "overlap_coeff", "programme_recall"):
        m = evaluate(rk_uniform, sc, name=f"uniform[{sc}]")
        metrics_rows.append(m.as_row())

    # ── (2) Refinement 1 — IDF cap selection ────────────────────────────────
    logger.info("=== (2) IDF cap selection sweep ===")
    idf_configs: dict[str, dict] = {
        "idf_uncapped_no_tier":     dict(idf_cap=None, use_tiers=False),
        "idf_cap3_no_tier":         dict(idf_cap=3.0,  use_tiers=False),
        "idf_cap25_no_tier":        dict(idf_cap=2.5,  use_tiers=False),
        "idf_uncapped_tier":        dict(idf_cap=None, use_tiers=True),
        "idf_cap3_tier":            dict(idf_cap=3.0,  use_tiers=True),
    }
    weighted_rk: dict[str, pd.DataFrame] = {}
    for cfg_name, kwargs in idf_configs.items():
        logger.info(f"  running {cfg_name} …")
        rk, _ = align_symbolic_weighted(
            df,
            top_n=20,
            use_programme_idf=False,
            implicit_confidence_mode="uniform",
            **kwargs,
        )
        weighted_rk[cfg_name] = rk
        for sc in ("weighted_jaccard", "programme_recall"):
            metrics_rows.append(evaluate(rk, sc, name=f"{cfg_name}[{sc}]").as_row())

    # Wilcoxon: uniform vs adopted cap3 (weighted_jaccard, top-20 mean)
    wstat, p_value = wilcoxon_top20_mean(
        rk_uniform, "weighted_jaccard",
        weighted_rk["idf_cap3_no_tier"], "weighted_jaccard",
    )
    logger.info(f"  Wilcoxon top-20 mean (uniform vs cap3): W={wstat:.1f}, p={p_value:.3e}")

    # ── (3) Refinement 2 — Programme-restricted IDF ─────────────────────────
    logger.info("=== (3) Programme IDF ===")
    rk_prog_idf, _ = align_symbolic_weighted(
        df,
        top_n=20,
        use_programme_idf=True,
        idf_cap=3.0,
        use_tiers=False,
        implicit_confidence_mode="uniform",
    )
    for sc in ("weighted_jaccard", "programme_recall"):
        metrics_rows.append(evaluate(rk_prog_idf, sc, name=f"cap3_no_tier_prog_idf[{sc}]").as_row())

    # ── (4) Refinement 3 — Confidence-aware implicit weighting ─────────────
    logger.info("=== (4) Implicit confidence sweep ===")
    impl_modes = ["uniform", "linear", "sqrt"]
    rk_impl: dict[str, pd.DataFrame] = {}
    for mode in impl_modes:
        logger.info(f"  running implicit={mode} …")
        rk, _ = align_symbolic_weighted(
            df,
            top_n=20,
            use_programme_idf=True,
            idf_cap=3.0,
            use_tiers=False,
            implicit_confidence_mode=mode,
        )
        rk_impl[mode] = rk
        for sc in ("weighted_jaccard", "programme_recall"):
            metrics_rows.append(evaluate(rk, sc, name=f"cap3_progidf_impl_{mode}[{sc}]").as_row())

    rk_full_base = rk_impl["sqrt"]

    # ── (5) Refinement 4a — High-IDF recall blend λ sweep ──────────────────
    logger.info("=== (5) λ sweep on high-IDF recall blend ===")
    lambda_rows: list[dict] = []
    lambdas = [0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    for lam in lambdas:
        blended = apply_recall_blend(rk_full_base, lam)
        m = evaluate(blended, "recall_blend", name=f"lambda_{lam:.2f}")
        lambda_rows.append({"lambda": lam, **m.as_row()})
    lambda_df = pd.DataFrame(lambda_rows)
    lambda_df.to_csv(OUTPUT_DIR / "lambda_sweep.csv", index=False)
    logger.info(f"λ sweep → {OUTPUT_DIR / 'lambda_sweep.csv'}")

    # Pick λ that maximises top-1 unique with smallest head-tied as tiebreak.
    lambda_df_sorted = lambda_df.sort_values(
        ["top1_unique", "head_tied"], ascending=[False, True]
    )
    chosen_lambda = float(lambda_df_sorted.iloc[0]["lambda"])
    logger.info(f"  λ_chosen = {chosen_lambda}")

    # ── (6) Refinement 4b — High-IDF F1 blend μ sweep ──────────────────────
    logger.info("=== (6) μ sweep on F1 blend ===")
    mu_rows: list[dict] = []
    mus = [0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    for mu in mus:
        blended = apply_f1_blend(rk_full_base, chosen_lambda, mu)
        m = evaluate(blended, "symbolic_signal", name=f"mu_{mu:.2f}")
        mu_rows.append({"mu": mu, **m.as_row()})
    mu_df = pd.DataFrame(mu_rows)
    mu_df.to_csv(OUTPUT_DIR / "mu_sweep.csv", index=False)
    logger.info(f"μ sweep → {OUTPUT_DIR / 'mu_sweep.csv'}")

    mu_df_sorted = mu_df.sort_values(
        ["top5_generalists", "top5_max_repeat", "top1_unique"],
        ascending=[True, True, False],
    )
    chosen_mu = float(mu_df_sorted.iloc[0]["mu"])
    logger.info(f"  μ_chosen = {chosen_mu}")

    # ── Final consolidated configuration ────────────────────────────────────
    final_rk = apply_f1_blend(rk_full_base, chosen_lambda, chosen_mu)
    final_m = evaluate(final_rk, "symbolic_signal", name="FINAL")
    metrics_rows.append(final_m.as_row())

    # ── Write outputs ──────────────────────────────────────────────────────
    all_df = pd.DataFrame(metrics_rows)
    all_df.to_csv(OUTPUT_DIR / "all_configs.csv", index=False)
    logger.info(f"All configs → {OUTPUT_DIR / 'all_configs.csv'}")

    summary = {
        "n_programmes": int(rk_uniform["programme_id"].nunique()),
        "n_jobs": int(rk_uniform["job_id"].nunique()),
        "n_pairs": int(len(rk_uniform)),
        "wilcoxon_uniform_vs_cap3": {"W": wstat, "p": p_value},
        "chosen_lambda": chosen_lambda,
        "chosen_mu": chosen_mu,
        "final": final_m.as_row(),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Symbolic-only sweep complete.")


if __name__ == "__main__":
    main()
