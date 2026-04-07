"""
Step 24 — Hybrid formula tuning.

Compares normalisation and scoring variants on the same candidate set
(Stage 1 semantic retrieval + Stage 2 symbolic recall already computed).

Variants:
  1. minmax           — current: α·minmax(cos) + (1-α)·minmax(rec)
  2. rank             — α·rank_norm(cos) + (1-α)·rank_norm(rec)
  3. minmax_agree     — minmax + β·agreement_boost
  4. rank_agree       — rank  + β·agreement_boost

Agreement boost = 1 - |norm(cos) - norm(rec)|  (rewards signal consensus).

Each variant is evaluated across alpha ∈ [0.0, 0.05, ..., 1.0] with and
without IPF.  Metrics: top-1 diversity, score CoV, Kendall-tau stability,
cross-strategy Spearman.

Output:
  experiments/results/sensitivity/formula_variants.json

Usage:
    python -m src.evaluation.formula_tuning
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import kendalltau, spearmanr

from src.alignment.semantic import align_semantic
from src.alignment.symbolic import align_symbolic_weighted
from src.scraping.config import DATA_DIR

DATASET_PATH = DATA_DIR / "dataset" / "dataset.parquet"
RESULTS_DIR = Path("experiments") / "results"


# ── Normalisation functions ───────────────────────────────────────────────────

def _minmax_norm(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    return (series - lo) / (hi - lo) if hi > lo else pd.Series(0.0, index=series.index)


def _rank_norm(series: pd.Series) -> pd.Series:
    """Rank-based normalisation: best rank → 1.0, worst → 0.0."""
    n = len(series)
    if n <= 1:
        return pd.Series(0.0, index=series.index)
    ranks = series.rank(ascending=True, method="average")
    return (ranks - 1) / (n - 1)


# ── IPF computation ──────────────────────────────────────────────────────────

def _apply_ipf(
    merged: pd.DataFrame,
    n_prog: int,
    ipf_top_k: int = 10,
    ipf_floor: float = 0.3,
) -> pd.Series:
    """Compute IPF multiplier. Returns a Series aligned with merged index."""
    merged = merged.copy()
    merged["_rank"] = merged.groupby("programme_id")["hybrid_score"].rank(
        ascending=False, method="first",
    )
    top_k_mask = merged["_rank"] <= ipf_top_k
    job_prog_count = (
        merged.loc[top_k_mask]
        .groupby("job_id")["programme_id"]
        .nunique()
        .rename("_prog_count")
    )
    merged = merged.merge(job_prog_count, on="job_id", how="left")
    merged["_prog_count"] = merged["_prog_count"].fillna(1.0)

    ipf_raw = np.log1p(n_prog / merged["_prog_count"])
    ipf_lo, ipf_hi = ipf_raw.min(), ipf_raw.max()
    if ipf_hi > ipf_lo:
        raw_norm = (ipf_raw - ipf_lo) / (ipf_hi - ipf_lo)
        ipf_mult = ipf_floor + (1.0 - ipf_floor) * raw_norm
    else:
        ipf_mult = pd.Series(1.0, index=merged.index)

    return ipf_mult


# ── Score a single variant ───────────────────────────────────────────────────

def _score_variant(
    candidates: pd.DataFrame,
    alpha: float,
    norm_fn: str,
    agreement_beta: float,
    ipf_top_k: int,
    n_prog: int,
    ipf_floor: float = 0.3,
    combine: str = "linear",
) -> pd.DataFrame:
    """Apply normalisation + scoring + IPF for one variant/alpha combo.

    Parameters
    ----------
    combine : how to fuse normalised cosine and recall.
        "linear"    — α·cos + (1-α)·rec   (default)
        "geometric" — cos^α · rec^(1-α)
        "harmonic"  — 1 / (α/cos + (1-α)/rec)
    """
    m = candidates.copy()

    norm = _minmax_norm if norm_fn == "minmax" else _rank_norm
    m["cosine_norm"] = m.groupby("programme_id")["cosine_score"].transform(norm)
    m["recall_norm"] = m.groupby("programme_id")["programme_recall"].transform(norm)

    cn = m["cosine_norm"]
    rn = m["recall_norm"]

    if combine == "linear":
        m["hybrid_score"] = alpha * cn + (1.0 - alpha) * rn
    elif combine == "geometric":
        # Shift by epsilon to avoid 0^α = 0 wiping the other signal
        eps = 1e-6
        m["hybrid_score"] = (cn + eps) ** alpha * (rn + eps) ** (1.0 - alpha)
    elif combine == "harmonic":
        eps = 1e-6
        m["hybrid_score"] = np.where(
            (cn + rn) > eps,
            1.0 / (alpha / (cn + eps) + (1.0 - alpha) / (rn + eps)),
            0.0,
        )

    if agreement_beta > 0:
        agreement = 1.0 - (m["cosine_norm"] - m["recall_norm"]).abs()
        m["hybrid_score"] = m["hybrid_score"] + agreement_beta * agreement

    if ipf_top_k > 0:
        ipf_mult = _apply_ipf(m, n_prog, ipf_top_k=ipf_top_k, ipf_floor=ipf_floor)
        m["hybrid_score"] = m["hybrid_score"] * ipf_mult

    m = m.sort_values(
        ["programme_id", "hybrid_score"], ascending=[True, False],
    ).reset_index(drop=True)
    return m


# ── Evaluate one scored ranking ──────────────────────────────────────────────

def _evaluate_ranking(
    ranked: pd.DataFrame,
    sem_rankings: pd.DataFrame,
    sym_rankings: pd.DataFrame,
) -> dict:
    """Compute evaluation metrics for a single scored ranking."""
    top1 = ranked.groupby("programme_id").first().reset_index()
    top5 = ranked.groupby("programme_id").head(5)
    top10 = ranked.groupby("programme_id").head(10)

    # Top-1 diversity
    n_progs = ranked["programme_id"].nunique()
    unique_top1 = top1["job_id"].nunique()

    # Top-5 generalist frequency
    top5_freq = top5["job_id"].value_counts()
    top5_gt5 = int((top5_freq > 5).sum())
    top5_max_freq = int(top5_freq.max()) if len(top5_freq) > 0 else 0

    # Score distribution
    hs = ranked["hybrid_score"]
    cov = float(hs.std() / hs.mean()) if hs.mean() > 0 else 0.0

    # Cross-strategy Spearman (per-programme, averaged)
    spearman_sem = []
    spearman_sym = []
    for pid in ranked["programme_id"].unique():
        hyb_top = ranked[ranked["programme_id"] == pid].head(20)
        hyb_jobs = hyb_top["job_id"].tolist()

        sem_prog = sem_rankings[sem_rankings["programme_id"] == pid]
        if len(sem_prog) > 0 and len(hyb_jobs) > 1:
            # Rank correlation on shared jobs
            shared = set(hyb_jobs) & set(sem_prog["job_id"])
            if len(shared) >= 5:
                hyb_ranks = {j: i for i, j in enumerate(hyb_jobs)}
                sem_sorted = sem_prog.sort_values("cosine_combined", ascending=False)
                sem_ranks = {j: i for i, j in enumerate(sem_sorted["job_id"])}
                common = sorted(shared)
                r_h = [hyb_ranks[j] for j in common]
                r_s = [sem_ranks[j] for j in common]
                rho, _ = spearmanr(r_h, r_s)
                if not np.isnan(rho):
                    spearman_sem.append(rho)

        sym_prog = sym_rankings[sym_rankings["programme_id"] == pid]
        if len(sym_prog) > 0 and len(hyb_jobs) > 1:
            shared = set(hyb_jobs) & set(sym_prog["job_id"])
            if len(shared) >= 5:
                hyb_ranks = {j: i for i, j in enumerate(hyb_jobs)}
                sym_sorted = sym_prog.sort_values("weighted_jaccard", ascending=False)
                sym_ranks = {j: i for i, j in enumerate(sym_sorted["job_id"])}
                common = sorted(shared)
                r_h = [hyb_ranks[j] for j in common]
                r_s = [sym_ranks[j] for j in common]
                rho, _ = spearmanr(r_h, r_s)
                if not np.isnan(rho):
                    spearman_sym.append(rho)

    return {
        "unique_top1": unique_top1,
        "n_programmes": n_progs,
        "top1_diversity": round(unique_top1 / max(n_progs, 1), 4),
        "top5_gt5": top5_gt5,
        "top5_max_freq": top5_max_freq,
        "score_mean": round(float(hs.mean()), 6),
        "score_std": round(float(hs.std()), 6),
        "score_cov": round(cov, 4),
        "score_max": round(float(hs.max()), 6),
        "top1_mean": round(float(top1["hybrid_score"].mean()), 6),
        "top5_mean": round(float(top5["hybrid_score"].mean()), 6),
        "spearman_sem_hyb": round(float(np.mean(spearman_sem)), 4) if spearman_sem else None,
        "spearman_sym_hyb": round(float(np.mean(spearman_sym)), 4) if spearman_sym else None,
    }


# ── Main sweep ───────────────────────────────────────────────────────────────

def run_formula_tuning(
    dataset_path: Path = DATASET_PATH,
    output_path: Path = RESULTS_DIR / "sensitivity" / "formula_variants.json",
    semantic_top_n: int = 50,
    ipf_top_k: int = 10,
    alpha_step: float = 0.05,
    agreement_betas: list[float] | None = None,
) -> dict:
    """Run all formula variants and persist comparison."""
    if agreement_betas is None:
        agreement_betas = [0.0, 0.1, 0.2, 0.3]

    logger.info(f"Loading dataset from {dataset_path}…")
    df = pd.read_parquet(dataset_path)
    n_prog = int((df["source_type"] == "programme").sum())

    # ── Stage 1 & 2: compute once, reuse across variants ────────────────────
    logger.info("Stage 1: semantic retrieval…")
    sem_full = align_semantic(df)[
        ["programme_id", "job_id", "programme_name", "job_title", "cosine_combined"]
    ].rename(columns={"cosine_combined": "cosine_score"})

    candidates = (
        sem_full.sort_values(["programme_id", "cosine_score"], ascending=[True, False])
        .groupby("programme_id", sort=False)
        .head(semantic_top_n)
        .reset_index(drop=True)
    )

    logger.info("Stage 2: symbolic refinement…")
    sym_full, _ = align_symbolic_weighted(df, top_n=semantic_top_n)
    sym_recall = sym_full[["programme_id", "job_id", "programme_recall"]]

    merged = candidates.merge(sym_recall, on=["programme_id", "job_id"], how="left")
    merged["programme_recall"] = merged["programme_recall"].fillna(0.0)

    # Load full rankings for cross-strategy Spearman
    sem_rankings = sem_full.copy()
    sem_rankings = sem_rankings.rename(columns={"cosine_score": "cosine_combined"})
    sym_rankings = sym_full.copy()

    # ── Sweep variants ──────────────────────────────────────────────────────
    alphas = [round(a, 2) for a in np.arange(0.0, 1.0 + alpha_step / 2, alpha_step)]

    variants = {}

    # --- Aspect 1: Normalisation × Agreement (linear combination) ---------
    for norm_fn in ["minmax", "rank"]:
        for beta in agreement_betas:
            variant_name = norm_fn
            if beta > 0:
                variant_name += f"_agree{beta:.1f}"

            logger.info(f"Variant: {variant_name}")
            variant_results = []

            for alpha in alphas:
                scored = _score_variant(
                    merged, alpha=alpha, norm_fn=norm_fn,
                    agreement_beta=beta, ipf_top_k=ipf_top_k, n_prog=n_prog,
                )
                metrics = _evaluate_ranking(scored, sem_rankings, sym_rankings)
                metrics["alpha"] = alpha
                variant_results.append(metrics)

            variants[variant_name] = variant_results
            best = max(variant_results, key=lambda m: (m["top1_diversity"], m["score_cov"]))
            logger.info(
                f"  Best alpha={best['alpha']:.2f}: "
                f"diversity={best['top1_diversity']:.2f}, "
                f"CoV={best['score_cov']:.3f}, "
                f"top5_gt5={best['top5_gt5']}"
            )

    # --- Aspect 2: Combination functions (geometric, harmonic) ------------
    for combine in ["geometric", "harmonic"]:
        variant_name = f"minmax_{combine}"
        logger.info(f"Variant: {variant_name}")
        variant_results = []
        for alpha in alphas:
            scored = _score_variant(
                merged, alpha=alpha, norm_fn="minmax",
                agreement_beta=0.0, ipf_top_k=ipf_top_k, n_prog=n_prog,
                combine=combine,
            )
            metrics = _evaluate_ranking(scored, sem_rankings, sym_rankings)
            metrics["alpha"] = alpha
            variant_results.append(metrics)
        variants[variant_name] = variant_results
        best = max(variant_results, key=lambda m: (m["top1_diversity"], m["score_cov"]))
        logger.info(
            f"  Best alpha={best['alpha']:.2f}: "
            f"diversity={best['top1_diversity']:.2f}, "
            f"CoV={best['score_cov']:.3f}, "
            f"top5_gt5={best['top5_gt5']}"
        )

    # --- Aspect 3: IPF parameter sweep ------------------------------------
    ipf_configs = [
        {"ipf_top_k": 5,  "ipf_floor": 0.3},
        {"ipf_top_k": 15, "ipf_floor": 0.3},
        {"ipf_top_k": 20, "ipf_floor": 0.3},
        {"ipf_top_k": 10, "ipf_floor": 0.1},
        {"ipf_top_k": 10, "ipf_floor": 0.5},
        {"ipf_top_k": 0,  "ipf_floor": 0.0},  # no IPF
    ]
    for cfg in ipf_configs:
        tk, fl = cfg["ipf_top_k"], cfg["ipf_floor"]
        variant_name = f"ipf_k{tk}_f{fl:.1f}"
        logger.info(f"Variant: {variant_name}")
        variant_results = []
        for alpha in alphas:
            scored = _score_variant(
                merged, alpha=alpha, norm_fn="minmax",
                agreement_beta=0.0, ipf_top_k=tk, n_prog=n_prog,
                ipf_floor=fl,
            )
            metrics = _evaluate_ranking(scored, sem_rankings, sym_rankings)
            metrics["alpha"] = alpha
            variant_results.append(metrics)
        variants[variant_name] = variant_results
        best = max(variant_results, key=lambda m: (m["top1_diversity"], m["score_cov"]))
        logger.info(
            f"  Best alpha={best['alpha']:.2f}: "
            f"diversity={best['top1_diversity']:.2f}, "
            f"CoV={best['score_cov']:.3f}, "
            f"top5_gt5={best['top5_gt5']}"
        )

    # --- Aspect 4: semantic_top_n sweep -----------------------------------
    for top_n in [20, 30, 75, 100]:
        variant_name = f"topn_{top_n}"
        logger.info(f"Variant: {variant_name} (re-slicing candidates)")
        cand_n = (
            sem_full.sort_values(
                ["programme_id", "cosine_score"], ascending=[True, False],
            )
            .groupby("programme_id", sort=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        merged_n = cand_n.merge(sym_recall, on=["programme_id", "job_id"], how="left")
        merged_n["programme_recall"] = merged_n["programme_recall"].fillna(0.0)

        variant_results = []
        for alpha in alphas:
            scored = _score_variant(
                merged_n, alpha=alpha, norm_fn="minmax",
                agreement_beta=0.0, ipf_top_k=ipf_top_k, n_prog=n_prog,
            )
            metrics = _evaluate_ranking(scored, sem_rankings, sym_rankings)
            metrics["alpha"] = alpha
            variant_results.append(metrics)
        variants[variant_name] = variant_results
        best = max(variant_results, key=lambda m: (m["top1_diversity"], m["score_cov"]))
        logger.info(
            f"  Best alpha={best['alpha']:.2f}: "
            f"diversity={best['top1_diversity']:.2f}, "
            f"CoV={best['score_cov']:.3f}, "
            f"top5_gt5={best['top5_gt5']}"
        )

    # ── Persist ──────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(
            {
                "parameters": {
                    "semantic_top_n": semantic_top_n,
                    "ipf_top_k": ipf_top_k,
                    "ipf_floor": 0.3,
                    "alpha_step": alpha_step,
                    "agreement_betas": agreement_betas,
                },
                "variants": variants,
            },
            fh,
            indent=2,
        )
    logger.info(f"Results → {output_path}")
    return variants


if __name__ == "__main__":
    run_formula_tuning()
