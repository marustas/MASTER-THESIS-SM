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


# ── Step 35 — Alpha rebalance sweep ──────────────────────────────────────────

def _apply_two_tier_ipf(
    merged: pd.DataFrame,
    n_prog: int,
    ipf_top_k: int = 30,
    ipf_floor: float = 0.1,
    ipf_strict_floor: float = 0.05,
    ipf_strict_threshold: float = 0.5,
) -> pd.DataFrame:
    """Recompute two-tier IPF on hybrid_score and multiply in-place."""
    m = merged.copy()
    m["_rank"] = m.groupby("programme_id")["hybrid_score"].rank(
        ascending=False, method="first",
    )
    top_k_mask = m["_rank"] <= ipf_top_k
    job_prog_count = (
        m.loc[top_k_mask]
        .groupby("job_id")["programme_id"]
        .nunique()
        .rename("_prog_count")
    )
    m = m.merge(job_prog_count, on="job_id", how="left")
    m["_prog_count"] = m["_prog_count"].fillna(1.0)

    strict_cutoff = max(2, int(n_prog * ipf_strict_threshold))
    m["_ipf"] = np.log1p(n_prog / m["_prog_count"])
    ipf_lo, ipf_hi = m["_ipf"].min(), m["_ipf"].max()
    if ipf_hi > ipf_lo:
        raw_norm = (m["_ipf"] - ipf_lo) / (ipf_hi - ipf_lo)
        per_job_floor = np.where(
            m["_prog_count"] >= strict_cutoff,
            ipf_strict_floor,
            ipf_floor,
        )
        m["_ipf_norm"] = per_job_floor + (1.0 - per_job_floor) * raw_norm
    else:
        m["_ipf_norm"] = 1.0

    m["hybrid_score"] = m["hybrid_score"] * m["_ipf_norm"]
    m = m.drop(columns=["_rank", "_prog_count", "_ipf", "_ipf_norm"])
    return m


def _evaluate_alpha(
    ranked: pd.DataFrame,
    sem_rankings: pd.DataFrame,
    sym_rankings: pd.DataFrame,
) -> dict:
    """Compute evaluation metrics for one alpha point."""
    top1 = ranked.groupby("programme_id").first().reset_index()
    top5 = ranked.groupby("programme_id").head(5)

    n_progs = ranked["programme_id"].nunique()
    unique_top1 = top1["job_id"].nunique()

    top5_freq = top5["job_id"].value_counts()
    top5_generalists = int((top5_freq > 5).sum())
    top5_max_freq = int(top5_freq.max()) if len(top5_freq) > 0 else 0

    hs = ranked["hybrid_score"]
    cov = float(hs.std() / hs.mean()) if hs.mean() > 0 else 0.0

    # Per-programme Spearman with pure strategies (top-20)
    spearman_sem_list, spearman_sym_list = [], []
    for pid in ranked["programme_id"].unique():
        hyb_top = ranked[ranked["programme_id"] == pid].head(20)
        hyb_jobs = hyb_top["job_id"].tolist()

        for baseline, col, out_list in [
            (sem_rankings, "cosine_combined", spearman_sem_list),
            (sym_rankings, "weighted_jaccard", spearman_sym_list),
        ]:
            bp = baseline[baseline["programme_id"] == pid]
            if bp.empty or len(hyb_jobs) < 2:
                continue
            shared = set(hyb_jobs) & set(bp["job_id"])
            if len(shared) < 5:
                continue
            hyb_ranks = {j: i for i, j in enumerate(hyb_jobs)}
            b_sorted = bp.sort_values(col, ascending=False)
            b_ranks = {j: i for i, j in enumerate(b_sorted["job_id"])}
            common = sorted(shared)
            rho, _ = spearmanr(
                [hyb_ranks[j] for j in common],
                [b_ranks[j] for j in common],
            )
            if not np.isnan(rho):
                out_list.append(rho)

    return {
        "unique_top1": unique_top1,
        "n_programmes": n_progs,
        "top1_diversity": round(unique_top1 / max(n_progs, 1), 4),
        "top5_generalists": top5_generalists,
        "top5_max_freq": top5_max_freq,
        "score_mean": round(float(hs.mean()), 6),
        "score_std": round(float(hs.std()), 6),
        "score_cov": round(cov, 4),
        "top1_mean": round(float(top1["hybrid_score"].mean()), 6),
        "top1_cov": round(
            float(top1["hybrid_score"].std() / top1["hybrid_score"].mean())
            if top1["hybrid_score"].mean() > 0 else 0.0, 4,
        ),
        "spearman_sem": round(float(np.mean(spearman_sem_list)), 4) if spearman_sem_list else None,
        "spearman_sym": round(float(np.mean(spearman_sym_list)), 4) if spearman_sym_list else None,
    }


def run_alpha_rebalance(
    dataset_path: Path | None = None,
    output_path: Path | None = None,
    alpha_min: float = 0.3,
    alpha_max: float = 0.7,
    alpha_step: float = 0.025,
    semantic_top_n: int = 50,
    ipf_top_k: int = 30,
    ipf_floor: float = 0.1,
    ipf_strict_floor: float = 0.05,
    ipf_strict_threshold: float = 0.5,
    norm_confidence: bool = True,
    gamma: float = 0.3,
    use_programme_idf: bool = True,
) -> dict:
    """
    Step 35: fine-grained alpha sweep with current full hybrid pipeline.

    Computes expensive stages once (semantic retrieval, symbolic with programme
    IDF, match quality refinement, confidence-aware normalisation), then sweeps
    alpha with two-tier IPF re-ranking at each value.
    """
    from src.alignment.semantic import align_semantic
    from src.alignment.symbolic import align_symbolic_weighted
    from src.skills.skill_weights import compute_corpus_idf, compute_median_idf

    if dataset_path is None:
        dataset_path = DATA_DIR / "dataset" / "dataset.parquet"
    if output_path is None:
        output_path = SENSITIVITY_DIR / "alpha_rebalance.json"

    logger.info(f"Loading dataset from {dataset_path}…")
    df = pd.read_parquet(dataset_path)
    n_prog = int((df["source_type"] == "programme").sum())

    # ── Stage 1: semantic retrieval (once) ────────────────────────────────
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

    # ── Stage 2: symbolic refinement with programme IDF (once) ────────────
    logger.info("Stage 2: symbolic refinement (programme IDF)…")
    sym_full, _ = align_symbolic_weighted(
        df, top_n=semantic_top_n, use_programme_idf=use_programme_idf,
    )
    sym_recall = sym_full[["programme_id", "job_id", "programme_recall"]]

    merged = candidates.merge(sym_recall, on=["programme_id", "job_id"], how="left")
    merged["programme_recall"] = merged["programme_recall"].fillna(0.0)

    # ── Match quality refinement (once, alpha-independent) ────────────────
    if gamma != 0.0:
        logger.info(f"  Applying match quality refinement (γ={gamma})…")
        all_uri_lists: list[list[str]] = []
        for _, row in df.iterrows():
            details = row.get("skill_details", [])
            if not isinstance(details, (list, np.ndarray)):
                details = []
            all_uri_lists.append(
                [s.get("esco_uri", "") for s in details if s.get("esco_uri")]
            )

        from src.alignment.hybrid import compute_match_quality
        uri_idfs = compute_corpus_idf(all_uri_lists)
        median_idf = compute_median_idf(uri_idfs)

        prog_uri_map: dict[int, list[str]] = {}
        for idx, row in df[df["source_type"] == "programme"].iterrows():
            details = row.get("skill_details", [])
            if not isinstance(details, (list, np.ndarray)):
                details = []
            prog_uri_map[idx] = [
                s.get("esco_uri", "") for s in details if s.get("esco_uri")
            ]

        job_uri_map: dict[int, list[str]] = {}
        for idx, row in df[df["source_type"] == "job_ad"].iterrows():
            details = row.get("skill_details", [])
            if not isinstance(details, (list, np.ndarray)):
                details = []
            job_uri_map[idx] = [
                s.get("esco_uri", "") for s in details if s.get("esco_uri")
            ]

        multipliers = []
        for _, row in merged.iterrows():
            p_uris = set(prog_uri_map.get(row["programme_id"], []))
            j_uris = job_uri_map.get(row["job_id"], [])
            matched = [u for u in j_uris if u in p_uris]
            qm = compute_match_quality(
                matched_uris=matched,
                job_uris=j_uris,
                uri_idfs=uri_idfs,
                median_idf=median_idf,
                gamma=gamma,
            )
            multipliers.append(qm["quality_multiplier"])

        merged["programme_recall"] = merged["programme_recall"] * multipliers

    # ── Confidence-aware min-max normalisation (once) ─────────────────────
    def _minmax(series: pd.Series) -> pd.Series:
        lo, hi = series.min(), series.max()
        return (series - lo) / (hi - lo) if hi > lo else pd.Series(0.0, index=series.index)

    merged["cosine_norm"] = merged.groupby("programme_id")["cosine_score"].transform(_minmax)
    if norm_confidence:
        ranges = merged.groupby("programme_id")["cosine_score"].transform(
            lambda s: s.max() - s.min()
        )
        prog_ranges = merged.groupby("programme_id")["cosine_score"].agg(
            lambda s: s.max() - s.min()
        )
        ref_range = prog_ranges.median()
        if ref_range > 0:
            confidence = (ranges / ref_range).clip(upper=1.0)
            merged["cosine_norm"] = merged["cosine_norm"] * confidence

    merged["recall_norm"] = merged.groupby("programme_id")["programme_recall"].transform(_minmax)

    # ── Full semantic/symbolic rankings for Spearman comparison ───────────
    sem_rankings = sem_full.rename(columns={"cosine_score": "cosine_combined"})
    sym_rankings = sym_full

    # ── Sweep alpha ───────────────────────────────────────────────────────
    alphas = [round(a, 3) for a in np.arange(alpha_min, alpha_max + alpha_step / 2, alpha_step)]
    logger.info(f"Sweeping alpha: {alphas[0]:.3f} → {alphas[-1]:.3f} ({len(alphas)} values)")

    results = []
    for alpha in alphas:
        m = merged.copy()
        m["hybrid_score"] = alpha * m["cosine_norm"] + (1.0 - alpha) * m["recall_norm"]

        if ipf_top_k > 0:
            m = _apply_two_tier_ipf(
                m, n_prog,
                ipf_top_k=ipf_top_k,
                ipf_floor=ipf_floor,
                ipf_strict_floor=ipf_strict_floor,
                ipf_strict_threshold=ipf_strict_threshold,
            )

        m = m.sort_values(
            ["programme_id", "hybrid_score"], ascending=[True, False],
        ).reset_index(drop=True)

        metrics = _evaluate_alpha(m, sem_rankings, sym_rankings)
        metrics["alpha"] = alpha
        results.append(metrics)
        logger.info(
            f"  α={alpha:.3f}: diversity={metrics['top1_diversity']:.3f}, "
            f"CoV={metrics['score_cov']:.3f}, generalists={metrics['top5_generalists']}"
        )

    # ── Find optimal alpha ────────────────────────────────────────────────
    # Primary: max top-1 diversity; tie-break: max score CoV
    best = max(results, key=lambda m: (m["top1_diversity"], m["score_cov"]))
    best_alpha = best["alpha"]
    logger.info(
        f"Best alpha={best_alpha:.3f}: diversity={best['top1_diversity']:.3f}, "
        f"CoV={best['score_cov']:.3f}, top5_generalists={best['top5_generalists']}"
    )

    # ── Persist ────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "best_alpha": best_alpha,
        "best_metrics": best,
        "current_alpha": 0.6,
        "parameters": {
            "alpha_range": [alpha_min, alpha_max],
            "alpha_step": alpha_step,
            "semantic_top_n": semantic_top_n,
            "ipf_top_k": ipf_top_k,
            "ipf_floor": ipf_floor,
            "ipf_strict_floor": ipf_strict_floor,
            "ipf_strict_threshold": ipf_strict_threshold,
            "norm_confidence": norm_confidence,
            "gamma": gamma,
            "use_programme_idf": use_programme_idf,
        },
        "sweep": results,
    }
    with open(output_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"Results → {output_path}")

    return summary


if __name__ == "__main__":
    run_sensitivity()
