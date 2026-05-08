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

      α (``alpha``) defaults to 0.55.  Varying α between 0 and 1 gives a
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
import math
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.alignment.cross_encoder import (
    load_cross_encoder,
    score_pairs,
    score_pairs_chunked,
    score_pairs_sectioned,
    score_pairs_sectioned_chunked,
)
from src.alignment.semantic import align_semantic
from src.alignment.symbolic import align_symbolic_weighted
from src.scraping.config import DATA_DIR
from src.skills.skill_weights import compute_corpus_idf, compute_median_idf

# ── Paths ──────────────────────────────────────────────────────────────────────

DATASET_PATH = DATA_DIR / "dataset" / "dataset.parquet"
RESULTS_DIR = DATA_DIR.parent / "experiments" / "results"


# ── Match quality refinement ───────────────────────────────────────────────────

def compute_match_quality(
    matched_uris: list[str],
    job_uris: list[str],
    uri_idfs: dict[str, float],
    median_idf: float,
    gamma: float = 0.3,
) -> dict[str, float]:
    """
    Compute a quality multiplier for programme_recall.

    Returns dict with keys: specificity_ratio, generic_penalty,
    quality_multiplier (product of the two).
    """
    # ── 1. Specificity ratio ──────────────────────────────────────────────
    if not matched_uris or not job_uris:
        specificity_ratio = 1.0
    else:
        mean_idf_matched = np.mean([uri_idfs.get(u, 1.0) for u in matched_uris])
        mean_idf_all_job = np.mean([uri_idfs.get(u, 1.0) for u in job_uris])
        raw = math.log(1.0 + mean_idf_matched) / math.log(1.0 + mean_idf_all_job)
        specificity_ratio = float(np.clip(raw, 0.5, 2.0))

    # ── 2. Generic penalty ────────────────────────────────────────────────
    if not matched_uris or gamma == 0.0:
        generic_penalty = 1.0
    else:
        matched_idfs = [uri_idfs.get(u, 1.0) for u in matched_uris]
        total_idf = sum(matched_idfs)
        if total_idf == 0.0:
            generic_penalty = 1.0
        else:
            generic_sum = sum(v for v in matched_idfs if v < median_idf)
            generic_frac = generic_sum / total_idf
            generic_penalty = 1.0 - gamma * generic_frac

    quality_multiplier = specificity_ratio * generic_penalty

    return {
        "specificity_ratio": specificity_ratio,
        "generic_penalty": generic_penalty,
        "quality_multiplier": quality_multiplier,
    }


# ── Core ───────────────────────────────────────────────────────────────────────

def align_hybrid(
    df: pd.DataFrame,
    semantic_top_n: int = 50,
    alpha: float = 0.55,
    ipf_top_k: int = 30,
    ipf_floor: float = 0.1,
    ipf_strict_floor: float = 0.05,
    ipf_strict_threshold: float = 0.5,
    norm_confidence: bool = True,
    gamma: float = 0.3,
    use_programme_idf: bool = True,
    implicit_confidence_mode: str = "sqrt",
    symbolic_signal_mode: str = "recall",
    hi_idf_blend_lambda: float = 0.3,
    cross_encoder_model: str | object | None = None,
    xe_alpha: float = 0.0,
    xe_batch_size: int = 64,
    xe_pool_mode: str = "single",
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
    ipf_top_k      : top-K threshold for inverse programme frequency.
                     Jobs appearing in top-K for many programmes are
                     down-weighted.  Set to 0 to disable.
    ipf_floor      : minimum IPF multiplier for moderately popular jobs.
    ipf_strict_floor : minimum IPF multiplier for universal generalists
                     (jobs in top-K of > ``ipf_strict_threshold`` programmes).
    ipf_strict_threshold : fraction of programmes above which a job is
                     considered a universal generalist (default 0.5 = 50%).
    norm_confidence : if True, dampen min-max normalisation when the raw
                     score range within a programme is small, preventing
                     noisy rankings for uniformly weak matches.
    gamma          : weight of the generic penalty in match quality
                     refinement.  0.0 disables the penalty.
    use_programme_idf : if True, weight programme skills by inter-programme
                     IDF in the symbolic refinement stage.
    implicit_confidence_mode : how implicit-skill weight scales with the
                     propagation confidence stored in ``skill_details``.
                     ``"sqrt"`` (default, Step 37) scales by sqrt of
                     normalised confidence; ``"uniform"`` keeps the paper's
                     flat 0.5; ``"linear"`` scales linearly between
                     conf=0.70 (→ 0) and conf=1.0 (→ 0.5).
    symbolic_signal_mode : which symbolic signal drives the Stage 2 refinement.
                     ``"recall"`` (default) uses ``programme_recall`` over
                     all skills.  ``"recall_hi_idf"`` *replaces* it with
                     ``programme_recall_high_idf`` — recall restricted to
                     URIs with corpus IDF above the median (mutual
                     specificity).  Tends to zero out the symbolic signal
                     for jobs that are entirely transversal, leaving cosine
                     to drive the ranking.  ``"recall_hi_idf_blend"``
                     keeps both signals as a convex combination
                     ``(1 − λ) · recall + λ · recall_hi_idf`` with λ set
                     by ``hi_idf_blend_lambda``; intended as the safer
                     default when the hard restriction is too aggressive.
    hi_idf_blend_lambda : weight on the high-IDF channel when
                     ``symbolic_signal_mode == "recall_hi_idf_blend"``.
                     Must be in [0, 1]; ignored otherwise.
    cross_encoder_model : optional cross-encoder for re-ranking the candidate
                     pool produced by Stage 1.  Either a HuggingFace model
                     name (str) or any object exposing ``predict(pairs)``.
                     Required when ``xe_alpha > 0``; ignored when
                     ``xe_alpha == 0``.
    xe_alpha       : weight on the cross-encoder channel in the hybrid
                     blend.  Final formula is
                     ``alpha · cos_norm + xe_alpha · xe_norm
                     + (1 − alpha − xe_alpha) · recall_norm``.
                     Set ``alpha=0`` for pure cross-encoder + recall (Step 38
                     variant 1); use a non-zero ``alpha`` for the
                     three-channel blend (variant 2).
    xe_batch_size  : batch size passed to the cross-encoder (default 64).
    xe_pool_mode   : how the cross-encoder scores a (programme, job) pair.
                     ``"single"`` (default) sends the full ``cleaned_text``
                     once — limited by the 512-token cross-encoder budget.
                     ``"section_weighted"`` parses the programme into
                     section groups (subjects, outcomes, identity,
                     specialisations, _remainder), scores each non-empty
                     section against the job, and weighted-pools using the
                     same SECTION_WEIGHTS as Step 34's programme embeddings.
                     ``"section_max"`` takes the best-scoring section
                     (useful for niche programmes whose discriminator
                     lives in one section).
                     ``"job_chunked_max"`` keeps the programme as a single
                     pass but chunks the job into 256-token pieces and
                     keeps the highest-scoring chunk — reduces the
                     long-generic-job advantage by letting short
                     specialised descriptions win on their best chunk.
                     ``"section_x_job_max"`` two-sided: programme by
                     sections (weighted-mean) and job by chunks (max).
                     Lifts both truncation ceilings; ~3-5× the cost of
                     ``"single"``.

    Returns
    -------
    rankings : pd.DataFrame
        Columns: programme_id, job_id, programme_name, job_title,
                 cosine_score, programme_recall, hybrid_score
                 (and ``xe_score`` when ``xe_alpha > 0``).
        One row per (programme, candidate) pair — at most
        ``semantic_top_n`` rows per programme.
        Sorted by (programme_id asc, hybrid_score desc).
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if not 0.0 <= xe_alpha <= 1.0:
        raise ValueError(f"xe_alpha must be in [0, 1], got {xe_alpha}")
    if alpha + xe_alpha > 1.0 + 1e-9:
        raise ValueError(
            f"alpha + xe_alpha must be ≤ 1, got {alpha} + {xe_alpha} = {alpha + xe_alpha}"
        )
    if xe_alpha > 0 and cross_encoder_model is None:
        raise ValueError("cross_encoder_model is required when xe_alpha > 0")
    valid_pool_modes = (
        "single",
        "section_weighted",
        "section_max",
        "job_chunked_max",
        "section_x_job_max",
    )
    if xe_pool_mode not in valid_pool_modes:
        raise ValueError(
            f"xe_pool_mode must be one of {valid_pool_modes}, got {xe_pool_mode!r}"
        )
    valid_symbolic_modes = ("recall", "recall_hi_idf", "recall_hi_idf_blend")
    if symbolic_signal_mode not in valid_symbolic_modes:
        raise ValueError(
            f"symbolic_signal_mode must be one of {valid_symbolic_modes}, "
            f"got {symbolic_signal_mode!r}"
        )
    if not 0.0 <= hi_idf_blend_lambda <= 1.0:
        raise ValueError(
            f"hi_idf_blend_lambda must be in [0, 1], got {hi_idf_blend_lambda}"
        )

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

    # ── Stage 1.5: cross-encoder re-ranking (optional) ──────────────────────
    if xe_alpha > 0:
        logger.info(
            f"Stage 1.5: cross-encoder re-ranking ({len(candidates):,} pairs, "
            f"pool={xe_pool_mode})…"
        )
        xe_model = load_cross_encoder(cross_encoder_model)
        text_map = df["cleaned_text"].fillna("").to_dict()
        pairs = [
            (text_map.get(pid, ""), text_map.get(jid, ""))
            for pid, jid in zip(candidates["programme_id"], candidates["job_id"])
        ]
        if xe_pool_mode == "single":
            candidates["xe_score"] = score_pairs(
                xe_model, pairs, batch_size=xe_batch_size,
            )
        elif xe_pool_mode == "job_chunked_max":
            candidates["xe_score"] = score_pairs_chunked(
                xe_model, pairs, pool="max", batch_size=xe_batch_size,
            )
        else:
            from src.embeddings.generator import (
                SECTION_WEIGHTS,
                parse_programme_sections,
            )
            if xe_pool_mode == "section_x_job_max":
                candidates["xe_score"] = score_pairs_sectioned_chunked(
                    xe_model, pairs,
                    section_parser=parse_programme_sections,
                    section_weights=SECTION_WEIGHTS,
                    prog_pool="weighted_mean",
                    job_pool="max",
                    batch_size=xe_batch_size,
                )
            else:
                pool = "max" if xe_pool_mode == "section_max" else "weighted_mean"
                candidates["xe_score"] = score_pairs_sectioned(
                    xe_model, pairs,
                    section_parser=parse_programme_sections,
                    section_weights=SECTION_WEIGHTS,
                    pool=pool,
                    batch_size=xe_batch_size,
                )

    # ── Stage 2: symbolic refinement (IDF-weighted programme recall) ────────
    logger.info(
        f"Stage 2: symbolic refinement "
        f"(signal={symbolic_signal_mode}, implicit_mode={implicit_confidence_mode})…"
    )
    sym, _ = align_symbolic_weighted(
        df, top_n=semantic_top_n, use_programme_idf=use_programme_idf,
        implicit_confidence_mode=implicit_confidence_mode,
    )
    sym = sym[["programme_id", "job_id", "programme_recall",
               "programme_recall_high_idf"]]

    merged = candidates.merge(sym, on=["programme_id", "job_id"], how="left")
    merged[["programme_recall", "programme_recall_high_idf"]] = (
        merged[["programme_recall", "programme_recall_high_idf"]].fillna(0.0)
    )

    if symbolic_signal_mode == "recall_hi_idf":
        merged["programme_recall"] = merged["programme_recall_high_idf"]
    elif symbolic_signal_mode == "recall_hi_idf_blend":
        merged["programme_recall"] = (
            (1.0 - hi_idf_blend_lambda) * merged["programme_recall"]
            + hi_idf_blend_lambda * merged["programme_recall_high_idf"]
        )
    # else "recall": leave programme_recall untouched.
    merged = merged.drop(columns=["programme_recall_high_idf"])

    # ── Match quality refinement ───────────────────────────────────────────────
    if gamma != 0.0:
        logger.info(f"  Applying match quality refinement (γ={gamma})…")

        # Build corpus IDF (same pattern as align_symbolic_weighted)
        all_uri_lists: list[list[str]] = []
        for _, row in df.iterrows():
            details = row.get("skill_details", [])
            if not isinstance(details, (list, np.ndarray)):
                details = []
            all_uri_lists.append(
                [s.get("esco_uri", "") for s in details if s.get("esco_uri")]
            )
        uri_idfs = compute_corpus_idf(all_uri_lists)
        median_idf = compute_median_idf(uri_idfs)

        # Build URI sets per row — use original df index as key because
        # align_semantic uses df.reset_index(drop=False) and stores the
        # original index as programme_id / job_id.
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

        # Apply quality multiplier per row
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

    # ── Per-programme min-max normalisation (confidence-aware) ─────────────────
    # When all candidates have similar raw scores, plain min-max stretches tiny
    # differences to [0, 1], making rankings fragile.  Confidence-aware mode
    # dampens the output by the ratio of the actual range to a reference range,
    # so programmes with uniformly weak matches keep normalised values near 0.
    #
    # Reference ranges are computed as the global median per-programme range
    # across the dataset — anything below that is "low confidence".

    def _minmax(series: pd.Series) -> pd.Series:
        lo, hi = series.min(), series.max()
        return (series - lo) / (hi - lo) if hi > lo else pd.Series(0.0, index=series.index)

    def _confident_minmax(col: str) -> pd.Series:
        raw_norm = merged.groupby("programme_id")[col].transform(_minmax)
        if not norm_confidence:
            return raw_norm
        # Compute per-programme range
        ranges = merged.groupby("programme_id")[col].transform(
            lambda s: s.max() - s.min()
        )
        # Reference = median of per-programme ranges (one value per programme)
        prog_ranges = merged.groupby("programme_id")[col].agg(lambda s: s.max() - s.min())
        ref_range = prog_ranges.median()
        if ref_range <= 0:
            return raw_norm
        # Confidence = min(range / ref, 1.0) — capped at 1 so strong ranges unaffected
        confidence = (ranges / ref_range).clip(upper=1.0)
        return raw_norm * confidence

    merged["cosine_norm"] = _confident_minmax("cosine_score")
    # Recall is not dampened: a narrow recall range still carries real signal
    # (skill overlap vs none), unlike narrow cosine which is genuinely noise.
    merged["recall_norm"] = merged.groupby("programme_id")["programme_recall"].transform(_minmax)

    if xe_alpha > 0:
        merged["xe_norm"] = _confident_minmax("xe_score")

    # ── Hybrid score ───────────────────────────────────────────────────────────
    recall_weight = 1.0 - alpha - xe_alpha
    if xe_alpha > 0:
        merged["hybrid_score"] = (
            alpha * merged["cosine_norm"]
            + xe_alpha * merged["xe_norm"]
            + recall_weight * merged["recall_norm"]
        )
    else:
        merged["hybrid_score"] = (
            alpha * merged["cosine_norm"] + recall_weight * merged["recall_norm"]
        )

    # ── Inverse programme frequency (two-tier generalist penalty) ────────────
    if ipf_top_k > 0:
        # Rank within each programme by initial hybrid score
        merged["_rank"] = merged.groupby("programme_id")["hybrid_score"].rank(
            ascending=False, method="first",
        )
        # Count how many programmes each job appears in top-K for
        top_k_mask = merged["_rank"] <= ipf_top_k
        job_prog_count = (
            merged.loc[top_k_mask]
            .groupby("job_id")["programme_id"]
            .nunique()
            .rename("_prog_count")
        )
        merged = merged.merge(job_prog_count, on="job_id", how="left")
        merged["_prog_count"] = merged["_prog_count"].fillna(1.0)

        # IPF = log(1 + N_prog / count) — smoothed, min-max normalised
        # Two-tier floor: universal generalists (>50% of programmes) get
        # ipf_strict_floor; moderately popular jobs get ipf_floor.
        strict_cutoff = max(2, int(n_prog * ipf_strict_threshold))
        merged["_ipf"] = np.log1p(n_prog / merged["_prog_count"])
        ipf_lo, ipf_hi = merged["_ipf"].min(), merged["_ipf"].max()
        if ipf_hi > ipf_lo:
            raw_norm = (merged["_ipf"] - ipf_lo) / (ipf_hi - ipf_lo)
            # Per-job floor: strict for universal generalists, standard otherwise
            per_job_floor = np.where(
                merged["_prog_count"] >= strict_cutoff,
                ipf_strict_floor,
                ipf_floor,
            )
            merged["_ipf_norm"] = per_job_floor + (1.0 - per_job_floor) * raw_norm
        else:
            merged["_ipf_norm"] = 1.0

        merged["hybrid_score"] = merged["hybrid_score"] * merged["_ipf_norm"]

        n_strict = int((merged["_prog_count"] >= strict_cutoff).any() and
                       job_prog_count[job_prog_count >= strict_cutoff].count())
        logger.info(
            f"  IPF penalty (top-{ipf_top_k}): "
            f"{int(job_prog_count[job_prog_count > 1].count())} generalist jobs penalised "
            f"({n_strict} universal, floor={ipf_strict_floor})"
        )
        merged = merged.drop(columns=["_rank", "_prog_count", "_ipf", "_ipf_norm"])

    drop_cols = ["cosine_norm", "recall_norm"]
    if "xe_norm" in merged.columns:
        drop_cols.append("xe_norm")
    rankings = (
        merged.drop(columns=drop_cols)
        .sort_values(["programme_id", "hybrid_score"], ascending=[True, False])
        .reset_index(drop=True)
    )

    logger.info(
        f"  → {len(rankings):,} candidate pairs "
        f"(avg {len(rankings) / max(n_prog, 1):.1f} per programme)"
    )
    return rankings


# ── Summary ────────────────────────────────────────────────────────────────────

def _compute_summary(
    rankings: pd.DataFrame,
    semantic_top_n: int,
    alpha: float,
    gamma: float = 0.3,
) -> dict:
    return {
        "parameters": {
            "semantic_top_n": semantic_top_n,
            "alpha": alpha,
            "gamma": gamma,
        },
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
    alpha: float = 0.55,
    ipf_top_k: int = 30,
    ipf_floor: float = 0.1,
    ipf_strict_floor: float = 0.05,
    ipf_strict_threshold: float = 0.5,
    norm_confidence: bool = True,
    gamma: float = 0.3,
    use_programme_idf: bool = True,
    implicit_confidence_mode: str = "sqrt",
    symbolic_signal_mode: str = "recall",
    hi_idf_blend_lambda: float = 0.3,
    cross_encoder_model: str | object | None = None,
    xe_alpha: float = 0.0,
    xe_batch_size: int = 64,
    xe_pool_mode: str = "single",
) -> None:
    """Load dataset, run hybrid alignment, persist results."""
    logger.info(f"Loading dataset from {dataset_path}…")
    df = pd.read_parquet(dataset_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    rankings = align_hybrid(
        df, semantic_top_n=semantic_top_n, alpha=alpha,
        ipf_top_k=ipf_top_k, ipf_floor=ipf_floor,
        ipf_strict_floor=ipf_strict_floor,
        ipf_strict_threshold=ipf_strict_threshold,
        norm_confidence=norm_confidence,
        gamma=gamma,
        use_programme_idf=use_programme_idf,
        implicit_confidence_mode=implicit_confidence_mode,
        symbolic_signal_mode=symbolic_signal_mode,
        hi_idf_blend_lambda=hi_idf_blend_lambda,
        cross_encoder_model=cross_encoder_model,
        xe_alpha=xe_alpha,
        xe_batch_size=xe_batch_size,
        xe_pool_mode=xe_pool_mode,
    )

    rankings_path = output_dir / "rankings.parquet"
    summary_path = output_dir / "summary.json"

    rankings.to_parquet(rankings_path, index=False)
    logger.info(f"Rankings → {rankings_path}  ({len(rankings):,} pairs)")

    summary = _compute_summary(rankings, semantic_top_n, alpha, gamma)
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"Summary → {summary_path}")


if __name__ == "__main__":
    run_hybrid_alignment()
