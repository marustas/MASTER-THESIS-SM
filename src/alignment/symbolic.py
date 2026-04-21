"""
Step 8 — Experiment 1: Skill-Based Symbolic Alignment.

Represents programmes and job ads as weighted ESCO skill-URI sets and
computes two complementary alignment scores per (programme, job) pair:

  weighted_jaccard  — symmetric similarity:
      sum(min(w_p, w_j)) / sum(max(w_p, w_j))  over all URI in union

  overlap_coeff — asymmetric coverage (how much of the smaller set is shared):
      sum(min(w_p, w_j)) / min(sum(w_p), sum(w_j))

Skill weights follow Gugnani & Misra (2020):
  explicit skills  → E1 weight = 1.0
  implicit skills  → E3 weight = 0.5  (inferred, lower confidence)

If a URI appears as both explicit and implicit the higher weight (1.0) wins.

Output  (experiments/results/exp1_symbolic/):
  rankings.parquet   — all (programme, job) pairs with scores; sorted by
                       (programme_id asc, weighted_jaccard desc)
  skill_gaps.parquet — per top-N pair: URIs the job has that the programme
                       lacks, with residual gap weight
  summary.json       — aggregate statistics

Usage:
    python -m src.alignment.symbolic
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from src.scraping.config import DATA_DIR
from src.skills.skill_weights import (
    build_reuse_level_map,
    build_weighted_skills as _build_tiered_skills,
    compute_corpus_idf,
    compute_programme_idf,
)

# ── Paths ──────────────────────────────────────────────────────────────────────

DATASET_PATH = DATA_DIR / "dataset" / "dataset.parquet"
RESULTS_DIR = DATA_DIR.parent / "experiments" / "results"

# ── Skill weights (Gugnani & Misra 2020) ──────────────────────────────────────

_EXPLICIT_WEIGHT: float = 1.0   # E1 — directly mentioned
_IMPLICIT_WEIGHT: float = 0.5   # E3 — inferred from similar documents


# ── Core helpers ───────────────────────────────────────────────────────────────

def _build_weighted_skills(skill_details: list[dict]) -> dict[str, float]:
    """
    Build {esco_uri: weight} from a row's skill_details list.

    Explicit skills get weight 1.0; implicit get 0.5.
    If a URI appears under both, the higher weight wins.
    """
    weights: dict[str, float] = {}
    for skill in skill_details:
        uri = skill.get("esco_uri", "")
        if not uri:
            continue
        if skill.get("explicit", False):
            w = _EXPLICIT_WEIGHT
        else:
            w = _IMPLICIT_WEIGHT
        weights[uri] = max(weights.get(uri, 0.0), w)
    return weights


def weighted_jaccard(w_a: dict[str, float], w_b: dict[str, float]) -> float:
    """
    Weighted Jaccard similarity between two weighted skill sets.

    Returns 0.0 when both sets are empty.
    """
    if not w_a and not w_b:
        return 0.0
    all_uris = set(w_a) | set(w_b)
    intersection = sum(min(w_a.get(u, 0.0), w_b.get(u, 0.0)) for u in all_uris)
    union_ = sum(max(w_a.get(u, 0.0), w_b.get(u, 0.0)) for u in all_uris)
    return intersection / union_ if union_ > 0.0 else 0.0


def overlap_coefficient(w_a: dict[str, float], w_b: dict[str, float]) -> float:
    """
    Weighted overlap coefficient: shared weight / min(total_a, total_b).

    Asymmetric but order-independent when used to measure coverage between
    a programme and a job ad — always normalises by the smaller set so a
    programme with few skills can still achieve a high score if they are a
    subset of the job's requirements.

    Returns 0.0 when either set is empty.
    """
    if not w_a or not w_b:
        return 0.0
    all_uris = set(w_a) | set(w_b)
    intersection = sum(min(w_a.get(u, 0.0), w_b.get(u, 0.0)) for u in all_uris)
    min_total = min(sum(w_a.values()), sum(w_b.values()))
    return intersection / min_total if min_total > 0.0 else 0.0


def programme_recall(
    w_prog: dict[str, float],
    w_job: dict[str, float],
) -> float:
    """
    Asymmetric recall: fraction of job-demanded skill weight the programme covers.

        recall = sum(min(w_p[u], w_j[u])) / sum(w_j)

    Answers: "how much of what the job requires does the programme provide?"

    Properties:
      - A programme covering all of a job's skills scores 1.0 regardless of
        how many extra skills the programme has.
      - A programme missing a job-critical skill is penalised proportionally
        to that skill's weight.
      - Jobs with few skills are not over-rewarded — they still need to match
        the programme's skills to achieve a high score.

    Returns 0.0 when the job has no skills.
    """
    if not w_job:
        return 0.0
    shared = sum(min(w_prog.get(u, 0.0), w_job[u]) for u in w_job)
    total_job = sum(w_job.values())
    return shared / total_job if total_job > 0.0 else 0.0


# ── Alignment ──────────────────────────────────────────────────────────────────

def align_symbolic(
    df: pd.DataFrame,
    top_n: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute pairwise symbolic alignment for all programmes × job ads.

    Parameters
    ----------
    df:
        Unified dataset with columns ``source_type``, ``skill_details``,
        and optionally ``name`` (programmes) / ``job_title`` (job ads).
    top_n:
        Number of top-ranked jobs per programme to include in skill gap
        analysis.

    Returns
    -------
    rankings : pd.DataFrame
        Columns: programme_id, job_id, programme_name, job_title,
                 weighted_jaccard, overlap_coeff.
        Sorted by (programme_id asc, weighted_jaccard desc).
    skill_gaps : pd.DataFrame
        Columns: programme_id, job_id, gap_uri, gap_weight.
        One row per URI the job has that the programme lacks or has at lower
        weight, restricted to the top-N job matches per programme.
    """
    programmes = df[df["source_type"] == "programme"]
    jobs = df[df["source_type"] == "job_ad"]

    logger.info(
        f"Symbolic alignment: {len(programmes)} programmes × {len(jobs)} job ads"
    )

    def _safe_details(row: pd.Series) -> list[dict]:
        import numpy as np
        details = row.get("skill_details", [])
        if isinstance(details, (list, np.ndarray)):
            return list(details)
        return []

    prog_skills: dict[int, dict[str, float]] = {
        idx: _build_weighted_skills(_safe_details(row))
        for idx, row in programmes.iterrows()
    }
    job_skills: dict[int, dict[str, float]] = {
        idx: _build_weighted_skills(_safe_details(row))
        for idx, row in jobs.iterrows()
    }

    has_prog_name = "name" in programmes.columns
    has_job_title = "job_title" in jobs.columns

    records = []
    for p_idx, p_ws in prog_skills.items():
        p_name = programmes.at[p_idx, "name"] if has_prog_name else str(p_idx)
        for j_idx, j_ws in job_skills.items():
            j_title = jobs.at[j_idx, "job_title"] if has_job_title else str(j_idx)
            records.append({
                "programme_id": p_idx,
                "job_id": j_idx,
                "programme_name": p_name,
                "job_title": j_title,
                "weighted_jaccard": weighted_jaccard(p_ws, j_ws),
                "overlap_coeff": overlap_coefficient(p_ws, j_ws),
                "programme_recall": programme_recall(p_ws, j_ws),
            })

    rankings = (
        pd.DataFrame(records)
        .sort_values(["programme_id", "weighted_jaccard"], ascending=[True, False])
        .reset_index(drop=True)
    )

    # ── Skill gap analysis ────────────────────────────────────────────────────
    gap_records = []
    for p_idx, p_ws in prog_skills.items():
        top_job_ids = (
            rankings[rankings["programme_id"] == p_idx]
            .head(top_n)["job_id"]
            .tolist()
        )
        for j_idx in top_job_ids:
            j_ws = job_skills[j_idx]
            for uri, j_weight in j_ws.items():
                p_weight = p_ws.get(uri, 0.0)
                if j_weight > p_weight:
                    gap_records.append({
                        "programme_id": p_idx,
                        "job_id": j_idx,
                        "gap_uri": uri,
                        "gap_weight": round(j_weight - p_weight, 4),
                    })

    skill_gaps = (
        pd.DataFrame(gap_records)
        if gap_records
        else pd.DataFrame(columns=["programme_id", "job_id", "gap_uri", "gap_weight"])
    )

    logger.info(
        f"  → {len(rankings)} pairs scored, "
        f"{len(skill_gaps)} skill-gap entries (top-{top_n} per programme)"
    )
    return rankings, skill_gaps


# ── Summary ────────────────────────────────────────────────────────────────────

def _compute_summary(
    rankings: pd.DataFrame,
    skill_gaps: pd.DataFrame,
    top_n: int,
) -> dict:
    top = rankings.groupby("programme_id").head(top_n)
    return {
        "n_programmes": int(rankings["programme_id"].nunique()),
        "n_jobs": int(rankings["job_id"].nunique()),
        "n_pairs": int(len(rankings)),
        "top_n": top_n,
        "weighted_jaccard_all": {
            "mean": float(rankings["weighted_jaccard"].mean()),
            "median": float(rankings["weighted_jaccard"].median()),
            "max": float(rankings["weighted_jaccard"].max()),
        },
        "weighted_jaccard_top_n": {
            "mean": float(top["weighted_jaccard"].mean()),
            "median": float(top["weighted_jaccard"].median()),
        },
        "overlap_coeff_all": {
            "mean": float(rankings["overlap_coeff"].mean()),
            "median": float(rankings["overlap_coeff"].median()),
        },
        "n_skill_gap_entries": int(len(skill_gaps)),
    }


# ── Weighted alignment (Step 23) ──────────────────────────────────────────────

def align_symbolic_weighted(
    df: pd.DataFrame,
    top_n: int = 20,
    esco_csv_path: Path | None = None,
    idf_cap: float | None = 3.0,
    use_tiers: bool = False,
    use_programme_idf: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Like ``align_symbolic`` but uses capped IDF weights instead of
    uniform 1.0 / 0.5.

    Parameters
    ----------
    df:
        Unified dataset (same schema as ``align_symbolic``).
    top_n:
        Top-N jobs per programme for skill gap analysis.
    esco_csv_path:
        Path to ESCO skills CSV.  ``None`` → default path.
    idf_cap:
        Upper bound on per-URI IDF.  Default 3.0 prevents a single rare
        skill from dominating.  ``None`` disables capping.
    use_tiers:
        If True, also multiply by ESCO reuse-level tier weight.
    use_programme_idf:
        If True, weight programme skills by inter-programme IDF instead
        of corpus-wide IDF.  Skills unique to one programme carry more
        weight than skills shared across all programmes.  Job skills
        still use corpus-wide IDF.

    Returns
    -------
    rankings, skill_gaps — same schema as ``align_symbolic``.
    """
    from src.skills.esco_loader import ESCO_CSV_PATH as _DEFAULT_CSV

    csv_path = esco_csv_path or _DEFAULT_CSV

    programmes = df[df["source_type"] == "programme"]
    jobs = df[df["source_type"] == "job_ad"]

    mode = "IDF + tier" if use_tiers else "IDF-only"
    cap_str = f"cap={idf_cap}" if idf_cap is not None else "uncapped"
    prog_idf_str = " + prog-IDF" if use_programme_idf else ""
    logger.info(
        f"Weighted symbolic alignment ({mode}{prog_idf_str}, {cap_str}): "
        f"{len(programmes)} programmes × {len(jobs)} job ads"
    )

    def _safe_details(row: pd.Series) -> list[dict]:
        import numpy as np
        details = row.get("skill_details", [])
        if isinstance(details, (list, np.ndarray)):
            return list(details)
        return []

    # ── Build reuse-level map ─────────────────────────────────────────────
    uri_reuse = build_reuse_level_map(csv_path)
    logger.info(f"  Reuse-level map: {len(uri_reuse)} URIs")

    # ── Compute corpus IDF over all documents ─────────────────────────────
    all_uri_lists: list[list[str]] = []
    for _, row in df.iterrows():
        details = _safe_details(row)
        all_uri_lists.append([s.get("esco_uri", "") for s in details if s.get("esco_uri")])
    uri_idfs = compute_corpus_idf(all_uri_lists)
    logger.info(f"  IDF computed for {len(uri_idfs)} URIs")

    # ── Programme-level IDF (Step 31) ─────────────────────────────────────
    if use_programme_idf:
        prog_uri_idfs = compute_programme_idf(df)
        logger.info(f"  Programme IDF computed for {len(prog_uri_idfs)} URIs")
    else:
        prog_uri_idfs = uri_idfs

    # ── Build weighted skill vectors ──────────────────────────────────────
    prog_skills: dict[int, dict[str, float]] = {
        idx: _build_tiered_skills(
            _safe_details(row), uri_reuse, prog_uri_idfs,
            idf_cap=idf_cap, use_tiers=use_tiers,
        )
        for idx, row in programmes.iterrows()
    }
    job_skills: dict[int, dict[str, float]] = {
        idx: _build_tiered_skills(
            _safe_details(row), uri_reuse, uri_idfs,
            idf_cap=idf_cap, use_tiers=use_tiers,
        )
        for idx, row in jobs.iterrows()
    }

    has_prog_name = "name" in programmes.columns
    has_job_title = "job_title" in jobs.columns

    records = []
    for p_idx, p_ws in prog_skills.items():
        p_name = programmes.at[p_idx, "name"] if has_prog_name else str(p_idx)
        for j_idx, j_ws in job_skills.items():
            j_title = jobs.at[j_idx, "job_title"] if has_job_title else str(j_idx)
            records.append({
                "programme_id": p_idx,
                "job_id": j_idx,
                "programme_name": p_name,
                "job_title": j_title,
                "weighted_jaccard": weighted_jaccard(p_ws, j_ws),
                "overlap_coeff": overlap_coefficient(p_ws, j_ws),
                "programme_recall": programme_recall(p_ws, j_ws),
            })

    rankings = (
        pd.DataFrame(records)
        .sort_values(["programme_id", "weighted_jaccard"], ascending=[True, False])
        .reset_index(drop=True)
    )

    # ── Skill gap analysis ────────────────────────────────────────────────
    gap_records = []
    for p_idx, p_ws in prog_skills.items():
        top_job_ids = (
            rankings[rankings["programme_id"] == p_idx]
            .head(top_n)["job_id"]
            .tolist()
        )
        for j_idx in top_job_ids:
            j_ws = job_skills[j_idx]
            for uri, j_weight in j_ws.items():
                p_weight = p_ws.get(uri, 0.0)
                if j_weight > p_weight:
                    gap_records.append({
                        "programme_id": p_idx,
                        "job_id": j_idx,
                        "gap_uri": uri,
                        "gap_weight": round(j_weight - p_weight, 4),
                    })

    skill_gaps = (
        pd.DataFrame(gap_records)
        if gap_records
        else pd.DataFrame(columns=["programme_id", "job_id", "gap_uri", "gap_weight"])
    )

    logger.info(
        f"  → {len(rankings)} pairs scored, "
        f"{len(skill_gaps)} skill-gap entries (top-{top_n} per programme)"
    )
    return rankings, skill_gaps


# ── Pipeline entry points ─────────────────────────────────────────────────────

def _persist_results(
    rankings: pd.DataFrame,
    skill_gaps: pd.DataFrame,
    output_dir: Path,
    top_n: int,
) -> None:
    """Write rankings, skill gaps and summary to *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)

    rankings_path = output_dir / "rankings.parquet"
    gaps_path = output_dir / "skill_gaps.parquet"
    summary_path = output_dir / "summary.json"

    rankings.to_parquet(rankings_path, index=False)
    logger.info(f"Rankings → {rankings_path}  ({len(rankings):,} pairs)")

    skill_gaps.to_parquet(gaps_path, index=False)
    logger.info(f"Skill gaps → {gaps_path}  ({len(skill_gaps):,} entries)")

    summary = _compute_summary(rankings, skill_gaps, top_n)
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"Summary → {summary_path}")


def run_symbolic_alignment(
    dataset_path: Path = DATASET_PATH,
    output_dir: Path = RESULTS_DIR / "exp1_symbolic",
    top_n: int = 20,
) -> None:
    """Load dataset, run uniform-weight symbolic alignment, persist results."""
    logger.info(f"Loading dataset from {dataset_path}…")
    df = pd.read_parquet(dataset_path)
    rankings, skill_gaps = align_symbolic(df, top_n=top_n)
    _persist_results(rankings, skill_gaps, output_dir, top_n)


def run_symbolic_alignment_weighted(
    dataset_path: Path = DATASET_PATH,
    output_dir: Path = RESULTS_DIR / "exp1_symbolic_weighted",
    top_n: int = 20,
) -> None:
    """Load dataset, run IDF + tier weighted symbolic alignment, persist results."""
    logger.info(f"Loading dataset from {dataset_path}…")
    df = pd.read_parquet(dataset_path)
    rankings, skill_gaps = align_symbolic_weighted(df, top_n=top_n)
    _persist_results(rankings, skill_gaps, output_dir, top_n)


if __name__ == "__main__":
    run_symbolic_alignment()
    run_symbolic_alignment_weighted()
