"""
Step 43 — Adaptive alpha experiment.

Targets the generalist-programme failure mode where broad curricula
("Informatics", "Information Systems", "Software Engineering") pick generic
IT roles (sysadmin / helpdesk / PM / industrial PLC programmer) as top-1
matches even though more domain-relevant jobs are present deeper in the
top-10.

Replaces the global ``alpha`` in ``align_hybrid`` with a per-programme
alpha that decreases with each programme's generalist score:

    alpha_p = max(alpha_floor, alpha - decay · generalist_score(p))

where ``generalist_score`` is the fraction of programme skill weight that
sits on URIs *below* the corpus median IDF.  Specialist programmes stay
at the base alpha; generalists shift toward the symbolic / specificity
side of the blend, where the Step 41 + 42 high-IDF channels actually
move the needle.

Sweep over ``decay ∈ {0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40}``.
Decay = 0 reproduces the canonical hybrid baseline exactly (and serves
as the smoke-test row).

Output:
  experiments/results/evaluation/adaptive_alpha/
    summary.json          — metrics per decay value, deltas vs baseline,
                             per-programme alpha distribution
    rankings_<decay>.parquet  — full ranking per decay setting
    FINDINGS.md           — narrative table + recommended decay

Usage:
  .venv/bin/python -m src.evaluation.adaptive_alpha_experiment
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

from src.alignment.hybrid import align_hybrid
from src.alignment.semantic import align_semantic
from src.alignment.symbolic import align_symbolic_weighted
from src.scraping.config import DATA_DIR
from src.skills.skill_weights import (
    build_weighted_skills,
    compute_corpus_idf,
    compute_median_idf,
    programme_generalist_score,
)

DATASET_PATH = DATA_DIR / "dataset" / "dataset.parquet"
RESULTS_DIR = DATA_DIR.parent / "experiments" / "results" / "evaluation" / "adaptive_alpha"

DECAY_SWEEP = (0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
ALPHA_BASE = 0.55
ALPHA_FLOOR = 0.35

# Programmes we know to be generalist failures from the domain-expert review.
GENERALIST_SUSPECTS = (
    ("Informatics", None),                     # 4 institutions
    ("Information Systems", None),             # 2 institutions
    ("Information Systems Engineering", None), # 2 institutions
    ("Software Engineering", None),            # 3 institutions
    ("Cyber Systems and Security", None),
    ("Information Systems Technology", None),
    ("Multimedia Technologies", None),
    ("Media Technologies", None),
)


# ── Metrics ────────────────────────────────────────────────────────────────────


def _compute_metrics(
    rankings: pd.DataFrame,
    sym_rankings: pd.DataFrame,
    sem_rankings: pd.DataFrame,
) -> dict[str, Any]:
    """Headline metrics — identical shape to other Step-* experiments."""
    rankings = rankings.sort_values(["programme_id", "hybrid_score"], ascending=[True, False])

    # Top-1
    top1 = rankings.groupby("programme_id").head(1)
    n_prog = top1["programme_id"].nunique()
    top1_counts = top1["job_id"].value_counts()

    # Top-1 vs Top-2 gap (head discrimination)
    head = rankings.groupby("programme_id").head(2)
    gaps = head.groupby("programme_id")["hybrid_score"].agg(
        lambda s: float(s.iloc[0] - s.iloc[1]) if len(s) >= 2 else 0.0
    )

    # Top-5 generalists (jobs that appear in many top-5s)
    top5 = rankings.groupby("programme_id").head(5)
    top5_counts = top5["job_id"].value_counts()

    # Cross-strategy Spearman
    sym_lookup = sym_rankings.set_index(["programme_id", "job_id"])["programme_recall"]
    sem_lookup = sem_rankings.set_index(["programme_id", "job_id"])["cosine_combined"]
    sp_sym, sp_sem = [], []
    for _, grp in rankings.groupby("programme_id"):
        if len(grp) < 5:
            continue
        idx = list(zip(grp["programme_id"], grp["job_id"]))
        sym_scores = sym_lookup.reindex(idx).fillna(0.0).values
        sem_scores = sem_lookup.reindex(idx).fillna(0.0).values
        hyb_scores = grp["hybrid_score"].values
        if np.std(sym_scores) > 0 and np.std(hyb_scores) > 0:
            sp_sym.append(spearmanr(sym_scores, hyb_scores).correlation)
        if np.std(sem_scores) > 0 and np.std(hyb_scores) > 0:
            sp_sem.append(spearmanr(sem_scores, hyb_scores).correlation)

    return {
        "n_programmes": int(n_prog),
        "unique_top1": int(top1["job_id"].nunique()),
        "top1_diversity": round(top1["job_id"].nunique() / max(n_prog, 1), 4),
        "top1_max_repeat": int(top1_counts.max()) if len(top1_counts) else 0,
        "top5_generalists": int((top5_counts > 5).sum()),
        "top5_max_repeat": int(top5_counts.max()) if len(top5_counts) else 0,
        "top1_score_mean": float(top1["hybrid_score"].mean()),
        "top1_score_max": float(top1["hybrid_score"].max()),
        "top1_score_cov": float(top1["hybrid_score"].std() / max(top1["hybrid_score"].mean(), 1e-9)),
        "gap_mean": float(gaps.mean()),
        "gap_lt_002": int((gaps < 0.02).sum()),
        "gap_lt_005": int((gaps < 0.05).sum()),
        "spearman_sym_hyb_mean": float(np.mean(sp_sym)) if sp_sym else None,
        "spearman_sem_hyb_mean": float(np.mean(sp_sem)) if sp_sem else None,
    }


# ── Helpers ────────────────────────────────────────────────────────────────────


def _compute_programme_alphas(
    df: pd.DataFrame,
    *,
    alpha_base: float,
    alpha_floor: float,
    decay: float,
    implicit_confidence_mode: str,
) -> dict[int, float]:
    """Mirror align_hybrid's internal computation so the report can show it."""
    all_uri_lists: list[list[str]] = []
    for _, row in df.iterrows():
        details = row.get("skill_details", [])
        if not isinstance(details, (list, np.ndarray)):
            details = []
        all_uri_lists.append(
            [s.get("esco_uri", "") for s in details if s.get("esco_uri")]
        )
    uri_idfs = compute_corpus_idf(all_uri_lists)
    high_idf_thr = compute_median_idf(uri_idfs)

    out: dict[int, float] = {}
    for idx, row in df[df["source_type"] == "programme"].iterrows():
        details = row.get("skill_details", [])
        if not isinstance(details, (list, np.ndarray)):
            details = []
        w_skills = build_weighted_skills(
            list(details),
            uri_reuse_levels={},
            uri_idfs=uri_idfs,
            implicit_confidence_mode=implicit_confidence_mode,
        )
        gen = programme_generalist_score(w_skills, uri_idfs, high_idf_thr)
        out[int(idx)] = float(max(alpha_floor, alpha_base - decay * gen))
    return out


def _suspect_top1_changes(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Report top-1 changes for known generalist programmes."""
    prog_rows = df[df["source_type"] == "programme"][["name", "institution"]].reset_index()
    prog_rows = prog_rows.rename(columns={"index": "programme_id"})

    job_titles = df[df["source_type"] == "job_ad"]["job_title"].to_dict()

    def _top1(rankings: pd.DataFrame) -> pd.DataFrame:
        return (
            rankings.sort_values(["programme_id", "hybrid_score"], ascending=[True, False])
            .groupby("programme_id")
            .head(1)[["programme_id", "job_id", "hybrid_score"]]
        )

    b = _top1(baseline).rename(columns={"job_id": "job_b", "hybrid_score": "score_b"})
    c = _top1(candidate).rename(columns={"job_id": "job_c", "hybrid_score": "score_c"})
    merged = b.merge(c, on="programme_id").merge(prog_rows, on="programme_id")

    suspects: list[dict[str, Any]] = []
    for name, _ in GENERALIST_SUSPECTS:
        sub = merged[merged["name"].str.lower() == name.lower()]
        for r in sub.itertuples():
            suspects.append({
                "programme": r.name,
                "institution": r.institution,
                "baseline_top1": job_titles.get(r.job_b, str(r.job_b))[:60],
                "candidate_top1": job_titles.get(r.job_c, str(r.job_c))[:60],
                "score_b": round(float(r.score_b), 4),
                "score_c": round(float(r.score_c), 4),
                "changed": bool(r.job_b != r.job_c),
            })
    return suspects


# ── Experiment ────────────────────────────────────────────────────────────────


def run_experiment(
    dataset_path: Path = DATASET_PATH,
    output_dir: Path = RESULTS_DIR,
    decays: tuple[float, ...] = DECAY_SWEEP,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset from {dataset_path}…")
    df = pd.read_parquet(dataset_path)

    # Cross-strategy references (one-time)
    logger.info("Computing semantic rankings (shared)…")
    sem_full = align_semantic(df)[["programme_id", "job_id", "cosine_combined"]]
    logger.info("Computing symbolic rankings (shared)…")
    sym_full, _ = align_symbolic_weighted(
        df, top_n=50, use_programme_idf=True, implicit_confidence_mode="sqrt",
    )
    sym_full = sym_full[["programme_id", "job_id", "programme_recall"]]

    results: dict[str, Any] = {
        "config": {
            "alpha_base": ALPHA_BASE,
            "alpha_floor": ALPHA_FLOOR,
            "decays": list(decays),
        },
        "by_decay": {},
        "deltas_vs_baseline": {},
        "suspect_changes": {},
        "programme_alphas": {},
    }

    baseline_rankings: pd.DataFrame | None = None

    for decay in decays:
        tag = f"d{decay:.2f}".replace(".", "_")
        logger.info(f"━━━ decay = {decay:.2f} ━━━")

        rankings = align_hybrid(
            df,
            semantic_top_n=50,
            alpha=ALPHA_BASE,
            adaptive_alpha=(decay > 0.0),
            alpha_generalist_decay=decay,
            alpha_floor=ALPHA_FLOOR,
        )
        rankings.to_parquet(output_dir / f"rankings_{tag}.parquet", index=False)

        metrics = _compute_metrics(rankings, sym_full, sem_full)
        results["by_decay"][f"{decay:.2f}"] = metrics

        if decay > 0.0:
            alphas = _compute_programme_alphas(
                df,
                alpha_base=ALPHA_BASE,
                alpha_floor=ALPHA_FLOOR,
                decay=decay,
                implicit_confidence_mode="sqrt",
            )
            vals = list(alphas.values())
            results["programme_alphas"][f"{decay:.2f}"] = {
                "min": float(min(vals)),
                "max": float(max(vals)),
                "mean": float(np.mean(vals)),
                "n_at_floor": int(sum(1 for v in vals if v <= ALPHA_FLOOR + 1e-9)),
            }

        if decay == 0.0:
            baseline_rankings = rankings
        else:
            assert baseline_rankings is not None
            base = results["by_decay"]["0.00"]
            results["deltas_vs_baseline"][f"{decay:.2f}"] = {
                "unique_top1": metrics["unique_top1"] - base["unique_top1"],
                "top1_score_mean": round(metrics["top1_score_mean"] - base["top1_score_mean"], 4),
                "top1_score_max": round(metrics["top1_score_max"] - base["top1_score_max"], 4),
                "top1_score_cov": round(metrics["top1_score_cov"] - base["top1_score_cov"], 4),
                "gap_lt_002": metrics["gap_lt_002"] - base["gap_lt_002"],
                "gap_lt_005": metrics["gap_lt_005"] - base["gap_lt_005"],
                "top5_generalists": metrics["top5_generalists"] - base["top5_generalists"],
                "top5_max_repeat": metrics["top5_max_repeat"] - base["top5_max_repeat"],
            }
            results["suspect_changes"][f"{decay:.2f}"] = _suspect_top1_changes(
                baseline_rankings, rankings, df,
            )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Wrote {summary_path}")
    return results


if __name__ == "__main__":
    run_experiment()
