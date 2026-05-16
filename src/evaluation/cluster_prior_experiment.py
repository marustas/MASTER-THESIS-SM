"""
Step 44 — Cluster prior experiment.

Independent attempt at the generalist-programme failure mode after Step 43
(adaptive alpha) saturated.  Blends each (programme, job) pair's
``programme_recall`` with the recall of the job against the *programme
cluster centroid* (mean weighted-skill profile across all programmes in
the same cluster):

    recall_blended = (1 − κ) · programme_recall + κ · cluster_recall

Rationale.  Many failing programmes ("Informatics", "Information Systems",
"Software Engineering") have thin curriculum text — few extracted skills,
mostly common ones.  Their per-row recall is dominated by transversal
vocabulary, which generic IT jobs (sysadmin, helpdesk, PM) cover trivially.
Pooling skills across cluster members enriches the profile with the
discipline-typical skills the individual curriculum description omitted.
The κ blend lets the per-row recall keep priority when it has signal but
falls back on the cluster centroid when it does not.

Requires programme cluster labels in dataset.parquet (run
src/clustering/programme_clustering.py first).  Independent of the IDF
channel that Step 43 over-relied on.

Sweep over ``κ ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.8}``.  κ = 0 reproduces
the canonical hybrid baseline exactly.

Output:
  experiments/results/evaluation/cluster_prior/
    summary.json   — per-κ metrics, deltas vs baseline, suspect changes
    rankings_<κ>.parquet
    FINDINGS.md

Usage:
  .venv/bin/python -m src.evaluation.cluster_prior_experiment
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

DATASET_PATH = DATA_DIR / "dataset" / "dataset.parquet"
RESULTS_DIR = DATA_DIR.parent / "experiments" / "results" / "evaluation" / "cluster_prior"

KAPPA_SWEEP = (0.0, 0.10, 0.20, 0.30, 0.50, 0.80)

GENERALIST_SUSPECTS = (
    "Informatics",
    "Information Systems",
    "Information Systems Engineering",
    "Software Engineering",
    "Cyber Systems and Security",
    "Information Systems Technology",
    "Multimedia Technologies",
    "Media Technologies",
)


# ── Metrics (same shape as adaptive_alpha_experiment) ─────────────────────────


def _compute_metrics(
    rankings: pd.DataFrame,
    sym_rankings: pd.DataFrame,
    sem_rankings: pd.DataFrame,
) -> dict[str, Any]:
    rankings = rankings.sort_values(["programme_id", "hybrid_score"], ascending=[True, False])

    top1 = rankings.groupby("programme_id").head(1)
    n_prog = top1["programme_id"].nunique()
    top1_counts = top1["job_id"].value_counts()

    head = rankings.groupby("programme_id").head(2)
    gaps = head.groupby("programme_id")["hybrid_score"].agg(
        lambda s: float(s.iloc[0] - s.iloc[1]) if len(s) >= 2 else 0.0
    )

    top5 = rankings.groupby("programme_id").head(5)
    top5_counts = top5["job_id"].value_counts()

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


def _suspect_top1_changes(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
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

    out: list[dict[str, Any]] = []
    for name in GENERALIST_SUSPECTS:
        sub = merged[merged["name"].str.lower() == name.lower()]
        for r in sub.itertuples():
            out.append({
                "programme": r.name,
                "institution": r.institution,
                "baseline_top1": job_titles.get(r.job_b, str(r.job_b))[:60],
                "candidate_top1": job_titles.get(r.job_c, str(r.job_c))[:60],
                "score_b": round(float(r.score_b), 4),
                "score_c": round(float(r.score_c), 4),
                "changed": bool(r.job_b != r.job_c),
            })
    return out


# ── Experiment ────────────────────────────────────────────────────────────────


def run_experiment(
    dataset_path: Path = DATASET_PATH,
    output_dir: Path = RESULTS_DIR,
    kappas: tuple[float, ...] = KAPPA_SWEEP,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset from {dataset_path}…")
    df = pd.read_parquet(dataset_path)

    if "cluster_label" not in df.columns:
        raise RuntimeError(
            "Dataset is missing 'cluster_label' — run "
            "src/clustering/programme_clustering.py and "
            "src/clustering/job_clustering.py before this experiment."
        )

    logger.info("Computing semantic rankings (shared)…")
    sem_full = align_semantic(df)[["programme_id", "job_id", "cosine_combined"]]
    logger.info("Computing symbolic rankings (shared)…")
    sym_full, _ = align_symbolic_weighted(
        df, top_n=50, use_programme_idf=True, implicit_confidence_mode="sqrt",
    )
    sym_full = sym_full[["programme_id", "job_id", "programme_recall"]]

    results: dict[str, Any] = {
        "config": {"kappas": list(kappas)},
        "by_kappa": {},
        "deltas_vs_baseline": {},
        "suspect_changes": {},
    }
    baseline_rankings: pd.DataFrame | None = None

    for kappa in kappas:
        tag = f"k{kappa:.2f}".replace(".", "_")
        logger.info(f"━━━ kappa = {kappa:.2f} ━━━")

        rankings = align_hybrid(
            df,
            semantic_top_n=50,
            cluster_prior_weight=kappa,
        )
        rankings.to_parquet(output_dir / f"rankings_{tag}.parquet", index=False)

        metrics = _compute_metrics(rankings, sym_full, sem_full)
        results["by_kappa"][f"{kappa:.2f}"] = metrics

        if kappa == 0.0:
            baseline_rankings = rankings
        else:
            assert baseline_rankings is not None
            base = results["by_kappa"]["0.00"]
            results["deltas_vs_baseline"][f"{kappa:.2f}"] = {
                "unique_top1": metrics["unique_top1"] - base["unique_top1"],
                "top1_score_mean": round(metrics["top1_score_mean"] - base["top1_score_mean"], 4),
                "top1_score_max": round(metrics["top1_score_max"] - base["top1_score_max"], 4),
                "top1_score_cov": round(metrics["top1_score_cov"] - base["top1_score_cov"], 4),
                "gap_lt_002": metrics["gap_lt_002"] - base["gap_lt_002"],
                "gap_lt_005": metrics["gap_lt_005"] - base["gap_lt_005"],
                "top5_generalists": metrics["top5_generalists"] - base["top5_generalists"],
                "top5_max_repeat": metrics["top5_max_repeat"] - base["top5_max_repeat"],
            }
            results["suspect_changes"][f"{kappa:.2f}"] = _suspect_top1_changes(
                baseline_rankings, rankings, df,
            )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Wrote {summary_path}")
    return results


if __name__ == "__main__":
    run_experiment()
