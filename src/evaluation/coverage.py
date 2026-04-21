"""
Step 32 — Programme coverage analysis for niche domain identification.

Identifies programmes with low job-ad coverage (few matching jobs above
a quality threshold) and flags them for potential corpus expansion.

Uses alignment scores from symbolic, semantic, and hybrid experiments to
compute per-programme coverage metrics:

  - n_matches: number of jobs above the score threshold
  - coverage_ratio: n_matches / total_jobs
  - top_score: best alignment score for the programme
  - domain_cluster: programme's cluster label (for identifying niche domains)

Programmes with n_matches < min_matches are flagged as low-coverage.

Usage:
    python -m src.evaluation.coverage
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.scraping.config import DATA_DIR

DATASET_PATH = DATA_DIR / "dataset" / "dataset.parquet"
RESULTS_DIR = DATA_DIR.parent / "experiments" / "results"


def analyse_coverage(
    dataset: pd.DataFrame,
    hybrid_rankings: pd.DataFrame,
    score_threshold: float = 0.15,
    min_matches: int = 5,
) -> pd.DataFrame:
    """
    Compute per-programme coverage from hybrid alignment rankings.

    Parameters
    ----------
    dataset:
        Unified dataset with ``source_type``, ``cluster_label``, ``name``.
    hybrid_rankings:
        Output of ``align_hybrid`` with ``programme_id``, ``job_id``,
        ``hybrid_score`` columns.
    score_threshold:
        Minimum hybrid score to count a job as a meaningful match.
    min_matches:
        Programmes with fewer matches than this are flagged low-coverage.

    Returns
    -------
    DataFrame with columns: programme_id, programme_name, cluster_label,
    n_matches, coverage_ratio, top_score, mean_score, low_coverage.
    """
    programmes = dataset[dataset["source_type"] == "programme"]
    n_jobs = int((dataset["source_type"] == "job_ad").sum())

    records = []
    for idx, row in programmes.iterrows():
        prog_rankings = hybrid_rankings[hybrid_rankings["programme_id"] == idx]
        above_threshold = prog_rankings[
            prog_rankings["hybrid_score"] >= score_threshold
        ]
        n_matches = len(above_threshold)
        top_score = float(prog_rankings["hybrid_score"].max()) if len(prog_rankings) > 0 else 0.0
        mean_score = float(above_threshold["hybrid_score"].mean()) if n_matches > 0 else 0.0

        records.append({
            "programme_id": idx,
            "programme_name": row.get("name", str(idx)),
            "cluster_label": row.get("cluster_label", -1),
            "n_matches": n_matches,
            "coverage_ratio": n_matches / max(n_jobs, 1),
            "top_score": top_score,
            "mean_score": mean_score,
            "low_coverage": n_matches < min_matches,
        })

    coverage = pd.DataFrame(records)

    n_low = int(coverage["low_coverage"].sum())
    logger.info(
        f"Coverage analysis: {len(coverage)} programmes, "
        f"{n_low} low-coverage (< {min_matches} matches above {score_threshold})"
    )
    return coverage


def identify_niche_clusters(
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate coverage by cluster to identify niche domain groups.

    Returns
    -------
    DataFrame with columns: cluster_label, n_programmes, n_low_coverage,
    low_coverage_ratio, mean_matches, mean_top_score.
    Sorted by low_coverage_ratio descending.
    """
    if coverage.empty:
        return pd.DataFrame(columns=[
            "cluster_label", "n_programmes", "n_low_coverage",
            "low_coverage_ratio", "mean_matches", "mean_top_score",
        ])

    grouped = coverage.groupby("cluster_label").agg(
        n_programmes=("programme_id", "count"),
        n_low_coverage=("low_coverage", "sum"),
        mean_matches=("n_matches", "mean"),
        mean_top_score=("top_score", "mean"),
    ).reset_index()

    grouped["low_coverage_ratio"] = grouped["n_low_coverage"] / grouped["n_programmes"]

    return grouped.sort_values("low_coverage_ratio", ascending=False).reset_index(drop=True)


def generate_expansion_recommendations(
    coverage: pd.DataFrame,
    dataset: pd.DataFrame,
) -> list[dict]:
    """
    Generate corpus expansion recommendations for low-coverage programmes.

    For each low-coverage programme, recommends:
      - The programme's top skill URIs (for targeted job search)
      - Suggested geographic expansion (EU-wide for niche domains)

    Returns
    -------
    List of dicts with keys: programme_id, programme_name, cluster_label,
    n_matches, top_skills, recommendation.
    """
    low_cov = coverage[coverage["low_coverage"] == True]  # noqa: E712
    if low_cov.empty:
        return []

    programmes = dataset[dataset["source_type"] == "programme"]
    recommendations = []

    for _, row in low_cov.iterrows():
        pid = row["programme_id"]
        prog_row = programmes.loc[pid] if pid in programmes.index else None
        if prog_row is None:
            continue

        # Extract programme's top skills
        details = prog_row.get("skill_details", [])
        if not isinstance(details, (list, np.ndarray)):
            details = []
        explicit_skills = [
            s.get("preferred_label", "")
            for s in details
            if s.get("explicit", False) and s.get("preferred_label")
        ][:10]

        recommendations.append({
            "programme_id": int(pid),
            "programme_name": row["programme_name"],
            "cluster_label": int(row["cluster_label"]),
            "n_matches": int(row["n_matches"]),
            "top_skills": explicit_skills,
            "recommendation": (
                f"Expand job corpus for '{row['programme_name']}' "
                f"(cluster {int(row['cluster_label'])}): "
                f"search EU-wide for jobs requiring {', '.join(explicit_skills[:5]) or 'N/A'}. "
                f"Currently only {int(row['n_matches'])} matching jobs."
            ),
        })

    return recommendations


def run_coverage_analysis(
    dataset_path: Path = DATASET_PATH,
    hybrid_rankings_path: Path = RESULTS_DIR / "exp3_hybrid" / "rankings.parquet",
    output_dir: Path = RESULTS_DIR / "coverage",
    score_threshold: float = 0.15,
    min_matches: int = 5,
) -> None:
    """Run full coverage analysis and persist results."""
    logger.info("Loading dataset and rankings…")
    dataset = pd.read_parquet(dataset_path)
    hybrid_rankings = pd.read_parquet(hybrid_rankings_path)

    coverage = analyse_coverage(
        dataset, hybrid_rankings,
        score_threshold=score_threshold, min_matches=min_matches,
    )
    clusters = identify_niche_clusters(coverage)
    recommendations = generate_expansion_recommendations(coverage, dataset)

    output_dir.mkdir(parents=True, exist_ok=True)

    coverage.to_parquet(output_dir / "programme_coverage.parquet", index=False)
    clusters.to_parquet(output_dir / "niche_clusters.parquet", index=False)

    summary = {
        "n_programmes": len(coverage),
        "n_low_coverage": int(coverage["low_coverage"].sum()),
        "score_threshold": score_threshold,
        "min_matches": min_matches,
        "low_coverage_programmes": [
            {"programme_name": r["programme_name"], "n_matches": r["n_matches"]}
            for r in recommendations
        ],
        "niche_clusters": clusters.to_dict(orient="records"),
        "expansion_recommendations": recommendations,
    }
    with open(output_dir / "coverage_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    logger.info(f"Coverage analysis → {output_dir}")


if __name__ == "__main__":
    run_coverage_analysis()
