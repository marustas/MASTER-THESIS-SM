"""Export study programme → job ad alignment results to a human-readable CSV.

Each export is saved as a timestamped file to track progress across runs:
    experiments/results/programme_job_mapping_YYYYMMDD_HHMMSS.csv

A symlink ``programme_job_mapping.csv`` always points to the latest export.

Columns:
    rank                  – job rank within the programme (1 = best match)
    programme_name        – study programme name
    institution           – university / college name
    job_title             – job advertisement title
    employer_sector       – industry / sector tag scraped with the job ad (may be empty)
    location              – job location string (may be empty)
    job_url               – link to the original job posting
    hybrid_score          – combined alignment score (0–1, higher = better match)
    semantic_score        – embedding cosine similarity (0–1)
    top_skill_gaps        – up to 10 ESCO skill labels the programme lacks vs. this job
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.scraping.config import DATA_DIR

# Same weighting scheme used inside src/alignment/symbolic.py
_EXPLICIT_WEIGHT = 1.0
_IMPLICIT_WEIGHT = 0.5
_TOP_GAPS_PER_JOB = 10

RESULTS_DIR = DATA_DIR.parent / "experiments" / "results"
DATASET_PATH = DATA_DIR / "dataset" / "dataset.parquet"
ESCO_PATH = DATA_DIR / "raw" / "esco" / "skills_en.csv"
EXPORTS_DIR = RESULTS_DIR / "exports"
LATEST_LINK = EXPORTS_DIR / "programme_job_mapping.csv"


def _load_esco_labels(path: Path) -> dict[str, str]:
    esco = pd.read_csv(path, usecols=["conceptUri", "preferredLabel"])
    return dict(zip(esco["conceptUri"], esco["preferredLabel"]))


def _skill_weights(details) -> dict[str, float]:
    weights: dict[str, float] = {}
    if details is None or (isinstance(details, float) and np.isnan(details)):
        return weights
    for skill in details:
        uri = skill.get("esco_uri", "")
        if not uri:
            continue
        w = _EXPLICIT_WEIGHT if skill.get("explicit", False) else _IMPLICIT_WEIGHT
        weights[uri] = max(weights.get(uri, 0.0), w)
    return weights


def _top_gap_labels(
    prog_details,
    job_details,
    uri_to_label: dict[str, str],
    n: int = _TOP_GAPS_PER_JOB,
) -> str:
    p = _skill_weights(prog_details)
    j = _skill_weights(job_details)
    gaps = [(uri, w - p.get(uri, 0.0)) for uri, w in j.items() if w > p.get(uri, 0.0)]
    gaps.sort(key=lambda x: -x[1])
    labels = [uri_to_label.get(uri, uri) for uri, _ in gaps[:n]]
    return "; ".join(labels)


def export() -> None:
    # Load base data
    rankings = pd.read_parquet(RESULTS_DIR / "exp3_hybrid" / "rankings.parquet")
    dataset = pd.read_parquet(DATASET_PATH)
    uri_to_label = _load_esco_labels(ESCO_PATH)

    # Index lookup tables
    programmes = dataset[dataset["source_type"] == "programme"][["institution", "skill_details"]].copy()
    jobs = dataset[dataset["source_type"] == "job_ad"][
        ["job_title", "url", "employer_sector", "location", "skill_details"]
    ].copy()
    prog_details = programmes["skill_details"]
    job_details = jobs["skill_details"]

    # Add rank within each programme (hybrid score already sorted per programme)
    rankings = rankings.sort_values(
        ["programme_id", "hybrid_score"], ascending=[True, False]
    )
    rankings["rank"] = rankings.groupby("programme_id").cumcount() + 1
    rankings = rankings[rankings["rank"] <= 10]

    # Enrich with institution and job metadata
    rankings["institution"] = rankings["programme_id"].map(programmes["institution"])
    rankings["job_url"] = rankings["job_id"].map(jobs["url"])
    rankings["employer_sector"] = rankings["job_id"].map(jobs["employer_sector"])
    rankings["location"] = rankings["job_id"].map(jobs["location"])

    # Compute top skill gaps directly from skill_details — independent of any
    # pre-computed symbolic top-N cap so every job shown in the UI has data.
    rankings["top_skill_gaps"] = rankings.apply(
        lambda r: _top_gap_labels(
            prog_details.get(r["programme_id"]),
            job_details.get(r["job_id"]),
            uri_to_label,
        ),
        axis=1,
    )

    # Select and rename columns
    output = rankings[
        [
            "rank",
            "programme_name",
            "institution",
            "job_title",
            "employer_sector",
            "location",
            "job_url",
            "hybrid_score",
            "cosine_score",
            "top_skill_gaps",
        ]
    ].rename(
        columns={
            "cosine_score": "semantic_score",
        }
    )

    output = output.round({"hybrid_score": 4, "semantic_score": 4})

    # Save timestamped file
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_path = EXPORTS_DIR / f"programme_job_mapping_{ts}.csv"
    output.to_csv(timestamped_path, index=False)

    # Update symlink to latest
    if LATEST_LINK.is_symlink() or LATEST_LINK.exists():
        LATEST_LINK.unlink()
    LATEST_LINK.symlink_to(timestamped_path.name)

    print(f"Exported {len(output):,} rows → {timestamped_path}")
    print(f"  {output['programme_name'].nunique()} programmes × {output['job_title'].nunique()} jobs")
    print(f"  Latest: {LATEST_LINK} → {timestamped_path.name}")


if __name__ == "__main__":
    export()
