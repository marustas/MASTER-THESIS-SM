"""Export study programme → job ad alignment results to a human-readable CSV.

Output: experiments/results/programme_job_mapping.csv

Columns:
    rank                  – job rank within the programme (1 = best match)
    programme_name        – study programme name
    institution           – university / college name
    job_title             – job advertisement title
    job_url               – link to the original job posting
    hybrid_score          – combined alignment score (0–1, higher = better match)
    semantic_score        – embedding cosine similarity (0–1)
    symbolic_score        – weighted Jaccard over ESCO skill sets (0–1)
    top_skill_gaps        – up to 3 ESCO skill labels the programme lacks vs. this job
"""

import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("experiments/results")
DATASET_PATH = Path("data/dataset/dataset.parquet")
ESCO_PATH = Path("data/raw/esco/skills_en.csv")
OUTPUT_PATH = RESULTS_DIR / "programme_job_mapping.csv"


def _load_esco_labels(path: Path) -> dict[str, str]:
    esco = pd.read_csv(path, usecols=["conceptUri", "preferredLabel"])
    return dict(zip(esco["conceptUri"], esco["preferredLabel"]))


def _top_gap_labels(
    skill_gaps: pd.DataFrame,
    programme_id: int,
    job_id: int,
    uri_to_label: dict[str, str],
    n: int = 3,
) -> str:
    mask = (skill_gaps["programme_id"] == programme_id) & (skill_gaps["job_id"] == job_id)
    rows = skill_gaps[mask].sort_values("gap_weight", ascending=False).head(n)
    labels = [uri_to_label.get(uri, uri) for uri in rows["gap_uri"]]
    return "; ".join(labels)


def export() -> None:
    # Load base data
    rankings = pd.read_parquet(RESULTS_DIR / "exp3_hybrid" / "rankings.parquet")
    skill_gaps = pd.read_parquet(RESULTS_DIR / "exp1_symbolic" / "skill_gaps.parquet")
    dataset = pd.read_parquet(DATASET_PATH)
    uri_to_label = _load_esco_labels(ESCO_PATH)

    # Index lookup tables
    programmes = dataset[dataset["source_type"] == "programme"][["institution"]].copy()
    jobs = dataset[dataset["source_type"] == "job_ad"][["job_title", "url"]].copy()

    # Add rank within each programme (hybrid score already sorted per programme)
    rankings = rankings.sort_values(
        ["programme_id", "hybrid_score"], ascending=[True, False]
    )
    rankings["rank"] = rankings.groupby("programme_id").cumcount() + 1

    # Enrich with institution and job URL
    rankings["institution"] = rankings["programme_id"].map(programmes["institution"])
    rankings["job_url"] = rankings["job_id"].map(jobs["url"])

    # Resolve top skill gaps
    rankings["top_skill_gaps"] = rankings.apply(
        lambda r: _top_gap_labels(skill_gaps, r["programme_id"], r["job_id"], uri_to_label),
        axis=1,
    )

    # Select and rename columns
    output = rankings[
        [
            "rank",
            "programme_name",
            "institution",
            "job_title",
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

    output = output.round({"hybrid_score": 4, "semantic_score": 4, "symbolic_score": 4})
    output.to_csv(OUTPUT_PATH, index=False)
    print(f"Exported {len(output):,} rows → {OUTPUT_PATH}")
    print(f"  {output['programme_name'].nunique()} programmes × {output['job_title'].nunique()} jobs")


if __name__ == "__main__":
    export()
