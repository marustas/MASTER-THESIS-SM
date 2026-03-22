"""
Step 7b — Job advertisement clustering.

Groups job ads by labour-market demand pattern using sentence-transformer
embeddings.  Reveals distinct skill-demand clusters in the ICT job market
(e.g. backend engineering, data/ML roles, DevOps, NLP/AI research).

Input:   data/dataset/dataset.parquet
Output:  data/dataset/dataset.parquet  with cluster_label columns added for
         source_type == "job_ad"

Usage:
    python -m src.clustering.job_clustering
"""

from __future__ import annotations

from pathlib import Path

from src.scraping.config import DATA_DIR
from src.clustering.clusterer import run_clustering

DATASET_PATH = DATA_DIR / "dataset" / "dataset.parquet"
N_CLUSTERS = 8   # broader variety expected in job market demand groups


def run(
    input_path: Path = DATASET_PATH,
    output_path: Path = DATASET_PATH,
    n_clusters: int = N_CLUSTERS,
) -> None:
    run_clustering(
        input_path=input_path,
        output_path=output_path,
        source_type="job_ad",
        mode="embedding",
        n_clusters=n_clusters,
    )


if __name__ == "__main__":
    run()
