"""
Step 7a — Study programme clustering.

Groups programmes by specialization pattern using sentence-transformer
embeddings.  Reveals structural groupings in Lithuanian ICT higher education
(e.g. AI/ML focus, software engineering, data science, cybersecurity).

Input:   data/dataset/dataset.parquet  (or programmes_embeddings.parquet)
Output:  data/dataset/dataset.parquet  with cluster_label columns added for
         source_type == "programme"

Usage:
    python -m src.clustering.programme_clustering
"""

from __future__ import annotations

from pathlib import Path

from src.scraping.config import DATA_DIR
from src.clustering.clusterer import ClusteringAlgorithm, run_clustering

DATASET_PATH = DATA_DIR / "dataset" / "dataset.parquet"
N_CLUSTERS = 6   # typical specialization groups in Lithuanian ICT programmes


def run(
    input_path: Path = DATASET_PATH,
    output_path: Path = DATASET_PATH,
    algorithm: ClusteringAlgorithm = "kmeans",
    n_clusters: int = N_CLUSTERS,
    min_cluster_size: int = 5,
) -> None:
    run_clustering(
        input_path=input_path,
        output_path=output_path,
        source_type="programme",
        mode="embedding",
        algorithm=algorithm,
        n_clusters=n_clusters,
        min_cluster_size=min_cluster_size,
    )


if __name__ == "__main__":
    run()
