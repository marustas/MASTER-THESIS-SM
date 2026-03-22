"""
Shared clustering logic for Step 7.

Supports two representation modes:
  "embedding"  — cluster on dense sentence-transformer vectors (default)
  "skills"     — cluster on a TF-IDF-weighted bag-of-ESCO-URIs

Algorithm: K-Means (default k=8).  UMAP is used for dimensionality reduction
before K-Means when the embedding dimension is large (>50), which improves
cluster quality and enables 2-D visualisation coordinates as a side product.

Output columns added to the DataFrame:
  cluster_label        — integer cluster id (0-based)
  cluster_label_2d_x   — UMAP x coordinate (for visualisation, optional)
  cluster_label_2d_y   — UMAP y coordinate (for visualisation, optional)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


RepresentationMode = Literal["embedding", "skills"]

_UMAP_THRESHOLD = 50   # reduce dimensions if embedding dim > this
_UMAP_N_COMPONENTS = 16
_UMAP_VIS_COMPONENTS = 2


def _extract_embedding_matrix(df: pd.DataFrame) -> np.ndarray:
    """Stack the 'embedding' list-column into a float32 matrix."""
    if "embedding" not in df.columns:
        raise ValueError("DataFrame has no 'embedding' column. Run Step 5 first.")
    matrix = np.array(df["embedding"].tolist(), dtype=np.float32)
    # Zero-norm rows (empty texts) get a small uniform vector so K-Means doesn't crash
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    zero_mask = (norms.ravel() < 1e-8)
    if zero_mask.any():
        logger.warning(f"  {zero_mask.sum()} zero-embedding rows replaced with uniform vector")
        matrix[zero_mask] = 1.0 / matrix.shape[1]
    return normalize(matrix, norm="l2")


def _extract_skill_matrix(df: pd.DataFrame) -> np.ndarray:
    """Build a TF-IDF matrix over skill_uris (space-joined strings)."""
    if "skill_uris" not in df.columns:
        raise ValueError("DataFrame has no 'skill_uris' column. Run Step 4 first.")
    docs = df["skill_uris"].apply(
        lambda uris: " ".join(uris) if isinstance(uris, list) else ""
    ).tolist()
    vec = TfidfVectorizer(min_df=1, sublinear_tf=True)
    return vec.fit_transform(docs).toarray().astype(np.float32)


def _umap_reduce(matrix: np.ndarray, n_components: int, random_state: int) -> np.ndarray:
    try:
        import umap  # type: ignore[import]
    except ImportError:
        logger.warning("umap-learn not available — skipping UMAP reduction")
        return matrix
    reducer = umap.UMAP(n_components=n_components, random_state=random_state, verbose=False)
    return reducer.fit_transform(matrix).astype(np.float32)


def fit_clusters(
    df: pd.DataFrame,
    mode: RepresentationMode = "embedding",
    n_clusters: int = 8,
    use_umap: bool = True,
    add_2d_coords: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Cluster the records in `df` and return a copy with cluster columns added.

    Parameters
    ----------
    df:
        DataFrame with an 'embedding' or 'skill_uris' column depending on `mode`.
    mode:
        "embedding" uses the dense sentence-transformer vectors.
        "skills"    uses a TF-IDF bag-of-ESCO-URIs representation.
    n_clusters:
        Number of K-Means clusters.
    use_umap:
        If True and embedding dim > _UMAP_THRESHOLD, apply UMAP before K-Means.
    add_2d_coords:
        If True, also compute 2-D UMAP coordinates for visualisation.
    random_state:
        Seed for reproducibility.
    """
    if len(df) < n_clusters:
        logger.warning(
            f"Fewer records ({len(df)}) than n_clusters ({n_clusters}) — "
            f"reducing n_clusters to {len(df)}"
        )
        n_clusters = max(1, len(df))

    logger.info(f"Extracting {mode} representation ({len(df)} records)…")
    matrix = _extract_embedding_matrix(df) if mode == "embedding" else _extract_skill_matrix(df)

    # Dimensionality reduction before clustering
    cluster_input = matrix
    if use_umap and matrix.shape[1] > _UMAP_THRESHOLD:
        logger.info(f"UMAP reduction: {matrix.shape[1]}D → {_UMAP_N_COMPONENTS}D")
        cluster_input = _umap_reduce(matrix, _UMAP_N_COMPONENTS, random_state)

    # K-Means
    logger.info(f"K-Means clustering: k={n_clusters}")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(cluster_input)

    result = df.copy()
    result["cluster_label"] = labels.astype(int)

    # 2-D coordinates for visualisation
    if add_2d_coords and matrix.shape[1] > _UMAP_VIS_COMPONENTS:
        logger.info("Computing 2-D UMAP coordinates for visualisation…")
        coords_2d = _umap_reduce(matrix, _UMAP_VIS_COMPONENTS, random_state)
        result["cluster_label_2d_x"] = coords_2d[:, 0]
        result["cluster_label_2d_y"] = coords_2d[:, 1]

    _log_cluster_stats(result)
    return result


def _log_cluster_stats(df: pd.DataFrame) -> None:
    sizes = df["cluster_label"].value_counts().sort_index()
    logger.info(f"Cluster sizes: {sizes.to_dict()}")

    if "all_skills" not in df.columns:
        return
    for cid in sorted(df["cluster_label"].unique()):
        subset = df[df["cluster_label"] == cid]
        from collections import Counter
        counter: Counter = Counter()
        for skills in subset["all_skills"].dropna():
            counter.update(skills)
        top5 = ", ".join(f"{s}({n})" for s, n in counter.most_common(5))
        logger.info(f"  Cluster {cid} (n={len(subset)}): {top5 or '—'}")


def run_clustering(
    input_path: Path,
    output_path: Path,
    source_type: str,
    mode: RepresentationMode = "embedding",
    n_clusters: int = 8,
    use_umap: bool = True,
) -> pd.DataFrame:
    """
    Load a parquet, cluster the records, save result with cluster columns.

    If the parquet contains multiple source_types, only `source_type` rows are
    clustered; other rows are passed through unchanged without cluster columns.
    """
    logger.info(f"Loading {input_path}")
    df = pd.read_parquet(input_path)

    if "source_type" in df.columns:
        mask = df["source_type"] == source_type
        target = df[mask].copy()
        rest = df[~mask].copy()
        logger.info(f"  Clustering {len(target)} '{source_type}' records (ignoring {len(rest)} others)")
    else:
        target = df.copy()
        rest = pd.DataFrame()

    if target.empty:
        logger.warning(f"No '{source_type}' records found in {input_path}")
        return df

    clustered = fit_clusters(target, mode=mode, n_clusters=n_clusters, use_umap=use_umap)

    if not rest.empty:
        result = pd.concat([clustered, rest], ignore_index=True)
    else:
        result = clustered

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    logger.info(f"Saved → {output_path}")
    return result
