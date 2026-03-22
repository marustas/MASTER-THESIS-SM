"""
Shared clustering logic for Step 7.

Supports two representation modes:
  "embedding"  — cluster on dense sentence-transformer vectors (default)
  "skills"     — cluster on a TF-IDF-weighted bag-of-ESCO-URIs

Supports three clustering algorithms:
  "kmeans"       — K-Means; needs n_clusters; fast; assumes spherical clusters.
  "hdbscan"      — Hierarchical density-based; determines k automatically;
                   handles noise (label=-1); best for varied-density embeddings.
                   Uses sklearn.cluster.HDBSCAN (requires scikit-learn ≥ 1.3).
  "agglomerative"— Agglomerative hierarchical; needs n_clusters; no centroid
                   assumption; suitable for smaller corpora.

UMAP is used for dimensionality reduction before clustering when the embedding
dimension is large (>50), improving cluster quality and producing 2-D
visualisation coordinates as a by-product.

Output columns added to the DataFrame:
  cluster_label        — integer cluster id (0-based; -1 = noise for HDBSCAN)
  cluster_label_2d_x   — UMAP x coordinate (for visualisation, optional)
  cluster_label_2d_y   — UMAP y coordinate (for visualisation, optional)
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


RepresentationMode = Literal["embedding", "skills"]
ClusteringAlgorithm = Literal["kmeans", "hdbscan", "agglomerative"]

_UMAP_THRESHOLD = 50    # reduce dimensions if embedding dim > this
_UMAP_N_COMPONENTS = 16
_UMAP_VIS_COMPONENTS = 2


# ── Representation extraction ──────────────────────────────────────────────────

def _extract_embedding_matrix(df: pd.DataFrame) -> np.ndarray:
    """Stack the 'embedding' list-column into an L2-normalised float32 matrix."""
    if "embedding" not in df.columns:
        raise ValueError("DataFrame has no 'embedding' column. Run Step 5 first.")
    matrix = np.array(df["embedding"].tolist(), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    zero_mask = norms.ravel() < 1e-8
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


# ── Dimensionality reduction ───────────────────────────────────────────────────

def _umap_reduce(matrix: np.ndarray, n_components: int, random_state: int) -> np.ndarray:
    try:
        import umap  # type: ignore[import]
    except ImportError:
        logger.warning("umap-learn not available — skipping UMAP reduction")
        return matrix
    reducer = umap.UMAP(n_components=n_components, random_state=random_state, verbose=False)
    return reducer.fit_transform(matrix).astype(np.float32)


# ── Algorithm dispatch ─────────────────────────────────────────────────────────

def _run_kmeans(
    matrix: np.ndarray, n_clusters: int, random_state: int
) -> np.ndarray:
    if len(matrix) < n_clusters:
        logger.warning(
            f"Fewer records ({len(matrix)}) than n_clusters ({n_clusters}) — "
            f"reducing to {len(matrix)}"
        )
        n_clusters = max(1, len(matrix))
    logger.info(f"K-Means: k={n_clusters}")
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto").fit_predict(matrix)


def _run_hdbscan(matrix: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """
    HDBSCAN via scikit-learn (≥1.3).  Points that do not belong to any dense
    cluster receive label -1 (noise).  min_cluster_size controls granularity:
    smaller values yield more (smaller) clusters.
    """
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError:
        raise ImportError("HDBSCAN requires scikit-learn ≥ 1.3")
    capped = max(2, min(min_cluster_size, len(matrix) // 2))
    logger.info(f"HDBSCAN: min_cluster_size={capped}")
    labels = HDBSCAN(min_cluster_size=capped).fit_predict(matrix)
    n_found = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    logger.info(f"  → {n_found} clusters, {n_noise} noise points")
    return labels


def _run_agglomerative(
    matrix: np.ndarray, n_clusters: int, linkage: str, random_state: int
) -> np.ndarray:
    if len(matrix) < n_clusters:
        logger.warning(
            f"Fewer records ({len(matrix)}) than n_clusters ({n_clusters}) — "
            f"reducing to {len(matrix)}"
        )
        n_clusters = max(1, len(matrix))
    logger.info(f"Agglomerative: k={n_clusters}, linkage={linkage}")
    return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit_predict(matrix)


# ── Public API ─────────────────────────────────────────────────────────────────

def fit_clusters(
    df: pd.DataFrame,
    mode: RepresentationMode = "embedding",
    algorithm: ClusteringAlgorithm = "kmeans",
    n_clusters: int = 8,
    min_cluster_size: int = 5,
    linkage: str = "ward",
    use_umap: bool = True,
    add_2d_coords: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Cluster the records in `df` and return a copy with cluster columns added.

    Parameters
    ----------
    df:
        DataFrame with an 'embedding' or 'skill_uris' column.
    mode:
        "embedding" | "skills" — which representation to cluster on.
    algorithm:
        "kmeans"        — K-Means (needs n_clusters).
        "hdbscan"       — Density-based; uses min_cluster_size; ignores n_clusters.
        "agglomerative" — Hierarchical; needs n_clusters and linkage.
    n_clusters:
        Target number of clusters (K-Means and Agglomerative only).
    min_cluster_size:
        Minimum points to form a core cluster (HDBSCAN only).
    linkage:
        Linkage criterion for Agglomerative: "ward" | "complete" | "average" | "single".
    use_umap:
        Apply UMAP reduction before clustering when dim > 50.
    add_2d_coords:
        Compute 2-D UMAP coordinates for scatter-plot visualisation.
    random_state:
        Seed for K-Means and UMAP.
    """
    logger.info(f"Extracting {mode} representation ({len(df)} records)…")
    matrix = _extract_embedding_matrix(df) if mode == "embedding" else _extract_skill_matrix(df)

    cluster_input = matrix
    if use_umap and matrix.shape[1] > _UMAP_THRESHOLD:
        logger.info(f"UMAP reduction: {matrix.shape[1]}D → {_UMAP_N_COMPONENTS}D")
        cluster_input = _umap_reduce(matrix, _UMAP_N_COMPONENTS, random_state)

    if algorithm == "kmeans":
        labels = _run_kmeans(cluster_input, n_clusters, random_state)
    elif algorithm == "hdbscan":
        labels = _run_hdbscan(cluster_input, min_cluster_size)
    elif algorithm == "agglomerative":
        labels = _run_agglomerative(cluster_input, n_clusters, linkage, random_state)
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose: kmeans, hdbscan, agglomerative")

    result = df.copy()
    result["cluster_label"] = labels.astype(int)

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
        counter: Counter = Counter()
        for skills in subset["all_skills"].dropna():
            counter.update(skills)
        tag = "(noise)" if cid == -1 else ""
        top5 = ", ".join(f"{s}({n})" for s, n in counter.most_common(5))
        logger.info(f"  Cluster {cid}{tag} (n={len(subset)}): {top5 or '—'}")


def run_clustering(
    input_path: Path,
    output_path: Path,
    source_type: str,
    mode: RepresentationMode = "embedding",
    algorithm: ClusteringAlgorithm = "kmeans",
    n_clusters: int = 8,
    min_cluster_size: int = 5,
    linkage: str = "ward",
    use_umap: bool = True,
) -> pd.DataFrame:
    """
    Load a parquet, cluster the records of one source_type, save with cluster columns.
    Records of other source types are passed through unchanged.
    """
    logger.info(f"Loading {input_path}")
    df = pd.read_parquet(input_path)

    if "source_type" in df.columns:
        mask = df["source_type"] == source_type
        target = df[mask].copy()
        rest = df[~mask].copy()
        logger.info(f"  Clustering {len(target)} '{source_type}' records")
    else:
        target = df.copy()
        rest = pd.DataFrame()

    if target.empty:
        logger.warning(f"No '{source_type}' records found in {input_path}")
        return df

    clustered = fit_clusters(
        target,
        mode=mode,
        algorithm=algorithm,
        n_clusters=n_clusters,
        min_cluster_size=min_cluster_size,
        linkage=linkage,
        use_umap=use_umap,
    )

    result = pd.concat([clustered, rest], ignore_index=True) if not rest.empty else clustered
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    logger.info(f"Saved → {output_path}")
    return result
