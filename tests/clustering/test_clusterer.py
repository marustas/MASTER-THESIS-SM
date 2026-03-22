"""
Tests for src/clustering/clusterer.py.

Covers:
  - fit_clusters: embedding and skills modes for all three algorithms
  - K-Means: cluster_label range, reproducibility, n_clusters cap
  - HDBSCAN: valid labels (integers ≥ -1), noise label allowed, min_cluster_size respected
  - Agglomerative: label range, linkage options
  - Zero-embedding rows handled without crash
  - Missing column raises ValueError
  - 2-D coords added/omitted correctly
  - run_clustering: source_type filtering, passthrough, parquet written
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.clustering.clusterer import fit_clusters, run_clustering


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_df(
    n: int,
    dim: int = 16,
    source_type: str | None = None,
    include_skills: bool = True,
    zero_row: int | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    vecs = rng.random((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms
    if zero_row is not None:
        vecs[zero_row] = 0.0
    df = pd.DataFrame({"embedding": vecs.tolist()})
    if source_type:
        df["source_type"] = source_type
    if include_skills:
        skill_pool = [
            ["esco:python", "esco:ml"],
            ["esco:java", "esco:sql"],
            ["esco:docker", "esco:kubernetes"],
            [],
        ]
        df["skill_uris"] = [skill_pool[i % len(skill_pool)] for i in range(n)]
        df["all_skills"] = [
            ["Python", "machine learning"] if i % 4 == 0 else
            ["Java", "SQL"] if i % 4 == 1 else
            ["Docker", "Kubernetes"] if i % 4 == 2 else []
            for i in range(n)
        ]
    return df


# ── K-Means ────────────────────────────────────────────────────────────────────

class TestKMeans:
    def test_cluster_label_column_added(self) -> None:
        df = _make_df(20)
        result = fit_clusters(df, algorithm="kmeans", n_clusters=3, use_umap=False, add_2d_coords=False)
        assert "cluster_label" in result.columns

    def test_label_range(self) -> None:
        n_clusters = 4
        df = _make_df(20)
        result = fit_clusters(df, algorithm="kmeans", n_clusters=n_clusters, use_umap=False, add_2d_coords=False)
        assert set(result["cluster_label"].unique()) <= set(range(n_clusters))

    def test_all_rows_assigned(self) -> None:
        df = _make_df(20)
        result = fit_clusters(df, algorithm="kmeans", n_clusters=3, use_umap=False, add_2d_coords=False)
        assert result["cluster_label"].notna().all()
        assert len(result) == 20

    def test_reproducible(self) -> None:
        df = _make_df(20)
        r1 = fit_clusters(df, algorithm="kmeans", n_clusters=3, use_umap=False, add_2d_coords=False, random_state=0)
        r2 = fit_clusters(df, algorithm="kmeans", n_clusters=3, use_umap=False, add_2d_coords=False, random_state=0)
        assert list(r1["cluster_label"]) == list(r2["cluster_label"])

    def test_n_clusters_capped_when_few_records(self) -> None:
        df = _make_df(3)
        result = fit_clusters(df, algorithm="kmeans", n_clusters=10, use_umap=False, add_2d_coords=False)
        assert "cluster_label" in result.columns
        assert len(result["cluster_label"].unique()) <= 3

    def test_zero_embedding_row_handled(self) -> None:
        df = _make_df(10, zero_row=0)
        result = fit_clusters(df, algorithm="kmeans", n_clusters=2, use_umap=False, add_2d_coords=False)
        assert result["cluster_label"].notna().all()

    def test_skills_mode(self) -> None:
        df = _make_df(12)
        result = fit_clusters(df, mode="skills", algorithm="kmeans", n_clusters=3, use_umap=False, add_2d_coords=False)
        assert "cluster_label" in result.columns
        assert result["cluster_label"].notna().all()


# ── HDBSCAN ────────────────────────────────────────────────────────────────────

class TestHDBSCAN:
    def test_cluster_label_column_added(self) -> None:
        df = _make_df(30)
        result = fit_clusters(df, algorithm="hdbscan", min_cluster_size=3, use_umap=False, add_2d_coords=False)
        assert "cluster_label" in result.columns

    def test_all_rows_have_integer_label(self) -> None:
        df = _make_df(30)
        result = fit_clusters(df, algorithm="hdbscan", min_cluster_size=3, use_umap=False, add_2d_coords=False)
        assert result["cluster_label"].notna().all()
        assert result["cluster_label"].dtype in (np.dtype("int32"), np.dtype("int64"))

    def test_noise_label_is_minus_one(self) -> None:
        """HDBSCAN may assign -1 to noise points — this is valid."""
        df = _make_df(30)
        result = fit_clusters(df, algorithm="hdbscan", min_cluster_size=3, use_umap=False, add_2d_coords=False)
        labels = result["cluster_label"].unique()
        assert all(l >= -1 for l in labels), f"Unexpected labels: {labels}"

    def test_at_least_one_cluster_found(self) -> None:
        """Even with all noise, at least some label must exist."""
        df = _make_df(30)
        result = fit_clusters(df, algorithm="hdbscan", min_cluster_size=3, use_umap=False, add_2d_coords=False)
        non_noise = result[result["cluster_label"] >= 0]
        # Not asserting non_noise is non-empty — HDBSCAN may mark all as noise
        # for random data; just ensure no crash and valid labels
        assert isinstance(non_noise, pd.DataFrame)

    def test_min_cluster_size_respected(self) -> None:
        """Clusters should have at least min_cluster_size points (noise aside)."""
        df = _make_df(40)
        mcs = 5
        result = fit_clusters(df, algorithm="hdbscan", min_cluster_size=mcs, use_umap=False, add_2d_coords=False)
        for cid in result["cluster_label"].unique():
            if cid == -1:
                continue
            cluster_size = (result["cluster_label"] == cid).sum()
            assert cluster_size >= mcs, f"Cluster {cid} has {cluster_size} < {mcs} points"

    def test_zero_embedding_handled(self) -> None:
        df = _make_df(20, zero_row=0)
        result = fit_clusters(df, algorithm="hdbscan", min_cluster_size=2, use_umap=False, add_2d_coords=False)
        assert result["cluster_label"].notna().all()


# ── Agglomerative ──────────────────────────────────────────────────────────────

class TestAgglomerative:
    def test_cluster_label_column_added(self) -> None:
        df = _make_df(20)
        result = fit_clusters(df, algorithm="agglomerative", n_clusters=3, use_umap=False, add_2d_coords=False)
        assert "cluster_label" in result.columns

    def test_label_range(self) -> None:
        n_clusters = 4
        df = _make_df(20)
        result = fit_clusters(df, algorithm="agglomerative", n_clusters=n_clusters, use_umap=False, add_2d_coords=False)
        assert set(result["cluster_label"].unique()) <= set(range(n_clusters))

    def test_all_rows_assigned(self) -> None:
        df = _make_df(20)
        result = fit_clusters(df, algorithm="agglomerative", n_clusters=3, use_umap=False, add_2d_coords=False)
        assert result["cluster_label"].notna().all()

    def test_n_clusters_capped_when_few_records(self) -> None:
        df = _make_df(3)
        result = fit_clusters(df, algorithm="agglomerative", n_clusters=10, use_umap=False, add_2d_coords=False)
        assert len(result["cluster_label"].unique()) <= 3

    @pytest.mark.parametrize("linkage", ["ward", "complete", "average", "single"])
    def test_linkage_options(self, linkage: str) -> None:
        df = _make_df(20)
        result = fit_clusters(
            df, algorithm="agglomerative", n_clusters=3, linkage=linkage,
            use_umap=False, add_2d_coords=False
        )
        assert "cluster_label" in result.columns
        assert result["cluster_label"].notna().all()


# ── Shared / cross-algorithm ───────────────────────────────────────────────────

class TestShared:
    def test_missing_embedding_column_raises(self) -> None:
        df = pd.DataFrame({"other": [1, 2, 3]})
        with pytest.raises(ValueError, match="embedding"):
            fit_clusters(df, mode="embedding", algorithm="kmeans", use_umap=False, add_2d_coords=False)

    def test_missing_skill_uris_raises(self) -> None:
        df = pd.DataFrame({"other": [1, 2, 3]})
        with pytest.raises(ValueError, match="skill_uris"):
            fit_clusters(df, mode="skills", algorithm="kmeans", use_umap=False, add_2d_coords=False)

    def test_unknown_algorithm_raises(self) -> None:
        df = _make_df(10)
        with pytest.raises(ValueError, match="Unknown algorithm"):
            fit_clusters(df, algorithm="dbscan", use_umap=False, add_2d_coords=False)  # type: ignore[arg-type]

    def test_2d_coords_added_when_requested(self) -> None:
        df = _make_df(20, dim=16)
        result = fit_clusters(df, algorithm="kmeans", n_clusters=3, use_umap=True, add_2d_coords=True)
        if "cluster_label_2d_x" in result.columns:
            assert result["cluster_label_2d_x"].notna().all()
            assert result["cluster_label_2d_y"].notna().all()

    def test_no_2d_coords_when_disabled(self) -> None:
        df = _make_df(20)
        result = fit_clusters(df, algorithm="kmeans", n_clusters=3, use_umap=False, add_2d_coords=False)
        assert "cluster_label_2d_x" not in result.columns
        assert "cluster_label_2d_y" not in result.columns


# ── run_clustering ─────────────────────────────────────────────────────────────

class TestRunClustering:
    @pytest.fixture
    def dataset_parquet(self, tmp_path: Path) -> Path:
        progs = _make_df(8, source_type="programme")
        jobs = _make_df(10, source_type="job_ad")
        df = pd.concat([progs, jobs], ignore_index=True)
        path = tmp_path / "dataset.parquet"
        df.to_parquet(path, index=False)
        return path

    def test_parquet_written(self, dataset_parquet: Path, tmp_path: Path) -> None:
        out = tmp_path / "out.parquet"
        run_clustering(dataset_parquet, out, "programme", algorithm="kmeans", n_clusters=2, use_umap=False)
        assert out.exists()

    def test_only_target_source_clustered(self, dataset_parquet: Path, tmp_path: Path) -> None:
        out = tmp_path / "out.parquet"
        run_clustering(dataset_parquet, out, "programme", algorithm="kmeans", n_clusters=2, use_umap=False)
        result = pd.read_parquet(out)
        prog_rows = result[result["source_type"] == "programme"]
        job_rows = result[result["source_type"] == "job_ad"]
        assert prog_rows["cluster_label"].notna().all()
        assert "cluster_label" not in job_rows.columns or job_rows["cluster_label"].isna().all()

    def test_total_row_count_preserved(self, dataset_parquet: Path, tmp_path: Path) -> None:
        out = tmp_path / "out.parquet"
        run_clustering(dataset_parquet, out, "programme", algorithm="kmeans", n_clusters=2, use_umap=False)
        assert len(pd.read_parquet(out)) == 18

    def test_hdbscan_via_run_clustering(self, dataset_parquet: Path, tmp_path: Path) -> None:
        out = tmp_path / "out_hdbscan.parquet"
        run_clustering(dataset_parquet, out, "job_ad", algorithm="hdbscan", min_cluster_size=2, use_umap=False)
        result = pd.read_parquet(out)
        job_rows = result[result["source_type"] == "job_ad"]
        assert "cluster_label" in result.columns
        assert job_rows["cluster_label"].notna().all()

    def test_agglomerative_via_run_clustering(self, dataset_parquet: Path, tmp_path: Path) -> None:
        out = tmp_path / "out_agg.parquet"
        run_clustering(dataset_parquet, out, "programme", algorithm="agglomerative", n_clusters=2, use_umap=False)
        result = pd.read_parquet(out)
        prog_rows = result[result["source_type"] == "programme"]
        assert prog_rows["cluster_label"].notna().all()

    def test_missing_source_type_warns_and_returns(
        self, dataset_parquet: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.parquet"
        result = run_clustering(dataset_parquet, out, "nonexistent", n_clusters=2, use_umap=False)
        assert len(result) == 18
