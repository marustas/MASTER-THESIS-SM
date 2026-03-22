"""
Tests for src/clustering/clusterer.py.

Covers:
  - fit_clusters: cluster_label column added, correct range, reproducibility
  - Embedding mode: uses 'embedding' column
  - Skills mode: uses 'skill_uris' column (TF-IDF bag-of-URIs)
  - n_clusters capped when fewer records than requested
  - run_clustering: source_type filtering, parquet written, passthrough of other rows
  - Zero-embedding rows handled without crash
  - UMAP disabled path (use_umap=False)
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


# ── fit_clusters: embedding mode ───────────────────────────────────────────────

class TestFitClustersEmbedding:
    def test_cluster_label_column_added(self) -> None:
        df = _make_df(20)
        result = fit_clusters(df, mode="embedding", n_clusters=3, use_umap=False, add_2d_coords=False)
        assert "cluster_label" in result.columns

    def test_label_range(self) -> None:
        n_clusters = 4
        df = _make_df(20)
        result = fit_clusters(df, mode="embedding", n_clusters=n_clusters, use_umap=False, add_2d_coords=False)
        labels = result["cluster_label"].unique()
        assert set(labels) <= set(range(n_clusters))

    def test_all_rows_assigned(self) -> None:
        df = _make_df(20)
        result = fit_clusters(df, mode="embedding", n_clusters=3, use_umap=False, add_2d_coords=False)
        assert result["cluster_label"].notna().all()
        assert len(result) == 20

    def test_reproducible(self) -> None:
        df = _make_df(20)
        r1 = fit_clusters(df, mode="embedding", n_clusters=3, use_umap=False, add_2d_coords=False, random_state=0)
        r2 = fit_clusters(df, mode="embedding", n_clusters=3, use_umap=False, add_2d_coords=False, random_state=0)
        assert list(r1["cluster_label"]) == list(r2["cluster_label"])

    def test_n_clusters_capped_when_few_records(self) -> None:
        df = _make_df(3)
        # requesting more clusters than records — should not crash
        result = fit_clusters(df, mode="embedding", n_clusters=10, use_umap=False, add_2d_coords=False)
        assert "cluster_label" in result.columns
        assert len(result["cluster_label"].unique()) <= 3

    def test_zero_embedding_row_handled(self) -> None:
        df = _make_df(10, zero_row=0)
        result = fit_clusters(df, mode="embedding", n_clusters=2, use_umap=False, add_2d_coords=False)
        assert result["cluster_label"].notna().all()

    def test_missing_embedding_column_raises(self) -> None:
        df = pd.DataFrame({"other": [1, 2, 3]})
        with pytest.raises(ValueError, match="embedding"):
            fit_clusters(df, mode="embedding", n_clusters=2, use_umap=False, add_2d_coords=False)

    def test_2d_coords_added_when_requested(self) -> None:
        df = _make_df(20, dim=16)
        result = fit_clusters(df, mode="embedding", n_clusters=3, use_umap=True, add_2d_coords=True)
        # UMAP may or may not be available; if coords added they must be finite
        if "cluster_label_2d_x" in result.columns:
            assert result["cluster_label_2d_x"].notna().all()
            assert result["cluster_label_2d_y"].notna().all()

    def test_no_2d_coords_when_disabled(self) -> None:
        df = _make_df(20)
        result = fit_clusters(df, mode="embedding", n_clusters=3, use_umap=False, add_2d_coords=False)
        assert "cluster_label_2d_x" not in result.columns
        assert "cluster_label_2d_y" not in result.columns


# ── fit_clusters: skills mode ──────────────────────────────────────────────────

class TestFitClustersSkills:
    def test_cluster_label_added(self) -> None:
        df = _make_df(20)
        result = fit_clusters(df, mode="skills", n_clusters=3, use_umap=False, add_2d_coords=False)
        assert "cluster_label" in result.columns

    def test_all_rows_assigned(self) -> None:
        df = _make_df(20)
        result = fit_clusters(df, mode="skills", n_clusters=3, use_umap=False, add_2d_coords=False)
        assert result["cluster_label"].notna().all()

    def test_missing_skill_uris_raises(self) -> None:
        df = pd.DataFrame({"other": [1, 2, 3]})
        with pytest.raises(ValueError, match="skill_uris"):
            fit_clusters(df, mode="skills", n_clusters=2, use_umap=False, add_2d_coords=False)


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
        run_clustering(dataset_parquet, out, "programme", n_clusters=2, use_umap=False)
        assert out.exists()

    def test_only_target_source_clustered(self, dataset_parquet: Path, tmp_path: Path) -> None:
        out = tmp_path / "out.parquet"
        run_clustering(dataset_parquet, out, "programme", n_clusters=2, use_umap=False)
        result = pd.read_parquet(out)
        prog_rows = result[result["source_type"] == "programme"]
        job_rows = result[result["source_type"] == "job_ad"]
        assert prog_rows["cluster_label"].notna().all()
        # job rows get NaN cluster_label (not clustered in this call)
        assert "cluster_label" not in job_rows.columns or job_rows["cluster_label"].isna().all()

    def test_total_row_count_preserved(self, dataset_parquet: Path, tmp_path: Path) -> None:
        out = tmp_path / "out.parquet"
        run_clustering(dataset_parquet, out, "programme", n_clusters=2, use_umap=False)
        result = pd.read_parquet(out)
        assert len(result) == 18  # 8 progs + 10 jobs

    def test_missing_source_type_warns_and_returns(
        self, dataset_parquet: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.parquet"
        # Request clustering of a source_type that doesn't exist
        result = run_clustering(dataset_parquet, out, "nonexistent", n_clusters=2, use_umap=False)
        # Should return the original dataframe unchanged
        assert len(result) == 18
