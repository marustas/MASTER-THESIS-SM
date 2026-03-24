"""
Tests for src/alignment/semantic.py.

Covers:
  - _to_matrix: stacks list-column into float32 matrix
  - _pairwise_scores: cosine and dot shapes, value range, symmetry
  - cosine == dot for L2-normalised input
  - align_semantic: output shape, column presence, sort order,
                    brief/extended NaN when columns absent,
                    brief/extended scores computed when columns present
  - run_semantic_alignment: output files written, rankings shape
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.alignment.semantic import (
    _pairwise_scores,
    _to_matrix,
    align_semantic,
    run_semantic_alignment,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _l2(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _make_embedding(seed: int, dim: int = 8) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return _l2(v).tolist()


def _make_df(
    n_prog: int,
    n_jobs: int,
    dim: int = 8,
    include_brief: bool = False,
    include_extended: bool = False,
) -> pd.DataFrame:
    rows = []
    for i in range(n_prog):
        row = {
            "source_type": "programme",
            "embedding": _make_embedding(i, dim),
            "name": f"Prog{i}",
        }
        if include_brief:
            row["embedding_brief"] = _make_embedding(i + 100, dim)
        if include_extended:
            row["embedding_extended"] = _make_embedding(i + 200, dim)
        rows.append(row)
    for i in range(n_jobs):
        rows.append({
            "source_type": "job_ad",
            "embedding": _make_embedding(i + 50, dim),
            "job_title": f"Job{i}",
        })
    return pd.DataFrame(rows)


# ── _to_matrix ─────────────────────────────────────────────────────────────────

class TestToMatrix:
    def test_shape(self):
        series = pd.Series([[1.0, 2.0], [3.0, 4.0]])
        mat = _to_matrix(series)
        assert mat.shape == (2, 2)
        assert mat.dtype == np.float32

    def test_values_preserved(self):
        series = pd.Series([[1.0, 0.0], [0.0, 1.0]])
        mat = _to_matrix(series)
        np.testing.assert_allclose(mat[0], [1.0, 0.0])
        np.testing.assert_allclose(mat[1], [0.0, 1.0])


# ── _pairwise_scores ───────────────────────────────────────────────────────────

class TestPairwiseScores:
    def _rand_l2(self, n: int, dim: int = 8, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        mat = rng.random((n, dim)).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / norms

    def test_output_shapes(self):
        p = self._rand_l2(3)
        j = self._rand_l2(5)
        cosine, dot = _pairwise_scores(p, j)
        assert cosine.shape == (3, 5)
        assert dot.shape == (3, 5)

    def test_cosine_range(self):
        p = self._rand_l2(4)
        j = self._rand_l2(6)
        cosine, _ = _pairwise_scores(p, j)
        assert (cosine >= -1.0 - 1e-5).all()
        assert (cosine <= 1.0 + 1e-5).all()

    def test_cosine_equals_dot_for_normalised(self):
        """For L2-normalised input, cosine and dot must be numerically identical."""
        p = self._rand_l2(4, seed=1)
        j = self._rand_l2(6, seed=2)
        cosine, dot = _pairwise_scores(p, j)
        np.testing.assert_allclose(cosine, dot, atol=1e-5)

    def test_self_similarity_is_one(self):
        p = self._rand_l2(3)
        cosine, dot = _pairwise_scores(p, p)
        np.testing.assert_allclose(np.diag(cosine), np.ones(3), atol=1e-5)
        np.testing.assert_allclose(np.diag(dot), np.ones(3), atol=1e-5)

    def test_orthogonal_vectors_score_zero(self):
        p = np.array([[1.0, 0.0]], dtype=np.float32)
        j = np.array([[0.0, 1.0]], dtype=np.float32)
        cosine, dot = _pairwise_scores(p, j)
        assert cosine[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert dot[0, 0] == pytest.approx(0.0, abs=1e-6)


# ── align_semantic ─────────────────────────────────────────────────────────────

class TestAlignSemantic:
    def test_output_shape(self):
        df = _make_df(2, 3)
        rankings = align_semantic(df)
        assert len(rankings) == 6  # 2 × 3

    def test_required_columns_present(self):
        rankings = align_semantic(_make_df(2, 3))
        for col in (
            "programme_id", "job_id",
            "cosine_combined", "dot_combined",
            "cosine_brief", "dot_brief",
            "cosine_extended", "dot_extended",
        ):
            assert col in rankings.columns

    def test_brief_extended_nan_when_absent(self):
        df = _make_df(2, 3, include_brief=False, include_extended=False)
        rankings = align_semantic(df)
        assert rankings["cosine_brief"].isna().all()
        assert rankings["cosine_extended"].isna().all()

    def test_brief_scores_present_when_column_exists(self):
        df = _make_df(2, 3, include_brief=True)
        rankings = align_semantic(df)
        assert rankings["cosine_brief"].notna().any()

    def test_extended_scores_present_when_column_exists(self):
        df = _make_df(2, 3, include_extended=True)
        rankings = align_semantic(df)
        assert rankings["cosine_extended"].notna().any()

    def test_cosine_equals_dot_for_normalised_embeddings(self):
        df = _make_df(3, 4)
        rankings = align_semantic(df)
        valid = rankings["cosine_combined"].notna() & rankings["dot_combined"].notna()
        np.testing.assert_allclose(
            rankings.loc[valid, "cosine_combined"].values,
            rankings.loc[valid, "dot_combined"].values,
            atol=1e-5,
        )

    def test_sorted_descending_within_programme(self):
        df = _make_df(3, 5)
        rankings = align_semantic(df)
        for p_id in rankings["programme_id"].unique():
            scores = rankings[rankings["programme_id"] == p_id]["cosine_combined"].tolist()
            assert scores == sorted(scores, reverse=True)

    def test_scores_in_valid_range(self):
        df = _make_df(2, 4)
        rankings = align_semantic(df)
        valid = rankings["cosine_combined"].dropna()
        assert (valid >= -1.0 - 1e-5).all()
        assert (valid <= 1.0 + 1e-5).all()

    def test_identical_embedding_scores_one(self):
        """A programme with the same embedding as a job should score 1.0."""
        vec = _make_embedding(42)
        df = pd.DataFrame([
            {"source_type": "programme", "embedding": vec, "name": "P0"},
            {"source_type": "job_ad",    "embedding": vec, "job_title": "J0"},
        ])
        rankings = align_semantic(df)
        assert rankings.iloc[0]["cosine_combined"] == pytest.approx(1.0, abs=1e-5)

    def test_n_pairs_equals_prog_times_jobs(self):
        df = _make_df(4, 7)
        rankings = align_semantic(df)
        assert len(rankings) == 4 * 7


# ── run_semantic_alignment ─────────────────────────────────────────────────────

class TestRunSemanticAlignment:
    def test_output_files_created(self, tmp_path):
        df = _make_df(2, 3)
        dataset_path = tmp_path / "dataset.parquet"
        df.to_parquet(dataset_path)
        output_dir = tmp_path / "exp2_semantic"
        run_semantic_alignment(
            dataset_path=dataset_path,
            output_dir=output_dir,
            top_n=3,
        )
        assert (output_dir / "rankings.parquet").exists()
        assert (output_dir / "summary.json").exists()

    def test_rankings_row_count(self, tmp_path):
        df = _make_df(2, 3)
        dataset_path = tmp_path / "dataset.parquet"
        df.to_parquet(dataset_path)
        output_dir = tmp_path / "exp2"
        run_semantic_alignment(dataset_path=dataset_path, output_dir=output_dir)
        rankings = pd.read_parquet(output_dir / "rankings.parquet")
        assert len(rankings) == 6  # 2 × 3
