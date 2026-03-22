"""
Tests for src/embeddings/generator.py.

All tests use a MockEmbeddingModel (same pattern as skill tests) so no
network access or model download is required.

Covers:
  - embed_texts: shape, L2-normalisation, empty/None text handling
  - embed_programmes: expected columns added, correct shapes
  - embed_job_ads: expected column added, correct shape
  - Zero-vector rows for empty text fields
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.skills.conftest import MockEmbeddingModel
from src.embeddings.generator import embed_texts, embed_programmes, embed_job_ads


# ── Shared fixture ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_model() -> MockEmbeddingModel:
    return MockEmbeddingModel()


# ── embed_texts ────────────────────────────────────────────────────────────────

class TestEmbedTexts:
    def test_output_shape(self, mock_model: MockEmbeddingModel) -> None:
        texts = ["Python developer", "Java SQL database", "machine learning"]
        result = embed_texts(mock_model, texts)
        assert result.shape == (3, MockEmbeddingModel.DIM)

    def test_l2_normalised(self, mock_model: MockEmbeddingModel) -> None:
        texts = ["Python developer", "Docker Kubernetes cloud"]
        result = embed_texts(mock_model, texts)
        norms = np.linalg.norm(result, axis=1)
        for norm in norms:
            assert abs(norm - 1.0) < 1e-5, f"Norm {norm} not ~1.0"

    def test_empty_string_is_zero_vector(self, mock_model: MockEmbeddingModel) -> None:
        result = embed_texts(mock_model, [""])
        assert result.shape == (1, MockEmbeddingModel.DIM)
        assert np.all(result[0] == 0.0), "Empty text should yield a zero vector"

    def test_none_treated_as_empty(self, mock_model: MockEmbeddingModel) -> None:
        result = embed_texts(mock_model, [None])  # type: ignore[list-item]
        assert np.all(result[0] == 0.0)

    def test_mixed_empty_and_non_empty(self, mock_model: MockEmbeddingModel) -> None:
        texts = ["Python developer", "", "SQL database"]
        result = embed_texts(mock_model, texts)
        assert result.shape == (3, MockEmbeddingModel.DIM)
        assert np.all(result[1] == 0.0), "Middle empty row should be zero"
        assert np.linalg.norm(result[0]) > 0
        assert np.linalg.norm(result[2]) > 0

    def test_all_empty_returns_zero_matrix(self, mock_model: MockEmbeddingModel) -> None:
        result = embed_texts(mock_model, ["", "  ", None])  # type: ignore[list-item]
        assert result.shape[0] == 3
        assert np.all(result == 0.0)

    def test_cosine_similarity_consistent(self, mock_model: MockEmbeddingModel) -> None:
        """Similar texts should have higher dot product than dissimilar ones."""
        texts = [
            "Python developer machine learning",
            "Python engineer ML background",
            "Java SQL database backend",
        ]
        result = embed_texts(mock_model, texts)
        sim_0_1 = float(result[0] @ result[1])
        sim_0_2 = float(result[0] @ result[2])
        assert sim_0_1 > sim_0_2, (
            f"Expected Python/ML texts to be more similar to each other "
            f"({sim_0_1:.3f}) than to Java/SQL ({sim_0_2:.3f})"
        )


# ── embed_programmes ───────────────────────────────────────────────────────────

class TestEmbedProgrammes:
    @pytest.fixture
    def sample_programmes_df(self, tmp_path: "Path") -> "Path":
        df = pd.DataFrame([
            {
                "name": "Computer Science BSc",
                "institution": "TU Vilnius",
                "cleaned_text": "Python machine learning data analysis",
                "brief_description": "A programme in CS and AI.",
                "extended_description": "Covers Python, ML, data engineering and cloud computing.",
            },
            {
                "name": "Software Engineering BSc",
                "institution": "KTU",
                "cleaned_text": "Java SQL software development agile",
                "brief_description": "Software engineering fundamentals.",
                "extended_description": "Java, SQL, software development, agile methodology.",
            },
            {
                "name": "AI MSc",
                "institution": "VU",
                "cleaned_text": "deep learning natural language processing neural networks",
                "brief_description": None,
                "extended_description": "",
            },
        ])
        path = tmp_path / "programmes_preprocessed.parquet"
        df.to_parquet(path, index=False)
        return path

    def test_embedding_column_added(
        self, mock_model: MockEmbeddingModel, sample_programmes_df: "Path", tmp_path: "Path"
    ) -> None:
        out = tmp_path / "out_programmes.parquet"
        result_df = embed_programmes(mock_model, sample_programmes_df, out)
        assert "embedding" in result_df.columns

    def test_brief_and_extended_columns_added(
        self, mock_model: MockEmbeddingModel, sample_programmes_df: "Path", tmp_path: "Path"
    ) -> None:
        out = tmp_path / "out_programmes2.parquet"
        result_df = embed_programmes(mock_model, sample_programmes_df, out)
        assert "embedding_brief" in result_df.columns
        assert "embedding_extended" in result_df.columns

    def test_embedding_dimension_correct(
        self, mock_model: MockEmbeddingModel, sample_programmes_df: "Path", tmp_path: "Path"
    ) -> None:
        out = tmp_path / "out_programmes3.parquet"
        result_df = embed_programmes(mock_model, sample_programmes_df, out)
        for emb in result_df["embedding"]:
            assert len(emb) == MockEmbeddingModel.DIM

    def test_row_count_preserved(
        self, mock_model: MockEmbeddingModel, sample_programmes_df: "Path", tmp_path: "Path"
    ) -> None:
        out = tmp_path / "out_programmes4.parquet"
        result_df = embed_programmes(mock_model, sample_programmes_df, out)
        assert len(result_df) == 3

    def test_parquet_saved(
        self, mock_model: MockEmbeddingModel, sample_programmes_df: "Path", tmp_path: "Path"
    ) -> None:
        out = tmp_path / "out_programmes5.parquet"
        embed_programmes(mock_model, sample_programmes_df, out)
        assert out.exists()
        reloaded = pd.read_parquet(out)
        assert "embedding" in reloaded.columns


# ── embed_job_ads ──────────────────────────────────────────────────────────────

class TestEmbedJobAds:
    @pytest.fixture
    def sample_jobs_df(self, tmp_path: "Path") -> "Path":
        df = pd.DataFrame([
            {
                "job_title": "Python Developer",
                "cleaned_text": "Python developer machine learning experience required.",
                "language": "en",
            },
            {
                "job_title": "Java Backend Engineer",
                "cleaned_text": "Java SQL agile software development.",
                "language": "en",
            },
            {
                "job_title": "Data Scientist",
                "cleaned_text": "",
                "language": "en",
            },
        ])
        path = tmp_path / "jobs_preprocessed.parquet"
        df.to_parquet(path, index=False)
        return path

    def test_embedding_column_added(
        self, mock_model: MockEmbeddingModel, sample_jobs_df: "Path", tmp_path: "Path"
    ) -> None:
        out = tmp_path / "out_jobs.parquet"
        result_df = embed_job_ads(mock_model, sample_jobs_df, out)
        assert "embedding" in result_df.columns

    def test_no_extra_columns(
        self, mock_model: MockEmbeddingModel, sample_jobs_df: "Path", tmp_path: "Path"
    ) -> None:
        """Job ads should not get brief/extended columns."""
        out = tmp_path / "out_jobs2.parquet"
        result_df = embed_job_ads(mock_model, sample_jobs_df, out)
        assert "embedding_brief" not in result_df.columns
        assert "embedding_extended" not in result_df.columns

    def test_empty_text_yields_zero_vector(
        self, mock_model: MockEmbeddingModel, sample_jobs_df: "Path", tmp_path: "Path"
    ) -> None:
        out = tmp_path / "out_jobs3.parquet"
        result_df = embed_job_ads(mock_model, sample_jobs_df, out)
        empty_row_emb = np.array(result_df["embedding"].iloc[2])
        assert np.all(empty_row_emb == 0.0), "Empty cleaned_text should yield zero vector"

    def test_row_count_preserved(
        self, mock_model: MockEmbeddingModel, sample_jobs_df: "Path", tmp_path: "Path"
    ) -> None:
        out = tmp_path / "out_jobs4.parquet"
        result_df = embed_job_ads(mock_model, sample_jobs_df, out)
        assert len(result_df) == 3

    def test_parquet_saved(
        self, mock_model: MockEmbeddingModel, sample_jobs_df: "Path", tmp_path: "Path"
    ) -> None:
        out = tmp_path / "out_jobs5.parquet"
        embed_job_ads(mock_model, sample_jobs_df, out)
        assert out.exists()
        reloaded = pd.read_parquet(out)
        assert "embedding" in reloaded.columns
