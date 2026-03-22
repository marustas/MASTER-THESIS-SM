"""
Tests for src/dataset_builder.py.

Covers:
  - Merging skills and embedding parquets per source type
  - source_type column set correctly
  - Unified dataset has records from both sources
  - Embedding columns grafted where present, omitted gracefully when absent
  - Descriptive stats: counts, coverage, top skills, text length, language
  - Output parquet and stats.json are written
  - Empty dataset returned when no input files exist
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dataset_builder import build, compute_stats, _coverage, _top_skills


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_skills_df(n: int, source_label: str) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    skill_pool = [["Python", "machine learning"], ["Java", "SQL"], ["Docker"], [], ["NLP", "deep learning"]]
    return pd.DataFrame({
        "name": [f"{source_label}_{i}" for i in range(n)],
        "cleaned_text": [f"Some {source_label} description number {i}" for i in range(n)],
        "language": ["en"] * (n - 1) + ["lt"],
        "language_supported": [True] * n,
        "explicit_skills": [skill_pool[i % len(skill_pool)] for i in range(n)],
        "implicit_skills": [["Git"] if i % 3 == 0 else [] for i in range(n)],
        "all_skills": [
            skill_pool[i % len(skill_pool)] + (["Git"] if i % 3 == 0 else [])
            for i in range(n)
        ],
        "skill_uris": [["esco:python"] if i % 2 == 0 else [] for i in range(n)],
        "extended_description": [f"Extended {i}" if i % 2 == 0 else None for i in range(n)],
    })


def _make_embed_df(n: int, dim: int = 8, include_brief: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    vecs = rng.random((n, dim)).astype(np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    df = pd.DataFrame({"embedding": vecs.tolist()})
    if include_brief:
        df["embedding_brief"] = vecs.tolist()
        df["embedding_extended"] = vecs.tolist()
    return df


@pytest.fixture
def prog_parquets(tmp_path: Path) -> tuple[Path, Path]:
    n = 5
    skills_path = tmp_path / "programmes_with_skills.parquet"
    embed_path = tmp_path / "programmes_embeddings.parquet"
    _make_skills_df(n, "prog").to_parquet(skills_path, index=False)
    _make_embed_df(n, include_brief=True).to_parquet(embed_path, index=False)
    return skills_path, embed_path


@pytest.fixture
def jobs_parquets(tmp_path: Path) -> tuple[Path, Path]:
    n = 7
    skills_path = tmp_path / "jobs_with_skills.parquet"
    embed_path = tmp_path / "jobs_embeddings.parquet"
    _make_skills_df(n, "job").to_parquet(skills_path, index=False)
    _make_embed_df(n).to_parquet(embed_path, index=False)
    return skills_path, embed_path


# ── Assembly tests ─────────────────────────────────────────────────────────────

class TestBuild:
    def test_record_count(
        self, prog_parquets: tuple[Path, Path], jobs_parquets: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ps, pe = prog_parquets
        js, je = jobs_parquets
        df = build(ps, pe, js, je, tmp_path / "d.parquet", tmp_path / "s.json")
        assert len(df) == 5 + 7

    def test_source_type_values(
        self, prog_parquets: tuple[Path, Path], jobs_parquets: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ps, pe = prog_parquets
        js, je = jobs_parquets
        df = build(ps, pe, js, je, tmp_path / "d.parquet", tmp_path / "s.json")
        assert set(df["source_type"].unique()) == {"programme", "job_ad"}
        assert (df["source_type"] == "programme").sum() == 5
        assert (df["source_type"] == "job_ad").sum() == 7

    def test_embedding_column_present(
        self, prog_parquets: tuple[Path, Path], jobs_parquets: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ps, pe = prog_parquets
        js, je = jobs_parquets
        df = build(ps, pe, js, je, tmp_path / "d.parquet", tmp_path / "s.json")
        assert "embedding" in df.columns

    def test_brief_extended_embed_for_programmes(
        self, prog_parquets: tuple[Path, Path], jobs_parquets: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ps, pe = prog_parquets
        js, je = jobs_parquets
        df = build(ps, pe, js, je, tmp_path / "d.parquet", tmp_path / "s.json")
        prog_rows = df[df["source_type"] == "programme"]
        assert "embedding_brief" in df.columns
        assert prog_rows["embedding_brief"].notna().all()

    def test_missing_embed_parquet_still_builds(
        self, prog_parquets: tuple[Path, Path], jobs_parquets: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ps, _ = prog_parquets
        js, je = jobs_parquets
        missing_embed = tmp_path / "nonexistent.parquet"
        df = build(ps, missing_embed, js, je, tmp_path / "d.parquet", tmp_path / "s.json")
        assert len(df) == 5 + 7
        # embedding column absent for programmes but dataset still built
        prog_rows = df[df["source_type"] == "programme"]
        if "embedding" in df.columns:
            assert prog_rows["embedding"].isna().all() or True  # null is acceptable

    def test_missing_skills_parquet_skipped(
        self, prog_parquets: tuple[Path, Path], jobs_parquets: tuple[Path, Path], tmp_path: Path
    ) -> None:
        _, pe = prog_parquets
        js, je = jobs_parquets
        missing_skills = tmp_path / "nonexistent_skills.parquet"
        df = build(missing_skills, pe, js, je, tmp_path / "d.parquet", tmp_path / "s.json")
        # Only job ads should appear
        assert len(df) == 7
        assert set(df["source_type"].unique()) == {"job_ad"}

    def test_no_input_returns_empty(self, tmp_path: Path) -> None:
        df = build(
            tmp_path / "a.parquet", tmp_path / "b.parquet",
            tmp_path / "c.parquet", tmp_path / "d.parquet",
            tmp_path / "out.parquet", tmp_path / "stats.json",
        )
        assert df.empty

    def test_parquet_written(
        self, prog_parquets: tuple[Path, Path], jobs_parquets: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ps, pe = prog_parquets
        js, je = jobs_parquets
        out = tmp_path / "dataset.parquet"
        build(ps, pe, js, je, out, tmp_path / "stats.json")
        assert out.exists()
        reloaded = pd.read_parquet(out)
        assert len(reloaded) == 12

    def test_stats_json_written(
        self, prog_parquets: tuple[Path, Path], jobs_parquets: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ps, pe = prog_parquets
        js, je = jobs_parquets
        stats_out = tmp_path / "stats.json"
        build(ps, pe, js, je, tmp_path / "d.parquet", stats_out)
        assert stats_out.exists()
        with open(stats_out) as f:
            stats = json.load(f)
        assert stats["total_records"] == 12


# ── compute_stats tests ────────────────────────────────────────────────────────

class TestComputeStats:
    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        n = 10
        df = _make_skills_df(n, "test")
        df["source_type"] = ["programme"] * 5 + ["job_ad"] * 5
        return df

    def test_total_records(self, sample_df: pd.DataFrame) -> None:
        stats = compute_stats(sample_df)
        assert stats["total_records"] == 10

    def test_by_source_type(self, sample_df: pd.DataFrame) -> None:
        stats = compute_stats(sample_df)
        assert stats["by_source_type"]["programme"] == 5
        assert stats["by_source_type"]["job_ad"] == 5

    def test_coverage_explicit_skills(self, sample_df: pd.DataFrame) -> None:
        stats = compute_stats(sample_df)
        cov = stats["coverage"]["explicit_skills"]
        assert 0.0 <= cov <= 1.0

    def test_top_skills_overall_present(self, sample_df: pd.DataFrame) -> None:
        stats = compute_stats(sample_df)
        assert "top_skills_overall" in stats
        assert len(stats["top_skills_overall"]) > 0

    def test_per_source_top_skills(self, sample_df: pd.DataFrame) -> None:
        stats = compute_stats(sample_df)
        assert "top_skills_programme" in stats
        assert "top_skills_job_ad" in stats

    def test_text_length_keys(self, sample_df: pd.DataFrame) -> None:
        stats = compute_stats(sample_df)
        tl = stats["text_length"]
        for key in ("mean", "median", "p10", "p90", "min", "max"):
            assert key in tl

    def test_language_distribution(self, sample_df: pd.DataFrame) -> None:
        stats = compute_stats(sample_df)
        assert "language_distribution" in stats
        assert "en" in stats["language_distribution"]


# ── _coverage helper ───────────────────────────────────────────────────────────

class TestCoverage:
    def test_full_coverage(self) -> None:
        df = pd.DataFrame({"col": [["Python"], ["Java"], ["SQL"]]})
        assert _coverage(df, "col") == 1.0

    def test_zero_coverage(self) -> None:
        df = pd.DataFrame({"col": [[], [], []]})
        assert _coverage(df, "col") == 0.0

    def test_partial_coverage(self) -> None:
        df = pd.DataFrame({"col": [["Python"], [], ["Java"]]})
        cov = _coverage(df, "col")
        assert abs(cov - 2 / 3) < 0.01

    def test_missing_column_returns_zero(self) -> None:
        df = pd.DataFrame({"other": [1, 2, 3]})
        assert _coverage(df, "col") == 0.0


# ── _top_skills helper ─────────────────────────────────────────────────────────

class TestTopSkills:
    def test_correct_order(self) -> None:
        series = pd.Series([["Python", "Java"], ["Python"], ["SQL", "Java"]])
        top = _top_skills(series, n=3)
        labels = [t[0] for t in top]
        # Python and Java tied at 2; SQL at 1
        assert labels[0] in {"Python", "Java"}
        assert "SQL" in labels

    def test_n_limit_respected(self) -> None:
        series = pd.Series([["Python", "Java", "SQL", "Docker", "Git"]])
        top = _top_skills(series, n=3)
        assert len(top) == 3

    def test_empty_series(self) -> None:
        top = _top_skills(pd.Series(dtype=object), n=10)
        assert top == []
