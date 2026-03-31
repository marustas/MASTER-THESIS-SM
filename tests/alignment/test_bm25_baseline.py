"""
Tests for src/alignment/bm25_baseline.py.

Covers:
  - tokenise: whitespace splitting, lowering, non-string input
  - align_bm25: ranking order, output shape, score column, matching text
  - run_bm25_alignment: output files created, summary structure
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.alignment.bm25_baseline import (
    tokenise,
    align_bm25,
    run_bm25_alignment,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_df(
    prog_texts: list[str],
    job_texts: list[str],
    prog_names: list[str] | None = None,
    job_titles: list[str] | None = None,
) -> pd.DataFrame:
    rows = []
    for i, text in enumerate(prog_texts):
        name = prog_names[i] if prog_names else f"Prog{i}"
        rows.append({"source_type": "programme", "cleaned_text": text, "name": name})
    for i, text in enumerate(job_texts):
        title = job_titles[i] if job_titles else f"Job{i}"
        rows.append({"source_type": "job_ad", "cleaned_text": text, "job_title": title})
    return pd.DataFrame(rows)


# ── tokenise ─────────────────────────────────────────────────────────────────

class TestTokenise:
    def test_basic(self):
        assert tokenise("Hello World") == ["hello", "world"]

    def test_empty_string(self):
        assert tokenise("") == []

    def test_non_string(self):
        assert tokenise(None) == []
        assert tokenise(42) == []

    def test_case_insensitive(self):
        assert tokenise("Python JAVA rust") == ["python", "java", "rust"]


# ── align_bm25 ───────────────────────────────────────────────────────────────

class TestAlignBm25:
    def test_output_shape(self):
        df = _make_df(
            ["python machine learning"],
            ["python developer", "java engineer"],
        )
        rankings = align_bm25(df)
        assert len(rankings) == 1 * 2  # 1 prog × 2 jobs
        assert "bm25_score" in rankings.columns

    def test_required_columns(self):
        df = _make_df(["data science"], ["data analyst"])
        rankings = align_bm25(df)
        for col in ("programme_id", "job_id", "programme_name", "job_title", "bm25_score"):
            assert col in rankings.columns

    def test_sorted_by_score_desc(self):
        df = _make_df(
            ["python python python"],
            ["python developer python", "java engineer"],
        )
        rankings = align_bm25(df)
        prog0 = rankings[rankings["programme_id"] == 0]
        scores = prog0["bm25_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_matching_text_scores_higher(self):
        df = _make_df(
            ["python data science"],
            ["python data science expert", "cooking baking restaurant"],
        )
        rankings = align_bm25(df)
        prog0 = rankings[rankings["programme_id"] == 0]
        top_job = prog0.iloc[0]["job_id"]
        assert top_job == 0  # python data science matches first job

    def test_multiple_programmes(self):
        df = _make_df(
            ["python programming", "cooking baking"],
            ["python programming language", "cooking baking chef", "unrelated topic xyz"],
        )
        rankings = align_bm25(df)
        assert len(rankings) == 2 * 3
        assert rankings["programme_id"].nunique() == 2
        # Programme 0 (python) should rank job 0 highest
        top0 = rankings[rankings["programme_id"] == 0].iloc[0]
        assert top0["job_id"] == 0
        # Programme 1 (cooking) should rank job 1 highest
        top1 = rankings[rankings["programme_id"] == 1].iloc[0]
        assert top1["job_id"] == 1

    def test_no_overlap_zero_score(self):
        df = _make_df(
            ["aardvark xylophone"],
            ["python developer machine learning"],
        )
        rankings = align_bm25(df)
        assert rankings.iloc[0]["bm25_score"] == 0.0

    def test_programme_names_preserved(self):
        df = _make_df(
            ["ai course"],
            ["ai job"],
            prog_names=["AI Engineering"],
            job_titles=["AI Developer"],
        )
        rankings = align_bm25(df)
        assert rankings.iloc[0]["programme_name"] == "AI Engineering"
        assert rankings.iloc[0]["job_title"] == "AI Developer"

    def test_custom_k1_b(self):
        # Varying doc lengths so b parameter has an effect
        df = _make_df(
            ["python"],
            [
                "python developer senior backend infrastructure cloud",
                "java developer",
                "rust engineer",
                "go programmer",
            ],
        )
        r1 = align_bm25(df, k1=1.5, b=0.75)
        r2 = align_bm25(df, k1=1.5, b=0.0)
        s1 = r1[r1["job_id"] == 0].iloc[0]["bm25_score"]
        s2 = r2[r2["job_id"] == 0].iloc[0]["bm25_score"]
        # Both should score > 0 and differ because b changes length normalisation
        assert s1 > 0.0
        assert s2 > 0.0
        assert s1 != s2

    def test_scores_non_negative(self):
        df = _make_df(
            ["python data analysis"],
            ["python developer", "data engineer", "java programmer"],
        )
        rankings = align_bm25(df)
        assert (rankings["bm25_score"] >= 0).all()


# ── run_bm25_alignment ──────────────────────────────────────────────────────

class TestRunBm25Alignment:
    def test_output_files_created(self, tmp_path: Path):
        df = _make_df(
            ["python machine learning", "data science statistics"],
            ["python developer", "data analyst", "java engineer"],
        )
        ds_path = tmp_path / "dataset.parquet"
        df.to_parquet(ds_path)
        out_dir = tmp_path / "output"

        run_bm25_alignment(dataset_path=ds_path, output_dir=out_dir)

        assert (out_dir / "rankings.parquet").exists()
        assert (out_dir / "summary.json").exists()

    def test_summary_structure(self, tmp_path: Path):
        df = _make_df(["ai"], ["ai job", "ml job"])
        ds_path = tmp_path / "ds.parquet"
        df.to_parquet(ds_path)
        out_dir = tmp_path / "out"

        run_bm25_alignment(dataset_path=ds_path, output_dir=out_dir)

        with open(out_dir / "summary.json") as f:
            data = json.load(f)
        assert data["n_programmes"] == 1
        assert data["n_jobs"] == 2
        assert "bm25_score_all" in data
        assert "mean" in data["bm25_score_all"]

    def test_rankings_parquet_readable(self, tmp_path: Path):
        df = _make_df(["test"], ["test job"])
        df.to_parquet(tmp_path / "ds.parquet")
        run_bm25_alignment(
            dataset_path=tmp_path / "ds.parquet",
            output_dir=tmp_path / "out",
        )
        result = pd.read_parquet(tmp_path / "out" / "rankings.parquet")
        assert "bm25_score" in result.columns
        assert len(result) == 1
