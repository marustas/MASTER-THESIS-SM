"""
Tests for src/alignment/symbolic.py.

Covers:
  - _build_weighted_skills: explicit/implicit/mixed URIs, empty input, dedup by max weight
  - weighted_jaccard: both empty, one empty, identical sets, partial overlap, no overlap
  - overlap_coefficient: both empty, one empty, subset, partial overlap
  - align_symbolic: ranking order, skill gap correctness, no-skills graceful handling
  - run_symbolic_alignment: end-to-end with tmp_path, output files written
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.alignment.symbolic import (
    _build_weighted_skills,
    align_symbolic,
    overlap_coefficient,
    run_symbolic_alignment,
    weighted_jaccard,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _skill(uri: str, *, explicit: bool = True, implicit: bool = False) -> dict:
    return {
        "esco_uri": uri,
        "preferred_label": uri.split(":")[-1],
        "matched_text": uri.split(":")[-1],
        "explicit": explicit,
        "implicit": implicit,
        "confidence": 1.0,
    }


def _make_df(
    programmes: list[list[dict]],
    jobs: list[list[dict]],
    prog_names: list[str] | None = None,
    job_titles: list[str] | None = None,
) -> pd.DataFrame:
    """Build a minimal unified dataset DataFrame."""
    rows = []
    for i, details in enumerate(programmes):
        name = prog_names[i] if prog_names else f"Prog{i}"
        rows.append({"source_type": "programme", "skill_details": details, "name": name})
    for i, details in enumerate(jobs):
        title = job_titles[i] if job_titles else f"Job{i}"
        rows.append({"source_type": "job_ad", "skill_details": details, "job_title": title})
    return pd.DataFrame(rows)


# ── _build_weighted_skills ─────────────────────────────────────────────────────

class TestBuildWeightedSkills:
    def test_explicit_skill_gets_weight_1(self):
        details = [_skill("esco:python", explicit=True)]
        result = _build_weighted_skills(details)
        assert result == {"esco:python": 1.0}

    def test_implicit_skill_gets_weight_05(self):
        details = [_skill("esco:sql", explicit=False, implicit=True)]
        result = _build_weighted_skills(details)
        assert result == {"esco:sql": 0.5}

    def test_explicit_wins_over_implicit_for_same_uri(self):
        details = [
            _skill("esco:ml", explicit=False, implicit=True),
            _skill("esco:ml", explicit=True, implicit=False),
        ]
        result = _build_weighted_skills(details)
        assert result["esco:ml"] == 1.0

    def test_empty_details_returns_empty(self):
        assert _build_weighted_skills([]) == {}

    def test_skill_without_uri_is_ignored(self):
        details = [{"esco_uri": "", "explicit": True, "implicit": False}]
        assert _build_weighted_skills(details) == {}

    def test_multiple_distinct_uris(self):
        details = [
            _skill("esco:python", explicit=True),
            _skill("esco:docker", explicit=False, implicit=True),
        ]
        result = _build_weighted_skills(details)
        assert result["esco:python"] == 1.0
        assert result["esco:docker"] == 0.5


# ── weighted_jaccard ───────────────────────────────────────────────────────────

class TestWeightedJaccard:
    def test_both_empty_returns_zero(self):
        assert weighted_jaccard({}, {}) == 0.0

    def test_one_empty_returns_zero(self):
        assert weighted_jaccard({"esco:x": 1.0}, {}) == 0.0
        assert weighted_jaccard({}, {"esco:x": 1.0}) == 0.0

    def test_identical_sets_returns_one(self):
        ws = {"esco:python": 1.0, "esco:ml": 0.5}
        assert weighted_jaccard(ws, ws) == pytest.approx(1.0)

    def test_no_overlap_returns_zero(self):
        a = {"esco:python": 1.0}
        b = {"esco:java": 1.0}
        assert weighted_jaccard(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # intersection min(1,1)=1; union max(1,1)+max(0.5,0)=1+0.5=1.5 → wait:
        # a={python:1, sql:0.5}, b={python:1, docker:1}
        # all_uris = {python, sql, docker}
        # intersection = min(1,1) + min(0.5,0) + min(0,1) = 1
        # union = max(1,1) + max(0.5,0) + max(0,1) = 1 + 0.5 + 1 = 2.5
        a = {"esco:python": 1.0, "esco:sql": 0.5}
        b = {"esco:python": 1.0, "esco:docker": 1.0}
        expected = 1.0 / 2.5
        assert weighted_jaccard(a, b) == pytest.approx(expected)

    def test_symmetric(self):
        a = {"esco:python": 1.0, "esco:ml": 0.5}
        b = {"esco:ml": 1.0, "esco:java": 0.5}
        assert weighted_jaccard(a, b) == pytest.approx(weighted_jaccard(b, a))


# ── overlap_coefficient ────────────────────────────────────────────────────────

class TestOverlapCoefficient:
    def test_both_empty_returns_zero(self):
        assert overlap_coefficient({}, {}) == 0.0

    def test_one_empty_returns_zero(self):
        assert overlap_coefficient({"esco:x": 1.0}, {}) == 0.0
        assert overlap_coefficient({}, {"esco:x": 1.0}) == 0.0

    def test_identical_sets_returns_one(self):
        ws = {"esco:python": 1.0, "esco:ml": 0.5}
        assert overlap_coefficient(ws, ws) == pytest.approx(1.0)

    def test_subset_returns_one(self):
        # a is a subset of b — overlap should be 1.0
        a = {"esco:python": 1.0}
        b = {"esco:python": 1.0, "esco:java": 1.0}
        assert overlap_coefficient(a, b) == pytest.approx(1.0)

    def test_no_overlap_returns_zero(self):
        a = {"esco:python": 1.0}
        b = {"esco:java": 1.0}
        assert overlap_coefficient(a, b) == pytest.approx(0.0)


# ── align_symbolic ─────────────────────────────────────────────────────────────

class TestAlignSymbolic:
    def _make_simple_df(self):
        """
        Prog0: {python:1.0, ml:0.5}
        Prog1: {java:1.0}
        Job0:  {python:1.0, ml:1.0}   ← strong match for Prog0
        Job1:  {java:1.0, sql:0.5}    ← match for Prog1
        Job2:  {docker:1.0}           ← no match for either
        """
        return _make_df(
            programmes=[
                [_skill("esco:python"), _skill("esco:ml", explicit=False, implicit=True)],
                [_skill("esco:java")],
            ],
            jobs=[
                [_skill("esco:python"), _skill("esco:ml")],
                [_skill("esco:java"), _skill("esco:sql", explicit=False, implicit=True)],
                [_skill("esco:docker")],
            ],
        )

    def test_output_shape(self):
        df = self._make_simple_df()
        rankings, gaps = align_symbolic(df, top_n=2)
        # 2 programmes × 3 jobs = 6 pairs
        assert len(rankings) == 6

    def test_ranking_columns_present(self):
        rankings, _ = align_symbolic(self._make_simple_df(), top_n=2)
        for col in ("programme_id", "job_id", "weighted_jaccard", "overlap_coeff"):
            assert col in rankings.columns

    def test_best_job_for_prog0_is_job0(self):
        df = self._make_simple_df()
        rankings, _ = align_symbolic(df, top_n=3)
        prog0_id = df[df["source_type"] == "programme"].index[0]
        top_for_prog0 = rankings[rankings["programme_id"] == prog0_id].iloc[0]
        # Job0 has python+ml, perfect overlap with Prog0
        job0_id = df[df["source_type"] == "job_ad"].index[0]
        assert top_for_prog0["job_id"] == job0_id

    def test_scores_in_0_1_range(self):
        rankings, _ = align_symbolic(self._make_simple_df(), top_n=2)
        assert (rankings["weighted_jaccard"] >= 0.0).all()
        assert (rankings["weighted_jaccard"] <= 1.0).all()
        assert (rankings["overlap_coeff"] >= 0.0).all()
        assert (rankings["overlap_coeff"] <= 1.0).all()

    def test_no_overlap_job_scores_zero(self):
        df = self._make_simple_df()
        rankings, _ = align_symbolic(df, top_n=2)
        prog1_id = df[df["source_type"] == "programme"].index[1]
        job2_id = df[df["source_type"] == "job_ad"].index[2]
        row = rankings[
            (rankings["programme_id"] == prog1_id) & (rankings["job_id"] == job2_id)
        ].iloc[0]
        assert row["weighted_jaccard"] == pytest.approx(0.0)

    def test_skill_gap_contains_missing_skills(self):
        df = self._make_simple_df()
        rankings, gaps = align_symbolic(df, top_n=3)
        prog0_id = df[df["source_type"] == "programme"].index[0]
        job1_id = df[df["source_type"] == "job_ad"].index[1]
        # Job1 has java:1.0 and sql:0.5; Prog0 has python:1.0 ml:0.5
        # Gap for (prog0, job1): java (1.0) and sql (0.5)
        prog0_job1_gaps = gaps[
            (gaps["programme_id"] == prog0_id) & (gaps["job_id"] == job1_id)
        ]
        gap_uris = set(prog0_job1_gaps["gap_uri"].tolist())
        assert "esco:java" in gap_uris
        assert "esco:sql" in gap_uris

    def test_empty_skills_handled_gracefully(self):
        df = _make_df(
            programmes=[[],  []],
            jobs=[[], []],
        )
        rankings, gaps = align_symbolic(df, top_n=5)
        assert len(rankings) == 4
        assert (rankings["weighted_jaccard"] == 0.0).all()
        assert len(gaps) == 0

    def test_sorted_descending_within_programme(self):
        df = self._make_simple_df()
        rankings, _ = align_symbolic(df, top_n=3)
        for p_id in rankings["programme_id"].unique():
            scores = rankings[rankings["programme_id"] == p_id]["weighted_jaccard"].tolist()
            assert scores == sorted(scores, reverse=True)


# ── run_symbolic_alignment ─────────────────────────────────────────────────────

class TestRunSymbolicAlignment:
    def test_output_files_created(self, tmp_path):
        df = _make_df(
            programmes=[[_skill("esco:python")]],
            jobs=[[_skill("esco:python")], [_skill("esco:java")]],
        )
        dataset_path = tmp_path / "dataset.parquet"
        df.to_parquet(dataset_path, index=True)

        output_dir = tmp_path / "exp1_symbolic"
        run_symbolic_alignment(
            dataset_path=dataset_path,
            output_dir=output_dir,
            top_n=2,
        )

        assert (output_dir / "rankings.parquet").exists()
        assert (output_dir / "skill_gaps.parquet").exists()
        assert (output_dir / "summary.json").exists()

    def test_rankings_shape_and_columns(self, tmp_path):
        df = _make_df(
            programmes=[[_skill("esco:python")]],
            jobs=[[_skill("esco:python")], [_skill("esco:java")]],
        )
        dataset_path = tmp_path / "dataset.parquet"
        df.to_parquet(dataset_path)
        output_dir = tmp_path / "exp1"
        run_symbolic_alignment(dataset_path=dataset_path, output_dir=output_dir, top_n=2)

        rankings = pd.read_parquet(output_dir / "rankings.parquet")
        assert len(rankings) == 2  # 1 programme × 2 jobs
        for col in ("programme_id", "job_id", "weighted_jaccard", "overlap_coeff"):
            assert col in rankings.columns
