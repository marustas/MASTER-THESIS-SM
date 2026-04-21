"""
Tests for Step 32 — Programme coverage analysis.

Covers:
  - analyse_coverage: basic coverage, threshold filtering, low_coverage flag
  - identify_niche_clusters: cluster aggregation, sorting
  - generate_expansion_recommendations: recommendation format, low-coverage only
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.coverage import (
    analyse_coverage,
    generate_expansion_recommendations,
    identify_niche_clusters,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _skill(uri: str, *, explicit: bool = True) -> dict:
    return {
        "esco_uri": uri,
        "preferred_label": uri.split(":")[-1],
        "matched_text": uri.split(":")[-1],
        "explicit": explicit,
        "implicit": not explicit,
        "confidence": 1.0,
    }


def _make_dataset(n_prog: int = 3, n_jobs: int = 10) -> pd.DataFrame:
    rows = []
    for i in range(n_prog):
        rows.append({
            "source_type": "programme",
            "name": f"Prog{i}",
            "cluster_label": i % 2,
            "skill_details": [_skill(f"esco:s{i}")],
            "skill_uris": [f"esco:s{i}"],
        })
    for i in range(n_jobs):
        rows.append({
            "source_type": "job_ad",
            "job_title": f"Job{i}",
            "cluster_label": i % 3,
            "skill_details": [_skill(f"esco:j{i}")],
            "skill_uris": [f"esco:j{i}"],
        })
    return pd.DataFrame(rows)


def _make_rankings(n_prog: int, n_jobs: int, high_score_count: int = 3) -> pd.DataFrame:
    """Create synthetic hybrid rankings with controlled score distribution."""
    records = []
    prog_start = 0
    job_start = n_prog
    for p in range(n_prog):
        for j in range(n_jobs):
            score = 0.3 if j < high_score_count else 0.05
            records.append({
                "programme_id": p,
                "job_id": job_start + j,
                "programme_name": f"Prog{p}",
                "job_title": f"Job{j}",
                "hybrid_score": score,
            })
    return pd.DataFrame(records)


# ── analyse_coverage ─────────────────────────────────────────────────────────

class TestAnalyseCoverage:
    def test_required_columns(self):
        dataset = _make_dataset(3, 10)
        rankings = _make_rankings(3, 10, high_score_count=6)
        result = analyse_coverage(dataset, rankings, score_threshold=0.1)
        for col in ("programme_id", "programme_name", "cluster_label",
                     "n_matches", "coverage_ratio", "top_score", "low_coverage"):
            assert col in result.columns

    def test_correct_match_count(self):
        dataset = _make_dataset(2, 5)
        rankings = _make_rankings(2, 5, high_score_count=3)
        result = analyse_coverage(dataset, rankings, score_threshold=0.1)
        # 3 jobs above 0.1 per programme
        assert (result["n_matches"] == 3).all()

    def test_low_coverage_flag(self):
        dataset = _make_dataset(2, 5)
        rankings = _make_rankings(2, 5, high_score_count=2)
        result = analyse_coverage(dataset, rankings, score_threshold=0.1, min_matches=5)
        assert result["low_coverage"].all()

    def test_high_coverage_not_flagged(self):
        dataset = _make_dataset(2, 10)
        rankings = _make_rankings(2, 10, high_score_count=8)
        result = analyse_coverage(dataset, rankings, score_threshold=0.1, min_matches=5)
        assert not result["low_coverage"].any()

    def test_coverage_ratio(self):
        dataset = _make_dataset(1, 10)
        rankings = _make_rankings(1, 10, high_score_count=5)
        result = analyse_coverage(dataset, rankings, score_threshold=0.1)
        assert result.iloc[0]["coverage_ratio"] == pytest.approx(0.5)

    def test_top_score(self):
        dataset = _make_dataset(1, 5)
        rankings = _make_rankings(1, 5, high_score_count=2)
        result = analyse_coverage(dataset, rankings, score_threshold=0.1)
        assert result.iloc[0]["top_score"] == pytest.approx(0.3)

    def test_empty_rankings(self):
        dataset = _make_dataset(2, 5)
        rankings = pd.DataFrame(columns=["programme_id", "job_id", "hybrid_score"])
        result = analyse_coverage(dataset, rankings, score_threshold=0.1)
        assert len(result) == 2
        assert (result["n_matches"] == 0).all()
        assert result["low_coverage"].all()

    def test_threshold_zero_counts_all(self):
        dataset = _make_dataset(1, 5)
        rankings = _make_rankings(1, 5, high_score_count=2)
        result = analyse_coverage(dataset, rankings, score_threshold=0.0)
        assert result.iloc[0]["n_matches"] == 5


# ── identify_niche_clusters ──────────────────────────────────────────────────

class TestIdentifyNicheClusters:
    def test_basic_aggregation(self):
        coverage = pd.DataFrame([
            {"programme_id": 0, "cluster_label": 0, "n_matches": 2, "top_score": 0.3, "low_coverage": True},
            {"programme_id": 1, "cluster_label": 0, "n_matches": 3, "top_score": 0.4, "low_coverage": True},
            {"programme_id": 2, "cluster_label": 1, "n_matches": 10, "top_score": 0.8, "low_coverage": False},
        ])
        result = identify_niche_clusters(coverage)
        assert len(result) == 2
        # Cluster 0 has 100% low coverage
        cluster0 = result[result["cluster_label"] == 0].iloc[0]
        assert cluster0["low_coverage_ratio"] == pytest.approx(1.0)

    def test_sorted_by_low_coverage_ratio(self):
        coverage = pd.DataFrame([
            {"programme_id": 0, "cluster_label": 0, "n_matches": 10, "top_score": 0.8, "low_coverage": False},
            {"programme_id": 1, "cluster_label": 1, "n_matches": 2, "top_score": 0.3, "low_coverage": True},
        ])
        result = identify_niche_clusters(coverage)
        assert result.iloc[0]["cluster_label"] == 1

    def test_empty_coverage(self):
        coverage = pd.DataFrame(columns=[
            "programme_id", "cluster_label", "n_matches", "top_score", "low_coverage",
        ])
        result = identify_niche_clusters(coverage)
        assert len(result) == 0


# ── generate_expansion_recommendations ───────────────────────────────────────

class TestGenerateExpansionRecommendations:
    def test_only_low_coverage(self):
        coverage = pd.DataFrame([
            {"programme_id": 0, "programme_name": "CS", "cluster_label": 0,
             "n_matches": 2, "low_coverage": True},
            {"programme_id": 1, "programme_name": "AI", "cluster_label": 1,
             "n_matches": 10, "low_coverage": False},
        ])
        dataset = pd.DataFrame([
            {"source_type": "programme", "name": "CS", "skill_details": [_skill("esco:python")]},
            {"source_type": "programme", "name": "AI", "skill_details": [_skill("esco:ml")]},
        ])
        recs = generate_expansion_recommendations(coverage, dataset)
        assert len(recs) == 1
        assert recs[0]["programme_name"] == "CS"

    def test_recommendation_contains_skills(self):
        coverage = pd.DataFrame([
            {"programme_id": 0, "programme_name": "GameDev", "cluster_label": 0,
             "n_matches": 1, "low_coverage": True},
        ])
        dataset = pd.DataFrame([
            {"source_type": "programme", "name": "GameDev",
             "skill_details": [_skill("esco:unity"), _skill("esco:csharp")]},
        ])
        recs = generate_expansion_recommendations(coverage, dataset)
        assert len(recs) == 1
        assert "unity" in recs[0]["top_skills"]
        assert "recommendation" in recs[0]

    def test_no_low_coverage_returns_empty(self):
        coverage = pd.DataFrame([
            {"programme_id": 0, "programme_name": "CS", "cluster_label": 0,
             "n_matches": 10, "low_coverage": False},
        ])
        dataset = pd.DataFrame([
            {"source_type": "programme", "name": "CS", "skill_details": [_skill("esco:python")]},
        ])
        recs = generate_expansion_recommendations(coverage, dataset)
        assert recs == []

    def test_empty_skill_details(self):
        coverage = pd.DataFrame([
            {"programme_id": 0, "programme_name": "New", "cluster_label": 0,
             "n_matches": 0, "low_coverage": True},
        ])
        dataset = pd.DataFrame([
            {"source_type": "programme", "name": "New", "skill_details": []},
        ])
        recs = generate_expansion_recommendations(coverage, dataset)
        assert len(recs) == 1
        assert recs[0]["top_skills"] == []
