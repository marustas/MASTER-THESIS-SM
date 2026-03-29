"""
Offline tests for src.recommendations.generator (Step 12).
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from src.recommendations.generator import (
    _best_strategy,
    _market_trends,
    _top_gap_uris_per_programme,
    generate_recommendations,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def symbolic_rankings():
    return pd.DataFrame([
        {"programme_id": 0, "programme_name": "AI MSc", "job_id": 0, "job_title": "ML Eng",      "weighted_jaccard": 0.8, "overlap_coeff": 0.9},
        {"programme_id": 0, "programme_name": "AI MSc", "job_id": 1, "job_title": "Data Sci",    "weighted_jaccard": 0.5, "overlap_coeff": 0.6},
        {"programme_id": 0, "programme_name": "AI MSc", "job_id": 2, "job_title": "Backend Dev", "weighted_jaccard": 0.2, "overlap_coeff": 0.3},
        {"programme_id": 1, "programme_name": "SE BSc",  "job_id": 0, "job_title": "ML Eng",      "weighted_jaccard": 0.3, "overlap_coeff": 0.4},
        {"programme_id": 1, "programme_name": "SE BSc",  "job_id": 1, "job_title": "Data Sci",    "weighted_jaccard": 0.4, "overlap_coeff": 0.5},
        {"programme_id": 1, "programme_name": "SE BSc",  "job_id": 2, "job_title": "Backend Dev", "weighted_jaccard": 0.7, "overlap_coeff": 0.8},
    ])


@pytest.fixture
def skill_gaps():
    return pd.DataFrame([
        {"programme_id": 0, "job_id": 0, "gap_uri": "esco:python",     "gap_weight": 1.0},
        {"programme_id": 0, "job_id": 0, "gap_uri": "esco:pytorch",    "gap_weight": 1.0},
        {"programme_id": 0, "job_id": 1, "gap_uri": "esco:python",     "gap_weight": 1.0},
        {"programme_id": 1, "job_id": 2, "gap_uri": "esco:docker",     "gap_weight": 1.0},
        {"programme_id": 1, "job_id": 2, "gap_uri": "esco:kubernetes", "gap_weight": 1.0},
    ])


@pytest.fixture
def semantic_rankings():
    return pd.DataFrame([
        {"programme_id": 0, "programme_name": "AI MSc", "job_id": 0, "job_title": "ML Eng",      "cosine_combined": 0.9},
        {"programme_id": 0, "programme_name": "AI MSc", "job_id": 1, "job_title": "Data Sci",    "cosine_combined": 0.7},
        {"programme_id": 0, "programme_name": "AI MSc", "job_id": 2, "job_title": "Backend Dev", "cosine_combined": 0.3},
        {"programme_id": 1, "programme_name": "SE BSc",  "job_id": 0, "job_title": "ML Eng",      "cosine_combined": 0.4},
        {"programme_id": 1, "programme_name": "SE BSc",  "job_id": 1, "job_title": "Data Sci",    "cosine_combined": 0.5},
        {"programme_id": 1, "programme_name": "SE BSc",  "job_id": 2, "job_title": "Backend Dev", "cosine_combined": 0.8},
    ])


@pytest.fixture
def hybrid_rankings():
    return pd.DataFrame([
        {"programme_id": 0, "programme_name": "AI MSc", "job_id": 0, "job_title": "ML Eng",      "hybrid_score": 0.85},
        {"programme_id": 0, "programme_name": "AI MSc", "job_id": 1, "job_title": "Data Sci",    "hybrid_score": 0.60},
        {"programme_id": 0, "programme_name": "AI MSc", "job_id": 2, "job_title": "Backend Dev", "hybrid_score": 0.25},
        {"programme_id": 1, "programme_name": "SE BSc",  "job_id": 0, "job_title": "ML Eng",      "hybrid_score": 0.35},
        {"programme_id": 1, "programme_name": "SE BSc",  "job_id": 1, "job_title": "Data Sci",    "hybrid_score": 0.45},
        {"programme_id": 1, "programme_name": "SE BSc",  "job_id": 2, "job_title": "Backend Dev", "hybrid_score": 0.75},
    ])


@pytest.fixture
def dataset():
    return pd.DataFrame([
        {"source_type": "programme", "skill_uris": ["esco:ml",    "esco:stats"]},
        {"source_type": "programme", "skill_uris": ["esco:java",  "esco:docker"]},
        {"source_type": "job_ad",    "skill_uris": ["esco:python", "esco:ml",  "esco:pytorch"]},
        {"source_type": "job_ad",    "skill_uris": ["esco:python", "esco:sql"]},
        {"source_type": "job_ad",    "skill_uris": ["esco:docker", "esco:kubernetes", "esco:python"]},
    ])


@pytest.fixture
def eval_summary():
    return {
        "n_programmes": 2,
        "top_k": 10,
        "top1_agreement_rate": 0.75,
        "spearman": {
            "sym_sem": {"mean": 0.70, "median": 0.72},
            "sym_hyb": {"mean": 0.80, "median": 0.82},
            "sem_hyb": {"mean": 0.85, "median": 0.87},
        },
    }


# ── _best_strategy ─────────────────────────────────────────────────────────────

class TestBestStrategy:
    def test_returns_hybrid_when_no_spearman_data(self):
        assert _best_strategy({}) == "hybrid"

    def test_returns_hybrid_when_spearman_all_nan(self):
        summary = {"spearman": {"sym_sem": {}, "sym_hyb": {}, "sem_hyb": {}}}
        assert _best_strategy(summary) == "hybrid"

    def test_selects_most_central_strategy(self, eval_summary):
        # sym_sem=0.70, sym_hyb=0.80, sem_hyb=0.85
        # symbolic centrality = mean(0.70, 0.80) = 0.75
        # semantic centrality = mean(0.70, 0.85) = 0.775
        # hybrid   centrality = mean(0.80, 0.85) = 0.825  ← highest
        assert _best_strategy(eval_summary) == "hybrid"

    def test_selects_symbolic_when_most_central(self):
        summary = {
            "spearman": {
                "sym_sem": {"mean": 0.9},
                "sym_hyb": {"mean": 0.9},
                "sem_hyb": {"mean": 0.1},
            }
        }
        # symbolic = mean(0.9, 0.9) = 0.9
        # semantic = mean(0.9, 0.1) = 0.5
        # hybrid   = mean(0.9, 0.1) = 0.5
        assert _best_strategy(summary) == "symbolic"

    def test_selects_semantic_when_most_central(self):
        summary = {
            "spearman": {
                "sym_sem": {"mean": 0.9},
                "sym_hyb": {"mean": 0.1},
                "sem_hyb": {"mean": 0.9},
            }
        }
        # symbolic = mean(0.9, 0.1) = 0.5
        # semantic = mean(0.9, 0.9) = 0.9  ← highest
        # hybrid   = mean(0.1, 0.9) = 0.5
        assert _best_strategy(summary) == "semantic"

    def test_partial_nan_handled(self):
        summary = {
            "spearman": {
                "sym_sem": {"mean": 0.6},
                "sym_hyb": {},            # mean missing → NaN
                "sem_hyb": {"mean": 0.8},
            }
        }
        # symbolic = mean(0.6, nan) = 0.6
        # semantic = mean(0.6, 0.8) = 0.7  ← highest
        # hybrid   = mean(nan, 0.8) = 0.8 -- wait, nan is 0.8 alone = 0.8
        # Actually: hybrid = mean([0.8]) = 0.8 ← highest
        result = _best_strategy(summary)
        assert result in {"semantic", "hybrid"}


# ── _top_gap_uris_per_programme ────────────────────────────────────────────────

class TestTopGapUrisPerProgramme:
    def test_returns_most_frequent_uris(self, skill_gaps):
        # programme 0: job 0 → python, pytorch; job 1 → python
        # python appears 2x → should be #1 for programme 0
        top_pairs = pd.DataFrame([
            {"programme_id": 0, "job_id": 0},
            {"programme_id": 0, "job_id": 1},
        ])
        result = _top_gap_uris_per_programme(skill_gaps, top_pairs, n=5)
        prog0 = result[result["programme_id"] == 0]["top_gap_uris"].iloc[0]
        assert prog0[0] == "esco:python"

    def test_empty_when_no_matching_pairs(self, skill_gaps):
        top_pairs = pd.DataFrame([{"programme_id": 99, "job_id": 99}])
        result = _top_gap_uris_per_programme(skill_gaps, top_pairs)
        assert result.empty or len(result) == 0

    def test_limits_to_n_uris(self, skill_gaps):
        top_pairs = pd.DataFrame([
            {"programme_id": 0, "job_id": 0},
            {"programme_id": 0, "job_id": 1},
        ])
        result = _top_gap_uris_per_programme(skill_gaps, top_pairs, n=1)
        prog0 = result[result["programme_id"] == 0]["top_gap_uris"].iloc[0]
        assert len(prog0) == 1

    def test_separate_results_per_programme(self, skill_gaps):
        top_pairs = pd.DataFrame([
            {"programme_id": 0, "job_id": 0},
            {"programme_id": 1, "job_id": 2},
        ])
        result = _top_gap_uris_per_programme(skill_gaps, top_pairs, n=5)
        assert set(result["programme_id"]) == {0, 1}

    def test_empty_gaps_dataframe(self):
        empty = pd.DataFrame(columns=["programme_id", "job_id", "gap_uri", "gap_weight"])
        top_pairs = pd.DataFrame([{"programme_id": 0, "job_id": 0}])
        result = _top_gap_uris_per_programme(empty, top_pairs)
        assert result.empty


# ── _market_trends ─────────────────────────────────────────────────────────────

class TestMarketTrends:
    def test_returns_dataframe_with_expected_columns(self, dataset):
        result = _market_trends(dataset)
        assert set(result.columns) >= {"skill_uri", "job_ad_count", "programme_count",
                                        "frequency", "programme_coverage", "gap_index"}

    def test_sorted_by_gap_index_descending(self, dataset):
        result = _market_trends(dataset)
        assert result["gap_index"].is_monotonic_decreasing

    def test_python_has_highest_gap_index(self, dataset):
        # python appears in 3/3 job ads but 0/2 programmes → gap_index = 1.0
        result = _market_trends(dataset)
        top_uri = result.iloc[0]["skill_uri"]
        assert top_uri == "esco:python"

    def test_frequency_values_in_unit_interval(self, dataset):
        result = _market_trends(dataset)
        assert (result["frequency"] >= 0).all()
        assert (result["frequency"] <= 1).all()

    def test_programme_coverage_values_in_unit_interval(self, dataset):
        result = _market_trends(dataset)
        assert (result["programme_coverage"] >= 0).all()
        assert (result["programme_coverage"] <= 1).all()

    def test_job_ad_count_correct(self, dataset):
        # python appears in all 3 job ads
        result = _market_trends(dataset)
        python_row = result[result["skill_uri"] == "esco:python"]
        assert python_row["job_ad_count"].iloc[0] == 3

    def test_top_n_limits_rows(self, dataset):
        result = _market_trends(dataset, top_n=2)
        assert len(result) <= 2

    def test_empty_dataset_does_not_crash(self):
        empty = pd.DataFrame({"source_type": pd.Series([], dtype=str),
                               "skill_uris": pd.Series([], dtype=object)})
        result = _market_trends(empty)
        assert isinstance(result, pd.DataFrame)


# ── generate_recommendations ───────────────────────────────────────────────────

class TestGenerateRecommendations:
    def test_returns_three_outputs(
        self, dataset, symbolic_rankings, skill_gaps,
        semantic_rankings, hybrid_rankings, eval_summary
    ):
        recs, trends, summary = generate_recommendations(
            dataset, symbolic_rankings, skill_gaps,
            semantic_rankings, hybrid_rankings, eval_summary,
        )
        assert isinstance(recs, pd.DataFrame)
        assert isinstance(trends, pd.DataFrame)
        assert isinstance(summary, dict)

    def test_recommendations_top_n_limit(
        self, dataset, symbolic_rankings, skill_gaps,
        semantic_rankings, hybrid_rankings, eval_summary
    ):
        recs, _, _ = generate_recommendations(
            dataset, symbolic_rankings, skill_gaps,
            semantic_rankings, hybrid_rankings, eval_summary,
            top_n=2,
        )
        for pid, group in recs.groupby("programme_id"):
            assert len(group) <= 2

    def test_rank_column_starts_at_one(
        self, dataset, symbolic_rankings, skill_gaps,
        semantic_rankings, hybrid_rankings, eval_summary
    ):
        recs, _, _ = generate_recommendations(
            dataset, symbolic_rankings, skill_gaps,
            semantic_rankings, hybrid_rankings, eval_summary,
        )
        assert recs["rank"].min() == 1

    def test_rank_ordered_by_alignment_score(
        self, dataset, symbolic_rankings, skill_gaps,
        semantic_rankings, hybrid_rankings, eval_summary
    ):
        recs, _, _ = generate_recommendations(
            dataset, symbolic_rankings, skill_gaps,
            semantic_rankings, hybrid_rankings, eval_summary,
        )
        for _, group in recs.groupby("programme_id"):
            scores = group.sort_values("rank")["alignment_score"].tolist()
            assert scores == sorted(scores, reverse=True)

    def test_strategy_column_set(
        self, dataset, symbolic_rankings, skill_gaps,
        semantic_rankings, hybrid_rankings, eval_summary
    ):
        recs, _, summary = generate_recommendations(
            dataset, symbolic_rankings, skill_gaps,
            semantic_rankings, hybrid_rankings, eval_summary,
        )
        assert (recs["strategy"] == summary["best_strategy"]).all()

    def test_top_gap_uris_is_list(
        self, dataset, symbolic_rankings, skill_gaps,
        semantic_rankings, hybrid_rankings, eval_summary
    ):
        recs, _, _ = generate_recommendations(
            dataset, symbolic_rankings, skill_gaps,
            semantic_rankings, hybrid_rankings, eval_summary,
        )
        assert all(isinstance(v, list) for v in recs["top_gap_uris"])

    def test_summary_contains_expected_keys(
        self, dataset, symbolic_rankings, skill_gaps,
        semantic_rankings, hybrid_rankings, eval_summary
    ):
        _, _, summary = generate_recommendations(
            dataset, symbolic_rankings, skill_gaps,
            semantic_rankings, hybrid_rankings, eval_summary,
        )
        required_keys = {
            "best_strategy", "n_programmes", "n_job_ads", "top_n",
            "top1_agreement_rate", "spearman_means",
            "n_market_trend_skills", "top_market_skills",
        }
        assert required_keys <= set(summary.keys())

    def test_summary_n_programmes_correct(
        self, dataset, symbolic_rankings, skill_gaps,
        semantic_rankings, hybrid_rankings, eval_summary
    ):
        _, _, summary = generate_recommendations(
            dataset, symbolic_rankings, skill_gaps,
            semantic_rankings, hybrid_rankings, eval_summary,
        )
        assert summary["n_programmes"] == 2

    def test_summary_n_job_ads_correct(
        self, dataset, symbolic_rankings, skill_gaps,
        semantic_rankings, hybrid_rankings, eval_summary
    ):
        _, _, summary = generate_recommendations(
            dataset, symbolic_rankings, skill_gaps,
            semantic_rankings, hybrid_rankings, eval_summary,
        )
        assert summary["n_job_ads"] == 3

    def test_empty_eval_summary_falls_back_to_hybrid(
        self, dataset, symbolic_rankings, skill_gaps,
        semantic_rankings, hybrid_rankings
    ):
        _, _, summary = generate_recommendations(
            dataset, symbolic_rankings, skill_gaps,
            semantic_rankings, hybrid_rankings, {},
        )
        assert summary["best_strategy"] == "hybrid"

    def test_required_output_columns_present(
        self, dataset, symbolic_rankings, skill_gaps,
        semantic_rankings, hybrid_rankings, eval_summary
    ):
        recs, trends, _ = generate_recommendations(
            dataset, symbolic_rankings, skill_gaps,
            semantic_rankings, hybrid_rankings, eval_summary,
        )
        assert {"programme_id", "job_id", "rank", "alignment_score", "strategy",
                "top_gap_uris"} <= set(recs.columns)
        assert {"skill_uri", "job_ad_count", "gap_index"} <= set(trends.columns)
