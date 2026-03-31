"""
Tests for src/evaluation/cluster_analysis.py.

Covers:
  - contingency_test: chi-squared on programme×job cluster table, empty case
  - per_cluster_scores: mean scores per cluster for all strategies
  - cluster_score_summary: aggregation by cluster
  - cluster_skill_gaps: top-N gap URIs per cluster
  - best_strategy_per_cluster: picks highest mean strategy
  - compute_cluster_analysis: full pipeline output structure
  - run_cluster_analysis: output files created
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.cluster_analysis import (
    contingency_test,
    per_cluster_scores,
    cluster_score_summary,
    cluster_skill_gaps,
    best_strategy_per_cluster,
    compute_cluster_analysis,
    run_cluster_analysis,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_dataset(
    n_progs: int = 4,
    n_jobs: int = 10,
    prog_clusters: list[int] | None = None,
    job_clusters: list[int] | None = None,
) -> pd.DataFrame:
    """Build a minimal dataset with source_type and cluster_label."""
    if prog_clusters is None:
        prog_clusters = [i % 2 for i in range(n_progs)]
    if job_clusters is None:
        job_clusters = [i % 3 for i in range(n_jobs)]

    progs = pd.DataFrame({
        "source_type": "programme",
        "cluster_label": prog_clusters,
    })
    jobs = pd.DataFrame({
        "source_type": "job_ad",
        "cluster_label": job_clusters,
    })
    return pd.concat([progs, jobs], ignore_index=True)


def _make_rankings(
    n_progs: int,
    n_jobs: int,
    score_col: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Build rankings with all (programme, job) pairs and random scores."""
    rng = np.random.RandomState(seed)
    rows = []
    for p in range(n_progs):
        for j in range(n_jobs):
            rows.append({
                "programme_id": p,
                "job_id": j,
                score_col: float(rng.rand()),
            })
    return pd.DataFrame(rows)


def _make_rankings_fixed(
    n_progs: int,
    n_jobs: int,
    score_col: str,
    high_score: float = 0.9,
    low_score: float = 0.1,
) -> pd.DataFrame:
    """Rankings where job_id == programme_id scores highest."""
    rows = []
    for p in range(n_progs):
        for j in range(n_jobs):
            rows.append({
                "programme_id": p,
                "job_id": j,
                score_col: high_score if j == p else low_score,
            })
    return pd.DataFrame(rows)


def _make_skill_gaps(programme_ids: list[int], gap_uris: list[list[str]]) -> pd.DataFrame:
    rows = []
    for p_id, uris in zip(programme_ids, gap_uris):
        for uri in uris:
            rows.append({"programme_id": p_id, "gap_uri": uri})
    return pd.DataFrame(rows)


# ── contingency_test ─────────────────────────────────────────────────────────

class TestContingencyTest:
    def test_returns_expected_keys(self):
        ds = _make_dataset(n_progs=4, n_jobs=10)
        rankings = _make_rankings(4, 10, "hybrid_score")
        result = contingency_test(ds, rankings, "hybrid_score", top_n=3)
        for key in ("contingency_table", "chi2", "p_value", "dof", "cramers_v"):
            assert key in result

    def test_chi2_numeric(self):
        ds = _make_dataset(n_progs=4, n_jobs=10)
        rankings = _make_rankings(4, 10, "hybrid_score")
        result = contingency_test(ds, rankings, "hybrid_score", top_n=3)
        assert isinstance(result["chi2"], float)
        assert result["chi2"] >= 0.0

    def test_cramers_v_between_0_and_1(self):
        ds = _make_dataset(n_progs=4, n_jobs=10)
        rankings = _make_rankings(4, 10, "hybrid_score")
        result = contingency_test(ds, rankings, "hybrid_score", top_n=3)
        if result["cramers_v"] is not None:
            assert 0.0 <= result["cramers_v"] <= 1.0

    def test_empty_rankings(self):
        ds = _make_dataset(n_progs=4, n_jobs=10)
        empty = pd.DataFrame(columns=["programme_id", "job_id", "hybrid_score"])
        result = contingency_test(ds, empty, "hybrid_score", top_n=3)
        assert result["chi2"] is None
        assert result["contingency_table"] == {}

    def test_top_n_limits_matches(self):
        ds = _make_dataset(n_progs=2, n_jobs=10)
        rankings = _make_rankings(2, 10, "hybrid_score")
        r1 = contingency_test(ds, rankings, "hybrid_score", top_n=1)
        r5 = contingency_test(ds, rankings, "hybrid_score", top_n=5)
        # With top_n=1, contingency table has fewer entries
        t1_total = sum(sum(v.values()) for v in r1["contingency_table"].values())
        t5_total = sum(sum(v.values()) for v in r5["contingency_table"].values())
        assert t1_total <= t5_total


# ── per_cluster_scores ───────────────────────────────────────────────────────

class TestPerClusterScores:
    def test_returns_all_programmes(self):
        ds = _make_dataset(n_progs=4, n_jobs=10)
        rankings = {
            "symbolic": _make_rankings(4, 10, "weighted_jaccard", seed=1),
            "semantic": _make_rankings(4, 10, "cosine_combined", seed=2),
            "hybrid": _make_rankings(4, 10, "hybrid_score", seed=3),
        }
        result = per_cluster_scores(ds, rankings, top_n=5)
        assert len(result) == 4
        assert set(result.columns) >= {"programme_id", "prog_cluster", "mean_symbolic", "mean_semantic", "mean_hybrid"}

    def test_scores_are_positive(self):
        ds = _make_dataset(n_progs=3, n_jobs=8)
        rankings = {
            "symbolic": _make_rankings(3, 8, "weighted_jaccard"),
            "semantic": _make_rankings(3, 8, "cosine_combined"),
            "hybrid": _make_rankings(3, 8, "hybrid_score"),
        }
        result = per_cluster_scores(ds, rankings, top_n=5)
        for col in ["mean_symbolic", "mean_semantic", "mean_hybrid"]:
            assert (result[col] >= 0).all()

    def test_cluster_labels_from_dataset(self):
        ds = _make_dataset(n_progs=4, prog_clusters=[10, 10, 20, 20])
        rankings = {
            "symbolic": _make_rankings(4, 5, "weighted_jaccard"),
            "semantic": _make_rankings(4, 5, "cosine_combined"),
            "hybrid": _make_rankings(4, 5, "hybrid_score"),
        }
        result = per_cluster_scores(ds, rankings)
        assert set(result["prog_cluster"]) == {10, 20}


# ── cluster_score_summary ───────────────────────────────────────────────────

class TestClusterScoreSummary:
    def test_keys_per_cluster(self):
        per_cluster = pd.DataFrame({
            "programme_id": [0, 1, 2, 3],
            "prog_cluster": [0, 0, 1, 1],
            "mean_symbolic": [0.1, 0.2, 0.3, 0.4],
            "mean_semantic": [0.5, 0.6, 0.7, 0.8],
            "mean_hybrid": [0.3, 0.4, 0.5, 0.6],
        })
        result = cluster_score_summary(per_cluster)
        assert set(result.keys()) == {0, 1}
        for cluster_data in result.values():
            assert "n_programmes" in cluster_data
            assert "mean_symbolic" in cluster_data

    def test_n_programmes_correct(self):
        per_cluster = pd.DataFrame({
            "programme_id": [0, 1, 2],
            "prog_cluster": [0, 0, 1],
            "mean_symbolic": [0.1, 0.2, 0.3],
        })
        result = cluster_score_summary(per_cluster)
        assert result[0]["n_programmes"] == 2
        assert result[1]["n_programmes"] == 1

    def test_mean_values_correct(self):
        per_cluster = pd.DataFrame({
            "programme_id": [0, 1],
            "prog_cluster": [0, 0],
            "mean_symbolic": [0.2, 0.4],
        })
        result = cluster_score_summary(per_cluster)
        assert abs(result[0]["mean_symbolic"] - 0.3) < 1e-4


# ── cluster_skill_gaps ───────────────────────────────────────────────────────

class TestClusterSkillGaps:
    def test_returns_per_cluster(self):
        ds = _make_dataset(n_progs=4, n_jobs=5, prog_clusters=[0, 0, 1, 1])
        gaps = _make_skill_gaps(
            [0, 0, 1, 2, 3],
            [["uri:a", "uri:b"], ["uri:a"], ["uri:c"], ["uri:a"], ["uri:d"]],
        )
        result = cluster_skill_gaps(ds, gaps)
        assert 0 in result
        assert 1 in result

    def test_most_common_gap_first(self):
        ds = _make_dataset(n_progs=3, n_jobs=5, prog_clusters=[0, 0, 0])
        gaps = _make_skill_gaps(
            [0, 1, 1, 2],
            [["uri:a"], ["uri:a", "uri:b"], ["uri:a"], ["uri:b"]],
        )
        result = cluster_skill_gaps(ds, gaps)
        top_gap_uri = result[0][0][0]
        assert top_gap_uri == "uri:a"

    def test_top_n_gaps_limits(self):
        ds = _make_dataset(n_progs=2, n_jobs=5, prog_clusters=[0, 0])
        uris = [f"uri:{i}" for i in range(20)]
        gaps = _make_skill_gaps([0, 1], [uris[:10], uris[10:]])
        result = cluster_skill_gaps(ds, gaps, top_n_gaps=5)
        assert len(result[0]) <= 5

    def test_empty_skill_gaps(self):
        ds = _make_dataset(n_progs=2, n_jobs=5)
        gaps = pd.DataFrame(columns=["programme_id", "gap_uri"])
        result = cluster_skill_gaps(ds, gaps)
        assert result == {}


# ── best_strategy_per_cluster ────────────────────────────────────────────────

class TestBestStrategyPerCluster:
    def test_picks_highest(self):
        per_cluster = pd.DataFrame({
            "programme_id": [0, 1],
            "prog_cluster": [0, 0],
            "mean_symbolic": [0.1, 0.1],
            "mean_semantic": [0.9, 0.9],
            "mean_hybrid": [0.5, 0.5],
        })
        result = best_strategy_per_cluster(per_cluster)
        assert result[0]["best_strategy"] == "semantic"

    def test_different_clusters_different_best(self):
        per_cluster = pd.DataFrame({
            "programme_id": [0, 1, 2, 3],
            "prog_cluster": [0, 0, 1, 1],
            "mean_symbolic": [0.8, 0.9, 0.1, 0.1],
            "mean_semantic": [0.1, 0.1, 0.8, 0.9],
            "mean_hybrid": [0.2, 0.2, 0.2, 0.2],
        })
        result = best_strategy_per_cluster(per_cluster)
        assert result[0]["best_strategy"] == "symbolic"
        assert result[1]["best_strategy"] == "semantic"

    def test_scores_dict_present(self):
        per_cluster = pd.DataFrame({
            "programme_id": [0],
            "prog_cluster": [0],
            "mean_symbolic": [0.5],
            "mean_semantic": [0.6],
            "mean_hybrid": [0.7],
        })
        result = best_strategy_per_cluster(per_cluster)
        assert "scores" in result[0]
        assert set(result[0]["scores"].keys()) == {"symbolic", "semantic", "hybrid"}


# ── compute_cluster_analysis ─────────────────────────────────────────────────

class TestComputeClusterAnalysis:
    def test_output_structure(self):
        ds = _make_dataset(n_progs=4, n_jobs=10)
        sym = _make_rankings(4, 10, "weighted_jaccard", seed=1)
        sem = _make_rankings(4, 10, "cosine_combined", seed=2)
        hyb = _make_rankings(4, 10, "hybrid_score", seed=3)
        per_cluster, summary = compute_cluster_analysis(ds, sym, sem, hyb, top_n=5)

        assert isinstance(per_cluster, pd.DataFrame)
        assert len(per_cluster) == 4
        assert "n_programme_clusters" in summary
        assert "contingency_test" in summary
        assert "cluster_scores" in summary
        assert "best_strategy_per_cluster" in summary

    def test_with_skill_gaps(self):
        ds = _make_dataset(n_progs=4, n_jobs=10, prog_clusters=[0, 0, 1, 1])
        sym = _make_rankings(4, 10, "weighted_jaccard")
        sem = _make_rankings(4, 10, "cosine_combined")
        hyb = _make_rankings(4, 10, "hybrid_score")
        gaps = _make_skill_gaps([0, 1, 2, 3], [["uri:a"], ["uri:b"], ["uri:c"], ["uri:d"]])

        _, summary = compute_cluster_analysis(ds, sym, sem, hyb, skill_gaps=gaps, top_n=5)
        assert "cluster_skill_gaps" in summary

    def test_without_skill_gaps(self):
        ds = _make_dataset(n_progs=2, n_jobs=5)
        sym = _make_rankings(2, 5, "weighted_jaccard")
        sem = _make_rankings(2, 5, "cosine_combined")
        hyb = _make_rankings(2, 5, "hybrid_score")

        _, summary = compute_cluster_analysis(ds, sym, sem, hyb, top_n=3)
        assert "cluster_skill_gaps" not in summary

    def test_n_programme_clusters(self):
        ds = _make_dataset(n_progs=6, n_jobs=10, prog_clusters=[0, 0, 1, 1, 2, 2])
        sym = _make_rankings(6, 10, "weighted_jaccard")
        sem = _make_rankings(6, 10, "cosine_combined")
        hyb = _make_rankings(6, 10, "hybrid_score")

        _, summary = compute_cluster_analysis(ds, sym, sem, hyb)
        assert summary["n_programme_clusters"] == 3


# ── run_cluster_analysis ─────────────────────────────────────────────────────

class TestRunClusterAnalysis:
    def test_output_files_created(self, tmp_path: Path):
        ds = _make_dataset(n_progs=4, n_jobs=10)
        sym = _make_rankings(4, 10, "weighted_jaccard")
        sem = _make_rankings(4, 10, "cosine_combined")
        hyb = _make_rankings(4, 10, "hybrid_score")

        ds_path = tmp_path / "dataset.parquet"
        sym_path = tmp_path / "sym.parquet"
        sem_path = tmp_path / "sem.parquet"
        hyb_path = tmp_path / "hyb.parquet"

        ds.to_parquet(ds_path)
        sym.to_parquet(sym_path)
        sem.to_parquet(sem_path)
        hyb.to_parquet(hyb_path)

        out_dir = tmp_path / "output"
        run_cluster_analysis(
            dataset_path=ds_path,
            symbolic_path=sym_path,
            semantic_path=sem_path,
            hybrid_path=hyb_path,
            skill_gaps_path=tmp_path / "nonexistent.parquet",
            output_dir=out_dir,
        )

        assert (out_dir / "cluster_analysis.parquet").exists()
        assert (out_dir / "cluster_analysis.json").exists()

        with open(out_dir / "cluster_analysis.json") as f:
            data = json.load(f)
        assert "contingency_test" in data
        assert "cluster_scores" in data

    def test_with_skill_gaps_file(self, tmp_path: Path):
        ds = _make_dataset(n_progs=3, n_jobs=8, prog_clusters=[0, 0, 1])
        sym = _make_rankings(3, 8, "weighted_jaccard")
        sem = _make_rankings(3, 8, "cosine_combined")
        hyb = _make_rankings(3, 8, "hybrid_score")
        gaps = _make_skill_gaps([0, 1, 2], [["uri:x"], ["uri:y"], ["uri:z"]])

        ds.to_parquet(tmp_path / "ds.parquet")
        sym.to_parquet(tmp_path / "sym.parquet")
        sem.to_parquet(tmp_path / "sem.parquet")
        hyb.to_parquet(tmp_path / "hyb.parquet")
        gaps.to_parquet(tmp_path / "gaps.parquet")

        out = tmp_path / "out"
        run_cluster_analysis(
            dataset_path=tmp_path / "ds.parquet",
            symbolic_path=tmp_path / "sym.parquet",
            semantic_path=tmp_path / "sem.parquet",
            hybrid_path=tmp_path / "hyb.parquet",
            skill_gaps_path=tmp_path / "gaps.parquet",
            output_dir=out,
        )

        with open(out / "cluster_analysis.json") as f:
            data = json.load(f)
        assert "cluster_skill_gaps" in data
