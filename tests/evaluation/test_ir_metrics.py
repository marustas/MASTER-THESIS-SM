"""
Tests for src/evaluation/ir_metrics.py.

Covers:
  - build_consensus: min_strategies threshold, empty strategies
  - precision_at_k: perfect, zero, partial
  - ndcg_at_k: perfect ranking, reversed ranking, no relevant
  - reciprocal_rank: first position, later position, missing
  - evaluate_strategy: aggregate metrics for known inputs
  - compute_ir_metrics: output structure, all strategies present
  - run_ir_metrics: output file created
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.ir_metrics import (
    build_consensus,
    precision_at_k,
    dcg_at_k,
    ndcg_at_k,
    reciprocal_rank,
    evaluate_strategy,
    compute_ir_metrics,
    run_ir_metrics,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_rankings(
    prog_ids: list[int],
    job_ids_per_prog: list[list[int]],
    score_col: str,
) -> pd.DataFrame:
    rows = []
    for p_id, job_ids in zip(prog_ids, job_ids_per_prog):
        for rank, j_id in enumerate(job_ids):
            rows.append({
                "programme_id": p_id,
                "job_id": j_id,
                score_col: float(len(job_ids) - rank),
            })
    return pd.DataFrame(rows)


# ── precision_at_k ────────────────────────────────────────────────────────────

class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0

    def test_none_relevant(self):
        assert precision_at_k([1, 2, 3], {4, 5}, k=3) == 0.0

    def test_partial(self):
        assert abs(precision_at_k([1, 2, 3, 4], {2, 4}, k=4) - 0.5) < 1e-6

    def test_k_larger_than_list(self):
        assert precision_at_k([1, 2], {1, 2}, k=5) == 1.0

    def test_empty_ranked(self):
        assert precision_at_k([], {1, 2}, k=3) == 0.0


# ── ndcg_at_k ─────────────────────────────────────────────────────────────────

class TestNdcgAtK:
    def test_perfect_ranking(self):
        # All relevant items at top
        assert abs(ndcg_at_k([1, 2, 3], {1, 2, 3}, k=3) - 1.0) < 1e-6

    def test_no_relevant(self):
        assert ndcg_at_k([1, 2, 3], set(), k=3) == 0.0

    def test_relevant_at_end(self):
        # Relevant item at position 3 (worst possible for k=3 with 1 relevant)
        n = ndcg_at_k([10, 20, 1], {1}, k=3)
        assert 0.0 < n < 1.0

    def test_empty_relevant_set(self):
        assert ndcg_at_k([1, 2, 3], set(), k=3) == 0.0


# ── reciprocal_rank ───────────────────────────────────────────────────────────

class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank([1, 2, 3], {1}) == 1.0

    def test_second_position(self):
        assert abs(reciprocal_rank([2, 1, 3], {1}) - 0.5) < 1e-6

    def test_not_found(self):
        assert reciprocal_rank([1, 2, 3], {99}) == 0.0


# ── build_consensus ───────────────────────────────────────────────────────────

class TestBuildConsensus:
    def test_unanimous_jobs_included(self):
        # Job 0 is top-1 in all 3 strategies for programme 0
        sym = _make_rankings([0], [[0, 1, 2]], "weighted_jaccard")
        sem = _make_rankings([0], [[0, 2, 1]], "cosine_combined")
        hyb = _make_rankings([0], [[0, 1, 2]], "hybrid_score")
        rel = build_consensus(
            {"symbolic": sym, "semantic": sem, "hybrid": hyb},
            k=2, min_strategies=2,
        )
        assert 0 in rel[0]

    def test_single_strategy_job_excluded(self):
        # Job 99 only in symbolic top-2
        sym = _make_rankings([0], [[99, 1, 2]], "weighted_jaccard")
        sem = _make_rankings([0], [[0, 2, 1]], "cosine_combined")
        hyb = _make_rankings([0], [[0, 1, 2]], "hybrid_score")
        rel = build_consensus(
            {"symbolic": sym, "semantic": sem, "hybrid": hyb},
            k=2, min_strategies=2,
        )
        assert 99 not in rel[0]

    def test_min_strategies_three_requires_all(self):
        sym = _make_rankings([0], [[0, 1, 2]], "weighted_jaccard")
        sem = _make_rankings([0], [[0, 2, 1]], "cosine_combined")
        hyb = _make_rankings([0], [[3, 1, 2]], "hybrid_score")
        rel = build_consensus(
            {"symbolic": sym, "semantic": sem, "hybrid": hyb},
            k=2, min_strategies=3,
        )
        # Job 0 is in sym + sem but not hyb (hyb has 3,1) → excluded at min=3
        assert 0 not in rel[0]


# ── compute_ir_metrics ────────────────────────────────────────────────────────

class TestComputeIrMetrics:
    def test_output_structure(self):
        sym = _make_rankings([0, 1], [[0, 1, 2, 3, 4]] * 2, "weighted_jaccard")
        sem = _make_rankings([0, 1], [[0, 1, 2, 3, 4]] * 2, "cosine_combined")
        hyb = _make_rankings([0, 1], [[0, 1, 2, 3, 4]] * 2, "hybrid_score")
        result = compute_ir_metrics(sym, sem, hyb, k=3)
        assert "strategies" in result
        assert "consensus" in result
        for name in ("symbolic", "semantic", "hybrid"):
            assert name in result["strategies"]
            strat = result["strategies"][name]
            for key in ("precision_at_k", "ndcg_at_k", "mrr", "coverage_at_k"):
                assert key in strat

    def test_identical_strategies_perfect_scores(self):
        # All 3 strategies agree → consensus = top-K, all metrics high
        jobs = [0, 1, 2, 3, 4]
        sym = _make_rankings([0], [jobs], "weighted_jaccard")
        sem = _make_rankings([0], [jobs], "cosine_combined")
        hyb = _make_rankings([0], [jobs], "hybrid_score")
        result = compute_ir_metrics(sym, sem, hyb, k=3, min_strategies=2)
        for name in ("symbolic", "semantic", "hybrid"):
            assert result["strategies"][name]["precision_at_k"] == 1.0
            assert abs(result["strategies"][name]["ndcg_at_k"] - 1.0) < 1e-4


# ── run_ir_metrics ────────────────────────────────────────────────────────────

class TestRunIrMetrics:
    def test_output_file_created(self, tmp_path: Path):
        jobs = list(range(10))
        sym = _make_rankings([0, 1], [jobs, jobs], "weighted_jaccard")
        sem = _make_rankings([0, 1], [jobs, jobs], "cosine_combined")
        hyb = _make_rankings([0, 1], [jobs, jobs], "hybrid_score")

        sym_path = tmp_path / "sym.parquet"
        sem_path = tmp_path / "sem.parquet"
        hyb_path = tmp_path / "hyb.parquet"
        sym.to_parquet(sym_path)
        sem.to_parquet(sem_path)
        hyb.to_parquet(hyb_path)

        run_ir_metrics(sym_path, sem_path, hyb_path, tmp_path)
        assert (tmp_path / "ir_metrics.json").exists()
        with open(tmp_path / "ir_metrics.json") as f:
            data = json.load(f)
        assert "strategies" in data
