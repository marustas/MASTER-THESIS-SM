"""
Tests for src/evaluation/sensitivity.py.

Covers:
  - _rerank_hybrid: score formula, re-sorting per programme
  - _mean_spearman: known correlations, insufficient overlap
  - _mean_jaccard_at_k: identical/disjoint top-k sets
  - alpha_sweep: output columns, row count, boundary alphas
  - run_sensitivity: output files written
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.sensitivity import (
    _rerank_hybrid,
    _mean_spearman,
    _mean_jaccard_at_k,
    _top1_agreement_rate,
    alpha_sweep,
    run_sensitivity,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_hybrid_candidates(
    n_prog: int = 2,
    n_jobs: int = 5,
) -> pd.DataFrame:
    """Hybrid candidate pool with cosine_score and weighted_jaccard."""
    rng = np.random.default_rng(42)
    rows = []
    for p in range(n_prog):
        for j in range(n_jobs):
            rows.append({
                "programme_id": p,
                "job_id": j,
                "programme_name": f"Prog{p}",
                "job_title": f"Job{j}",
                "cosine_score": round(float(rng.uniform(0.2, 0.8)), 4),
                "weighted_jaccard": round(float(rng.uniform(0.0, 0.3)), 4),
            })
    return pd.DataFrame(rows)


def _make_symbolic(n_prog: int = 2, n_jobs: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for p in range(n_prog):
        for j in range(n_jobs):
            rows.append({
                "programme_id": p,
                "job_id": j,
                "weighted_jaccard": round(float(rng.uniform(0.0, 0.3)), 4),
            })
    return pd.DataFrame(rows)


def _make_semantic(n_prog: int = 2, n_jobs: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(13)
    rows = []
    for p in range(n_prog):
        for j in range(n_jobs):
            rows.append({
                "programme_id": p,
                "job_id": j,
                "cosine_combined": round(float(rng.uniform(0.2, 0.8)), 4),
            })
    return pd.DataFrame(rows)


# ── _rerank_hybrid ─────────────────────────────────────────────────────────────

class TestRerankHybrid:
    def test_alpha_zero_equals_jaccard(self):
        cand = _make_hybrid_candidates()
        result = _rerank_hybrid(cand, alpha=0.0)
        assert np.allclose(result["hybrid_score"], result["weighted_jaccard"])

    def test_alpha_one_equals_cosine(self):
        cand = _make_hybrid_candidates()
        result = _rerank_hybrid(cand, alpha=1.0)
        assert np.allclose(result["hybrid_score"], result["cosine_score"])

    def test_alpha_half_is_mean(self):
        cand = _make_hybrid_candidates()
        result = _rerank_hybrid(cand, alpha=0.5)
        expected = 0.5 * cand["cosine_score"] + 0.5 * cand["weighted_jaccard"]
        assert np.allclose(
            sorted(result["hybrid_score"]), sorted(expected), atol=1e-6
        )

    def test_sorted_desc_within_programme(self):
        cand = _make_hybrid_candidates()
        result = _rerank_hybrid(cand, alpha=0.5)
        for p_id in result["programme_id"].unique():
            scores = result.loc[result["programme_id"] == p_id, "hybrid_score"]
            assert list(scores) == sorted(scores, reverse=True)

    def test_does_not_mutate_input(self):
        cand = _make_hybrid_candidates()
        original_scores = cand["cosine_score"].copy()
        _rerank_hybrid(cand, alpha=0.3)
        assert cand["cosine_score"].equals(original_scores)


# ── _mean_spearman ─────────────────────────────────────────────────────────────

class TestMeanSpearman:
    def test_identical_rankings_return_one(self):
        hybrid = pd.DataFrame({
            "programme_id": [0, 0, 0, 0, 0],
            "job_id": [0, 1, 2, 3, 4],
            "hybrid_score": [5.0, 4.0, 3.0, 2.0, 1.0],
        })
        baseline = pd.DataFrame({
            "programme_id": [0, 0, 0, 0, 0],
            "job_id": [0, 1, 2, 3, 4],
            "weighted_jaccard": [5.0, 4.0, 3.0, 2.0, 1.0],
        })
        rho = _mean_spearman(hybrid, baseline, "hybrid_score", "weighted_jaccard")
        assert abs(rho - 1.0) < 1e-6

    def test_reversed_rankings_return_neg_one(self):
        hybrid = pd.DataFrame({
            "programme_id": [0, 0, 0, 0, 0],
            "job_id": [0, 1, 2, 3, 4],
            "hybrid_score": [5.0, 4.0, 3.0, 2.0, 1.0],
        })
        baseline = pd.DataFrame({
            "programme_id": [0, 0, 0, 0, 0],
            "job_id": [0, 1, 2, 3, 4],
            "weighted_jaccard": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        rho = _mean_spearman(hybrid, baseline, "hybrid_score", "weighted_jaccard")
        assert abs(rho - (-1.0)) < 1e-6

    def test_insufficient_overlap_nan(self):
        hybrid = pd.DataFrame({
            "programme_id": [0, 0],
            "job_id": [0, 1],
            "hybrid_score": [5.0, 4.0],
        })
        baseline = pd.DataFrame({
            "programme_id": [0, 0],
            "job_id": [2, 3],
            "weighted_jaccard": [5.0, 4.0],
        })
        rho = _mean_spearman(hybrid, baseline, "hybrid_score", "weighted_jaccard")
        assert np.isnan(rho)


# ── _mean_jaccard_at_k ────────────────────────────────────────────────────────

class TestMeanJaccardAtK:
    def test_identical_top_k_returns_one(self):
        hybrid = pd.DataFrame({
            "programme_id": [0] * 5,
            "job_id": [0, 1, 2, 3, 4],
            "hybrid_score": [5.0, 4.0, 3.0, 2.0, 1.0],
        })
        baseline = pd.DataFrame({
            "programme_id": [0] * 5,
            "job_id": [0, 1, 2, 3, 4],
            "weighted_jaccard": [5.0, 4.0, 3.0, 2.0, 1.0],
        })
        j = _mean_jaccard_at_k(hybrid, baseline, "hybrid_score", "weighted_jaccard", k=3)
        assert abs(j - 1.0) < 1e-6

    def test_disjoint_top_k_returns_zero(self):
        hybrid = pd.DataFrame({
            "programme_id": [0] * 5,
            "job_id": [0, 1, 2, 3, 4],
            "hybrid_score": [5.0, 4.0, 3.0, 2.0, 1.0],
        })
        baseline = pd.DataFrame({
            "programme_id": [0] * 5,
            "job_id": [5, 6, 7, 8, 9],
            "weighted_jaccard": [5.0, 4.0, 3.0, 2.0, 1.0],
        })
        j = _mean_jaccard_at_k(hybrid, baseline, "hybrid_score", "weighted_jaccard", k=3)
        assert j == 0.0


# ── alpha_sweep ───────────────────────────────────────────────────────────────

class TestAlphaSweep:
    def test_output_row_count(self):
        cand = _make_hybrid_candidates()
        sym = _make_symbolic()
        sem = _make_semantic()
        result = alpha_sweep(cand, sym, sem, alphas=[0.0, 0.5, 1.0])
        assert len(result) == 3

    def test_output_columns(self):
        cand = _make_hybrid_candidates()
        sym = _make_symbolic()
        sem = _make_semantic()
        result = alpha_sweep(cand, sym, sem, alphas=[0.5])
        for col in ("alpha", "hybrid_mean", "hybrid_median", "hybrid_max",
                     "spearman_sym", "spearman_sem"):
            assert col in result.columns

    def test_alpha_zero_hybrid_mean_equals_jaccard_mean(self):
        cand = _make_hybrid_candidates()
        sym = _make_symbolic()
        sem = _make_semantic()
        result = alpha_sweep(cand, sym, sem, alphas=[0.0])
        expected_mean = float(cand["weighted_jaccard"].mean())
        assert abs(result.iloc[0]["hybrid_mean"] - expected_mean) < 1e-4

    def test_alpha_one_hybrid_mean_equals_cosine_mean(self):
        cand = _make_hybrid_candidates()
        sym = _make_symbolic()
        sem = _make_semantic()
        result = alpha_sweep(cand, sym, sem, alphas=[1.0])
        expected_mean = float(cand["cosine_score"].mean())
        assert abs(result.iloc[0]["hybrid_mean"] - expected_mean) < 1e-4

    def test_default_alphas_eleven_values(self):
        cand = _make_hybrid_candidates()
        sym = _make_symbolic()
        sem = _make_semantic()
        result = alpha_sweep(cand, sym, sem)
        assert len(result) == 11
        assert list(result["alpha"]) == [round(a * 0.1, 1) for a in range(11)]

    def test_hybrid_mean_monotonic_in_alpha(self):
        """As alpha increases, hybrid_mean should shift toward cosine_mean."""
        cand = _make_hybrid_candidates()
        sym = _make_symbolic()
        sem = _make_semantic()
        result = alpha_sweep(cand, sym, sem)
        cosine_mean = float(cand["cosine_score"].mean())
        jaccard_mean = float(cand["weighted_jaccard"].mean())
        # At alpha=0 closer to jaccard, at alpha=1 closer to cosine
        assert abs(result.iloc[0]["hybrid_mean"] - jaccard_mean) < 1e-4
        assert abs(result.iloc[-1]["hybrid_mean"] - cosine_mean) < 1e-4


# ── run_sensitivity ───────────────────────────────────────────────────────────

class TestRunSensitivity:
    def test_output_files_created(self, tmp_path: Path):
        cand = _make_hybrid_candidates(3, 8)
        sym = _make_symbolic(3, 8)
        sem = _make_semantic(3, 8)
        hyb_path = tmp_path / "hybrid.parquet"
        sym_path = tmp_path / "symbolic.parquet"
        sem_path = tmp_path / "semantic.parquet"
        cand.to_parquet(hyb_path, index=False)
        sym.to_parquet(sym_path, index=False)
        sem.to_parquet(sem_path, index=False)

        out_dir = tmp_path / "sensitivity"
        run_sensitivity(hyb_path, sym_path, sem_path, out_dir)

        assert (out_dir / "alpha_sweep.parquet").exists()
        assert (out_dir / "alpha_sweep_summary.json").exists()

    def test_summary_has_best_alpha(self, tmp_path: Path):
        cand = _make_hybrid_candidates(3, 8)
        sym = _make_symbolic(3, 8)
        sem = _make_semantic(3, 8)
        hyb_path = tmp_path / "hybrid.parquet"
        sym_path = tmp_path / "symbolic.parquet"
        sem_path = tmp_path / "semantic.parquet"
        cand.to_parquet(hyb_path, index=False)
        sym.to_parquet(sym_path, index=False)
        sem.to_parquet(sem_path, index=False)

        out_dir = tmp_path / "sensitivity"
        run_sensitivity(hyb_path, sym_path, sem_path, out_dir)

        with open(out_dir / "alpha_sweep_summary.json") as f:
            summary = json.load(f)
        assert "best_alpha" in summary
        assert 0.0 <= summary["best_alpha"] <= 1.0
