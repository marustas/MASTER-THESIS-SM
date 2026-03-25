"""
Tests for src/evaluation/cross_strategy.py.

Covers:
  - _spearman_pair: identical rankings → ρ=1, reversed → ρ=-1,
                    insufficient overlap → NaN, intersection logic
  - _top_k_jaccard: identical sets → 1, disjoint → 0, partial overlap
  - evaluate: output shape, required columns, top-1 agreement flag,
              Spearman/Jaccard values for known inputs,
              programmes absent from one strategy are skipped
  - run_evaluation: output files created, per_programme row count
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.cross_strategy import (
    _spearman_pair,
    _top_k_jaccard,
    evaluate,
    run_evaluation,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_rankings(
    programme_ids: list[int],
    job_id_lists: list[list[int]],
    score_col: str,
    score_lists: list[list[float]] | None = None,
) -> pd.DataFrame:
    """Build a minimal rankings DataFrame for one strategy."""
    rows = []
    for p_id, job_ids, in zip(programme_ids, job_id_lists):
        scores = (
            score_lists[programme_ids.index(p_id)]
            if score_lists
            else list(range(len(job_ids), 0, -1))
        )
        for job_id, score in zip(job_ids, scores):
            rows.append({
                "programme_id": p_id,
                "job_id": job_id,
                score_col: float(score),
                "programme_name": f"Prog{p_id}",
                "job_title": f"Job{job_id}",
            })
    return pd.DataFrame(rows)


def _sym(programme_ids, job_id_lists, score_lists=None):
    return _make_rankings(programme_ids, job_id_lists, "weighted_jaccard", score_lists)

def _sem(programme_ids, job_id_lists, score_lists=None):
    return _make_rankings(programme_ids, job_id_lists, "cosine_combined", score_lists)

def _hyb(programme_ids, job_id_lists, score_lists=None):
    return _make_rankings(programme_ids, job_id_lists, "hybrid_score", score_lists)


def _simple_trio(n_prog=2, n_jobs=5):
    """Three strategies with the same job sets for easy assertion."""
    prog_ids = list(range(n_prog))
    job_ids = [list(range(n_jobs))] * n_prog
    return (
        _sym(prog_ids, job_ids),
        _sem(prog_ids, job_ids),
        _hyb(prog_ids, job_ids),
    )


# ── _spearman_pair ─────────────────────────────────────────────────────────────

class TestSpearmanPair:
    def _series(self, jobs, scores):
        return pd.Series(scores), pd.Series(jobs)

    def test_identical_scores_returns_one(self):
        scores = pd.Series([3.0, 2.0, 1.0])
        jobs = pd.Series([1, 2, 3])
        rho = _spearman_pair(scores, scores, jobs, jobs)
        assert rho == pytest.approx(1.0)

    def test_reversed_scores_returns_minus_one(self):
        jobs = pd.Series([1, 2, 3])
        a = pd.Series([3.0, 2.0, 1.0])
        b = pd.Series([1.0, 2.0, 3.0])
        rho = _spearman_pair(a, b, jobs, jobs)
        assert rho == pytest.approx(-1.0)

    def test_insufficient_overlap_returns_nan(self):
        # Only 2 shared jobs — below minimum of 3
        a_jobs = pd.Series([1, 2, 99])
        b_jobs = pd.Series([1, 2, 88])
        a_scores = pd.Series([3.0, 2.0, 1.0])
        b_scores = pd.Series([3.0, 2.0, 1.0])
        rho = _spearman_pair(a_scores, b_scores, a_jobs, b_jobs)
        assert np.isnan(rho)

    def test_intersection_used(self):
        # a has jobs 1,2,3; b has jobs 1,2,3,4 — should correlate on 1,2,3
        a_jobs = pd.Series([1, 2, 3])
        b_jobs = pd.Series([1, 2, 3, 4])
        a_scores = pd.Series([3.0, 2.0, 1.0])
        b_scores = pd.Series([3.0, 2.0, 1.0, 0.5])
        rho = _spearman_pair(a_scores, b_scores, a_jobs, b_jobs)
        assert rho == pytest.approx(1.0)


# ── _top_k_jaccard ─────────────────────────────────────────────────────────────

class TestTopKJaccard:
    def test_identical_sets_returns_one(self):
        jobs = pd.Series([1, 2, 3, 4, 5])
        assert _top_k_jaccard(jobs, jobs, k=3) == pytest.approx(1.0)

    def test_disjoint_sets_returns_zero(self):
        a = pd.Series([1, 2, 3])
        b = pd.Series([4, 5, 6])
        assert _top_k_jaccard(a, b, k=3) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = pd.Series([1, 2, 3])
        b = pd.Series([2, 3, 4])
        # intersection={2,3}, union={1,2,3,4} → 2/4 = 0.5
        assert _top_k_jaccard(a, b, k=3) == pytest.approx(0.5)

    def test_k_limits_comparison(self):
        a = pd.Series([1, 2, 3, 4, 5])
        b = pd.Series([1, 2, 99, 98, 97])
        # top-2 from each: {1,2} vs {1,2} → 1.0
        assert _top_k_jaccard(a, b, k=2) == pytest.approx(1.0)
        # top-3: {1,2,3} vs {1,2,99} → 2/4 = 0.5
        assert _top_k_jaccard(a, b, k=3) == pytest.approx(0.5)

    def test_empty_series_returns_nan(self):
        empty = pd.Series([], dtype=int)
        assert np.isnan(_top_k_jaccard(empty, empty, k=5))


# ── evaluate ───────────────────────────────────────────────────────────────────

class TestEvaluate:
    def test_output_shape(self):
        sym, sem, hyb = _simple_trio(n_prog=3, n_jobs=5)
        pp, _ = evaluate(sym, sem, hyb, top_k=4)
        assert len(pp) == 3

    def test_required_columns_present(self):
        sym, sem, hyb = _simple_trio()
        pp, _ = evaluate(sym, sem, hyb, top_k=4)
        for col in (
            "programme_id",
            "spearman_sym_sem", "spearman_sym_hyb", "spearman_sem_hyb",
            "top1_symbolic", "top1_semantic", "top1_hybrid",
            "top1_all_agree",
        ):
            assert col in pp.columns

    def test_identical_rankings_spearman_is_one(self):
        jobs = list(range(6))
        scores = [float(i) for i in range(6, 0, -1)]
        sym = _sym([0], [jobs], [scores])
        # Use same job order and scores for all three strategies
        sem = _sem([0], [jobs], [scores])
        hyb = _hyb([0], [jobs], [scores])
        pp, _ = evaluate(sym, sem, hyb, top_k=4)
        assert pp.iloc[0]["spearman_sym_sem"] == pytest.approx(1.0, abs=1e-5)
        assert pp.iloc[0]["spearman_sym_hyb"] == pytest.approx(1.0, abs=1e-5)
        assert pp.iloc[0]["spearman_sem_hyb"] == pytest.approx(1.0, abs=1e-5)

    def test_identical_top1_sets_agree_flag_true(self):
        jobs = [10, 20, 30]
        scores = [3.0, 2.0, 1.0]
        sym = _sym([0], [jobs], [scores])
        sem = _sem([0], [jobs], [scores])
        hyb = _hyb([0], [jobs], [scores])
        pp, _ = evaluate(sym, sem, hyb, top_k=3)
        assert pp.iloc[0]["top1_all_agree"] == True

    def test_different_top1_sets_agree_flag_false(self):
        sym = _sym([0], [[1, 2, 3]], [[3.0, 2.0, 1.0]])
        sem = _sem([0], [[2, 1, 3]], [[3.0, 2.0, 1.0]])   # job 2 is top
        hyb = _hyb([0], [[1, 2, 3]], [[3.0, 2.0, 1.0]])
        pp, _ = evaluate(sym, sem, hyb, top_k=3)
        assert pp.iloc[0]["top1_all_agree"] == False

    def test_programme_absent_from_symbolic_is_skipped(self):
        # sym only has programme 0; sem and hyb have 0 and 1
        sym = _sym([0], [[1, 2, 3]])
        sem = _sem([0, 1], [[1, 2, 3], [4, 5, 6]])
        hyb = _hyb([0, 1], [[1, 2, 3], [4, 5, 6]])
        pp, _ = evaluate(sym, sem, hyb, top_k=2)
        assert set(pp["programme_id"].tolist()) == {0}

    def test_summary_agreement_rate(self):
        jobs = [1, 2, 3]
        scores = [3.0, 2.0, 1.0]
        sym = _sym([0, 1], [jobs, jobs], [scores, scores])
        sem = _sem([0, 1], [jobs, jobs], [scores, scores])
        hyb = _hyb([0, 1], [jobs, jobs], [scores, scores])
        _, summary = evaluate(sym, sem, hyb, top_k=3)
        assert summary["top1_agreement_rate"] == pytest.approx(1.0)
        assert summary["top1_agreements"] == 2

    def test_jaccard_columns_present(self):
        sym, sem, hyb = _simple_trio(n_jobs=6)
        pp, _ = evaluate(sym, sem, hyb, top_k=4)
        for col in (
            "jaccard_sym_sem_at_2", "jaccard_sym_sem_at_4",
            "jaccard_sym_hyb_at_2", "jaccard_sym_hyb_at_4",
            "jaccard_sem_hyb_at_2", "jaccard_sem_hyb_at_4",
        ):
            assert col in pp.columns


# ── run_evaluation ─────────────────────────────────────────────────────────────

class TestRunEvaluation:
    def _write_parquets(self, tmp_path, n_prog=2, n_jobs=5):
        sym, sem, hyb = _simple_trio(n_prog=n_prog, n_jobs=n_jobs)
        (tmp_path / "exp1").mkdir()
        (tmp_path / "exp2").mkdir()
        (tmp_path / "exp3").mkdir()
        sym.to_parquet(tmp_path / "exp1" / "rankings.parquet")
        sem.to_parquet(tmp_path / "exp2" / "rankings.parquet")
        hyb.to_parquet(tmp_path / "exp3" / "rankings.parquet")
        return (
            tmp_path / "exp1" / "rankings.parquet",
            tmp_path / "exp2" / "rankings.parquet",
            tmp_path / "exp3" / "rankings.parquet",
        )

    def test_output_files_created(self, tmp_path):
        sym_p, sem_p, hyb_p = self._write_parquets(tmp_path)
        out = tmp_path / "evaluation"
        run_evaluation(sym_p, sem_p, hyb_p, output_dir=out, top_k=3)
        assert (out / "per_programme.parquet").exists()
        assert (out / "summary.json").exists()

    def test_per_programme_row_count(self, tmp_path):
        sym_p, sem_p, hyb_p = self._write_parquets(tmp_path, n_prog=3)
        out = tmp_path / "eval"
        run_evaluation(sym_p, sem_p, hyb_p, output_dir=out, top_k=3)
        pp = pd.read_parquet(out / "per_programme.parquet")
        assert len(pp) == 3

    def test_summary_json_valid(self, tmp_path):
        sym_p, sem_p, hyb_p = self._write_parquets(tmp_path)
        out = tmp_path / "eval"
        run_evaluation(sym_p, sem_p, hyb_p, output_dir=out, top_k=4)
        with open(out / "summary.json") as f:
            summary = json.load(f)
        assert "n_programmes" in summary
        assert "top1_agreement_rate" in summary
        assert "spearman" in summary
        assert "jaccard" in summary
