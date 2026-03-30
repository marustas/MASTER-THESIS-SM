"""
Tests for src/evaluation/significance.py.

Covers:
  - bootstrap_ci: known mean, CI width, empty input, deterministic seed
  - paired_wilcoxon: identical arrays, different arrays, insufficient data
  - compute_significance: output structure, all expected keys
  - run_significance: output file created
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.significance import (
    bootstrap_ci,
    paired_wilcoxon,
    compute_significance,
    run_significance,
)


# ── bootstrap_ci ──────────────────────────────────────────────────────────────

class TestBootstrapCI:
    def test_mean_correct(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = bootstrap_ci(values)
        assert abs(result["mean"] - 3.0) < 1e-4

    def test_ci_contains_mean(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = bootstrap_ci(values)
        assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]

    def test_narrow_ci_for_constant_values(self):
        values = np.array([5.0] * 100)
        result = bootstrap_ci(values)
        assert result["ci_lower"] == result["ci_upper"] == 5.0

    def test_empty_returns_none(self):
        result = bootstrap_ci(np.array([]))
        assert result["mean"] is None

    def test_nan_values_excluded(self):
        values = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        result = bootstrap_ci(values)
        assert abs(result["mean"] - 3.0) < 1e-4

    def test_deterministic_with_seed(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r1 = bootstrap_ci(values, seed=42)
        r2 = bootstrap_ci(values, seed=42)
        assert r1 == r2


# ── paired_wilcoxon ───────────────────────────────────────────────────────────

class TestPairedWilcoxon:
    def test_identical_arrays_high_p(self):
        a = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        result = paired_wilcoxon(a, a)
        # All differences are zero → insufficient non-zero diffs
        assert result["p_value"] is None

    def test_different_arrays_low_p(self):
        a = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        b = np.array([0.9, 0.95, 0.85, 0.92, 0.88, 0.91, 0.87, 0.93])
        result = paired_wilcoxon(a, b)
        assert result["p_value"] is not None
        assert result["p_value"] < 0.05

    def test_effect_size_bounded(self):
        a = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        b = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        result = paired_wilcoxon(a, b)
        if result["effect_size"] is not None:
            assert -1.0 <= result["effect_size"] <= 1.0

    def test_nan_pairs_excluded(self):
        a = np.array([0.1, np.nan, 0.3, 0.4, 0.5])
        b = np.array([0.9, 0.8, np.nan, 0.6, 0.4])
        result = paired_wilcoxon(a, b)
        assert result["n"] == 3

    def test_insufficient_nonzero_diffs(self):
        a = np.array([0.5, 0.5, 0.6])
        b = np.array([0.5, 0.5, 0.6])
        result = paired_wilcoxon(a, b)
        assert result["p_value"] is None


# ── compute_significance ──────────────────────────────────────────────────────

class TestComputeSignificance:
    @pytest.fixture
    def sample_per_programme(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        n = 20
        return pd.DataFrame({
            "programme_id": range(n),
            "spearman_sym_sem": rng.uniform(0.1, 0.5, n),
            "spearman_sym_hyb": rng.uniform(0.5, 0.9, n),
            "spearman_sem_hyb": rng.uniform(0.5, 0.9, n),
            "jaccard_sym_sem_at_5": rng.uniform(0.0, 0.3, n),
            "jaccard_sym_sem_at_10": rng.uniform(0.0, 0.3, n),
            "jaccard_sym_hyb_at_5": rng.uniform(0.1, 0.4, n),
            "jaccard_sym_hyb_at_10": rng.uniform(0.1, 0.4, n),
            "jaccard_sem_hyb_at_5": rng.uniform(0.3, 0.6, n),
            "jaccard_sem_hyb_at_10": rng.uniform(0.3, 0.6, n),
        })

    def test_output_has_bootstrap_ci(self, sample_per_programme):
        result = compute_significance(sample_per_programme, n_boot=100)
        assert "bootstrap_ci" in result
        assert "spearman_sym_sem" in result["bootstrap_ci"]

    def test_output_has_wilcoxon(self, sample_per_programme):
        result = compute_significance(sample_per_programme, n_boot=100)
        assert "wilcoxon_tests" in result
        assert len(result["wilcoxon_tests"]) == 3

    def test_all_spearman_cols_have_ci(self, sample_per_programme):
        result = compute_significance(sample_per_programme, n_boot=100)
        for col in ["spearman_sym_sem", "spearman_sym_hyb", "spearman_sem_hyb"]:
            assert col in result["bootstrap_ci"]
            ci = result["bootstrap_ci"][col]
            assert ci["ci_lower"] <= ci["mean"] <= ci["ci_upper"]

    def test_jaccard_cols_have_ci(self, sample_per_programme):
        result = compute_significance(sample_per_programme, n_boot=100)
        jaccard_keys = [k for k in result["bootstrap_ci"] if k.startswith("jaccard_")]
        assert len(jaccard_keys) == 6

    def test_n_programmes(self, sample_per_programme):
        result = compute_significance(sample_per_programme, n_boot=100)
        assert result["n_programmes"] == 20


# ── run_significance ──────────────────────────────────────────────────────────

class TestRunSignificance:
    def test_output_file_created(self, tmp_path: Path):
        rng = np.random.default_rng(42)
        n = 15
        df = pd.DataFrame({
            "programme_id": range(n),
            "spearman_sym_sem": rng.uniform(0.1, 0.5, n),
            "spearman_sym_hyb": rng.uniform(0.5, 0.9, n),
            "spearman_sem_hyb": rng.uniform(0.5, 0.9, n),
            "jaccard_sym_sem_at_5": rng.uniform(0.0, 0.3, n),
            "jaccard_sym_sem_at_10": rng.uniform(0.0, 0.3, n),
            "jaccard_sym_hyb_at_5": rng.uniform(0.1, 0.4, n),
            "jaccard_sym_hyb_at_10": rng.uniform(0.1, 0.4, n),
            "jaccard_sem_hyb_at_5": rng.uniform(0.3, 0.6, n),
            "jaccard_sem_hyb_at_10": rng.uniform(0.3, 0.6, n),
        })
        pp_path = tmp_path / "per_programme.parquet"
        df.to_parquet(pp_path, index=False)

        run_significance(pp_path, tmp_path, n_boot=50)

        sig_path = tmp_path / "significance.json"
        assert sig_path.exists()
        with open(sig_path) as f:
            data = json.load(f)
        assert "bootstrap_ci" in data
        assert "wilcoxon_tests" in data
