"""
Tests for src/evaluation/stability.py.

Covers:
  - _rank_vector: rank ordering, ties, missing programmes
  - compute_kendall_tau: perfect correlation, reversed, partial overlap
  - run_stability: output structure, strategy keys, metric ranges
  - run: output file created
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.stability import (
    _rank_vector,
    compute_kendall_tau,
    run_stability,
    run,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_rankings(
    programme_ids: list[int],
    job_ids: list[int],
    score_col: str = "score",
    scores: list[float] | None = None,
) -> pd.DataFrame:
    """Build a rankings DataFrame with all programme×job pairs."""
    rows = []
    idx = 0
    for pid in programme_ids:
        for jid in job_ids:
            s = scores[idx] if scores else float(jid)
            rows.append({
                "programme_id": pid,
                "job_id": jid,
                "programme_name": f"Prog{pid}",
                "job_title": f"Job{jid}",
                score_col: s,
            })
            idx += 1
    return pd.DataFrame(rows)


def _skill(uri: str, *, explicit: bool = True) -> dict:
    return {
        "esco_uri": uri,
        "preferred_label": uri.split(":")[-1],
        "matched_text": uri.split(":")[-1],
        "explicit": explicit,
        "implicit": not explicit,
        "confidence": 1.0,
    }


def _make_dataset(
    n_progs: int = 3,
    n_jobs: int = 10,
) -> pd.DataFrame:
    """Build a minimal dataset with all required alignment columns."""
    rng = np.random.default_rng(42)
    rows = []
    skills_pool = ["esco:python", "esco:java", "esco:sql", "esco:ml", "esco:docker"]

    for i in range(n_progs):
        emb = rng.normal(size=16).astype(np.float32)
        emb /= np.linalg.norm(emb)
        n_skills = rng.integers(1, 4)
        chosen = rng.choice(skills_pool, size=n_skills, replace=False).tolist()
        rows.append({
            "source_type": "programme",
            "name": f"Prog{i}",
            "cleaned_text": f"programme {i} about {' '.join(chosen)}",
            "embedding": emb,
            "skill_details": [_skill(u) for u in chosen],
        })

    for j in range(n_jobs):
        emb = rng.normal(size=16).astype(np.float32)
        emb /= np.linalg.norm(emb)
        n_skills = rng.integers(1, 4)
        chosen = rng.choice(skills_pool, size=n_skills, replace=False).tolist()
        rows.append({
            "source_type": "job_ad",
            "job_title": f"Job{j}",
            "cleaned_text": f"job {j} requires {' '.join(chosen)}",
            "embedding": emb,
            "skill_details": [_skill(u) for u in chosen],
        })

    return pd.DataFrame(rows)


# ── _rank_vector ─────────────────────────────────────────────────────────────


class TestRankVector:
    def test_basic_ordering(self):
        rankings = _make_rankings([0], [10, 20, 30], "score", [3.0, 1.0, 2.0])
        ranks = _rank_vector(rankings, 0, "score")
        # score order: 10(3.0) > 30(2.0) > 20(1.0) → ranks 0, 1, 2
        assert ranks[10] == 0
        assert ranks[30] == 1
        assert ranks[20] == 2

    def test_multiple_programmes(self):
        rankings = _make_rankings([0, 1], [10, 20], "score", [2.0, 1.0, 1.0, 2.0])
        r0 = _rank_vector(rankings, 0, "score")
        r1 = _rank_vector(rankings, 1, "score")
        assert r0[10] == 0  # prog 0: job 10 scores higher
        assert r1[20] == 0  # prog 1: job 20 scores higher


# ── compute_kendall_tau ──────────────────────────────────────────────────────


class TestComputeKendallTau:
    def test_identical_rankings(self):
        rankings = _make_rankings([0], [10, 20, 30, 40], "score")
        taus = compute_kendall_tau(rankings, rankings, [0], "score")
        assert len(taus) == 1
        assert taus[0] == pytest.approx(1.0)

    def test_reversed_rankings(self):
        full = _make_rankings([0], [10, 20, 30, 40], "score", [4.0, 3.0, 2.0, 1.0])
        rev = _make_rankings([0], [10, 20, 30, 40], "score", [1.0, 2.0, 3.0, 4.0])
        taus = compute_kendall_tau(full, rev, [0], "score")
        assert len(taus) == 1
        assert taus[0] == pytest.approx(-1.0)

    def test_partial_overlap(self):
        """Resampled rankings may have fewer jobs — only common jobs compared."""
        full = _make_rankings([0], [10, 20, 30, 40], "score", [4.0, 3.0, 2.0, 1.0])
        partial = _make_rankings([0], [10, 20, 30], "score", [3.0, 2.0, 1.0])
        taus = compute_kendall_tau(full, partial, [0], "score")
        assert len(taus) == 1
        assert taus[0] == pytest.approx(1.0)  # same relative order

    def test_too_few_common_jobs(self):
        """Fewer than 3 common jobs → skipped."""
        full = _make_rankings([0], [10, 20], "score")
        partial = _make_rankings([0], [10], "score")
        taus = compute_kendall_tau(full, partial, [0], "score")
        assert len(taus) == 0

    def test_multiple_programmes(self):
        full = _make_rankings([0, 1], [10, 20, 30], "score")
        taus = compute_kendall_tau(full, full, [0, 1], "score")
        assert len(taus) == 2
        assert all(t == pytest.approx(1.0) for t in taus)


# ── run_stability ────────────────────────────────────────────────────────────


class TestRunStability:
    def _mock_strategies(self):
        """Lightweight strategies that just return score based on index."""
        def _symbolic(df):
            from src.alignment.symbolic import align_symbolic
            return align_symbolic(df, top_n=5)[0]

        return {
            "symbolic": {
                "fn": _symbolic,
                "score_col": "weighted_jaccard",
            },
        }

    def test_output_structure(self):
        df = _make_dataset(n_progs=3, n_jobs=15)
        results = run_stability(
            df,
            n_resamples=5,
            sample_fraction=0.8,
            strategies=self._mock_strategies(),
        )
        assert "symbolic" in results
        metrics = results["symbolic"]
        assert "mean_tau" in metrics
        assert "std_tau" in metrics
        assert "ci_95_lower" in metrics
        assert "ci_95_upper" in metrics
        assert "n_resamples" in metrics
        assert metrics["n_resamples"] == 5

    def test_tau_in_valid_range(self):
        df = _make_dataset(n_progs=3, n_jobs=15)
        results = run_stability(
            df,
            n_resamples=5,
            sample_fraction=0.8,
            strategies=self._mock_strategies(),
        )
        tau = results["symbolic"]["mean_tau"]
        assert -1.0 <= tau <= 1.0

    def test_high_stability_with_full_sample(self):
        """100% sample should give high (but not necessarily perfect) correlation."""
        df = _make_dataset(n_progs=2, n_jobs=10)
        results = run_stability(
            df,
            n_resamples=3,
            sample_fraction=1.0,
            strategies=self._mock_strategies(),
        )
        # Not exactly 1.0 because index reordering can change tie-breaking
        assert results["symbolic"]["mean_tau"] > 0.5

    def test_seed_reproducibility(self):
        df = _make_dataset(n_progs=2, n_jobs=15)
        r1 = run_stability(df, n_resamples=5, strategies=self._mock_strategies(), seed=99)
        r2 = run_stability(df, n_resamples=5, strategies=self._mock_strategies(), seed=99)
        assert r1["symbolic"]["mean_tau"] == r2["symbolic"]["mean_tau"]


# ── run (pipeline entry point) ───────────────────────────────────────────────


class TestRunEntryPoint:
    def test_output_file(self, tmp_path):
        df = _make_dataset(n_progs=2, n_jobs=10)
        dataset_path = tmp_path / "dataset.parquet"
        df.to_parquet(dataset_path)

        import src.evaluation.stability as stability_mod

        def _mock_symbolic(df):
            from src.alignment.symbolic import align_symbolic
            return align_symbolic(df, top_n=5)[0]

        original_strategies = stability_mod.STRATEGIES
        stability_mod.STRATEGIES = {
            "symbolic": {"fn": _mock_symbolic, "score_col": "weighted_jaccard"},
        }
        try:
            run(
                dataset_path=dataset_path,
                output_dir=tmp_path,
                n_resamples=3,
            )
        finally:
            stability_mod.STRATEGIES = original_strategies

        assert (tmp_path / "stability.json").exists()
        with open(tmp_path / "stability.json") as fh:
            data = json.load(fh)
        assert "symbolic" in data
        assert "mean_tau" in data["symbolic"]
