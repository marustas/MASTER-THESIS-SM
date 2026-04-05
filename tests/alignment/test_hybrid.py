"""
Tests for src/alignment/hybrid.py.

Covers:
  - align_hybrid: output columns, row count ≤ semantic_top_n per programme,
                  hybrid_score formula with per-programme min-max normalisation,
                  sort order, alpha boundary values, invalid alpha raises ValueError
  - hybrid_score uses normalised cosine and programme_recall
  - run_hybrid_alignment: output files written, rankings shape
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.alignment.hybrid import align_hybrid, run_hybrid_alignment


# ── Helpers ────────────────────────────────────────────────────────────────────

def _l2(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _emb(seed: int, dim: int = 8) -> list[float]:
    rng = np.random.default_rng(seed)
    return _l2(rng.random(dim).astype(np.float32)).tolist()


def _skill(uri: str, *, explicit: bool = True) -> dict:
    return {
        "esco_uri": uri,
        "preferred_label": uri.split(":")[-1],
        "matched_text": uri.split(":")[-1],
        "explicit": explicit,
        "implicit": not explicit,
        "confidence": 1.0,
    }


def _make_df(
    n_prog: int,
    n_jobs: int,
    dim: int = 8,
    with_skills: bool = True,
) -> pd.DataFrame:
    skill_pool = [
        [_skill("esco:python"), _skill("esco:ml", explicit=False)],
        [_skill("esco:java")],
        [_skill("esco:docker"), _skill("esco:k8s")],
        [],
    ]
    rows = []
    for i in range(n_prog):
        rows.append({
            "source_type": "programme",
            "embedding": _emb(i, dim),
            "name": f"Prog{i}",
            "skill_details": skill_pool[i % len(skill_pool)] if with_skills else [],
        })
    for i in range(n_jobs):
        rows.append({
            "source_type": "job_ad",
            "embedding": _emb(i + 50, dim),
            "job_title": f"Job{i}",
            "skill_details": skill_pool[(i + 1) % len(skill_pool)] if with_skills else [],
        })
    return pd.DataFrame(rows)


# ── align_hybrid ───────────────────────────────────────────────────────────────

class TestAlignHybrid:
    def test_required_columns(self):
        df = _make_df(2, 4)
        rankings = align_hybrid(df, semantic_top_n=3)
        for col in ("programme_id", "job_id", "cosine_score",
                    "programme_recall", "hybrid_score"):
            assert col in rankings.columns

    def test_rows_capped_at_semantic_top_n(self):
        df = _make_df(3, 10)
        rankings = align_hybrid(df, semantic_top_n=4)
        for p_id in rankings["programme_id"].unique():
            assert len(rankings[rankings["programme_id"] == p_id]) <= 4

    def test_rows_not_exceed_available_jobs(self):
        df = _make_df(2, 3)
        rankings = align_hybrid(df, semantic_top_n=10)
        for p_id in rankings["programme_id"].unique():
            assert len(rankings[rankings["programme_id"] == p_id]) <= 3

    def test_sorted_by_hybrid_score_desc(self):
        df = _make_df(3, 8)
        rankings = align_hybrid(df, semantic_top_n=5)
        for p_id in rankings["programme_id"].unique():
            scores = rankings[rankings["programme_id"] == p_id]["hybrid_score"].tolist()
            assert scores == sorted(scores, reverse=True)

    def test_hybrid_score_uses_normalised_components(self):
        """Hybrid score = α·norm(cosine) + (1-α)·norm(recall), both min-max per programme."""
        df = _make_df(2, 5)
        alpha = 0.6
        rankings = align_hybrid(df, semantic_top_n=5, alpha=alpha)
        for p_id in rankings["programme_id"].unique():
            grp = rankings[rankings["programme_id"] == p_id]
            cos = grp["cosine_score"]
            rec = grp["programme_recall"]
            # normalise
            cos_lo, cos_hi = cos.min(), cos.max()
            rec_lo, rec_hi = rec.min(), rec.max()
            cos_n = (cos - cos_lo) / (cos_hi - cos_lo) if cos_hi > cos_lo else 0.0
            rec_n = (rec - rec_lo) / (rec_hi - rec_lo) if rec_hi > rec_lo else 0.0
            expected = alpha * cos_n + (1 - alpha) * rec_n
            pd.testing.assert_series_equal(
                grp["hybrid_score"].reset_index(drop=True).round(6),
                expected.reset_index(drop=True).round(6),
                check_names=False,
            )

    def test_hybrid_score_in_0_1_after_normalisation(self):
        df = _make_df(3, 8)
        rankings = align_hybrid(df, semantic_top_n=5)
        assert (rankings["hybrid_score"] >= -1e-9).all()
        assert (rankings["hybrid_score"] <= 1.0 + 1e-9).all()

    def test_alpha_one_hybrid_equals_normalised_cosine(self):
        df = _make_df(2, 4)
        rankings = align_hybrid(df, semantic_top_n=4, alpha=1.0)
        # With alpha=1, hybrid = norm(cosine); top per programme should be 1.0
        for p_id in rankings["programme_id"].unique():
            grp = rankings[rankings["programme_id"] == p_id]
            if len(grp) > 1:
                assert grp["hybrid_score"].iloc[0] == pytest.approx(1.0, abs=1e-6)

    def test_alpha_zero_hybrid_equals_normalised_recall(self):
        df = _make_df(2, 4)
        rankings = align_hybrid(df, semantic_top_n=4, alpha=0.0)
        # With alpha=0, hybrid = norm(recall); top per programme should be 1.0
        for p_id in rankings["programme_id"].unique():
            grp = rankings[rankings["programme_id"] == p_id]
            if len(grp) > 1:
                assert grp["hybrid_score"].iloc[0] == pytest.approx(1.0, abs=1e-6)

    def test_invalid_alpha_raises(self):
        df = _make_df(2, 3)
        with pytest.raises(ValueError, match="alpha"):
            align_hybrid(df, alpha=1.5)
        with pytest.raises(ValueError, match="alpha"):
            align_hybrid(df, alpha=-0.1)

    def test_no_skills_runs_without_error(self):
        df = _make_df(2, 4, with_skills=False)
        rankings = align_hybrid(df, semantic_top_n=3)
        assert len(rankings) > 0
        assert (rankings["programme_recall"] == 0.0).all()


# ── run_hybrid_alignment ───────────────────────────────────────────────────────

class TestRunHybridAlignment:
    def test_output_files_created(self, tmp_path):
        df = _make_df(2, 4)
        dataset_path = tmp_path / "dataset.parquet"
        df.to_parquet(dataset_path)
        output_dir = tmp_path / "exp3_hybrid"
        run_hybrid_alignment(
            dataset_path=dataset_path,
            output_dir=output_dir,
            semantic_top_n=3,
            alpha=0.5,
        )
        assert (output_dir / "rankings.parquet").exists()
        assert (output_dir / "summary.json").exists()

    def test_rankings_capped_at_top_n(self, tmp_path):
        df = _make_df(2, 6)
        dataset_path = tmp_path / "dataset.parquet"
        df.to_parquet(dataset_path)
        output_dir = tmp_path / "exp3"
        run_hybrid_alignment(
            dataset_path=dataset_path,
            output_dir=output_dir,
            semantic_top_n=3,
        )
        rankings = pd.read_parquet(output_dir / "rankings.parquet")
        for p_id in rankings["programme_id"].unique():
            assert len(rankings[rankings["programme_id"] == p_id]) <= 3
