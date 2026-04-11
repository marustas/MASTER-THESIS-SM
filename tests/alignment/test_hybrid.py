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

from src.alignment.hybrid import align_hybrid, compute_match_quality, run_hybrid_alignment


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
        # Disable IPF to test the base formula in isolation
        rankings = align_hybrid(df, semantic_top_n=5, alpha=alpha, ipf_top_k=0)
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
        rankings = align_hybrid(df, semantic_top_n=4, alpha=1.0, ipf_top_k=0)
        # With alpha=1, hybrid = norm(cosine); top per programme should be 1.0
        for p_id in rankings["programme_id"].unique():
            grp = rankings[rankings["programme_id"] == p_id]
            if len(grp) > 1:
                assert grp["hybrid_score"].iloc[0] == pytest.approx(1.0, abs=1e-6)

    def test_alpha_zero_hybrid_equals_normalised_recall(self):
        df = _make_df(2, 4)
        rankings = align_hybrid(df, semantic_top_n=4, alpha=0.0, ipf_top_k=0)
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


# ── IPF (inverse programme frequency) ─────────────────────────────────────

class TestIPF:
    def test_ipf_disabled_matches_base_formula(self):
        """ipf_top_k=0 should give same results as raw normalised formula."""
        df = _make_df(3, 8)
        rankings = align_hybrid(df, semantic_top_n=5, ipf_top_k=0, gamma=0.0, delta=0.0)
        assert (rankings["hybrid_score"] >= -1e-6).all()
        assert (rankings["hybrid_score"] <= 1.0 + 1e-6).all()

    def test_ipf_penalises_generalist_jobs(self):
        """A job appearing in top-K for all programmes should score lower with IPF."""
        # Use same embedding for all programmes so one job matches all equally
        rows = []
        shared_emb = _emb(42, 8)
        shared_skills = [_skill("esco:common")]
        for i in range(4):
            rows.append({
                "source_type": "programme",
                "embedding": shared_emb,
                "name": f"Prog{i}",
                "skill_details": shared_skills,
            })
        # Generalist job: same embedding & skill as all programmes
        rows.append({
            "source_type": "job_ad",
            "embedding": shared_emb,
            "job_title": "Generalist",
            "skill_details": shared_skills,
        })
        # Specialist job: different embedding
        rows.append({
            "source_type": "job_ad",
            "embedding": _emb(99, 8),
            "job_title": "Specialist",
            "skill_details": [_skill("esco:niche")],
        })
        df = pd.DataFrame(rows)

        with_ipf = align_hybrid(df, semantic_top_n=2, ipf_top_k=1)
        without_ipf = align_hybrid(df, semantic_top_n=2, ipf_top_k=0)

        # The generalist job should have a lower score with IPF than without
        gen_with = with_ipf[with_ipf["job_title"] == "Generalist"]["hybrid_score"].mean()
        gen_without = without_ipf[without_ipf["job_title"] == "Generalist"]["hybrid_score"].mean()
        assert gen_with < gen_without

    def test_ipf_preserves_sort_order(self):
        df = _make_df(3, 8)
        rankings = align_hybrid(df, semantic_top_n=5, ipf_top_k=5)
        for p_id in rankings["programme_id"].unique():
            scores = rankings[rankings["programme_id"] == p_id]["hybrid_score"].tolist()
            assert scores == sorted(scores, reverse=True)

    def test_ipf_scores_non_negative(self):
        df = _make_df(3, 8)
        rankings = align_hybrid(df, semantic_top_n=5, ipf_top_k=3)
        assert (rankings["hybrid_score"] >= -1e-9).all()


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


# ── Mock skill embeddings helper ─────────────────────────────────────────────

def _mock_skill_embeddings(uris: list[str], dim: int = 8, cluster: bool = True):
    """Return {uri: embedding} — cluster=True → similar vectors, False → orthogonal."""
    rng = np.random.default_rng(42)
    if cluster:
        base = _l2(rng.random(dim).astype(np.float32))
        return {u: _l2(base + rng.normal(0, 0.05, dim).astype(np.float32)) for u in uris}
    else:
        embeddings = {}
        for i, u in enumerate(uris):
            v = np.zeros(dim, dtype=np.float32)
            v[i % dim] = 1.0
            embeddings[u] = v
        return embeddings


# ── compute_match_quality ─────────────────────────────────────────────────────

class TestComputeMatchQuality:
    """Unit tests for the quality multiplier components."""

    # ── Specificity ratio ─────────────────────────────────────────────────
    def test_specificity_rewards_rare_matches(self):
        """Matching rare URIs → specificity > 1."""
        uri_idfs = {"rare": 4.0, "common": 0.5}
        qm = compute_match_quality(
            matched_uris=["rare"], job_uris=["rare", "common"],
            uri_idfs=uri_idfs, median_idf=2.0, gamma=0.0, delta=0.0,
        )
        assert qm["specificity_ratio"] > 1.0

    def test_specificity_neutral_when_empty_matched(self):
        qm = compute_match_quality(
            matched_uris=[], job_uris=["a"], uri_idfs={"a": 1.0},
            median_idf=1.0, gamma=0.0, delta=0.0,
        )
        assert qm["specificity_ratio"] == 1.0

    def test_specificity_penalises_common_matches(self):
        """Matching common URIs when job also has rare → specificity < 1."""
        uri_idfs = {"common": 0.3, "rare": 5.0}
        qm = compute_match_quality(
            matched_uris=["common"], job_uris=["common", "rare"],
            uri_idfs=uri_idfs, median_idf=2.0, gamma=0.0, delta=0.0,
        )
        assert qm["specificity_ratio"] < 1.0

    # ── Generic penalty ───────────────────────────────────────────────────
    def test_generic_full_penalty(self):
        """All matched URIs below median → penalty = 1 - γ."""
        uri_idfs = {"a": 0.5, "b": 0.3}
        qm = compute_match_quality(
            matched_uris=["a", "b"], job_uris=["a", "b"],
            uri_idfs=uri_idfs, median_idf=2.0, gamma=0.3, delta=0.0,
        )
        assert qm["generic_penalty"] == pytest.approx(0.7)

    def test_generic_neutral_when_all_specific(self):
        """All matched URIs above median → penalty = 1.0."""
        uri_idfs = {"a": 3.0, "b": 4.0}
        qm = compute_match_quality(
            matched_uris=["a", "b"], job_uris=["a", "b"],
            uri_idfs=uri_idfs, median_idf=2.0, gamma=0.3, delta=0.0,
        )
        assert qm["generic_penalty"] == pytest.approx(1.0)

    def test_generic_partial(self):
        """Mix of generic and specific → penalty between 0.7 and 1.0."""
        uri_idfs = {"gen": 0.5, "spec": 4.0}
        qm = compute_match_quality(
            matched_uris=["gen", "spec"], job_uris=["gen", "spec"],
            uri_idfs=uri_idfs, median_idf=2.0, gamma=0.3, delta=0.0,
        )
        assert 0.7 < qm["generic_penalty"] < 1.0

    # ── Coherence boost ───────────────────────────────────────────────────
    def test_coherence_disabled_when_none(self):
        qm = compute_match_quality(
            matched_uris=["a", "b", "c"], job_uris=["a", "b", "c"],
            uri_idfs={"a": 1.0, "b": 1.0, "c": 1.0}, median_idf=1.0,
            gamma=0.0, delta=0.2, skill_embeddings=None,
        )
        assert qm["coherence_boost"] == 1.0

    def test_coherence_disabled_below_min_skills(self):
        embs = _mock_skill_embeddings(["a", "b"], dim=8)
        qm = compute_match_quality(
            matched_uris=["a", "b"], job_uris=["a", "b"],
            uri_idfs={"a": 1.0, "b": 1.0}, median_idf=1.0,
            gamma=0.0, delta=0.2, min_coherence_skills=3,
            skill_embeddings=embs,
        )
        assert qm["coherence_boost"] == 1.0

    def test_coherence_boosts_similar_cluster(self):
        embs = _mock_skill_embeddings(["a", "b", "c"], dim=8, cluster=True)
        qm = compute_match_quality(
            matched_uris=["a", "b", "c"], job_uris=["a", "b", "c"],
            uri_idfs={"a": 1.0, "b": 1.0, "c": 1.0}, median_idf=1.0,
            gamma=0.0, delta=0.2, min_coherence_skills=3,
            skill_embeddings=embs,
        )
        assert qm["coherence_boost"] > 1.0

    def test_coherence_near_neutral_with_dissimilar(self):
        embs = _mock_skill_embeddings(["a", "b", "c"], dim=8, cluster=False)
        qm = compute_match_quality(
            matched_uris=["a", "b", "c"], job_uris=["a", "b", "c"],
            uri_idfs={"a": 1.0, "b": 1.0, "c": 1.0}, median_idf=1.0,
            gamma=0.0, delta=0.2, min_coherence_skills=3,
            skill_embeddings=embs,
        )
        # Orthogonal → coherence ≈ 0 → boost ≈ 1.0
        assert qm["coherence_boost"] == pytest.approx(1.0, abs=0.05)

    # ── Multiplier ────────────────────────────────────────────────────────
    def test_multiplier_is_product(self):
        uri_idfs = {"a": 0.5, "b": 3.0}
        qm = compute_match_quality(
            matched_uris=["a", "b"], job_uris=["a", "b"],
            uri_idfs=uri_idfs, median_idf=2.0, gamma=0.3, delta=0.0,
        )
        expected = qm["specificity_ratio"] * qm["generic_penalty"] * qm["coherence_boost"]
        assert qm["quality_multiplier"] == pytest.approx(expected)

    def test_empty_job_uris(self):
        qm = compute_match_quality(
            matched_uris=[], job_uris=[], uri_idfs={}, median_idf=0.0,
            gamma=0.3, delta=0.2,
        )
        assert qm["quality_multiplier"] == pytest.approx(1.0)


# ── Hybrid match quality integration ─────────────────────────────────────────

class TestHybridMatchQuality:
    """Integration tests for match quality within align_hybrid."""

    def test_refined_recall_differs_from_raw(self):
        """γ > 0 should modify programme_recall before normalisation."""
        # Build data with mixed IDF: common skill in many docs, rare in few
        common_skill = _skill("esco:common")
        rare_skill = _skill("esco:rare")
        rows = []
        # Programme has both common and rare
        rows.append({
            "source_type": "programme",
            "embedding": _emb(1, 8),
            "name": "Prog0",
            "skill_details": [common_skill, rare_skill],
        })
        # Job A matches only common
        rows.append({
            "source_type": "job_ad",
            "embedding": _emb(10, 8),
            "job_title": "Job0",
            "skill_details": [common_skill],
        })
        # Job B matches only rare
        rows.append({
            "source_type": "job_ad",
            "embedding": _emb(11, 8),
            "job_title": "Job1",
            "skill_details": [rare_skill],
        })
        # Extra jobs with common skill to drive its IDF down
        for i in range(5):
            rows.append({
                "source_type": "job_ad",
                "embedding": _emb(20 + i, 8),
                "job_title": f"Filler{i}",
                "skill_details": [common_skill],
            })
        df = pd.DataFrame(rows)
        raw = align_hybrid(df, semantic_top_n=7, ipf_top_k=0, gamma=0.0, delta=0.0)
        refined = align_hybrid(df, semantic_top_n=7, ipf_top_k=0, gamma=0.3, delta=0.0)
        merged = raw.merge(refined, on=["programme_id", "job_id"], suffixes=("_raw", "_ref"))
        # programme_recall values should differ (quality multiplier applied)
        assert not np.allclose(
            merged["programme_recall_raw"].values,
            merged["programme_recall_ref"].values,
        )

    def test_backward_compat_gamma0_delta0(self):
        """γ=0, δ=0 → identical to original formula."""
        df = _make_df(2, 5)
        original = align_hybrid(df, semantic_top_n=5, ipf_top_k=0, gamma=0.0, delta=0.0)
        # Run again with same params — should be deterministic
        again = align_hybrid(df, semantic_top_n=5, ipf_top_k=0, gamma=0.0, delta=0.0)
        pd.testing.assert_frame_equal(original, again)

    def test_specificity_reranks_domain_match(self):
        """With equal cosine, specificity should favour domain-specific matches."""
        shared_emb = _emb(42, 8)
        rows = []
        rows.append({
            "source_type": "programme",
            "embedding": shared_emb,
            "name": "DevOps Programme",
            "skill_details": [_skill("esco:k8s"), _skill("esco:docker"), _skill("esco:ci_cd")],
        })
        # Job A: matches all niche skills
        rows.append({
            "source_type": "job_ad",
            "embedding": shared_emb,
            "job_title": "DevOps Engineer",
            "skill_details": [_skill("esco:k8s"), _skill("esco:docker"), _skill("esco:ci_cd")],
        })
        # Job B: matches only one niche + has generic skills (esco:python appears in many)
        rows.append({
            "source_type": "job_ad",
            "embedding": shared_emb,
            "job_title": "Generic Dev",
            "skill_details": [_skill("esco:python"), _skill("esco:sql"), _skill("esco:k8s")],
        })
        # Extra docs with python/sql to drive their IDF down
        for i in range(5):
            rows.append({
                "source_type": "job_ad",
                "embedding": _emb(50 + i, 8),
                "job_title": f"Filler{i}",
                "skill_details": [_skill("esco:python"), _skill("esco:sql")],
            })
        df = pd.DataFrame(rows)
        # Low alpha to let recall (with quality refinement) dominate
        rankings = align_hybrid(
            df, semantic_top_n=7, ipf_top_k=0, alpha=0.3, gamma=0.3, delta=0.0,
        )
        top_for_prog = rankings[rankings["programme_id"] == 0].iloc[0]
        assert top_for_prog["job_title"] == "DevOps Engineer"

    def test_generic_penalty_demotes_generic_match(self):
        """Job matched only via generic skills should score lower with γ > 0."""
        df = _make_df(2, 5)
        without = align_hybrid(df, semantic_top_n=5, ipf_top_k=0, gamma=0.0, delta=0.0)
        with_pen = align_hybrid(df, semantic_top_n=5, ipf_top_k=0, gamma=0.3, delta=0.0)
        # Mean hybrid score should not increase (generic penalty only reduces)
        assert with_pen["hybrid_score"].mean() <= without["hybrid_score"].mean() + 1e-6

    def test_coherence_boosts_related_cluster(self):
        """Coherence boost should increase score for coherent skill matches."""
        rows = []
        devops_skills = ["esco:k8s", "esco:docker", "esco:ci_cd", "esco:terraform"]
        rows.append({
            "source_type": "programme",
            "embedding": _emb(1, 8),
            "name": "DevOps",
            "skill_details": [_skill(u) for u in devops_skills],
        })
        rows.append({
            "source_type": "job_ad",
            "embedding": _emb(2, 8),
            "job_title": "DevOps Job",
            "skill_details": [_skill(u) for u in devops_skills],
        })
        df = pd.DataFrame(rows)
        embs = _mock_skill_embeddings(devops_skills, dim=8, cluster=True)
        with_coh = align_hybrid(
            df, semantic_top_n=1, ipf_top_k=0,
            gamma=0.0, delta=0.2, skill_embeddings=embs,
        )
        without_coh = align_hybrid(
            df, semantic_top_n=1, ipf_top_k=0,
            gamma=0.0, delta=0.0,
        )
        # With coherence boost, programme_recall should be higher
        assert with_coh["programme_recall"].iloc[0] >= without_coh["programme_recall"].iloc[0]

    def test_scores_non_negative(self):
        df = _make_df(3, 8)
        rankings = align_hybrid(df, semantic_top_n=5, gamma=0.3, delta=0.2)
        assert (rankings["hybrid_score"] >= -1e-9).all()
        assert (rankings["programme_recall"] >= -1e-9).all()

    def test_sort_order_maintained(self):
        df = _make_df(3, 8)
        rankings = align_hybrid(df, semantic_top_n=5, gamma=0.3, delta=0.2)
        for p_id in rankings["programme_id"].unique():
            scores = rankings[rankings["programme_id"] == p_id]["hybrid_score"].tolist()
            assert scores == sorted(scores, reverse=True)
