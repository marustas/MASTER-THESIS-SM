"""
Tests for src/alignment/cross_encoder.py and the cross-encoder re-rank
stage in src/alignment/hybrid.py.

All tests use ``MockCrossEncoder`` from tests/conftest.py — fully offline,
no HuggingFace download required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.alignment.cross_encoder import (
    DEFAULT_MODEL,
    chunk_text_by_tokens,
    load_cross_encoder,
    score_pairs,
    score_pairs_chunked,
    score_pairs_sectioned,
    score_pairs_sectioned_chunked,
)
from src.alignment.hybrid import align_hybrid


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


def _make_df(n_prog: int, n_jobs: int, dim: int = 8) -> pd.DataFrame:
    """Build a minimal dataset including ``cleaned_text`` for the re-ranker."""
    skill_pool = [
        [_skill("esco:python"), _skill("esco:ml", explicit=False)],
        [_skill("esco:java")],
        [_skill("esco:docker"), _skill("esco:kubernetes")],
        [_skill("esco:sql"), _skill("esco:data")],
    ]
    text_pool_p = [
        "python machine learning data analysis",
        "java software development",
        "docker kubernetes container orchestration",
        "sql database engineering",
    ]
    text_pool_j = [
        "java backend development with sql",
        "python data science nlp",
        "kubernetes docker cloud devops",
        "machine learning deep learning python",
    ]
    rows = []
    for i in range(n_prog):
        rows.append({
            "source_type": "programme",
            "embedding": _emb(i, dim),
            "name": f"Prog{i}",
            "cleaned_text": text_pool_p[i % len(text_pool_p)],
            "skill_details": skill_pool[i % len(skill_pool)],
        })
    for i in range(n_jobs):
        rows.append({
            "source_type": "job_ad",
            "embedding": _emb(i + 50, dim),
            "job_title": f"Job{i}",
            "cleaned_text": text_pool_j[i % len(text_pool_j)],
            "skill_details": skill_pool[(i + 1) % len(skill_pool)],
        })
    return pd.DataFrame(rows)


# ── load_cross_encoder ────────────────────────────────────────────────────────

class TestLoadCrossEncoder:
    def test_returns_object_with_predict_unchanged(self, mock_cross_encoder):
        loaded = load_cross_encoder(mock_cross_encoder)
        assert loaded is mock_cross_encoder

    def test_default_model_constant(self):
        assert DEFAULT_MODEL.startswith("cross-encoder/")

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            load_cross_encoder(42)


# ── score_pairs ───────────────────────────────────────────────────────────────

class TestScorePairs:
    def test_returns_one_score_per_pair(self, mock_cross_encoder):
        pairs = [
            ("python machine learning", "python data science"),
            ("java", "kubernetes docker"),
            ("docker", "docker kubernetes"),
        ]
        scores = score_pairs(mock_cross_encoder, pairs)
        assert scores.shape == (3,)
        assert scores.dtype == np.float32

    def test_empty_input_returns_empty(self, mock_cross_encoder):
        scores = score_pairs(mock_cross_encoder, [])
        assert scores.shape == (0,)
        assert scores.dtype == np.float32

    def test_higher_overlap_scores_higher(self, mock_cross_encoder):
        identical = [("python machine learning", "python machine learning")]
        disjoint = [("python", "kubernetes docker")]
        s_id = score_pairs(mock_cross_encoder, identical)[0]
        s_dj = score_pairs(mock_cross_encoder, disjoint)[0]
        assert s_id > s_dj

    def test_empty_strings_get_neg_inf(self, mock_cross_encoder):
        scores = score_pairs(
            mock_cross_encoder,
            [("python", ""), ("", ""), ("python", "python")],
        )
        assert np.isneginf(scores[0])
        assert np.isneginf(scores[1])
        assert np.isfinite(scores[2])

    def test_batch_size_does_not_change_results(self, mock_cross_encoder):
        pairs = [
            ("python", "python data"),
            ("java", "kubernetes"),
            ("docker", "docker container"),
            ("sql", "data analysis"),
            ("nlp", "natural language processing"),
        ]
        s1 = score_pairs(mock_cross_encoder, pairs, batch_size=1)
        s2 = score_pairs(mock_cross_encoder, pairs, batch_size=64)
        np.testing.assert_array_almost_equal(s1, s2)


# ── align_hybrid + cross-encoder integration ─────────────────────────────────

class TestHybridWithCrossEncoder:
    def test_xe_alpha_zero_no_xe_score_column(self, mock_cross_encoder):
        df = _make_df(2, 4)
        rankings = align_hybrid(df, semantic_top_n=3, xe_alpha=0.0)
        assert "xe_score" not in rankings.columns

    def test_xe_alpha_positive_adds_xe_score(self, mock_cross_encoder):
        df = _make_df(2, 4)
        rankings = align_hybrid(
            df, semantic_top_n=3,
            cross_encoder_model=mock_cross_encoder, xe_alpha=0.5,
            alpha=0.0, ipf_top_k=0,
        )
        assert "xe_score" in rankings.columns
        assert rankings["xe_score"].notna().all()

    def test_xe_alpha_without_model_raises(self):
        df = _make_df(2, 3)
        with pytest.raises(ValueError, match="cross_encoder_model"):
            align_hybrid(df, alpha=0.0, xe_alpha=0.5)

    def test_alpha_plus_xe_alpha_over_one_raises(self, mock_cross_encoder):
        df = _make_df(2, 3)
        with pytest.raises(ValueError, match="alpha \\+ xe_alpha"):
            align_hybrid(
                df, alpha=0.7, xe_alpha=0.5,
                cross_encoder_model=mock_cross_encoder,
            )

    def test_invalid_xe_alpha_raises(self, mock_cross_encoder):
        df = _make_df(2, 3)
        with pytest.raises(ValueError, match="xe_alpha"):
            align_hybrid(
                df, xe_alpha=1.5, cross_encoder_model=mock_cross_encoder,
            )

    def test_variant_1_replace_cosine(self, mock_cross_encoder):
        """alpha=0, xe_alpha=0.55 → cos channel is gone, xe + recall remain."""
        df = _make_df(3, 6)
        rankings = align_hybrid(
            df, semantic_top_n=5,
            alpha=0.0, xe_alpha=0.55,
            cross_encoder_model=mock_cross_encoder,
            ipf_top_k=0, norm_confidence=False,
        )
        assert (rankings["hybrid_score"] >= -1e-6).all()
        assert (rankings["hybrid_score"] <= 1.0 + 1e-6).all()

    def test_variant_2_three_channel_blend(self, mock_cross_encoder):
        """alpha + xe_alpha < 1 → all three channels contribute."""
        df = _make_df(3, 6)
        rankings = align_hybrid(
            df, semantic_top_n=5,
            alpha=0.3, xe_alpha=0.3,
            cross_encoder_model=mock_cross_encoder,
            ipf_top_k=0, norm_confidence=False,
        )
        # hybrid = 0.3·cos + 0.3·xe + 0.4·recall — all per-programme min-max
        # Verify formula by recomputing
        for p_id in rankings["programme_id"].unique():
            grp = rankings[rankings["programme_id"] == p_id].copy()
            for col, w in [("cosine_score", 0.3), ("xe_score", 0.3),
                           ("programme_recall", 0.4)]:
                lo, hi = grp[col].min(), grp[col].max()
                grp[f"_{col}_n"] = (
                    (grp[col] - lo) / (hi - lo) if hi > lo
                    else pd.Series(0.0, index=grp.index)
                )
            expected = (
                0.3 * grp["_cosine_score_n"]
                + 0.3 * grp["_xe_score_n"]
                + 0.4 * grp["_programme_recall_n"]
            )
            pd.testing.assert_series_equal(
                grp["hybrid_score"].reset_index(drop=True).round(6),
                expected.reset_index(drop=True).round(6),
                check_names=False,
            )

    def test_xe_alpha_one_hybrid_equals_normalised_xe(self, mock_cross_encoder):
        df = _make_df(2, 4)
        rankings = align_hybrid(
            df, semantic_top_n=4,
            alpha=0.0, xe_alpha=1.0,
            cross_encoder_model=mock_cross_encoder,
            ipf_top_k=0, norm_confidence=False,
        )
        for p_id in rankings["programme_id"].unique():
            grp = rankings[rankings["programme_id"] == p_id]
            if len(grp) > 1 and grp["xe_score"].max() > grp["xe_score"].min():
                assert grp["hybrid_score"].iloc[0] == pytest.approx(1.0, abs=1e-6)

    def test_invalid_xe_pool_mode_raises(self, mock_cross_encoder):
        df = _make_df(2, 3)
        with pytest.raises(ValueError, match="xe_pool_mode"):
            align_hybrid(
                df, alpha=0.0, xe_alpha=0.5,
                cross_encoder_model=mock_cross_encoder,
                xe_pool_mode="bogus",
            )

    def test_section_weighted_mode_runs(self, mock_cross_encoder):
        """Programme texts include a 'subjects:' header so the parser finds sections."""
        df = _make_df(2, 4)
        df.loc[df["source_type"] == "programme", "cleaned_text"] = (
            "subjects:\npython data analysis\noutcomes:\nmachine learning"
        )
        rankings = align_hybrid(
            df, semantic_top_n=3,
            alpha=0.0, xe_alpha=0.5,
            cross_encoder_model=mock_cross_encoder,
            xe_pool_mode="section_weighted",
            ipf_top_k=0, norm_confidence=False,
        )
        assert "xe_score" in rankings.columns
        assert rankings["xe_score"].notna().all()


# ── score_pairs_sectioned ────────────────────────────────────────────────────

class TestScorePairsSectioned:
    @staticmethod
    def _trivial_parser(text: str) -> dict[str, str]:
        # Split on lines starting with "@" — '@subjects line1' goes to 'subjects'
        groups: dict[str, list[str]] = {"subjects": [], "outcomes": [], "_remainder": []}
        cur = "_remainder"
        for line in text.split("\n"):
            if line.startswith("@"):
                cur = line[1:].strip() if line[1:].strip() in groups else "_remainder"
            else:
                groups[cur].append(line)
        return {g: "\n".join(v).strip() for g, v in groups.items()}

    _WEIGHTS = {"subjects": 0.7, "outcomes": 0.3, "_remainder": 0.0}

    def test_weighted_mean_combines_sections(self, mock_cross_encoder):
        pairs = [(
            "@subjects\npython machine learning\n@outcomes\ndata analysis",
            "python data analysis machine learning",
        )]
        scores = score_pairs_sectioned(
            mock_cross_encoder, pairs,
            section_parser=self._trivial_parser,
            section_weights=self._WEIGHTS,
            pool="weighted_mean",
        )
        assert scores.shape == (1,)
        assert np.isfinite(scores[0])

    def test_max_pool_picks_best_section(self, mock_cross_encoder):
        pairs = [(
            "@subjects\nkubernetes docker\n@outcomes\npython python python",
            "python data science",
        )]
        max_score = score_pairs_sectioned(
            mock_cross_encoder, pairs,
            section_parser=self._trivial_parser,
            section_weights=self._WEIGHTS,
            pool="max",
        )[0]
        wm_score = score_pairs_sectioned(
            mock_cross_encoder, pairs,
            section_parser=self._trivial_parser,
            section_weights=self._WEIGHTS,
            pool="weighted_mean",
        )[0]
        # The 'outcomes' python-heavy section overlaps the job; max should
        # at least match weighted-mean and typically exceeds it.
        assert max_score >= wm_score - 1e-6

    def test_empty_programme_returns_neg_inf(self, mock_cross_encoder):
        scores = score_pairs_sectioned(
            mock_cross_encoder, [("", "python data")],
            section_parser=self._trivial_parser,
            section_weights=self._WEIGHTS,
        )
        assert np.isneginf(scores[0])

    def test_invalid_pool_raises(self, mock_cross_encoder):
        with pytest.raises(ValueError, match="pool"):
            score_pairs_sectioned(
                mock_cross_encoder, [("a", "b")],
                section_parser=self._trivial_parser,
                section_weights=self._WEIGHTS,
                pool="bogus",
            )

    def test_empty_input_returns_empty(self, mock_cross_encoder):
        scores = score_pairs_sectioned(
            mock_cross_encoder, [],
            section_parser=self._trivial_parser,
            section_weights=self._WEIGHTS,
        )
        assert scores.shape == (0,)


# ── chunk_text_by_tokens ─────────────────────────────────────────────────────

class TestChunkTextByTokens:
    def test_no_tokenizer_returns_single_chunk(self):
        chunks = chunk_text_by_tokens("python data analysis", tokenizer=None)
        assert chunks == ["python data analysis"]

    def test_empty_text_returns_empty_list(self):
        assert chunk_text_by_tokens("", tokenizer=None) == []
        assert chunk_text_by_tokens("   ", tokenizer=None) == []

    def test_short_text_returns_single_chunk(self):
        class FakeTok:
            def encode(self, text, add_special_tokens=False):
                return text.split()  # 1 word = 1 token
            def decode(self, ids, skip_special_tokens=True):
                return " ".join(ids)
        chunks = chunk_text_by_tokens("a b c", FakeTok(), max_tokens=8)
        assert chunks == ["a b c"]

    def test_long_text_is_split(self):
        class FakeTok:
            def encode(self, text, add_special_tokens=False):
                return text.split()
            def decode(self, ids, skip_special_tokens=True):
                return " ".join(ids)
        chunks = chunk_text_by_tokens(
            "one two three four five six", FakeTok(), max_tokens=2,
        )
        assert chunks == ["one two", "three four", "five six"]


# ── score_pairs_chunked (job-side only) ──────────────────────────────────────

class TestScorePairsChunked:
    def test_max_pool_runs(self, mock_cross_encoder):
        # MockCrossEncoder has no tokenizer → falls back to single chunk
        pairs = [
            ("python data", "python data analysis machine learning"),
            ("java", "kubernetes docker container orchestration"),
        ]
        scores = score_pairs_chunked(
            mock_cross_encoder, pairs, pool="max",
        )
        assert scores.shape == (2,)
        assert np.isfinite(scores).all()

    def test_mean_pool_runs(self, mock_cross_encoder):
        pairs = [("python", "python python python")]
        scores = score_pairs_chunked(
            mock_cross_encoder, pairs, pool="mean",
        )
        assert scores.shape == (1,)
        assert np.isfinite(scores[0])

    def test_invalid_pool_raises(self, mock_cross_encoder):
        with pytest.raises(ValueError, match="pool"):
            score_pairs_chunked(
                mock_cross_encoder, [("a", "b")], pool="bogus",
            )

    def test_empty_input_returns_empty(self, mock_cross_encoder):
        assert score_pairs_chunked(mock_cross_encoder, []).shape == (0,)

    def test_empty_strings_get_neg_inf(self, mock_cross_encoder):
        scores = score_pairs_chunked(
            mock_cross_encoder, [("python", ""), ("python", "python")],
        )
        assert np.isneginf(scores[0])
        assert np.isfinite(scores[1])


# ── score_pairs_sectioned_chunked (two-sided) ────────────────────────────────

class TestScorePairsSectionedChunked:
    @staticmethod
    def _parser(text: str) -> dict[str, str]:
        groups = {"subjects": [], "outcomes": [], "_remainder": []}
        cur = "_remainder"
        for line in text.split("\n"):
            if line.startswith("@"):
                cur = line[1:].strip() if line[1:].strip() in groups else "_remainder"
            else:
                groups[cur].append(line)
        return {g: "\n".join(v).strip() for g, v in groups.items()}

    _WEIGHTS = {"subjects": 0.7, "outcomes": 0.3, "_remainder": 0.0}

    def test_runs_with_mock(self, mock_cross_encoder):
        pairs = [(
            "@subjects\npython machine learning\n@outcomes\ndata analysis",
            "python data analysis machine learning",
        )]
        scores = score_pairs_sectioned_chunked(
            mock_cross_encoder, pairs,
            section_parser=self._parser,
            section_weights=self._WEIGHTS,
        )
        assert scores.shape == (1,)
        assert np.isfinite(scores[0])

    def test_invalid_prog_pool_raises(self, mock_cross_encoder):
        with pytest.raises(ValueError, match="prog_pool"):
            score_pairs_sectioned_chunked(
                mock_cross_encoder, [("a", "b")],
                section_parser=self._parser,
                section_weights=self._WEIGHTS,
                prog_pool="bogus",
            )

    def test_invalid_job_pool_raises(self, mock_cross_encoder):
        with pytest.raises(ValueError, match="job_pool"):
            score_pairs_sectioned_chunked(
                mock_cross_encoder, [("a", "b")],
                section_parser=self._parser,
                section_weights=self._WEIGHTS,
                job_pool="bogus",
            )

    def test_empty_input_returns_empty(self, mock_cross_encoder):
        scores = score_pairs_sectioned_chunked(
            mock_cross_encoder, [],
            section_parser=self._parser,
            section_weights=self._WEIGHTS,
        )
        assert scores.shape == (0,)
