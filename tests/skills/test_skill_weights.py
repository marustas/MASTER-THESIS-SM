"""
Tests for src/skills/skill_weights.py.

Covers:
  - tier_weight: all known levels, unknown, None
  - compute_corpus_idf: empty, single doc, multi-doc, dedup within doc
  - build_weighted_skills: tier × IDF × explicit/implicit, dedup by max, empty
"""

from __future__ import annotations

import math

import pytest

from src.skills.skill_weights import (
    DEFAULT_TIER_WEIGHT,
    REUSE_TIER_WEIGHTS,
    build_weighted_skills,
    compute_corpus_idf,
    tier_weight,
)


# ── tier_weight ───────────────────────────────────────────────────────────────

class TestTierWeight:
    def test_transversal(self):
        assert tier_weight("transversal") == 0.3

    def test_cross_sector(self):
        assert tier_weight("cross-sector") == 0.5

    def test_sector_specific(self):
        assert tier_weight("sector-specific") == 0.8

    def test_occupation_specific(self):
        assert tier_weight("occupation-specific") == 1.0

    def test_none_returns_default(self):
        assert tier_weight(None) == DEFAULT_TIER_WEIGHT

    def test_unknown_returns_default(self):
        assert tier_weight("something-else") == DEFAULT_TIER_WEIGHT

    def test_case_insensitive(self):
        assert tier_weight("Cross-Sector") == 0.5

    def test_whitespace_stripped(self):
        assert tier_weight("  transversal  ") == 0.3


# ── compute_corpus_idf ───────────────────────────────────────────────────────

class TestComputeCorpusIdf:
    def test_empty_corpus(self):
        assert compute_corpus_idf([]) == {}

    def test_single_doc_single_uri(self):
        result = compute_corpus_idf([["uri:a"]])
        # log(1 + 1/1) = log(2)
        assert result["uri:a"] == pytest.approx(math.log(2.0))

    def test_two_docs_one_shared(self):
        result = compute_corpus_idf([["uri:a", "uri:b"], ["uri:a"]])
        # uri:a appears in 2/2 docs → log(1 + 2/2) = log(2)
        assert result["uri:a"] == pytest.approx(math.log(2.0))
        # uri:b appears in 1/2 docs → log(1 + 2/1) = log(3)
        assert result["uri:b"] == pytest.approx(math.log(3.0))

    def test_duplicate_uris_within_doc_counted_once(self):
        result = compute_corpus_idf([["uri:a", "uri:a", "uri:a"]])
        assert result["uri:a"] == pytest.approx(math.log(2.0))

    def test_rare_uri_has_higher_idf(self):
        corpus = [["uri:common", "uri:rare"]] + [["uri:common"]] * 9
        result = compute_corpus_idf(corpus)
        assert result["uri:rare"] > result["uri:common"]


# ── build_weighted_skills ────────────────────────────────────────────────────

def _skill(uri: str, *, explicit: bool = True) -> dict:
    return {
        "esco_uri": uri,
        "preferred_label": uri.split(":")[-1],
        "matched_text": uri.split(":")[-1],
        "explicit": explicit,
        "implicit": not explicit,
        "confidence": 1.0,
    }


class TestBuildWeightedSkills:
    def test_empty(self):
        assert build_weighted_skills([], {}, {}) == {}

    def test_explicit_with_tier_and_idf(self):
        details = [_skill("uri:python", explicit=True)]
        reuse = {"uri:python": "occupation-specific"}
        idfs = {"uri:python": math.log(3.0)}
        result = build_weighted_skills(details, reuse, idfs)
        expected = 1.0 * math.log(3.0) * 1.0  # tier=1.0, idf=log3, explicit=1.0
        assert result["uri:python"] == pytest.approx(expected)

    def test_implicit_halves_weight(self):
        details = [_skill("uri:sql", explicit=False)]
        reuse = {"uri:sql": "cross-sector"}
        idfs = {"uri:sql": 2.0}
        result = build_weighted_skills(details, reuse, idfs)
        expected = 0.5 * 2.0 * 0.5  # tier=0.5, idf=2.0, implicit=0.5
        assert result["uri:sql"] == pytest.approx(expected)

    def test_explicit_wins_over_implicit(self):
        details = [
            _skill("uri:ml", explicit=False),
            _skill("uri:ml", explicit=True),
        ]
        reuse = {"uri:ml": "sector-specific"}
        idfs = {"uri:ml": 1.5}
        result = build_weighted_skills(details, reuse, idfs)
        w_explicit = 0.8 * 1.5 * 1.0
        w_implicit = 0.8 * 1.5 * 0.5
        assert result["uri:ml"] == pytest.approx(max(w_explicit, w_implicit))

    def test_missing_reuse_level_uses_default(self):
        details = [_skill("uri:x", explicit=True)]
        result = build_weighted_skills(details, {}, {"uri:x": 1.0})
        expected = DEFAULT_TIER_WEIGHT * 1.0 * 1.0
        assert result["uri:x"] == pytest.approx(expected)

    def test_missing_idf_uses_default(self):
        details = [_skill("uri:x", explicit=True)]
        result = build_weighted_skills(details, {"uri:x": "transversal"}, {})
        # default_idf=1.0
        expected = 0.3 * 1.0 * 1.0
        assert result["uri:x"] == pytest.approx(expected)

    def test_transversal_common_skill_gets_low_weight(self):
        """Transversal skill appearing in every doc should have low weight."""
        details = [_skill("uri:communication", explicit=True)]
        reuse = {"uri:communication": "transversal"}
        # IDF for uri in all 10 docs: log(1 + 10/10) = log(2) ≈ 0.693
        idfs = {"uri:communication": math.log(2.0)}
        result = build_weighted_skills(details, reuse, idfs)
        expected = 0.3 * math.log(2.0) * 1.0
        assert result["uri:communication"] == pytest.approx(expected)
        assert result["uri:communication"] < 0.25  # much lower than uniform 1.0

    def test_rare_specific_skill_gets_high_weight(self):
        """Occupation-specific rare skill should get high weight."""
        details = [_skill("uri:kubernetes", explicit=True)]
        reuse = {"uri:kubernetes": "occupation-specific"}
        # IDF for uri in 1 of 100 docs: log(1 + 100/1) = log(101) ≈ 4.615
        idfs = {"uri:kubernetes": math.log(101.0)}
        result = build_weighted_skills(details, reuse, idfs)
        expected = 1.0 * math.log(101.0) * 1.0
        assert result["uri:kubernetes"] == pytest.approx(expected)
        assert result["uri:kubernetes"] > 4.0  # much higher than uniform 1.0

    def test_skip_empty_uri(self):
        details = [{"esco_uri": "", "explicit": True}]
        assert build_weighted_skills(details, {}, {}) == {}
