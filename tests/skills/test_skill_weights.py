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
    DEFAULT_IDF_CAP,
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
    """Default mode: IDF-only with cap=3.0, no tier weighting."""

    def test_empty(self):
        assert build_weighted_skills([], {}, {}) == {}

    def test_explicit_idf_only(self):
        """Default (no tiers): weight = min(idf, cap) × 1.0."""
        details = [_skill("uri:python", explicit=True)]
        idfs = {"uri:python": 2.5}
        result = build_weighted_skills(details, {}, idfs)
        assert result["uri:python"] == pytest.approx(2.5 * 1.0)

    def test_implicit_halves_weight(self):
        details = [_skill("uri:sql", explicit=False)]
        idfs = {"uri:sql": 2.0}
        result = build_weighted_skills(details, {}, idfs)
        assert result["uri:sql"] == pytest.approx(2.0 * 0.5)

    def test_idf_cap_applied(self):
        """IDF above cap is clamped to cap (default 3.0)."""
        details = [_skill("uri:rare", explicit=True)]
        idfs = {"uri:rare": 5.5}
        result = build_weighted_skills(details, {}, idfs)
        assert result["uri:rare"] == pytest.approx(DEFAULT_IDF_CAP * 1.0)

    def test_idf_cap_none_disables_capping(self):
        details = [_skill("uri:rare", explicit=True)]
        idfs = {"uri:rare": 5.5}
        result = build_weighted_skills(details, {}, idfs, idf_cap=None)
        assert result["uri:rare"] == pytest.approx(5.5 * 1.0)

    def test_explicit_wins_over_implicit(self):
        details = [
            _skill("uri:ml", explicit=False),
            _skill("uri:ml", explicit=True),
        ]
        idfs = {"uri:ml": 1.5}
        result = build_weighted_skills(details, {}, idfs)
        assert result["uri:ml"] == pytest.approx(1.5 * 1.0)

    def test_missing_idf_uses_default(self):
        details = [_skill("uri:x", explicit=True)]
        result = build_weighted_skills(details, {}, {})
        # default_idf=1.0, no tier, cap doesn't bite
        assert result["uri:x"] == pytest.approx(1.0)

    def test_tiers_ignored_by_default(self):
        """Reuse levels have no effect when use_tiers=False (default)."""
        details = [_skill("uri:x", explicit=True)]
        reuse = {"uri:x": "transversal"}
        idfs = {"uri:x": 2.0}
        result = build_weighted_skills(details, reuse, idfs)
        # tier ignored → weight = min(2.0, 3.0) × 1.0 = 2.0
        assert result["uri:x"] == pytest.approx(2.0)

    def test_skip_empty_uri(self):
        details = [{"esco_uri": "", "explicit": True}]
        assert build_weighted_skills(details, {}, {}) == {}

    def test_common_skill_lower_than_rare(self):
        """Common skill (low IDF) gets lower weight than rare skill."""
        details = [
            _skill("uri:common", explicit=True),
            _skill("uri:rare", explicit=True),
        ]
        # common in 100/100 docs, rare in 1/100 docs
        idfs = {"uri:common": math.log(2.0), "uri:rare": math.log(101.0)}
        result = build_weighted_skills(details, {}, idfs)
        # rare is capped at 3.0
        assert result["uri:common"] == pytest.approx(math.log(2.0))
        assert result["uri:rare"] == pytest.approx(DEFAULT_IDF_CAP)
        assert result["uri:rare"] > result["uri:common"]


class TestBuildWeightedSkillsWithTiers:
    """Tier mode enabled: weight = tier × min(idf, cap) × explicit/implicit."""

    def test_explicit_with_tier_and_idf(self):
        details = [_skill("uri:python", explicit=True)]
        reuse = {"uri:python": "occupation-specific"}
        idfs = {"uri:python": math.log(3.0)}
        result = build_weighted_skills(
            details, reuse, idfs, idf_cap=None, use_tiers=True,
        )
        expected = 1.0 * math.log(3.0) * 1.0
        assert result["uri:python"] == pytest.approx(expected)

    def test_transversal_common_skill_gets_low_weight(self):
        details = [_skill("uri:communication", explicit=True)]
        reuse = {"uri:communication": "transversal"}
        idfs = {"uri:communication": math.log(2.0)}
        result = build_weighted_skills(
            details, reuse, idfs, idf_cap=None, use_tiers=True,
        )
        expected = 0.3 * math.log(2.0) * 1.0
        assert result["uri:communication"] == pytest.approx(expected)
        assert result["uri:communication"] < 0.25

    def test_missing_reuse_level_uses_default_tier(self):
        details = [_skill("uri:x", explicit=True)]
        result = build_weighted_skills(
            details, {}, {"uri:x": 1.0}, idf_cap=None, use_tiers=True,
        )
        expected = DEFAULT_TIER_WEIGHT * 1.0 * 1.0
        assert result["uri:x"] == pytest.approx(expected)

    def test_tier_and_cap_combined(self):
        details = [_skill("uri:rare", explicit=True)]
        reuse = {"uri:rare": "sector-specific"}
        idfs = {"uri:rare": 5.5}
        result = build_weighted_skills(
            details, reuse, idfs, idf_cap=3.0, use_tiers=True,
        )
        expected = 0.8 * 3.0 * 1.0
        assert result["uri:rare"] == pytest.approx(expected)
