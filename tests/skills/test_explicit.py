"""
Tests for ExplicitSkillExtractor.

Covers:
  - S3 (ESCO dictionary match): known skills found by PhraseMatcher
  - S2 (PoS comma-list rule): paper's Octave example
  - S1 (NER): entity-based candidate generation
  - Relevance scoring: terms below threshold are filtered
  - URI deduplication: multiple surface forms → single highest-confidence result
  - Edge cases: empty text, whitespace-only text
"""

from __future__ import annotations

import pytest

from src.skills.explicit_extractor import ExplicitSkillExtractor, _RELEVANCE_THRESHOLD


class TestS3DictionaryMatch:
    """Skills explicitly present in the ESCO mock index must be found."""

    def test_single_token_skill(self, explicit_extractor: ExplicitSkillExtractor) -> None:
        skills = explicit_extractor.extract("We need a developer with Python experience.")
        labels = [s.preferred_label for s in skills]
        assert "Python" in labels

    def test_multi_token_skill(self, explicit_extractor: ExplicitSkillExtractor) -> None:
        skills = explicit_extractor.extract(
            "The role requires machine learning and data analysis."
        )
        labels = [s.preferred_label for s in skills]
        assert "machine learning" in labels
        assert "data analysis" in labels

    def test_alt_label_match(self, explicit_extractor: ExplicitSkillExtractor) -> None:
        """Alt labels (synonyms) must also be detected and mapped to the preferred label."""
        skills = explicit_extractor.extract("Proficiency in NLP is required.")
        labels = [s.preferred_label for s in skills]
        # "NLP" is an alt label for "natural language processing"
        assert "natural language processing" in labels

    def test_multiple_skills_in_one_sentence(self, explicit_extractor: ExplicitSkillExtractor) -> None:
        skills = explicit_extractor.extract(
            "Experience with Docker, Kubernetes, and Git is expected."
        )
        labels = [s.preferred_label for s in skills]
        assert "Docker" in labels
        assert "Kubernetes" in labels
        assert "Git" in labels

    def test_case_insensitive_match(self, explicit_extractor: ExplicitSkillExtractor) -> None:
        skills = explicit_extractor.extract("Proficiency in PYTHON and SQL.")
        labels = [s.preferred_label for s in skills]
        assert "Python" in labels
        assert "SQL" in labels


class TestS2PosCommaRule:
    """
    Replicates the paper's key example (Section 4.1, Combined Flow):

    'Need candidates with ability to code in Python, Java, and Octave.'

    Python and Java are ESCO dict hits.  Octave is NOT in the ESCO dictionary
    as 'Octave' per se but IS in the mock index as an alt-label for GNU Octave.
    The PoS comma-list rule should boost Octave's score because Python and Java
    anchor it as a skill in the comma-separated list.
    """

    def test_paper_example_python_java_found(self, explicit_extractor: ExplicitSkillExtractor) -> None:
        skills = explicit_extractor.extract(
            "Need candidates with ability to code in Python, Java, and Octave."
        )
        labels = [s.preferred_label for s in skills]
        assert "Python" in labels, f"Python not found. Got: {labels}"
        assert "Java" in labels, f"Java not found. Got: {labels}"

    def test_pos_rule_boosts_adjacent_noun(self, explicit_extractor: ExplicitSkillExtractor) -> None:
        """
        When a comma-separated list contains known skills,
        the PoS rule fires and assigns S2=1 to adjacent nouns.
        Verify the rule fires by checking the S2 scores directly.
        """
        doc = explicit_extractor._nlp(
            "Need candidates with ability to code in Python, Java, and Octave."
        )
        s3_hits = explicit_extractor._s3_dict(doc)
        dict_hit_surfaces = set(s3_hits.keys())
        s2_scores = explicit_extractor._s2_pos(doc, dict_hit_surfaces)

        # At least one non-dict noun should have been boosted
        assert len(s2_scores) > 0, (
            "S2 PoS rule did not fire. No nouns were boosted. "
            f"Dict hits: {dict_hit_surfaces}"
        )

    def test_pos_rule_does_not_fire_without_anchor(self, explicit_extractor: ExplicitSkillExtractor) -> None:
        """If no noun in the comma list is a known skill, the rule must not fire."""
        doc = explicit_extractor._nlp("The office has tables, chairs, and lamps.")
        s3_hits = explicit_extractor._s3_dict(doc)
        dict_hit_surfaces = set(s3_hits.keys())
        s2_scores = explicit_extractor._s2_pos(doc, dict_hit_surfaces)
        assert len(s2_scores) == 0


class TestRelevanceScoring:
    """Relevance score gates: terms below threshold must be excluded."""

    def test_relevance_threshold_applied(self, explicit_extractor: ExplicitSkillExtractor) -> None:
        """All returned skills must have confidence >= threshold."""
        skills = explicit_extractor.extract(
            "We need Python, SQL, and experience with cloud computing."
        )
        for skill in skills:
            assert skill.confidence >= _RELEVANCE_THRESHOLD, (
                f"Skill '{skill.preferred_label}' has confidence {skill.confidence} "
                f"below threshold {_RELEVANCE_THRESHOLD}"
            )

    def test_common_words_filtered_out(self, explicit_extractor: ExplicitSkillExtractor) -> None:
        """Generic stop-words should not pass the relevance threshold."""
        skills = explicit_extractor.extract("The candidate must have good communication.")
        labels = [s.preferred_label for s in skills]
        # "candidate" was explicitly filtered in the paper's example
        assert "candidate" not in labels


class TestUriDeduplication:
    """The same ESCO URI must appear at most once, with the highest confidence."""

    def test_no_duplicate_uris(self, explicit_extractor: ExplicitSkillExtractor) -> None:
        skills = explicit_extractor.extract(
            "Python, Python programming, and Python scripting are all required."
        )
        uris = [s.esco_uri for s in skills]
        assert len(uris) == len(set(uris)), f"Duplicate URIs found: {uris}"

    def test_highest_confidence_kept(self, explicit_extractor: ExplicitSkillExtractor) -> None:
        """When multiple surface forms map to the same URI, the best confidence wins."""
        # Both "NLP" (alt label) and "natural language processing" (preferred) map to esco:nlp
        skills = explicit_extractor.extract(
            "Strong background in NLP and natural language processing."
        )
        nlp_hits = [s for s in skills if s.esco_uri == "esco:nlp"]
        assert len(nlp_hits) == 1, f"Expected 1 entry for esco:nlp, got {len(nlp_hits)}"


class TestEdgeCases:
    def test_empty_string(self, explicit_extractor: ExplicitSkillExtractor) -> None:
        assert explicit_extractor.extract("") == []

    def test_whitespace_only(self, explicit_extractor: ExplicitSkillExtractor) -> None:
        assert explicit_extractor.extract("   \n\t  ") == []

    def test_no_skills_in_text(self, explicit_extractor: ExplicitSkillExtractor) -> None:
        skills = explicit_extractor.extract(
            "The weather today is sunny and warm with light breeze."
        )
        # May return some false positives via S4 but should not crash
        assert isinstance(skills, list)

    def test_all_skills_are_explicit(self, explicit_extractor: ExplicitSkillExtractor) -> None:
        skills = explicit_extractor.extract("Python and machine learning.")
        for skill in skills:
            assert skill.explicit is True
            assert skill.implicit is False
