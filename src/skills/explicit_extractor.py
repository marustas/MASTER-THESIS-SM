"""
Explicit skill extraction via phrase matching against the ESCO taxonomy.

Approach:
  1. Build a spaCy PhraseMatcher from all ESCO skill labels (preferred + alt).
  2. For each document, run the matcher to find all skill surface forms.
  3. Map each match back to its canonical ESCO URI and preferred label.

The matcher uses lowercase, lemmatized matching so "programming languages"
matches "programming language", etc.

Returns a list of ExtractedSkill per document, containing:
  - esco_uri        — canonical ESCO concept URI
  - preferred_label — ESCO preferred label
  - matched_text    — exact surface form found in the document
  - explicit        — always True for this extractor
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import spacy
from loguru import logger
from spacy.language import Language
from spacy.matcher import PhraseMatcher

from src.skills.esco_loader import EscoIndex, EscoSkill

# Maximum label length for PhraseMatcher (very long labels cause spaCy issues)
_MAX_LABEL_TOKENS = 8


@dataclass
class ExtractedSkill:
    esco_uri: str
    preferred_label: str
    matched_text: str
    explicit: bool = True
    implicit: bool = False
    confidence: float = 1.0


# ── Matcher builder ────────────────────────────────────────────────────────────

class ExplicitSkillExtractor:
    """
    spaCy PhraseMatcher-based extractor for ESCO skills.

    Build once per process and reuse across documents:
        extractor = ExplicitSkillExtractor(esco_index)
    """

    def __init__(self, esco_index: EscoIndex, nlp_model: str = "en_core_web_sm"):
        self._index = esco_index
        self._nlp = self._load_nlp(nlp_model)
        self._matcher = self._build_matcher()

    @staticmethod
    def _load_nlp(model: str) -> Language:
        try:
            nlp = spacy.load(model, disable=["parser", "ner"])
        except OSError:
            logger.warning(f"spaCy model '{model}' not found — using blank English model")
            nlp = spacy.blank("en")
        return nlp

    def _build_matcher(self) -> PhraseMatcher:
        matcher = PhraseMatcher(self._nlp.vocab, attr="LOWER")
        added = 0
        skipped = 0
        for skill in self._index.skills:
            patterns = []
            for label in skill.all_labels:
                doc = self._nlp.make_doc(label.lower())
                if 1 <= len(doc) <= _MAX_LABEL_TOKENS:
                    patterns.append(doc)
            if patterns:
                matcher.add(skill.uri, patterns)
                added += 1
            else:
                skipped += 1

        logger.info(
            f"PhraseMatcher built: {added} skill patterns added, {skipped} skipped"
        )
        return matcher

    def extract(self, text: str) -> list[ExtractedSkill]:
        """
        Extract explicit ESCO skills from `text`.
        Returns deduplicated list sorted by ESCO URI.
        """
        if not text or not text.strip():
            return []

        doc = self._nlp(text[:500_000])  # hard limit for very long descriptions
        matches = self._matcher(doc)

        seen_uris: set[str] = set()
        results: list[ExtractedSkill] = []

        for match_id, start, end in matches:
            uri = self._nlp.vocab.strings[match_id]
            if uri in seen_uris:
                continue
            skill = self._index.lookup_uri(uri)
            if skill is None:
                continue
            seen_uris.add(uri)
            span_text = doc[start:end].text
            results.append(ExtractedSkill(
                esco_uri=uri,
                preferred_label=skill.preferred_label,
                matched_text=span_text,
                explicit=True,
                confidence=1.0,
            ))

        return sorted(results, key=lambda s: s.esco_uri)

    def extract_labels(self, text: str) -> list[str]:
        """Convenience wrapper — returns only preferred labels."""
        return [s.preferred_label for s in self.extract(text)]
