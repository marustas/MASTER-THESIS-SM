"""
Explicit skill extraction — ensemble approach per Gugnani & Misra (2020).

The paper uses four parallel modules combined via a weighted relevance score:
  S1 — NER: entities, keywords, concepts identified by a NLU system
  S2 — PoS patterns: syntactic rules (e.g. comma-separated noun lists)
  S3 — Skill Dictionary: direct lookup against known skill terms
  S4 — Word2Vec: cosine similarity of candidate phrases to skill dictionary

Relevance score (Equation 1 in the paper):
  x = (α·S1 + β·S2 + Σγn·S3n + λ·S4) / (α + β + Σγn + λ)

Terms with x < 0.35 are dropped.

Adaptation for this thesis:
  - S3 (Skill Dictionary) uses the ESCO taxonomy instead of ONet/Hope/Wikipedia.
    ESCO is richer and EU-aligned; matching via spaCy PhraseMatcher.
  - S1 (NER) uses spaCy's en_core_web_sm NER instead of Watson NLU.
  - S2 (PoS) implements the paper's comma-separated noun-list rule.
  - S4 uses sentence-transformer embeddings instead of Word2Vec (stronger).

Weights (Table 1 in the paper, adapted):
  α (NER)              = 1
  β (PoS)              = 1
  γ (ESCO Dictionary)  = 20   (replaces ONet=20 + Hope=10 + Wikipedia=20)
  λ (Embedding sim)    = 2
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import spacy
from loguru import logger
from sentence_transformers import SentenceTransformer
from spacy.language import Language
from spacy.matcher import PhraseMatcher

from src.skills.esco_loader import EscoIndex

_MAX_LABEL_TOKENS = 8
_RELEVANCE_THRESHOLD = 0.35

# Weights (paper Table 1, adapted for ESCO)
_W_NER = 1        # α
_W_POS = 1        # β
_W_DICT = 20      # γ (ESCO replaces three dictionaries → combined weight)
_W_EMBED = 2      # λ
_W_TOTAL = _W_NER + _W_POS + _W_DICT + _W_EMBED

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@dataclass
class ExtractedSkill:
    esco_uri: str
    preferred_label: str
    matched_text: str
    explicit: bool = True
    implicit: bool = False
    confidence: float = 1.0   # relevance score [0, 1]


class ExplicitSkillExtractor:
    """
    Four-module ensemble explicit skill extractor aligned with Gugnani & Misra (2020).

    Build once per process and reuse across documents:
        extractor = ExplicitSkillExtractor(esco_index)
    """

    def __init__(
        self,
        esco_index: EscoIndex,
        nlp_model: str = "en_core_web_sm",
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self._index = esco_index
        self._nlp = self._load_nlp(nlp_model)
        self._phrase_matcher = self._build_phrase_matcher()
        self._embed_model = SentenceTransformer(embedding_model)

        # Pre-compute ESCO label embeddings for S4 scoring (normalised)
        logger.info("Pre-computing ESCO label embeddings for relevance scoring…")
        labels = [s.preferred_label for s in self._index.skills]
        self._label_embeddings: np.ndarray = self._embed_model.encode(
            labels, batch_size=256, normalize_embeddings=True, show_progress_bar=False
        )
        self._label_uris = [s.uri for s in self._index.skills]
        logger.info("ExplicitSkillExtractor ready")

    # ── spaCy + PhraseMatcher setup ────────────────────────────────────────────

    @staticmethod
    def _load_nlp(model: str) -> Language:
        try:
            # Keep NER enabled for S1; parser disabled for speed
            nlp = spacy.load(model, disable=["parser"])
        except OSError:
            logger.warning(f"spaCy model '{model}' not found — using blank model (no NER/PoS)")
            nlp = spacy.blank("en")
        return nlp

    def _build_phrase_matcher(self) -> PhraseMatcher:
        """Build ESCO PhraseMatcher for S3 (dictionary matching)."""
        matcher = PhraseMatcher(self._nlp.vocab, attr="LOWER")
        added = 0
        for skill in self._index.skills:
            patterns = []
            for label in skill.all_labels:
                doc = self._nlp.make_doc(label.lower())
                if 1 <= len(doc) <= _MAX_LABEL_TOKENS:
                    patterns.append(doc)
            if patterns:
                matcher.add(skill.uri, patterns)
                added += 1
        logger.info(f"PhraseMatcher: {added} ESCO skills indexed")
        return matcher

    # ── Module S1: NER ─────────────────────────────────────────────────────────

    @staticmethod
    def _s1_ner(doc) -> dict[str, float]:
        """
        Extract candidate terms via spaCy NER.
        Mirrors the paper's Watson NLU entity/keyword/concept extraction.
        Score = 1.0 for entity labels relevant to skills (PRODUCT, ORG, etc.),
                0.5 for other entity types.
        """
        skill_ent_labels = {"PRODUCT", "ORG", "LANGUAGE", "WORK_OF_ART", "LAW", "GPE"}
        scores: dict[str, float] = {}
        for ent in doc.ents:
            phrase = ent.text.strip().lower()
            if len(phrase) < 2:
                continue
            s = 1.0 if ent.label_ in skill_ent_labels else 0.5
            scores[phrase] = max(scores.get(phrase, 0.0), s)
        return scores

    # ── Module S2: PoS patterns ────────────────────────────────────────────────

    def _s2_pos(self, doc, dict_hits: set[str]) -> dict[str, float]:
        """
        Apply the paper's comma-separated noun-list rule:
        'If a sentence has a comma-separated list of nouns where one or more
        nouns is a known skill, then the other nouns are probably skills too.'

        Returns candidate terms with S2 score = 1.0 if the rule fires.
        """
        scores: dict[str, float] = {}
        for sent in doc.sents if doc.has_annotation("SENT_START") else [doc[:]]:
            nouns = self._comma_separated_nouns(sent)
            if len(nouns) < 2:
                continue
            # Check if at least one noun is a known ESCO skill
            anchor_found = any(n.lower() in dict_hits for n in nouns)
            if anchor_found:
                for noun in nouns:
                    key = noun.lower()
                    if key not in dict_hits:  # only score the non-explicit ones
                        scores[key] = 1.0
        return scores

    @staticmethod
    def _comma_separated_nouns(span) -> list[str]:
        """Extract nouns from a comma-separated token sequence."""
        nouns = []
        tokens = [t for t in span if not t.is_space]
        for i, token in enumerate(tokens):
            if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop:
                # Check that neighbours are commas or conjunctions
                left_ok = i == 0 or tokens[i - 1].text in {",", "and", "or", "&"}
                right_ok = i == len(tokens) - 1 or tokens[i + 1].text in {",", "and", "or", "&"}
                if left_ok or right_ok:
                    nouns.append(token.text)
        return nouns

    # ── Module S3: ESCO dictionary match ──────────────────────────────────────

    def _s3_dict(self, doc) -> dict[str, tuple[str, str]]:
        """
        Run PhraseMatcher against ESCO.
        Returns {surface_form_lower: (uri, preferred_label)}.
        Score = 1.0 for any match (binary in the paper for individual dicts).
        """
        matches = self._phrase_matcher(doc)
        hits: dict[str, tuple[str, str]] = {}
        for match_id, start, end in matches:
            uri = self._nlp.vocab.strings[match_id]
            skill = self._index.lookup_uri(uri)
            if skill:
                surface = doc[start:end].text.lower()
                hits[surface] = (uri, skill.preferred_label)
        return hits

    # ── Module S4: Embedding similarity ───────────────────────────────────────

    def _s4_embed(self, candidates: list[str]) -> dict[str, tuple[float, str, str]]:
        """
        Compute cosine similarity between each candidate phrase embedding
        and all ESCO label embeddings (W2V analogue, paper Section 4.1).
        Returns {candidate: (max_similarity, best_uri, best_label)}.
        """
        if not candidates:
            return {}
        cand_embs = self._embed_model.encode(
            candidates, normalize_embeddings=True, show_progress_bar=False
        )
        sim_matrix = cand_embs @ self._label_embeddings.T  # (n_cands, n_skills)
        result = {}
        for i, cand in enumerate(candidates):
            best_idx = int(np.argmax(sim_matrix[i]))
            result[cand] = (
                float(sim_matrix[i, best_idx]),
                self._label_uris[best_idx],
                self._index.skills[best_idx].preferred_label,
            )
        return result

    # ── Combined extraction ────────────────────────────────────────────────────

    def extract(self, text: str) -> list[ExtractedSkill]:
        """
        Run all four modules, compute relevance scores, and return skills
        with score ≥ _RELEVANCE_THRESHOLD.
        """
        if not text or not text.strip():
            return []

        doc = self._nlp(text[:500_000])

        # S3 first — dictionary hits are needed by S2
        s3_hits = self._s3_dict(doc)
        dict_hit_surfaces = set(s3_hits.keys())

        # S1 and S2 candidates
        s1_scores = self._s1_ner(doc)
        s2_scores = self._s2_pos(doc, dict_hit_surfaces)

        # Collect all unique candidate phrases
        all_candidates: set[str] = set(s1_scores) | set(s2_scores) | dict_hit_surfaces

        # S4 embedding similarity for all candidates
        s4_results = self._s4_embed(list(all_candidates))

        # ── Compute relevance score per candidate (Equation 1) ─────────────────
        results: list[ExtractedSkill] = []
        seen_uris: set[str] = set()

        for candidate in all_candidates:
            s1 = s1_scores.get(candidate, 0.0)
            s2 = s2_scores.get(candidate, 0.0)
            s3 = 1.0 if candidate in dict_hit_surfaces else 0.0
            s4_sim, best_uri, best_label = s4_results.get(candidate, (0.0, "", ""))

            relevance = (
                _W_NER * s1 + _W_POS * s2 + _W_DICT * s3 + _W_EMBED * s4_sim
            ) / _W_TOTAL

            if relevance < _RELEVANCE_THRESHOLD:
                continue

            # Resolve the best ESCO URI for this candidate
            if candidate in dict_hit_surfaces:
                uri, label = s3_hits[candidate]
            elif best_uri:
                uri, label = best_uri, best_label
            else:
                continue  # no ESCO concept resolved

            if uri in seen_uris:
                continue
            seen_uris.add(uri)

            results.append(ExtractedSkill(
                esco_uri=uri,
                preferred_label=label,
                matched_text=candidate,
                explicit=True,
                confidence=round(relevance, 4),
            ))

        return sorted(results, key=lambda s: (-s.confidence, s.esco_uri))

    def extract_labels(self, text: str) -> list[str]:
        return [s.preferred_label for s in self.extract(text)]
