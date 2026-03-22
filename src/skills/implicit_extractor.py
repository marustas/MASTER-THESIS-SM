"""
Implicit skill extraction via similar-document skill propagation.

This is a faithful adaptation of Gugnani & Misra (2020), Section 4.2.

Paper's method (Section 4.2 — "Identifying Similar JDs"):
  1. Train Doc2Vec on 1.1 Million JDs; project all JDs into semantic space.
  2. For each target JD, find top-10 most similar JDs (similarity ≥ 0.59).
  3. Extract explicit skills from those similar JDs.
  4. Skills present in similar JDs but ABSENT from the target JD = implicit skills.
  5. These implicit skills are weighted at E3=0.5 (vs. E3=1.0 for explicit) in matching.

Adaptation for this thesis:
  - Instead of Doc2Vec trained on 1.1M external JDs, we use sentence-transformer
    embeddings of our own collected documents. This is justified because our dataset
    is domain-specific (ICT/AI) and sentence-transformers outperform Doc2Vec on
    semantic similarity for short-to-medium texts.
  - Similarity threshold adapted from 0.59 (paper) → configurable (default 0.60).
  - Top-K neighbours: default 10 (same as paper).
  - The corpus for neighbour search is the same document collection (job ads search
    within job ads; programmes search within programmes).

Usage:
    extractor = ImplicitSkillExtractor(explicit_extractor)
    # Fit on the full corpus first (computes and indexes all embeddings)
    extractor.fit(texts, explicit_skills_per_doc)
    # Then for any document:
    implicit = extractor.extract(text, explicit_uris_for_this_doc)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.skills.explicit_extractor import ExtractedSkill, ExplicitSkillExtractor

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 10
DEFAULT_SIM_THRESHOLD = 0.60   # paper uses 0.59; slightly raised for domain focus
IMPLICIT_SCORE = 0.5           # paper's E3 weight for implicit skills


class ImplicitSkillExtractor:
    """
    Corpus-level implicit skill extractor.

    Implements the Gugnani & Misra (2020) approach:
    find similar documents, take skills present in them but missing from the target.

    Workflow:
        extractor = ImplicitSkillExtractor(explicit_extractor)
        extractor.fit(corpus_texts, corpus_explicit_skills)
        implicit_skills = extractor.extract(target_text, target_explicit_uris)
    """

    def __init__(
        self,
        explicit_extractor: ExplicitSkillExtractor,
        model_name: str = EMBEDDING_MODEL,
        top_k: int = DEFAULT_TOP_K,
        sim_threshold: float = DEFAULT_SIM_THRESHOLD,
    ):
        self._explicit_extractor = explicit_extractor
        self._model = SentenceTransformer(model_name)
        self._top_k = top_k
        self._sim_threshold = sim_threshold

        # Set after fit()
        self._corpus_embeddings: np.ndarray | None = None   # (N, dim), L2-normalised
        self._corpus_skill_sets: list[list[ExtractedSkill]] = []   # explicit skills per doc

    # ── Corpus indexing ────────────────────────────────────────────────────────

    def fit(
        self,
        texts: list[str],
        explicit_skills_per_doc: list[list[ExtractedSkill]] | None = None,
        batch_size: int = 256,
    ) -> "ImplicitSkillExtractor":
        """
        Embed all corpus documents and store their explicit skill sets.

        Parameters
        ----------
        texts:
            Cleaned text for every document in the corpus.
        explicit_skills_per_doc:
            Pre-computed explicit skills for each document.
            If None, the explicit extractor is run on each text (slow).
        """
        logger.info(f"Fitting ImplicitSkillExtractor on {len(texts)} documents…")

        self._corpus_embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        if explicit_skills_per_doc is not None:
            self._corpus_skill_sets = explicit_skills_per_doc
        else:
            logger.info("Running explicit extraction on corpus (this may take a while)…")
            self._corpus_skill_sets = [
                self._explicit_extractor.extract(t) for t in texts
            ]

        logger.info("ImplicitSkillExtractor ready")
        return self

    # ── Per-document implicit extraction ──────────────────────────────────────

    def extract(
        self,
        text: str,
        explicit_uris: set[str],
        doc_idx: int | None = None,
    ) -> list[ExtractedSkill]:
        """
        Find implicit skills for a single document.

        Parameters
        ----------
        text:
            Cleaned text of the target document.
        explicit_uris:
            Set of ESCO URIs already found explicitly — these are excluded.
        doc_idx:
            If provided, this document is excluded from its own neighbour search
            (prevents self-match).

        Returns
        -------
        List of implicit ExtractedSkill objects, sorted by confidence descending.
        Confidence = cosine similarity of the neighbour document to the target.
        """
        if self._corpus_embeddings is None:
            raise RuntimeError("Call fit() before extract()")

        if not text or not text.strip():
            return []

        # Step 1 — Embed the target document
        target_emb = self._model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )[0]  # (dim,)

        # Step 2 — Cosine similarity against all corpus embeddings
        sims: np.ndarray = self._corpus_embeddings @ target_emb  # (N,)

        # Exclude self
        if doc_idx is not None and 0 <= doc_idx < len(sims):
            sims[doc_idx] = -1.0

        # Step 3 — Select top-K neighbours above threshold
        ranked_indices = np.argsort(sims)[::-1]
        neighbour_indices = [
            int(i) for i in ranked_indices
            if sims[i] >= self._sim_threshold
        ][: self._top_k]

        if not neighbour_indices:
            return []

        # Step 4 — Collect skills from neighbours, exclude already-explicit ones
        implicit_candidates: dict[str, tuple[ExtractedSkill, float]] = {}
        # uri → (skill, max_neighbour_similarity)

        for idx in neighbour_indices:
            neighbour_sim = float(sims[idx])
            for skill in self._corpus_skill_sets[idx]:
                if skill.esco_uri in explicit_uris:
                    continue
                if skill.esco_uri not in implicit_candidates:
                    implicit_candidates[skill.esco_uri] = (skill, neighbour_sim)
                else:
                    # Keep the highest neighbour similarity as confidence proxy
                    existing_sim = implicit_candidates[skill.esco_uri][1]
                    if neighbour_sim > existing_sim:
                        implicit_candidates[skill.esco_uri] = (skill, neighbour_sim)

        # Step 5 — Build output list with implicit flag and confidence
        results: list[ExtractedSkill] = []
        for uri, (skill, sim) in implicit_candidates.items():
            results.append(ExtractedSkill(
                esco_uri=uri,
                preferred_label=skill.preferred_label,
                matched_text=skill.matched_text,
                explicit=False,
                implicit=True,
                confidence=round(sim * IMPLICIT_SCORE, 4),  # scale by E3=0.5 (paper)
            ))

        return sorted(results, key=lambda s: -s.confidence)

    # ── Batch convenience ──────────────────────────────────────────────────────

    def extract_batch(
        self,
        texts: list[str],
        explicit_skills_list: list[list[ExtractedSkill]],
    ) -> list[list[ExtractedSkill]]:
        """
        Extract implicit skills for all documents in the corpus.
        Useful when the corpus IS the dataset (fit and extract on the same set).
        """
        results = []
        for i, (text, explicit) in enumerate(zip(texts, explicit_skills_list)):
            explicit_uris = {s.esco_uri for s in explicit}
            implicit = self.extract(text, explicit_uris=explicit_uris, doc_idx=i)
            results.append(implicit)
            if (i + 1) % 100 == 0:
                logger.info(f"  Implicit extraction: {i + 1}/{len(texts)}")
        return results
