"""
Implicit skill extraction via document embeddings.

Approach (inspired by Gugnani & Misra, 2020):
  The original paper showed that extracting implicit skills — competencies
  *implied* by the document context but not explicitly named — improved
  job–resume matching reciprocal rank by 29.4%.

  Implementation:
  1. Embed all ESCO skill descriptions using a sentence-transformer model.
     These embeddings are computed once and cached to disk.
  2. For each document, extract candidate phrases:
       - Noun chunks from spaCy dependency parse
       - Named entities (ORG, PRODUCT, LANGUAGE tags often capture tech skills)
       - Bigrams/trigrams from sliding window
  3. Embed each candidate phrase.
  4. Compute cosine similarity between the phrase embedding and all ESCO skill
     embeddings.
  5. Candidates whose top-1 ESCO match exceeds `IMPLICIT_THRESHOLD` and whose
     surface form does NOT appear in the explicit skill set → implicit skills.

  The final skill set per document = explicit ∪ implicit, each tagged accordingly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import spacy
from loguru import logger
from sentence_transformers import SentenceTransformer
from spacy.language import Language

from src.skills.esco_loader import EscoIndex, EscoSkill
from src.skills.explicit_extractor import ExtractedSkill
from src.scraping.config import DATA_DIR

# ── Configuration ──────────────────────────────────────────────────────────────
IMPLICIT_THRESHOLD: float = 0.72       # cosine similarity cutoff for implicit match
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"   # fast, strong multilingual performance
ESCO_EMBEDDINGS_CACHE: Path = DATA_DIR / "raw" / "esco" / "esco_skill_embeddings.npz"
MAX_CANDIDATE_TOKENS: int = 6          # max tokens in a candidate phrase
MIN_CANDIDATE_CHARS: int = 3           # minimum candidate phrase length
BATCH_SIZE: int = 256                  # embedding batch size


# ── Implicit extractor ─────────────────────────────────────────────────────────

class ImplicitSkillExtractor:
    """
    Embedding-based implicit skill extractor.

    Build once per process (loading/computing ESCO embeddings is expensive):
        extractor = ImplicitSkillExtractor(esco_index)
    """

    def __init__(
        self,
        esco_index: EscoIndex,
        model_name: str = EMBEDDING_MODEL,
        threshold: float = IMPLICIT_THRESHOLD,
        nlp_model: str = "en_core_web_sm",
    ):
        self._index = esco_index
        self._threshold = threshold
        self._model = SentenceTransformer(model_name)
        self._nlp = self._load_nlp(nlp_model)
        self._esco_embeddings: Optional[np.ndarray] = None   # (n_skills, dim)
        self._esco_uris: list[str] = []
        self._load_or_compute_esco_embeddings()

    # ── Setup ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_nlp(model: str) -> Language:
        try:
            return spacy.load(model, disable=["ner"])
        except OSError:
            logger.warning(f"spaCy '{model}' not found — using blank model (no noun chunks)")
            return spacy.blank("en")

    def _load_or_compute_esco_embeddings(self) -> None:
        """Load ESCO skill embeddings from cache, or compute and cache them."""
        cache_key = self._embeddings_cache_key()
        cache_path = ESCO_EMBEDDINGS_CACHE.with_stem(
            f"esco_skill_embeddings_{cache_key[:8]}"
        )

        if cache_path.exists():
            logger.info(f"Loading ESCO embeddings from cache: {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            self._esco_embeddings = data["embeddings"]
            self._esco_uris = list(data["uris"])
            logger.info(f"Loaded {len(self._esco_uris)} ESCO skill embeddings")
            return

        logger.info(
            f"Computing embeddings for {len(self._index.skills)} ESCO skills "
            f"(model: {EMBEDDING_MODEL}) — this runs once and is cached…"
        )
        texts = [
            f"{s.preferred_label}. {s.description or ''}"
            for s in self._index.skills
        ]
        self._esco_uris = [s.uri for s in self._index.skills]
        self._esco_embeddings = self._model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            embeddings=self._esco_embeddings,
            uris=np.array(self._esco_uris),
        )
        logger.info(f"ESCO embeddings cached → {cache_path}")

    def _embeddings_cache_key(self) -> str:
        """Hash over skill URIs + model name to detect stale caches."""
        content = EMBEDDING_MODEL + "".join(s.uri for s in self._index.skills[:100])
        return hashlib.md5(content.encode()).hexdigest()

    # ── Candidate extraction ───────────────────────────────────────────────────

    def _extract_candidates(self, text: str) -> list[str]:
        """
        Extract candidate phrases from `text`:
          - spaCy noun chunks
          - Named entities (PRODUCT, ORG, LANGUAGE, etc.)
          - Sliding n-gram window (bigrams and trigrams) over tokens
        """
        doc = self._nlp(text[:100_000])
        candidates: set[str] = set()

        # Noun chunks
        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip()
            if MIN_CANDIDATE_CHARS <= len(phrase) and len(chunk) <= MAX_CANDIDATE_TOKENS:
                candidates.add(phrase.lower())

        # Named entities
        for ent in doc.ents:
            if ent.label_ in {"PRODUCT", "ORG", "LANGUAGE", "GPE", "WORK_OF_ART"}:
                phrase = ent.text.strip().lower()
                if len(phrase) >= MIN_CANDIDATE_CHARS:
                    candidates.add(phrase)

        # Sliding n-grams over non-stop, non-punct tokens (2–3 tokens)
        tokens = [
            t.lemma_.lower()
            for t in doc
            if not t.is_stop and not t.is_punct and not t.is_space and len(t.text) >= 2
        ]
        for n in (2, 3):
            for i in range(len(tokens) - n + 1):
                candidates.add(" ".join(tokens[i : i + n]))

        return list(candidates)

    # ── Implicit extraction ────────────────────────────────────────────────────

    def extract(
        self,
        text: str,
        explicit_uris: set[str],
    ) -> list[ExtractedSkill]:
        """
        Extract implicit skills from `text`.

        `explicit_uris` — set of ESCO URIs already found by the explicit extractor.
        Returns only skills NOT in `explicit_uris` whose embedding similarity
        exceeds `self._threshold`.
        """
        if self._esco_embeddings is None or not text.strip():
            return []

        candidates = self._extract_candidates(text)
        if not candidates:
            return []

        # Embed all candidates in one batch
        cand_embeddings: np.ndarray = self._model.encode(
            candidates,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Cosine similarity: (n_candidates, n_esco_skills)
        # Both embeddings are L2-normalised → dot product = cosine similarity
        sim_matrix: np.ndarray = cand_embeddings @ self._esco_embeddings.T

        results: list[ExtractedSkill] = []
        seen_uris: set[str] = set(explicit_uris)

        for i, candidate in enumerate(candidates):
            best_idx = int(np.argmax(sim_matrix[i]))
            best_sim = float(sim_matrix[i, best_idx])

            if best_sim < self._threshold:
                continue

            uri = self._esco_uris[best_idx]
            if uri in seen_uris:
                continue

            skill = self._index.lookup_uri(uri)
            if skill is None:
                continue

            seen_uris.add(uri)
            results.append(ExtractedSkill(
                esco_uri=uri,
                preferred_label=skill.preferred_label,
                matched_text=candidate,
                explicit=False,
                implicit=True,
                confidence=round(best_sim, 4),
            ))

        return sorted(results, key=lambda s: (-s.confidence, s.esco_uri))
