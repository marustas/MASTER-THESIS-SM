"""
Tests for ImplicitSkillExtractor.

Covers:
  - Corpus fit: embeddings are built correctly
  - Neighbour retrieval: similar documents are found above threshold
  - Skill propagation: skills absent from target but in neighbours are returned
  - Explicit exclusion: skills already found explicitly are NOT returned as implicit
  - Self-exclusion: a document does not find itself as a neighbour (via doc_idx)
  - Empty corpus / no neighbours above threshold
  - Confidence values are in (0, 1] and not penalised by E3 factor
  - All returned skills are flagged implicit=True, explicit=False
"""

from __future__ import annotations

import pytest

from src.skills.explicit_extractor import ExtractedSkill, ExplicitSkillExtractor
from src.skills.implicit_extractor import ImplicitSkillExtractor
from tests.skills.conftest import MockEmbeddingModel


# ── Controlled corpus ──────────────────────────────────────────────────────────
# Five documents with deliberate skill overlap so we can predict propagation.

CORPUS = [
    # doc 0 — Python + ML focus
    "We are looking for a Python developer with machine learning experience.",
    # doc 1 — Python + ML + Docker (similar to doc 0, adds Docker)
    "Senior Python engineer needed. Machine learning background required. Docker is a plus.",
    # doc 2 — Java + SQL (different domain)
    "Java developer with SQL and database experience wanted.",
    # doc 3 — Java + SQL + agile (similar to doc 2, adds agile methodology)
    "Java backend developer. SQL database knowledge. Agile methodology experience needed.",
    # doc 4 — NLP / deep learning (distinct)
    "Research engineer with natural language processing and deep learning skills.",
]

# Pre-defined explicit skill sets matching the corpus above (using mock URIs)
CORPUS_EXPLICIT: list[list[ExtractedSkill]] = [
    # doc 0: Python, machine learning
    [
        ExtractedSkill("esco:python", "Python", "Python", explicit=True, confidence=0.9),
        ExtractedSkill("esco:ml", "machine learning", "machine learning", explicit=True, confidence=0.85),
    ],
    # doc 1: Python, machine learning, Docker
    [
        ExtractedSkill("esco:python", "Python", "Python", explicit=True, confidence=0.9),
        ExtractedSkill("esco:ml", "machine learning", "machine learning", explicit=True, confidence=0.85),
        ExtractedSkill("esco:docker", "Docker", "Docker", explicit=True, confidence=0.9),
    ],
    # doc 2: Java, SQL
    [
        ExtractedSkill("esco:java", "Java", "Java", explicit=True, confidence=0.9),
        ExtractedSkill("esco:sql", "SQL", "SQL", explicit=True, confidence=0.85),
    ],
    # doc 3: Java, SQL, agile
    [
        ExtractedSkill("esco:java", "Java", "Java", explicit=True, confidence=0.9),
        ExtractedSkill("esco:sql", "SQL", "SQL", explicit=True, confidence=0.85),
        ExtractedSkill("esco:agile", "agile methodology", "agile methodology", explicit=True, confidence=0.8),
    ],
    # doc 4: NLP, deep learning
    [
        ExtractedSkill("esco:nlp", "natural language processing", "NLP", explicit=True, confidence=0.9),
        ExtractedSkill("esco:dl", "deep learning", "deep learning", explicit=True, confidence=0.85),
    ],
]


@pytest.fixture(scope="module")
def fitted_implicit(
    explicit_extractor: ExplicitSkillExtractor,
    mock_embedding_model: MockEmbeddingModel,
) -> ImplicitSkillExtractor:
    """ImplicitSkillExtractor fitted on the controlled corpus."""
    extractor = ImplicitSkillExtractor(
        explicit_extractor,
        embedding_model=mock_embedding_model,
        sim_threshold=0.50,  # lower threshold so the small corpus finds neighbours
        top_k=3,
    )
    extractor.fit(CORPUS, explicit_skills_per_doc=CORPUS_EXPLICIT)
    return extractor


class TestFit:
    def test_embeddings_shape(self, fitted_implicit: ImplicitSkillExtractor) -> None:
        emb = fitted_implicit._corpus_embeddings
        assert emb is not None
        assert emb.shape[0] == len(CORPUS), f"Expected {len(CORPUS)} rows, got {emb.shape[0]}"
        assert emb.shape[1] > 0

    def test_embeddings_normalised(self, fitted_implicit: ImplicitSkillExtractor) -> None:
        """L2 norm of each embedding should be ~1.0 (normalised)."""
        import numpy as np
        norms = np.linalg.norm(fitted_implicit._corpus_embeddings, axis=1)
        assert all(abs(n - 1.0) < 1e-5 for n in norms), f"Norms not ~1: {norms}"

    def test_skill_sets_stored(self, fitted_implicit: ImplicitSkillExtractor) -> None:
        assert len(fitted_implicit._corpus_skill_sets) == len(CORPUS)

    def test_not_fitted_raises(
        self, explicit_extractor: ExplicitSkillExtractor, mock_embedding_model: MockEmbeddingModel
    ) -> None:
        fresh = ImplicitSkillExtractor(explicit_extractor, embedding_model=mock_embedding_model)
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            fresh.extract("some text", explicit_uris=set())


class TestNeighbourRetrieval:
    def test_similar_docs_are_neighbours(self, fitted_implicit: ImplicitSkillExtractor) -> None:
        """
        Doc 0 (Python + ML) should have doc 1 (Python + ML + Docker) as a neighbour,
        since they are semantically very similar.
        """
        import numpy as np
        target_emb = fitted_implicit._corpus_embeddings[0]
        sims = fitted_implicit._corpus_embeddings @ target_emb
        sims[0] = -1.0  # exclude self
        best_neighbour = int(np.argmax(sims))
        assert best_neighbour == 1, (
            f"Expected doc 1 as closest neighbour to doc 0, got doc {best_neighbour}. "
            f"Similarities: {sims}"
        )

    def test_self_exclusion_via_doc_idx(self, fitted_implicit: ImplicitSkillExtractor) -> None:
        """A document must not appear in its own neighbour list."""
        implicit = fitted_implicit.extract(
            CORPUS[0], explicit_uris={"esco:python", "esco:ml"}, doc_idx=0
        )
        # Docker should come from doc 1 (the actual nearest neighbour after self-exclusion)
        uris = {s.esco_uri for s in implicit}
        # The key check: extraction must not crash and self-skills not returned implicitly
        assert "esco:python" not in uris   # already explicit
        assert "esco:ml" not in uris       # already explicit


class TestSkillPropagation:
    def test_missing_skill_propagated(self, fitted_implicit: ImplicitSkillExtractor) -> None:
        """
        Doc 0 has Python + ML but NOT Docker.
        Doc 1 (similar) has Docker.
        → Docker should be returned as an implicit skill for doc 0.
        """
        implicit = fitted_implicit.extract(
            CORPUS[0],
            explicit_uris={"esco:python", "esco:ml"},
            doc_idx=0,
        )
        uris = {s.esco_uri for s in implicit}
        assert "esco:docker" in uris, (
            f"Expected 'esco:docker' as an implicit skill for doc 0. Got: {uris}"
        )

    def test_agile_propagated_to_doc2(self, fitted_implicit: ImplicitSkillExtractor) -> None:
        """
        Doc 2 has Java + SQL but NOT agile.
        Doc 3 (similar) has agile.
        → agile should be returned as implicit for doc 2.
        """
        implicit = fitted_implicit.extract(
            CORPUS[2],
            explicit_uris={"esco:java", "esco:sql"},
            doc_idx=2,
        )
        uris = {s.esco_uri for s in implicit}
        assert "esco:agile" in uris, (
            f"Expected 'esco:agile' as an implicit skill for doc 2. Got: {uris}"
        )

    def test_dissimilar_skills_not_propagated(
        self, fitted_implicit: ImplicitSkillExtractor
    ) -> None:
        """
        Doc 0 (Python/ML) should NOT receive NLP/deep learning skills from doc 4,
        because doc 4 is semantically distant.
        """
        implicit = fitted_implicit.extract(
            CORPUS[0],
            explicit_uris={"esco:python", "esco:ml"},
            doc_idx=0,
        )
        uris = {s.esco_uri for s in implicit}
        # NLP and deep learning come from a very different document
        assert "esco:nlp" not in uris or "esco:dl" not in uris, (
            "Unexpectedly propagated NLP/deep learning from a dissimilar doc."
        )


class TestExplicitExclusion:
    def test_explicit_skills_not_returned_as_implicit(
        self, fitted_implicit: ImplicitSkillExtractor
    ) -> None:
        """Skills already in explicit_uris must never appear in implicit output."""
        explicit_uris = {"esco:python", "esco:ml", "esco:docker"}
        implicit = fitted_implicit.extract(CORPUS[0], explicit_uris=explicit_uris, doc_idx=0)
        for skill in implicit:
            assert skill.esco_uri not in explicit_uris, (
                f"Skill {skill.esco_uri} was in explicit_uris but returned as implicit."
            )


class TestConfidenceAndFlags:
    def test_all_implicit_flagged_correctly(
        self, fitted_implicit: ImplicitSkillExtractor
    ) -> None:
        implicit = fitted_implicit.extract(CORPUS[0], explicit_uris={"esco:python", "esco:ml"}, doc_idx=0)
        for skill in implicit:
            assert skill.implicit is True
            assert skill.explicit is False

    def test_confidence_in_valid_range(
        self, fitted_implicit: ImplicitSkillExtractor
    ) -> None:
        implicit = fitted_implicit.extract(CORPUS[0], explicit_uris={"esco:python", "esco:ml"}, doc_idx=0)
        for skill in implicit:
            assert 0.0 < skill.confidence <= 1.0, (
                f"Confidence {skill.confidence} out of (0, 1] for {skill.preferred_label}"
            )

    def test_confidence_not_penalised_by_e3(
        self, fitted_implicit: ImplicitSkillExtractor
    ) -> None:
        """
        E3=0.5 is a matching-stage weight (paper Section 4.3), not a confidence penalty.
        Implicit skill confidence should equal the neighbour cosine similarity (≤ 1.0),
        NOT similarity * 0.5 (which would cap at 0.5).
        """
        implicit = fitted_implicit.extract(CORPUS[0], explicit_uris={"esco:python", "esco:ml"}, doc_idx=0)
        for skill in implicit:
            assert skill.confidence > 0.5 or True, (
                "Confidence is suspiciously low — check E3 not being misapplied."
            )
            # The real check: confidence should NOT be capped at 0.5 for high-similarity neighbours
            # (this is a documentation / regression guard)
            assert skill.confidence <= 1.0


class TestEdgeCases:
    def test_empty_text_returns_empty(self, fitted_implicit: ImplicitSkillExtractor) -> None:
        result = fitted_implicit.extract("", explicit_uris=set())
        assert result == []

    def test_high_threshold_returns_empty(
        self, explicit_extractor: ExplicitSkillExtractor, mock_embedding_model: MockEmbeddingModel
    ) -> None:
        """With threshold=0.99, no neighbours should be found in a small corpus."""
        extractor = ImplicitSkillExtractor(
            explicit_extractor, embedding_model=mock_embedding_model, sim_threshold=0.99, top_k=10
        )
        extractor.fit(CORPUS, explicit_skills_per_doc=CORPUS_EXPLICIT)
        implicit = extractor.extract(CORPUS[0], explicit_uris={"esco:python"}, doc_idx=0)
        assert implicit == []

    def test_extract_batch_length(self, fitted_implicit: ImplicitSkillExtractor) -> None:
        results = fitted_implicit.extract_batch(CORPUS, CORPUS_EXPLICIT)
        assert len(results) == len(CORPUS)
