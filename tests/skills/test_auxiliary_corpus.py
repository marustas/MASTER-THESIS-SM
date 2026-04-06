"""
Tests for auxiliary corpus integration in skill_mapper.process_dataframe.

Covers:
  - Auxiliary texts enlarge the implicit extractor's fit corpus
  - Implicit skills are only extracted for main corpus documents (not auxiliary)
  - Auxiliary corpus enables skill propagation from auxiliary → main docs
  - Without auxiliary corpus, behaviour is unchanged (backwards compat)
  - Output shape matches main DataFrame regardless of auxiliary size
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.skills.explicit_extractor import ExtractedSkill, ExplicitSkillExtractor
from src.skills.implicit_extractor import ImplicitSkillExtractor
from src.skills.skill_mapper import process_dataframe
from tests.conftest import MockEmbeddingModel


# ── Fixtures ──────────────────────────────────────────────────────────────────

MAIN_TEXTS = [
    "We need a Python developer with machine learning skills.",
    "Java backend developer wanted for SQL database work.",
]

AUXILIARY_TEXTS = [
    "Python engineer needed. Docker experience is a plus.",
    "Senior Python ML engineer. Kubernetes deployment required.",
    "Java developer with agile methodology and REST API design.",
]

AUXILIARY_EXPLICIT: list[list[ExtractedSkill]] = [
    [
        ExtractedSkill("esco:python", "Python", "Python", explicit=True, confidence=0.9),
        ExtractedSkill("esco:docker", "Docker", "Docker", explicit=True, confidence=0.85),
    ],
    [
        ExtractedSkill("esco:python", "Python", "Python", explicit=True, confidence=0.9),
        ExtractedSkill("esco:ml", "machine learning", "ML", explicit=True, confidence=0.85),
        ExtractedSkill("esco:kubernetes", "Kubernetes", "Kubernetes", explicit=True, confidence=0.8),
    ],
    [
        ExtractedSkill("esco:java", "Java", "Java", explicit=True, confidence=0.9),
        ExtractedSkill("esco:agile", "agile methodology", "agile", explicit=True, confidence=0.8),
        ExtractedSkill("esco:restapi", "REST API", "REST API", explicit=True, confidence=0.85),
    ],
]


def _make_main_df() -> pd.DataFrame:
    return pd.DataFrame({
        "cleaned_text": MAIN_TEXTS,
    })


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestAuxiliaryCorpus:
    def test_output_shape_matches_main_df(
        self,
        explicit_extractor: ExplicitSkillExtractor,
        mock_embedding_model: MockEmbeddingModel,
    ) -> None:
        """Output has same number of rows as main df, regardless of auxiliary size."""
        df = _make_main_df()
        imp = ImplicitSkillExtractor(
            explicit_extractor, embedding_model=mock_embedding_model,
            sim_threshold=0.3, top_k=5,
        )
        result = process_dataframe(
            df, explicit_extractor, imp,
            auxiliary_texts=AUXILIARY_TEXTS,
            auxiliary_explicit=AUXILIARY_EXPLICIT,
        )
        assert len(result) == len(MAIN_TEXTS)

    def test_enrichment_columns_present(
        self,
        explicit_extractor: ExplicitSkillExtractor,
        mock_embedding_model: MockEmbeddingModel,
    ) -> None:
        df = _make_main_df()
        imp = ImplicitSkillExtractor(
            explicit_extractor, embedding_model=mock_embedding_model,
            sim_threshold=0.3, top_k=5,
        )
        result = process_dataframe(
            df, explicit_extractor, imp,
            auxiliary_texts=AUXILIARY_TEXTS,
            auxiliary_explicit=AUXILIARY_EXPLICIT,
        )
        for col in ("explicit_skills", "implicit_skills", "all_skills",
                     "skill_uris", "skill_details"):
            assert col in result.columns

    def test_fit_corpus_includes_auxiliary(
        self,
        explicit_extractor: ExplicitSkillExtractor,
        mock_embedding_model: MockEmbeddingModel,
    ) -> None:
        """Implicit extractor should be fitted on main + auxiliary docs."""
        df = _make_main_df()
        imp = ImplicitSkillExtractor(
            explicit_extractor, embedding_model=mock_embedding_model,
            sim_threshold=0.3, top_k=5,
        )
        process_dataframe(
            df, explicit_extractor, imp,
            auxiliary_texts=AUXILIARY_TEXTS,
            auxiliary_explicit=AUXILIARY_EXPLICIT,
        )
        # The fit corpus should have main + auxiliary documents
        expected_size = len(MAIN_TEXTS) + len(AUXILIARY_TEXTS)
        assert imp._corpus_embeddings.shape[0] == expected_size

    def test_auxiliary_enables_more_implicit_skills(
        self,
        explicit_extractor: ExplicitSkillExtractor,
        mock_embedding_model: MockEmbeddingModel,
    ) -> None:
        """With auxiliary corpus, more implicit skills should be available for propagation."""
        df = _make_main_df()

        # Without auxiliary
        imp_no_aux = ImplicitSkillExtractor(
            explicit_extractor, embedding_model=mock_embedding_model,
            sim_threshold=0.3, top_k=5,
        )
        result_no_aux = process_dataframe(df.copy(), explicit_extractor, imp_no_aux)
        implicit_no_aux = sum(len(row) for row in result_no_aux["implicit_skills"])

        # With auxiliary
        imp_with_aux = ImplicitSkillExtractor(
            explicit_extractor, embedding_model=mock_embedding_model,
            sim_threshold=0.3, top_k=5,
        )
        result_with_aux = process_dataframe(
            df.copy(), explicit_extractor, imp_with_aux,
            auxiliary_texts=AUXILIARY_TEXTS,
            auxiliary_explicit=AUXILIARY_EXPLICIT,
        )
        implicit_with_aux = sum(len(row) for row in result_with_aux["implicit_skills"])

        # Auxiliary corpus should provide at least as many implicit skills
        assert implicit_with_aux >= implicit_no_aux

    def test_without_auxiliary_unchanged(
        self,
        explicit_extractor: ExplicitSkillExtractor,
        mock_embedding_model: MockEmbeddingModel,
    ) -> None:
        """Without auxiliary, process_dataframe behaves identically to before."""
        df = _make_main_df()
        imp = ImplicitSkillExtractor(
            explicit_extractor, embedding_model=mock_embedding_model,
            sim_threshold=0.3, top_k=5,
        )
        result = process_dataframe(df, explicit_extractor, imp)
        assert len(result) == len(MAIN_TEXTS)
        assert imp._corpus_embeddings.shape[0] == len(MAIN_TEXTS)

    def test_none_auxiliary_same_as_omitted(
        self,
        explicit_extractor: ExplicitSkillExtractor,
        mock_embedding_model: MockEmbeddingModel,
    ) -> None:
        """Passing None for auxiliary should behave same as omitting it."""
        df = _make_main_df()
        imp = ImplicitSkillExtractor(
            explicit_extractor, embedding_model=mock_embedding_model,
            sim_threshold=0.3, top_k=5,
        )
        result = process_dataframe(
            df, explicit_extractor, imp,
            auxiliary_texts=None,
            auxiliary_explicit=None,
        )
        assert len(result) == len(MAIN_TEXTS)
        assert imp._corpus_embeddings.shape[0] == len(MAIN_TEXTS)

    def test_auxiliary_skills_can_propagate(
        self,
        explicit_extractor: ExplicitSkillExtractor,
        mock_embedding_model: MockEmbeddingModel,
    ) -> None:
        """
        Docker is only in auxiliary corpus (aux doc 0).
        Main doc 0 (Python + ML) should get Docker as implicit if
        aux doc 0 (Python + Docker) is a similar neighbour.
        """
        df = _make_main_df()
        imp = ImplicitSkillExtractor(
            explicit_extractor, embedding_model=mock_embedding_model,
            sim_threshold=0.3, top_k=5,
        )
        result = process_dataframe(
            df, explicit_extractor, imp,
            auxiliary_texts=AUXILIARY_TEXTS,
            auxiliary_explicit=AUXILIARY_EXPLICIT,
        )
        # Check if Docker was propagated to main doc 0
        main_0_implicit_uris = set()
        for detail in result.iloc[0]["skill_details"]:
            if detail.get("implicit"):
                main_0_implicit_uris.add(detail["esco_uri"])

        # Docker should be propagated from auxiliary Python + Docker doc
        assert "esco:docker" in main_0_implicit_uris, (
            f"Expected Docker to propagate from auxiliary. Got implicit URIs: {main_0_implicit_uris}"
        )
