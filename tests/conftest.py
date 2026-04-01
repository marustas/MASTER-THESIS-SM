"""
Shared pytest fixtures available to all test packages.

Moved from tests/skills/conftest.py so that tests/evaluation/ and other
packages can also access mock ESCO, mock embedding model, and extractors.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.skills.esco_loader import EscoIndex, EscoSkill
from src.skills.explicit_extractor import ExplicitSkillExtractor
from src.skills.implicit_extractor import ImplicitSkillExtractor


# ── Mock ESCO skills ───────────────────────────────────────────────────────────

MOCK_SKILLS: list[dict] = [
    {"uri": "esco:python",        "preferred_label": "Python",               "alt_labels": ["Python programming", "Python scripting"]},
    {"uri": "esco:java",          "preferred_label": "Java",                 "alt_labels": ["Java programming", "Java development"]},
    {"uri": "esco:sql",           "preferred_label": "SQL",                  "alt_labels": ["structured query language"]},
    {"uri": "esco:octave",        "preferred_label": "GNU Octave",           "alt_labels": ["Octave", "octave programming"]},
    {"uri": "esco:ml",            "preferred_label": "machine learning",     "alt_labels": ["ML", "statistical learning"]},
    {"uri": "esco:da",            "preferred_label": "data analysis",        "alt_labels": ["data analytics", "analysing data"]},
    {"uri": "esco:de",            "preferred_label": "data engineering",     "alt_labels": ["data pipelines"]},
    {"uri": "esco:docker",        "preferred_label": "Docker",               "alt_labels": ["containerisation", "container technology"]},
    {"uri": "esco:kubernetes",    "preferred_label": "Kubernetes",           "alt_labels": ["K8s", "container orchestration"]},
    {"uri": "esco:agile",         "preferred_label": "agile methodology",    "alt_labels": ["Scrum", "agile development"]},
    {"uri": "esco:oop",           "preferred_label": "object-oriented programming", "alt_labels": ["OOP", "OOP programming"]},
    {"uri": "esco:softdev",       "preferred_label": "software development", "alt_labels": ["software engineering"]},
    {"uri": "esco:nlp",           "preferred_label": "natural language processing", "alt_labels": ["NLP", "text mining"]},
    {"uri": "esco:cv",            "preferred_label": "computer vision",      "alt_labels": ["image recognition"]},
    {"uri": "esco:dl",            "preferred_label": "deep learning",        "alt_labels": ["neural networks", "DNN"]},
    {"uri": "esco:cloud",         "preferred_label": "cloud computing",      "alt_labels": ["AWS", "Azure", "GCP"]},
    {"uri": "esco:git",           "preferred_label": "Git",                  "alt_labels": ["version control", "GitHub"]},
    {"uri": "esco:pytorch",       "preferred_label": "PyTorch",              "alt_labels": ["torch"]},
    {"uri": "esco:tensorflow",    "preferred_label": "TensorFlow",           "alt_labels": ["tf"]},
    {"uri": "esco:restapi",       "preferred_label": "REST API",             "alt_labels": ["RESTful API", "REST web service"]},
]


# ── Deterministic mock embedding model ────────────────────────────────────────

class MockEmbeddingModel:
    """
    Network-free embedding model for tests.

    Maps texts to 16-dimensional L2-normalised vectors using keyword presence.
    Each keyword boosts a fixed dimension; this encodes domain similarity:

      Dim  0 — python
      Dim  1 — java
      Dim  2 — sql / database
      Dim  3 — machine / learning (shared by ML and "deep learning" at a lower weight)
      Dim  4 — docker / container
      Dim  5 — kubernetes
      Dim  6 — agile / scrum
      Dim  7 — natural / language / processing / nlp / text
      Dim  8 — deep / neural
      Dim  9 — cloud / aws / azure / gcp
      Dim 10 — git / github / version
      Dim 11 — octave
      Dim 12 — pytorch / torch
      Dim 13 — tensorflow
      Dim 14 — rest / api
      Dim 15 — data / analysis / analytics / engineering
    """

    DIM = 16

    _KW: dict[str, int] = {
        "python": 0,
        "java": 1,
        "sql": 2, "database": 2,
        "machine": 3, "learning": 3,
        "docker": 4, "container": 4, "containerisation": 4,
        "kubernetes": 5, "k8s": 5,
        "agile": 6, "scrum": 6,
        "natural": 7, "language": 7, "processing": 7, "nlp": 7, "text": 7,
        "deep": 8, "neural": 8,
        "cloud": 9, "aws": 9, "azure": 9, "gcp": 9,
        "git": 10, "github": 10, "version": 10,
        "octave": 11,
        "pytorch": 12, "torch": 12,
        "tensorflow": 13,
        "rest": 14, "api": 14, "restful": 14,
        "data": 15, "analysis": 15, "analytics": 15, "engineering": 15,
    }

    def get_sentence_embedding_dimension(self) -> int:
        return self.DIM

    def encode(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool = True,
        batch_size: int = 256,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        result = []
        for text in texts:
            vec = np.zeros(self.DIM, dtype=float)
            lower = text.lower()
            for kw, dim in self._KW.items():
                if kw in lower:
                    vec[dim] += 1.0
            # Avoid zero vectors (e.g. for generic ESCO labels with no keyword match)
            norm = np.linalg.norm(vec)
            if norm < 1e-8:
                # Spread uniformly so cosine similarity with others stays low
                vec = np.full(self.DIM, 1.0 / self.DIM)
                norm = np.linalg.norm(vec)
            if normalize_embeddings:
                vec = vec / norm
            result.append(vec)
        return np.array(result, dtype=np.float32)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def mock_embedding_model() -> MockEmbeddingModel:
    return MockEmbeddingModel()


@pytest.fixture(scope="session")
def mock_esco_index() -> EscoIndex:
    skills = [
        EscoSkill(
            uri=s["uri"],
            preferred_label=s["preferred_label"],
            alt_labels=s.get("alt_labels", []),
        )
        for s in MOCK_SKILLS
    ]
    index = EscoIndex(skills=skills)
    index.build()
    return index


@pytest.fixture(scope="session")
def explicit_extractor(
    mock_esco_index: EscoIndex,
    mock_embedding_model: MockEmbeddingModel,
) -> ExplicitSkillExtractor:
    """Build the explicit extractor once for the whole test session (no network)."""
    return ExplicitSkillExtractor(mock_esco_index, embedding_model=mock_embedding_model)


@pytest.fixture(scope="session")
def implicit_extractor(
    explicit_extractor: ExplicitSkillExtractor,
    mock_embedding_model: MockEmbeddingModel,
) -> ImplicitSkillExtractor:
    return ImplicitSkillExtractor(explicit_extractor, embedding_model=mock_embedding_model)
