"""
Shared pytest fixtures for skill extraction tests.

Uses a minimal mock ESCO index (~20 ICT skills) so tests run without
the full ESCO CSV download.  Skills are chosen to cover:
  - Single-token skills (Python, Java, SQL)
  - Multi-token skills (machine learning, data analysis)
  - The paper's example terms (Octave)
  - Skills used for implicit propagation checks
"""

from __future__ import annotations

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
def explicit_extractor(mock_esco_index: EscoIndex) -> ExplicitSkillExtractor:
    """Build the explicit extractor once for the whole test session (slow init)."""
    return ExplicitSkillExtractor(mock_esco_index)


@pytest.fixture(scope="session")
def implicit_extractor(explicit_extractor: ExplicitSkillExtractor) -> ImplicitSkillExtractor:
    return ImplicitSkillExtractor(explicit_extractor)
