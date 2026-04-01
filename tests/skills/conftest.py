"""
Skill extraction test fixtures.

All shared fixtures (MockEmbeddingModel, mock_esco_index, explicit_extractor,
implicit_extractor) are defined in tests/conftest.py and inherited automatically.

MockEmbeddingModel and MOCK_SKILLS are re-exported here for modules that
import them directly (e.g. tests/embeddings/test_generator.py).
"""

from tests.conftest import MockEmbeddingModel, MOCK_SKILLS  # noqa: F401
