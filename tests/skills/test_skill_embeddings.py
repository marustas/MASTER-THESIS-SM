"""
Tests for Step 27 — ESCO skill description embeddings.

Covers:
  - build_skill_description_embeddings: basic flow, empty descriptions, filtering
  - save_skill_embeddings / load roundtrip
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.skills.skill_weights import (
    build_skill_description_embeddings,
    save_skill_embeddings,
)


# ── Mock ESCO data ───────────────────────────────────────────────────────────

class _MockEscoSkill:
    def __init__(self, uri: str, preferred_label: str, description: str | None = None):
        self.uri = uri
        self.preferred_label = preferred_label
        self.alt_labels: list[str] = []
        self.skill_type = "skill/competence"
        self.reuse_level = "sector-specific"
        self.description = description

    @property
    def all_labels(self) -> list[str]:
        return [self.preferred_label] + self.alt_labels


class _MockEscoIndex:
    def __init__(self, skills: list[_MockEscoSkill]):
        self.skills = skills


class _MockEmbeddingModel:
    """Deterministic 8-dim embedding for testing."""
    def __init__(self, dim: int = 8):
        self._dim = dim

    def encode(self, texts: list[str], *, normalize_embeddings: bool = True, **kwargs) -> np.ndarray:
        rng = np.random.default_rng(42)
        embeddings = rng.random((len(texts), self._dim)).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            embeddings = embeddings / norms
        return embeddings

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim


# ── build_skill_description_embeddings ───────────────────────────────────────

class TestBuildSkillDescriptionEmbeddings:
    def test_basic_flow(self, tmp_path, monkeypatch):
        """Embeds descriptions and returns {uri: embedding} dict."""
        skills = [
            _MockEscoSkill("uri:a", "Python", "Programming language for general use"),
            _MockEscoSkill("uri:b", "SQL", "Query language for databases"),
        ]
        index = _MockEscoIndex(skills)
        monkeypatch.setattr(
            "src.skills.skill_weights.load_from_csv",
            lambda path: index,
        )

        model = _MockEmbeddingModel(dim=8)
        result = build_skill_description_embeddings(model, csv_path=tmp_path / "fake.csv")

        assert len(result) == 2
        assert "uri:a" in result
        assert "uri:b" in result
        assert result["uri:a"].shape == (8,)
        # L2-normalised
        assert np.linalg.norm(result["uri:a"]) == pytest.approx(1.0, abs=1e-4)

    def test_skips_skills_without_description(self, tmp_path, monkeypatch):
        """Skills with no description are excluded."""
        skills = [
            _MockEscoSkill("uri:a", "Python", "Programming language"),
            _MockEscoSkill("uri:b", "SQL", None),  # no description
            _MockEscoSkill("uri:c", "Java", ""),    # empty description
        ]
        index = _MockEscoIndex(skills)
        monkeypatch.setattr(
            "src.skills.skill_weights.load_from_csv",
            lambda path: index,
        )

        model = _MockEmbeddingModel(dim=8)
        result = build_skill_description_embeddings(model, csv_path=tmp_path / "fake.csv")

        assert len(result) == 1
        assert "uri:a" in result

    def test_empty_index(self, tmp_path, monkeypatch):
        """Empty ESCO index returns empty dict."""
        index = _MockEscoIndex([])
        monkeypatch.setattr(
            "src.skills.skill_weights.load_from_csv",
            lambda path: index,
        )

        model = _MockEmbeddingModel(dim=8)
        result = build_skill_description_embeddings(model, csv_path=tmp_path / "fake.csv")
        assert result == {}

    def test_all_skills_no_descriptions(self, tmp_path, monkeypatch):
        """All skills without descriptions returns empty dict."""
        skills = [
            _MockEscoSkill("uri:a", "Python", None),
            _MockEscoSkill("uri:b", "SQL", None),
        ]
        index = _MockEscoIndex(skills)
        monkeypatch.setattr(
            "src.skills.skill_weights.load_from_csv",
            lambda path: index,
        )

        model = _MockEmbeddingModel(dim=8)
        result = build_skill_description_embeddings(model, csv_path=tmp_path / "fake.csv")
        assert result == {}


# ── save_skill_embeddings ────────────────────────────────────────────────────

class TestSaveSkillEmbeddings:
    def test_roundtrip(self, tmp_path):
        """Save and reload produces identical results."""
        rng = np.random.default_rng(42)
        embeddings = {
            "uri:a": rng.random(8).astype(np.float32),
            "uri:b": rng.random(8).astype(np.float32),
        }
        path = tmp_path / "skill_embeddings.npz"
        save_skill_embeddings(embeddings, path=path)

        assert path.exists()
        data = np.load(path)
        loaded = dict(zip(data["uris"], data["embeddings"]))
        assert set(loaded.keys()) == {"uri:a", "uri:b"}
        np.testing.assert_array_almost_equal(loaded["uri:a"], embeddings["uri:a"])

    def test_empty_dict_no_file(self, tmp_path):
        """Empty dict does not create a file."""
        path = tmp_path / "skill_embeddings.npz"
        save_skill_embeddings({}, path=path)
        assert not path.exists()
