"""
Tests for Steps 27 & 31 — ESCO description embeddings and programme IDF.

Covers:
  - build_skill_description_embeddings: basic flow, empty descriptions, filtering
  - save_skill_embeddings / load roundtrip
  - compute_programme_idf: basic IDF, programme-only filtering (Step 31)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.skills.skill_weights import (
    build_skill_description_embeddings,
    compute_programme_idf,
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


# ── compute_programme_idf (Step 31) ──────────────────────────────────────────

def _skill(uri: str, *, explicit: bool = True) -> dict:
    return {
        "esco_uri": uri,
        "preferred_label": uri.split(":")[-1],
        "matched_text": uri.split(":")[-1],
        "explicit": explicit,
        "implicit": not explicit,
        "confidence": 1.0,
    }


class TestComputeProgrammeIdf:
    def test_basic_programme_idf(self):
        """IDF computed only from programme rows."""
        rows = [
            {"source_type": "programme", "skill_details": [_skill("uri:python"), _skill("uri:ml")]},
            {"source_type": "programme", "skill_details": [_skill("uri:python")]},
            {"source_type": "job_ad", "skill_details": [_skill("uri:python"), _skill("uri:sql")]},
        ]
        df = pd.DataFrame(rows)
        result = compute_programme_idf(df)

        # 2 programmes: python in 2/2, ml in 1/2
        assert result["uri:python"] == pytest.approx(math.log(2.0))
        assert result["uri:ml"] == pytest.approx(math.log(3.0))
        # sql is only in job_ad, not in programmes — should not appear
        assert "uri:sql" not in result

    def test_ignores_job_ads(self):
        """Job ad skills do not affect programme IDF."""
        rows = [
            {"source_type": "programme", "skill_details": [_skill("uri:python")]},
            {"source_type": "job_ad", "skill_details": [_skill("uri:python"), _skill("uri:java")]},
            {"source_type": "job_ad", "skill_details": [_skill("uri:java")]},
        ]
        df = pd.DataFrame(rows)
        result = compute_programme_idf(df)

        assert "uri:python" in result
        assert "uri:java" not in result

    def test_empty_programmes(self):
        """No programmes returns empty dict."""
        rows = [
            {"source_type": "job_ad", "skill_details": [_skill("uri:python")]},
        ]
        df = pd.DataFrame(rows)
        result = compute_programme_idf(df)
        assert result == {}

    def test_unique_skill_higher_idf(self):
        """Skill unique to one programme has higher IDF than shared skill."""
        rows = [
            {"source_type": "programme", "skill_details": [_skill("uri:shared"), _skill("uri:unique")]},
            {"source_type": "programme", "skill_details": [_skill("uri:shared")]},
            {"source_type": "programme", "skill_details": [_skill("uri:shared")]},
        ]
        df = pd.DataFrame(rows)
        result = compute_programme_idf(df)

        assert result["uri:unique"] > result["uri:shared"]

    def test_no_skill_details_handled(self):
        """Rows with missing or empty skill_details are handled gracefully."""
        rows = [
            {"source_type": "programme", "skill_details": []},
            {"source_type": "programme", "skill_details": [_skill("uri:a")]},
        ]
        df = pd.DataFrame(rows)
        result = compute_programme_idf(df)
        assert "uri:a" in result
