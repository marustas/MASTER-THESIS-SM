"""
Tests for Step 31 — Programme-level IDF.

Covers:
  - compute_programme_idf: basic IDF, programme-only filtering
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.skills.skill_weights import compute_programme_idf


# ── helpers ──────────────────────────────────────────────────────────────────

def _skill(uri: str, *, explicit: bool = True) -> dict:
    return {
        "esco_uri": uri,
        "preferred_label": uri.split(":")[-1],
        "matched_text": uri.split(":")[-1],
        "explicit": explicit,
        "implicit": not explicit,
        "confidence": 1.0,
    }


# ── compute_programme_idf (Step 31) ──────────────────────────────────────────

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
