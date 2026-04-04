"""
Step 23 — IDF + ESCO reuse-level skill weighting.

Provides two weighting factors that replace the uniform skill weights
in symbolic alignment:

1. **Reuse-level tier weight** — maps ESCO ``reuseLevel`` to a weight
   reflecting specificity:
     transversal = 0.3, cross-sector = 0.5,
     sector-specific = 0.8, occupation-specific = 1.0

2. **Corpus IDF factor** — ``log(1 + N / df(uri))`` where N = total
   documents and df = number of documents containing the URI.

Final per-skill weight:
    tier_weight(uri) × idf_factor(uri) × (1.0 if explicit, 0.5 if implicit)

Usage:
    python -m src.skills.skill_weights
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import pandas as pd
from loguru import logger

from src.scraping.config import DATA_DIR
from src.skills.esco_loader import ESCO_CSV_PATH, load_from_csv

# ── Tier weights ──────────────────────────────────────────────────────────────

REUSE_TIER_WEIGHTS: dict[str, float] = {
    "transversal": 0.3,
    "cross-sector": 0.5,
    "sector-specific": 0.8,
    "occupation-specific": 1.0,
}

DEFAULT_TIER_WEIGHT: float = 0.5  # fallback for missing/unknown reuse level


def build_reuse_level_map(
    csv_path: Path = ESCO_CSV_PATH,
) -> dict[str, str]:
    """Return {esco_uri: reuse_level} from the ESCO CSV."""
    index = load_from_csv(csv_path)
    return {
        skill.uri: skill.reuse_level
        for skill in index.skills
        if skill.reuse_level
    }


def tier_weight(reuse_level: str | None) -> float:
    """Map a reuse level string to its numeric tier weight."""
    if reuse_level is None:
        return DEFAULT_TIER_WEIGHT
    return REUSE_TIER_WEIGHTS.get(reuse_level.strip().lower(), DEFAULT_TIER_WEIGHT)


# ── Corpus IDF ────────────────────────────────────────────────────────────────

def compute_corpus_idf(
    skill_uri_lists: list[list[str]],
) -> dict[str, float]:
    """
    Compute IDF for each ESCO URI across the corpus.

    Parameters
    ----------
    skill_uri_lists:
        One list of ESCO URIs per document (duplicates within a doc are
        ignored — each URI counts once per document).

    Returns
    -------
    {uri: log(1 + N / df)} where N = total documents, df = document frequency.
    """
    n_docs = len(skill_uri_lists)
    if n_docs == 0:
        return {}

    doc_freq: Counter[str] = Counter()
    for uris in skill_uri_lists:
        doc_freq.update(set(uris))

    return {
        uri: math.log(1.0 + n_docs / df)
        for uri, df in doc_freq.items()
    }


# ── Combined weight builder ──────────────────────────────────────────────────

def build_weighted_skills(
    skill_details: list[dict],
    uri_reuse_levels: dict[str, str],
    uri_idfs: dict[str, float],
    default_idf: float = 1.0,
) -> dict[str, float]:
    """
    Build {esco_uri: weight} using tier × IDF × explicit/implicit factors.

    Parameters
    ----------
    skill_details:
        Row's skill_details list (dicts with esco_uri, explicit, implicit).
    uri_reuse_levels:
        {uri: reuse_level_string} from ESCO CSV.
    uri_idfs:
        {uri: idf_value} from ``compute_corpus_idf``.
    default_idf:
        Fallback IDF for URIs not in the corpus (neutral = 1.0).
    """
    weights: dict[str, float] = {}
    for skill in skill_details:
        uri = skill.get("esco_uri", "")
        if not uri:
            continue

        t_w = tier_weight(uri_reuse_levels.get(uri))
        idf = uri_idfs.get(uri, default_idf)
        expl_impl = 1.0 if skill.get("explicit", False) else 0.5

        w = t_w * idf * expl_impl
        weights[uri] = max(weights.get(uri, 0.0), w)

    return weights
