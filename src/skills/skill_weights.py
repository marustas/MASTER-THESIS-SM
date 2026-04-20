"""
Step 23 — IDF-based skill weighting for symbolic alignment.

Replaces the uniform 1.0 / 0.5 weights with corpus-IDF weighting
so that rare, informative skills contribute more than ubiquitous ones.

**Default configuration (IDF-only, cap=3.0):**

    weight(uri) = min(idf(uri), 3.0) × (1.0 if explicit, 0.5 if implicit)

The IDF cap prevents a single rare skill from dominating the Jaccard
score.  Tier weighting (ESCO ``reuseLevel``) is available but disabled
by default — experiments showed IDF-only with cap=3.0 gives the best
trade-off between discriminative power (CoV +18%) and ranking quality
(only 4 of 46 programmes get a semantically worse top-1 match, vs 10
when tier weighting is active).

Usage:
    python -m src.skills.skill_weights
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
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

# ── IDF defaults ─────────────────────────────────────────────────────────────

DEFAULT_IDF_CAP: float = 3.0  # prevent single rare skill from dominating


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


def compute_median_idf(uri_idfs: dict[str, float]) -> float:
    """Return the median IDF value across all URIs.  Returns 0.0 if empty."""
    if not uri_idfs:
        return 0.0
    vals = sorted(uri_idfs.values())
    n = len(vals)
    if n % 2 == 1:
        return vals[n // 2]
    return (vals[n // 2 - 1] + vals[n // 2]) / 2.0


# ── Combined weight builder ──────────────────────────────────────────────────

def build_weighted_skills(
    skill_details: list[dict],
    uri_reuse_levels: dict[str, str],
    uri_idfs: dict[str, float],
    default_idf: float = 1.0,
    idf_cap: float | None = DEFAULT_IDF_CAP,
    use_tiers: bool = False,
) -> dict[str, float]:
    """
    Build {esco_uri: weight} using IDF × explicit/implicit factors.

    Parameters
    ----------
    skill_details:
        Row's skill_details list (dicts with esco_uri, explicit, implicit).
    uri_reuse_levels:
        {uri: reuse_level_string} from ESCO CSV.  Only used when
        *use_tiers* is True.
    uri_idfs:
        {uri: idf_value} from ``compute_corpus_idf``.
    default_idf:
        Fallback IDF for URIs not in the corpus (neutral = 1.0).
    idf_cap:
        Upper bound on IDF value.  ``None`` disables capping.
    use_tiers:
        If True, multiply by ESCO reuse-level tier weight.  Disabled by
        default — experiments showed tier weighting over-penalises
        transversal skills and creates too many false reranking.
    """
    weights: dict[str, float] = {}
    for skill in skill_details:
        uri = skill.get("esco_uri", "")
        if not uri:
            continue

        t_w = tier_weight(uri_reuse_levels.get(uri)) if use_tiers else 1.0
        idf = uri_idfs.get(uri, default_idf)
        if idf_cap is not None:
            idf = min(idf, idf_cap)
        expl_impl = 1.0 if skill.get("explicit", False) else 0.5

        w = t_w * idf * expl_impl
        weights[uri] = max(weights.get(uri, 0.0), w)

    return weights


# ── ESCO description embeddings (Step 27) ────────────────────────────────────

SKILL_EMBEDDINGS_PATH = DATA_DIR / "dataset" / "skill_embeddings.npz"


def build_skill_description_embeddings(
    embedding_model: object,
    csv_path: Path = ESCO_CSV_PATH,
) -> dict[str, np.ndarray]:
    """
    Embed ESCO skill descriptions (1-3 sentences) for coherence boost.

    Parameters
    ----------
    embedding_model:
        Any object with ``.encode(texts, normalize_embeddings=True) -> np.ndarray``
        (e.g. ``SentenceTransformer`` or ``MockEmbeddingModel``).
    csv_path:
        Path to the ESCO skills CSV.

    Returns
    -------
    ``{esco_uri: L2-normalised embedding}`` for all skills that have a
    non-empty description.  Skills without a description are skipped.
    """
    index = load_from_csv(csv_path)
    uris: list[str] = []
    descriptions: list[str] = []
    for skill in index.skills:
        if skill.description:
            uris.append(skill.uri)
            descriptions.append(skill.description)

    if not descriptions:
        logger.warning("No ESCO descriptions found — returning empty embeddings")
        return {}

    logger.info(f"Embedding {len(descriptions)} ESCO skill descriptions…")
    embeddings = embedding_model.encode(descriptions, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    return dict(zip(uris, embeddings))
