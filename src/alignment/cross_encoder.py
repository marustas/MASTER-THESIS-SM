"""
Step 38 — Cross-encoder re-ranking helper.

Wraps a HuggingFace cross-encoder (default ``cross-encoder/ms-marco-MiniLM-L-6-v2``)
behind a thin ``score_pairs(pairs)`` API.  Used by the hybrid alignment to
re-evaluate (programme_text, job_text) pairs from scratch with full
token-level attention, producing fresh ranking variance inside the candidate
pool that the bi-encoder cosine has already exhausted.

Model is injected the same way as ``ExplicitSkillExtractor.embedding_model``:

  * ``str`` — passed to ``sentence_transformers.CrossEncoder`` (downloads
    from HuggingFace unless ``HF_HUB_OFFLINE=1`` and the model is cached)
  * any object with ``predict(pairs) -> np.ndarray`` — used directly
    (e.g. ``MockCrossEncoder`` in tests, no network required)

The default model is small (~80 MB, 6-layer MiniLM) and CPU-tractable for
the ~2 250 pairs (45 programmes × 50 candidates) the hybrid pipeline
generates.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from loguru import logger

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_LENGTH = 512


def load_cross_encoder(model: str | object = DEFAULT_MODEL):
    """
    Resolve a cross-encoder model spec to a callable scorer.

    Returns the object unchanged if it already exposes ``predict``.
    Otherwise loads ``sentence_transformers.CrossEncoder`` from the
    given name.
    """
    if hasattr(model, "predict"):
        return model

    if not isinstance(model, str):
        raise TypeError(
            f"cross_encoder_model must be a str or expose .predict(), got {type(model).__name__}"
        )

    from sentence_transformers import CrossEncoder

    logger.info(f"Loading cross-encoder: {model}")
    return CrossEncoder(model, max_length=DEFAULT_MAX_LENGTH)


def score_pairs(
    model,
    pairs: Sequence[tuple[str, str]],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """
    Score a sequence of (query, document) pairs with a cross-encoder.

    Empty / missing texts are replaced with a single space so the model
    never receives an empty string (which some tokenisers reject).
    The corresponding output score for any all-empty pair is set to
    ``-inf`` after the fact so it ranks last.

    Returns a 1-D float32 ndarray of length ``len(pairs)``.
    """
    if len(pairs) == 0:
        return np.zeros(0, dtype=np.float32)

    safe_pairs: list[tuple[str, str]] = []
    empty_mask: list[bool] = []
    for q, d in pairs:
        q_ok = bool(q and q.strip())
        d_ok = bool(d and d.strip())
        empty_mask.append(not (q_ok and d_ok))
        safe_pairs.append((q if q_ok else " ", d if d_ok else " "))

    raw = model.predict(
        safe_pairs,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )
    scores = np.asarray(raw, dtype=np.float32).reshape(-1)

    if any(empty_mask):
        scores = scores.copy()
        scores[np.asarray(empty_mask)] = -np.inf

    return scores


def score_pairs_sectioned(
    model,
    pairs: Sequence[tuple[str, str]],
    *,
    section_parser: Callable[[str], dict[str, str]],
    section_weights: dict[str, float],
    pool: str = "weighted_mean",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """
    Score (programme, job) pairs by splitting each programme into sections,
    scoring (section_text, job_text) independently, and pooling.

    The cross-encoder shares its 512-token budget across the pair, so each
    side gets ~256 tokens.  Splitting the programme into sections lifts the
    truncation ceiling: every section gets its own re-rank pass against the
    full job text, and the pool combines the per-section evidence.

    Pooling
    -------
    ``"weighted_mean"`` :  Σ_g w_g · s_g  /  Σ_g w_g (over non-empty groups).
                            Uses ``section_weights`` directly.  Mirrors the
                            section-weighted programme embedding (Step 34).
    ``"max"``           :  ``max_g s_g``.  Best-matching section dominates;
                            useful for niche programmes whose discriminator
                            lives in a single section.

    Empty programme/job pairs (where every section is empty or the job text
    is empty) get -inf so they rank last.
    """
    if pool not in ("weighted_mean", "max"):
        raise ValueError(f"pool must be 'weighted_mean' or 'max', got {pool!r}")

    if len(pairs) == 0:
        return np.zeros(0, dtype=np.float32)

    # Parse all programme texts once.  Cache per unique programme text so
    # repeated programme→jobs scoring does not re-parse.
    parse_cache: dict[str, dict[str, str]] = {}

    flat_pairs: list[tuple[str, str]] = []
    flat_meta: list[tuple[int, str, float]] = []  # (pair_idx, group, weight)
    empty_pair: list[bool] = []

    for i, (q, d) in enumerate(pairs):
        d_clean = d if (d and d.strip()) else ""
        if not d_clean or not (q and q.strip()):
            empty_pair.append(True)
            continue
        empty_pair.append(False)
        if q not in parse_cache:
            parse_cache[q] = section_parser(q)
        sections = parse_cache[q]
        for group, weight in section_weights.items():
            text = sections.get(group, "").strip()
            if not text:
                continue
            flat_pairs.append((text, d_clean))
            flat_meta.append((i, group, weight))

    if not flat_pairs:
        return np.full(len(pairs), -np.inf, dtype=np.float32)

    raw = model.predict(
        flat_pairs, batch_size=batch_size, show_progress_bar=False,
    )
    flat_scores = np.asarray(raw, dtype=np.float32).reshape(-1)

    # Pool per pair index
    out = np.full(len(pairs), -np.inf, dtype=np.float32)
    if pool == "weighted_mean":
        weighted_sum = np.zeros(len(pairs), dtype=np.float64)
        weight_total = np.zeros(len(pairs), dtype=np.float64)
        for s, (i, _g, w) in zip(flat_scores, flat_meta):
            weighted_sum[i] += w * s
            weight_total[i] += w
        non_empty = weight_total > 0
        out[non_empty] = (weighted_sum[non_empty] / weight_total[non_empty]).astype(np.float32)
    else:  # max
        for s, (i, _g, _w) in zip(flat_scores, flat_meta):
            if s > out[i]:
                out[i] = s

    # Pairs that started empty (caller side) stay -inf — already initialised.
    return out
