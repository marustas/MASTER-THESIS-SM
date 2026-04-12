"""
Language detection and normalization.

The dataset contains two primary languages:
  - Lithuanian (lt) — study programme descriptions from LAMA BPO / university sites
  - English (en)    — most EU job advertisements

Strategy:
  - Detect language using `langdetect` (fast, offline).
  - Keep records whose detected language is Lithuanian or English.
  - Tag each record with its detected language so downstream steps can apply
    the correct tokenizer / spaCy model.
  - Optionally translate Lithuanian → English for joint embedding (placeholder
    for Step 5 — actual translation deferred to that step).
"""

from __future__ import annotations

from functools import lru_cache

from langdetect import DetectorFactory, LangDetectException, detect, detect_langs
from loguru import logger

# Make langdetect deterministic
DetectorFactory.seed = 42

SUPPORTED_LANGUAGES: frozenset[str] = frozenset({"lt", "en"})

# Language codes accepted for job ads (broader — EU postings may also be in
# German, Latvian, etc.; we keep them but flag non-lt/en for transparency).
ACCEPTED_JOB_LANGUAGES: frozenset[str] = frozenset({"lt", "en", "lv", "et", "pl", "de", "fr", "nl", "sv", "es"})


def detect_language(text: str) -> str | None:
    """
    Return the ISO 639-1 language code for `text`, or None on failure.

    Requires at least 20 characters for a reliable detection.
    """
    if not text or len(text.strip()) < 20:
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None


def detect_language_with_confidence(text: str) -> tuple[str | None, float]:
    """
    Return (language_code, probability) for the top-ranked language.
    Returns (None, 0.0) when detection fails.
    """
    if not text or len(text.strip()) < 20:
        return None, 0.0
    try:
        langs = detect_langs(text)
        if langs:
            top = langs[0]
            return top.lang, round(top.prob, 3)
    except LangDetectException:
        pass
    return None, 0.0


def is_supported(language_code: str | None, *, for_job_ads: bool = False) -> bool:
    """Return True if the language is acceptable for the given data type."""
    if language_code is None:
        return False
    pool = ACCEPTED_JOB_LANGUAGES if for_job_ads else SUPPORTED_LANGUAGES
    return language_code in pool


def tag_language(text: str, *, for_job_ads: bool = False) -> dict:
    """
    Detect language and return a dict with:
      {
        "language": "en" | "lt" | ...,
        "language_confidence": 0.95,
        "language_supported": True | False,
      }
    Used to enrich preprocessed records.
    """
    lang, conf = detect_language_with_confidence(text)
    supported = is_supported(lang, for_job_ads=for_job_ads)
    if not supported:
        logger.debug(f"Unsupported language detected: {lang!r} (conf={conf})")
    return {
        "language": lang,
        "language_confidence": conf,
        "language_supported": supported,
    }
