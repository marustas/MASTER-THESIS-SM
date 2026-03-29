"""
Step 3 — Unified text preprocessing pipeline.

Processes both study programme descriptions and job advertisements through
a shared sequence of steps:

  1. HTML / markup stripping
  2. Unicode and whitespace normalization
  3. Boilerplate line removal
  4. Language detection and tagging
  5. Near-duplicate removal (MinHash)
  6. Tokenization (spaCy, with language-appropriate model)

Input:
  data/raw/programmes/lama_bpo_programmes_extended.json
  data/raw/job_ads/all_jobs.json

Output:
  data/processed/programmes/programmes_preprocessed.parquet
  data/processed/job_ads/jobs_preprocessed.parquet

Each output row retains all original fields plus:
  cleaned_text        — main text after full cleaning pipeline
  tokens              — whitespace-joined list of lemmatized, filtered tokens
  language            — detected ISO 639-1 code
  language_confidence — langdetect confidence score
  language_supported  — bool flag
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import spacy
from loguru import logger
from spacy.language import Language

from src.preprocessing.deduplication import deduplicate
from src.preprocessing.language import tag_language
from src.preprocessing.text_cleaner import clean
from src.preprocessing.translate import translate_lt_to_en

# ── Paths ──────────────────────────────────────────────────────────────────────
from src.scraping.config import DATA_DIR

PROCESSED_PROGRAMMES_DIR = DATA_DIR / "processed" / "programmes"
PROCESSED_JOB_ADS_DIR = DATA_DIR / "processed" / "job_ads"
PROCESSED_PROGRAMMES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_JOB_ADS_DIR.mkdir(parents=True, exist_ok=True)

# ── spaCy model registry ───────────────────────────────────────────────────────
# Models must be installed separately:
#   python -m spacy download en_core_web_sm
#   python -m spacy download lt_core_news_sm   (if available)
_SPACY_MODELS: dict[str, str] = {
    "en": "en_core_web_sm",
    "lt": "lt_core_news_sm",
}
_FALLBACK_MODEL = "en_core_web_sm"

_nlp_cache: dict[str, Language] = {}


def _get_nlp(lang: str | None) -> Language:
    model_name = _SPACY_MODELS.get(lang or "", _FALLBACK_MODEL)
    if model_name not in _nlp_cache:
        try:
            _nlp_cache[model_name] = spacy.load(model_name, disable=["parser", "ner"])
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.warning(
                f"spaCy model '{model_name}' not found — falling back to '{_FALLBACK_MODEL}'"
            )
            if _FALLBACK_MODEL not in _nlp_cache:
                _nlp_cache[_FALLBACK_MODEL] = spacy.load(
                    _FALLBACK_MODEL, disable=["parser", "ner"]
                )
            _nlp_cache[model_name] = _nlp_cache[_FALLBACK_MODEL]
    return _nlp_cache[model_name]


def tokenize(text: str, lang: str | None) -> list[str]:
    """
    Tokenize `text` using a language-appropriate spaCy model.

    Returns a list of lowercased lemmas, excluding:
      - stop words
      - punctuation and whitespace tokens
      - tokens shorter than 2 characters
    """
    nlp = _get_nlp(lang)
    doc = nlp(text[:1_000_000])  # spaCy has an internal limit safeguard
    return [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and len(token.text) >= 2
    ]


# ── Per-record processing ──────────────────────────────────────────────────────

def process_record(record: dict, *, text_fields: list[str], is_job_ad: bool = False) -> dict:
    """
    Apply the full preprocessing pipeline to a single record dict.

    `text_fields` specifies which fields to concatenate as the main text.
    The cleaned and tokenized result is stored in `cleaned_text` and `tokens`.
    """
    # Concatenate relevant text fields
    raw_parts = [str(record.get(f) or "") for f in text_fields]
    raw_text = " ".join(p for p in raw_parts if p.strip())

    cleaned = clean(raw_text, strip_html_tags=True, min_length=30)
    if cleaned is None:
        record["cleaned_text"] = None
        record["tokens"] = []
        record["language"] = None
        record["language_confidence"] = 0.0
        record["language_supported"] = False
        return record

    lang_info = tag_language(cleaned, for_job_ads=is_job_ad)

    # Translate Lithuanian job ads to English before tokenization so that
    # downstream skill extraction and embeddings operate on English text.
    if is_job_ad and lang_info["language"] == "lt":
        cleaned = translate_lt_to_en(cleaned)
        lang_info["language"] = "en"
        lang_info["language_confidence"] = 1.0

    tokens = tokenize(cleaned, lang_info["language"])

    record["cleaned_text"] = cleaned  # may be translated if was Lithuanian job ad
    record["tokens"] = tokens
    record.update(lang_info)
    return record


# ── Pipeline runners ───────────────────────────────────────────────────────────

def run_programmes(
    input_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """
    Preprocess study programme records.

    Text fields used (in order of preference):
      extended_description → brief_description → name
    """
    logger.info(f"Preprocessing programmes from {input_path}")
    with open(input_path, encoding="utf-8") as f:
        records: list[dict] = json.load(f)
    logger.info(f"Loaded {len(records)} programme records")

    processed = []
    for i, rec in enumerate(records, 1):
        rec = process_record(
            rec,
            text_fields=["extended_description", "brief_description", "name"],
            is_job_ad=False,
        )
        processed.append(rec)
        if i % 50 == 0:
            logger.info(f"  Programmes processed: {i}/{len(records)}")

    # Deduplicate on cleaned_text, using lama_bpo_url as exact key
    dedup = deduplicate(
        processed,
        text_field="cleaned_text",
        key_field="lama_bpo_url",
        near_duplicate=True,
    )

    # Drop records with no usable text
    valid = [r for r in dedup.kept if r.get("cleaned_text")]
    logger.info(f"Programmes after dedup + filtering: {len(valid)}")

    df = pd.DataFrame(valid)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved → {output_path}")
    return df


def run_job_ads(
    input_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """
    Preprocess job advertisement records.

    Text fields used: description → job_title
    """
    logger.info(f"Preprocessing job ads from {input_path}")
    with open(input_path, encoding="utf-8") as f:
        records: list[dict] = json.load(f)
    logger.info(f"Loaded {len(records)} job ad records")

    processed = []
    for i, rec in enumerate(records, 1):
        rec = process_record(
            rec,
            text_fields=["description", "job_title"],
            is_job_ad=True,
        )
        processed.append(rec)
        if i % 200 == 0:
            logger.info(f"  Job ads processed: {i}/{len(records)}")

    # Deduplicate: URL as exact key, cleaned_text for near-dup
    dedup = deduplicate(
        processed,
        text_field="cleaned_text",
        key_field="url",
        near_duplicate=True,
    )

    # Keep only language-supported records
    valid = [
        r for r in dedup.kept
        if r.get("cleaned_text") and r.get("language_supported")
    ]
    logger.info(f"Job ads after dedup + language filter: {len(valid)}")

    df = pd.DataFrame(valid)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved → {output_path}")
    return df


# ── CLI entry-point ────────────────────────────────────────────────────────────

def run(
    programmes_input: Path = DATA_DIR / "raw" / "programmes" / "lama_bpo_programmes_extended.json",
    jobs_input: Path = DATA_DIR / "raw" / "job_ads" / "all_jobs.json",
) -> None:
    run_programmes(
        programmes_input,
        PROCESSED_PROGRAMMES_DIR / "programmes_preprocessed.parquet",
    )
    run_job_ads(
        jobs_input,
        PROCESSED_JOB_ADS_DIR / "jobs_preprocessed.parquet",
    )


if __name__ == "__main__":
    run()
