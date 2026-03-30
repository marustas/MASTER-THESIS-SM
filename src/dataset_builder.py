"""
Step 6 — Dataset Assembly & Descriptive Validation.

Merges all upstream artefacts into one unified, machine-readable dataset:

  Step 3 → preprocessed text, tokens, language tags
  Step 4 → explicit_skills, implicit_skills, all_skills, skill_uris, skill_details
  Step 5 → embedding, embedding_brief*, embedding_extended*

The join strategy per source:
  - programmes : programmes_with_skills  LEFT JOIN  programmes_embeddings  ON index
  - job ads    : jobs_with_skills        LEFT JOIN  jobs_embeddings        ON index
  Then both are unioned with a `source_type` discriminator column.

Descriptive statistics logged and saved to data/dataset/stats.json:
  - Record counts by source type
  - Coverage rates (explicit skills, implicit skills, embeddings, extended descriptions)
  - Top-20 skills overall, per-programme, per-job-ad
  - Text length distribution (mean / median / p10 / p90)
  - Language distribution

Output:
  data/dataset/dataset.parquet   — unified dataset (all columns)
  data/dataset/stats.json        — descriptive statistics dict

Usage:
    python -m src.dataset_builder
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.scraping.config import DATA_DIR

# ── Paths ──────────────────────────────────────────────────────────────────────
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
DATASET_DIR = DATA_DIR / "dataset"

_PROG_SKILLS = PROCESSED_DIR / "programmes" / "programmes_with_skills.parquet"
_PROG_EMBED = EMBEDDINGS_DIR / "programmes_embeddings.parquet"
_JOBS_SKILLS = PROCESSED_DIR / "job_ads" / "jobs_with_skills.parquet"
_JOBS_EMBED = EMBEDDINGS_DIR / "jobs_embeddings.parquet"

DATASET_OUT = DATASET_DIR / "dataset.parquet"
STATS_OUT = DATASET_DIR / "stats.json"

# Columns from the embedding parquet to graft onto the skills parquet.
# The skills parquet was produced from the same preprocessed base, so indices align.
_EMBED_COLS = ["embedding", "embedding_brief", "embedding_extended"]


# ── Join helpers ───────────────────────────────────────────────────────────────

def _merge_skills_and_embeddings(
    skills_path: Path,
    embed_path: Path,
    source_type: str,
) -> pd.DataFrame:
    """
    Load a skills parquet and graft embedding columns from the corresponding
    embedding parquet.  Both files share the same row order (derived from the
    same preprocessed parquet), so a positional index join is safe.
    """
    if not skills_path.exists():
        logger.warning(f"Skills parquet not found, skipping: {skills_path}")
        return pd.DataFrame()

    df = pd.read_parquet(skills_path).reset_index(drop=True)

    if embed_path.exists():
        embed_df = pd.read_parquet(embed_path).reset_index(drop=True)
        cols_to_add = [c for c in _EMBED_COLS if c in embed_df.columns and c not in df.columns]
        if cols_to_add:
            df = pd.concat([df, embed_df[cols_to_add]], axis=1)
            logger.info(f"  Grafted embedding columns {cols_to_add} onto {source_type} records")
    else:
        logger.warning(f"Embedding parquet not found, embeddings omitted: {embed_path}")

    df["source_type"] = source_type
    return df


# ── Descriptive statistics ─────────────────────────────────────────────────────

def _text_length_stats(series: pd.Series) -> dict[str, float]:
    lengths = series.dropna().str.len()
    if lengths.empty:
        return {}
    return {
        "mean": round(float(lengths.mean()), 1),
        "median": round(float(lengths.median()), 1),
        "p10": round(float(lengths.quantile(0.10)), 1),
        "p90": round(float(lengths.quantile(0.90)), 1),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
    }


def _top_skills(skill_lists: pd.Series, n: int = 20) -> list[tuple[str, int]]:
    counter: Counter = Counter()
    for skills in skill_lists.dropna():
        counter.update(skills)
    return counter.most_common(n)


def _coverage(df: pd.DataFrame, col: str) -> float:
    """Fraction of rows where col is non-null and non-empty."""
    if col not in df.columns:
        return 0.0
    def _has_value(v: Any) -> bool:
        if v is None:
            return False
        if isinstance(v, list):
            return len(v) > 0
        if isinstance(v, float) and np.isnan(v):
            return False
        return bool(str(v).strip())
    return round(float(df[col].apply(_has_value).mean()), 4)


def compute_stats(df: pd.DataFrame) -> dict:
    stats: dict[str, Any] = {}

    # Record counts
    stats["total_records"] = len(df)
    stats["by_source_type"] = df["source_type"].value_counts().to_dict()

    # Coverage rates
    stats["coverage"] = {
        "explicit_skills": _coverage(df, "explicit_skills"),
        "implicit_skills": _coverage(df, "implicit_skills"),
        "embedding": _coverage(df, "embedding"),
        "extended_description": _coverage(df, "extended_description"),
        "language_supported": round(float(df["language_supported"].mean()), 4)
            if "language_supported" in df.columns else None,
    }

    # Skill frequency — overall and per source type
    if "all_skills" in df.columns:
        stats["top_skills_overall"] = _top_skills(df["all_skills"])
        for stype in df["source_type"].unique():
            subset = df.loc[df["source_type"] == stype, "all_skills"]
            stats[f"top_skills_{stype}"] = _top_skills(subset)

        skill_counts = df["all_skills"].apply(
            lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
        )
        stats["skills_per_record"] = {
            "mean": round(float(skill_counts.mean()), 2),
            "median": float(skill_counts.median()),
            "zero_skill_records": int((skill_counts == 0).sum()),
        }

    # Text length distribution
    if "cleaned_text" in df.columns:
        stats["text_length"] = _text_length_stats(df["cleaned_text"])

    # Language distribution
    if "language" in df.columns:
        stats["language_distribution"] = df["language"].value_counts().to_dict()

    return stats


# ── Main assembly ──────────────────────────────────────────────────────────────

def build(
    prog_skills: Path = _PROG_SKILLS,
    prog_embed: Path = _PROG_EMBED,
    jobs_skills: Path = _JOBS_SKILLS,
    jobs_embed: Path = _JOBS_EMBED,
    dataset_out: Path = DATASET_OUT,
    stats_out: Path = STATS_OUT,
) -> pd.DataFrame:
    """
    Assemble and validate the unified dataset.

    Returns the combined DataFrame (empty if no input files exist).
    """
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    parts = []

    prog_df = _merge_skills_and_embeddings(prog_skills, prog_embed, "programme")
    if not prog_df.empty:
        logger.info(f"Programmes loaded: {len(prog_df)} records")
        parts.append(prog_df)

    jobs_df = _merge_skills_and_embeddings(jobs_skills, jobs_embed, "job_ad")
    if not jobs_df.empty:
        logger.info(f"Job ads loaded: {len(jobs_df)} records")
        parts.append(jobs_df)

    if not parts:
        logger.warning("No input data found — dataset is empty.")
        return pd.DataFrame()

    dataset = pd.concat(parts, ignore_index=True)
    logger.info(f"Unified dataset: {len(dataset)} records total")

    # Compute and log stats
    stats = compute_stats(dataset)
    _log_stats(stats)

    # Persist
    dataset.to_parquet(dataset_out, index=False)
    logger.info(f"Dataset saved → {dataset_out}")

    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info(f"Stats saved  → {stats_out}")

    return dataset


def _log_stats(stats: dict) -> None:
    logger.info(f"Total records: {stats['total_records']}")
    logger.info(f"By source:     {stats['by_source_type']}")
    cov = stats.get("coverage", {})
    logger.info(
        f"Coverage — explicit_skills: {cov.get('explicit_skills', 0):.1%} | "
        f"implicit_skills: {cov.get('implicit_skills', 0):.1%} | "
        f"embedding: {cov.get('embedding', 0):.1%} | "
        f"extended_description: {cov.get('extended_description', 0):.1%}"
    )
    spr = stats.get("skills_per_record", {})
    if spr:
        logger.info(
            f"Skills/record — mean: {spr.get('mean', 0)} | "
            f"median: {spr.get('median', 0)} | "
            f"zero-skill records: {spr.get('zero_skill_records', 0)}"
        )
    top = stats.get("top_skills_overall", [])
    if top:
        top_str = ", ".join(f"{s}({n})" for s, n in top[:10])
        logger.info(f"Top-10 skills: {top_str}")


if __name__ == "__main__":
    build()
