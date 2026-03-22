"""
Step 4 — Skill extraction & ESCO ontology mapping coordinator.

Orchestrates explicit + implicit extraction for both preprocessed data sources
and enriches each record with:
  - explicit_skills:  list of ESCO preferred labels (explicitly mentioned)
  - implicit_skills:  list of ESCO preferred labels (implied, embedding-based)
  - all_skills:       union of explicit + implicit
  - skill_uris:       corresponding ESCO URIs for all_skills
  - skill_details:    full ExtractedSkill dicts for downstream analysis

Input:
  data/processed/programmes/programmes_preprocessed.parquet
  data/processed/job_ads/jobs_preprocessed.parquet

Output:
  data/processed/programmes/programmes_with_skills.parquet
  data/processed/job_ads/jobs_with_skills.parquet
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from loguru import logger

from src.skills.esco_loader import EscoIndex, load_esco_index
from src.skills.explicit_extractor import ExplicitSkillExtractor, ExtractedSkill
from src.skills.implicit_extractor import ImplicitSkillExtractor
from src.scraping.config import DATA_DIR

PROCESSED_DIR = DATA_DIR / "processed"


def _extract_skills_for_record(
    text: str,
    explicit_extractor: ExplicitSkillExtractor,
    implicit_extractor: ImplicitSkillExtractor,
) -> dict:
    """Run both extractors on a single cleaned text and return enrichment fields."""
    explicit_skills = explicit_extractor.extract(text)
    explicit_uris = {s.esco_uri for s in explicit_skills}

    implicit_skills = implicit_extractor.extract(text, explicit_uris=explicit_uris)

    all_skills: list[ExtractedSkill] = explicit_skills + implicit_skills

    return {
        "explicit_skills": [s.preferred_label for s in explicit_skills],
        "implicit_skills": [s.preferred_label for s in implicit_skills],
        "all_skills": [s.preferred_label for s in all_skills],
        "skill_uris": [s.esco_uri for s in all_skills],
        "skill_details": [asdict(s) for s in all_skills],
    }


def process_dataframe(
    df: pd.DataFrame,
    explicit_extractor: ExplicitSkillExtractor,
    implicit_extractor: ImplicitSkillExtractor,
    text_column: str = "cleaned_text",
    log_every: int = 100,
) -> pd.DataFrame:
    """Apply skill extraction to every row of a DataFrame."""
    enrichments = []
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows(), 1):
        text = row.get(text_column) or ""
        enrichment = _extract_skills_for_record(text, explicit_extractor, implicit_extractor)
        enrichments.append(enrichment)
        if i % log_every == 0:
            logger.info(f"  Skill extraction: {i}/{total}")

    enrichment_df = pd.DataFrame(enrichments, index=df.index)
    return pd.concat([df, enrichment_df], axis=1)


def run(
    esco_csv_path: Path | None = None,
    programmes_input: Path = PROCESSED_DIR / "programmes" / "programmes_preprocessed.parquet",
    jobs_input: Path = PROCESSED_DIR / "job_ads" / "jobs_preprocessed.parquet",
    programmes_output: Path = PROCESSED_DIR / "programmes" / "programmes_with_skills.parquet",
    jobs_output: Path = PROCESSED_DIR / "job_ads" / "jobs_with_skills.parquet",
) -> None:
    # Load ESCO index
    logger.info("Loading ESCO taxonomy…")
    esco_index: EscoIndex = load_esco_index() if esco_csv_path is None else load_esco_index(esco_csv_path)

    # Build extractors (expensive — done once, reused for both datasets)
    logger.info("Building explicit skill extractor (PhraseMatcher)…")
    explicit_extractor = ExplicitSkillExtractor(esco_index)

    logger.info("Building implicit skill extractor (SentenceTransformer)…")
    implicit_extractor = ImplicitSkillExtractor(esco_index)

    # ── Programmes ─────────────────────────────────────────────────────────────
    if programmes_input.exists():
        logger.info(f"Processing programmes: {programmes_input}")
        df_prog = pd.read_parquet(programmes_input)
        df_prog = process_dataframe(df_prog, explicit_extractor, implicit_extractor, log_every=20)
        df_prog.to_parquet(programmes_output, index=False)
        logger.info(f"Saved → {programmes_output}")
        _log_skill_stats(df_prog, label="Programmes")
    else:
        logger.warning(f"Programmes input not found: {programmes_input}")

    # ── Job ads ────────────────────────────────────────────────────────────────
    if jobs_input.exists():
        logger.info(f"Processing job ads: {jobs_input}")
        df_jobs = pd.read_parquet(jobs_input)
        df_jobs = process_dataframe(df_jobs, explicit_extractor, implicit_extractor, log_every=200)
        df_jobs.to_parquet(jobs_output, index=False)
        logger.info(f"Saved → {jobs_output}")
        _log_skill_stats(df_jobs, label="Job ads")
    else:
        logger.warning(f"Job ads input not found: {jobs_input}")


def _log_skill_stats(df: pd.DataFrame, label: str) -> None:
    if "all_skills" not in df.columns:
        return
    skill_counts = df["all_skills"].apply(len)
    explicit_counts = df["explicit_skills"].apply(len)
    implicit_counts = df["implicit_skills"].apply(len)
    logger.info(
        f"{label} skill stats — "
        f"avg total: {skill_counts.mean():.1f}, "
        f"avg explicit: {explicit_counts.mean():.1f}, "
        f"avg implicit: {implicit_counts.mean():.1f}, "
        f"records with 0 skills: {(skill_counts == 0).sum()}"
    )


if __name__ == "__main__":
    run()
