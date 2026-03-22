"""
Step 4 — Skill extraction & ESCO ontology mapping coordinator.

Orchestrates explicit + implicit extraction per Gugnani & Misra (2020):
  1. Run explicit extraction (4-module ensemble) on every document.
  2. Fit the implicit extractor on the full corpus (builds document embeddings).
  3. For each document, find top-K similar documents in the corpus and propagate
     skills that are absent from the target → implicit skills.

Enriches each record with:
  - explicit_skills:  list of ESCO preferred labels (explicitly mentioned)
  - implicit_skills:  list of ESCO preferred labels (inferred from similar docs)
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

from dataclasses import asdict
from pathlib import Path

import pandas as pd
from loguru import logger

from src.skills.esco_loader import EscoIndex, load_esco_index
from src.skills.explicit_extractor import ExplicitSkillExtractor, ExtractedSkill
from src.skills.implicit_extractor import ImplicitSkillExtractor
from src.scraping.config import DATA_DIR

PROCESSED_DIR = DATA_DIR / "processed"


def process_dataframe(
    df: pd.DataFrame,
    explicit_extractor: ExplicitSkillExtractor,
    implicit_extractor: ImplicitSkillExtractor,
    text_column: str = "cleaned_text",
    log_every: int = 100,
) -> pd.DataFrame:
    """
    Apply skill extraction to every row of a DataFrame.

    Step A — Run explicit extraction on all documents (row by row).
    Step B — Fit the implicit extractor on the full corpus.
    Step C — Extract implicit skills per document using corpus neighbours.
    """
    texts = df[text_column].fillna("").tolist()
    total = len(texts)

    # ── Step A: Explicit extraction ─────────────────────────────────────────────
    logger.info(f"Step A: explicit extraction on {total} documents…")
    all_explicit: list[list[ExtractedSkill]] = []
    for i, text in enumerate(texts, 1):
        skills = explicit_extractor.extract(text)
        all_explicit.append(skills)
        if i % log_every == 0:
            logger.info(f"  Explicit: {i}/{total}")

    # ── Step B: Fit implicit extractor on corpus ─────────────────────────────────
    logger.info("Step B: fitting implicit extractor on corpus embeddings…")
    implicit_extractor.fit(texts, explicit_skills_per_doc=all_explicit)

    # ── Step C: Implicit extraction per document ─────────────────────────────────
    logger.info("Step C: extracting implicit skills via similar-document propagation…")
    all_implicit = implicit_extractor.extract_batch(texts, all_explicit)

    # ── Assemble enrichment columns ──────────────────────────────────────────────
    enrichments = []
    for explicit, implicit in zip(all_explicit, all_implicit):
        combined: list[ExtractedSkill] = explicit + implicit
        enrichments.append({
            "explicit_skills": [s.preferred_label for s in explicit],
            "implicit_skills": [s.preferred_label for s in implicit],
            "all_skills": [s.preferred_label for s in combined],
            "skill_uris": [s.esco_uri for s in combined],
            "skill_details": [asdict(s) for s in combined],
        })

    enrichment_df = pd.DataFrame(enrichments, index=df.index)
    result = pd.concat([df, enrichment_df], axis=1)
    _log_skill_stats(result, label=f"Dataset ({total} records)")
    return result


def run(
    esco_csv_path: Path | None = None,
    programmes_input: Path = PROCESSED_DIR / "programmes" / "programmes_preprocessed.parquet",
    jobs_input: Path = PROCESSED_DIR / "job_ads" / "jobs_preprocessed.parquet",
    programmes_output: Path = PROCESSED_DIR / "programmes" / "programmes_with_skills.parquet",
    jobs_output: Path = PROCESSED_DIR / "job_ads" / "jobs_with_skills.parquet",
) -> None:
    logger.info("Loading ESCO taxonomy…")
    esco_index: EscoIndex = (
        load_esco_index() if esco_csv_path is None else load_esco_index(esco_csv_path)
    )

    # Build extractors once — reused for both datasets
    logger.info("Building explicit skill extractor (4-module ensemble)…")
    explicit_extractor = ExplicitSkillExtractor(esco_index)

    # Implicit extractor is stateful (fit per dataset)
    logger.info("Initialising implicit skill extractor…")

    # ── Programmes ──────────────────────────────────────────────────────────────
    if programmes_input.exists():
        logger.info(f"Processing programmes: {programmes_input}")
        df_prog = pd.read_parquet(programmes_input)
        implicit_extractor_prog = ImplicitSkillExtractor(explicit_extractor)
        df_prog = process_dataframe(df_prog, explicit_extractor, implicit_extractor_prog, log_every=20)
        df_prog.to_parquet(programmes_output, index=False)
        logger.info(f"Saved → {programmes_output}")
    else:
        logger.warning(f"Programmes input not found: {programmes_input}")

    # ── Job ads ─────────────────────────────────────────────────────────────────
    if jobs_input.exists():
        logger.info(f"Processing job ads: {jobs_input}")
        df_jobs = pd.read_parquet(jobs_input)
        implicit_extractor_jobs = ImplicitSkillExtractor(explicit_extractor)
        df_jobs = process_dataframe(df_jobs, explicit_extractor, implicit_extractor_jobs, log_every=200)
        df_jobs.to_parquet(jobs_output, index=False)
        logger.info(f"Saved → {jobs_output}")
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
        f"avg total: {skill_counts.mean():.1f} | "
        f"avg explicit: {explicit_counts.mean():.1f} | "
        f"avg implicit: {implicit_counts.mean():.1f} | "
        f"zero-skill records: {(skill_counts == 0).sum()}"
    )


if __name__ == "__main__":
    run()
