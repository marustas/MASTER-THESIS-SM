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
    auxiliary_texts: list[str] | None = None,
    auxiliary_explicit: list[list[ExtractedSkill]] | None = None,
) -> pd.DataFrame:
    """
    Apply skill extraction to every row of a DataFrame.

    Step A — Run explicit extraction on all documents (row by row).
    Step B — Fit the implicit extractor on the full corpus (main + auxiliary).
    Step C — Extract implicit skills per document using corpus neighbours.

    Parameters
    ----------
    auxiliary_texts : optional list of cleaned texts from auxiliary corpus.
        Used to enlarge the neighbour pool when fitting the implicit extractor.
        These documents are NOT included in the output.
    auxiliary_explicit : optional pre-computed explicit skills for auxiliary docs.
        Must match auxiliary_texts in length.
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

    # ── Step B: Fit implicit extractor on corpus (+ auxiliary if provided) ─────
    if auxiliary_texts and auxiliary_explicit:
        n_aux = len(auxiliary_texts)
        logger.info(
            f"Step B: fitting implicit extractor on {total} main + {n_aux} auxiliary "
            f"= {total + n_aux} documents…"
        )
        fit_texts = texts + auxiliary_texts
        fit_explicit = all_explicit + auxiliary_explicit
    else:
        logger.info(f"Step B: fitting implicit extractor on {total} documents…")
        fit_texts = texts
        fit_explicit = all_explicit

    implicit_extractor.fit(fit_texts, explicit_skills_per_doc=fit_explicit)

    # ── Step C: Implicit extraction per document (main corpus only) ───────────
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


def _load_auxiliary_corpus(
    auxiliary_path: Path,
    explicit_extractor: ExplicitSkillExtractor,
    text_column: str = "cleaned_text",
    log_every: int = 200,
) -> tuple[list[str], list[list[ExtractedSkill]]]:
    """Load auxiliary corpus and run explicit extraction on it."""
    logger.info(f"Loading auxiliary corpus from {auxiliary_path}…")
    df_aux = pd.read_parquet(auxiliary_path)
    aux_texts = df_aux[text_column].fillna("").tolist()
    n = len(aux_texts)
    logger.info(f"Auxiliary corpus: {n} documents — running explicit extraction…")

    aux_explicit: list[list[ExtractedSkill]] = []
    for i, text in enumerate(aux_texts, 1):
        skills = explicit_extractor.extract(text)
        aux_explicit.append(skills)
        if i % log_every == 0:
            logger.info(f"  Auxiliary explicit: {i}/{n}")

    avg_skills = sum(len(s) for s in aux_explicit) / max(n, 1)
    logger.info(f"Auxiliary explicit extraction done — avg {avg_skills:.1f} skills per doc")
    return aux_texts, aux_explicit


def run(
    esco_csv_path: Path | None = None,
    programmes_input: Path = PROCESSED_DIR / "programmes" / "programmes_preprocessed.parquet",
    jobs_input: Path = PROCESSED_DIR / "job_ads" / "jobs_preprocessed.parquet",
    programmes_output: Path = PROCESSED_DIR / "programmes" / "programmes_with_skills.parquet",
    jobs_output: Path = PROCESSED_DIR / "job_ads" / "jobs_with_skills.parquet",
    auxiliary_input: Path = PROCESSED_DIR / "job_ads" / "auxiliary_preprocessed.parquet",
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

    # ── Load auxiliary corpus if available ───────────────────────────────────────
    aux_texts: list[str] | None = None
    aux_explicit: list[list[ExtractedSkill]] | None = None
    if auxiliary_input.exists():
        aux_texts, aux_explicit = _load_auxiliary_corpus(
            auxiliary_input, explicit_extractor,
        )
    else:
        logger.info("No auxiliary corpus found — implicit extraction uses main corpus only")

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

    # ── Job ads (with auxiliary corpus for implicit fitting) ────────────────────
    if jobs_input.exists():
        logger.info(f"Processing job ads: {jobs_input}")
        df_jobs = pd.read_parquet(jobs_input)
        implicit_extractor_jobs = ImplicitSkillExtractor(explicit_extractor)
        df_jobs = process_dataframe(
            df_jobs, explicit_extractor, implicit_extractor_jobs,
            log_every=200,
            auxiliary_texts=aux_texts,
            auxiliary_explicit=aux_explicit,
        )
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
