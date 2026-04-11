"""
Auxiliary LinkedIn job scraper for implicit skill extraction corpus.

Scrapes ~2000 EU-wide IT/CS job postings across all time periods.
These jobs are used ONLY to enrich the implicit skill extractor's
neighbour corpus — they never enter the alignment dataset.

Same search queries as the main LinkedIn scraper but with:
  - 8 EU country locations (not just Lithuania)
  - TimeFilters.ANY (all time, not just past month)
  - 12 queries × 8 locations = 96 combinations, 25 per query ≈ 2400 max

Output:
  data/raw/job_ads/auxiliary_jobs.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from loguru import logger

from src.scraping.config import RAW_JOB_ADS_DIR
from src.scraping.job_ads import _extract_skills_from_text
from src.scraping.models import JobAd

from linkedin_jobs_scraper import LinkedinScraper
from linkedin_jobs_scraper.events import Events, EventData
from linkedin_jobs_scraper.filters import (
    ExperienceLevelFilters,
    IndustryFilters,
    TimeFilters,
    TypeFilters,
)
from linkedin_jobs_scraper.query import Query, QueryFilters, QueryOptions

# ── Configuration ────────────────────────────────────────────────────────────

LI_AT_COOKIE = os.getenv("LI_AT_COOKIE", "")

AUXILIARY_QUERIES: list[str] = [
    "Software Engineer",
    "Software Developer",
    "Data Engineer",
    "Data Scientist",
    "Machine Learning",
    "DevOps Engineer",
    "Backend Developer",
    "Frontend Developer",
    "Full Stack Developer",
    "Cybersecurity",
    "IT Specialist",
    "Systems Administrator",
]

# Specific EU countries — LinkedIn doesn't recognise "European Union" as a location
AUXILIARY_LOCATIONS: list[str] = [
    "Germany",
    "Netherlands",
    "Poland",
    "France",
    "Spain",
    "Ireland",
    "Sweden",
    "Lithuania",
]

AUXILIARY_INDUSTRIES: list[IndustryFilters] = [
    IndustryFilters.SOFTWARE_DEVELOPMENT,
    IndustryFilters.TECHNOLOGY_INTERNET,
    IndustryFilters.IT_SERVICES,
    IndustryFilters.COMPUTER_GAMES
]

# Keep per-query limit low to avoid timeouts; we generate
# len(queries) × len(locations) = 12 × 8 = 96 query combinations
AUXILIARY_LIMIT_PER_QUERY: int = 25


# ── Scraper ──────────────────────────────────────────────────────────────────

def scrape_auxiliary_linkedin(
    output_path: Path = RAW_JOB_ADS_DIR / "auxiliary_jobs.json",
) -> list[JobAd]:
    """Scrape EU-wide LinkedIn IT jobs for auxiliary implicit extraction corpus."""
    if not LI_AT_COOKIE:
        raise RuntimeError(
            "LI_AT_COOKIE environment variable is not set. "
            "Extract the li_at cookie from LinkedIn in your browser DevTools."
        )

    collected: list[JobAd] = []
    seen_ids: set[str] = set()

    def on_data(data: EventData) -> None:
        job_id = data.job_id or ""
        if job_id in seen_ids:
            return
        seen_ids.add(job_id)

        description = data.description or ""
        remote = "remote" in description.lower()

        job = JobAd(
            job_title=data.title or "",
            description=description or None,
            required_skills=_extract_skills_from_text(description),
            employer_sector=None,
            location=data.place or None,
            country=None,
            employment_type=None,
            remote=remote,
            posting_date=data.date or None,
            url=data.link or None,
            source="linkedin_auxiliary",
        )
        collected.append(job)
        if len(collected) % 100 == 0:
            logger.info(f"Auxiliary LinkedIn: {len(collected)} jobs collected so far")

    def on_error(error: Exception) -> None:
        logger.warning(f"LinkedIn auxiliary scraper error: {error}")

    def on_end() -> None:
        logger.info(f"LinkedIn auxiliary scraping finished — {len(collected)} jobs collected")

    scraper = LinkedinScraper(
        chrome_executable_path=None,
        headless=True,
        max_workers=1,
        slow_mo=2.5,
        page_load_timeout=60,
    )

    scraper.on(Events.DATA, on_data)
    scraper.on(Events.ERROR, on_error)
    scraper.on(Events.END, on_end)

    # One Query per (keyword, location) pair for better coverage and
    # to keep each query short enough to avoid LinkedIn timeouts.
    queries = [
        Query(
            query=keyword,
            options=QueryOptions(
                locations=[location],
                limit=AUXILIARY_LIMIT_PER_QUERY,
                filters=QueryFilters(
                    time=TimeFilters.ANY,
                    type=[TypeFilters.FULL_TIME],
                    experience=[
                        ExperienceLevelFilters.ENTRY_LEVEL,
                        ExperienceLevelFilters.ASSOCIATE,
                        ExperienceLevelFilters.MID_SENIOR,
                    ],
                    industry=AUXILIARY_INDUSTRIES,
                ),
            ),
        )
        for keyword in AUXILIARY_QUERIES
        for location in AUXILIARY_LOCATIONS
    ]

    logger.info(
        f"Starting auxiliary LinkedIn scraper: {len(queries)} queries "
        f"({len(AUXILIARY_QUERIES)} keywords × {len(AUXILIARY_LOCATIONS)} locations), "
        f"limit {AUXILIARY_LIMIT_PER_QUERY} per query, all time"
    )
    scraper.run(queries)

    # Deduplicate by title (lowercase)
    unique: list[JobAd] = []
    seen_titles: set[str] = set()
    for job in collected:
        key = job.job_title.strip().lower()
        if key and key not in seen_titles:
            seen_titles.add(key)
            unique.append(job)

    logger.info(f"After dedup: {len(unique)} unique auxiliary jobs (from {len(collected)} total)")

    # Persist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            [j.model_dump(mode="json") for j in unique],
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"Saved {len(unique)} auxiliary LinkedIn jobs → {output_path}")

    return unique


# ── CLI entry-point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    scrape_auxiliary_linkedin()
