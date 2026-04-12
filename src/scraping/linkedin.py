"""
Step 2b — LinkedIn job advertisement data collection.

Scrapes IT/CS job postings from LinkedIn using py-linkedin-jobs-scraper.

Industry filters: SOFTWARE_DEVELOPMENT, TECHNOLOGY_INTERNET, IT_SERVICES.
Location: Lithuania (to match CVbankas scope).

Requires the ``LI_AT_COOKIE`` environment variable (extract the ``li_at``
cookie value from browser DevTools → Application → Cookies → linkedin.com).

Output:
  data/raw/job_ads/linkedin_jobs.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from loguru import logger

from src.scraping.config import RAW_JOB_ADS_DIR
from src.scraping.job_ads import _deduplicate_by_identity, _extract_skills_from_text, _load_jobs
from src.scraping.models import JobAd

# ── LinkedIn scraper imports ─────────────────────────────────────────────────

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

LINKEDIN_QUERIES: list[str] = [
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

LINKEDIN_LOCATIONS: list[str] = ["Lithuania"]

LINKEDIN_INDUSTRIES: list[IndustryFilters] = [
    IndustryFilters.SOFTWARE_DEVELOPMENT,
    IndustryFilters.TECHNOLOGY_INTERNET,
    IndustryFilters.IT_SERVICES,
    IndustryFilters.COMPUTER_GAMES
]

LINKEDIN_LIMIT_PER_QUERY: int = 50


# ── Scraper ──────────────────────────────────────────────────────────────────

def scrape_linkedin(
    output_path: Path = RAW_JOB_ADS_DIR / "linkedin_jobs.json",
) -> list[JobAd]:
    """Run LinkedIn job scraper and return collected JobAd objects."""
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
            country="Lithuania",
            employment_type=None,
            remote=remote,
            posting_date=data.date or None,
            url=data.link or None,
            source="linkedin",
        )
        collected.append(job)
        logger.debug(f"LinkedIn job #{len(collected)}: {job.job_title}")

    def on_error(error: Exception) -> None:
        logger.warning(f"LinkedIn scraper error: {error}")

    def on_end() -> None:
        logger.info(f"LinkedIn scraping finished — {len(collected)} jobs collected")

    scraper = LinkedinScraper(
        chrome_executable_path=None,
        headless=True,
        max_workers=1,
        slow_mo=1.5,
        page_load_timeout=40,
    )

    scraper.on(Events.DATA, on_data)
    scraper.on(Events.ERROR, on_error)
    scraper.on(Events.END, on_end)

    queries = [
        Query(
            query=keyword,
            options=QueryOptions(
                locations=LINKEDIN_LOCATIONS,
                limit=LINKEDIN_LIMIT_PER_QUERY,
                filters=QueryFilters(
                    time=TimeFilters.MONTH,
                    type=[TypeFilters.FULL_TIME],
                    experience=[
                        ExperienceLevelFilters.ENTRY_LEVEL,
                        ExperienceLevelFilters.ASSOCIATE,
                        ExperienceLevelFilters.MID_SENIOR,
                    ],
                    industry=LINKEDIN_INDUSTRIES,
                ),
            ),
        )
        for keyword in LINKEDIN_QUERIES
    ]

    logger.info(
        f"Starting LinkedIn scraper: {len(queries)} queries × "
        f"limit {LINKEDIN_LIMIT_PER_QUERY} per query"
    )
    scraper.run(queries)

    existing = _load_jobs(output_path)
    unique = _deduplicate_by_identity([*existing, *collected])
    logger.info(
        f"Collected {len(collected)} LinkedIn jobs; merged with existing {len(existing)} "
        f"-> {len(unique)} total"
    )

    # Persist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            [j.model_dump(mode="json") for j in unique],
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"Saved {len(unique)} LinkedIn jobs → {output_path}")

    return unique


# ── CLI entry-point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    scrape_linkedin()
