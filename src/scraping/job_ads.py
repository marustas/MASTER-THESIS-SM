"""
Step 2 — Job advertisement data collection.

Scrapes ICT/AI job postings from CVbankas English site (en.cvbankas.lt).

For each posting the following fields are extracted:
  job_title, description, required_skills, employer_sector,
  location, country, employment_type, remote, posting_date, url, source

Temporal filter: postings not older than MAX_POSTING_AGE_DAYS days.

Output:
    data/raw/job_ads/cvbankas_jobs.json
    data/raw/job_ads/linkedin_jobs.json
    data/raw/job_ads/all_jobs.json
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

from bs4 import BeautifulSoup
from loguru import logger

from src.scraping.base import BaseScraper
from src.scraping.config import (
    CVBANKAS_BASE_URL,
    CVBANKAS_IT_CATEGORIES,
    CVBANKAS_MAX_PAGES,
    MAX_POSTING_AGE_DAYS,
    RAW_JOB_ADS_DIR,
)
from src.scraping.models import JobAd


# ── CVbankas scraper ───────────────────────────────────────────────────────────

class CVbankasJobScraper(BaseScraper):
    """Scrapes IT job listings from en.cvbankas.lt using pagination."""

    SOURCE = "cvbankas"

    async def scrape_all(self) -> list[JobAd]:
        jobs: list[JobAd] = []
        for category_id in CVBANKAS_IT_CATEGORIES:
            category_jobs = await self._scrape_category(category_id)
            jobs.extend(category_jobs)
            logger.info(f"CVbankas category {category_id}: {len(category_jobs)} jobs collected")
        return self._deduplicate(jobs)

    async def _scrape_category(self, category_id: int) -> list[JobAd]:
        jobs: list[JobAd] = []
        cutoff = datetime.utcnow() - timedelta(days=MAX_POSTING_AGE_DAYS)

        for page_num in range(1, CVBANKAS_MAX_PAGES + 1):
            url = f"{CVBANKAS_BASE_URL}/?padalinys[]={category_id}&page={page_num}"
            logger.info(f"CVbankas page {page_num}: {url}")

            try:
                page = await self.fetch_page(url)
                cards: list[dict] = await page.evaluate("""
                    () => Array.from(document.querySelectorAll('article[id^="job_ad_"]')).map(article => {
                        const link = article.querySelector('a.list_a');
                        const timeEl = article.querySelector('time');
                        const locationEl = article.querySelector(
                            '[class*="city"], [class*="miestas"], [class*="location"]'
                        );
                        return {
                            url: link ? link.href : null,
                            title: link ? link.innerText.trim().split('\\n')[0].trim() : '',
                            location: locationEl ? locationEl.innerText.trim() : null,
                            posting_date: timeEl
                                ? (timeEl.getAttribute('datetime') || timeEl.innerText.trim())
                                : null,
                        };
                    }).filter(c => c.url && c.url.includes('cvbankas') && c.title.length > 0)
                """)
                await page.close()
            except Exception as exc:
                logger.warning(f"Failed to fetch page {page_num}: {exc}")
                break

            if not cards:
                logger.info(f"No cards on page {page_num} — stopping")
                break

            too_old_count = 0
            for card in cards:
                if card.get("posting_date") and self._is_too_old(card["posting_date"], cutoff):
                    too_old_count += 1
                    continue
                detail = await self._scrape_detail(card["url"])
                jobs.append(JobAd(
                    job_title=card.get("title") or detail.get("title", ""),
                    description=detail.get("description"),
                    required_skills=detail.get("skills", []),
                    employer_sector=detail.get("sector"),
                    location=card.get("location"),
                    country="Lithuania",
                    employment_type=detail.get("employment_type"),
                    remote=detail.get("remote"),
                    posting_date=card.get("posting_date"),
                    url=card["url"],
                    source=self.SOURCE,
                ))

            if too_old_count > len(cards) * 0.5:
                logger.info("Majority of cards are too old — stopping")
                break

        return jobs

    async def _scrape_detail(self, url: str) -> dict:
        if not url:
            return {}
        try:
            page = await self.fetch_page(url)
            html = await page.content()
            await page.close()
        except Exception as exc:
            logger.debug(f"Detail page failed {url}: {exc}")
            return {}

        soup = BeautifulSoup(html, "lxml")
        for tag in soup.select("header, footer, nav, script, style"):
            tag.decompose()

        # Title fallback from detail page
        title_el = soup.select_one("h1")
        title = title_el.get_text(strip=True) if title_el else ""

        description = None
        for sel in [
            "[id*='jobad_desc']",
            "[class*='jobad-description']",
            "[class*='jobad_description']",
            "[class*='job_description']",
            "article section",
            "article",
            "main",
        ]:
            el = soup.select_one(sel)
            if el:
                text = el.get_text(separator=" ", strip=True)
                if len(text) > 100:
                    description = text
                    break

        skills = _extract_skills_from_text(description or "")
        sector_el = soup.select_one("[class*='sector']") or soup.select_one("[class*='sritis']")
        employment_el = (
            soup.select_one("[class*='employment_type']")
            or soup.select_one("[class*='darbo_tipas']")
        )
        remote = "remote" in (description or "").lower() or "nuotolinis" in (description or "").lower()

        return {
            "title": title,
            "description": description,
            "skills": skills,
            "sector": sector_el.get_text(strip=True) if sector_el else None,
            "employment_type": employment_el.get_text(strip=True) if employment_el else None,
            "remote": remote,
        }

    @staticmethod
    def _is_too_old(date_str: str, cutoff: datetime) -> bool:
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%d.%m.%Y", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                dt = datetime.strptime(date_str[:len(fmt)], fmt)
                return dt < cutoff
            except ValueError:
                continue
        return False

    @staticmethod
    def _deduplicate(jobs: list[JobAd]) -> list[JobAd]:
        seen: set[str] = set()
        unique = []
        for job in jobs:
            key = (job.url or job.job_title).lower()
            if key not in seen:
                seen.add(key)
                unique.append(job)
        return unique


# ── Shared utilities ───────────────────────────────────────────────────────────

# Curated list of common ICT/AI skills to extract from free text.
# Will be replaced by ESCO-based extraction in Step 4.
_SKILL_PATTERNS: list[str] = [
    "python", "java", "javascript", "typescript", "c\\+\\+", "c#", "go", "rust", "scala",
    "sql", "nosql", "postgresql", "mysql", "mongodb", "redis",
    "docker", "kubernetes", "aws", "azure", "gcp", "terraform", "ci/cd", "git",
    "machine learning", "deep learning", "nlp", "computer vision", "pytorch", "tensorflow",
    "scikit-learn", "hugging face", "transformers", "llm", "rag",
    "react", "angular", "vue", "node\\.js", "django", "fastapi", "spring",
    "linux", "bash", "rest api", "graphql", "microservices",
    "data engineering", "spark", "kafka", "airflow", "dbt",
    "agile", "scrum", "jira",
]
_SKILL_REGEX = re.compile(
    r"\b(" + "|".join(_SKILL_PATTERNS) + r")\b",
    re.IGNORECASE,
)


def _extract_skills_from_text(text: str) -> list[str]:
    """Crude regex skill extraction — refined in Step 4 (ESCO mapping)."""
    if not text:
        return []
    return sorted(set(m.group(0).lower() for m in _SKILL_REGEX.finditer(text)))


# ── CLI entry-point ────────────────────────────────────────────────────────────

async def run(output_dir: Path = RAW_JOB_ADS_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    async with CVbankasJobScraper() as scraper:
        jobs_new = await scraper.scrape_all()

    cvbankas_path = output_dir / "cvbankas_jobs.json"
    jobs_existing = _load_jobs(cvbankas_path)
    jobs = _deduplicate_by_identity([*jobs_new, *jobs_existing])
    logger.info(
        f"CVbankas mined {len(jobs_new)} jobs; merged with existing {len(jobs_existing)} "
        f"-> {len(jobs)} total"
    )
    _save(jobs, cvbankas_path)

    try:
        from src.scraping.linkedin import scrape_linkedin

        scrape_linkedin(output_path=output_dir / "linkedin_jobs.json")
    except RuntimeError as exc:
        logger.info(f"LinkedIn scrape skipped: {exc}")

    _merge_all_jobs(output_dir)


_NON_IT_TITLE_PATTERNS: list[str] = [
    "teacher", "teaching", "ppc specialist", "seo specialist",
    "account executive", "lead generation specialist",
    "head of treasury", "communications projects manager",
]


def _is_non_it(job: JobAd) -> bool:
    """Return True if the job title matches a known non-IT pattern."""
    title = job.job_title.strip().lower()
    return any(pat in title for pat in _NON_IT_TITLE_PATTERNS)


def _merge_all_jobs(output_dir: Path) -> None:
    """Merge CVbankas + LinkedIn jobs into all_jobs.json without title collapse."""
    import json as _json

    all_jobs: list[JobAd] = []

    cvbankas_path = output_dir / "cvbankas_jobs.json"
    all_jobs.extend(_load_jobs(cvbankas_path))

    linkedin_path = output_dir / "linkedin_jobs.json"
    all_jobs.extend(_load_jobs(linkedin_path))

    # Filter non-IT jobs
    before = len(all_jobs)
    all_jobs = [j for j in all_jobs if not _is_non_it(j)]
    removed = before - len(all_jobs)
    if removed:
        logger.info(f"Removed {removed} non-IT jobs by title filter")

    # Deduplicate only exact same posting identities.
    # Title-only dedup removes many legitimate jobs (same title, different company/post).
    unique = _deduplicate_by_identity(all_jobs)

    logger.info(
        f"Merged {before} total jobs → {len(unique)} unique "
        f"(CVbankas + LinkedIn, after dedup + filter)"
    )
    _save(unique, output_dir / "all_jobs.json")


def _save(jobs: list[JobAd], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([j.model_dump(mode="json") for j in jobs], f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(jobs)} jobs → {path}")


def _load_jobs(path: Path) -> list[JobAd]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [JobAd(**j) for j in json.load(f)]


def _deduplicate_by_identity(jobs: list[JobAd]) -> list[JobAd]:
    seen: set[str] = set()
    unique: list[JobAd] = []
    for job in jobs:
        key = (job.url or "").strip().lower()
        if not key:
            key = "|".join([
                (job.source or "").strip().lower(),
                (job.job_title or "").strip().lower(),
                (job.location or "").strip().lower(),
                (job.posting_date or "").strip().lower(),
            ])
        if key in seen:
            continue
        seen.add(key)
        unique.append(job)
    return unique


if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
