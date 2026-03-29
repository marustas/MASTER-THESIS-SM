"""
Step 2 — Job advertisement data collection.

Scrapes ICT/AI job postings from CVbankas English site (en.cvbankas.lt).

For each posting the following fields are extracted:
  job_title, company, description, required_skills, employer_sector,
  location, country, employment_type, remote, posting_date, url, source

Temporal filter: postings not older than MAX_POSTING_AGE_DAYS days.

Output:
  data/raw/job_ads/cvbankas_jobs.json
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
    """Scrapes IT job listings from en.cvbankas.lt using scroll-based pagination."""

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
                        const companyEl = article.querySelector(
                            '[class*="company"], [class*="darbdavys"], [class*="employer"]'
                        );
                        const locationEl = article.querySelector(
                            '[class*="city"], [class*="miestas"], [class*="location"]'
                        );
                        return {
                            url: link ? link.href : null,
                            title: link ? link.innerText.trim().split('\\n')[0].trim() : '',
                            company: companyEl ? companyEl.innerText.trim() : null,
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
                    company=card.get("company"),
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
            key = (job.url or f"{job.job_title}|{job.company}").lower()
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
        jobs = await scraper.scrape_all()
    logger.info(f"CVbankas total: {len(jobs)} jobs")
    _save(jobs, output_dir / "cvbankas_jobs.json")
    _save(jobs, output_dir / "all_jobs.json")


def _save(jobs: list[JobAd], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([j.model_dump(mode="json") for j in jobs], f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(jobs)} jobs → {path}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
