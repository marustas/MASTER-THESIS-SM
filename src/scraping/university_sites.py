"""
Step 1b — University website curriculum scraper.

For each programme collected by the LAMA BPO scraper that has a university_url,
this module visits the university's own programme page and extracts:
  - extended curriculum description (learning outcomes, programme overview)
  - list of course/module names with credits and descriptions where available

The extracted text is merged back into the Programme objects as `extended_description`
and optionally as a list of CourseModule entries.

Output:
  data/raw/programmes/lama_bpo_programmes_extended.json   (updated Programme list)
  data/raw/programmes/course_modules.json                 (CourseModule list)
"""

from __future__ import annotations

import json
from pathlib import Path

from bs4 import BeautifulSoup
from loguru import logger
from playwright.async_api import Page

from src.scraping.base import BaseScraper
from src.scraping.config import RAW_PROGRAMMES_DIR
from src.scraping.models import CourseModule, Programme


# CSS / text selectors tried in order for extracting curriculum descriptions.
# Ordered from most-specific (university CMS patterns) to generic fallbacks.
_DESCRIPTION_SELECTORS: list[str] = [
    # Common Lithuanian university CMS patterns
    ".programme-description",
    ".study-programme-description",
    ".programa-aprasymas",
    "[class*='learning-outcomes']",
    "[class*='programme-content']",
    "[class*='curriculum']",
    # Generic article/content containers
    "article .content",
    "main article",
    ".page-content",
    ".entry-content",
    # Very generic fallback
    "main",
]

_COURSE_TABLE_SELECTORS: list[str] = [
    "table.curriculum",
    "table.timetable",
    "table.modules",
    ".course-list table",
    "table",  # last resort
]


class UniversityScraper(BaseScraper):
    """Visits university programme pages and extracts extended curriculum descriptions."""

    async def enrich_programmes(
        self, programmes: list[Programme]
    ) -> tuple[list[Programme], list[CourseModule]]:
        """
        For each programme with a university_url, scrape extended description
        and course modules. Programmes without a URL are returned unchanged.
        """
        all_modules: list[CourseModule] = []

        for i, programme in enumerate(programmes, 1):
            if not programme.university_url:
                logger.debug(f"[{i}/{len(programmes)}] No university URL — skipping: {programme.name}")
                continue

            logger.info(
                f"[{i}/{len(programmes)}] Scraping university site: "
                f"{programme.name} @ {programme.institution}"
            )
            try:
                extended, modules = await self._scrape_programme_page(
                    programme.university_url,
                    programme.name,
                    programme.institution,
                )
                programme.extended_description = extended
                all_modules.extend(modules)
            except Exception as exc:
                logger.warning(
                    f"Failed to scrape {programme.university_url}: {exc}"
                )

        return programmes, all_modules

    # ── Core scraping logic ────────────────────────────────────────────────────

    async def _scrape_programme_page(
        self, url: str, programme_name: str, institution: str
    ) -> tuple[str | None, list[CourseModule]]:
        page = await self.fetch_page(url)
        try:
            html = await page.content()
        finally:
            await page.close()

        soup = BeautifulSoup(html, "lxml")
        self._remove_boilerplate(soup)

        description = self._extract_description(soup)
        modules = self._extract_course_modules(soup, programme_name, institution)

        return description, modules

    # ── Description extraction ─────────────────────────────────────────────────

    @staticmethod
    def _extract_description(soup: BeautifulSoup) -> str | None:
        for selector in _DESCRIPTION_SELECTORS:
            el = soup.select_one(selector)
            if el:
                text = el.get_text(separator=" ", strip=True)
                if len(text) > 100:
                    return text
        return None

    # ── Course module extraction ───────────────────────────────────────────────

    @staticmethod
    def _extract_course_modules(
        soup: BeautifulSoup, programme_name: str, institution: str
    ) -> list[CourseModule]:
        modules: list[CourseModule] = []

        for selector in _COURSE_TABLE_SELECTORS:
            tables = soup.select(selector)
            for table in tables:
                rows = table.select("tr")
                if len(rows) < 2:
                    continue

                # Try to detect header row to identify column positions
                headers = [th.get_text(strip=True).lower() for th in rows[0].select("th, td")]
                name_col = _find_col(headers, ["pavadinimas", "modulis", "dalykas", "subject", "name", "module"])
                credits_col = _find_col(headers, ["kreditai", "ects", "credits", "hp"])
                desc_col = _find_col(headers, ["aprašymas", "description", "tikslai", "outcomes"])

                for row in rows[1:]:
                    cells = row.select("td")
                    if not cells:
                        continue

                    name = cells[name_col].get_text(strip=True) if name_col is not None and name_col < len(cells) else cells[0].get_text(strip=True)
                    if not name or len(name) < 3:
                        continue

                    credits_text = cells[credits_col].get_text(strip=True) if credits_col is not None and credits_col < len(cells) else None
                    credits = _parse_int(credits_text)

                    description = cells[desc_col].get_text(strip=True) if desc_col is not None and desc_col < len(cells) else None

                    modules.append(CourseModule(
                        programme_name=programme_name,
                        institution=institution,
                        module_name=name,
                        credits_ects=credits,
                        description=description if description and len(description) > 10 else None,
                    ))

                if modules:
                    return modules  # stop after first productive table

        return modules

    # ── HTML cleanup ───────────────────────────────────────────────────────────

    @staticmethod
    def _remove_boilerplate(soup: BeautifulSoup) -> None:
        for tag in soup.select("header, footer, nav, script, style, .cookie-banner, .menu"):
            tag.decompose()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_col(headers: list[str], keywords: list[str]) -> int | None:
    for keyword in keywords:
        for i, h in enumerate(headers):
            if keyword in h:
                return i
    return None


def _parse_int(text: str | None) -> int | None:
    if not text:
        return None
    digits = "".join(c for c in text if c.isdigit())
    return int(digits) if digits else None


# ── CLI entry-point ────────────────────────────────────────────────────────────

async def run(
    input_path: Path = RAW_PROGRAMMES_DIR / "lama_bpo_programmes.json",
    output_path: Path = RAW_PROGRAMMES_DIR / "lama_bpo_programmes_extended.json",
    modules_path: Path = RAW_PROGRAMMES_DIR / "course_modules.json",
) -> None:
    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)
    programmes = [Programme(**p) for p in raw]

    async with UniversityScraper() as scraper:
        programmes, modules = await scraper.enrich_programmes(programmes)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([p.model_dump(mode="json") for p in programmes], f, ensure_ascii=False, indent=2)

    with open(modules_path, "w", encoding="utf-8") as f:
        json.dump([m.model_dump(mode="json") for m in modules], f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(programmes)} programmes → {output_path}")
    logger.info(f"Saved {len(modules)} course modules → {modules_path}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
