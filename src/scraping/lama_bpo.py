"""
Step 1a — LAMA BPO programme listing scraper.

Scrapes Bachelor-level CS/ICT/AI programmes from the LAMA BPO registry:
  https://www.lamabpo.lt/pirmosios-pakopos-ir-vientisosios-studijos/programu-sarasas/

For each programme it captures:
  - name, institution, city, field group, field, study mode
  - brief description from the registry detail page
  - link to the university's own programme page (for Step 1b)

Output: data/raw/programmes/lama_bpo_programmes.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger
from playwright.async_api import Page

from src.scraping.base import BaseScraper
from src.scraping.config import (
    LAMA_BPO_BASE_URL,
    LAMA_BPO_PROGRAMMES_URL,
    RAW_PROGRAMMES_DIR,
    REQUEST_DELAY,
    TARGET_FIELD_GROUPS,
    TARGET_FIELDS,
)
from src.scraping.models import Programme


class LamaBpoScraper(BaseScraper):
    """Scrapes the LAMA BPO programme listing and detail pages."""

    # ── Public API ─────────────────────────────────────────────────────────────

    async def scrape_all(self) -> list[Programme]:
        """
        Full scraping run:
          1. Load the listing page and read available filter options.
          2. Identify CS/ICT/AI relevant field groups and fields.
          3. For each matching filter combination, collect programme card data.
          4. Deduplicate and enrich with detail page data.
        Returns a list of Programme objects.
        """
        logger.info("Starting LAMA BPO scrape")
        page = await self.fetch_page(LAMA_BPO_PROGRAMMES_URL)

        filter_options = await self._read_filter_options(page)
        logger.info(
            f"Filter options found — field_groups: {len(filter_options['field_groups'])}, "
            f"fields: {len(filter_options['fields'])}"
        )

        target_field_groups = self._match_targets(
            filter_options["field_groups"], TARGET_FIELD_GROUPS
        )
        target_fields = self._match_targets(filter_options["fields"], TARGET_FIELDS)
        logger.info(
            f"Matched targets — field_groups: {target_field_groups}, fields: {target_fields}"
        )

        raw_entries = await self._collect_programme_cards(
            page, target_field_groups, target_fields
        )
        await page.close()
        logger.info(f"Collected {len(raw_entries)} raw programme entries")

        programmes = await self._enrich_with_detail_pages(raw_entries)
        programmes = self._deduplicate(programmes)
        logger.info(f"Final programme count after dedup: {len(programmes)}")
        return programmes

    # ── Filter helpers ─────────────────────────────────────────────────────────

    async def _read_filter_options(self, page: Page) -> dict[str, list[str]]:
        """Extract all available option values from the filter dropdowns."""
        await page.wait_for_load_state("networkidle")

        field_groups: list[str] = await page.evaluate("""
            () => {
                const sel = document.querySelector('select[name*="krypcių"], select[name*="krypciu"], select[id*="krypciu"], select[id*="kryp"]');
                if (!sel) return [];
                return Array.from(sel.options)
                    .map(o => o.text.trim())
                    .filter(t => t.length > 0 && t !== '-');
            }
        """)

        fields: list[str] = await page.evaluate("""
            () => {
                const selects = Array.from(document.querySelectorAll('select'));
                for (const sel of selects) {
                    const name = (sel.name || sel.id || '').toLowerCase();
                    if (name.includes('krypt') && name.includes('is')) {
                        return Array.from(sel.options)
                            .map(o => o.text.trim())
                            .filter(t => t.length > 0 && t !== '-');
                    }
                }
                return [];
            }
        """)

        # Fallback: dump all select options grouped by select index
        if not field_groups and not fields:
            all_selects: list[dict] = await page.evaluate("""
                () => Array.from(document.querySelectorAll('select')).map((sel, i) => ({
                    index: i,
                    name: sel.name || sel.id || '',
                    options: Array.from(sel.options).map(o => o.text.trim()).filter(t => t)
                }))
            """)
            logger.debug(f"All selects on page: {json.dumps(all_selects, ensure_ascii=False)}")

        return {"field_groups": field_groups, "fields": fields}

    @staticmethod
    def _match_targets(available: list[str], targets: list[str]) -> list[str]:
        """Return items from `available` that contain any keyword from `targets`."""
        matched = []
        for option in available:
            for target in targets:
                if target.lower() in option.lower():
                    matched.append(option)
                    break
        return matched if matched else available  # if no match, try all

    # ── Card collection ────────────────────────────────────────────────────────

    async def _collect_programme_cards(
        self,
        page: Page,
        field_groups: list[str],
        fields: list[str],
    ) -> list[dict]:
        """
        Iterate over filter combinations and scrape programme cards from the table.
        Falls back to scraping without filters if filtering produces no results.
        """
        entries: dict[str, dict] = {}

        filter_values = fields if fields else field_groups
        for value in filter_values:
            batch = await self._apply_filter_and_scrape(page, value)
            for entry in batch:
                key = f"{entry.get('name', '')}|{entry.get('institution', '')}"
                entries[key] = entry

        # Fallback: scrape everything without filter
        if not entries:
            logger.warning("No entries with filters — scraping without filter")
            batch = await self._apply_filter_and_scrape(page, filter_value=None)
            for entry in batch:
                key = f"{entry.get('name', '')}|{entry.get('institution', '')}"
                entries[key] = entry

        return list(entries.values())

    async def _apply_filter_and_scrape(
        self, page: Page, filter_value: Optional[str]
    ) -> list[dict]:
        """Apply a single filter value (or no filter) and return all table rows."""
        if filter_value:
            applied = await page.evaluate(
                """
                (value) => {
                    const selects = Array.from(document.querySelectorAll('select'));
                    for (const sel of selects) {
                        for (const opt of sel.options) {
                            if (opt.text.trim() === value) {
                                sel.value = opt.value;
                                sel.dispatchEvent(new Event('change', {bubbles: true}));
                                return true;
                            }
                        }
                    }
                    return false;
                }
                """,
                filter_value,
            )
            if not applied:
                logger.debug(f"Filter value not found on page: {filter_value!r}")
                return []
            await page.wait_for_timeout(2000)

        return await self._extract_table_rows(page)

    async def _extract_table_rows(self, page: Page) -> list[dict]:
        """Extract programme data from all visible table rows."""
        await page.wait_for_load_state("networkidle")

        rows: list[dict] = await page.evaluate("""
            () => {
                const results = [];
                const rows = document.querySelectorAll('table tbody tr, .program-row, .programme-item');
                rows.forEach(row => {
                    const cells = Array.from(row.querySelectorAll('td, .cell'));
                    if (cells.length < 2) return;

                    const link = row.querySelector('a');
                    results.push({
                        name: cells[0]?.innerText?.trim() || '',
                        institution: cells[1]?.innerText?.trim() || '',
                        city: cells[2]?.innerText?.trim() || null,
                        study_mode: cells[3]?.innerText?.trim() || null,
                        lama_bpo_url: link ? link.href : null,
                        raw_html: row.innerHTML,
                    });
                });
                return results.filter(r => r.name.length > 0);
            }
        """)

        # If table structure differs, try a more generic extraction
        if not rows:
            rows = await self._extract_generic_entries(page)

        return rows

    async def _extract_generic_entries(self, page: Page) -> list[dict]:
        """Fallback: find all links that look like programme detail pages."""
        links: list[dict] = await page.evaluate(f"""
            () => {{
                const base = '{LAMA_BPO_BASE_URL}';
                return Array.from(document.querySelectorAll('a[href]'))
                    .filter(a => a.href.includes('/programa/') || a.href.includes('/studiju-programa/'))
                    .map(a => ({{
                        name: a.innerText.trim(),
                        institution: '',
                        lama_bpo_url: a.href,
                    }}))
                    .filter(e => e.name.length > 0);
            }}
        """)
        return links

    # ── Detail page enrichment ─────────────────────────────────────────────────

    async def _enrich_with_detail_pages(self, entries: list[dict]) -> list[Programme]:
        """Visit each programme's LAMA BPO detail page to get brief description and university URL."""
        programmes: list[Programme] = []
        for i, entry in enumerate(entries, 1):
            url = entry.get("lama_bpo_url")
            logger.info(f"[{i}/{len(entries)}] Enriching: {entry.get('name')} @ {entry.get('institution')}")

            brief_description = None
            university_url = None

            if url:
                try:
                    detail = await self._scrape_detail_page(url)
                    brief_description = detail.get("brief_description")
                    university_url = detail.get("university_url")
                except Exception as exc:
                    logger.warning(f"Detail page failed for {url}: {exc}")

            programme = Programme(
                name=entry.get("name", ""),
                institution=entry.get("institution", ""),
                city=entry.get("city"),
                field_group=entry.get("field_group"),
                field=entry.get("field"),
                study_mode=entry.get("study_mode"),
                lama_bpo_url=url,
                university_url=university_url,
                brief_description=brief_description,
                scraped_at=datetime.utcnow(),
            )

            # Exclude programmes with no descriptive text at all (per thesis methodology)
            if not programme.brief_description and not programme.extended_description:
                logger.debug(f"Excluding (no description): {programme.name} @ {programme.institution}")
                continue

            programmes.append(programme)

        return programmes

    async def _scrape_detail_page(self, url: str) -> dict:
        """Extract brief description and university link from a LAMA BPO programme detail page."""
        page = await self.fetch_page(url)
        try:
            data: dict = await page.evaluate("""
                () => {
                    // Brief description — look for common content containers
                    const descSelectors = [
                        '.programme-description', '.program-description',
                        '.studiju-programa-aprasymas', '.aprasymas',
                        '[class*="description"]', '[class*="aprasymas"]',
                        'article p', '.content p',
                    ];
                    let description = null;
                    for (const sel of descSelectors) {
                        const el = document.querySelector(sel);
                        if (el && el.innerText.trim().length > 50) {
                            description = el.innerText.trim();
                            break;
                        }
                    }

                    // University programme URL — look for an external link
                    const uniLinkSelectors = [
                        'a[href*="vgtu.lt"]', 'a[href*="vu.lt"]', 'a[href*="ktu.lt"]',
                        'a[href*="mykolas.lt"]', 'a[href*="ism.lt"]', 'a[href*="esf.lt"]',
                        'a[href*=".lt"][class*="university"]',
                        'a[href*=".lt"][class*="programa"]',
                    ];
                    let uniUrl = null;
                    for (const sel of uniLinkSelectors) {
                        const el = document.querySelector(sel);
                        if (el) { uniUrl = el.href; break; }
                    }
                    // Generic fallback: any outbound .lt link
                    if (!uniUrl) {
                        const allLinks = Array.from(document.querySelectorAll('a[href]'));
                        const ext = allLinks.find(a =>
                            a.href.startsWith('http') &&
                            !a.href.includes('lamabpo.lt') &&
                            a.href.includes('.lt')
                        );
                        if (ext) uniUrl = ext.href;
                    }

                    return { brief_description: description, university_url: uniUrl };
                }
            """)
            return data
        finally:
            await page.close()

    # ── Utilities ──────────────────────────────────────────────────────────────

    @staticmethod
    def _deduplicate(programmes: list[Programme]) -> list[Programme]:
        seen: set[str] = set()
        unique = []
        for p in programmes:
            key = f"{p.name.lower()}|{p.institution.lower()}"
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique


# ── CLI entry-point ────────────────────────────────────────────────────────────

async def run(output_path: Path = RAW_PROGRAMMES_DIR / "lama_bpo_programmes.json") -> None:
    async with LamaBpoScraper() as scraper:
        programmes = await scraper.scrape_all()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            [p.model_dump(mode="json") for p in programmes],
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"Saved {len(programmes)} programmes → {output_path}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
