"""
Step 1a — LAMA BPO programme listing scraper (English).

Scrapes Bachelor-level CS/ICT/AI programmes from the LAMA BPO English registry:
  https://lamabpo.lt/en/bachelors-studies/study-programmes/

For each programme it captures:
  - name, institution, city, field group, field, study mode
  - brief description from the LAMA BPO listing row
  - AIKOS link (national register) from the listing row
  - extended description from the AIKOS detail page:
      programme objective + learning outcomes + career paths

Output: data/raw/programmes/lama_bpo_programmes.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from loguru import logger
from playwright.async_api import Page

from src.scraping.base import BaseScraper
from src.scraping.config import (
    LAMA_BPO_BASE_URL,
    LAMA_BPO_PROGRAMMES_URL,
    RAW_PROGRAMMES_DIR,
    TARGET_FIELD_GROUPS,
    TARGET_FIELDS,
)
from src.scraping.models import Programme


class LamaBpoScraper(BaseScraper):
    """Scrapes LAMA BPO programme listing and AIKOS detail pages."""

    # ── Public API ─────────────────────────────────────────────────────────────

    async def scrape_all(self) -> list[Programme]:
        """
        Full scraping run:
          1. Load the listing page and read available filter options.
          2. Identify CS/ICT/AI relevant field groups and fields.
          3. Collect programme rows for each matching filter value.
          4. Deduplicate and enrich via AIKOS detail pages.
        Returns a list of Programme objects.
        """
        logger.info("Starting LAMA BPO scrape (EN)")
        page = await self.fetch_page(LAMA_BPO_PROGRAMMES_URL)

        filter_options = await self._read_filter_options(page)
        logger.info(
            f"Filter options — field_groups: {len(filter_options['field_groups'])}, "
            f"fields: {len(filter_options['fields'])}"
        )

        target_field_groups = self._match_targets(
            filter_options["field_groups"], TARGET_FIELD_GROUPS
        )
        target_fields = self._match_targets(filter_options["fields"], TARGET_FIELDS)
        logger.info(
            f"Matched — field_groups: {target_field_groups}, fields: {target_fields}"
        )

        raw_entries = await self._collect_programme_cards(
            page, target_field_groups, target_fields
        )
        await page.close()
        logger.info(f"Collected {len(raw_entries)} raw programme entries")

        programmes = await self._enrich_with_aikos(raw_entries)
        programmes = self._deduplicate(programmes)
        logger.info(f"Final programme count after dedup: {len(programmes)}")
        return programmes

    # ── Filter helpers ─────────────────────────────────────────────────────────

    async def _read_filter_options(self, page: Page) -> dict[str, list[str]]:
        """Extract option values from the Group of study field and Field of studies dropdowns."""
        await page.wait_for_load_state("networkidle")

        # Dump all selects so we can identify them by position/content
        all_selects: list[dict] = await page.evaluate("""
            () => Array.from(document.querySelectorAll('select')).map((sel, i) => ({
                index: i,
                name: sel.name || sel.id || '',
                options: Array.from(sel.options)
                    .map(o => o.text.trim())
                    .filter(t => t && t !== '---' && t !== '-')
            }))
        """)

        logger.debug(f"Selects on page: {json.dumps(all_selects, ensure_ascii=False)}")

        # The EN page uses name="field" for the broad study-field group
        # and name="direction" for the specific field of studies.
        field_groups: list[str] = []
        fields: list[str] = []

        for sel in all_selects:
            name = sel["name"]
            opts = sel["options"]
            if name == "field":
                field_groups = opts
            elif name == "direction":
                fields = opts

        return {"field_groups": field_groups, "fields": fields}

    @staticmethod
    def _match_targets(available: list[str], targets: list[str]) -> list[str]:
        """Return items from `available` whose text contains any target keyword."""
        matched = []
        for option in available:
            for target in targets:
                if target.lower() in option.lower():
                    matched.append(option)
                    break
        return matched if matched else available  # no match → try all

    # ── Card collection ────────────────────────────────────────────────────────

    async def _collect_programme_cards(
        self,
        page: Page,
        field_groups: list[str],
        fields: list[str],
    ) -> list[dict]:
        """
        Iterate filter values and collect programme rows.
        Prefers `fields` over `field_groups` for finer-grained filtering.
        Falls back to unfiltered scrape if nothing is collected.
        """
        entries: dict[str, dict] = {}

        # Prefer the broader group filter (e.g. "Computer Sciences" gets everything
        # in one pass); fall back to individual direction values if no groups matched.
        filter_values = field_groups if field_groups else fields
        for value in filter_values:
            batch = await self._apply_filter_and_scrape(page, value)
            for entry in batch:
                key = f"{entry.get('name', '')}|{entry.get('institution', '')}"
                entries[key] = entry

        if not entries:
            logger.warning("No entries matched filters — scraping without filter")
            batch = await self._apply_filter_and_scrape(page, filter_value=None)
            for entry in batch:
                key = f"{entry.get('name', '')}|{entry.get('institution', '')}"
                entries[key] = entry

        return list(entries.values())

    async def _apply_filter_and_scrape(
        self, page: Page, filter_value: str | None
    ) -> list[dict]:
        """Apply a single dropdown filter value (or none) and return all table rows."""
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
                logger.debug(f"Filter value not applied: {filter_value!r}")
                return []

            # Click a search/submit button if one exists after setting the filter
            await page.evaluate("""
                () => {
                    const btn = document.querySelector(
                        'button[type="submit"], input[type="submit"], '
                        + 'button.search, button.filter, .search-btn, .btn-search'
                    );
                    if (btn) btn.click();
                }
            """)
            await page.wait_for_timeout(3000)

        return await self._extract_table_rows(page)

    async def _extract_table_rows(self, page: Page) -> list[dict]:
        """Extract programme data from all visible table rows including AIKOS links."""
        await page.wait_for_load_state("networkidle")

        rows: list[dict] = await page.evaluate("""
            () => {
                const results = [];
                const rows = document.querySelectorAll('table tbody tr, .program-row, .programme-item');
                rows.forEach(row => {
                    const cells = Array.from(row.querySelectorAll('td, .cell'));
                    // Require at least 5 columns:
                    // [0] field group  [1] direction  [2] institution  [3] city  [4] programme name
                    if (cells.length < 5) return;

                    const aikosLink = row.querySelector('a[href*="aikos.smm.lt"]');
                    const nameText  = cells[4]?.innerText?.trim() || '';
                    if (!nameText) return;

                    results.push({
                        name:        nameText,
                        institution: cells[2]?.innerText?.trim() || '',
                        city:        cells[3]?.innerText?.trim() || null,
                        field_group: cells[0]?.innerText?.trim() || null,
                        field:       cells[1]?.innerText?.trim() || null,
                        study_mode:  cells[5]?.innerText?.trim() || null,
                        aikos_url:   aikosLink ? aikosLink.href : null,
                    });
                });
                return results;
            }
        """)

        if not rows:
            rows = await self._extract_generic_entries(page)

        return rows

    async def _extract_generic_entries(self, page: Page) -> list[dict]:
        """Fallback: find all links pointing to programme or AIKOS pages."""
        entries: list[dict] = await page.evaluate(f"""
            () => {{
                const aikos = Array.from(document.querySelectorAll('a[href*="aikos.smm.lt"]'))
                    .map(a => ({{
                        name: a.closest('tr')?.querySelector('td')?.innerText?.trim()
                              || a.innerText.trim(),
                        institution: '',
                        lama_bpo_url: null,
                        aikos_url: a.href,
                        brief_description: null,
                    }}))
                    .filter(e => e.name.length > 0);

                if (aikos.length > 0) return aikos;

                // Last resort: any programme-looking link on the LAMA BPO domain
                return Array.from(document.querySelectorAll('a[href]'))
                    .filter(a => a.href.includes('{LAMA_BPO_BASE_URL}') &&
                                 (a.href.includes('/program') || a.href.includes('/programme')))
                    .map(a => ({{
                        name: a.innerText.trim(),
                        institution: '',
                        lama_bpo_url: a.href,
                        aikos_url: null,
                        brief_description: null,
                    }}))
                    .filter(e => e.name.length > 0);
            }}
        """)
        return entries

    # ── AIKOS enrichment ───────────────────────────────────────────────────────

    async def _enrich_with_aikos(self, entries: list[dict]) -> list[Programme]:
        """
        For each raw entry, fetch the AIKOS page (if available) to build the
        extended_description from programme objective, learning outcomes and
        career paths.  Entries with neither a brief nor extended description
        are excluded per the thesis methodology.
        """
        programmes: list[Programme] = []
        for i, entry in enumerate(entries, 1):
            aikos_url = entry.get("aikos_url")
            logger.info(
                f"[{i}/{len(entries)}] {entry.get('name')} @ {entry.get('institution')}"
            )

            extended_description: str | None = None

            if aikos_url:
                try:
                    extended_description = await self._scrape_aikos_page(aikos_url)
                except Exception as exc:
                    logger.warning(f"AIKOS page failed for {aikos_url}: {exc}")

            programme = Programme(
                name=entry.get("name", ""),
                institution=entry.get("institution", ""),
                city=entry.get("city"),
                field_group=entry.get("field_group"),
                field=entry.get("field"),
                study_mode=entry.get("study_mode"),
                aikos_url=aikos_url,
                brief_description=None,
                extended_description=extended_description,
                scraped_at=datetime.utcnow(),
            )

            if not programme.brief_description and not programme.extended_description:
                logger.debug(f"Excluding (no description): {programme.name}")
                continue

            programmes.append(programme)

        return programmes

    async def _scrape_aikos_page(self, url: str) -> str | None:
        """
        Fetch and return structured text from an AIKOS programme detail page.

        Extracts (in order of appearance):
          - Programme Objective
          - Core Learning Outcomes
          - Teaching Methods
          - Career Paths

        Returns a single string with labelled sections, or None if nothing useful
        is found.
        """
        page = await self.fetch_page(url)
        try:
            data: dict = await page.evaluate("""
                () => {
                    // Helper: return visible text of the first matching element
                    const get = (...sels) => {
                        for (const sel of sels) {
                            const el = document.querySelector(sel);
                            if (el) {
                                const t = el.innerText.trim();
                                if (t.length > 10) return t;
                            }
                        }
                        return null;
                    };

                    // Helper: extract a named section by scanning heading+sibling text
                    const section = (keyword) => {
                        const els = Array.from(document.querySelectorAll('h1,h2,h3,h4,td,th,label,strong,b'));
                        for (const el of els) {
                            if (el.innerText.toLowerCase().includes(keyword.toLowerCase())) {
                                // Try next sibling or parent's next sibling text
                                const next = el.nextElementSibling || el.parentElement?.nextElementSibling;
                                if (next) {
                                    const t = next.innerText.trim();
                                    if (t.length > 10) return t;
                                }
                                // Try same row's next cell (table layout)
                                const row = el.closest('tr');
                                if (row) {
                                    const cells = Array.from(row.querySelectorAll('td'));
                                    if (cells.length >= 2) return cells[cells.length - 1].innerText.trim();
                                }
                            }
                        }
                        return null;
                    };

                    const objective = section('objective') || section('programme aim');
                    const outcomes  = section('learning outcome') || section('competenc');
                    const methods   = section('teaching method') || section('study method');
                    const careers   = section('career') || section('employment');

                    // Fallback: grab all visible text from the main content zone
                    const body = get(
                        '.ms-rtestate-field',
                        '#contentBox',
                        '.ms-webpart-zone',
                        '.content-area',
                        '#ctl00_PlaceHolderMain_ctl00',
                        'main article',
                        '.program-details',
                        'article',
                        'main',
                    ) || (() => {
                        // Last resort: all paragraph text on the page
                        const paras = Array.from(document.querySelectorAll('p, li'))
                            .map(el => el.innerText.trim())
                            .filter(t => t.length > 30);
                        return paras.length > 0 ? paras.join('\\n') : null;
                    })();

                    return { objective, outcomes, methods, careers, body };
                }
            """)

            parts: list[str] = []
            if data.get("objective"):
                parts.append(f"Programme Objective:\n{data['objective']}")
            if data.get("outcomes"):
                parts.append(f"Learning Outcomes:\n{data['outcomes']}")
            if data.get("methods"):
                parts.append(f"Teaching Methods:\n{data['methods']}")
            if data.get("careers"):
                parts.append(f"Career Paths:\n{data['careers']}")

            if not parts and data.get("body"):
                parts.append(data["body"])

            return "\n\n".join(parts) if parts else None

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
