from __future__ import annotations

import asyncio
import time
from typing import Any

from loguru import logger
from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from src.scraping.config import (
    MAX_RETRIES,
    NAVIGATION_TIMEOUT,
    PAGE_TIMEOUT,
    REQUEST_DELAY,
    USER_AGENT,
)


class BaseScraper:
    """Async Playwright-based scraper with retry logic and rate limiting."""

    def __init__(self, headless: bool = True, delay: float = REQUEST_DELAY):
        self.headless = headless
        self.delay = delay
        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None

    async def __aenter__(self) -> "BaseScraper":
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        self._context = await self._browser.new_context(
            user_agent=USER_AGENT,
            locale="lt-LT",
            extra_http_headers={"Accept-Language": "lt-LT,lt;q=0.9,en;q=0.8"},
        )
        self._context.set_default_timeout(PAGE_TIMEOUT)
        self._context.set_default_navigation_timeout(NAVIGATION_TIMEOUT)
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def new_page(self) -> Page:
        assert self._context is not None, "Scraper not started — use as async context manager"
        return await self._context.new_page()

    async def fetch_page(self, url: str, wait_for: str | None = None) -> Page:
        """Open a URL in a new page and optionally wait for a CSS selector."""
        page = await self.new_page()
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                await page.goto(url, wait_until="domcontentloaded")
                if wait_for:
                    await page.wait_for_selector(wait_for, timeout=PAGE_TIMEOUT)
                await asyncio.sleep(self.delay)
                return page
            except Exception as exc:
                logger.warning(f"Attempt {attempt}/{MAX_RETRIES} failed for {url}: {exc}")
                if attempt == MAX_RETRIES:
                    raise
                await asyncio.sleep(self.delay * attempt)
        raise RuntimeError(f"All retries exhausted for {url}")  # unreachable
