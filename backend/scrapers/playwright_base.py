"""
Playwright base class for browser-driven county scrapers.

Usage
-----
Subclass PlaywrightBaseScraper instead of BaseCountyScraper whenever the
target portal is a JavaScript SPA (Tyler Odyssey, GovOS PublicSearch, etc.)
that cannot be scraped with plain HTTP requests.

    class MyCountyScraper(PlaywrightBaseScraper):
        county_fips = "48113"
        source_name = "My County Court (Playwright)"
        indicator_types = ["lien"]

        async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
            async with self.browser_context() as ctx:
                page = await ctx.new_page()
                await page.goto("https://example-court.gov/search")
                # ... interact with the page ...
                yield RawIndicatorRecord(...)

Architecture notes
------------------
- The `playwright-worker` Docker service (Dockerfile.playwright) runs this
  on the `web_scrape` Celery queue.  The regular `worker` service handles
  `county_api`, `mls`, and `default` queues.
- A fresh BrowserContext (incognito-equivalent) is created per scraper run
  so cookies/state never bleed between runs.
- `--no-sandbox` is required inside Docker containers (no user namespace).
- `--disable-blink-features=AutomationControlled` removes the navigator
  .webdriver flag that trivial bot-detection checks for.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from playwright.async_api import (
    Browser,
    BrowserContext,
    async_playwright,
)

from scrapers.base import BaseCountyScraper


class PlaywrightBaseScraper(BaseCountyScraper):
    """
    Abstract base for Playwright-driven scrapers.

    Subclasses call `async with self.browser_context() as ctx:` to obtain a
    ready-to-use BrowserContext.  Browser and Playwright are torn down
    automatically on exit.
    """

    _headless: bool = True
    _user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    )

    @asynccontextmanager
    async def browser_context(
        self, **context_kwargs
    ) -> AsyncGenerator[BrowserContext, None]:
        """
        Async context manager — yields a fresh Playwright BrowserContext.

        Extra keyword args are forwarded to ``browser.new_context()``, so
        callers can set ``extra_http_headers``, ``storage_state``, etc.

        Example::

            async with self.browser_context(
                extra_http_headers={"X-Custom": "value"}
            ) as ctx:
                page = await ctx.new_page()
        """
        async with async_playwright() as pw:
            browser: Browser = await pw.chromium.launch(
                headless=self._headless,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                ],
            )
            try:
                ctx: BrowserContext = await browser.new_context(
                    user_agent=self._user_agent,
                    viewport={"width": 1280, "height": 800},
                    locale="en-US",
                    **context_kwargs,
                )
                try:
                    yield ctx
                finally:
                    await ctx.close()
            finally:
                await browser.close()
