"""
Dallas County Lis Pendens Scraper (Texas - FIPS 48113)

Source: Dallas County Clerk — Official Public Records
Portal: GovOS/Kofile PublicSearch at https://dallas.tx.publicsearch.us

Data:   Lis Pendens ("suit pending") instruments recorded with the county
        clerk.  A Lis Pendens puts the world on notice that the property is
        subject to a pending lawsuit — most commonly a lender's foreclosure
        action or a title dispute.  Recording date + grantor (property owner)
        name are the key fields; the recorded document does NOT include a
        street address (address enrichment must come from assessor lookups).

Signal: indicator_type = "lien"
        Active Lis Pendens = ongoing legal action that encumbers title.
        Lenders file these before foreclosure auctions; they signal severe
        financial distress.  Especially actionable when combined with tax
        delinquency or code violations.

Strategy:
        GovOS PublicSearch is a React SPA — the results page renders
        client-side via XHR/fetch calls to an internal JSON API.  We use
        Playwright to:
          1. Navigate to the search URL with LP docType + date-range params.
          2. Intercept every JSON API response matching the results endpoint.
          3. Collect instrument records across paginated responses.
          4. Click "next page" inside the browser to trigger subsequent pages.

        If the API interception yields nothing (portal changed endpoints),
        the scraper falls back to DOM parsing of the rendered results table.
        Both approaches log enough detail to diagnose failures.

Doctype: "LP" is the most common code.  The scraper also tries "LIS" if LP
        returns 0 results (GovOS version differences across counties).

Address: Lis Pendens filings do not include street addresses — only a legal
        description (lot/block/subdivision).  The scraper stores a placeholder
        address ("Lis Pendens {num}, Dallas County, TX") and the grantor name
        in owner_name so the record can be enriched via assessor APN lookup.

Rate:   5 req/min — conservative; Playwright navigations are slow anyway.
"""

from __future__ import annotations

import asyncio
import json
import structlog
from datetime import date, datetime, timedelta
from typing import AsyncIterator

from scrapers.base import RawIndicatorRecord
from scrapers.playwright_base import PlaywrightBaseScraper

logger = structlog.get_logger(__name__)

COUNTY_FIPS = "48113"
PORTAL_BASE = "https://dallas.tx.publicsearch.us"
SOURCE_URL = f"{PORTAL_BASE}/"

# Search window: active liens (recent filings; old ones age off or get cancelled)
LOOKBACK_DAYS = 90

# GovOS document type codes for Lis Pendens — try in order
_LP_DOC_TYPES = ["LP", "LIS", "LISP", "LIS PENDENS"]

# Substrings that identify the JSON search-results API response URL
_API_URL_HINTS = (
    "/api/search",
    "/api/instruments",
    "/search/instruments",
    "docType",
    "instrument",
)

# GovOS pagination page size
PAGE_SIZE = 25


def _coerce(d: dict, *keys: str, default: str = "") -> str:
    for k in keys:
        v = d.get(k)
        if v is not None:
            return str(v).strip()
    return default


def _join_names(parties: list[dict]) -> str | None:
    """Extract and join party names from a GovOS parties array."""
    names = []
    for p in parties or []:
        name = _coerce(p, "name", "fullName", "Name").strip()
        if name:
            names.append(name.title())
    return "; ".join(names) if names else None


class DallasLisPendensScraper(PlaywrightBaseScraper):
    county_fips = COUNTY_FIPS
    source_name = "Dallas County Clerk Lis Pendens (GovOS PublicSearch)"
    indicator_types = ["lien"]
    rate_limit_per_minute = 5  # Playwright navigations are slow; be conservative

    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        start_date = (date.today() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        end_date = date.today().strftime("%Y-%m-%d")

        total = 0

        for doc_type in _LP_DOC_TYPES:
            instruments = await self._fetch_by_doc_type(doc_type, start_date, end_date)
            if instruments:
                logger.info(
                    "dallas_lp_doc_type_hit",
                    doc_type=doc_type,
                    count=len(instruments),
                )
                for inst in instruments:
                    record = self._build_record(inst)
                    if record and await self.validate_record(record):
                        yield record
                        total += 1
                break  # don't double-count; stop at first successful type
            else:
                logger.debug("dallas_lp_doc_type_no_results", doc_type=doc_type)

        if total == 0:
            logger.warning(
                "dallas_lp_no_records",
                hint=(
                    "All LP doc type codes returned 0 results. "
                    "Open https://dallas.tx.publicsearch.us in a browser, "
                    "search for Lis Pendens, and check the Network tab for "
                    "the actual API endpoint URL and doc type code."
                ),
            )
        else:
            logger.info("dallas_lp_complete", total_yielded=total)

    async def _fetch_by_doc_type(
        self, doc_type: str, start_date: str, end_date: str
    ) -> list[dict]:
        """
        Open the portal in a Playwright browser, navigate to the search
        results page for the given doc type + date range, and collect all
        instrument records across pages.

        Strategy A: Intercept JSON API responses (preferred — clean data).
        Strategy B: DOM parse the rendered results table (fallback).
        """
        search_url = (
            f"{PORTAL_BASE}/results"
            f"?search=&docTypes={doc_type}"
            f"&dateRecordedMin={start_date}"
            f"&dateRecordedMax={end_date}"
        )

        captured: list[dict] = []  # instrument records collected from API responses
        api_hit = False

        async with self.browser_context() as ctx:
            page = await ctx.new_page()

            # --- Strategy A: Intercept JSON API responses ---
            async def handle_response(response):
                nonlocal api_hit
                url = response.url
                # Heuristic: look for any response that might be the search API
                if response.status != 200:
                    return
                content_type = response.headers.get("content-type", "")
                if "json" not in content_type:
                    return
                if not any(hint in url for hint in _API_URL_HINTS):
                    return
                try:
                    data = await response.json()
                    instruments = self._extract_instruments(data, url)
                    if instruments:
                        api_hit = True
                        captured.extend(instruments)
                        logger.debug(
                            "dallas_lp_api_intercepted",
                            url=url[:120],
                            count=len(instruments),
                        )
                except Exception as exc:
                    logger.debug("dallas_lp_intercept_parse_error", url=url[:80], error=str(exc)[:100])

            page.on("response", handle_response)

            # Navigate to the search URL
            try:
                await page.goto(search_url, wait_until="networkidle", timeout=30_000)
            except Exception as exc:
                logger.warning("dallas_lp_navigation_error", url=search_url[:100], error=str(exc)[:120])
                return []

            # Wait a moment for any deferred XHR
            await asyncio.sleep(3)

            if api_hit:
                # Paginate: click "next" until exhausted
                captured.extend(
                    await self._paginate_via_clicks(page, doc_type, start_date, end_date)
                )
                return captured

            # --- Strategy B: DOM fallback ---
            logger.info(
                "dallas_lp_fallback_to_dom",
                doc_type=doc_type,
                hint="No API response intercepted; attempting DOM parse",
            )
            return await self._scrape_dom(page)

    async def _paginate_via_clicks(
        self, page, doc_type: str, start_date: str, end_date: str
    ) -> list[dict]:
        """
        Click the 'Next' pagination button repeatedly, collecting instrument
        records from each subsequent API response.
        """
        additional: list[dict] = []
        page_num = 2

        while True:
            # Try common "Next page" selectors used in GovOS portals
            next_btn = None
            for selector in (
                "button[aria-label='Next page']",
                "button[aria-label='next']",
                "li.pagination-next:not(.disabled) a",
                "[data-testid='next-page']",
                "button:has-text('Next')",
                "a:has-text('Next')",
            ):
                try:
                    btn = page.locator(selector)
                    if await btn.count() > 0 and await btn.is_enabled():
                        next_btn = btn
                        break
                except Exception:
                    continue

            if next_btn is None:
                logger.debug("dallas_lp_pagination_done", pages=page_num - 1)
                break

            page_captured: list[dict] = []

            async def handle_page_response(response):
                url = response.url
                if response.status != 200:
                    return
                content_type = response.headers.get("content-type", "")
                if "json" not in content_type:
                    return
                if not any(hint in url for hint in _API_URL_HINTS):
                    return
                try:
                    data = await response.json()
                    instruments = self._extract_instruments(data, url)
                    if instruments:
                        page_captured.extend(instruments)
                except Exception:
                    pass

            page.on("response", handle_page_response)

            try:
                await next_btn.click()
                await page.wait_for_load_state("networkidle", timeout=15_000)
                await asyncio.sleep(2)
            except Exception as exc:
                logger.debug("dallas_lp_next_click_error", page=page_num, error=str(exc)[:80])
                break
            finally:
                page.remove_listener("response", handle_page_response)

            if not page_captured:
                logger.debug("dallas_lp_no_results_on_page", page=page_num)
                break

            additional.extend(page_captured)
            logger.debug("dallas_lp_page_fetched", page=page_num, count=len(page_captured))
            page_num += 1

            # Safety: cap at 200 pages (5,000 instruments) to prevent runaway
            if page_num > 200:
                logger.warning("dallas_lp_pagination_cap_reached", pages=page_num)
                break

        return additional

    async def _scrape_dom(self, page) -> list[dict]:
        """
        Fallback: attempt to parse a results table rendered in the DOM.
        GovOS portals render a table with columns like:
          Instrument # | Doc Type | Grantor | Grantee | Recorded Date
        Returns synthetic instrument dicts to be processed by _build_record.
        """
        instruments = []
        try:
            # Wait for any table or results list to appear
            await page.wait_for_selector(
                "table tr, [class*='result'], [class*='instrument']",
                timeout=10_000,
            )
        except Exception:
            logger.warning("dallas_lp_dom_no_results_element")
            return []

        rows = await page.query_selector_all("table tbody tr")
        if not rows:
            # Try card-style layout
            rows = await page.query_selector_all("[class*='result-row'], [class*='instrument-row']")

        for row in rows:
            cells = await row.query_selector_all("td")
            texts = [await c.inner_text() for c in cells]
            if len(texts) < 3:
                continue

            # GovOS standard column order:
            # [instrument_number, doc_type, grantor, grantee, recorded_date, book, page]
            instrument: dict = {}
            if len(texts) >= 1:
                instrument["instrumentNumber"] = texts[0].strip()
            if len(texts) >= 3:
                instrument["grantors"] = [{"name": texts[2].strip()}]
            if len(texts) >= 4:
                instrument["grantees"] = [{"name": texts[3].strip()}]
            if len(texts) >= 5:
                instrument["recordedDate"] = texts[4].strip()

            if instrument.get("instrumentNumber"):
                instruments.append(instrument)

        logger.info("dallas_lp_dom_scraped", count=len(instruments))
        return instruments

    @staticmethod
    def _extract_instruments(data, url: str) -> list[dict]:
        """
        Pull an instrument list out of a GovOS JSON response.
        GovOS uses several wrapper shapes across portal versions.
        """
        if isinstance(data, list):
            # Bare array of instrument objects
            return data

        if isinstance(data, dict):
            # Try known wrapper keys (GovOS v3/v4/v5 variations)
            for key in (
                "instruments",
                "results",
                "data",
                "Records",
                "records",
                "items",
                "Instruments",
            ):
                val = data.get(key)
                if isinstance(val, list) and val:
                    return val

            # Some versions nest under a search result object
            inner = data.get("searchResults") or data.get("SearchResults") or {}
            if isinstance(inner, dict):
                for key in ("instruments", "results", "data", "items"):
                    val = inner.get(key)
                    if isinstance(val, list) and val:
                        return val

        return []

    def _build_record(self, inst: dict) -> RawIndicatorRecord | None:
        instrument_number = _coerce(
            inst, "instrumentNumber", "InstrumentNumber", "instrument_number", "id"
        )
        if not instrument_number:
            return None

        # Grantor = property owner / borrower
        grantor_name = _join_names(
            inst.get("grantors") or inst.get("Grantors") or []
        )
        # Grantee = lender / plaintiff filing the LP
        grantee_name = _join_names(
            inst.get("grantees") or inst.get("Grantees") or []
        )

        raw_date = _coerce(
            inst,
            "recordedDate", "RecordedDate", "recorded_date",
            "dateRecorded", "DateRecorded", "filingDate",
        )
        filing_date: date | None = None
        if raw_date:
            try:
                filing_date = datetime.fromisoformat(raw_date[:10]).date()
            except ValueError:
                pass

        doc_type = _coerce(inst, "docType", "DocType", "doc_type", "documentType")
        legal_desc = _coerce(
            inst, "legalDescription", "LegalDescription", "legal_description"
        )

        # Lis Pendens do not carry street addresses — use placeholder
        address_raw = f"Lis Pendens {instrument_number}, Dallas County, TX"

        return RawIndicatorRecord(
            indicator_type="lien",
            address_raw=address_raw,
            county_fips=self.county_fips,
            owner_name=grantor_name,
            filing_date=filing_date,
            case_number=instrument_number,
            source_url=SOURCE_URL,
            raw_payload={
                "instrument_number": instrument_number,
                "doc_type": doc_type,
                "grantor": grantor_name,
                "grantee": grantee_name,
                "recorded_date": raw_date,
                "legal_description": legal_desc[:300] if legal_desc else "",
                "book": _coerce(inst, "book", "Book"),
                "page": _coerce(inst, "page", "Page"),
            },
        )
