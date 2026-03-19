"""
Dallas County Lis Pendens Scraper (Texas - FIPS 48113)

Source: Dallas County Clerk — Official Public Records
Portal: GovOS/Kofile PublicSearch at https://dallas.tx.publicsearch.us

Data:   Lis Pendens ("suit pending") instruments recorded with the county
        clerk.  A Lis Pendens puts the world on notice that the property is
        subject to a pending lawsuit — most commonly a lender's foreclosure
        action or a title dispute.

Signal: indicator_type = "lien"
        Active Lis Pendens = ongoing legal action that encumbers title.

Strategy:
        The results URL with department=RP&docTypes=LP renders an HTML table
        of LP instruments directly.  We:
          1. Navigate to the results URL with LP filter + date-range.
          2. Parse the HTML table rows (columns: Grantor, Grantee, Doc Type,
             Recorded Date, Doc Number, Book/Volume/Page, Town, Legal Description).
          3. Paginate by clicking numbered page buttons (50 rows/page).
          4. Yield one RawIndicatorRecord per LP instrument.

URL format:
        https://dallas.tx.publicsearch.us/results
            ?department=RP
            &docTypes=LP
            &recordedDateRange=YYYYMMDD%2CYYYYMMDD
            &searchType=advancedSearch

Address: Lis Pendens filings carry a legal description (lot/block/subdivision)
        and a "Town" field, but no street address.  The scraper stores a
        placeholder ("Lis Pendens {doc_num}, {town}, Dallas County, TX") and
        keeps grantor name as owner_name for downstream enrichment.

Rate:   5 req/min — Playwright navigations are slow; be conservative.
"""

from __future__ import annotations

import asyncio
import re
import structlog
from datetime import date, datetime, timedelta
from typing import AsyncIterator

from scrapers.base import RawIndicatorRecord
from scrapers.playwright_base import PlaywrightBaseScraper

logger = structlog.get_logger(__name__)

COUNTY_FIPS = "48113"
PORTAL_BASE = "https://dallas.tx.publicsearch.us"
SOURCE_URL = f"{PORTAL_BASE}/"

LOOKBACK_DAYS = 90

# Table column indices (0-based) — first 3 cols are empty icon columns
_COL_GRANTOR = 3
_COL_GRANTEE = 4
_COL_DOC_TYPE = 5
_COL_RECORDED_DATE = 6
_COL_DOC_NUMBER = 7
_COL_BOOK_VOL_PAGE = 8
_COL_TOWN = 9
_COL_LEGAL_DESC = 10


def _build_search_url(start_date: str, end_date: str) -> str:
    """Build the GovOS advanced search URL for LP doc type.

    Dates must be YYYYMMDD format.  The comma separator is URL-encoded as %2C.
    """
    return (
        f"{PORTAL_BASE}/results"
        f"?department=RP"
        f"&docTypes=LP"
        f"&recordedDateRange={start_date}%2C{end_date}"
        f"&searchType=advancedSearch"
    )


def _parse_recorded_date(raw: str) -> date | None:
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    return None


class DallasLisPendensScraper(PlaywrightBaseScraper):
    county_fips = COUNTY_FIPS
    source_name = "Dallas County Clerk Lis Pendens (GovOS PublicSearch)"
    indicator_types = ["lien"]
    rate_limit_per_minute = 5

    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        start_date = (date.today() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
        end_date = date.today().strftime("%Y%m%d")
        search_url = _build_search_url(start_date, end_date)

        total = 0
        async with self.browser_context() as ctx:
            page = await ctx.new_page()

            # Navigate to homepage first so the React SPA establishes session
            # state, then go to the results URL.  Going directly to /results
            # sometimes triggers a blank page on repeat visits.
            try:
                await page.goto(PORTAL_BASE + "/", wait_until="domcontentloaded", timeout=30_000)
                await asyncio.sleep(2)
            except Exception:
                pass  # non-fatal; attempt results URL anyway

            logger.info("dallas_lp_navigate", url=search_url[:100])
            try:
                await page.goto(search_url, wait_until="domcontentloaded", timeout=45_000)
            except Exception as exc:
                logger.warning("dallas_lp_navigation_error", error=str(exc)[:120])
                return

            # Wait for the React SPA to render the results table
            try:
                await page.wait_for_selector("table tbody tr", timeout=25_000)
            except Exception:
                logger.warning("dallas_lp_table_timeout", hint="No table rows after 25s")
            await asyncio.sleep(2)

            # Count total pages from pagination
            page_count = await self._get_page_count(page)
            logger.info("dallas_lp_pages_found", pages=page_count, url=search_url[:80])

            for page_num in range(1, page_count + 1):
                if page_num > 1:
                    clicked = await self._go_to_page(page, page_num)
                    if not clicked:
                        logger.warning("dallas_lp_page_nav_failed", page=page_num)
                        break
                    await asyncio.sleep(2)

                rows = await self._parse_table(page)
                logger.info("dallas_lp_page_parsed", page=page_num, rows=len(rows))

                for row in rows:
                    record = self._build_record(row)
                    if record and await self.validate_record(record):
                        yield record
                        total += 1

        if total == 0:
            logger.warning(
                "dallas_lp_no_records",
                hint=(
                    "0 LP records yielded. Check "
                    f"https://dallas.tx.publicsearch.us/results"
                    "?department=RP&docTypes=LP&recordedDateRange=YYYYMMDD%2CYYYYMMDD"
                    "&searchType=advancedSearch"
                ),
            )
        else:
            logger.info("dallas_lp_complete", total_yielded=total)

    async def _get_page_count(self, page) -> int:
        """Return the number of result pages from the pagination buttons."""
        try:
            # Pagination renders as ◀ 1 2 3 ▶ buttons
            page_nums = await page.evaluate("""() => {
                const btns = Array.from(document.querySelectorAll(
                    '[class*="pagination"] button, [aria-label*="page"], [class*="page-btn"]'
                ));
                const nums = btns
                    .map(b => parseInt(b.textContent.trim(), 10))
                    .filter(n => !isNaN(n));
                return nums.length ? Math.max(...nums) : 0;
            }""")
            if page_nums and page_nums > 0:
                return int(page_nums)

            # Fallback: check if table has any rows at all
            row_count = await page.evaluate(
                "() => document.querySelectorAll('table tbody tr').length"
            )
            return 1 if row_count > 0 else 0
        except Exception as exc:
            logger.debug("dallas_lp_page_count_error", error=str(exc)[:80])
            return 1

    async def _go_to_page(self, page, page_num: int) -> bool:
        """Click the numbered page button to navigate to the given page."""
        try:
            # Try various selectors for page number buttons
            for selector in [
                f"button:has-text('{page_num}')",
                f"[aria-label='Page {page_num}']",
                f"[aria-label='page {page_num}']",
            ]:
                btn = page.locator(selector).first
                if await btn.count() > 0:
                    await btn.click()
                    await page.wait_for_load_state("networkidle", timeout=15_000)
                    await asyncio.sleep(1.5)
                    return True
        except Exception as exc:
            logger.debug("dallas_lp_page_click_error", page=page_num, error=str(exc)[:80])
        return False

    async def _parse_table(self, page) -> list[dict]:
        """Extract all rows from the current results table."""
        try:
            await page.wait_for_selector("table tbody tr", timeout=10_000)
        except Exception:
            logger.warning("dallas_lp_no_table_rows")
            return []

        rows = await page.evaluate("""() => {
            const rows = document.querySelectorAll('table tbody tr');
            return Array.from(rows).map(row => {
                const cells = row.querySelectorAll('td');
                const texts = Array.from(cells).map(c => c.textContent.trim().replace(/\\s+/g, ' '));
                // Also try to grab the doc detail href
                const links = row.querySelectorAll('a[href]');
                const href = links.length > 0 ? links[0].href : '';
                return {cells: texts, href: href};
            });
        }""")

        parsed = []
        for row_data in rows:
            cells = row_data.get("cells", [])
            if len(cells) < 8:
                continue
            parsed.append({
                "grantor": cells[_COL_GRANTOR] if len(cells) > _COL_GRANTOR else "",
                "grantee": cells[_COL_GRANTEE] if len(cells) > _COL_GRANTEE else "",
                "doc_type": cells[_COL_DOC_TYPE] if len(cells) > _COL_DOC_TYPE else "",
                "recorded_date": cells[_COL_RECORDED_DATE] if len(cells) > _COL_RECORDED_DATE else "",
                "doc_number": cells[_COL_DOC_NUMBER] if len(cells) > _COL_DOC_NUMBER else "",
                "book_vol_page": cells[_COL_BOOK_VOL_PAGE] if len(cells) > _COL_BOOK_VOL_PAGE else "",
                "town": cells[_COL_TOWN] if len(cells) > _COL_TOWN else "",
                "legal_description": cells[_COL_LEGAL_DESC] if len(cells) > _COL_LEGAL_DESC else "",
                "href": row_data.get("href", ""),
            })
        return parsed

    def _build_record(self, row: dict) -> RawIndicatorRecord | None:
        doc_number = (row.get("doc_number") or "").strip()
        if not doc_number or doc_number in ("--", "N/A", ""):
            return None

        grantor = (row.get("grantor") or "").strip().title()
        grantee = (row.get("grantee") or "").strip().title()
        town = (row.get("town") or "").strip().title()
        legal_desc = (row.get("legal_description") or "").strip()
        recorded_date_raw = (row.get("recorded_date") or "").strip()

        filing_date = _parse_recorded_date(recorded_date_raw)

        # Build address: LP filings don't have street addresses, use placeholder
        town_part = town if town and town not in ("N/A", "--", "No Town") else "Dallas County"
        address_raw = f"Lis Pendens {doc_number}, {town_part}, TX"

        source = row.get("href") or SOURCE_URL

        return RawIndicatorRecord(
            indicator_type="lien",
            address_raw=address_raw,
            county_fips=self.county_fips,
            owner_name=grantor or None,
            filing_date=filing_date,
            case_number=doc_number,
            source_url=source,
            raw_payload={
                "doc_number": doc_number,
                "doc_type": row.get("doc_type", "LIS PENDENS (NOTICE OF)"),
                "grantor": grantor,
                "grantee": grantee,
                "recorded_date": recorded_date_raw,
                "town": town,
                "legal_description": legal_desc[:300] if legal_desc else "",
                "book_vol_page": row.get("book_vol_page", ""),
            },
        )
