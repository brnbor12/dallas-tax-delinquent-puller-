"""
Hillsborough County Code Enforcement Scraper (Florida - FIPS 12057)

Source: Hillsborough County Code Enforcement — HillsGovHub (Accela Citizen Access)
Portal: https://aca-prod.accela.com/HCFL/Cap/CapHome.aspx?module=Enforcement&TabName=Enforcement

Signal: indicator_type = "code_violation"

Strategy:
  1. Navigate to the enforcement home/search page and click Search with the
     default date range (do NOT change Start Date — altering it triggers a
     .NET DateTime validation error).
  2. Parse the results grid: cells are
       [0]=checkbox  [1]=date  [2]=record_no(+link)  [3]=type
       [4]=description  [5]=project_name  [6]=related  [7]=status
  3. Filter rows to enforcement/code types only.
  4. Skip records older than LOOKBACK_DAYS (filter in Python).
  5. For each matching row, navigate directly to the CapDetail.aspx URL
     and extract the street address from the "Work Location" section.
  6. Paginate via Next > button.

Detail page address: under the "Work Location" h1, the address text sits in
the sibling div of the heading container (extracted via JavaScript).

Rate: 3 req/min
"""

from __future__ import annotations

import asyncio
import re
import structlog
from datetime import date, timedelta
from typing import AsyncIterator

from scrapers.base import RawIndicatorRecord
from scrapers.playwright_base import PlaywrightBaseScraper

logger = structlog.get_logger(__name__)

COUNTY_FIPS = "12057"
HOME_URL = (
    "https://aca-prod.accela.com/HCFL/Cap/CapHome.aspx"
    "?module=Enforcement&TabName=Enforcement"
)
SOURCE_URL = HOME_URL

LOOKBACK_DAYS = 180  # filter records in Python — don't change the form dates

# Record type prefixes that count as code enforcement violations
_ENFORCEMENT_RE = re.compile(
    r"\b(enforcement|citizen board support code|water enforcement|"
    r"regulatory compliance violation)\b",
    re.IGNORECASE,
)

# Statuses that mean the case is closed — skip
_CLOSED_RE = re.compile(
    r"\b(closed|finaled|withdrawn|dismissed|void|cancelled|canceled|complete)\b",
    re.IGNORECASE,
)


def _parse_date(raw: str):
    from datetime import datetime
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    return None


class HillsboroughCodeEnforcementScraper(PlaywrightBaseScraper):
    county_fips = COUNTY_FIPS
    source_name = "Hillsborough County Code Enforcement (HillsGovHub)"
    indicator_types = ["code_violation"]
    rate_limit_per_minute = 3

    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        lookback = int(self.config.get("lookback_days", LOOKBACK_DAYS))
        cutoff = date.today() - timedelta(days=lookback)

        total = 0
        async with self.browser_context() as ctx:
            page = await ctx.new_page()

            # Step 1: Load the search page and submit with default date range
            # (DO NOT change the Start Date field — doing so causes a .NET
            # DateTime parse error on the server side)
            logger.info("hcfl_ce_navigate")
            await page.goto(HOME_URL, wait_until="domcontentloaded", timeout=30_000)
            await asyncio.sleep(3)

            search_ok = await self._click_search(page)
            if not search_ok:
                logger.error("hcfl_ce_search_failed")
                return

            await asyncio.sleep(4)

            # Step 2: Paginate
            page_num = 1
            while True:
                logger.info("hcfl_ce_page", page=page_num)
                rows = await self._parse_results_page(page)

                if not rows:
                    logger.info("hcfl_ce_no_more_rows", pages=page_num)
                    break

                yielded_this_page = 0
                for row in rows:
                    record_type = row.get("record_type", "")
                    status = row.get("status", "")
                    filed_raw = row.get("date_filed", "")

                    # Filter by type
                    if not _ENFORCEMENT_RE.search(record_type):
                        continue
                    # Filter out closed
                    if _CLOSED_RE.search(status):
                        continue
                    # Filter by date
                    filed = _parse_date(filed_raw)
                    if filed and filed < cutoff:
                        continue  # too old — keep paginating (newer ones may follow)

                    detail_url = row.get("detail_url", "")
                    if not detail_url:
                        continue

                    address = await self._get_work_location(ctx, detail_url)
                    await asyncio.sleep(1)

                    logger.debug("hcfl_ce_address",
                                 record_no=row.get("record_no", ""),
                                 address=address[:80] if address else "(empty)",
                                 starts_digit=bool(address and address[0].isdigit()))

                    record = self._build_record(row, address)
                    if record and await self.validate_record(record):
                        yield record
                        total += 1
                        yielded_this_page += 1

                logger.info("hcfl_ce_page_done",
                            page=page_num, rows=len(rows), yielded=yielded_this_page)

                has_next = await self._go_next_page(page)
                if not has_next:
                    break
                page_num += 1
                await asyncio.sleep(3)

        if total == 0:
            logger.warning("hcfl_ce_no_records")
        else:
            logger.info("hcfl_ce_complete", total_yielded=total)

    async def _click_search(self, page) -> bool:
        """Click the Search link on the General Search form (no date changes)."""
        try:
            # The Search link: <a href="javascript:...btnNewSearch...">Search</a>
            for sel in [
                "a[href*='btnNewSearch']",
                "a[href*='PlaceHolderMain$btnNewSearch']",
            ]:
                btn = page.locator(sel).first
                if await btn.count() > 0:
                    await btn.click()
                    return True
            logger.warning("hcfl_ce_search_btn_not_found")
            return False
        except Exception as exc:
            logger.error("hcfl_ce_search_click_error", error=str(exc)[:200])
            return False

    async def _parse_results_page(self, page) -> list[dict]:
        """
        Extract rows from the results grid.
        Column layout confirmed from DOM inspection:
          cells[0] = checkbox
          cells[1] = date filed  (e.g. "03/09/2026")
          cells[2] = record number (contains <a href="/HCFL/Cap/CapDetail.aspx?...">)
          cells[3] = record type
          cells[4] = description
          cells[5] = project name / category
          cells[6] = related records count
          cells[7] = status
        """
        try:
            return await page.evaluate("""() => {
                const rows = [];
                // Find the results table by locating rows with CapDetail links
                const tables = document.querySelectorAll('table');
                let tbl = null;
                for (const t of tables) {
                    if (t.querySelector('a[href*="CapDetail"]')) {
                        tbl = t;
                        break;
                    }
                }
                if (!tbl) return rows;

                for (const tr of tbl.querySelectorAll('tr')) {
                    const cells = tr.querySelectorAll('td');
                    if (cells.length < 5) continue;

                    const texts = Array.from(cells).map(td =>
                        (td.innerText || '').trim()
                    );

                    // Cell[2] has the record number link
                    const link = cells[2] && cells[2].querySelector('a[href*="CapDetail"]');
                    const href = link ? link.href : '';
                    if (!href) continue;

                    rows.push({
                        date_filed:   texts[1] || '',
                        record_no:    texts[2] || '',
                        record_type:  texts[3] || '',
                        description:  texts[4] || '',
                        project_name: texts[5] || '',
                        status:       texts[7] || texts[6] || '',
                        detail_url:   href,
                    });
                }
                return rows;
            }""")
        except Exception as exc:
            logger.debug("hcfl_ce_parse_error", error=str(exc)[:120])
            return []

    async def _get_work_location(self, ctx, detail_url: str) -> str:
        """
        Open the CapDetail page and return the Work Location address text.

        Extraction: the address appears in document.body.innerText immediately
        after the heading "Work Location" and before the next section heading
        ("Record Details", "Owner:", etc.).  The sibling-div approach fails
        because the address div is empty in the DOM (lazy-rendered); the text
        IS present in innerText after domcontentloaded.
        """
        detail_page = await ctx.new_page()
        try:
            await detail_page.goto(
                detail_url, wait_until="domcontentloaded", timeout=25_000
            )
            await asyncio.sleep(1)

            address = await detail_page.evaluate("""() => {
                // Address lives in body.innerText immediately after 'Work Location'
                const bodyText = document.body.innerText || '';
                const wlIdx = bodyText.indexOf('Work Location');
                if (wlIdx >= 0) {
                    const after = bodyText.substring(wlIdx + 'Work Location'.length);
                    const lines = after.split('\\n').map(l => l.replace(/\\*/g, '').trim()).filter(l => l && l !== ' ');
                    const STOP = /^(Record Details|Project Description|Owner|More Details|Parcel|Fees|Inspections|Attachments|Processing|Related|Right Of Way|Custom|Skip to|View Additional|>>)/i;
                    const addrLines = [];
                    for (const line of lines) {
                        if (STOP.test(line)) break;
                        addrLines.push(line);
                    }
                    return addrLines.join(' ')
                        .replace(/View Additional Locations?>?>?/gi, '')
                        .replace(/\\s+/g, ' ').trim();
                }
                return '';
            }""")
            return address or ""
        except Exception as exc:
            logger.debug("hcfl_ce_detail_error",
                         url=detail_url[:80], error=str(exc)[:120])
            return ""
        finally:
            try:
                await detail_page.close()
            except Exception:
                pass

    async def _go_next_page(self, page) -> bool:
        """Click Next > pagination link if present and enabled."""
        try:
            for sel in [
                "a:has-text('Next >')",
                "a:has-text('Next>')",
                "a[title='Next Page']",
                ".ACA_SmBtn_Next",
                "a:has-text('>')",
            ]:
                btn = page.locator(sel).first
                if await btn.count() > 0:
                    disabled = await btn.get_attribute("disabled")
                    cls = await btn.get_attribute("class") or ""
                    if disabled or "disabled" in cls.lower():
                        return False
                    await btn.click()
                    await asyncio.sleep(3)
                    return True
            return False
        except Exception:
            return False

    def _build_record(self, row: dict, address: str) -> RawIndicatorRecord | None:
        record_no = row.get("record_no", "").strip()
        record_type = row.get("record_type", "").strip()
        status = row.get("status", "").strip()
        date_filed_raw = row.get("date_filed", "").strip()
        detail_url = row.get("detail_url", "")

        if not record_no:
            return None

        addr = address.strip()
        # Must start with a digit to be a usable street address
        if not addr or not addr[0].isdigit():
            return None

        if ", FL" not in addr.upper() and "FLORIDA" not in addr.upper():
            addr = f"{addr}, Hillsborough County, FL"

        filing_date = _parse_date(date_filed_raw)

        return RawIndicatorRecord(
            indicator_type="code_violation",
            address_raw=addr,
            county_fips=self.county_fips,
            case_number=record_no,
            filing_date=filing_date,
            source_url=SOURCE_URL,
            raw_payload={
                "record_no": record_no,
                "record_type": record_type,
                "status": status,
                "date_filed": date_filed_raw,
                "detail_url": detail_url,
                "description": row.get("description", "")[:200],
            },
        )
