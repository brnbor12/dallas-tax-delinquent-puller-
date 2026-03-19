"""
Polk County Probate Scraper (Florida - FIPS 12105)

Source: Polk County Clerk of Courts — PRO (Polk Records Online)
Portal: https://pro.polkcountyclerk.net/PRO/PublicSearch/PublicSearch

Data:   Probate case filings (case type CP) — estates being administered
        through the 10th Judicial Circuit Court.

Signal: indicator_type = "probate"
        Estate filings indicate the property owner has died; heirs may be
        motivated sellers who want to liquidate inherited real estate quickly.

Strategy:
        PRO is an ASP.NET MVC app with a public case search.
        We search by case type prefix "CP" for recent years to find probate
        cases, then scrape party/address details from individual case pages.

        Florida Uniform Case Numbers: YY-NNNNNN-CP-XX
        We construct case numbers and search sequentially.

Rate:   Conservative — 10 req/min to avoid overloading the clerk's site.
"""

from __future__ import annotations

import re
import structlog
from datetime import date, timedelta
from typing import AsyncIterator

import httpx

from scrapers.base import BaseCountyScraper, RawIndicatorRecord

logger = structlog.get_logger(__name__)

COUNTY_FIPS = "12105"
PRO_BASE = "https://pro.polkcountyclerk.net/PRO"
SEARCH_URL = f"{PRO_BASE}/PublicSearch/PublicSearch"
SOURCE_URL = "https://pro.polkcountyclerk.net/PRO/PublicSearch/PublicSearch"

LOOKBACK_DAYS = 180

# Party roles indicating the decedent
_DECEDENT_ROLES = {"decedent", "deceased", "ward", "incapacitated person"}

# Non-individual entity patterns to flag
_ENTITY_RE = re.compile(
    r"\b(llc|l\.l\.c|inc|corp|trust|bank|mortgage|assoc|association)\b",
    re.IGNORECASE,
)


def _parse_date(raw: str) -> date | None:
    if not raw or not raw.strip():
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y"):
        try:
            from datetime import datetime
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    return None


class PolkProbateScraper(BaseCountyScraper):
    county_fips = COUNTY_FIPS
    source_name = "Polk County Clerk PRO — Probate Cases (CP)"
    indicator_types = ["probate"]
    rate_limit_per_minute = 10

    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        today = date.today()
        cutoff = today - timedelta(days=LOOKBACK_DAYS)

        async with httpx.AsyncClient(
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
            follow_redirects=True,
            timeout=30.0,
        ) as client:
            # Initialize session — GET the search page first for cookies/tokens
            try:
                init_resp = await client.get(SEARCH_URL)
                if init_resp.status_code != 200:
                    logger.error("polk_probate_init_failed", status=init_resp.status_code)
                    return
            except Exception as exc:
                logger.error("polk_probate_init_error", error=str(exc)[:120])
                return

            # Extract ASP.NET anti-forgery token if present
            token = _extract_verification_token(init_resp.text)

            # Search for probate cases by last name patterns
            # PRO requires at least a last name — search common names + wildcards
            # Or search by partial case number if supported
            total = 0
            seen_cases: set[str] = set()

            # Strategy: search using date range with common last name prefixes
            # to get broad coverage. PRO limits results to 100 per search.
            prefixes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

            for prefix in prefixes:
                await self._rate_limit_sleep()
                try:
                    results = await self._search_cases(client, token, prefix)
                    for case in results:
                        case_num = case.get("case_number", "")
                        if not case_num or case_num in seen_cases:
                            continue

                        # Filter: must be CP (probate) case type
                        if "CP" not in case_num.upper():
                            continue

                        filing_date = _parse_date(case.get("filing_date", ""))
                        if filing_date and filing_date < cutoff:
                            continue

                        seen_cases.add(case_num)

                        record = self._build_record(case)
                        if record and await self.validate_record(record):
                            yield record
                            total += 1

                except Exception as exc:
                    logger.debug("polk_probate_search_error", prefix=prefix, error=str(exc)[:80])
                    continue

            if total == 0:
                logger.warning("polk_probate_no_records")
            else:
                logger.info("polk_probate_complete", total_yielded=total, cases_seen=len(seen_cases))

    async def _search_cases(
        self, client: httpx.AsyncClient, token: str | None, last_name_prefix: str
    ) -> list[dict]:
        """Search PRO for probate cases matching a last name prefix."""
        form_data = {
            "LastName": last_name_prefix,
            "SearchType": "CP",  # Probate case type
        }
        if token:
            form_data["__RequestVerificationToken"] = token

        try:
            resp = await client.post(SEARCH_URL, data=form_data, timeout=30.0)
            if resp.status_code != 200:
                return []
            return _parse_search_results(resp.text)
        except Exception as exc:
            logger.debug("polk_probate_search_failed", prefix=last_name_prefix, error=str(exc)[:80])
            return []

    def _build_record(self, case: dict) -> RawIndicatorRecord | None:
        address_raw = (case.get("address") or "").strip()
        if not address_raw or not address_raw[0:1].isdigit():
            return None

        # Append city/state/zip if not already present
        if "FL" not in address_raw.upper():
            address_raw = f"{address_raw}, FL"

        owner_name = (case.get("party_name") or "").strip().title()
        filing_date = _parse_date(case.get("filing_date", ""))
        case_number = case.get("case_number", "")

        return RawIndicatorRecord(
            indicator_type="probate",
            address_raw=address_raw,
            county_fips=self.county_fips,
            owner_name=owner_name or None,
            filing_date=filing_date,
            case_number=case_number,
            source_url=SOURCE_URL,
            raw_payload={
                "case_number": case_number,
                "case_type": case.get("case_type", "CP"),
                "party_name": owner_name,
                "party_role": case.get("party_role", ""),
                "filing_date": case.get("filing_date", ""),
                "title": case.get("title", ""),
                "status": case.get("status", ""),
            },
        )


def _extract_verification_token(html: str) -> str | None:
    """Extract ASP.NET __RequestVerificationToken from form HTML."""
    match = re.search(
        r'<input[^>]*name="__RequestVerificationToken"[^>]*value="([^"]*)"',
        html,
    )
    return match.group(1) if match else None


def _parse_search_results(html: str) -> list[dict]:
    """Parse PRO search result HTML into a list of case dicts.

    PRO returns an HTML table with case results. We extract:
    - Case number, title/style, filing date, party name, address.
    """
    results = []

    # Look for table rows with case data
    # PRO uses <tr> rows in a results table
    row_pattern = re.compile(
        r'<tr[^>]*class="[^"]*search[Rr]esult[^"]*"[^>]*>(.*?)</tr>',
        re.DOTALL,
    )
    cell_pattern = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL)

    for row_match in row_pattern.finditer(html):
        row_html = row_match.group(1)
        cells = [
            re.sub(r"<[^>]+>", "", c.group(1)).strip()
            for c in cell_pattern.finditer(row_html)
        ]
        if len(cells) >= 3:
            results.append({
                "case_number": cells[0] if cells else "",
                "title": cells[1] if len(cells) > 1 else "",
                "filing_date": cells[2] if len(cells) > 2 else "",
                "party_name": cells[3] if len(cells) > 3 else "",
                "address": cells[4] if len(cells) > 4 else "",
                "party_role": cells[5] if len(cells) > 5 else "",
                "status": cells[6] if len(cells) > 6 else "",
                "case_type": "CP",
            })

    return results
