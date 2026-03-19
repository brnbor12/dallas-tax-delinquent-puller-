"""
Polk County Foreclosure Scraper (Florida - FIPS 12105)

Source: Polk County Clerk of Courts — PRO (Polk Records Online)
Portal: https://pro.polkcountyclerk.net/PRO/PublicSearch/PublicSearch

Data:   Mortgage foreclosure cases (case type CA — Circuit Civil).
        In Florida, foreclosures are filed as Circuit Civil actions.

Signal: indicator_type = "pre_foreclosure"
        An active foreclosure filing means the homeowner has defaulted on
        their mortgage and the lender is pursuing a judicial foreclosure.
        The owner may be highly motivated to sell before the sale date.

Strategy:
        Search PRO for CA (Circuit Civil) cases, then filter titles
        containing "mortgage foreclosure" or "foreclosure" keywords.
        Extract defendant (homeowner) address as the property.

Rate:   Conservative — 10 req/min.
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

# Title patterns that indicate foreclosure
_FORECLOSURE_RE = re.compile(
    r"\b(mortgage\s+foreclos|foreclos|mtg\s+fcl|lis\s+pendens)\b",
    re.IGNORECASE,
)

# Non-individual plaintiff patterns (banks, servicers) — expected for foreclosure
_LENDER_RE = re.compile(
    r"\b(bank|trust|mortgage|loan|servicing|llc|inc\b|corp|association|"
    r"federal|national|fha|freddie|fannie|mers|wells\s+fargo|chase|"
    r"citibank|bof?a|us\s+bank|pennymac|nationstar|ocwen|ditech)\b",
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


class PolkForeclosureScraper(BaseCountyScraper):
    county_fips = COUNTY_FIPS
    source_name = "Polk County Clerk PRO — Foreclosure Cases (CA)"
    indicator_types = ["pre_foreclosure"]
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
            try:
                init_resp = await client.get(SEARCH_URL)
                if init_resp.status_code != 200:
                    logger.error("polk_foreclosure_init_failed", status=init_resp.status_code)
                    return
            except Exception as exc:
                logger.error("polk_foreclosure_init_error", error=str(exc)[:120])
                return

            token = _extract_verification_token(init_resp.text)

            total = 0
            seen_cases: set[str] = set()
            prefixes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

            for prefix in prefixes:
                await self._rate_limit_sleep()
                try:
                    results = await self._search_cases(client, token, prefix)
                    for case in results:
                        case_num = case.get("case_number", "")
                        if not case_num or case_num in seen_cases:
                            continue

                        # Must be CA (Circuit Civil) with foreclosure in title
                        title = case.get("title", "")
                        if not _FORECLOSURE_RE.search(title):
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
                    logger.debug("polk_foreclosure_search_error", prefix=prefix, error=str(exc)[:80])
                    continue

            if total == 0:
                logger.warning("polk_foreclosure_no_records")
            else:
                logger.info("polk_foreclosure_complete", total_yielded=total, cases_seen=len(seen_cases))

    async def _search_cases(
        self, client: httpx.AsyncClient, token: str | None, last_name_prefix: str
    ) -> list[dict]:
        form_data = {
            "LastName": last_name_prefix,
            "SearchType": "CA",  # Circuit Civil
        }
        if token:
            form_data["__RequestVerificationToken"] = token

        try:
            resp = await client.post(SEARCH_URL, data=form_data, timeout=30.0)
            if resp.status_code != 200:
                return []
            return _parse_search_results(resp.text)
        except Exception as exc:
            logger.debug("polk_foreclosure_search_failed", prefix=last_name_prefix, error=str(exc)[:80])
            return []

    def _build_record(self, case: dict) -> RawIndicatorRecord | None:
        address_raw = (case.get("address") or "").strip()
        if not address_raw or not address_raw[0:1].isdigit():
            return None

        if "FL" not in address_raw.upper():
            address_raw = f"{address_raw}, FL"

        # Defendant = homeowner in foreclosure
        owner_name = (case.get("party_name") or "").strip().title()
        # Skip if defendant looks like an entity (unusual for foreclosure defendant)
        if _LENDER_RE.search(owner_name):
            owner_name = None

        filing_date = _parse_date(case.get("filing_date", ""))
        case_number = case.get("case_number", "")

        return RawIndicatorRecord(
            indicator_type="pre_foreclosure",
            address_raw=address_raw,
            county_fips=self.county_fips,
            owner_name=owner_name or None,
            filing_date=filing_date,
            case_number=case_number,
            source_url=SOURCE_URL,
            raw_payload={
                "case_number": case_number,
                "case_type": "CA",
                "title": case.get("title", ""),
                "party_name": owner_name or "",
                "filing_date": case.get("filing_date", ""),
                "status": case.get("status", ""),
            },
        )


def _extract_verification_token(html: str) -> str | None:
    match = re.search(
        r'<input[^>]*name="__RequestVerificationToken"[^>]*value="([^"]*)"',
        html,
    )
    return match.group(1) if match else None


def _parse_search_results(html: str) -> list[dict]:
    results = []
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
                "status": cells[5] if len(cells) > 5 else "",
                "case_type": "CA",
            })
    return results
