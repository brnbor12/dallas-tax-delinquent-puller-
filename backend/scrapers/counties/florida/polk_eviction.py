"""
Polk County Eviction Scraper (Florida - FIPS 12105)

Source: Polk County Clerk of Courts — PRO (Polk Records Online)
Portal: https://pro.polkcountyclerk.net/PRO/PublicSearch/PublicSearch

Data:   Eviction (landlord-tenant) case filings — case types CC (County Civil)
        and SC (Small Claims) that involve residential evictions.

Signal: indicator_type = "eviction"
        Active eviction filing = landlord-tenant conflict; landlord may be
        motivated to sell to exit a problem tenancy.

Strategy:
        PRO public search allows searching by party last name.
        We search for recent eviction-type cases (CC/SC), extract the
        defendant address (= rental property), and yield records.

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

LOOKBACK_DAYS = 90

# Case type codes that indicate eviction
_EVICTION_CASE_TYPES = {"CC", "SC"}

# Title patterns that indicate eviction
_EVICTION_TITLE_RE = re.compile(
    r"\b(evict|tenant|landlord|possess|unlawful\s+detain|rent|lease)\b",
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


class PolkEvictionScraper(BaseCountyScraper):
    county_fips = COUNTY_FIPS
    source_name = "Polk County Clerk PRO — Eviction Cases (CC/SC)"
    indicator_types = ["eviction"]
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
                    logger.error("polk_eviction_init_failed", status=init_resp.status_code)
                    return
            except Exception as exc:
                logger.error("polk_eviction_init_error", error=str(exc)[:120])
                return

            token = _extract_verification_token(init_resp.text)

            total = 0
            seen_cases: set[str] = set()
            prefixes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

            for prefix in prefixes:
                await self._rate_limit_sleep()
                try:
                    for case_type in _EVICTION_CASE_TYPES:
                        results = await self._search_cases(client, token, prefix, case_type)
                        for case in results:
                            case_num = case.get("case_number", "")
                            if not case_num or case_num in seen_cases:
                                continue

                            # Verify it looks like an eviction case
                            title = case.get("title", "")
                            case_code = case.get("case_type", "").upper()
                            if case_code not in _EVICTION_CASE_TYPES:
                                if not _EVICTION_TITLE_RE.search(title):
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
                    logger.debug("polk_eviction_search_error", prefix=prefix, error=str(exc)[:80])
                    continue

            if total == 0:
                logger.warning("polk_eviction_no_records")
            else:
                logger.info("polk_eviction_complete", total_yielded=total, cases_seen=len(seen_cases))

    async def _search_cases(
        self, client: httpx.AsyncClient, token: str | None,
        last_name_prefix: str, case_type: str,
    ) -> list[dict]:
        form_data = {
            "LastName": last_name_prefix,
            "SearchType": case_type,
        }
        if token:
            form_data["__RequestVerificationToken"] = token

        try:
            resp = await client.post(SEARCH_URL, data=form_data, timeout=30.0)
            if resp.status_code != 200:
                return []
            return _parse_search_results(resp.text, case_type)
        except Exception as exc:
            logger.debug("polk_eviction_search_failed", prefix=last_name_prefix, error=str(exc)[:80])
            return []

    def _build_record(self, case: dict) -> RawIndicatorRecord | None:
        address_raw = (case.get("address") or "").strip()
        if not address_raw or not address_raw[0:1].isdigit():
            return None

        if "FL" not in address_raw.upper():
            address_raw = f"{address_raw}, FL"

        # For evictions: defendant = tenant, plaintiff = landlord/owner
        plaintiff = (case.get("plaintiff") or "").strip().title()
        defendant = (case.get("party_name") or "").strip().title()
        filing_date = _parse_date(case.get("filing_date", ""))
        case_number = case.get("case_number", "")

        return RawIndicatorRecord(
            indicator_type="eviction",
            address_raw=address_raw,
            county_fips=self.county_fips,
            owner_name=plaintiff or None,  # landlord = property owner
            filing_date=filing_date,
            case_number=case_number,
            source_url=SOURCE_URL,
            raw_payload={
                "case_number": case_number,
                "case_type": case.get("case_type", ""),
                "title": case.get("title", ""),
                "plaintiff": plaintiff,
                "defendant": defendant,
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


def _parse_search_results(html: str, case_type: str) -> list[dict]:
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
                "plaintiff": cells[5] if len(cells) > 5 else "",
                "status": cells[6] if len(cells) > 6 else "",
                "case_type": case_type,
            })
    return results
