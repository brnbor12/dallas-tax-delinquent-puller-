"""
Polk County Lis Pendens Scraper (Florida - FIPS 12105)

Source: Polk County Clerk of Courts — SearchNG Official Records
Portal: https://apps.polkcountyclerk.net/SearchNG_Application/

Data:   Lis Pendens documents recorded in Official Records.
        In Florida, a lis pendens is recorded to provide notice that a
        property is subject to pending litigation — most commonly mortgage
        foreclosure, but also HOA liens, code enforcement liens, and
        title disputes.

Signal: indicator_type = "lien"
        A recorded lis pendens = property is in legal jeopardy. The owner
        may be motivated to sell before the litigation concludes.

Strategy:
        SearchNG is an ASP.NET application for searching recorded documents.
        We search by document type "LIS PENDENS" with a date range filter
        to find recently recorded LP instruments, then extract party names
        and legal descriptions.

        Since LP documents contain legal descriptions (not street addresses),
        we store a placeholder address with the legal description snippet.
        Downstream address enrichment via the Polk PA parcel API can resolve
        the legal description to a street address.

Rate:   Conservative — 10 req/min. The server is known to be slow.
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
SEARCHNG_BASE = "https://apps.polkcountyclerk.net/SearchNG_Application"
SOURCE_URL = "https://apps.polkcountyclerk.net/SearchNG_Application/"

LOOKBACK_DAYS = 90


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


class PolkLisPendensScraper(BaseCountyScraper):
    county_fips = COUNTY_FIPS
    source_name = "Polk County Clerk SearchNG — Lis Pendens (Official Records)"
    indicator_types = ["lien"]
    rate_limit_per_minute = 10

    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        today = date.today()
        start_date = today - timedelta(days=LOOKBACK_DAYS)

        async with httpx.AsyncClient(
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
            follow_redirects=True,
            timeout=60.0,
        ) as client:
            # Initialize session
            try:
                init_resp = await client.get(SEARCHNG_BASE)
                if init_resp.status_code != 200:
                    logger.error("polk_lis_pendens_init_failed", status=init_resp.status_code)
                    return
            except Exception as exc:
                logger.error("polk_lis_pendens_init_error", error=str(exc)[:120])
                return

            # Search for lis pendens documents in date range
            total = 0
            seen_instruments: set[str] = set()

            try:
                results = await self._search_documents(
                    client, start_date, today,
                )
                for doc in results:
                    inst_num = doc.get("instrument_number", "")
                    if not inst_num or inst_num in seen_instruments:
                        continue

                    seen_instruments.add(inst_num)
                    record = self._build_record(doc)
                    if record and await self.validate_record(record):
                        yield record
                        total += 1

            except Exception as exc:
                logger.error("polk_lis_pendens_search_error", error=str(exc)[:120])

            if total == 0:
                logger.warning("polk_lis_pendens_no_records")
            else:
                logger.info("polk_lis_pendens_complete", total_yielded=total)

    async def _search_documents(
        self, client: httpx.AsyncClient,
        start_date: date, end_date: date,
    ) -> list[dict]:
        """Search SearchNG for lis pendens documents in a date range."""
        search_url = f"{SEARCHNG_BASE}/Search"

        form_data = {
            "DocType": "LIS PENDENS",
            "StartDate": start_date.strftime("%m/%d/%Y"),
            "EndDate": end_date.strftime("%m/%d/%Y"),
        }

        try:
            await self._rate_limit_sleep()
            resp = await client.post(search_url, data=form_data, timeout=60.0)
            if resp.status_code != 200:
                logger.warning("polk_lis_pendens_search_failed", status=resp.status_code)
                return []
            return _parse_searchng_results(resp.text)
        except Exception as exc:
            logger.error("polk_lis_pendens_search_error", error=str(exc)[:120])
            return []

    def _build_record(self, doc: dict) -> RawIndicatorRecord | None:
        # Lis pendens have legal descriptions, not street addresses
        # Use the legal description as address_raw for downstream enrichment
        legal_desc = (doc.get("legal_description") or "").strip()
        grantor = (doc.get("grantor") or "").strip().title()
        grantee = (doc.get("grantee") or "").strip().title()

        # Try to build an address-like string from the legal description
        # The ingestor will attempt geocoding or parcel lookup
        if legal_desc:
            address_raw = f"LP {doc.get('instrument_number', '')} — {legal_desc[:150]}, Polk County, FL"
        else:
            return None  # No legal desc = no way to locate the property

        recording_date = _parse_date(doc.get("recording_date", ""))
        instrument = doc.get("instrument_number", "")

        return RawIndicatorRecord(
            indicator_type="lien",
            address_raw=address_raw,
            county_fips=self.county_fips,
            owner_name=grantor or None,  # grantor = property owner
            filing_date=recording_date,
            case_number=instrument,
            source_url=SOURCE_URL,
            raw_payload={
                "instrument_number": instrument,
                "doc_type": "LIS PENDENS",
                "grantor": grantor,
                "grantee": grantee,
                "recording_date": doc.get("recording_date", ""),
                "legal_description": legal_desc[:300],
                "book": doc.get("book", ""),
                "page": doc.get("page", ""),
                "consideration": doc.get("consideration", ""),
            },
        )


def _parse_searchng_results(html: str) -> list[dict]:
    """Parse SearchNG result HTML into document dicts."""
    results = []

    row_pattern = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL)
    cell_pattern = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL)

    rows = list(row_pattern.finditer(html))
    # Skip header row(s)
    for row_match in rows[1:]:
        row_html = row_match.group(1)
        cells = [
            re.sub(r"<[^>]+>", "", c.group(1)).strip()
            for c in cell_pattern.finditer(row_html)
        ]
        if len(cells) >= 4:
            results.append({
                "instrument_number": cells[0] if cells else "",
                "doc_type": cells[1] if len(cells) > 1 else "",
                "recording_date": cells[2] if len(cells) > 2 else "",
                "legal_description": cells[3] if len(cells) > 3 else "",
                "grantor": cells[4] if len(cells) > 4 else "",
                "grantee": cells[5] if len(cells) > 5 else "",
                "book": cells[6] if len(cells) > 6 else "",
                "page": cells[7] if len(cells) > 7 else "",
                "consideration": cells[8] if len(cells) > 8 else "",
            })

    return results
