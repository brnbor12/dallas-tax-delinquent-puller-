"""
Hillsborough County Eviction Scraper (Florida - FIPS 12057)

Source: Hillsborough County Clerk of Circuit Court — Public Bulk Data
Portal: https://publicrec.hillsclerk.com/Civil/bulkdata/

Data:   Active LT (landlord-tenant) eviction case filings in Circuit/County Court.
        Covers Residential Eviction (possession only and past-due rent),
        Unlawful Detainer, and Delinquent Tenant cases.

Signal: indicator_type = "eviction"
        Active eviction filing = landlord-tenant conflict; landlord may be
        motivated to sell to exit a problem tenancy.

Strategy:
        The clerk publishes free monthly bulk CSV files:
          - Case File (one per month, ~1MB): CaseNbr, Style, CaseType, DtFile, ...
          - Party File (daily, ~2.5MB):     CaseNbr, Party, Name, Address1,
                                            Address2, City, State, ZIP

        We:
          1. Download the most recent Case File and previous month's if needed
             for a 90-day lookback.
          2. Filter for eviction case types (LT Residential Eviction*,
             LT Unlawful Detainer, Delinquent Tenant).
          3. Download the latest Party File.
          4. Join on CaseNbr to get the Defendant's address
             (Defendant = tenant = property address).
          5. Yield one RawIndicatorRecord per active eviction case.

Address: Defendant's address from the Party file = rental property address.
         Falls back to placeholder if no defendant address found.

Rate:   Simple HTTP downloads, no rate limit needed.
        Files are 1-5MB each — download once per run.
"""

from __future__ import annotations

import csv
import io
import re
import structlog
from datetime import date, datetime, timedelta
from typing import AsyncIterator

import httpx

from scrapers.base import BaseCountyScraper, RawIndicatorRecord

logger = structlog.get_logger(__name__)

COUNTY_FIPS = "12057"
BULK_BASE = "https://publicrec.hillsclerk.com/Civil/bulkdata"
SOURCE_URL = "https://publicrec.hillsclerk.com/Civil/bulkdata/"

LOOKBACK_DAYS = 90

# Case types to capture as evictions
_EVICTION_TYPES = (
    "LT Residential Eviction",
    "LT Unlawful Detainer",
    "Delinquent Tenant",
    "LT Commercial Eviction",
)

# Only ingest open/active cases
_ACTIVE_STATUSES = {"open", "active", "pending"}


def _eviction_case_type(case_type: str) -> bool:
    ct = case_type.strip()
    return any(ct.startswith(et) or et in ct for et in _EVICTION_TYPES)


def _parse_date(raw: str) -> date | None:
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _case_file_url(ref_date: date) -> str:
    """Build URL for the monthly case file closest to ref_date.
    Files are named MM-10-YYYY.csv (generated on the 10th of each month).
    """
    # Try current month's file, then previous month
    month = ref_date.month
    year = ref_date.year
    return f"{BULK_BASE}/Bulk%20Data%20Case%20File_%20{month:02d}-10-{year}.csv"


def _party_file_url(ref_date: date) -> str:
    """Build URL for a recent party file. Party files are generated weekdays."""
    return f"{BULK_BASE}/Bulk%20Data%20Party%20File_%20{ref_date.month:02d}-{ref_date.day:02d}-{ref_date.year}.csv"


async def _download_csv(client: httpx.AsyncClient, url: str) -> list[dict] | None:
    """Download a CSV URL and return parsed rows, or None on failure."""
    try:
        resp = await client.get(url, timeout=60.0, follow_redirects=True)
        if resp.status_code != 200:
            return None
        content = resp.content.decode("utf-8-sig", errors="replace")
        reader = csv.DictReader(io.StringIO(content))
        return list(reader)
    except Exception as exc:
        logger.debug("hillsborough_eviction_download_failed", url=url[:80], error=str(exc)[:80])
        return None


class HillsboroughEvictionScraper(BaseCountyScraper):
    county_fips = COUNTY_FIPS
    source_name = "Hillsborough County Clerk Civil Bulk Data — Evictions"
    indicator_types = ["eviction"]
    rate_limit_per_minute = 60  # simple HTTP downloads, not rate-sensitive

    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        today = date.today()
        cutoff = today - timedelta(days=LOOKBACK_DAYS)

        async with httpx.AsyncClient(
            headers={"User-Agent": "Mozilla/5.0 (compatible; motivated-seller-bot/1.0)"},
        ) as client:
            # --- Download Case Files (try current and previous month) ---
            cases: dict[str, dict] = {}  # CaseNbr -> row

            months_to_try = [today]
            # Also try previous month in case current isn't published yet
            first_day = today.replace(day=1)
            prev_month = (first_day - timedelta(days=1)).replace(day=1)
            months_to_try.append(prev_month)

            for ref_month in months_to_try:
                url = _case_file_url(ref_month)
                rows = await _download_csv(client, url)
                if rows:
                    logger.info("hillsborough_eviction_case_file", url=url[-40:], rows=len(rows))
                    for row in rows:
                        cnbr = (row.get("CaseNbr") or "").strip()
                        if not cnbr:
                            continue
                        if cnbr not in cases:
                            cases[cnbr] = row
                else:
                    logger.debug("hillsborough_eviction_case_file_missing", url=url[-40:])

            if not cases:
                logger.warning("hillsborough_eviction_no_cases", hint="No case files downloaded")
                return

            # Filter to eviction types within lookback window
            eviction_cases: dict[str, dict] = {}
            for cnbr, row in cases.items():
                ct = row.get("CaseType", "")
                if not _eviction_case_type(ct):
                    continue
                status = (row.get("CaseStatus") or "").strip().lower()
                if status and status not in _ACTIVE_STATUSES:
                    continue
                dt = _parse_date(row.get("DtFile", ""))
                if dt and dt < cutoff:
                    continue
                eviction_cases[cnbr] = row

            logger.info("hillsborough_eviction_filtered", total_cases=len(cases), eviction_cases=len(eviction_cases))

            if not eviction_cases:
                logger.warning("hillsborough_eviction_no_matching_cases")
                return

            # --- Download Party File (try recent weekdays) ---
            parties: dict[str, list[dict]] = {}  # CaseNbr -> list of party rows

            for days_back in range(0, 10):
                ref_date = today - timedelta(days=days_back)
                if ref_date.weekday() >= 5:  # skip weekends
                    continue
                url = _party_file_url(ref_date)
                rows = await _download_csv(client, url)
                if rows:
                    logger.info("hillsborough_eviction_party_file", url=url[-40:], rows=len(rows))
                    for row in rows:
                        cnbr = (row.get("CaseNbr") or "").strip()
                        if cnbr not in parties:
                            parties[cnbr] = []
                        parties[cnbr].append(row)
                    break  # got a party file, stop looking

            if not parties:
                logger.warning("hillsborough_eviction_no_party_file", hint="No party file found in last 10 days")

            # --- Build Records ---
            total = 0
            for cnbr, case_row in eviction_cases.items():
                record = self._build_record(cnbr, case_row, parties.get(cnbr, []))
                if record and await self.validate_record(record):
                    yield record
                    total += 1

        if total == 0:
            logger.warning("hillsborough_eviction_no_records")
        else:
            logger.info("hillsborough_eviction_complete", total_yielded=total)

    def _build_record(self, cnbr: str, case: dict, party_rows: list[dict]) -> RawIndicatorRecord | None:
        case_type = (case.get("CaseType") or "").strip()
        style = (case.get("Style") or "").strip()
        filing_date = _parse_date(case.get("DtFile", ""))

        # Find defendant address (tenant = property address)
        address_raw = None
        plaintiff_name = None
        defendant_name = None

        for party in party_rows:
            role = (party.get("Party") or "").strip().lower()
            name = (party.get("Name") or "").strip()
            addr1 = (party.get("Address1") or "").strip()
            addr2 = (party.get("Address2") or "").strip()
            city = (party.get("City") or "").strip()
            state = (party.get("State") or "FL").strip()
            zip_code = (party.get("ZIP") or "").strip()

            if role in ("plaintiff", "petitioner") and name:
                plaintiff_name = name.title()
            elif role in ("defendant", "respondent") and not address_raw:
                defendant_name = name.title() if name else None
                if addr1 and addr1[0].isdigit():
                    parts = [addr1]
                    if addr2 and addr2 not in (addr1, ""):
                        parts.append(addr2)
                    parts.append(f"{city}, {state} {zip_code}".strip())
                    address_raw = ", ".join(p for p in parts if p)

        if not address_raw:
            return None  # Skip cases without a real property address

        # Plaintiff = landlord = property owner for evictions
        owner_name = plaintiff_name

        return RawIndicatorRecord(
            indicator_type="eviction",
            address_raw=address_raw,
            county_fips=self.county_fips,
            owner_name=owner_name,
            filing_date=filing_date,
            case_number=cnbr,
            source_url=SOURCE_URL,
            raw_payload={
                "case_number": cnbr,
                "case_type": case_type,
                "style": style[:200],
                "status": (case.get("CaseStatus") or "").strip(),
                "division": (case.get("Division") or "").strip(),
                "judge": (case.get("Judge") or "").strip(),
                "plaintiff": plaintiff_name or "",
                "defendant": defendant_name or "",
                "filing_date": case.get("DtFile", ""),
            },
        )
