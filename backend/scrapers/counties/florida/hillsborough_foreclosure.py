"""
Hillsborough County Mortgage Foreclosure Scraper (Florida - FIPS 12057)

Source: Hillsborough County Clerk of Circuit Court — Civil Undisposed Weekly CSVs
Portal: https://publicrec.hillsclerk.com/Civil/undisposed/

Data:   Active mortgage foreclosure cases filed in Circuit Court.
        Each weekly CSV covers new filings for that 7-day window and includes
        party addresses inline — no secondary party-file join needed.

Signal: indicator_type = "pre_foreclosure"

Strategy:
  1. Fetch the directory listing at /Civil/undisposed/
  2. Download the most recent N weekly files (covering ~6-month lookback).
  3. Filter rows where CaseTypeDescription contains "Mortgage Foreclosure"
     and PartyType == "Defendant".
  4. Per case, pick the best defendant address: first FL individual (not
     a bank/HOA/government) with a digit-starting Address1.
  5. Yield one RawIndicatorRecord per unique case.

Address: Defendant (homeowner) Address1/City/State/Zip from the CSV.
         Cases without a usable FL street address are skipped.
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
UNDISPOSED_BASE = "https://publicrec.hillsclerk.com/Civil/undisposed"
SOURCE_URL = "https://publicrec.hillsclerk.com/Civil/undisposed/"

LOOKBACK_DAYS = 180  # 6 months of weekly files

# Substrings that indicate a party is NOT the homeowner we want
_SKIP_PARTY_PATTERNS = re.compile(
    r"\b(bank|trust|mortgage|loan|servicing|llc|inc\b|corp|association|hoa|"
    r"unknown|portfolio|recovery|housing|finance|county|clerk|government|"
    r"secretary|department|commissioner|attorney|trustee of|as trustee|"
    r"federal|national|fha|va\b|veterans|freddie|fannie|mers)\b",
    re.IGNORECASE,
)


def _parse_date(raw: str) -> date | None:
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _is_good_defendant(row: dict) -> bool:
    """Return True if this defendant row looks like the actual homeowner."""
    state = (row.get("State") or "").strip()
    addr1 = (row.get("Address1") or "").strip()
    last = (row.get("LastName") or "").strip()
    if state != "FL":
        return False
    if not addr1 or not addr1[0].isdigit():
        return False
    # Skip corporate/institutional parties
    name_check = f"{last} {row.get('FirstName', '')}"
    if _SKIP_PARTY_PATTERNS.search(name_check):
        return False
    return True


def _build_address(row: dict) -> str:
    addr1 = (row.get("Address1") or "").strip().rstrip(",")
    addr2 = (row.get("Address2") or "").strip().rstrip(",")
    city = (row.get("City") or "").strip().rstrip(",")
    state = (row.get("State") or "FL").strip()
    zip_code = (row.get("Zip") or "").strip()
    parts = [addr1]
    if addr2 and addr2 != addr1:
        parts.append(addr2)
    parts.append(f"{city}, {state} {zip_code}".strip())
    return ", ".join(p for p in parts if p)


async def _download_csv(client: httpx.AsyncClient, url: str) -> list[dict] | None:
    try:
        resp = await client.get(url, timeout=60.0, follow_redirects=True)
        if resp.status_code != 200:
            return None
        content = resp.content.decode("utf-8-sig", errors="replace")
        reader = csv.DictReader(io.StringIO(content), delimiter="|")
        return list(reader)
    except Exception as exc:
        logger.debug("hillsborough_fc_download_failed", url=url[-60:], error=str(exc)[:80])
        return None


async def _list_undisposed_files(client: httpx.AsyncClient) -> list[str]:
    """Fetch directory listing and return CSV filenames, most recent first."""
    try:
        resp = await client.get(UNDISPOSED_BASE + "/", timeout=30.0)
        if resp.status_code != 200:
            return []
        urls = re.findall(r'href="([^"]*CivilUndisposed[^"]*\.csv)"', resp.text, re.IGNORECASE)
        # Sort descending (filenames are YYYYMMDD date-ordered)
        return sorted(set(urls), reverse=True)
    except Exception as exc:
        logger.warning("hillsborough_fc_dir_listing_failed", error=str(exc)[:80])
        return []


class HillsboroughForeclosureScraper(BaseCountyScraper):
    county_fips = COUNTY_FIPS
    source_name = "Hillsborough County Clerk Civil Undisposed — Mortgage Foreclosure"
    indicator_types = ["pre_foreclosure"]
    rate_limit_per_minute = 60

    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        today = date.today()
        cutoff = today - timedelta(days=LOOKBACK_DAYS)

        async with httpx.AsyncClient(
            headers={"User-Agent": "Mozilla/5.0 (compatible; motivated-seller-bot/1.0)"},
        ) as client:
            filenames = await _list_undisposed_files(client)
            if not filenames:
                logger.warning("hillsborough_fc_no_dir_listing")
                return

            # Accumulate rows per case across weekly files
            cases: dict[str, dict] = {}  # case_number -> {"_rows": [...], "_case_row": row}

            files_processed = 0
            for fname in filenames:
                # Check if file is within lookback window
                date_match = re.search(r"(\d{8})_(\d{8})", fname)
                if date_match:
                    try:
                        file_end = datetime.strptime(date_match.group(2), "%Y%m%d").date()
                        if file_end < cutoff:
                            break  # Sorted newest-first; stop when out of window
                    except ValueError:
                        pass

                # hrefs may be full paths (/Civil/undisposed/...) or bare filenames
                if fname.startswith("/"):
                    url = f"https://publicrec.hillsclerk.com{fname}"
                else:
                    url = f"{UNDISPOSED_BASE}/{fname}"
                rows = await _download_csv(client, url)
                if rows is None:
                    continue

                logger.info("hillsborough_fc_undisposed_file", file=fname[-40:], rows=len(rows))
                files_processed += 1

                for row in rows:
                    if "Mortgage Foreclosure" not in (row.get("CaseTypeDescription") or ""):
                        continue
                    cnbr = (row.get("CaseNumber") or "").strip().strip('"')
                    if not cnbr:
                        continue
                    if cnbr not in cases:
                        cases[cnbr] = {"_rows": [], "_case_row": row}
                    cases[cnbr]["_rows"].append(row)

            logger.info(
                "hillsborough_fc_files_processed",
                files=files_processed,
                unique_cases=len(cases),
            )

            if not cases:
                logger.warning("hillsborough_fc_no_cases")
                return

            total = 0
            for cnbr, case_data in cases.items():
                record = self._build_record(cnbr, case_data)
                if record and await self.validate_record(record):
                    yield record
                    total += 1

        if total == 0:
            logger.warning("hillsborough_fc_no_records")
        else:
            logger.info("hillsborough_fc_complete", total_yielded=total)

    def _build_record(self, cnbr: str, case_data: dict) -> RawIndicatorRecord | None:
        all_rows = case_data["_rows"]
        case_row = case_data["_case_row"]

        case_type = (case_row.get("CaseTypeDescription") or "").strip().strip('"')
        title = (case_row.get("Title") or "").strip().strip('"')
        filing_date = _parse_date((case_row.get("FilingDate") or "").strip().strip('"'))

        # Find plaintiff (the lender)
        plaintiff_name = None
        for row in all_rows:
            ptype = (row.get("PartyType") or "").strip().lower()
            if ptype in ("plaintiff", "petitioner"):
                last = (row.get("LastName") or "").strip()
                first = (row.get("FirstName") or "").strip()
                plaintiff_name = f"{first} {last}".strip().title() or None
                break

        # Pass 1: FL individual homeowner defendant
        address_raw = None
        owner_name = None
        for row in all_rows:
            ptype = (row.get("PartyType") or "").strip().lower()
            if ptype not in ("defendant", "respondent"):
                continue
            if _is_good_defendant(row):
                last = (row.get("LastName") or "").strip()
                first = (row.get("FirstName") or "").strip()
                owner_name = f"{first} {last}".strip().title() or None
                address_raw = _build_address(row)
                break

        # Pass 2: any FL defendant with digit address
        if not address_raw:
            for row in all_rows:
                ptype = (row.get("PartyType") or "").strip().lower()
                if ptype not in ("defendant", "respondent"):
                    continue
                state = (row.get("State") or "").strip()
                addr1 = (row.get("Address1") or "").strip()
                if state == "FL" and addr1 and addr1[0].isdigit():
                    last = (row.get("LastName") or "").strip()
                    first = (row.get("FirstName") or "").strip()
                    owner_name = f"{first} {last}".strip().title() or None
                    address_raw = _build_address(row)
                    break

        if not address_raw:
            return None  # Skip cases without a usable property address

        return RawIndicatorRecord(
            indicator_type="pre_foreclosure",
            address_raw=address_raw,
            county_fips=self.county_fips,
            owner_name=owner_name,
            filing_date=filing_date,
            case_number=cnbr,
            source_url=SOURCE_URL,
            raw_payload={
                "case_number": cnbr,
                "case_type": case_type,
                "title": title[:200],
                "plaintiff": plaintiff_name or "",
                "defendant": owner_name or "",
                "filing_date": (case_row.get("FilingDate") or "").strip(),
            },
        )
