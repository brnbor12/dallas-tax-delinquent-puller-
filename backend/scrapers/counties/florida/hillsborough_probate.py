"""
Hillsborough County Probate Scraper (Florida - FIPS 12057)

Source: Hillsborough County Clerk of Circuit Court — Probate Daily Filings
Portal: https://publicrec.hillsclerk.com/Probate/dailyfilings/

Data:   Probate case filings — estates being administered through the court.
        The decedent's address (= their primary property) is directly included.

Signal: indicator_type = "probate"
        Estate filings indicate the property owner has died; heirs may be
        motivated sellers who want to liquidate inherited real estate quickly.

Strategy:
        The clerk publishes free daily CSV files:
          ProbateFiling_YYYYMMDD.csv
          Columns: CaseCategory, CaseTypeDescription, CaseNumber, Title,
                   FilingDate, PartyType, FirstName, MiddleName,
                   LastName/CompanyName, DateofDeath, PartyAddress, Attorney

        We download the last LOOKBACK_DAYS of daily files, filter for
        Decedent party rows (which contain the property address and date of
        death), deduplicate by CaseNumber, and yield one record per estate.

Address: PartyAddress from the Decedent row = decedent's home (the property).

Rate:   Simple HTTP downloads — files are ~20KB each.
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
PROBATE_BASE = "https://publicrec.hillsclerk.com/Probate/dailyfilings"
SOURCE_URL = "https://publicrec.hillsclerk.com/Probate/dailyfilings/"

LOOKBACK_DAYS = 90


def _parse_date(raw: str) -> date | None:
    if not raw or not raw.strip():
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    return None


class HillsboroughProbateScraper(BaseCountyScraper):
    county_fips = COUNTY_FIPS
    source_name = "Hillsborough County Clerk Probate Daily Filings"
    indicator_types = ["probate"]
    rate_limit_per_minute = 60

    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        today = date.today()
        cutoff = today - timedelta(days=LOOKBACK_DAYS)

        # Collect all decedent rows from recent daily files, keyed by CaseNumber
        # (multiple daily files may reference the same case — deduplicate)
        estates: dict[str, dict] = {}  # CaseNumber -> best decedent row

        async with httpx.AsyncClient(
            headers={"User-Agent": "Mozilla/5.0 (compatible; motivated-seller-bot/1.0)"},
        ) as client:
            for days_back in range(0, LOOKBACK_DAYS + 1):
                ref_date = today - timedelta(days=days_back)
                url = f"{PROBATE_BASE}/ProbateFiling_{ref_date.strftime('%Y%m%d')}.csv"
                try:
                    resp = await client.get(url, timeout=30.0, follow_redirects=True)
                    if resp.status_code != 200:
                        continue
                    content = resp.content.decode("utf-8-sig", errors="replace")
                    if len(content) < 200:
                        continue  # empty/header-only file (weekends return 151 bytes)

                    reader = csv.DictReader(io.StringIO(content))
                    rows = list(reader)
                    if not rows:
                        continue

                    file_new = 0
                    for row in rows:
                        party_type = (row.get("PartyType") or "").strip().lower()
                        if party_type != "decedent":
                            continue

                        case_num = (row.get("CaseNumber") or "").strip()
                        if not case_num:
                            continue

                        filing_date = _parse_date(row.get("FilingDate", ""))
                        if filing_date and filing_date < cutoff:
                            continue

                        # Keep the most complete decedent row per case
                        existing = estates.get(case_num)
                        addr = (row.get("PartyAddress") or "").strip()
                        if not existing or (addr and not (existing.get("PartyAddress") or "").strip()):
                            estates[case_num] = row
                            file_new += 1

                    if file_new > 0:
                        logger.debug("hillsborough_probate_file_loaded",
                                     date=ref_date.isoformat(), new_estates=file_new)
                except Exception as exc:
                    logger.debug("hillsborough_probate_file_error",
                                 date=ref_date.isoformat(), error=str(exc)[:60])

        logger.info("hillsborough_probate_estates_found", count=len(estates))

        total = 0
        for case_num, row in estates.items():
            record = self._build_record(case_num, row)
            if record and await self.validate_record(record):
                yield record
                total += 1

        if total == 0:
            logger.warning("hillsborough_probate_no_records")
        else:
            logger.info("hillsborough_probate_complete", total_yielded=total)

    def _build_record(self, case_num: str, row: dict) -> RawIndicatorRecord | None:
        address_raw = (row.get("PartyAddress") or "").strip()
        if not address_raw or not address_raw[0].isdigit():
            # No valid street address — skip instead of creating placeholder
            logger.debug("hillsborough_probate_no_address", case_num=case_num)
            return None

        first = (row.get("FirstName") or "").strip()
        middle = (row.get("MiddleName") or "").strip()
        last = (row.get("LastName/CompanyName") or "").strip()
        decedent_name = " ".join(p for p in [first, middle, last] if p).title()

        filing_date = _parse_date(row.get("FilingDate", ""))
        date_of_death = _parse_date(row.get("DateofDeath", ""))

        title = (row.get("Title") or "").strip()
        case_type = (row.get("CaseTypeDescription") or "").strip()

        return RawIndicatorRecord(
            indicator_type="probate",
            address_raw=address_raw,
            county_fips=self.county_fips,
            owner_name=decedent_name or None,
            filing_date=filing_date,
            case_number=case_num,
            source_url=SOURCE_URL,
            raw_payload={
                "case_number": case_num,
                "case_type": case_type,
                "title": title[:200],
                "decedent_name": decedent_name,
                "date_of_death": row.get("DateofDeath", ""),
                "filing_date": row.get("FilingDate", ""),
                "attorney": (row.get("Attorney") or "").strip(),
            },
        )
