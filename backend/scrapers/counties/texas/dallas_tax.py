"""
Dallas County Tax Delinquent Scraper (Texas - FIPS 48113)

Source: Dallas County Tax Office — Tax Roll (TRW) File
Data:   All property tax accounts; delinquent accounts identified by roll code
URL:    https://www.dallascounty.org/departments/tax/tax-roll.php
File:   https://www.dallascounty.org/Assets/uploads/docs/tax/trw/trwfile.720880.zip

The TRW (Tax Roll Workfile) is an ASCII fixed-width file updated every Friday.
It contains one record per account+year+jurisdiction combination. Properties
with unpaid prior-year balances are flagged as tax_delinquent.

Field positions come from the official layout doc embedded in the sample ZIP:
  https://www.dallascounty.org/Assets/uploads/docs/tax/trw/trwfile.441510_SampleFile.zip
  (extract trwfile_layout.txt.* for the full spec)

Known confirmed positions (from Dallas County Tax Office docs):
  Field 1  ACCOUNT      cols  1– 34  (34 chars)  — Account identifier
  Field 2  YEAR         cols 35– 38  ( 4 chars)  — Tax year (e.g. 2024)
  Field 3  JURISDICTION cols 39– 42  ( 4 chars)  — Taxing entity code
  Field 4  TAX-UNIT-ACCT cols 43–76  (34 chars)  — Account in that jurisdiction

Remaining fields are estimated from the Dallas County TRW FAQ and typical Texas
county TRW layouts. VERIFY against trwfile_layout.txt before relying on this data.
Update FIELD_MAP below if positions are wrong.

Rate limit: 2 req/min — large file download, be conservative.
"""

from __future__ import annotations

import io
import re
import structlog
import zipfile
from datetime import date, datetime
from typing import AsyncIterator

import httpx

from scrapers.base import BaseCountyScraper, RawIndicatorRecord

logger = structlog.get_logger(__name__)

# Index page — the actual ZIP filename changes weekly
TRW_INDEX_URL = "https://www.dallascounty.org/departments/tax/tax-roll.php"
TRW_BASE_URL = "https://www.dallascounty.org"

# ---------------------------------------------------------------------------
# Field layout — 1-indexed, end column inclusive.
# Verified against trwfile_layout.txt from the official Dallas County sample ZIP.
# ---------------------------------------------------------------------------
FIELD_MAP: dict[str, tuple[int, int]] = {
    "account":       (1,   34),   # ✓ unique property account id
    "year":          (35,  38),   # ✓ tax year
    "jurisdiction":  (39,  42),   # ✓ taxing entity code
    "tax_unit_acct": (43,  76),   # ✓ appraisal district account no (APN)
    "levy":          (77,  87),   # levy amount (implied decimal, 11 digits)
    "date_paid":     (93,  100),  # YYYYMMDD or blank if unpaid
    "levy_balance":  (111, 121),  # remaining levy due (implied decimal) — USE THIS for delinquency
    "suit":          (122, 122),  # 'A,J,L' if suit pending
    "cause_no":      (123, 162),  # cause number of suit (40 chars)
    "owner":         (226, 265),  # owner name (40 chars)
    "addr2":         (266, 305),  # owner mailing street (40 chars)
    "addr3":         (306, 345),  # owner mailing address line 3
    "city":          (386, 425),  # owner mailing city (40 chars)
    "state":         (426, 427),  # owner mailing state (2 chars)
    "zip":           (428, 439),  # owner mailing zip (12 chars)
    "roll_code":     (440, 440),  # property roll code (1 char)
    "parcel_no":     (441, 448),  # property street number (8 chars)
    "parcel_name":   (449, 488),  # property street name + city abbrev (40 chars)
    "tot_amt_due":   (490, 500),  # total amt due incl. penalties — 0 for current-year accounts
}

# Dallas County FIPS
COUNTY_FIPS = "48113"
CURRENT_YEAR = datetime.now().year
# Only flag delinquencies from the last 10 years — older ones are typically
# tied up in suits/judgments and not actionable for investors.
DELINQUENT_YEAR_CUTOFF = CURRENT_YEAR - 10


class DallasCountyTaxRollScraper(BaseCountyScraper):
    county_fips = COUNTY_FIPS
    source_name = "Dallas County Tax Roll (TRW) — Delinquent Accounts"
    indicator_types = ["tax_delinquent"]
    rate_limit_per_minute = 2  # Large file; download only a few times per run

    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        """
        Download the TRW ZIP, extract the flat data file, and yield one
        RawIndicatorRecord per delinquent *property* (deduplicated by APN).

        The TRW has one row per account-year-jurisdiction, so the same property
        appears many times. We aggregate: sum up all delinquent-year balances
        and yield a single record with the total owed. We limit to the last 10
        years to focus on actionable leads.
        """
        trw_url = await self._find_current_trw_url()
        if not trw_url:
            logger.error("dallas_tax_no_url_found", index=TRW_INDEX_URL)
            return

        zip_bytes = await self._download_trw_zip(trw_url)
        if not zip_bytes:
            logger.error("dallas_tax_zip_download_failed")
            return

        flat_file = self._extract_flat_file(zip_bytes)
        if flat_file is None:
            logger.error("dallas_tax_flat_file_not_found_in_zip")
            return

        line_count = flat_file.count('\n')
        logger.info("dallas_tax_parsing_flat_file", size_bytes=len(flat_file), lines=line_count)

        # Aggregate delinquent amounts per account (APN)
        # key: apn → {amount, most_recent_year, metadata for record construction}
        accounts: dict[str, dict] = {}

        for line_num, line in enumerate(flat_file.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                entry = self._parse_line_aggregate(line, line_num)
                if entry:
                    apn = entry["apn"]
                    if apn not in accounts:
                        accounts[apn] = entry
                    else:
                        # Accumulate balance; keep most recent year's metadata
                        accounts[apn]["amount"] += entry["amount"]
                        if entry["year"] > accounts[apn]["year"]:
                            accounts[apn].update({k: v for k, v in entry.items() if k != "amount"})
                            accounts[apn]["amount"] += entry["amount"]
            except Exception as exc:
                logger.warning("dallas_tax_line_parse_failed", line=line_num, error=str(exc))

            if line_num % 500_000 == 0:
                logger.info("dallas_tax_progress", line=line_num, accounts=len(accounts))

        logger.info("dallas_tax_aggregated", total_delinquent_accounts=len(accounts))

        for entry in accounts.values():
            record = self._build_record(entry)
            if record and await self.validate_record(record):
                yield record

    async def _find_current_trw_url(self) -> str | None:
        """Scrape the index page to find the current week's TRW ZIP URL."""
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                resp = await client.get(TRW_INDEX_URL)
                resp.raise_for_status()

            # Find the data ZIP (exclude sample file)
            links = re.findall(
                r'href=["\']([^"\']*trw[^"\']*\.zip[^"\']*)["\']',
                resp.text, re.IGNORECASE,
            )
            data_links = [l for l in links if "sample" not in l.lower()]
            if not data_links:
                logger.error("dallas_tax_no_zip_link", all_links=links)
                return None

            url = data_links[0]
            if url.startswith("/"):
                url = TRW_BASE_URL + url
            logger.info("dallas_tax_found_url", url=url)
            return url
        except Exception as exc:
            logger.error("dallas_tax_index_fetch_failed", error=str(exc))
            return None

    async def _download_trw_zip(self, url: str) -> bytes | None:
        """Download the weekly TRW ZIP file (large file — up to 500 MB, 30+ min)."""
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=30, read=3600, write=30, pool=30),
                follow_redirects=True,
            ) as client:
                logger.info("dallas_tax_downloading_zip", url=url)
                chunks = []
                total = 0
                async with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_bytes(chunk_size=1024 * 1024):
                        chunks.append(chunk)
                        total += len(chunk)
                        if total % (50 * 1024 * 1024) == 0:  # log every 50 MB
                            logger.info("dallas_tax_download_progress", mb=total // 1024 // 1024)
                data = b"".join(chunks)
                logger.info("dallas_tax_download_complete", mb=len(data) // 1024 // 1024)
                await self._rate_limit_sleep()
                return data
        except httpx.HTTPError as exc:
            logger.error("dallas_tax_download_failed", url=url, error=str(exc))
            return None

    def _extract_flat_file(self, zip_bytes: bytes) -> str | None:
        """
        Extract the main data file from the ZIP archive.

        The ZIP contains files like:
          flat404.DALLASCOUNTY.YYYYMMDD.NNNNNN  ← main data file
          tcs404p.DALLASCOUNTY.YYYYMMDD.NNNNNN
          Important_read_me.DALLASCOUNTY.NNNNNN
          trwfile_layout.txt.NNNNNN              ← field layout spec
        """
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                # Log all files in the ZIP for debugging
                names = zf.namelist()
                logger.info("dallas_tax_zip_contents", files=names)

                # Find the flat data file (contains "flat404" — may have directory prefix)
                flat_name = next(
                    (n for n in names if "flat404" in n.lower()), None
                )
                if flat_name is None:
                    logger.error("dallas_tax_no_flat_file", zip_contents=names)
                    return None

                raw = zf.read(flat_name)
                # Try common encodings for legacy ASCII files
                for enc in ("latin-1", "cp1252", "utf-8"):
                    try:
                        return raw.decode(enc)
                    except UnicodeDecodeError:
                        continue
                return None

        except zipfile.BadZipFile as exc:
            logger.error("dallas_tax_bad_zip", error=str(exc))
            return None

    def _parse_line_aggregate(self, line: str, line_num: int) -> dict | None:
        """
        Parse one TRW line into an intermediate dict for aggregation.
        Returns None if the record is not delinquent or out of range.
        """
        if len(line) < 500:
            return None

        def col(name: str) -> str:
            start, end = FIELD_MAP[name]
            return line[start - 1 : end].strip()

        account = col("account")
        if not account:
            return None

        try:
            tax_year = int(col("year"))
        except ValueError:
            return None

        # Only recent, prior-year, unpaid accounts
        if tax_year >= CURRENT_YEAR or tax_year < DELINQUENT_YEAR_CUTOFF:
            return None
        if col("date_paid"):
            return None

        # levy_balance = remaining levy (always set when unpaid)
        # tot_amt_due = levy + penalties (0 for current-year, shows full delinquency for prior years)
        try:
            amount = int(col("levy_balance")) / 100.0
        except ValueError:
            return None

        # Minimum $100 balance per row — skip trivially small amounts
        if amount < 100.0:  # $100 levy_balance per year/jurisdiction
            return None

        # Use tot_amt_due (with penalties) if available, otherwise levy_balance
        tot_raw = col("tot_amt_due")
        try:
            tot = int(tot_raw) / 100.0
            if tot > 0:
                amount = tot
        except ValueError:
            pass

        parcel_no = col("parcel_no")
        parcel_name = col("parcel_name")
        mail_city = col("city")
        mail_state = col("state") or "TX"
        mail_zip = col("zip")[:5]
        street_name = parcel_name.split(",")[0].strip() if parcel_name else ""

        if parcel_no and street_name:
            address_raw = f"{parcel_no} {street_name}, {mail_city or 'Dallas'}, {mail_state} {mail_zip}".strip()
        else:
            mail_addr = col("addr2")
            if not mail_addr:
                return None
            address_raw = f"{mail_addr}, {mail_city or 'Dallas'}, {mail_state} {mail_zip}".strip()

        apn = col("tax_unit_acct") or account.strip()

        return {
            "apn": apn,
            "account": account.strip(),
            "year": tax_year,
            "amount": amount,
            "address_raw": address_raw,
            "owner_name": col("owner") or None,
            "mail_addr": col("addr2") or None,
            "mail_city": mail_city or None,
            "mail_state": mail_state or None,
            "mail_zip": mail_zip or None,
            "roll_code": col("roll_code"),
            "suit": col("suit"),
        }

    def _build_record(self, entry: dict) -> RawIndicatorRecord | None:
        """Convert an aggregated account dict into a RawIndicatorRecord."""
        filing_date: date | None = None
        try:
            filing_date = date(entry["year"] + 1, 2, 1)
        except ValueError:
            pass

        return RawIndicatorRecord(
            indicator_type="tax_delinquent",
            address_raw=entry["address_raw"],
            county_fips=self.county_fips,
            apn=entry["apn"],
            owner_name=entry["owner_name"],
            owner_mailing_address=entry["mail_addr"],
            owner_mailing_city=entry["mail_city"],
            owner_mailing_state=entry["mail_state"],
            owner_mailing_zip=entry["mail_zip"],
            amount=entry["amount"],
            filing_date=filing_date,
            case_number=entry["account"],
            source_url=TRW_INDEX_URL,
            raw_payload={
                "account": entry["account"],
                "most_recent_year": entry["year"],
                "roll_code": entry["roll_code"],
                "suit": entry["suit"],
            },
        )
