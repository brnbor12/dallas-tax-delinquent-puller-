"""
Harris County Absentee Owner Scraper (Texas - FIPS 48201)

Source: Harris County Appraisal District (HCAD) — Annual CAMA Bulk Data
URL:    https://download.hcad.org/data/CAMA/2025/Real_acct_owner.zip
Portal: https://download.hcad.org/

Data:   Real_acct_owner.zip — single zip with seven text files.
        We use real_acct.txt (tab-delimited, ~857 MB uncompressed), which
        contains both property site address and owner mailing address in one row.

Signal: indicator_type = "absentee_owner"
        A property is flagged when the owner's mailing address is out-of-state
        (mail_state != 'TX'), indicating the owner does not live at the property.
        Out-of-state owners are often more motivated to sell.

Scope:  Residential parcels (state_class in A1-A4) with a valid street number
        in site_addr_1. Excludes commercial, exempt, and vacant land.

Key columns used from real_acct.txt:
    acct         — Harris County parcel ID (APN)
    mailto       — owner name (mailing to)
    mail_addr_1  — owner mailing street address
    mail_city    — owner mailing city
    mail_state   — owner mailing state
    mail_zip     — owner mailing zip
    site_addr_1  — property street address
    site_addr_2  — property city
    site_addr_3  — property zip
    state_class  — property classification (A1=SFR, A2=duplex, A3=condo, A4=townhouse)
    assessed_val — HCAD assessed value

File size: 200 MB compressed → 857 MB uncompressed. Processed with streaming
csv reader to avoid loading the entire file into memory.

Schedule: Annual (HCAD updates yearly). Run once after HCAD releases data (~spring).
"""

from __future__ import annotations

import csv
import io
import zipfile
from typing import AsyncIterator

import httpx
import structlog

from scrapers.base import BaseCountyScraper, RawIndicatorRecord

logger = structlog.get_logger(__name__)

COUNTY_FIPS = "48201"
HCAD_ZIP_URL = "https://download.hcad.org/data/CAMA/2025/Real_acct_owner.zip"
SOURCE_URL = "https://download.hcad.org/"

# Residential state_class codes in HCAD
_RESIDENTIAL_CLASSES = {"A1", "A2", "A3", "A4"}

# Owner name tokens that indicate an entity (not individual)
_ENTITY_TOKENS = {
    "LLC", "L.L.C", "INC", "CORP", "TRUST", "TR", "ESTATE",
    "BANK", "MORTGAGE", "ASSOC", "ASSOCIATION", "PARTNERS", "LP", "LTD",
    "MANAGEMENT", "PROPERTIES", "HOLDINGS", "VENTURES", "GROUP",
}


def _owner_type(name: str) -> str:
    upper = name.upper()
    for token in _ENTITY_TOKENS:
        if token in upper:
            return "LLC"
    return "individual"


def _build_site_address(row: dict) -> str | None:
    """Build a full property address from real_acct.txt site fields."""
    street = (row.get("site_addr_1") or "").strip()
    city = (row.get("site_addr_2") or "").strip().title()
    zip_code = (row.get("site_addr_3") or "").strip()

    if not street or street.startswith("0 "):
        return None  # no valid address (e.g. "0 COMMERCE ST" = unaddressed parcel)

    parts = [street]
    if city:
        parts.append(city)
    parts.append("TX")
    if zip_code:
        parts.append(zip_code)
    return ", ".join(parts)


class HarrisAbsenteeOwnerScraper(BaseCountyScraper):
    """
    Downloads HCAD Real_acct_owner.zip and yields absentee owner records
    for residential properties with an out-of-state mailing address.

    Config keys (all optional):
        zip_url      — override the HCAD download URL (default: 2025 data)
        include_tx_absentee — if True, also include in-state TX absentee owners
                              (mail address != site address). Default: False.
    """

    county_fips = COUNTY_FIPS
    source_name = "Harris County Appraisal District — Absentee Owner (HCAD Bulk Data)"
    indicator_types = ["absentee_owner"]
    rate_limit_per_minute = 1  # single large download, no repeated requests

    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        zip_url = self.config.get("zip_url") or HCAD_ZIP_URL

        logger.info("hcad_absentee_download_start", url=zip_url)

        async with httpx.AsyncClient(
            headers={"User-Agent": "Mozilla/5.0 (compatible; motivated-seller-bot/1.0)"},
            follow_redirects=True,
            timeout=600.0,
        ) as client:
            resp = await client.get(zip_url)
            resp.raise_for_status()

        content = resp.content
        logger.info("hcad_absentee_download_complete", size_mb=len(content) // 1024 // 1024)

        total = skipped_non_res = skipped_no_addr = skipped_in_state = 0

        try:
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                with z.open("real_acct.txt") as raw_file:
                    reader = csv.DictReader(
                        io.TextIOWrapper(raw_file, encoding="latin-1"),
                        delimiter="\t",
                    )
                    for row in reader:
                        acct = (row.get("acct") or "").strip()
                        if not acct:
                            continue

                        # Filter: residential only
                        state_class = (row.get("state_class") or "").strip().upper()
                        if state_class not in _RESIDENTIAL_CLASSES:
                            skipped_non_res += 1
                            continue

                        # Filter: out-of-state mailing address
                        mail_state = (row.get("mail_state") or "").strip().upper()
                        if mail_state in ("TX", ""):
                            skipped_in_state += 1
                            continue

                        # Filter: valid site address
                        address_raw = _build_site_address(row)
                        if not address_raw:
                            skipped_no_addr += 1
                            continue

                        record = self._build_record(acct, row, address_raw, mail_state)
                        if record and await self.validate_record(record):
                            yield record
                            total += 1

        except zipfile.BadZipFile as exc:
            logger.error("hcad_absentee_bad_zip", error=str(exc))
            return

        logger.info(
            "hcad_absentee_complete",
            total_yielded=total,
            skipped_non_res=skipped_non_res,
            skipped_no_addr=skipped_no_addr,
            skipped_in_state=skipped_in_state,
        )

    def _build_record(
        self,
        acct: str,
        row: dict,
        address_raw: str,
        mail_state: str,
    ) -> RawIndicatorRecord | None:
        owner_name = (row.get("mailto") or "").strip().title()
        if not owner_name or owner_name.upper() in ("CURRENT OWNER", "UNKNOWN"):
            return None

        mail_addr1 = (row.get("mail_addr_1") or "").strip()
        mail_addr2 = (row.get("mail_addr_2") or "").strip()
        mail_city = (row.get("mail_city") or "").strip().title()
        mail_zip = (row.get("mail_zip") or "").strip()

        mailing_full = mail_addr1
        if mail_addr2:
            mailing_full = f"{mail_addr1}, {mail_addr2}"

        owner_t = _owner_type(owner_name)

        try:
            assessed = float(row.get("assessed_val") or 0) or None
        except (ValueError, TypeError):
            assessed = None

        return RawIndicatorRecord(
            indicator_type="absentee_owner",
            address_raw=address_raw,
            county_fips=self.county_fips,
            owner_name=owner_name,
            owner_mailing_address=mailing_full or None,
            owner_mailing_city=mail_city or None,
            owner_mailing_state=mail_state or None,
            owner_mailing_zip=mail_zip or None,
            owner_type=owner_t,
            apn=acct,
            amount=assessed,
            source_url=SOURCE_URL,
            raw_payload={
                "acct": acct,
                "owner_name": owner_name,
                "mailing_state": mail_state,
                "mailing_address": mailing_full,
                "mailing_city": mail_city,
                "mailing_zip": mail_zip,
                "state_class": row.get("state_class", ""),
                "site_addr_1": row.get("site_addr_1", ""),
                "site_addr_2": row.get("site_addr_2", ""),
                "site_addr_3": row.get("site_addr_3", ""),
                "assessed_val": row.get("assessed_val", ""),
                "tot_appr_val": row.get("tot_appr_val", ""),
            },
        )
