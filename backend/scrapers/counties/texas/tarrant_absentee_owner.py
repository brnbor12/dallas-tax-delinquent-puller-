"""
Tarrant County Absentee Owner Scraper (Texas - FIPS 48439)

Source: Tarrant Appraisal District (TAD) — Daily PropertyData Bulk Download
URL:    https://www.tad.org/content/data-download/PropertyData_R.ZIP
Portal: https://www.tad.org/resources/data-downloads

Data:   PropertyData_R.ZIP — residential-only pipe-delimited flat file.
        Revised daily. Includes owner name, mailing address, property site
        address, classification codes, and homestead exemption status.

Signal: indicator_type = "absentee_owner"
        A property is flagged when the owner's mailing state is NOT Texas,
        indicating an out-of-state owner who may be more motivated to sell.

Scope:  Residential parcels (Property_Class starting with A or B) with a
        valid digit-starting site address. Excludes commercial, exempt, and
        vacant land.

Key columns used from PropertyData_R (pipe-delimited, header row):
    Account_Num      — TAD parcel ID (APN)
    Owner_Name       — owner of record
    Owner_Address    — owner mailing street address
                       (first char may be bad-address flag: -, =, *)
    Owner_CityState  — combined "CITY STATE" field (e.g. "FORT WORTH TX")
    Owner_Zip        — 5-digit zip code
    Situs_Address    — property site address
    Property_Class   — classification code (A/A1-A5/B/B2-B4 = residential)
    Appraised_Value  — market value for tax purposes

Schedule: Daily (TAD updates daily). Celery Beat: monthly to avoid churn.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from typing import AsyncIterator

import httpx
import structlog

from scrapers.base import BaseCountyScraper, RawIndicatorRecord

logger = structlog.get_logger(__name__)

COUNTY_FIPS = "48439"
TAD_ZIP_URL = "https://www.tad.org/content/data-download/PropertyData_R.ZIP"
SOURCE_URL = "https://www.tad.org/resources/data-downloads"

# Property_Class values that are residential
# A/A1-A5 = SFR, mobile, condo, townhouse, PUD; B/B2-B4 = multi-family
_RESIDENTIAL_CLASSES = re.compile(r"^A[1-5]?$|^B[2-4]?$", re.IGNORECASE)

# Owner name tokens indicating entity rather than individual
_ENTITY_TOKENS = {
    "LLC", "L.L.C", "INC", "CORP", "TRUST", "TR", "ESTATE",
    "BANK", "MORTGAGE", "ASSOC", "ASSOCIATION", "PARTNERS", "LP", "LTD",
    "MANAGEMENT", "PROPERTIES", "HOLDINGS", "VENTURES", "GROUP",
}

# Bad-address flag characters that TAD prepends to Owner_Address
_BAD_ADDR_FLAGS = {"-", "=", "*"}


def _owner_type(name: str) -> str:
    upper = name.upper()
    for token in _ENTITY_TOKENS:
        if token in upper:
            return "LLC"
    return "individual"


def _parse_owner_citystate(citystate: str) -> tuple[str, str]:
    """
    Parse TAD's combined Owner_CityState field (e.g. "FORT WORTH TX").
    Returns (city, state_code).  State code is the last whitespace-delimited
    token if it looks like a 2-letter US state code, else ("", "").
    """
    s = citystate.strip()
    if not s:
        return "", ""
    parts = s.rsplit(None, 1)  # split on last whitespace
    if len(parts) == 2 and len(parts[1]) == 2 and parts[1].isalpha():
        return parts[0].strip().title(), parts[1].upper()
    # Fallback: last 2 chars
    if len(s) >= 2 and s[-2:].isalpha():
        return s[:-2].strip().title(), s[-2:].upper()
    return s.title(), ""


def _clean_owner_address(addr: str) -> str:
    """Strip TAD bad-address flag from first character of mailing address."""
    if addr and addr[0] in _BAD_ADDR_FLAGS:
        return addr[1:].strip()
    return addr.strip()


class TarrantAbsenteeOwnerScraper(BaseCountyScraper):
    """
    Downloads TAD PropertyData_R.ZIP and yields absentee owner records
    for residential properties with an out-of-state mailing address.

    Config keys (all optional):
        zip_url   — override the TAD download URL
    """

    county_fips = COUNTY_FIPS
    source_name = "Tarrant Appraisal District — Absentee Owner (TAD Bulk Data)"
    indicator_types = ["absentee_owner"]
    rate_limit_per_minute = 1  # single large download

    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        zip_url = self.config.get("zip_url") or TAD_ZIP_URL

        logger.info("tad_absentee_download_start", url=zip_url)

        async with httpx.AsyncClient(
            headers={"User-Agent": "Mozilla/5.0 (compatible; motivated-seller-bot/1.0)"},
            follow_redirects=True,
            timeout=600.0,
        ) as client:
            resp = await client.get(zip_url)
            resp.raise_for_status()

        content = resp.content
        logger.info("tad_absentee_download_complete", size_mb=len(content) // 1024 // 1024)

        total = skipped_non_res = skipped_no_addr = skipped_in_state = 0

        try:
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                # TAD zip typically contains a single .txt or .csv file
                names = z.namelist()
                logger.info("tad_absentee_zip_contents", files=names)
                data_file = names[0] if names else None
                if not data_file:
                    logger.error("tad_absentee_empty_zip")
                    return

                with z.open(data_file) as raw_file:
                    reader = csv.DictReader(
                        io.TextIOWrapper(raw_file, encoding="latin-1"),
                        delimiter="|",
                    )
                    for row in reader:
                        acct = (row.get("Account_Num") or "").strip()
                        if not acct:
                            continue

                        # Filter: residential only
                        prop_class = (row.get("Property_Class") or "").strip()
                        if not _RESIDENTIAL_CLASSES.match(prop_class):
                            skipped_non_res += 1
                            continue

                        # Parse owner mailing state
                        citystate_raw = (row.get("Owner_CityState") or "").strip()
                        owner_city, owner_state = _parse_owner_citystate(citystate_raw)

                        # Filter: out-of-state mailing address only
                        if not owner_state or owner_state == "TX":
                            skipped_in_state += 1
                            continue

                        # Build property site address
                        situs = (row.get("Situs_Address") or "").strip()
                        if not situs or not situs[0].isdigit():
                            skipped_no_addr += 1
                            continue
                        # TAD Situs_Address may already include city/zip or be bare street
                        # Append ", Tarrant County, TX" if no state indicator present
                        if ", TX" not in situs.upper() and "TEXAS" not in situs.upper():
                            address_raw = f"{situs}, TX"
                        else:
                            address_raw = situs

                        record = self._build_record(acct, row, address_raw, owner_city, owner_state)
                        if record and await self.validate_record(record):
                            yield record
                            total += 1

        except zipfile.BadZipFile as exc:
            logger.error("tad_absentee_bad_zip", error=str(exc))
            return

        logger.info(
            "tad_absentee_complete",
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
        owner_city: str,
        owner_state: str,
    ) -> RawIndicatorRecord | None:
        owner_name = (row.get("Owner_Name") or "").strip().title()
        if not owner_name or owner_name.upper() in ("CURRENT OWNER", "UNKNOWN"):
            return None

        mail_addr_raw = (row.get("Owner_Address") or "").strip()
        mail_addr = _clean_owner_address(mail_addr_raw)
        mail_zip = (row.get("Owner_Zip") or "").strip()[:5]

        try:
            assessed = float(row.get("Appraised_Value") or 0) or None
        except (ValueError, TypeError):
            assessed = None

        return RawIndicatorRecord(
            indicator_type="absentee_owner",
            address_raw=address_raw,
            county_fips=self.county_fips,
            owner_name=owner_name,
            owner_mailing_address=mail_addr or None,
            owner_mailing_city=owner_city or None,
            owner_mailing_state=owner_state or None,
            owner_mailing_zip=mail_zip or None,
            owner_type=_owner_type(owner_name),
            apn=acct,
            amount=assessed,
            source_url=SOURCE_URL,
            raw_payload={
                "acct": acct,
                "owner_name": owner_name,
                "mailing_address": mail_addr,
                "mailing_city": owner_city,
                "mailing_state": owner_state,
                "mailing_zip": mail_zip,
                "property_class": row.get("Property_Class", ""),
                "situs_address": row.get("Situs_Address", ""),
                "appraised_value": row.get("Appraised_Value", ""),
            },
        )
