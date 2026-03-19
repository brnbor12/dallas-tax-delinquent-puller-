"""
Polk County Absentee Owner Scraper (Florida - FIPS 12105)

Source: Polk County Property Appraiser — FTP Bulk Data
Portal: https://polkflpa.gov/FTPPage/ftpdefault.aspx

Data:   CAMA (Computer Assisted Mass Appraisal) bulk data files, updated nightly.
        Joins three files on PARCEL_ID:
          ftp_owner.txt  — owner name + mailing address (where the owner lives)
          ftp_site.txt   — property site address (where the property is)
          ftp_parcel.txt — land use code (RES/COM/etc.) + homestead exemption

Signal: indicator_type = "absentee_owner"
        A property is flagged when the owner's mailing address is out-of-state
        (STATE != 'FL'), indicating the owner does not live at the property.
        Out-of-state owners are often more motivated to sell — they have no
        local ties, face higher carrying costs, and frequently want to liquidate.

Scope:  Residential parcels (DORDESC == 'RES') with valid site addresses only.
        Excludes vacant land (STR_NUM == '0'), government parcels (GOV/EX),
        and institutional/exempt parcels.

Files:
  ftp_owner.zip   (~13 MB) — owner mailing addresses
  ftp_site.zip    (~3 MB)  — property site addresses
  ftp_parcel.zip  (~14 MB) — land use and homestead data

    Owner fields: PARCEL_ID, LN_NUM, NAME, PCTOWN, MAILTO,
                  ADDR_1, ADDR_2, ADDR_3, CITY, STATE, ZIP
    Site fields:  PARCEL_ID, LN_NUM, BLD_NUM, STR, STR_PFX, STR_NUM,
                  STR_NUM_SFX, STR_SFX, STR_SFX_DIR, STR_UNIT, ZIP, CITY
    Parcel fields: PARCEL_ID, DORDESC, DORDESC1, HOMESTEAD, ASSESSVAL, YR_IMPROVED

Rate:   Three simple HTTPS downloads (~30 MB total). No rate limiting needed.
        Files are regenerated nightly; run weekly or monthly to avoid re-ingesting
        the same set of static ownership records.
"""

from __future__ import annotations

import csv
import io
import os
import zipfile
import structlog
from typing import AsyncIterator

import httpx

from scrapers.base import BaseCountyScraper, RawIndicatorRecord

logger = structlog.get_logger(__name__)

COUNTY_FIPS = "12105"
PA_BASE = "https://polkflpa.gov/FTPPage/downloader.ashx"
SOURCE_URL = "https://polkflpa.gov/FTPPage/ftpdefault.aspx"

_FILE_PARAMS = {
    "owner":  {"filename": "ftp_owner.zip",  "dir": r"\AppraisalData\ ".strip()},
    "site":   {"filename": "ftp_site.zip",   "dir": r"\AppraisalData\ ".strip()},
    "parcel": {"filename": "ftp_parcel.zip", "dir": r"\AppraisalData\ ".strip()},
}

# Residential land-use code
_RESIDENTIAL_DORDESC = {"RES"}

# States to flag as out-of-state (everything except FL and empty)
_SKIP_STATES = {"FL", ""}

# Owner name suffixes that indicate non-individual (LLC, trust, bank, etc.)
_ENTITY_TOKENS = {
    "LLC", "L.L.C", "INC", "CORP", "TRUST", "TR", "ESTATE",
    "BANK", "MORTGAGE", "ASSOC", "ASSOCIATION", "PARTNERS", "LP", "LTD",
}


def _build_url(file_key: str) -> str:
    p = _FILE_PARAMS[file_key]
    return f"{PA_BASE}?filename={p['filename']}&dir=%5CAppraisalData%5C"


async def _download_zip_csv(client: httpx.AsyncClient, file_key: str) -> list[dict]:
    """Download a zip file and return parsed CSV rows."""
    url = _build_url(file_key)
    try:
        resp = await client.get(url, timeout=120.0)
        resp.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        txt_file = next(n for n in z.namelist() if n.endswith(".txt"))
        content = z.read(txt_file).decode("latin-1")
        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)
        logger.info("polk_absentee_downloaded", file=file_key, rows=len(rows))
        return rows
    except Exception as exc:
        logger.error("polk_absentee_download_failed", file=file_key, error=str(exc)[:120])
        return []


def _owner_type(name: str) -> str:
    """Classify owner as individual or entity."""
    upper = name.upper()
    for token in _ENTITY_TOKENS:
        if token in upper:
            return "LLC"
    return "individual"


def _build_site_address(row: dict) -> str | None:
    """Assemble a street address from ftp_site fields."""
    num = row.get("STR_NUM", "").strip()
    if not num or num == "0":
        return None  # no valid street number

    pfx = row.get("STR_PFX", "").strip()
    street = row.get("STR", "").strip()
    sfx = row.get("STR_SFX", "").strip()
    dir_ = row.get("STR_SFX_DIR", "").strip()
    unit = row.get("STR_UNIT", "").strip()
    city = row.get("CITY", "").strip().title()
    zip_ = row.get("ZIP", "").strip()

    parts = [p for p in [num, pfx, street, sfx, dir_] if p]
    addr = " ".join(parts)
    if unit:
        addr += f" #{unit}"

    return f"{addr}, {city}, FL {zip_}"


class PolkAbsenteeOwnerScraper(BaseCountyScraper):
    county_fips = COUNTY_FIPS
    source_name = "Polk County Property Appraiser — Absentee Owner (FTP Bulk Data)"
    indicator_types = ["absentee_owner"]
    rate_limit_per_minute = 10  # 3 downloads, no repeated requests

    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        from scrapers.gcp_proxy import make_proxied_client

        async with make_proxied_client(
            headers={"User-Agent": "Mozilla/5.0 (compatible; motivated-seller-bot/1.0)"},
            follow_redirects=True,
        ) as client:
            owner_rows, site_rows, parcel_rows = await _download_zip_csv(
                client, "owner"), await _download_zip_csv(
                client, "site"), await _download_zip_csv(
                client, "parcel")

        if not owner_rows or not site_rows or not parcel_rows:
            logger.warning("polk_absentee_missing_data",
                           owner=len(owner_rows), site=len(site_rows), parcel=len(parcel_rows))
            return

        # Build lookup: PARCEL_ID -> parcel row (land use + homestead)
        parcel_map: dict[str, dict] = {
            r["PARCEL_ID"]: r for r in parcel_rows
        }

        # Build lookup: PARCEL_ID -> site address string
        site_map: dict[str, str] = {}
        for r in site_rows:
            pid = r.get("PARCEL_ID", "").strip()
            if pid and pid not in site_map:
                addr = _build_site_address(r)
                if addr:
                    site_map[pid] = addr

        logger.info("polk_absentee_lookups_built",
                    parcels=len(parcel_map), sites=len(site_map))

        total = 0
        skipped_non_res = 0
        skipped_no_site = 0
        skipped_in_state = 0

        for row in owner_rows:
            # Only primary owner (LN_NUM == "1")
            if row.get("LN_NUM", "").strip() != "1":
                continue

            pid = row.get("PARCEL_ID", "").strip()
            if not pid:
                continue

            # Filter: residential parcels only
            parcel = parcel_map.get(pid)
            if not parcel:
                continue
            if parcel.get("DORDESC", "").strip() not in _RESIDENTIAL_DORDESC:
                skipped_non_res += 1
                continue

            # Filter: out-of-state mailing address only
            mailing_state = row.get("STATE", "").strip().upper()
            if mailing_state in _SKIP_STATES:
                skipped_in_state += 1
                continue

            # Filter: must have a valid site address
            address_raw = site_map.get(pid)
            if not address_raw:
                skipped_no_site += 1
                continue

            # Build record
            record = self._build_record(pid, row, parcel, address_raw, mailing_state)
            if record and await self.validate_record(record):
                yield record
                total += 1

        logger.info(
            "polk_absentee_complete",
            total_yielded=total,
            skipped_non_res=skipped_non_res,
            skipped_no_site=skipped_no_site,
            skipped_in_state=skipped_in_state,
        )
        if total == 0:
            logger.warning("polk_absentee_no_records")

    def _build_record(
        self,
        pid: str,
        owner: dict,
        parcel: dict,
        address_raw: str,
        mailing_state: str,
    ) -> RawIndicatorRecord | None:
        owner_name = owner.get("NAME", "").strip().title()
        if not owner_name:
            return None

        mailing_addr1 = owner.get("ADDR_1", "").strip()
        mailing_addr2 = owner.get("ADDR_2", "").strip()
        mailing_city = owner.get("CITY", "").strip().title()
        mailing_zip = owner.get("ZIP", "").strip()[:5]

        mailing_full = mailing_addr1
        if mailing_addr2:
            mailing_full = f"{mailing_addr1}, {mailing_addr2}"

        owner_t = _owner_type(owner_name)

        # Assessed value from parcel
        try:
            assessed = float(parcel.get("ASSESSVAL", "") or 0)
        except ValueError:
            assessed = None

        homestead = parcel.get("HOMESTEAD", "").strip()
        is_homestead = bool(homestead and homestead != "0")

        return RawIndicatorRecord(
            indicator_type="absentee_owner",
            address_raw=address_raw,
            county_fips=self.county_fips,
            owner_name=owner_name or None,
            owner_mailing_address=mailing_full or None,
            owner_mailing_city=mailing_city or None,
            owner_mailing_state=mailing_state or None,
            owner_mailing_zip=mailing_zip or None,
            owner_type=owner_t,
            apn=pid,
            amount=assessed if assessed else None,
            source_url=SOURCE_URL,
            raw_payload={
                "parcel_id": pid,
                "owner_name": owner_name,
                "mailing_state": mailing_state,
                "mailing_address": mailing_full,
                "mailing_city": mailing_city,
                "mailing_zip": mailing_zip,
                "dordesc": parcel.get("DORDESC", ""),
                "dordesc1": parcel.get("DORDESC1", ""),
                "homestead": homestead,
                "is_homestead": is_homestead,
                "assessed_value": assessed,
                "yr_improved": parcel.get("YR_IMPROVED", ""),
            },
        )
