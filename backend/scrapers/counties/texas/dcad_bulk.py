"""
Dallas County (DCAD) Bulk Data Scraper — Ownership Changes & Value Signals
(Texas - FIPS 48113)

Source: Dallas Central Appraisal District — Data Products
Portal: https://www.dallascad.org/DataProducts.aspx

Data:   Two CSV ZIP files:
          DCAD{YYYY}_CURRENT.ZIP — current year appraisal data
          DCAD{YYYY-1}_CERTIFIED_*.zip or DCAD{YYYY-1}_CURRENT.ZIP — prior year

        Each ZIP contains ACCOUNT_INFO.CSV and ACCOUNT_APPRL_YEAR.CSV.

Signals produced:
  1. "recent_sale" — owner name changed between years (= deed transfer).
     In non-disclosure Texas, this is the best free proxy for sales data.
     Appraised value serves as an approximate sale price.

  2. "absentee_owner" — owner mailing state != TX (out-of-state owner).

  3. Value drop detection — PREV_MKT_VAL > TOT_VAL indicates declining value,
     which may signal distress or motivation to sell.

Scope:  Residential properties only (DIVISION_CD = 'RES').

Rate:   Two HTTP downloads (~150 MB each). No rate limiting needed.
        Run monthly — DCAD updates the "CURRENT" files periodically.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
import structlog
from datetime import date, datetime
from typing import AsyncIterator

import httpx

from scrapers.base import BaseCountyScraper, RawIndicatorRecord

logger = structlog.get_logger(__name__)

COUNTY_FIPS = "48113"
DCAD_BASE = "https://www.dallascad.org/ViewPDFs.aspx"
SOURCE_URL = "https://www.dallascad.org/DataProducts.aspx"

# Skip entity owners for absentee detection
_ENTITY_RE = re.compile(
    r"\b(llc|l\.l\.c|inc|corp|trust|bank|mortgage|assoc|lp|ltd|partners)\b",
    re.IGNORECASE,
)


def _build_url(year: int, certified: bool = False) -> str:
    """Build download URL for DCAD data product."""
    if certified:
        # Certified files have date suffix — try common patterns
        path = f"\\\\DCAD.ORG\\WEB\\WEBDATA\\WEBFORMS\\DATA PRODUCTS\\DCAD{year}_CERTIFIED_07242025.zip"
    else:
        path = f"\\\\DCAD.ORG\\WEB\\WEBDATA\\WEBFORMS\\DATA PRODUCTS\\DCAD{year}_CURRENT.ZIP"
    return f"{DCAD_BASE}?type=3&id={path}"


def _parse_date(raw: str) -> date | None:
    if not raw or not raw.strip():
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%Y%m%d", "%m/%d/%y"):
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _parse_amount(raw: str) -> float | None:
    if not raw or not raw.strip():
        return None
    try:
        val = float(raw.strip().replace(",", ""))
        return val if val > 0 else None
    except ValueError:
        return None


def _build_address_from_slim(info: dict) -> str | None:
    """Build street address from slim ACCOUNT_INFO dict."""
    num = info.get("street_num", "")
    if not num or not num[0:1].isdigit():
        return None
    street = info.get("street", "")
    if not street:
        return None

    addr = num
    half = info.get("half", "")
    if half:
        addr += f" {half}"
    addr += f" {street}"
    bldg = info.get("bldg", "")
    unit = info.get("unit", "")
    if bldg:
        addr += f" BLDG {bldg}"
    if unit:
        addr += f" #{unit}"

    city = info.get("city", "")
    city = re.sub(r"\s*\(.*?\)\s*$", "", city).strip()
    zip_code = info.get("zip", "")

    return f"{addr}, {city}, TX {zip_code}"


async def _download_dcad_zip(client: httpx.AsyncClient, year: int) -> zipfile.ZipFile | None:
    """Download DCAD ZIP and return the ZipFile object."""
    import tempfile, os
    url = _build_url(year)
    logger.info("dcad_downloading", year=year, url=url[:80])

    try:
        resp = await client.get(url, timeout=300.0)
        resp.raise_for_status()
    except Exception as exc:
        logger.error("dcad_download_failed", year=year, error=str(exc)[:120])
        return None

    # Write to temp file to avoid holding full ZIP in memory
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp.write(resp.content)
    tmp.close()
    del resp  # free the response memory

    try:
        return zipfile.ZipFile(tmp.name)
    except zipfile.BadZipFile:
        logger.error("dcad_bad_zip", year=year)
        os.unlink(tmp.name)
        return None


def _extract_account_info_slim(z: zipfile.ZipFile) -> dict[str, dict]:
    """Extract only the fields we need from ACCOUNT_INFO — keyed by account num.
    Returns {account_num: {owner, address, state, deed_date, division}}."""
    result = {}
    for name in z.namelist():
        if "ACCOUNT_INFO" not in name.upper():
            continue
        content = z.read(name).decode("latin-1")
        reader = csv.DictReader(io.StringIO(content))
        for row in reader:
            acct = (row.get("ACCOUNT_NUM") or "").strip()
            div = (row.get("DIVISION_CD") or "").strip().upper()
            if not acct or div != "RES":
                continue
            result[acct] = {
                "owner": (row.get("OWNER_NAME1") or "").strip(),
                "state": (row.get("OWNER_STATE") or "").strip().upper(),
                "deed_date": (row.get("DEED_TXFR_DATE") or "").strip(),
                "street_num": (row.get("STREET_NUM") or "").strip(),
                "street": (row.get("FULL_STREET_NAME") or "").strip(),
                "half": (row.get("STREET_HALF_NUM") or "").strip(),
                "bldg": (row.get("BLDG_ID") or "").strip(),
                "unit": (row.get("UNIT_ID") or "").strip(),
                "city": (row.get("PROPERTY_CITY") or "").strip(),
                "zip": (row.get("PROPERTY_ZIPCODE") or "").strip()[:5],
            }
        logger.info("dcad_loaded_slim", file=name, residential=len(result))
    return result


def _extract_values_slim(z: zipfile.ZipFile) -> dict[str, tuple]:
    """Extract (tot_val, prev_mkt_val) from ACCOUNT_APPRL_YEAR for RES only."""
    result = {}
    for name in z.namelist():
        if "ACCOUNT_APPRL_YEAR" not in name.upper():
            continue
        content = z.read(name).decode("latin-1")
        reader = csv.DictReader(io.StringIO(content))
        for row in reader:
            acct = (row.get("ACCOUNT_NUM") or "").strip()
            div = (row.get("DIVISION_CD") or "").strip().upper()
            if not acct or div != "RES":
                continue
            tot = _parse_amount(row.get("TOT_VAL", ""))
            prev = _parse_amount(row.get("PREV_MKT_VAL", ""))
            result[acct] = (tot, prev)
        logger.info("dcad_values_loaded", file=name, count=len(result))
    return result


def _extract_owners_only(z: zipfile.ZipFile) -> dict[str, str]:
    """Extract just account -> owner_name from ACCOUNT_INFO."""
    result = {}
    for name in z.namelist():
        if "ACCOUNT_INFO" not in name.upper():
            continue
        content = z.read(name).decode("latin-1")
        reader = csv.DictReader(io.StringIO(content))
        for row in reader:
            acct = (row.get("ACCOUNT_NUM") or "").strip()
            owner = (row.get("OWNER_NAME1") or "").strip().upper()
            if acct and owner:
                result[acct] = owner
        logger.info("dcad_owners_loaded", file=name, count=len(result))
    return result


class DCADBulkScraper(BaseCountyScraper):
    county_fips = COUNTY_FIPS
    source_name = "DCAD Bulk Data — Ownership Changes & Value Signals"
    indicator_types = ["absentee_owner"]
    rate_limit_per_minute = 5

    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        import gc, os
        current_year = date.today().year

        async with httpx.AsyncClient(
            headers={"User-Agent": "Mozilla/5.0 (compatible; motivated-seller-bot/1.0)"},
            follow_redirects=True,
        ) as client:
            curr_zip = await _download_dcad_zip(client, current_year)
            if not curr_zip:
                logger.warning("dcad_no_current_data")
                return

            # Extract slim data and free the zip
            curr_accounts = _extract_account_info_slim(curr_zip)
            val_map = _extract_values_slim(curr_zip)
            tmp_path = curr_zip.filename
            curr_zip.close()
            if tmp_path:
                os.unlink(tmp_path)
            gc.collect()

            if not curr_accounts:
                logger.warning("dcad_no_residential_accounts")
                return

            # Download prior year for owner comparison
            prev_zip = await _download_dcad_zip(client, current_year - 1)

        prev_owners: dict[str, str] = {}
        if prev_zip:
            prev_owners = _extract_owners_only(prev_zip)
            tmp_path = prev_zip.filename
            prev_zip.close()
            if tmp_path:
                os.unlink(tmp_path)
            gc.collect()

        logger.info(
            "dcad_lookups_built",
            current=len(curr_accounts),
            prev_owners=len(prev_owners),
            values=len(val_map),
        )

        total = 0
        owner_changes = 0
        absentee_count = 0
        value_drops = 0

        for acct, info in curr_accounts.items():
            address_raw = _build_address_from_slim(info)
            if not address_raw:
                continue

            owner_name = info["owner"]
            owner_state = info["state"]
            deed_date = _parse_date(info["deed_date"])

            # Get values
            vals = val_map.get(acct)
            tot_val = vals[0] if vals else None
            prev_val = vals[1] if vals else None

            # --- Signal 1: Ownership change ---
            prev_owner = prev_owners.get(acct, "")
            curr_upper = owner_name.upper()
            is_owner_change = (
                prev_owner and curr_upper
                and prev_owner != curr_upper
                and deed_date
                and (date.today() - deed_date).days < 540
            )

            # --- Signal 2: Out-of-state owner ---
            is_absentee = (
                owner_state
                and owner_state not in ("TX", "TEXAS", "")
                and not _ENTITY_RE.search(owner_name)
            )

            # --- Signal 3: Value drop ---
            is_value_drop = (
                tot_val and prev_val
                and tot_val < prev_val * 0.9
            )

            if is_absentee:
                record = RawIndicatorRecord(
                    indicator_type="absentee_owner",
                    address_raw=address_raw,
                    county_fips=self.county_fips,
                    apn=acct,
                    owner_name=owner_name.title() or None,
                    owner_mailing_state=owner_state,
                    amount=tot_val,
                    filing_date=deed_date,
                    source_url=SOURCE_URL,
                    raw_payload={
                        "account_num": acct,
                        "owner_state": owner_state,
                        "owner_change": is_owner_change,
                        "value_drop": is_value_drop,
                        "tot_val": tot_val,
                        "prev_val": prev_val,
                        "deed_date": str(deed_date) if deed_date else "",
                        "prev_owner": prev_owner[:50] if prev_owner else "",
                    },
                )
                if await self.validate_record(record):
                    yield record
                    total += 1
                    absentee_count += 1

            if is_owner_change:
                owner_changes += 1
            if is_value_drop:
                value_drops += 1

            if total > 0 and total % 500 == 0:
                logger.info("dcad_progress", total=total, changes=owner_changes, absentee=absentee_count)

        logger.info(
            "dcad_bulk_complete",
            total_yielded=total,
            owner_changes=owner_changes,
            absentee_owners=absentee_count,
            value_drops=value_drops,
            residential=len(curr_accounts),
        )
