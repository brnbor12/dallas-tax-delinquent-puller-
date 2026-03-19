"""
Polk County Tax Delinquent Scraper (Florida - FIPS 12105)

Source: Polk County Property Appraiser — FTP Bulk Data
Portal: https://polkflpa.gov/FTPPage/ftpdefault.aspx

Data:   Joins three bulk files on PARCEL_ID:
          ftp_parceltax.txt — parcel tax data (delinquent amounts, year, status)
          ftp_site.txt      — property site addresses
          ftp_owner.txt     — owner name + mailing address

Signal: indicator_type = "tax_delinquent"
        A property is flagged when it has unpaid / delinquent taxes.
        Delinquent owners face certificate sales and eventual tax deed
        auctions — they're highly motivated to settle or sell.

Scope:  All residential parcels with a delinquent tax balance > 0.
        Excludes vacant land (STR_NUM == '0'), government/exempt parcels.

Files:
  ftp_parceltax.zip (~33 MB) — parcel tax amounts and status
  ftp_site.zip      (~3 MB)  — property site addresses
  ftp_owner.zip     (~13 MB) — owner mailing addresses

Rate:   Three HTTPS downloads (~50 MB total). No rate limiting needed.
        Files are regenerated nightly; run weekly to capture new delinquencies.

Proxy:  The PA site uses WebKnight WAF that blocks datacenter IPs.
        Set PROXY_URL env var or pass config.proxy_url to route through
        a residential proxy (e.g. http://user:pass@proxy:port).
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
    "parceltax": {"filename": "ftp_parceltax.zip"},
    "site":      {"filename": "ftp_site.zip"},
    "owner":     {"filename": "ftp_owner.zip"},
}

# Residential land-use codes (from ftp_parcel DORDESC field, joined via owner file)
# We key off site address existence + non-zero tax balance instead of DORDESC
# since parceltax doesn't carry DORDESC.

# Tax status values that indicate delinquency
_DELINQUENT_STATUSES = {"D", "DELINQUENT", "DLQ", "CERT", "CERTIFICATE", "TDA"}


def _build_url(file_key: str) -> str:
    p = _FILE_PARAMS[file_key]
    return f"{PA_BASE}?filename={p['filename']}&dir=%5CAppraisalData%5C"


async def _download_zip_csv(
    client: httpx.AsyncClient, file_key: str
) -> list[dict]:
    """Download a zip file and return parsed CSV rows."""
    url = _build_url(file_key)
    try:
        resp = await client.get(url, timeout=180.0)
        resp.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        txt_file = next(n for n in z.namelist() if n.endswith(".txt"))
        content = z.read(txt_file).decode("latin-1")
        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)
        logger.info("polk_tax_downloaded", file=file_key, rows=len(rows))
        return rows
    except Exception as exc:
        logger.error("polk_tax_download_failed", file=file_key, error=str(exc)[:120])
        return []


def _build_site_address(row: dict) -> str | None:
    """Assemble a street address from ftp_site fields."""
    num = row.get("STR_NUM", "").strip()
    if not num or num == "0":
        return None

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


def _parse_amount(raw: str) -> float | None:
    if not raw or not raw.strip():
        return None
    try:
        val = float(raw.strip().replace(",", "").replace("$", ""))
        return val if val != 0 else None
    except ValueError:
        return None


class PolkTaxDelinquentScraper(BaseCountyScraper):
    county_fips = COUNTY_FIPS
    source_name = "Polk County Property Appraiser — Tax Delinquent (FTP Bulk Data)"
    indicator_types = ["tax_delinquent"]
    rate_limit_per_minute = 10

    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        from scrapers.gcp_proxy import make_proxied_client

        async with make_proxied_client(
            headers={"User-Agent": "Mozilla/5.0 (compatible; motivated-seller-bot/1.0)"},
            follow_redirects=True,
        ) as client:
            parceltax_rows = await _download_zip_csv(client, "parceltax")
            site_rows = await _download_zip_csv(client, "site")
            owner_rows = await _download_zip_csv(client, "owner")

        if not parceltax_rows:
            logger.warning("polk_tax_no_parceltax_data")
            return

        if not site_rows:
            logger.warning("polk_tax_no_site_data")
            return

        # Build lookup: PARCEL_ID -> site address
        site_map: dict[str, str] = {}
        for r in site_rows:
            pid = r.get("PARCEL_ID", "").strip()
            if pid and pid not in site_map:
                addr = _build_site_address(r)
                if addr:
                    site_map[pid] = addr

        # Build lookup: PARCEL_ID -> owner info (primary owner only)
        owner_map: dict[str, dict] = {}
        for r in owner_rows:
            if r.get("LN_NUM", "").strip() != "1":
                continue
            pid = r.get("PARCEL_ID", "").strip()
            if pid and pid not in owner_map:
                owner_map[pid] = r

        logger.info(
            "polk_tax_lookups_built",
            parceltax=len(parceltax_rows),
            sites=len(site_map),
            owners=len(owner_map),
        )

        # Detect which columns carry delinquent tax info
        # Common field names in Polk PA parceltax: TAX_DUE, BALANCE, STATUS, etc.
        sample = parceltax_rows[0] if parceltax_rows else {}
        sample_keys = {k.upper(): k for k in sample.keys()}
        logger.info("polk_tax_columns", columns=list(sample.keys())[:20])

        # Find the amount column — try common names
        amount_col = None
        for candidate in ["TAXESDUE", "TAX_DUE", "BALANCE", "AMT_DUE", "DELINQUENT",
                          "TOTAL_DUE", "BAL_DUE", "CERT_AMT", "TAX_BALANCE", "AMOUNT"]:
            if candidate in sample_keys:
                amount_col = sample_keys[candidate]
                break

        # Find status column
        status_col = None
        for candidate in ["STATUS", "TAX_STATUS", "CERT_STATUS", "DLQ_STATUS"]:
            if candidate in sample_keys:
                status_col = sample_keys[candidate]
                break

        # Find year column
        year_col = None
        for candidate in ["TAX_YEAR", "YEAR", "TAX_YR", "YR", "TAXDIST"]:
            if candidate in sample_keys:
                year_col = sample_keys[candidate]
                break

        logger.info(
            "polk_tax_detected_columns",
            amount=amount_col,
            status=status_col,
            year=year_col,
        )

        # Aggregate: sum TAXESDUE across all districts per parcel
        # A parcel appears once per tax district — aggregate to one total
        parcel_tax: dict[str, float] = {}
        for row in parceltax_rows:
            pid = row.get("PARCEL_ID", "").strip()
            if not pid:
                continue
            amount = _parse_amount(row.get(amount_col, "")) if amount_col else None
            if amount and amount > 0:
                parcel_tax[pid] = parcel_tax.get(pid, 0) + amount

        logger.info("polk_tax_delinquent_parcels", delinquent=len(parcel_tax))

        total = 0
        skipped_no_site = 0

        for pid, total_due in parcel_tax.items():
            address_raw = site_map.get(pid)
            if not address_raw:
                skipped_no_site += 1
                continue

            owner = owner_map.get(pid, {})
            owner_name = owner.get("NAME", "").strip().title() if owner else None

            record = RawIndicatorRecord(
                indicator_type="tax_delinquent",
                address_raw=address_raw,
                county_fips=self.county_fips,
                apn=pid,
                owner_name=owner_name or None,
                amount=total_due,
                source_url=SOURCE_URL,
                raw_payload={
                    "parcel_id": pid,
                    "total_taxes_due": total_due,
                    "owner_name": owner_name or "",
                },
            )

            if await self.validate_record(record):
                yield record
                total += 1

        logger.info(
            "polk_tax_delinquent_complete",
            total_yielded=total,
            skipped_no_site=skipped_no_site,
            skipped_no_balance=skipped_no_balance,
        )
        if total == 0:
            logger.warning("polk_tax_delinquent_no_records")
