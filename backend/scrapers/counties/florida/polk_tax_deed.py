"""
Polk County Tax Deed Sales Scraper (Florida - FIPS 12105)

Source: apps.polkcountyclerk.net/browserviewtd/  (NewVision AngularJS app)
Data:   Tax deed sale listings — properties where a certificate holder filed
        a Tax Deed Application (TDA), forcing a public auction when the owner
        fails to pay delinquent property taxes.

Signal: indicator_type = "tax_delinquent"
        A scheduled tax deed sale means:
          1. Property taxes have been delinquent for ≥ 2 years.
          2. A certificate holder has formally applied for a tax deed.
          3. The county clerk has scheduled a forced auction.
        The owner has until the auction gavel falls to redeem or sell.

API:    The AngularJS app POSTs RSA-encrypted search criteria to:
          POST https://apps.polkcountyclerk.net/browserviewtd/api/search
        Encryption: RSA-PKCS1v15 with a hard-coded public key embedded in
          the site's services.js.  Parameters: SaleDate (YYYYMMDD), or
          LandAvailable ("LANDA") + AvailDate for unsold "lands available".

TLS note: The server uses TLS 1.0 / weak cipher suites.  Python 3.12 +
        OpenSSL 3.x reject this by default.  We configure the SSL context with
        SECLEVEL=0 and minimum_version=TLSv1 to allow the connection.
        This is acceptable for a public data scraper with no sensitive traffic.

Fields returned (per sale record):
  deed_id      — internal ID (not used for matching)
  tax_number   — tax certificate number
  strap_num    — Polk County parcel number ("strap number")
  last_name    — assessed owner name (full name despite field label)
  deed_status  — SCHEDULED | SOLD | REDEEMED | CANCELLED | ...
  sale_date    — auction date (ISO datetime string)
  trans_amt    — base bid / minimum to redeem
  bid_amt      — highest bid (0 if unsold)
  ref_1        — Tax Deed ID (NNNNN-YYYY format, our case_number)

Note: the API does NOT return the property address.  We construct a partial
      address from the parcel number and rely on the address_normalized fuzzy
      matcher or geocoder to resolve later.  If a better address source is
      found (e.g. Polk PA parcel API), swap in _get_address().
"""

from __future__ import annotations

import base64
import ssl
import warnings
import re
from datetime import date, datetime, timedelta
from typing import AsyncIterator

import httpx
import structlog
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding

from scrapers.base import BaseCountyScraper, RawIndicatorRecord

logger = structlog.get_logger(__name__)

COUNTY_FIPS = "12105"
API_URL = "https://apps.polkcountyclerk.net/browserviewtd/api/search"

# RSA public key hard-coded in the site's Scripts/app/services.js
# Used for PKCS1v15 encryption of all search parameters.
_PUBLIC_KEY_PEM = b"""-----BEGIN PUBLIC KEY-----
MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQDXBPCKXRqaD74rYrPXU/DA4Z5H
mJbNivwCYijae6QXu/QLqS3GbyGrxkrEmdODbYWOLJfWBvaQSALcolSyKQUvtkjz
g61bJC2/xNk4HTHFrA4uAMMvC+49RlSgtEm5dI10+YOp0TGId1d4E0Ey0RDQxNWa
ev2TeleyipADuctnqwIDAQAB
-----END PUBLIC KEY-----"""

# Scan window: look for sales in this range relative to today
DAYS_LOOKBACK = 30
DAYS_LOOKAHEAD = 90

# Only ingest properties that are still actionable (not yet sold)
ACTIVE_STATUSES = {"SCHEDULED", "PENDING", "ACTIVE", "LANDA"}


def _make_ssl_context() -> ssl.SSLContext:
    """
    Build an SSL context that accepts the Polk clerk server's legacy TLS 1.0.
    SECLEVEL=0 disables OpenSSL's minimum key-length and cipher requirements.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.minimum_version = ssl.TLSVersion.TLSv1
        ctx.set_ciphers("DEFAULT@SECLEVEL=0")
    return ctx


def _make_encryptor():
    """Load the RSA public key and return an encrypt(str) → str function."""
    key = serialization.load_pem_public_key(_PUBLIC_KEY_PEM, backend=default_backend())

    def encrypt(value: str) -> str:
        enc_bytes = key.encrypt(value.encode("utf-8"), padding.PKCS1v15())
        return base64.b64encode(enc_bytes).decode("ascii")

    return encrypt


def _format_strap(strap_num: str) -> str:
    """
    Convert a raw strap number like '152622000000044130' to the standard
    Polk County display format: 'SS-TT-RR-SSSSSS-PPPPPP-UUUU'
    (Section-Township-Range-Subdivision-Parcel-Unit).
    Used only as a fallback address when no real address is available.
    """
    s = re.sub(r"\D", "", str(strap_num))
    if len(s) == 18:
        return f"{s[0:2]}-{s[2:4]}-{s[4:6]}-{s[6:12]}-{s[12:18]}"
    return s


class PolkTaxDeedScraper(BaseCountyScraper):
    county_fips = COUNTY_FIPS
    source_name = "Polk County Tax Deed Sales (Clerk's Browser)"
    indicator_types = ["tax_delinquent"]
    rate_limit_per_minute = 15  # ~4 s between requests; polite to the county server

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._encrypt = _make_encryptor()
        self._ssl_ctx = _make_ssl_context()

    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        """
        Search for tax deed sales by date over a rolling window.
        Yields one record per property with an active/upcoming sale status.
        """
        scan_dates = self._get_scan_dates()
        logger.info(
            "polk_td_scan_start",
            dates=len(scan_dates),
            first=str(scan_dates[0]),
            last=str(scan_dates[-1]),
        )

        seen_straps: set[str] = set()

        async with httpx.AsyncClient(
            verify=self._ssl_ctx,
            timeout=httpx.Timeout(connect=15, read=30, write=15, pool=15),
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"},
        ) as client:
            for sale_date in scan_dates:
                records = await self._fetch_sale_date(client, sale_date)
                for item in records:
                    strap = str(item.get("strap_num", "")).strip()
                    if not strap or strap in seen_straps:
                        continue

                    status = (item.get("deed_status") or "").upper().strip()
                    # Only ingest active/upcoming sales (not already sold)
                    if status and status not in ACTIVE_STATUSES:
                        continue

                    seen_straps.add(strap)
                    record = self._build_record(item, sale_date)
                    if record and await self.validate_record(record):
                        yield record

            # Also search Lands Available (LANDA) — unsold properties
            landa_records = await self._fetch_lands_available(client)
            for item in landa_records:
                strap = str(item.get("strap_num", "")).strip()
                if not strap or strap in seen_straps:
                    continue
                seen_straps.add(strap)
                record = self._build_record(item, date.today())
                if record and await self.validate_record(record):
                    yield record

        logger.info("polk_td_complete", total_yielded=len(seen_straps))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_scan_dates(self) -> list[date]:
        today = date.today()
        start = today - timedelta(days=DAYS_LOOKBACK)
        end = today + timedelta(days=DAYS_LOOKAHEAD)
        dates = []
        d = start
        while d <= end:
            dates.append(d)
            d += timedelta(days=1)
        return dates

    async def _fetch_sale_date(
        self, client: httpx.AsyncClient, sale_date: date
    ) -> list[dict]:
        """Fetch all tax deed records for a specific sale date."""
        date_str = sale_date.strftime("%Y%m%d")
        payload = {
            "SaleDate": self._encrypt(date_str),
            "MaxRows": -1,
            "RowsPerPage": " 0",
            "StartRow": " 0",
        }
        return await self._post_search(client, payload, context=f"SaleDate={date_str}")

    async def _fetch_lands_available(self, client: httpx.AsyncClient) -> list[dict]:
        """Fetch 'Lands Available' — unsold properties from past failed auctions."""
        today_str = date.today().strftime("%Y%m%d")
        payload = {
            "LandAvailable": self._encrypt("LANDA"),
            "AvailDate": self._encrypt(today_str),
            "MaxRows": -1,
            "RowsPerPage": " 0",
            "StartRow": " 0",
        }
        return await self._post_search(client, payload, context="LandsAvailable")

    async def _post_search(
        self, client: httpx.AsyncClient, payload: dict, context: str = ""
    ) -> list[dict]:
        """POST search payload and return parsed list of records."""
        try:
            resp = await client.post(
                API_URL,
                json=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            await self._rate_limit_sleep()
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data:
                logger.debug(
                    "polk_td_search_result",
                    context=context,
                    records=data[0].get("_total_rows", len(data)),
                )
            return data if isinstance(data, list) else []
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning(
                "polk_td_search_error",
                context=context,
                error=str(exc)[:200],
            )
            return []

    def _build_record(self, item: dict, fallback_date: date) -> RawIndicatorRecord | None:
        """Convert an API result dict to a RawIndicatorRecord."""
        strap_num = str(item.get("strap_num", "")).strip()
        if not strap_num:
            return None

        owner_name = str(item.get("last_name", "")).strip() or None

        # Amount: use base bid (trans_amt); fall back to bid_amt
        amount: float | None = None
        for amt_field in ("trans_amt", "bid_amt"):
            raw = item.get(amt_field)
            if raw:
                try:
                    amount = float(raw)
                    if amount > 0:
                        break
                except (ValueError, TypeError):
                    pass

        # Sale date
        filing_date = fallback_date
        raw_sale = item.get("sale_date")
        if raw_sale:
            try:
                filing_date = datetime.fromisoformat(raw_sale[:10]).date()
            except ValueError:
                pass

        # Tax Deed ID is the case number (format "NNNNN-YYYY")
        case_number = str(item.get("ref_1", "")).strip() or None

        # Address: not provided by the API.
        # Format as "STRAP <strap_num>, Polk County, FL" so the ingestor
        # can at least store the record and attempt APN-based matching.
        # TODO: replace with polkpa.org parcel lookup for real addresses.
        formatted_strap = _format_strap(strap_num)
        address_raw = f"Parcel {formatted_strap}, Polk County, FL"

        return RawIndicatorRecord(
            indicator_type="tax_delinquent",
            address_raw=address_raw,
            county_fips=self.county_fips,
            apn=strap_num,
            owner_name=owner_name,
            amount=amount,
            filing_date=filing_date,
            case_number=case_number,
            source_url="https://apps.polkcountyclerk.net/browserviewtd/",
            raw_payload={
                "deed_id": item.get("deed_id"),
                "tax_number": item.get("tax_number"),
                "strap_num": strap_num,
                "deed_status": item.get("deed_status"),
                "type_code": item.get("type_code"),
                "trans_amt": item.get("trans_amt"),
                "bid_amt": item.get("bid_amt"),
                "sale_date": item.get("sale_date"),
                "cr_date": item.get("cr_date"),
            },
        )
