"""
Dallas County — Supabase-backed scrapers (Texas - FIPS 48113)

Source:  Existing GCP pipeline that writes to Supabase (project wjlbfmxxxmzzktpomptq).
Tables:
  - gov_leads              — code violations (ArcGIS) + tax delinquent records
  - gov_foreclosure_notice_raw — foreclosure notices scraped from county PDFs

These scrapers ingest data already collected by separate Cloud Run jobs rather
than scraping county websites directly.  They are designed for:
  - Initial full sync (first run, no date filter)
  - Incremental sync (subsequent runs, filters on first_seen_at >= last run)

Scraper keys:
  tx_dallas_supabase_leads       — gov_leads (code_violation + tax_delinquent)
  tx_dallas_supabase_foreclosure — gov_foreclosure_notice_raw
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import AsyncIterator

import structlog

from scrapers.base import BaseCountyScraper, RawIndicatorRecord
from scrapers.platforms.supabase_source import SupabasePagedReader

logger = structlog.get_logger(__name__)

COUNTY_FIPS = "48113"
SOURCE_URL = "https://supabase.com"  # internal source, no public URL

# Maps Supabase source_type → our indicator_type
_SOURCE_TYPE_MAP: dict[str, str] = {
    "code_violations_arcgis": "code_violation",
    "tax_delinquent": "tax_delinquent",
    "tax_delinquent_trw": "tax_delinquent",
    # Foreclosure notices extracted from county clerk PDFs (NOD / pre-foreclosure)
    "foreclosure_notice": "pre_foreclosure",
}

# Legal description fragments that mean the address field is unusable
_LEGAL_DESC_MARKERS = (
    "deed of trust",
    "described herein",
    "described therein",
    "real property",
    "legal description",
    "lot block",
    "square feet",
)

# Trailing county/area abbreviations appended to addresses in the tax roll export
# e.g. "4930 BROOKVIEW DR, DA"  →  "4930 BROOKVIEW DR"
_TRAILING_ABBR_RE = re.compile(r",\s*[A-Z]{1,4}$")


def _clean_address(raw: str) -> str:
    """Strip trailing county abbreviations like ', DA' from truncated tax roll addresses."""
    return _TRAILING_ABBR_RE.sub("", (raw or "").strip()).strip()


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except (ValueError, TypeError):
        return None


def _build_address(row: dict) -> str:
    """Reconstruct a full address from gov_leads columns."""
    street = _clean_address(row.get("address") or "")
    city = (row.get("city") or "").strip().title()
    state = (row.get("state") or "TX").strip().upper()
    zip_code = (row.get("zip") or "").strip()

    if not street:
        return ""

    parts = [street]
    if city:
        parts.append(city)
    if state:
        parts.append(state)
    if zip_code:
        parts.append(zip_code)
    return ", ".join(parts)


class DallasGovLeadsScraper(BaseCountyScraper):
    """
    Ingests gov_leads from Supabase: code violations + tax delinquent records.

    Config keys (all optional):
        supabase_url  — override default from app settings
        supabase_key  — override default from app settings
        full_sync     — if True, ignore lookback and sync all records
        lookback_days — days to look back for incremental sync (default: 30)
    """

    county_fips = COUNTY_FIPS
    source_name = "Dallas Supabase gov_leads (code violations + tax delinquent + foreclosure notices)"
    indicator_types = ["code_violation", "tax_delinquent", "pre_foreclosure"]
    rate_limit_per_minute = 120  # Supabase REST, no strict rate limit

    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        from app.core.config import settings

        url = self.config.get("supabase_url") or settings.supabase_url
        key = self.config.get("supabase_key") or settings.supabase_key
        full_sync: bool = self.config.get("full_sync", False)
        lookback_days: int = int(self.config.get("lookback_days", 30))

        since: datetime | None = None
        if not full_sync:
            from datetime import timedelta
            since = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        reader = SupabasePagedReader(url, key, "gov_leads")
        total = 0

        for source_type, indicator_type in _SOURCE_TYPE_MAP.items():
            logger.info(
                "dallas_supabase_leads_start",
                source_type=source_type,
                indicator_type=indicator_type,
                full_sync=full_sync,
            )

            async for row in reader.paginate(
                since=since,
                since_field="first_seen_at",
                **{f"source_type": f"eq.{source_type}"},
            ):
                record = self._build_record(row, indicator_type)
                if record and await self.validate_record(record):
                    yield record
                    total += 1

        logger.info("dallas_supabase_leads_complete", total_yielded=total)

    def _build_record(self, row: dict, indicator_type: str) -> RawIndicatorRecord | None:
        address_raw = _build_address(row)
        if not address_raw:
            return None

        # Skip addresses that are clearly legal descriptions extracted from PDFs
        addr_lower = address_raw.lower()
        if len(address_raw) > 150 or any(m in addr_lower for m in _LEGAL_DESC_MARKERS):
            return None

        filing_date = _parse_date(row.get("first_seen_at"))
        case_number = str(row.get("raw_source_key") or row.get("id") or "")

        return RawIndicatorRecord(
            indicator_type=indicator_type,
            address_raw=address_raw,
            county_fips=self.county_fips,
            owner_name=None,
            filing_date=filing_date,
            case_number=case_number or None,
            source_url=row.get("source_url") or SOURCE_URL,
            raw_payload={
                "supabase_id": row.get("id"),
                "source_type": row.get("source_type"),
                "dedupe_key": row.get("dedupe_key"),
                "tags": row.get("tags"),
                "first_seen_at": row.get("first_seen_at"),
                "last_seen_at": row.get("last_seen_at"),
            },
        )


class DallasForeclosureNoticeSupabaseScraper(BaseCountyScraper):
    """
    Ingests gov_foreclosure_notice_raw from Supabase.

    These are foreclosure notices scraped from Dallas County Clerk PDFs by
    the GCP Cloud Run job (dallas-preforeclosure-scraper).

    Config keys (all optional):
        supabase_url  — override default from app settings
        supabase_key  — override default from app settings
        full_sync     — if True, sync all records regardless of date
    """

    county_fips = COUNTY_FIPS
    source_name = "Dallas Supabase foreclosure notices (from county PDFs)"
    indicator_types = ["foreclosure"]
    rate_limit_per_minute = 120

    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        from app.core.config import settings

        url = self.config.get("supabase_url") or settings.supabase_url
        key = self.config.get("supabase_key") or settings.supabase_key
        full_sync: bool = self.config.get("full_sync", False)
        lookback_days: int = int(self.config.get("lookback_days", 90))

        since: datetime | None = None
        if not full_sync:
            from datetime import timedelta
            since = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        reader = SupabasePagedReader(url, key, "gov_foreclosure_notice_raw")
        total = 0

        logger.info(
            "dallas_supabase_foreclosure_start",
            full_sync=full_sync,
            since=since.isoformat() if since else None,
        )

        async for row in reader.paginate(since=since, since_field="pulled_at"):
            record = self._build_record(row)
            if record and await self.validate_record(record):
                yield record
                total += 1

        logger.info("dallas_supabase_foreclosure_complete", total_yielded=total)

    def _build_record(self, row: dict) -> RawIndicatorRecord | None:
        address_raw = (row.get("property_address") or "").strip()
        # Remove stray unicode middle dots (·) that appear in some addresses
        address_raw = address_raw.replace("\u00b7", "").strip()
        if not address_raw:
            return None

        # Prefer sale_date, fall back to filing_date, then pulled_at
        filing_date = (
            _parse_date(row.get("sale_date"))
            or _parse_date(row.get("filing_date"))
            or _parse_date(row.get("pulled_at"))
        )

        case_number = str(row.get("case_number") or "").strip() or None
        source_url = row.get("source_pdf") or SOURCE_URL
        owner_name = (row.get("owner_name") or "").strip().title() or None

        return RawIndicatorRecord(
            indicator_type="foreclosure",
            address_raw=address_raw,
            county_fips=self.county_fips,
            owner_name=owner_name,
            filing_date=filing_date,
            case_number=case_number,
            source_url=source_url,
            raw_payload={
                "supabase_id": row.get("id"),
                "county": row.get("county"),
                "city": row.get("city"),
                "legal_description": row.get("legal_description"),
                "sale_date": row.get("sale_date"),
                "source_pdf": row.get("source_pdf"),
                "pulled_at": row.get("pulled_at"),
            },
        )
