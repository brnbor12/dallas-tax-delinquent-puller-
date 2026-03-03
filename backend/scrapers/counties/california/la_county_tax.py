"""
LA County Tax Delinquent Scraper

Source: LA County Treasurer and Tax Collector
Data: Tax-defaulted properties (publicly available)
URL:  https://ttc.lacounty.gov/

LA County publishes its tax-defaulted property list via their open data portal.
This scraper fetches properties where property taxes are delinquent.

Rate limit: 30 req/min (well within what the county portal allows)
"""

from __future__ import annotations

import structlog
from datetime import date
from typing import AsyncIterator

import httpx

from scrapers.base import BaseCountyScraper, RawIndicatorRecord

logger = structlog.get_logger(__name__)

# LA County Assessor open data — tax defaulted parcels
# Real endpoint — published by LA County on their open data portal
LA_TAX_DEFAULTED_URL = "https://assessor.lacounty.gov/wp-json/wp/v2/"

# Fallback: Socrata-based dataset (LA County data portal)
# This uses the publicly available dataset for demonstration.
# The actual production endpoint should be verified at:
# https://data.lacounty.gov/datasets/tax-defaulted-property
SOCRATA_URL = "https://data.lacounty.gov/resource/7naa-ktzv.json"


class LACountyTaxDelinquentScraper(BaseCountyScraper):
    county_fips = "06037"
    source_name = "LA County Tax Defaulted Properties"
    indicator_types = ["tax_delinquent"]
    rate_limit_per_minute = 30

    async def fetch_records(
        self,
        limit: int = 1000,
        **kwargs,
    ) -> AsyncIterator[RawIndicatorRecord]:
        """
        Fetch tax-delinquent properties from the LA County open data portal.

        Paginates automatically through all records using Socrata's $offset param.
        Yields one RawIndicatorRecord per property.
        """
        offset = 0

        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                try:
                    resp = await client.get(
                        SOCRATA_URL,
                        params={
                            "$limit":  limit,
                            "$offset": offset,
                            "$order":  "assessor_id",
                        },
                        headers={"Accept": "application/json"},
                    )
                    resp.raise_for_status()
                    rows = resp.json()
                except httpx.HTTPError as exc:
                    logger.error("la_tax_fetch_failed", offset=offset, error=str(exc))
                    break

                if not rows:
                    break  # No more records

                for row in rows:
                    record = self._parse_row(row)
                    if record and await self.validate_record(record):
                        yield record

                offset += limit
                await self._rate_limit_sleep()

    def _parse_row(self, row: dict) -> RawIndicatorRecord | None:
        """Parse a Socrata row into a RawIndicatorRecord."""
        try:
            # Build address from available fields
            address_parts = [
                row.get("situs_address", ""),
                row.get("situs_city", ""),
                "CA",
                row.get("situs_zip", ""),
            ]
            address_raw = ", ".join(p for p in address_parts if p).strip()
            if not address_raw:
                return None

            # Parse tax amount
            amount: float | None = None
            raw_amount = row.get("total_amount_due") or row.get("defaulted_amount")
            if raw_amount:
                try:
                    amount = float(str(raw_amount).replace("$", "").replace(",", ""))
                except ValueError:
                    pass

            # Parse filing/default date
            filing_date: date | None = None
            raw_date = row.get("default_date") or row.get("fiscal_year_defaulted")
            if raw_date:
                try:
                    from datetime import datetime
                    filing_date = datetime.strptime(str(raw_date)[:10], "%Y-%m-%d").date()
                except ValueError:
                    pass

            return RawIndicatorRecord(
                indicator_type="tax_delinquent",
                address_raw=address_raw,
                county_fips=self.county_fips,
                apn=row.get("assessor_id") or row.get("ain"),
                owner_name=row.get("owner_name"),
                owner_mailing_address=row.get("owner_address"),
                owner_mailing_city=row.get("owner_city"),
                owner_mailing_state=row.get("owner_state"),
                owner_mailing_zip=row.get("owner_zip"),
                amount=amount,
                filing_date=filing_date,
                source_url=SOCRATA_URL,
                raw_payload=row,
            )

        except Exception as exc:
            logger.warning("la_tax_parse_failed", row=row, error=str(exc))
            return None
