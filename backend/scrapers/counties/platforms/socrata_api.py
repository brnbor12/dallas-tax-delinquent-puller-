"""
Generic Socrata API Scraper

Socrata is the open data platform used by hundreds of counties and cities
(data.gov, data.lacounty.gov, data.cityofchicago.org, etc.).

This single scraper handles any Socrata dataset by accepting the endpoint URL
and field mappings in the job config. This means dozens of county datasets can
be scraped without writing new Python code — just add a row to scrape_jobs.

Example scrape_jobs.config:
{
    "scraper_key": "socrata_generic",
    "endpoint": "https://data.cookcountyil.gov/resource/fc6m-7aad.json",
    "indicator_type": "tax_delinquent",
    "county_fips": "17031",
    "field_map": {
        "address_raw":    "property_address",
        "apn":            "pin",
        "owner_name":     "owner_name",
        "amount":         "total_due",
        "filing_date":    "sale_date"
    }
}
"""

from __future__ import annotations

import structlog
from datetime import datetime
from typing import AsyncIterator

import httpx

from scrapers.base import BaseCountyScraper, RawIndicatorRecord

logger = structlog.get_logger(__name__)


class SocrataAPIScraper(BaseCountyScraper):
    """
    Generic scraper for any Socrata open data endpoint.
    All configuration is passed via self.config (stored in scrape_jobs.config).
    """

    county_fips = "*"
    source_name = "Socrata Generic Scraper"
    indicator_types = []  # set from config
    rate_limit_per_minute = 30

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        cfg = self.config
        self.endpoint: str = cfg.get("endpoint", "")
        self.indicator_type: str = cfg.get("indicator_type", "tax_delinquent")
        self.county_fips = cfg.get("county_fips", "")  # type: ignore[assignment]
        self.field_map: dict[str, str] = cfg.get("field_map", {})
        self.indicator_types = [self.indicator_type]
        self.page_size: int = cfg.get("page_size", 1000)
        self.where_clause: str = cfg.get("where", "")  # optional Socrata $where filter

    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        if not self.endpoint:
            logger.error("socrata_scraper_no_endpoint")
            return

        offset = 0
        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                params: dict = {
                    "$limit":  self.page_size,
                    "$offset": offset,
                }
                if self.where_clause:
                    params["$where"] = self.where_clause

                try:
                    resp = await client.get(
                        self.endpoint,
                        params=params,
                        headers={"Accept": "application/json"},
                    )
                    resp.raise_for_status()
                    rows = resp.json()
                except Exception as exc:
                    logger.error("socrata_fetch_failed", endpoint=self.endpoint, offset=offset, error=str(exc))
                    break

                if not rows:
                    break

                for row in rows:
                    record = self._map_row(row)
                    if record and await self.validate_record(record):
                        yield record

                offset += self.page_size
                await self._rate_limit_sleep()

    def _map_row(self, row: dict) -> RawIndicatorRecord | None:
        fm = self.field_map

        def get(field: str) -> str | None:
            key = fm.get(field)
            return str(row[key]).strip() if key and key in row and row[key] else None

        address_raw = get("address_raw")
        if not address_raw:
            return None

        amount: float | None = None
        raw_amount = get("amount")
        if raw_amount:
            try:
                amount = float(raw_amount.replace("$", "").replace(",", ""))
            except ValueError:
                pass

        filing_date = None
        raw_date = get("filing_date")
        if raw_date:
            for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S"):
                try:
                    filing_date = datetime.strptime(raw_date[:10], fmt[:len(raw_date[:10])]).date()
                    break
                except ValueError:
                    continue

        return RawIndicatorRecord(
            indicator_type=self.indicator_type,
            address_raw=address_raw,
            county_fips=self.county_fips,
            apn=get("apn"),
            owner_name=get("owner_name"),
            owner_mailing_address=get("owner_mailing_address"),
            owner_mailing_state=get("owner_mailing_state"),
            amount=amount,
            filing_date=filing_date,
            case_number=get("case_number"),
            source_url=self.endpoint,
            raw_payload=row,
        )
