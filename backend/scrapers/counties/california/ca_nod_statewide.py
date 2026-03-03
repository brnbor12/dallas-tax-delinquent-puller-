"""
California NOD (Notice of Default) Helper

In California, Notices of Default are recorded with county recorders and must
be publicly available under Government Code § 27280. Many counties publish
daily/weekly NOD lists.

This module provides a helper base for county recorder scrapers. Each CA county
recorder has a slightly different portal, but they share common patterns.

For production: integrate with a licensed data provider (ATTOM, PropertyRadar)
that aggregates CA NOD data statewide rather than scraping 58 county websites.
"""

from __future__ import annotations

import structlog
from typing import AsyncIterator

from scrapers.base import BaseCountyScraper, RawIndicatorRecord

logger = structlog.get_logger(__name__)


class CANODStatewideHelper(BaseCountyScraper):
    """
    Placeholder / integration point for CA statewide NOD data.

    In Phase 1: replace this with ATTOM Data Solutions API call.
    In Phase 2: add individual county recorder scrapers for priority counties.
    """

    county_fips = "*"  # Multi-county
    source_name = "California NOD (ATTOM Data)"
    indicator_types = ["pre_foreclosure"]
    rate_limit_per_minute = 60

    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        """
        When ATTOM_API_KEY is configured, fetch NOD data via ATTOM API.
        Otherwise, log a warning and yield nothing.
        """
        from app.core.config import settings

        if not settings.attom_api_key:
            logger.warning(
                "ca_nod_scraper_skipped",
                reason="ATTOM_API_KEY not configured. "
                       "Set it to enable California NOD data ingestion.",
            )
            return

        async for record in self._fetch_attom_nod(**kwargs):
            yield record

    async def _fetch_attom_nod(self, state: str = "CA", **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        """
        ATTOM Data Solutions API — Pre-Foreclosure / NOD endpoint.

        ATTOM provides licensed, aggregated NOD data for all CA counties.
        Docs: https://api.developer.attomdata.com/docs
        """
        import httpx
        from app.core.config import settings

        base_url = "https://api.developer.attomdata.com/propertyapi/v1.0.0"
        headers = {
            "apikey": settings.attom_api_key,
            "accept": "application/json",
        }

        page = 1
        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                try:
                    resp = await client.get(
                        f"{base_url}/assessment/detail",
                        headers=headers,
                        params={
                            "state": state,
                            "eventtype": "NTD",  # Notice to Default
                            "page": page,
                            "pagesize": 100,
                        },
                    )
                    if resp.status_code == 404:
                        break
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as exc:
                    logger.error("attom_nod_failed", page=page, error=str(exc))
                    break

                properties = data.get("property", [])
                if not properties:
                    break

                for prop in properties:
                    record = self._parse_attom_record(prop)
                    if record:
                        yield record

                page += 1
                await self._rate_limit_sleep()

    def _parse_attom_record(self, prop: dict) -> RawIndicatorRecord | None:
        try:
            addr = prop.get("address", {})
            address_raw = f"{addr.get('line1', '')}, {addr.get('city', '')}, {addr.get('state', '')} {addr.get('postal1', '')}".strip(", ")

            location = prop.get("location", {})
            county_fips = location.get("fips", "06000")

            return RawIndicatorRecord(
                indicator_type="pre_foreclosure",
                address_raw=address_raw,
                county_fips=county_fips,
                apn=prop.get("identifier", {}).get("apn"),
                source_url="https://api.developer.attomdata.com",
                raw_payload=prop,
            )
        except Exception as exc:
            logger.warning("attom_parse_failed", error=str(exc))
            return None
