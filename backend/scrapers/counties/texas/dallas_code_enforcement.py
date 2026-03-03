"""
Dallas City Code Enforcement Violations Scraper (Texas - FIPS 48113)

Source: City of Dallas Open Data Portal (Socrata SODA2 API)
Dataset: Building Inspection Code Enforcement Cases
Dataset ID: 7h2m-3um5
API: https://www.dallasopendata.com/resource/7h2m-3um5.json

Data: Municipal code enforcement cases filed by Dallas Building Inspection.
      Covers substandard structures, debris/nuisance, vegetation violations,
      unsafe conditions, and other property maintenance code violations.

Signal: indicator_type = "code_violation"
        Open code enforcement cases indicate property neglect or financial
        distress — correlated with absentee ownership and deferred maintenance.
        Properties with active violations are strong motivated-seller candidates.

Filter: ACTIVE cases only — case_status IN ('OPEN', 'PENDING HEARING',
        'REFERRED TO LEGAL', 'PENDING'). Historical closed cases are excluded
        so we only surface current distress signals.

Auth:   No API key required for public access. Provide socrata_app_token in
        scraper config to raise the per-IP rate limit from 1,000 to 100,000
        requests/day (register free at dallasopendata.com).

Rate limit: 20 req/min (Socrata is generous; conservative anyway)
"""

from __future__ import annotations

import structlog
from datetime import datetime
from typing import AsyncIterator

import httpx

from scrapers.base import BaseCountyScraper, RawIndicatorRecord

logger = structlog.get_logger(__name__)

COUNTY_FIPS = "48113"
# Dataset: "311 Service Requests October 1, 2020 to Present" — updated daily.
# Covers all 311 service requests; we filter to department='Code Compliance'.
DATASET_URL = "https://www.dallasopendata.com/resource/d7e7-envw.json"
PAGE_SIZE = 1000
SOURCE_URL = "https://www.dallasopendata.com/City-Services/311-Service-Requests-October-1-2020-to-Present/d7e7-envw"

# Status values that mean the request is CLOSED / no longer active
_CLOSED_STATUSES = (
    "'Closed'",
    "'Closed (Duplicate)'",
    "'Closed (Transferred)'",
    "'Canceled'",
)

# Only ingest open/active Code Compliance requests
_WHERE_CLAUSE = (
    "department='Code Compliance' "
    "AND status NOT IN (" + ", ".join(_CLOSED_STATUSES) + ")"
)


class DallasCodeEnforcementScraper(BaseCountyScraper):
    county_fips = COUNTY_FIPS
    source_name = "Dallas City Code Enforcement (Open Data Socrata)"
    indicator_types = ["code_violation"]
    rate_limit_per_minute = 20

    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        """
        Paginate all OPEN/ACTIVE code enforcement cases from Dallas Open Data.
        Yields one RawIndicatorRecord per active case.
        """
        headers: dict[str, str] = {"Accept": "application/json"}
        app_token = self.config.get("socrata_app_token", "")
        if app_token:
            headers["X-App-Token"] = app_token

        total = 0
        offset = 0

        async with httpx.AsyncClient(timeout=30.0) as client:
            while True:
                try:
                    resp = await client.get(
                        DATASET_URL,
                        params={
                            "$where": _WHERE_CLAUSE,
                            "$limit": PAGE_SIZE,
                            "$offset": offset,
                            "$order": "created_date DESC",
                        },
                        headers=headers,
                    )
                    await self._rate_limit_sleep()
                    resp.raise_for_status()
                    rows = resp.json()
                except (httpx.HTTPError, ValueError) as exc:
                    logger.warning(
                        "dallas_code_fetch_error",
                        offset=offset,
                        error=str(exc)[:200],
                    )
                    break

                if not rows:
                    break

                for row in rows:
                    record = self._build_record(row)
                    if record and await self.validate_record(record):
                        yield record
                        total += 1

                logger.info(
                    "dallas_code_page",
                    offset=offset,
                    batch=len(rows),
                    total_yielded=total,
                )

                if len(rows) < PAGE_SIZE:
                    break
                offset += PAGE_SIZE

        logger.info("dallas_code_complete", total_yielded=total)

    def _build_record(self, row: dict) -> RawIndicatorRecord | None:
        # Address comes pre-formatted as "1234 MAIN ST, DALLAS, TX, 75201"
        address = (row.get("address") or "").strip()
        if not address or not address[0].isdigit():
            return None

        # Normalise "ADDR, DALLAS, TX, 75201" → "ADDR, DALLAS, TX 75201" (no trailing comma)
        address_raw = address.replace(", TX, ", ", TX ")

        filing_date = None
        raw_date = row.get("created_date", "")
        if raw_date:
            try:
                filing_date = datetime.fromisoformat(raw_date[:10]).date()
            except ValueError:
                pass

        # service_request_number is the unique case identifier in this dataset
        case_number = (row.get("service_request_number") or "").strip() or None

        return RawIndicatorRecord(
            indicator_type="code_violation",
            address_raw=address_raw,
            county_fips=self.county_fips,
            filing_date=filing_date,
            case_number=case_number,
            source_url=SOURCE_URL,
            raw_payload={
                "service_request_number": case_number,
                "service_request_type": row.get("service_request_type", ""),
                "department": row.get("department", ""),
                "status": row.get("status", ""),
                "priority": row.get("priority", ""),
                "council_district": row.get("city_council_district", ""),
                "created_date": raw_date,
                "update_date": row.get("update_date", ""),
            },
        )
