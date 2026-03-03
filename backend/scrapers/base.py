"""
Abstract base class for all county / court / MLS scrapers.

Every scraper must:
  1. Declare class-level metadata (county_fips, source_name, indicator_types).
  2. Implement `fetch_records()` as an async generator yielding RawIndicatorRecord.

The ingestor only ever speaks RawIndicatorRecord — it knows nothing about the
source. This means new counties/scrapers can be added without touching the
ingestor, the API, or the scoring engine.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import AsyncIterator


@dataclass
class RawIndicatorRecord:
    """
    Standardized output from any scraper.

    Every field except indicator_type, address_raw, and county_fips is optional
    because different sources expose different amounts of structured data.
    """
    indicator_type: str        # Must be a value in INDICATOR_TYPES
    address_raw: str           # Full address string — ingestor will geocode
    county_fips: str           # 5-digit FIPS code e.g. "06037"

    apn: str | None = None                    # Assessor Parcel Number
    owner_name: str | None = None
    owner_mailing_address: str | None = None
    owner_mailing_city: str | None = None
    owner_mailing_state: str | None = None
    owner_mailing_zip: str | None = None
    owner_type: str | None = None             # individual | LLC | trust | bank

    amount: float | None = None               # Dollar amount (tax debt, lien, etc.)
    filing_date: date | None = None
    expiry_date: date | None = None
    case_number: str | None = None
    days_on_market: int | None = None

    source_url: str = ""
    raw_payload: dict = field(default_factory=dict)


class BaseCountyScraper(ABC):
    """
    All scrapers inherit from this class.

    Subclasses must set class-level attributes:
      county_fips:     str   — "06037" for LA County, "*" for multi-county scrapers
      source_name:     str   — Human-readable name for logging
      indicator_types: list  — Which indicator types this scraper produces
    """

    county_fips: str = ""
    source_name: str = ""
    indicator_types: list[str] = []
    rate_limit_per_minute: int = 20  # Override per scraper

    def __init__(self, config: dict | None = None) -> None:
        self.config: dict = config or {}
        self._request_count = 0

    @abstractmethod
    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        """
        Yield RawIndicatorRecord instances one at a time.

        Implementors should:
        - Handle pagination internally (don't load everything into memory)
        - Respect self.rate_limit_per_minute
        - Log errors per record and continue (don't raise on bad rows)
        """
        ...

    async def validate_record(self, record: RawIndicatorRecord) -> bool:
        """Basic validation — override for county-specific rules."""
        return bool(record.address_raw and record.county_fips and record.indicator_type)

    async def _rate_limit_sleep(self) -> None:
        """Sleep to respect rate limit. Call after each HTTP request."""
        self._request_count += 1
        seconds_per_request = 60.0 / max(self.rate_limit_per_minute, 1)
        await asyncio.sleep(seconds_per_request)
