"""
Rentcast Listing Scraper — Active & Pending Sale Listings

Source: Rentcast API (https://developers.rentcast.io)
Endpoint: GET /listings/sale

Data:   Individual property listings with asking price, listing status
        (active, pending, sold), days on market, MLS info, property details.
        Covers any US market — configured per county via scraper config.

Signals produced:
  - Properties with active/pending listings are matched against existing
    distress indicators. A listed property with stacked signals = highly
    motivated seller.
  - Listing data enriches existing property records with market price,
    DOM, and listing status.

Config:
  api_key       — Rentcast API key (or set RENTCAST_API_KEY env var)
  county_fips   — Target county FIPS code
  state         — State abbreviation (e.g. "TX")
  cities        — List of cities to search (e.g. ["Dallas", "Irving", "Garland"])
  zip_codes     — Alternative: list of zip codes to search

Rate:   Free tier = 50 calls/month. Each call returns up to 500 listings.
        Paginate through results for full coverage.
"""

from __future__ import annotations

import os
import structlog
from datetime import date, datetime
from typing import AsyncIterator

import httpx

from scrapers.base import BaseCountyScraper, RawIndicatorRecord

logger = structlog.get_logger(__name__)

RENTCAST_BASE = "https://api.rentcast.io/v1"
SOURCE_URL = "https://www.rentcast.io"

# Default Dallas County config
DEFAULT_CONFIG = {
    "county_fips": "48113",
    "state": "TX",
    "cities": [
        "Dallas", "Irving", "Garland", "Mesquite", "Grand Prairie",
        "Richardson", "Carrollton", "Farmers Branch", "Duncanville",
        "DeSoto", "Cedar Hill", "Lancaster", "Rowlett", "Sachse",
        "Balch Springs", "Hutchins", "Wilmer", "Sunnyvale", "Seagoville",
    ],
}

# Listing statuses we care about
_ACTIVE_STATUSES = {"Active", "Pending", "Under Contract", "Contingent"}


def _parse_date(raw: str | None) -> date | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except (ValueError, TypeError):
        pass
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw[:10], fmt).date()
        except (ValueError, TypeError):
            continue
    return None


class RentcastListingScraper(BaseCountyScraper):
    county_fips = "48113"  # default, overridden by config
    source_name = "Rentcast API — Sale Listings"
    indicator_types = []  # enrichment scraper, doesn't create new indicators
    rate_limit_per_minute = 10

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        merged = {**DEFAULT_CONFIG, **(config or {})}
        self.county_fips = merged.get("county_fips", "48113")
        self._api_key = (
            merged.get("api_key")
            or os.environ.get("RENTCAST_API_KEY", "")
        )
        self._state = merged.get("state", "TX")
        self._cities = merged.get("cities", [])
        self._zip_codes = merged.get("zip_codes", [])

    async def fetch_records(self, **kwargs) -> AsyncIterator[RawIndicatorRecord]:
        if not self._api_key:
            logger.error("rentcast_no_api_key", hint="Set RENTCAST_API_KEY env var or pass config.api_key")
            return

        headers = {
            "Accept": "application/json",
            "X-Api-Key": self._api_key,
        }

        async with httpx.AsyncClient(
            headers=headers,
            base_url=RENTCAST_BASE,
            timeout=30.0,
        ) as client:
            total = 0
            seen_addresses: set[str] = set()

            # Search by city or zip code
            search_params = []
            if self._zip_codes:
                for zc in self._zip_codes:
                    search_params.append({"zipCode": zc})
            elif self._cities:
                for city in self._cities:
                    search_params.append({"city": city, "state": self._state})
            else:
                search_params.append({"state": self._state})

            for params in search_params:
                offset = 0
                while True:
                    await self._rate_limit_sleep()
                    try:
                        query = {
                            **params,
                            "status": "Active",
                            "propertyType": "Single Family,Condo,Townhouse,Multi Family",
                            "limit": 500,
                            "offset": offset,
                        }
                        resp = await client.get("/listings/sale", params=query)

                        if resp.status_code == 429:
                            logger.warning("rentcast_rate_limited", params=params)
                            break
                        if resp.status_code == 402:
                            logger.warning("rentcast_quota_exceeded")
                            return
                        if resp.status_code != 200:
                            logger.warning("rentcast_error", status=resp.status_code, body=resp.text[:200])
                            break

                        listings = resp.json()
                        if not listings:
                            break

                        logger.info("rentcast_page", params=str(params)[:40], offset=offset, count=len(listings))

                        for listing in listings:
                            record = self._build_record(listing)
                            if not record:
                                continue
                            # Dedupe by address
                            addr_key = record.address_raw.lower()
                            if addr_key in seen_addresses:
                                continue
                            seen_addresses.add(addr_key)

                            if await self.validate_record(record):
                                yield record
                                total += 1

                        # Paginate
                        if len(listings) < 500:
                            break
                        offset += 500

                    except Exception as exc:
                        logger.error("rentcast_request_error", params=params, error=str(exc)[:120])
                        break

            if total == 0:
                logger.warning("rentcast_no_listings")
            else:
                logger.info("rentcast_complete", total_yielded=total)

    def _build_record(self, listing: dict) -> RawIndicatorRecord | None:
        # Build address
        addr_line = (listing.get("formattedAddress") or "").strip()
        if not addr_line:
            addr1 = (listing.get("addressLine1") or "").strip()
            city = (listing.get("city") or "").strip()
            state = (listing.get("state") or "").strip()
            zip_code = (listing.get("zipCode") or "").strip()
            if not addr1:
                return None
            addr_line = f"{addr1}, {city}, {state} {zip_code}"

        if not addr_line or not addr_line[0].isdigit():
            return None

        price = listing.get("price")
        status = listing.get("status", "")
        listing_date = _parse_date(listing.get("listedDate"))
        last_seen = _parse_date(listing.get("lastSeenDate"))
        days_on_market = listing.get("daysOnMarket")
        sqft = listing.get("squareFootage")
        beds = listing.get("bedrooms")
        baths = listing.get("bathrooms")
        property_type = listing.get("propertyType", "")
        mls_number = listing.get("mlsNumber", "")

        # Use listing as enrichment — indicator_type signals the property is listed
        # We use a generic type that the scoring engine can weight
        return RawIndicatorRecord(
            indicator_type="active_listing",
            address_raw=addr_line,
            county_fips=self.county_fips,
            amount=float(price) if price else None,
            filing_date=listing_date or last_seen,
            days_on_market=days_on_market,
            case_number=mls_number or None,
            source_url=SOURCE_URL,
            raw_payload={
                "price": price,
                "status": status,
                "listing_date": str(listing_date) if listing_date else "",
                "last_seen": str(last_seen) if last_seen else "",
                "days_on_market": days_on_market,
                "property_type": property_type,
                "sqft": sqft,
                "bedrooms": beds,
                "bathrooms": baths,
                "mls_number": mls_number,
                "mls_name": listing.get("mlsName", ""),
                "listing_agent": listing.get("listingAgent", ""),
                "listing_office": listing.get("listingOffice", ""),
                "price_per_sqft": round(price / sqft, 2) if price and sqft else None,
            },
        )


# Pre-configured instances for different counties
class DallasRentcastScraper(RentcastListingScraper):
    """Rentcast listings for Dallas County, TX."""
    county_fips = "48113"
    source_name = "Rentcast API — Dallas County Sale Listings"

    def __init__(self, config: dict | None = None):
        super().__init__({
            "county_fips": "48113",
            "state": "TX",
            "cities": [
                "Dallas", "Irving", "Garland", "Mesquite", "Grand Prairie",
                "Richardson", "Carrollton", "Farmers Branch", "Duncanville",
                "DeSoto", "Cedar Hill", "Lancaster", "Rowlett", "Sachse",
                "Balch Springs", "Hutchins", "Sunnyvale", "Seagoville",
            ],
            **(config or {}),
        })


class HillsboroughRentcastScraper(RentcastListingScraper):
    """Rentcast listings for Hillsborough County, FL."""
    county_fips = "12057"
    source_name = "Rentcast API — Hillsborough County Sale Listings"

    def __init__(self, config: dict | None = None):
        super().__init__({
            "county_fips": "12057",
            "state": "FL",
            "cities": [
                "Tampa", "Brandon", "Riverview", "Valrico", "Plant City",
                "Temple Terrace", "Lutz", "Ruskin", "Seffner", "Thonotosassa",
                "Apollo Beach", "Gibsonton", "Lithia", "Dover", "Wimauma",
            ],
            **(config or {}),
        })


class PolkRentcastScraper(RentcastListingScraper):
    """Rentcast listings for Polk County, FL."""
    county_fips = "12105"
    source_name = "Rentcast API — Polk County Sale Listings"

    def __init__(self, config: dict | None = None):
        super().__init__({
            "county_fips": "12105",
            "state": "FL",
            "cities": [
                "Lakeland", "Winter Haven", "Bartow", "Auburndale",
                "Lake Wales", "Haines City", "Davenport", "Kissimmee",
                "Mulberry", "Fort Meade", "Frostproof", "Dundee",
            ],
            **(config or {}),
        })
