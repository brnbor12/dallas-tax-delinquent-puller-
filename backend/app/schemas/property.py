"""Pydantic response schemas for property API endpoints."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class OwnerSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name_raw: str
    mailing_address: str | None
    mailing_city: str | None
    mailing_state: str | None
    mailing_zip: str | None
    owner_type: str | None
    is_absentee: bool | None
    is_out_of_state: bool | None


class IndicatorSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    indicator_type: str
    status: str
    source: str
    source_url: str | None
    amount_cents: int | None
    filing_date: date | None
    expiry_date: date | None
    case_number: str | None
    created_at: datetime

    @property
    def amount_dollars(self) -> float | None:
        return self.amount_cents / 100 if self.amount_cents else None


class ScoreSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    total_score: float
    score_tier: str | None
    score_breakdown: dict | None
    indicator_count: int
    last_scored_at: datetime


class ListingSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    source: str
    list_price: int | None
    original_price: int | None
    days_on_market: int | None
    price_reductions: int
    listing_status: str | None
    listed_date: date | None
    last_price_cut: date | None


class CountySchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    fips_code: str
    name: str
    state_abbr: str


class PropertyListItem(BaseModel):
    """Lightweight property representation for list view."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    address_raw: str
    address_line1: str | None
    address_city: str | None
    address_state: str | None
    address_zip: str | None
    property_type: str | None
    assessed_value: int | None
    score: ScoreSchema | None
    active_indicator_types: list[str] = []
    lat: float | None = None
    lng: float | None = None


class PropertyDetail(BaseModel):
    """Full property detail for the slide-in panel."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    apn: str | None
    address_raw: str
    address_line1: str | None
    address_city: str | None
    address_state: str | None
    address_zip: str | None
    property_type: str | None
    year_built: int | None
    sqft: int | None
    bedrooms: int | None
    bathrooms: float | None
    lot_size_sqft: int | None
    zoning: str | None
    assessed_value: int | None
    market_value: int | None
    last_sale_date: date | None
    last_sale_price: int | None
    data_source: str | None
    created_at: datetime
    updated_at: datetime

    county: CountySchema | None
    owners: list[OwnerSchema]
    indicators: list[IndicatorSchema]
    score: ScoreSchema | None
    listings: list[ListingSchema]

    lat: float | None = None
    lng: float | None = None


class PropertyListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    results: list[PropertyListItem]


class FilterParams(BaseModel):
    """Query parameters for property list endpoint."""
    state: str | None = None
    # Comma-separated FIPS codes, e.g. "12057" or "12057,12105"
    county_fips: str | None = None
    # Comma-separated zip codes
    zip_codes: str | None = None
    # Comma-separated indicator types
    indicator_types: str | None = None
    score_min: float | None = None
    score_tier: str | None = None
    property_type: str | None = None
    year_built_min: int | None = None
    year_built_max: int | None = None
    assessed_min: int | None = None
    assessed_max: int | None = None
    bedrooms_min: int | None = None
    dom_min: int | None = None
    out_of_state_only: bool = False
    lat: float | None = None
    lng: float | None = None
    radius_km: float | None = None
    sort_by: str = "score_desc"
    page: int = 1
    page_size: int = 50

    def to_filter_dict(self) -> dict:
        d = self.model_dump(exclude_none=True, exclude={"page", "page_size"})
        # Split comma-separated list params into actual lists
        for key in ("county_fips", "zip_codes", "indicator_types"):
            if key in d and isinstance(d[key], str):
                d[key] = [v.strip() for v in d[key].split(",") if v.strip()]
                if not d[key]:
                    del d[key]
        return d
