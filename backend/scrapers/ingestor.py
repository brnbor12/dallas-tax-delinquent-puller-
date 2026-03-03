"""
Scraper Ingestor — the single point of entry for all scraped data.

Responsibilities:
  1. Validate incoming RawIndicatorRecord.
  2. Geocode the address (Google or Nominatim).
  3. Look up / create the County and Property rows.
  4. Deduplicate indicators (APN+county, then fuzzy address).
  5. Upsert Owner.
  6. Upsert PropertyIndicator.
  7. Enqueue a score recalculation task.

This module is intentionally synchronous-at-the-DB-level (uses SQLAlchemy
sync calls inside Celery tasks) to keep things simple. Async variant can be
added if needed.
"""

from __future__ import annotations

import structlog
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from sqlalchemy import text

from app.models.county import County
from app.models.indicator import PropertyIndicator
from app.models.owner import Owner
from app.models.property import Property
from scrapers.address_utils import normalize_address, parse_address_components
from scrapers.base import RawIndicatorRecord
from scrapers.geocoder import geocode_address

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


class IngestResult:
    def __init__(self) -> None:
        self.found = 0
        self.upserted = 0
        self.failed = 0
        self.errors: list[str] = []


def ingest_record(session: Session, record: RawIndicatorRecord, geocode: bool = True) -> bool:
    """
    Ingest a single RawIndicatorRecord into the database.
    Returns True on success, False on failure.

    geocode=False: skip geocoding — useful for bulk imports where coordinates
    can be filled in by a later batch job. Properties without coordinates are
    still indexed (by APN / address) and can receive indicators.
    """
    try:
        # 1. Resolve county
        county = _get_or_create_county(session, record.county_fips)

        # 2. Geocode address (skipped for bulk imports)
        lat, lng = (None, None)
        if geocode:
            lat, lng = geocode_address(record.address_raw)

        # 3. Resolve property (APN lookup → fuzzy address → create)
        prop = _get_or_create_property(session, record, county, lat, lng)

        # 4. Upsert owner
        if record.owner_name:
            _upsert_owner(session, prop, record)

        # 5. Upsert indicator
        _upsert_indicator(session, prop, record)

        session.flush()
        return True

    except Exception as exc:
        logger.error(
            "ingest_record_failed",
            address=record.address_raw,
            county_fips=record.county_fips,
            error=str(exc),
        )
        session.rollback()
        return False


def _get_or_create_county(session: Session, fips_code: str) -> County:
    county = session.execute(
        select(County).where(County.fips_code == fips_code)
    ).scalar_one_or_none()

    if county is None:
        county = County(fips_code=fips_code, name=fips_code, state_abbr="", state_fips=fips_code[:2])
        session.add(county)
        session.flush()

    return county


def _get_or_create_property(
    session: Session,
    record: RawIndicatorRecord,
    county: County,
    lat: float | None,
    lng: float | None,
) -> Property:
    # Try APN lookup first
    if record.apn:
        prop = session.execute(
            select(Property).where(
                Property.apn == record.apn,
                Property.county_id == county.id,
            )
        ).scalar_one_or_none()
        if prop:
            _update_property_location(prop, lat, lng)
            return prop

    addr_norm = normalize_address(record.address_raw)

    # Exact normalized address match within county (fast, index hit)
    if county.id and addr_norm:
        prop = session.execute(
            select(Property).where(
                Property.county_id == county.id,
                Property.address_normalized == addr_norm,
            )
        ).scalar_one_or_none()
        if prop:
            _update_property_location(prop, lat, lng)
            return prop

    # Fuzzy address match via pg_trgm (house-number-aware, catches DRIVE vs DR etc.)
    # Only for records without APN (PDF-sourced) — links them to TRW/API records.
    if county.id and addr_norm and not record.apn and addr_norm[0].isdigit():
        row = session.execute(
            text(
                """
                SELECT id, address_raw FROM properties
                WHERE county_id = :county_id
                  AND address_normalized IS NOT NULL
                  AND apn IS NOT NULL
                  -- Exact house number match (first numeric token)
                  AND substring(address_normalized FROM E'^\\d+')
                    = substring(:addr FROM E'^\\d+')
                  -- Fuzzy street name match (similarity >= 0.65 avoids false positives)
                  AND similarity(
                    regexp_replace(address_normalized, E'^\\d+\\s*', ''),
                    regexp_replace(:addr, E'^\\d+\\s*', '')
                  ) >= 0.65
                ORDER BY similarity(
                    regexp_replace(address_normalized, E'^\\d+\\s*', ''),
                    regexp_replace(:addr, E'^\\d+\\s*', '')
                ) DESC
                LIMIT 1
                """
            ),
            {"county_id": county.id, "addr": addr_norm},
        ).fetchone()
        if row:
            prop = session.get(Property, row[0])
            if prop:
                logger.info(
                    "ingestor_fuzzy_match",
                    input=record.address_raw,
                    matched=prop.address_raw,
                )
                _update_property_location(prop, lat, lng)
                return prop

    # Create new property
    location_wkt = f"SRID=4326;POINT({lng} {lat})" if lat and lng else None
    line1, city, state, zip_code = parse_address_components(record.address_raw)
    prop = Property(
        apn=record.apn,
        county_id=county.id,
        address_raw=record.address_raw,
        address_normalized=addr_norm or None,
        address_line1=line1,
        address_city=city,
        address_state=state,
        address_zip=zip_code,
        data_source="scraper",
        raw_data=record.raw_payload,
    )
    if location_wkt:
        prop.location = location_wkt  # type: ignore[assignment]

    session.add(prop)
    session.flush()
    return prop


def _update_property_location(prop: Property, lat: float | None, lng: float | None) -> None:
    if lat and lng and prop.location is None:
        prop.location = f"SRID=4326;POINT({lng} {lat})"  # type: ignore[assignment]


def _upsert_owner(session: Session, prop: Property, record: RawIndicatorRecord) -> None:
    owner = session.execute(
        select(Owner).where(Owner.property_id == prop.id)
    ).scalar_one_or_none()

    prop_state = prop.address_state
    mailing_state = record.owner_mailing_state
    is_out_of_state = (
        bool(mailing_state and prop_state and mailing_state.upper() != prop_state.upper())
    )
    is_absentee = is_out_of_state or bool(
        record.owner_mailing_address and
        record.owner_mailing_address != prop.address_raw
    )

    if owner is None:
        owner = Owner(
            property_id=prop.id,
            name_raw=record.owner_name or "",
            mailing_address=record.owner_mailing_address,
            mailing_city=record.owner_mailing_city,
            mailing_state=mailing_state,
            mailing_zip=record.owner_mailing_zip,
            owner_type=record.owner_type,
            is_absentee=is_absentee,
            is_out_of_state=is_out_of_state,
        )
        session.add(owner)
    else:
        owner.name_raw = record.owner_name or owner.name_raw
        owner.is_absentee = is_absentee
        owner.is_out_of_state = is_out_of_state


def _upsert_indicator(session: Session, prop: Property, record: RawIndicatorRecord) -> None:
    amount_cents = int(record.amount * 100) if record.amount is not None else None

    # Dedup on (property_id, indicator_type, case_number) if case_number given
    existing = None
    if record.case_number:
        existing = session.execute(
            select(PropertyIndicator).where(
                PropertyIndicator.property_id == prop.id,
                PropertyIndicator.indicator_type == record.indicator_type,
                PropertyIndicator.case_number == record.case_number,
            )
        ).scalar_one_or_none()

    if existing:
        existing.amount_cents = amount_cents or existing.amount_cents
        existing.updated_at = datetime.now(timezone.utc)
    else:
        indicator = PropertyIndicator(
            property_id=prop.id,
            indicator_type=record.indicator_type,
            status="active",
            source=record.source_url or "scraper",
            source_url=record.source_url,
            amount_cents=amount_cents,
            filing_date=record.filing_date,
            expiry_date=record.expiry_date,
            case_number=record.case_number,
            raw_data=record.raw_payload,
        )
        session.add(indicator)
